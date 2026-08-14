"""Inference-only splat variants on the trained b2ck5kfd (heartl1_w050) model. NO retraining.

Variants per val subject (shared forward pass, same Δ field):
  A: current   — splat 518² world_points + 518-upsampled intensity → (D,256,256)
  B: native256 — bilinear-downsample world_points 518→256, intensity = the ORIGINAL native
                 256² canonical slices (phases[t,z]), splat → (D,256,256)
  C: high-z    — same 518² points/intensity as A, but splat into (128,256,256) with
                 z_scale scaled so the 128 planes span the same physical z extent.

Run: micromamba run -n svr python <this file>
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = "/home/minsukc/vggt"
sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
os.chdir(REPO)
from omegaconf import OmegaConf
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from vggt.utils.splat import splat_to_volume
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local

CKPT_SRC = os.path.join(REPO, "scratch/logs/213520194_mri_volume_heartl1_w050_dynamic_axial_cmrx24only/ckpts/checkpoint_best.pt")
FIGS = os.path.join(REPO, "figs")
OUT_DIR = os.path.join(REPO, "result", "resolution_experiments")
OUT_JSON = os.path.join(OUT_DIR, "b2ck5kfd_splat_variants.json")
N_VAL = 10
FIG_SEQS = [0, 5]
DZ_HI = 128


def build_dataset():
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    return MRIDataset(
        common_conf, os.path.join(REPO, "scratch/data"),
        split="val", split_file=os.path.join(REPO, "training/splits/cmrx24only.txt"),
        mode="dynamic", mri_mode="axial", num_slices=20, target_size=518,
        reference_slot=True, one_frame_per_slice=True, continuous_z=False,
        t_target_fixed=None, defer_input_images=False,
    )


def build_batch(data, device, seq_index):
    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(device)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0  # (1,S,3,518,518) [0,1]
    batch = {
        "images": imgs,
        "scanner_coords": st("scanner_coords"),
        "z_indices": st("z_indices"),
        "t_indices": st("t_indices"),
        "timesteps": st("timesteps", np.int64),
        "slice_indices": st("slice_indices", np.int64),
        "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0).to(device),
        "anatomy_bbox": torch.from_numpy(np.asarray(data["anatomy_bbox"]).astype(np.int64)).unsqueeze(0).to(device),
        "phases": torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).unsqueeze(0).to(device),
        "z_scale": torch.from_numpy(np.asarray(data["z_scale"]).astype(np.float32)).to(device),
        "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=device),
    }
    return batch


def splat(pos, inten, grid, z_scale):
    B, S, h, w, _ = pos.shape
    p = pos.reshape(B, S * h * w, 3); i = inten.reshape(B, S * h * w)
    wgt = (i > 1e-3).to(i.dtype)
    return splat_to_volume(p, i, grid, z_scale, weight=wgt)


def psnr(pred, gt, mask=None):
    if mask is not None:
        d = ((pred - gt) ** 2)[mask]
    else:
        d = (pred - gt) ** 2
    mse = d.mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def main():
    dev = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    model = VGGT(depth=24, embed_dim=1024, num_heads=16,
                 use_z_pose_embedding=True, reference_slot=True,
                 use_reference_token=True, train_on_residual_dvf=True).to(dev)
    sd = torch.load(stage_checkpoint_to_local(CKPT_SRC), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    ds = build_dataset()
    print(f"val subjects: {len(ds.subjects)}")

    recs, fig_cache = [], {}
    for i in range(N_VAL):
        data = ds.get_data(seq_index=i, img_per_seq=20)
        batch = build_batch(data, dev, i)
        V_gt = batch["gt_target_volume"]                      # (1,D,256,256)
        D = V_gt.shape[1]
        z_scale = float(batch["z_scale"].reshape(-1)[0])
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        wp = preds["world_points"].float()                    # (1,S,518,518,3)
        inten518 = batch["images"].float().mean(dim=2)        # (1,S,518,518) [0,1]

        # native 256 intensities: phases[t_s, z_s]
        S = wp.shape[1]
        ts = batch["timesteps"][0].reshape(S).tolist()
        zs = batch["slice_indices"][0].reshape(S).tolist()
        nat = torch.stack([batch["phases"][0, int(t), :, :, ...][int(z)] if batch["phases"].dim() == 5
                           else batch["phases"][0, int(t)][int(z)] for t, z in zip(ts, zs)]).unsqueeze(0)
        # sanity: upsampled native slice should match the model input
        up = F.interpolate(nat.reshape(S, 1, 256, 256), size=(518, 518),
                           mode="bilinear", align_corners=True).reshape(1, S, 518, 518)
        err = (up - inten518).abs().max().item()
        if i == 0:
            print(f"native-vs-input consistency max err: {err:.2e} (expect ~1e-2 or less; "
                  f"nonzero only from uint8 quantization in 'images')")

        # A: current
        Va, Ca = splat(wp, inten518, (D, 256, 256), z_scale)
        # B: downsample wp to 256, native intensity
        wp256 = F.interpolate(wp[0].permute(0, 3, 1, 2), size=(256, 256),
                              mode="bilinear", align_corners=True).permute(0, 2, 3, 1).unsqueeze(0)
        Vb, Cb = splat(wp256, nat, (D, 256, 256), z_scale)
        # C: high-z grid, same physical extent → z_scale' = z_scale*(DZ_HI-1)/(D-1)
        zs_hi = z_scale * (DZ_HI - 1) / (D - 1)
        Vc, Cc = splat(wp, inten518, (DZ_HI, 256, 256), zs_hi)

        bb = batch["anatomy_bbox"][0].tolist()
        z0, z1, y0, y1, x0, x1 = [int(v) for v in bb]
        mask = torch.zeros_like(V_gt, dtype=torch.bool)
        mask[:, z0:z1, y0:y1, x0:x1] = True
        rec = dict(seq=i, D=D,
                   A_full=psnr(Va, V_gt), A_bbox=psnr(Va, V_gt, mask),
                   B_full=psnr(Vb, V_gt), B_bbox=psnr(Vb, V_gt, mask),
                   AB_disagree=float((Va - Vb).abs().mean().item()),
                   C_cov_frac=float((Cc > 1e-6).float().mean().item()),
                   A_cov_frac=float((Ca > 1e-6).float().mean().item()))
        recs.append(rec)
        print(f"seq{i:2d} D={D:2d} | A(518): bbox {rec['A_bbox']:.2f} full {rec['A_full']:.2f}"
              f" | B(256native): bbox {rec['B_bbox']:.2f} full {rec['B_full']:.2f}"
              f" | Δbbox {rec['B_bbox']-rec['A_bbox']:+.2f} | covC {rec['C_cov_frac']:.3f} covA {rec['A_cov_frac']:.3f}")

        if i in FIG_SEQS:
            zc = (z0 + z1) // 2
            # coronal view: fix y = mid of bbox → (Dz, x)
            yc = (y0 + y1) // 2
            fig_cache[i] = dict(
                gt_ax=V_gt[0, zc].cpu().numpy(), A_ax=Va[0, zc].cpu().numpy(), B_ax=Vb[0, zc].cpu().numpy(),
                gt_cor=V_gt[0, :, yc, :].cpu().numpy(), A_cor=Va[0, :, yc, :].cpu().numpy(),
                C_cor=Vc[0, :, yc, :].cpu().numpy(), Ccov_cor=Cc[0, :, yc, :].cpu().numpy(),
                bbox=bb, rec=rec, D=D)

    # aggregate
    mean = lambda k: float(np.mean([r[k] for r in recs]))
    summary = {k: mean(k) for k in ("A_full", "A_bbox", "B_full", "B_bbox", "AB_disagree", "A_cov_frac", "C_cov_frac")}
    print("\n=== MEAN over", len(recs), "val subjects (clean, no resp) ===")
    print(f"  bbox PSNR   A(518 splat): {summary['A_bbox']:.3f}   B(256 native): {summary['B_bbox']:.3f}   Δ {summary['B_bbox']-summary['A_bbox']:+.3f}")
    print(f"  full PSNR   A: {summary['A_full']:.3f}   B: {summary['B_full']:.3f}   Δ {summary['B_full']-summary['A_full']:+.3f}")
    print(f"  coverage>0  A(D-grid): {summary['A_cov_frac']:.3f}   C(128-grid): {summary['C_cov_frac']:.3f}")
    json.dump(dict(ckpt=CKPT_SRC, n=len(recs), summary=summary, records=recs), open(OUT_JSON, "w"), indent=2)

    # ── fig 1: A vs B axial ──
    seqs = sorted(fig_cache)
    fig, axes = plt.subplots(len(seqs), 4, figsize=(16, 4.2 * len(seqs)), squeeze=False)
    for r, sq in enumerate(seqs):
        d = fig_cache[sq]; z0, z1, y0, y1, x0, x1 = [int(v) for v in d["bbox"]]
        crop = lambda im: im[y0:y1, x0:x1]
        vmax = float(np.percentile(crop(d["gt_ax"]), 99.5)) or 1.0
        panels = [("GT (target phase)", crop(d["gt_ax"]), "gray", vmax),
                  (f"A: 518 splat (current)\nbbox {d['rec']['A_bbox']:.2f} dB", crop(d["A_ax"]), "gray", vmax),
                  (f"B: 256-native splat\nbbox {d['rec']['B_bbox']:.2f} dB", crop(d["B_ax"]), "gray", vmax),
                  ("|A − B|", np.abs(crop(d["A_ax"]) - crop(d["B_ax"])), "magma", None)]
        for c, (t, im, cm, vm) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(im, cmap=cm, vmin=0, vmax=(vm if vm else np.percentile(im, 99.5) + 1e-6))
            ax.set_title(f"seq{sq}  {t}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "splatvar_A_vs_B_axial.png"), dpi=130); plt.close(fig)

    # ── fig 2: high-z coronal ──
    fig, axes = plt.subplots(len(seqs), 4, figsize=(16, 4.2 * len(seqs)), squeeze=False)
    for r, sq in enumerate(seqs):
        d = fig_cache[sq]; z0, z1, y0, y1, x0, x1 = [int(v) for v in d["bbox"]]
        D = d["D"]
        gt_up = F.interpolate(torch.from_numpy(d["gt_cor"])[None, None], size=(DZ_HI, 256),
                              mode="bilinear", align_corners=True)[0, 0].numpy()
        A_up = F.interpolate(torch.from_numpy(d["A_cor"])[None, None], size=(DZ_HI, 256),
                             mode="bilinear", align_corners=True)[0, 0].numpy()
        vmax = float(np.percentile(gt_up, 99.5)) or 1.0
        panels = [(f"GT coronal (D={D}, z-interp to {DZ_HI})", gt_up[:, x0:x1], "gray", vmax),
                  (f"A: splat at D={D}, z-interp up", A_up[:, x0:x1], "gray", vmax),
                  (f"C: direct splat into {DZ_HI} z-planes", d["C_cor"][:, x0:x1], "gray", vmax),
                  (f"C coverage (log1p)", np.log1p(d["Ccov_cor"][:, x0:x1]), "viridis", None)]
        for c, (t, im, cm, vm) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(im, cmap=cm, vmin=0, vmax=(vm if vm else im.max() + 1e-6), aspect="auto")
            ax.set_title(f"seq{sq}  {t}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "splatvar_highz_coronal.png"), dpi=130); plt.close(fig)
    print("figs written: figs/splatvar_A_vs_B_axial.png, figs/splatvar_highz_coronal.png")


if __name__ == "__main__":
    main()
