"""Side-by-side: splat at 518² (current) vs native 256² render — NO retraining.

The current pipeline upsamples each native-256 canonical input slice to 518² for DINOv2,
the DPT head emits Δ at 518², and the splat scatters the 518-UPSAMPLED intensity into the
256³ grid (~4 points/voxel in-plane, each carrying a bilinearly-interpolated value).

Proposed change (user): render at native 256 — DPT outputs Δ at 256, splat the native-256
input-slice intensity (~1 point/voxel, no upsample interpolation).

This is a NO-TRAIN first-order proxy: we take the trained model's 518 Δ field and
downsample world_points + intensity to 256, then splat. (The true version re-fits the head
to a 256 render; since Δ is smooth, downsampling it is a faithful first-order stand-in.)
Both variants share the same Δ and same intensity content — only the RENDER resolution
differs, so any delta is purely the splat-resolution effect.

Run: micromamba run -n svr python tools/compare_splat_resolution.py --n-val 24
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)

from vggt.utils.splat import splat_to_volume                          # noqa: E402
from loss import compute_volume_intensity_loss                        # noqa: E402
import tools.eval_variants_matrix as H                                # noqa: E402

OUT = os.path.join(REPO, "result", "limits_eval"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(H.LOGS, "218747856_mri_volume_resp_allphases_aggft_z_no_t", "ckpts", "checkpoint_last.pt")
GRID = (12, 256, 256)


def splat_at(world_points, intensity, grid):
    """world_points (1,S,h,w,3), intensity (1,S,h,w) → (V_canon, coverage) at grid res."""
    B, S, h, w, _ = world_points.shape
    pos = world_points.reshape(B, S * h * w, 3)
    inten = intensity.reshape(B, S * h * w)
    wgt = (inten > 1e-3).to(inten.dtype)
    return splat_to_volume(pos, inten, grid, weight=wgt)


def downsample(t_bshwc_or_bshw, size):
    """Bilinear-resize the spatial dims to (size,size). Handles (1,S,h,w,3) and (1,S,h,w)."""
    if t_bshwc_or_bshw.dim() == 5:                 # world_points (1,S,h,w,3)
        B, S, h, w, C = t_bshwc_or_bshw.shape
        x = t_bshwc_or_bshw.permute(0, 1, 4, 2, 3).reshape(B * S, C, h, w)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)
        return x.reshape(B, S, C, size, size).permute(0, 1, 3, 4, 2).contiguous()
    else:                                          # intensity (1,S,h,w)
        B, S, h, w = t_bshwc_or_bshw.shape
        x = t_bshwc_or_bshw.reshape(B * S, 1, h, w)
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=True)
        return x.reshape(B, S, size, size)


def metrics(V_canon, coverage, batch):
    """Reuse the trainer's exact metric code by pre-supplying V_canon/coverage."""
    preds = {"world_points": batch["scanner_coords"], "V_canon": V_canon, "coverage": coverage}
    out = compute_volume_intensity_loss(preds, batch, grid_shape=GRID, tv_weight=0.0)
    f = lambda k: (float(out[k].item()) if k in out and out[k] is not None else None)
    return dict(full=f("metric_psnr_3d_full"), bbox=f("metric_psnr_3d_bbox"),
                motion=f("metric_psnr_3d_motion"), ssim=f("metric_ssim_3d_full"),
                cov_frac=f("metric_coverage_frac"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--fig-samples", default="0,7", help="seq indices to render side-by-side")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} ckpt={os.path.basename(CKPT)}")
    ds = H.build_dataset(); print(f"val subjects: {len(ds.subjects)}")
    model, info = H.make_model(use_t=False, ckpt_path=CKPT, device=dev)
    print("backbone:", {k: info[k] for k in ("missing", "unexpected", "ckpt_epoch")})
    resp = H.PROTOCOLS["breathing"]                 # the real corrupted→clean task

    recs, fig_cache = [], {}
    fig_seqs = {int(x) for x in args.fig_samples.split(",")}
    for i in range(args.n_val):
        data = ds.get_data(seq_index=i, img_per_seq=H.NUM_SLICES)
        batch = H.build_batch(data, dev, seq_index=i)
        batch = H.gpu_augment_batch(batch, None, dev, respiratory_cfg=resp, train=False)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        wp = preds["world_points"].float()                       # (1,S,518,518,3)
        inten = batch["images"].float().mean(dim=2)              # (1,S,518,518) in [0,1]

        # Variant A: current 518 render
        Va, Ca = splat_at(wp, inten, GRID)
        # Variant B: native-256 render (downsample Δ field + intensity, then splat)
        wp256, in256 = downsample(wp, 256), downsample(inten, 256)
        Vb, Cb = splat_at(wp256, in256, GRID)

        ma, mb = metrics(Va, Ca, batch), metrics(Vb, Cb, batch)
        recs.append(dict(seq=i, t=int(np.asarray(data["t_target"]).flatten()[0]), A=ma, B=mb))
        print(f"seq{i:2d} t{recs[-1]['t']:2d} | 518: motion {ma['motion']:.2f} bbox {ma['bbox']:.2f} ssim {ma['ssim']:.3f}"
              f"  | 256: motion {mb['motion']:.2f} bbox {mb['bbox']:.2f} ssim {mb['ssim']:.3f}"
              f"  | Δmotion {mb['motion']-ma['motion']:+.2f} Δbbox {mb['bbox']-ma['bbox']:+.2f}")
        if i in fig_seqs:
            z0, z1 = int(batch["anatomy_bbox"][0][0]), int(batch["anatomy_bbox"][0][1])
            zc = (z0 + z1) // 2
            fig_cache[i] = dict(gt=batch["gt_target_volume"][0, zc].cpu().numpy(),
                                A=Va[0, zc].cpu().numpy(), B=Vb[0, zc].cpu().numpy(),
                                t=recs[-1]["t"], zc=zc, bbox=[int(v) for v in batch["anatomy_bbox"][0].tolist()],
                                mA=ma, mB=mb)

    # ── aggregate ──
    def agg(side, key):
        v = [r[side][key] for r in recs if r[side][key] is not None and np.isfinite(r[side][key])]
        return float(np.mean(v)) if v else None
    summ = {s: {k: agg(s, k) for k in ("full", "bbox", "motion", "ssim", "cov_frac")} for s in ("A", "B")}
    summ["delta"] = {k: (summ["B"][k] - summ["A"][k]) if (summ["A"][k] and summ["B"][k]) else None
                     for k in ("full", "bbox", "motion", "ssim", "cov_frac")}
    print("\n=== MEAN over", len(recs), "samples (breathing val) ===")
    for k in ("motion", "bbox", "full", "ssim", "cov_frac"):
        print(f"  {k:9s}  518: {summ['A'][k]:.3f}   256: {summ['B'][k]:.3f}   Δ(256-518): {summ['delta'][k]:+.3f}")
    json.dump(dict(ckpt=CKPT, n=len(recs), summary=summ, records=recs),
              open(os.path.join(OUT, "splat_res_compare.json"), "w"), indent=2)

    # ── figure ──
    seqs = sorted(fig_cache)
    nrow = len(seqs)
    fig, axes = plt.subplots(nrow, 4, figsize=(15.5, 4.0 * nrow), squeeze=False)
    for r, sq in enumerate(seqs):
        d = fig_cache[sq]
        z0, z1, y0, y1, x0, x1 = d["bbox"]
        vmax = float(np.percentile(d["gt"][y0:y1, x0:x1], 99.5)) or 1.0
        crop = lambda im: im[y0:y1, x0:x1]
        panels = [("GT (target phase)", crop(d["gt"]), "gray"),
                  (f"518² splat (current)\nmotion {d['mA']['motion']:.1f} / bbox {d['mA']['bbox']:.1f} dB", crop(d["A"]), "gray"),
                  (f"256² native splat\nmotion {d['mB']['motion']:.1f} / bbox {d['mB']['bbox']:.1f} dB", crop(d["B"]), "gray"),
                  (f"|256 − 518| diff\n(where renders disagree)", np.abs(crop(d["B"]) - crop(d["A"])), "magma")]
        for c, (title, im, cm) in enumerate(panels):
            ax = axes[r][c]
            vm = vmax if c < 3 else float(np.percentile(im, 99.5) + 1e-6)
            ax.imshow(im, cmap=cm, vmin=0, vmax=vm); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(title, fontsize=10.5)
        axes[r][0].set_ylabel(f"seq{sq}  t={d['t']}  z={d['zc']}\n(heart-cropped)", fontsize=9.5)
    fig.suptitle(f"Splat resolution: 518² (current) vs native 256²  —  same trained model, no retraining\n"
                 f"mean over {len(recs)} breathing-val samples:  motion {summ['A']['motion']:.2f}→{summ['B']['motion']:.2f} "
                 f"({summ['delta']['motion']:+.2f}),  bbox {summ['A']['bbox']:.2f}→{summ['B']['bbox']:.2f} "
                 f"({summ['delta']['bbox']:+.2f}),  SSIM {summ['A']['ssim']:.3f}→{summ['B']['ssim']:.3f} "
                 f"({summ['delta']['ssim']:+.3f})", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94 if nrow > 1 else 0.88])
    p = os.path.join(OUT, "fig_splat_res_compare.png")
    fig.savefig(p, dpi=120, bbox_inches="tight", facecolor="white"); print("saved", p)


if __name__ == "__main__":
    main()
