"""Standalone per-target-phase analysis of two trained models.

  M1 = allphases_aggft (fc8d065g) — z/t/target_t all ON, all 12 target phases.
  M2 = t0t7_aggft_no_zt (vry47r4f) — z/t OFF, target_t ON, phases {0,7}.

Not a comparison — characterizes each. Runs inference on val subjects across
target phases, computes full/bbox/motion PSNR + SSIM, and renders per-phase
reconstruction figures. Outputs to _html/assets/ + a metrics JSON.
"""
import os, sys, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss, compute_motion_mask

DATA_ROOT = "scratch/data/CMRxRecon2024/Cine_combined"
SPLIT = "training/splits/random_8_1_1.txt"
OUT = "_html/assets"
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

MODELS = {
    "m1_allphases": dict(
        ckpt="scratch/logs/219576158_mri_volume_allphases_aggft/ckpts/checkpoint_last.pt",
        flags=dict(use_z_pose_embedding=True, use_t_pose_embedding=True, use_target_t_pose_embedding=True),
        phases=list(range(12)),
    ),
    "m2_nozt": dict(
        ckpt="scratch/logs/219575690_mri_volume_t0t7_aggft_no_zt/ckpts/checkpoint_last.pt",
        flags=dict(use_z_pose_embedding=False, use_t_pose_embedding=False, use_target_t_pose_embedding=True),
        phases=[0, 7],
    ),
}
N_METRIC_SUBJ = 8          # val subjects for per-phase metric aggregation
VIS_SUBJ = [0, 1]          # val subjects rendered in detail


def build_ds():
    conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                             "rescale_aug": False, "landscape_check": False,
                             "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(conf, DATA_ROOT, split="val", split_file=SPLIT,
                      mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)


def make_batch(ds, seq_index, t_target):
    ds.t_target_fixed = t_target
    data = ds.get_data(seq_index=seq_index, img_per_seq=12)
    st = lambda k, dt=np.float32: torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    batch = {
        "images": st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0,
        "scanner_coords": st("scanner_coords"), "world_points": st("world_points"),
        "z_indices": st("z_indices"), "t_indices": st("t_indices"),
        "target_t_indices": st("target_t_indices"),
        "timesteps": st("timesteps", np.int64), "slice_indices": st("slice_indices", np.int64),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
        "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0),
        "t_target": torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0),
    }
    phases = torch.from_numpy(data["phases"].astype(np.float32)).unsqueeze(0)  # (1,T,D,H,W)
    return batch, phases, data


def psnr(a, b, m=None):
    if m is None:
        mse = ((a - b) ** 2).mean()
    else:
        m = m.float(); n = m.sum().clamp(min=1)
        mse = (((a - b) ** 2) * m).sum() / n
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def ssim3d(a, b):
    try:
        from fused_ssim import fused_ssim3d
        return float(fused_ssim3d(a[None, None].float().contiguous(), b[None, None].float().contiguous(), train=False))
    except Exception:
        return float("nan")


def load_model(cfg):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             train_on_residual_dvf=True, **cfg["flags"]).to(DEV)
    ck = torch.load(cfg["ckpt"], map_location=DEV, weights_only=False)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m


@torch.no_grad()
def run_one(model, ds, seq_index, t_target):
    batch, phases, data = make_batch(ds, seq_index, t_target)
    bd = {k: v.to(DEV) for k, v in batch.items()}
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(bd["images"], batch=bd)
    preds["world_points"] = preds["world_points"].float()
    out = compute_volume_intensity_loss(preds, bd, grid_shape=(12, 256, 256), tv_weight=0.1)
    Vc = out["V_canon"][0].float().cpu()
    Vg = out["V_gt"][0].float().cpu()
    mot = compute_motion_mask(phases)[0].cpu()   # (D,H,W) bool
    anat = (Vg > 1e-3)
    met = dict(full=psnr(Vc, Vg), bbox=psnr(Vc, Vg, anat),
               motion=psnr(Vc, Vg, mot), ssim=ssim3d(Vc, Vg),
               t_target=int(t_target), subj=os.path.basename(data["subject_path"]) if "subject_path" in data else f"val{seq_index}")
    return Vc.numpy(), Vg.numpy(), mot.numpy(), anat.numpy(), met


def best_z(vg):
    return int(np.argmax(vg.reshape(vg.shape[0], -1).sum(1)))


def main():
    ds = build_ds()
    subj_names = [os.path.basename(p) for p in ds.subjects]
    print(f"val subjects: {len(subj_names)}", flush=True)
    results = {}
    for mkey, cfg in MODELS.items():
        print(f"\n===== {mkey} :: loading {cfg['ckpt']}", flush=True)
        model = load_model(cfg)
        phases = cfg["phases"]
        # ---- metrics over N_METRIC_SUBJ subjects x phases ----
        per = []
        for s in range(N_METRIC_SUBJ):
            for t in phases:
                Vc, Vg, mot, anat, met = run_one(model, ds, s, t)
                met["subj_idx"] = s
                per.append(met)
                print(f"  {mkey} subj{s}({subj_names[s]}) t={t}: full={met['full']:.2f} bbox={met['bbox']:.2f} motion={met['motion']:.2f} ssim={met['ssim']:.3f}", flush=True)
        results[mkey] = per
        json.dump(results, open(f"{OUT}/analysis_metrics.json", "w"), indent=1)

        # ---- per-phase mean curve ----
        ts = phases
        def agg(metric):
            return [float(np.mean([r[metric] for r in per if r["t_target"] == t])) for t in ts], \
                   [float(np.std([r[metric] for r in per if r["t_target"] == t])) for t in ts]
        fig, ax = plt.subplots(figsize=(max(5, len(ts)*0.8), 4))
        for metric, c in [("motion", "#d62728"), ("bbox", "#1f77b4"), ("full", "#2ca02c")]:
            mu, sd = agg(metric)
            ax.errorbar(ts, mu, yerr=sd, marker="o", capsize=3, label=metric, color=c)
        ax.set_xlabel("target phase t"); ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"{mkey}: per-target-phase PSNR (mean±std, {N_METRIC_SUBJ} val subj)")
        ax.set_xticks(ts); ax.grid(alpha=.3); ax.legend()
        plt.tight_layout(); plt.savefig(f"{OUT}/{mkey}_perphase.png", dpi=120); plt.close()
        print(f"  wrote {mkey}_perphase.png", flush=True)

        # ---- detailed visuals ----
        if mkey == "m1_allphases":
            # 3 x 12 montage (GT/pred/diff rows, 12 phase cols) at fixed mid-z, per VIS_SUBJ
            for s in VIS_SUBJ:
                # use a reference GT (t0) to set z & vmax
                Vc0, Vg0, _, _, _ = run_one(model, ds, s, 0)
                z = best_z(Vg0); vmax = float(np.percentile(Vg0[z], 99.5))
                fig, ax = plt.subplots(3, 12, figsize=(20, 5.2))
                for j, t in enumerate(range(12)):
                    Vc, Vg, _, _, met = run_one(model, ds, s, t)
                    ax[0, j].imshow(Vg[z], cmap="gray", vmin=0, vmax=vmax)
                    ax[0, j].set_title(f"t{t}\nGTbbox{met['bbox']:.1f}", fontsize=7)
                    ax[1, j].imshow(Vc[z], cmap="gray", vmin=0, vmax=vmax)
                    ax[2, j].imshow(np.abs(Vc[z]-Vg[z]), cmap="magma", vmin=0, vmax=vmax*0.5)
                    for r in range(3):
                        ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
                for r, lab in enumerate(["V_gt", "V_canon", "|err|"]):
                    ax[r, 0].set_ylabel(lab, fontsize=10)
                fig.suptitle(f"{mkey} — subj{s} ({subj_names[s]}) reconstruction across all 12 target phases (mid-z={z})", fontsize=11)
                plt.tight_layout(); plt.savefig(f"{OUT}/m1_cycle_subj{s}.png", dpi=110); plt.close()
                print(f"  wrote m1_cycle_subj{s}.png", flush=True)
        else:
            # no_zt: per-z montage for t=0 and t=7, per VIS_SUBJ
            for s in VIS_SUBJ:
                for t in [0, 7]:
                    Vc, Vg, mot, anat, met = run_one(model, ds, s, t)
                    zs = [z for z in range(Vg.shape[0]) if (Vg[z] > 1e-3).sum() > 50]
                    vmax = float(np.percentile(Vg[Vg > 1e-3], 99.5))
                    fig, ax = plt.subplots(3, len(zs), figsize=(1.5*len(zs), 4.8))
                    if len(zs) == 1: ax = ax.reshape(3, 1)
                    for j, z in enumerate(zs):
                        ax[0, j].imshow(Vg[z], cmap="gray", vmin=0, vmax=vmax); ax[0, j].set_title(f"z{z}", fontsize=8)
                        ax[1, j].imshow(Vc[z], cmap="gray", vmin=0, vmax=vmax)
                        ax[2, j].imshow(np.abs(Vc[z]-Vg[z]), cmap="magma", vmin=0, vmax=vmax*0.5)
                        for r in range(3): ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
                    for r, lab in enumerate(["V_gt", "V_canon", "|err|"]):
                        ax[r, 0].set_ylabel(lab, fontsize=10)
                    fig.suptitle(f"{mkey} — subj{s} ({subj_names[s]}) t={t}: per-z  (full={met['full']:.1f} bbox={met['bbox']:.1f} motion={met['motion']:.1f} dB)", fontsize=10)
                    plt.tight_layout(); plt.savefig(f"{OUT}/m2_subj{s}_t{t}.png", dpi=110); plt.close()
                    print(f"  wrote m2_subj{s}_t{t}.png", flush=True)
        del model; torch.cuda.empty_cache()

    json.dump(results, open(f"{OUT}/analysis_metrics.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
