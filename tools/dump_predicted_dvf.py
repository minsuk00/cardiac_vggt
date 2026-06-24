"""Dump the TRUE predicted DVF (in mm) for the t59w6nqy run over the val set.

No GT motion needed — this is a magnitude sanity check: feed real (breathing-
augmented) input slices, run the trained model, recover the predicted residual
Δ = world_points - scanner_coords, convert to physical mm per axis, and report
whether the displacements sit in a reasonable range (not ~0, not absurd).

Matches the run exactly: respiratory ON (defaults), use_z=T use_t=F use_target_t=T,
aggregator-finetuned weights (only patch_embed was frozen — irrelevant at eval).

mm/norm (splat align_corners, pos=(p+1)/2*(size-1)):
    in-plane (256 vox @1.4mm) → 178.5 mm per norm unit (Δx, Δy)
    through  (12 vox  @12.0mm) →  66.0 mm per norm unit (Δz)

Run: PYTHONPATH=training:. micromamba run -n svr python tools/dump_predicted_dvf.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT))

from data.datasets.mri_dataset import MRIDataset  # noqa: E402
from data.gpu_aug import gpu_augment_batch  # noqa: E402
from data.respiratory import RespiratoryConfig  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402

CKPT = str(ROOT / "scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt")
DATA_ROOT = str(ROOT / "scratch/data/CMRxRecon2024/Cine_combined")
SPLIT_FILE = str(ROOT / "training/splits/random_8_1_1.txt")
OUT_PNG = str(ROOT / "result/predicted_dvf_ranges.png")
NUM_SLICES = 12
IN_PLANE_MM = (256 - 1) / 2.0 * 1.4      # 178.5
THROUGH_MM = (12 - 1) / 2.0 * 12.0       # 66.0 (12mm true pitch)
MASK_THRESH = 0.05                        # input-intensity gate (real anatomy pixels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False, "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=NUM_SLICES, target_size=518)
    n_subj = len(ds.subjects)
    print(f"val subjects: {n_subj}   device: {device}")

    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=False,
                 use_target_t_pose_embedding=True, train_on_residual_dvf=True).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded ckpt (epoch={ck.get('epoch','?')})  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    resp_cfg = RespiratoryConfig(enable=True)   # run used config defaults

    dx_all, dy_all, dz_all = [], [], []         # masked per-pixel mm
    applied_d, slot_dz, slot_inplane = [], [], []  # per-slot summaries

    for i in range(n_subj):
        data = ds.get_data(seq_index=i, img_per_seq=NUM_SLICES)

        def st(k, dt=np.float32):
            return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(device)

        batch = {
            "images": st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0,
            "scanner_coords": st("scanner_coords"),
            "z_indices": st("z_indices"),
            "t_indices": st("t_indices"),
            "target_t_indices": st("target_t_indices"),
            "phases": torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).unsqueeze(0).to(device),
            "timesteps": st("timesteps", np.int64),
            "slice_indices": st("slice_indices", np.int64),
            "seq_index": torch.tensor([[i]], dtype=torch.int64, device=device),
        }
        batch = gpu_augment_batch(batch, None, device, respiratory_cfg=resp_cfg, train=False)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        wp = preds["world_points"].float()                       # (1,S,H,W,3)
        delta = (wp - batch["scanner_coords"]).float()           # normalized residual
        dmm = delta.clone()
        dmm[..., 0] *= IN_PLANE_MM
        dmm[..., 1] *= IN_PLANE_MM
        dmm[..., 2] *= THROUGH_MM
        dmm = dmm[0].cpu().numpy()                                # (S,H,W,3)

        imgs = batch["images"][0].mean(0 if False else 1).cpu().numpy()  # (S,H,W) channel-mean
        mask = imgs > MASK_THRESH                                  # (S,H,W)
        applied = batch["resp_disp_mm"][0].norm(dim=-1).cpu().numpy()  # (S,) mm

        S = dmm.shape[0]
        for s in range(S):
            m = mask[s]
            if m.sum() < 50:
                continue
            dx = dmm[s, ..., 0][m]; dy = dmm[s, ..., 1][m]; dz = dmm[s, ..., 2][m]
            dx_all.append(dx); dy_all.append(dy); dz_all.append(dz)
            applied_d.append(float(applied[s]))
            slot_dz.append(float(np.mean(dz)))                    # signed mean Δz
            slot_inplane.append(float(np.mean(np.sqrt(dx**2 + dy**2))))
        if (i + 1) % 5 == 0:
            print(f"  ...{i+1}/{n_subj} subjects")

    dx_all = np.concatenate(dx_all); dy_all = np.concatenate(dy_all); dz_all = np.concatenate(dz_all)
    inplane_mag = np.sqrt(dx_all**2 + dy_all**2)
    applied_d = np.array(applied_d); slot_dz = np.array(slot_dz); slot_inplane = np.array(slot_inplane)

    def stats(name, a):
        ab = np.abs(a)
        print(f"  {name:14s} mean|·|={ab.mean():6.2f}  p50={np.percentile(ab,50):6.2f}  "
              f"p95={np.percentile(ab,95):6.2f}  p99={np.percentile(ab,99):6.2f}  max={ab.max():7.2f}  (mm)")

    print("\n=== per-pixel predicted |Δ| over anatomy pixels (mm) ===")
    stats("Δx in-plane", dx_all)
    stats("Δy in-plane", dy_all)
    stats("|Δ| in-plane", inplane_mag)
    stats("Δz through", dz_all)

    print("\n=== breathing test: per-slot signed mean Δz vs applied |d| ===")
    if applied_d.std() > 1e-6:
        r = float(np.corrcoef(applied_d, slot_dz)[0, 1])
        slope, intercept = np.polyfit(applied_d, slot_dz, 1)
        print(f"  n_slots={len(applied_d)}   corr(|d|, meanΔz)={r:+.3f}")
        print(f"  linear fit: meanΔz ≈ {slope:+.3f}·|d| {intercept:+.2f} mm   (slope ~±1 = full rigid correction)")
        for lo, hi in [(0, 2), (2, 8), (8, 16), (16, 30)]:
            sel = (applied_d >= lo) & (applied_d < hi)
            if sel.sum():
                print(f"    applied |d|∈[{lo:2d},{hi:2d})mm  n={sel.sum():3d}  "
                      f"meanΔz={slot_dz[sel].mean():+6.2f}  mean|Δz|={np.abs(slot_dz[sel]).mean():5.2f}  "
                      f"mean in-plane|Δ|={slot_inplane[sel].mean():5.2f} mm")

    # ── scatter figure ───────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), dpi=120)
    ax[0].scatter(applied_d, slot_dz, s=14, alpha=0.5)
    xs = np.linspace(0, applied_d.max(), 50)
    ax[0].plot(xs, slope * xs + intercept, "r-", lw=2, label=f"fit slope={slope:+.2f}")
    ax[0].axhline(0, color="#bbb", lw=0.8)
    ax[0].set_xlabel("applied breathing |d| (mm)"); ax[0].set_ylabel("predicted signed mean Δz (mm)")
    ax[0].set_title(f"breathing: through-plane correction vs input (corr={r:+.2f})"); ax[0].legend(fontsize=8)
    ax[1].hist(np.abs(dz_all), bins=80, alpha=0.6, label="|Δz| through", color="#d62728", density=True)
    ax[1].hist(inplane_mag, bins=80, alpha=0.6, label="|Δ| in-plane", color="#1f77b4", density=True)
    ax[1].set_xlabel("predicted |Δ| (mm)"); ax[1].set_ylabel("density"); ax[1].set_xlim(0, 30)
    ax[1].set_title("predicted displacement magnitude distribution"); ax[1].legend(fontsize=8)
    fig.tight_layout(); os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True); fig.savefig(OUT_PNG)
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
