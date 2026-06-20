"""Why does OOD OCMR V_canon look CLEANER than in-distribution CMRxRecon val?

Hypothesis: the val visuals apply respiratory simulation (16+/-8 mm SI shift) to the
INPUT slices while the reference stays unshifted, so the model must undo a large
through-plane shift. It under-corrects (docs/07 slope 0.42) -> uneven splat coverage ->
dark spots + blur. OCMR has NO synthetic breathing -> small Delta -> clean splat.

DECISIVE TEST: run the SAME val subjects with breathing ON vs OFF, OCMR-style render
(mid-z V_canon, no reference) + quantified splat-cleanliness metrics. If OFF is clean
and ON is spotty, breathing — not OOD-ness — is the driver. OCMR is then ≈ the OFF regime.
We also compute the identical metrics on OCMR for a 3-way comparison.

Run: PYTHONPATH=training:. micromamba run -n svr python tools/diagnose_ood_clean_paradox.py
"""
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vggt.utils.splat import splat_predictions
from tools.eval_ocmr_inference import (
    load_model, load_cine, percentile_scale, assign_canonical_z, build_batch,
    GRID_SHAPE, MM_PER_NORM, D_CANON, DEFAULT_CKPT,
)

CKPT = "/tmp/vggt_ckpt/resp_z_no_t_last.pt"
if not os.path.exists(CKPT):
    CKPT = DEFAULT_CKPT
OUT = os.path.join(_ROOT, "result", "ood_paradox")
os.makedirs(OUT, exist_ok=True)
N_VAL = 5
N_OCMR = 4
DEV = torch.device("cuda")


def splat_cleanliness(coverage):
    """Robust (median-based, outlier-insensitive) splat-coverage metrics. No reference needed.
    body = voxels with coverage > 10% of the MEDIAN positive coverage (not max → robust to peaks).
    hole_frac: among body voxels, fraction with <25% of body-median coverage (speckle holes).
    disp_iqr: IQR/median of coverage over body (dispersion). body_vox: size of the splatted region."""
    cov = coverage.reshape(-1)
    pos = cov[cov > 0]
    if pos.size == 0:
        return dict(hole_frac=0.0, disp_iqr=0.0, body_vox=0)
    med_pos = np.median(pos)
    body = cov > 0.1 * med_pos
    c = cov[body]
    medb = np.median(c)
    q1, q3 = np.percentile(c, [25, 75])
    return dict(
        hole_frac=float((c < 0.25 * medb).mean()),
        disp_iqr=float((q3 - q1) / (medb + 1e-9)),
        body_vox=int(body.sum()),
    )


def mean_abs_dz(world_points, scanner_coords, intensity):
    """Mean |Δz| (mm) over input anatomy pixels — the through-plane correction magnitude."""
    dz = np.abs((world_points[..., 2] - scanner_coords[..., 2]) * MM_PER_NORM[2])
    m = intensity > 0.05
    return float(dz[m].mean()) if m.any() else 0.0


# ────────────────────────── val (CMRxRecon) ──────────────────────────
def build_val_dataset():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29563")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(_ROOT, "training", "config")):
        cfg = compose(config_name="mri_volume", overrides=["data.augmentation.respiratory.enable=true"])
    from training.data.respiratory import RespiratoryConfig
    rcfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)
    val = instantiate(cfg.data.val, _recursive_=False)
    mri_ds = val.dataset.base_dataset.datasets[0]
    return mri_ds, rcfg


def val_forward(model, mri_ds, rcfg, seq_index, breathing):
    """Build a val batch (filmstrip-style), apply breathing or not, splat at the subject's
    stratified target phase. Returns V_canon, coverage, world_points, scanner_coords, intensity, V_gt."""
    from data.gpu_aug import gpu_augment_batch
    from loss import compute_volume_intensity_loss
    T = mri_ds.gt_grid_shape[0]
    S0 = mri_ds.num_slices
    data = mri_ds.get_data(seq_index=seq_index, img_per_seq=S0)

    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(DEV)

    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    S = imgs.shape[1]
    batch = {"images": imgs, "scanner_coords": st("scanner_coords"),
             "z_indices": st("z_indices"), "t_indices": st("t_indices")}
    phases = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(DEV)
    batch["phases"] = phases.unsqueeze(0)
    batch["timesteps"] = st("timesteps", np.int64)
    batch["slice_indices"] = st("slice_indices", np.int64)
    batch["seq_index"] = torch.tensor([[seq_index]], dtype=torch.int64, device=DEV)
    batch = gpu_augment_batch(batch, None, DEV,
                              respiratory_cfg=(rcfg if breathing else None), train=False)
    t_target = seq_index % T
    t_norm = t_target / max(1, T) * 2.0 - 1.0
    batch["target_t_indices"] = torch.full((1, S, 1), t_norm, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    wp = preds["world_points"].float()
    V_canon, coverage = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    # PSNR on the PROJECT-PRIMARY metric: motion voxels (dynamic across the cardiac cycle),
    # plus full for reference. Motion mask from the UNSHIFTED phases bundle (identical ON/OFF).
    from loss import compute_motion_mask
    mmask = compute_motion_mask(batch["phases"])[0]          # (D,H,W) bool
    Vc, Vg = V_canon[0], phases[t_target]
    se = (Vc - Vg) ** 2
    psnr = lambda mse: float(10.0 * torch.log10(1.0 / mse.clamp(min=1e-10)))
    psnr_motion = psnr(se[mmask].mean()) if mmask.any() else float("nan")
    psnr_full = psnr(se.mean())
    return dict(
        V_canon=Vc.cpu().numpy(), coverage=coverage[0].cpu().numpy(),
        wp=wp[0].cpu().numpy(), sc=batch["scanner_coords"][0].cpu().numpy(),
        intensity=batch["images"][0].mean(1).cpu().numpy(),
        V_gt=Vg.cpu().numpy(), t_target=int(t_target),
        psnr_motion=psnr_motion, psnr_full=psnr_full,
        motion_frac=float(mmask.float().mean()),
    )


# ────────────────────────── OCMR ──────────────────────────
def ocmr_forward(model, subj_dir):
    cine, meta = load_cine(subj_dir)
    scale = percentile_scale(cine)
    z_map = assign_canonical_z(meta["slice_positions_mm"])
    rng = np.random.default_rng(0)
    batch, S, picks = build_batch(cine, meta, scale, z_map, rng, DEV)
    # ED (phase 0) normalizes to -1.0 (t/T*2-1); 0.0 would be phase 6. Match the report convention.
    batch["target_t_indices"] = torch.full((1, S, 1), -1.0, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    wp = preds["world_points"].float()
    V_canon, coverage = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    return dict(
        V_canon=V_canon[0].cpu().numpy(), coverage=coverage[0].cpu().numpy(),
        wp=wp[0].cpu().numpy(), sc=batch["scanner_coords"][0].cpu().numpy(),
        intensity=batch["images"][0].mean(1).cpu().numpy(),
    )


def panel(ax_row, V, cov, title):
    D = V.shape[0]
    zs = [D // 4, D // 2, 3 * D // 4]
    vmax = float(V.max()) or 1e-3
    for i, z in enumerate(zs):
        ax_row[i].imshow(V[z], cmap="gray", vmin=0, vmax=vmax); ax_row[i].axis("off")
        ax_row[i].set_title(f"{title} z={z}", fontsize=7)
    ax_row[3].imshow(cov[D // 2], cmap="magma"); ax_row[3].axis("off")
    ax_row[3].set_title("coverage (mid-z)", fontsize=7)


def main():
    model = load_model(CKPT, DEV)
    rows = {"val_breath_on": [], "val_breath_off": [], "ocmr": []}

    print("=== building val dataset (may rebuild monai cache, ~min) ===", flush=True)
    mri_ds, rcfg = build_val_dataset()
    print(f"val MRIDataset ready; respiratory cfg amplitude={rcfg.amplitude_mm}+/-{rcfg.amplitude_jitter}mm", flush=True)

    for i in range(N_VAL):
        on = val_forward(model, mri_ds, rcfg, i, breathing=True)
        off = val_forward(model, mri_ds, rcfg, i, breathing=False)
        m_on = {**splat_cleanliness(on["coverage"]), "mean_dz_mm": mean_abs_dz(on["wp"], on["sc"], on["intensity"])}
        m_off = {**splat_cleanliness(off["coverage"]), "mean_dz_mm": mean_abs_dz(off["wp"], off["sc"], off["intensity"])}
        # recon error vs unshifted reference (val only)
        for r, o in ((m_on, on), (m_off, off)):
            msk = o["V_gt"] > 0.05
            r["recon_l1"] = float(np.abs(o["V_canon"] - o["V_gt"])[msk].mean()) if msk.any() else 0.0
        rows["val_breath_on"].append(m_on); rows["val_breath_off"].append(m_off)
        print(f"val subj{i} t{on['t_target']}: ON hole={m_on['hole_frac']:.3f} iqr={m_on['disp_iqr']:.2f} body={m_on['body_vox']} dz={m_on['mean_dz_mm']:.1f}mm L1={m_on['recon_l1']:.4f} | "
              f"OFF hole={m_off['hole_frac']:.3f} iqr={m_off['disp_iqr']:.2f} body={m_off['body_vox']} dz={m_off['mean_dz_mm']:.1f}mm L1={m_off['recon_l1']:.4f}", flush=True)
        # render comparison for first 3 subjects
        if i < 3:
            fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
            panel(axes[0], on["V_canon"], on["coverage"], "ON")
            panel(axes[1], off["V_canon"], off["coverage"], "OFF")
            fig.suptitle(f"val subj{i} (t={on['t_target']}) — breathing ON (top) vs OFF (bottom)", fontsize=10)
            fig.tight_layout(); fig.savefig(os.path.join(OUT, f"val_subj{i}_on_vs_off.png"), dpi=95); plt.close(fig)

    ocmr_dirs = sorted(d for d in
                       [os.path.join(_ROOT, "scratch/data/ocmr/recon", x) for x in
                        ["us_0084_1_5T", "us_0173_pt_1_5T", "us_0197_pt_1_5T", "us_0183_pt_1_5T"]]
                       if os.path.exists(os.path.join(d, "sax_cine.nii.gz")))[:N_OCMR]
    for d in ocmr_dirs:
        o = ocmr_forward(model, d)
        m = {**splat_cleanliness(o["coverage"]), "mean_dz_mm": mean_abs_dz(o["wp"], o["sc"], o["intensity"])}
        rows["ocmr"].append(m)
        print(f"ocmr {os.path.basename(d)}: hole={m['hole_frac']:.3f} iqr={m['disp_iqr']:.2f} body={m['body_vox']} dz={m['mean_dz_mm']:.1f}mm", flush=True)
        if d == ocmr_dirs[0]:
            globals()["_ocmr_demo"] = o
        fig, axes = plt.subplots(1, 4, figsize=(10, 2.7))
        panel(axes, o["V_canon"], o["coverage"], "OCMR")
        fig.suptitle(f"OCMR {os.path.basename(d)} (target t=0)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"ocmr_{os.path.basename(d)}.png"), dpi=95); plt.close(fig)

    # ── decisive 3-way matched figure: same render for val-ON / val-OFF / OCMR ──
    demo_on = val_forward(model, mri_ds, rcfg, 0, breathing=True)
    demo_off = val_forward(model, mri_ds, rcfg, 0, breathing=False)
    demo_oc = globals().get("_ocmr_demo")
    fig, axes = plt.subplots(3, 2, figsize=(6, 8.6))
    for r, (lbl, o) in enumerate([("val breathing ON", demo_on),
                                  ("val breathing OFF", demo_off),
                                  ("OCMR (OOD, no synth breathing)", demo_oc)]):
        D = o["V_canon"].shape[0]; mid = D // 2
        axes[r][0].imshow(o["V_canon"][mid], cmap="gray", vmin=0, vmax=float(o["V_canon"].max()) or 1e-3)
        axes[r][0].set_ylabel(lbl, fontsize=9); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        axes[r][0].set_title("V_canon (mid-z)", fontsize=8)
        axes[r][1].imshow(o["coverage"][mid], cmap="magma"); axes[r][1].axis("off")
        axes[r][1].set_title("splat coverage (mid-z)", fontsize=8)
    fig.suptitle("Why OCMR looks cleaner: breathing scatters slices → speckled coverage", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "DECISIVE_3way.png"), dpi=110); plt.close(fig)

    def agg(key):
        rs = rows[key]
        return {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]}
    summary = {k: agg(k) for k in rows}
    json.dump(summary, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print("\n=== SUMMARY (means) ===")
    for k, v in summary.items():
        print(f"{k:16s}", v)
    print("DONE")


if __name__ == "__main__":
    main()
