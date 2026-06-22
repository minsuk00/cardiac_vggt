"""Settle 3 questions:

Q1. Rendering gap demo. Same V_refined volume rendered FOUR ways:
      [max + native]  [max + 2x bilinear upscale]
      [pct99.5 + native] [pct99.5 + 2x bilinear upscale]
    Honest: native 256x256 is what the model actually produces. Upscale is interpolation.
    Percentile clipping just changes where the white point sits; max can let one bright
    outlier voxel kill contrast. Now you can see exactly what each choice does.

Q2. OLD pre-refiner ckpt V_canon vs NEW joint-refiner ckpt V_refined on N=5 OCMR.
    Both windowings shown so the comparison isn't biased by display choice.
    Native 256x256 — no upscale.

Q3. Per-subject all-z filmstrips of V_canon vs V_refined (new ckpt) on N=5 val ON and val OFF.
    Native 256x256. Percentile-99.5 windowing with disclosure in title.
"""
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from vggt.models.vggt import VGGT
from vggt.utils.splat import splat_predictions
from tools.eval_ocmr_inference import load_cine, percentile_scale, assign_canonical_z, build_batch
from tools.diagnose_ood_clean_paradox import build_val_dataset

NEW_CKPT = "/home/minsukc/vggt/scratch/logs/218349151_mri_refiner_joint/ckpts/checkpoint_last.pt"
OLD_CKPT = "/home/minsukc/vggt/scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
OUT = os.path.join(_ROOT, "result", "4way_refiner")
DEV = torch.device("cuda")
GRID_SHAPE = (12, 256, 256)

OCMR_SUBJECTS = ["us_0084_1_5T", "us_0173_pt_1_5T", "us_0183_pt_1_5T",
                 "us_0197_pt_1_5T", "us_0169_pt_1_5T"]
VAL_SEQS = [0, 1, 2, 3, 4]


def build_model(use_refiner):
    return VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
        enable_refiner=use_refiner, refiner_use_coverage=use_refiner, grid_shape=GRID_SHAPE,
    ).to(DEV).eval()


def load_state(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    return model


def _forward(model, batch, target_t=-1.0, gt_phase=0):
    """Returns (V_canon, V_refined or None, V_gt or None) at native 256x256x12.
    For val, batch["phases"] is (1, T, D, H, W) and respiratory aug leaves it UNSHIFTED
    (only input slices get the breathing sim), so V_gt = batch["phases"][0, gt_phase].
    """
    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), target_t, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    V_can = preds.get("V_canon")
    if V_can is None:
        wp = preds["world_points"].float()
        V_can, _cov = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    V_can = V_can[0].float().cpu().numpy()
    V_ref = preds["V_refined"][0].float().cpu().numpy() if "V_refined" in preds else None
    V_gt = (batch["phases"][0, gt_phase].float().cpu().numpy()
            if "phases" in batch else None)
    return V_can, V_ref, V_gt


def val_batch(ds, rcfg, seq_index, breathing):
    from data.gpu_aug import gpu_augment_batch
    S0 = ds.num_slices
    data = ds.get_data(seq_index=seq_index, img_per_seq=S0)
    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(DEV)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    batch = {"images": imgs, "scanner_coords": st("scanner_coords"),
             "z_indices": st("z_indices"), "t_indices": st("t_indices"),
             "timesteps": st("timesteps", np.int64),
             "slice_indices": st("slice_indices", np.int64),
             "phases": torch.from_numpy(np.asarray(data["phases"]).astype(np.float32))
                        .to(DEV).unsqueeze(0),
             "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=DEV)}
    return gpu_augment_batch(batch, None, DEV,
                              respiratory_cfg=(rcfg if breathing else None), train=False)


def ocmr_batch(subj_dir):
    cine, meta = load_cine(subj_dir)
    scale = percentile_scale(cine)
    z_map = assign_canonical_z(meta["slice_positions_mm"])
    rng = np.random.default_rng(0)
    batch, _S, _picks = build_batch(cine, meta, scale, z_map, rng, DEV)
    return batch


# ───────── windowing helpers ─────────
def window_max(sl, V):
    """Wandb-style: max over the whole volume."""
    vmax = float(V.max()) or 1e-3
    return np.clip(sl / vmax, 0, 1)


def window_pct(sl, V):
    """Percentile-99.5 (top 0.5% saturate to white, bottom 1% to black)."""
    hi = float(np.percentile(V, 99.5))
    lo = float(np.percentile(V, 1.0))
    return np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1)


def upscale_2x(a01):
    """Bilinear 2x upscale via PIL (cosmetic only; documented as such)."""
    a8 = (a01 * 255).astype(np.uint8)
    up = np.asarray(Image.fromarray(a8, "L").resize(
        (a8.shape[1] * 2, a8.shape[0] * 2), Image.BILINEAR)) / 255.0
    return up


# ───────── Q1: 4-way render demo on the SAME volume + V_gt reference ─────────
def render_q1_rendering_demo(V_refined, V_gt, label, path):
    """5 panels of the same mid-z slice: V_gt + 4 rendering choices for V_refined."""
    mid = V_refined.shape[0] // 2
    sl = V_refined[mid]
    gt_sl = V_gt[mid] if V_gt is not None else None
    panels = [
        ("V_gt (reference)\npct99.5 windowing",
         window_pct(gt_sl, V_gt) if V_gt is not None else np.zeros_like(sl)),
        ("V_refined: max windowing\nnative 256x256\n(wandb does this)",
         window_max(sl, V_refined)),
        ("V_refined: max windowing\n2x bilinear upscale\n(cosmetic only)",
         upscale_2x(window_max(sl, V_refined))),
        ("V_refined: pct99.5 windowing\nnative 256x256",
         window_pct(sl, V_refined)),
        ("V_refined: pct99.5 windowing\n2x bilinear upscale",
         upscale_2x(window_pct(sl, V_refined))),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Q1 — V_gt + same V_refined volume ({label}), 4 rendering choices",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ───────── Q2: 3-way refiner-isolation test on OCMR ─────────
def render_q2_refiner_isolation(old_canons, new_canons, new_refineds, labs, path):
    """3 rows × N subjects, mid-z, pct99.5 windowing:
      Row 1: OLD ckpt V_canon (the _html/13 model)
      Row 2: NEW ckpt V_canon (raw splat from the refiner model, BEFORE refiner runs)
      Row 3: NEW ckpt V_refined (post-refiner)
    Row1 ≈ Row2 + Row3 worse  → refiner is the degrader.
    Row1 ≠ Row2                → it's the new model's aggregator/point head, not the refiner.
    """
    N = len(labs)
    fig, axes = plt.subplots(3, N, figsize=(N * 3.0, 9.0))
    sources = [
        ("OLD ckpt\nV_canon\n(the _html/13 model)", old_canons),
        ("NEW ckpt\nV_canon\n(raw splat, pre-refiner)", new_canons),
        ("NEW ckpt\nV_refined\n(post-refiner)", new_refineds),
    ]
    for r, (rlbl, vols) in enumerate(sources):
        for i, (lbl, V) in enumerate(zip(labs, vols)):
            mid = V.shape[0] // 2
            axes[r, i].imshow(window_pct(V[mid], V), cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
            if r == 0:
                axes[r, i].set_title(lbl, fontsize=8)
            if i == 0:
                axes[r, i].text(-0.20, 0.5, rlbl, transform=axes[r, i].transAxes,
                                fontsize=8, ha="right", va="center")
    fig.suptitle("Q2 — OCMR refiner-isolation test  (native 256x256, pct99.5 per panel)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ───────── Q3: per-subject filmstrips combining ON and OFF ─────────
def render_q3_per_subject(subj_label, gt_on, canon_on, ref_on, gt_off, canon_off, ref_off, path):
    """One subject, 6 rows × 12 z cols (large tiles for readability):
      [ON V_gt, ON V_canon, ON V_refined, OFF V_gt, OFF V_canon, OFF V_refined]
    Native 256x256, percentile-99.5 per volume.
    """
    D = 12
    rows = [
        ("ON V_gt", gt_on), ("ON V_canon", canon_on), ("ON V_refined", ref_on),
        ("OFF V_gt", gt_off), ("OFF V_canon", canon_off), ("OFF V_refined", ref_off),
    ]
    fig, axes = plt.subplots(6, D, figsize=(D * 2.0, 6 * 2.0))
    for r, (rlbl, V) in enumerate(rows):
        for z in range(D):
            ax = axes[r, z]
            if V is not None:
                ax.imshow(window_pct(V[z], V), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"z{z}", fontsize=10)
            if z == 0:
                ax.text(-0.10, 0.5, rlbl, transform=ax.transAxes,
                        fontsize=10, ha="right", va="center")
    fig.suptitle(f"Q3 — val {subj_label} @ ED  (native 256x256, pct99.5 per volume)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)

    print("\n[setup] Loading NEW ckpt (joint refiner) ...")
    new_model = load_state(build_model(use_refiner=True), NEW_CKPT)
    val_ds, rcfg = build_val_dataset()

    # ── Q1: rendering demo on one val OFF volume ──
    print("\n[Q1] rendering demo on val seq0 OFF ...")
    b = val_batch(val_ds, rcfg, 0, breathing=False)
    _, V_ref_demo, V_gt_demo = _forward(new_model, b)
    render_q1_rendering_demo(V_ref_demo, V_gt_demo, "val seq0 OFF (new ckpt)",
                             os.path.join(OUT, "Q1_rendering_demo.png"))

    # ── Q3: per-subject filmstrips combining ON and OFF ──
    print("\n[Q3] per-subject filmstrips (V_gt | V_canon | V_refined, all z, ON + OFF) ...")
    for i in VAL_SEQS:
        b_on = val_batch(val_ds, rcfg, i, breathing=True)
        Vc_on, Vr_on, Vg_on = _forward(new_model, b_on)
        b_off = val_batch(val_ds, rcfg, i, breathing=False)
        Vc_off, Vr_off, Vg_off = _forward(new_model, b_off)
        print(f"  seq{i}: V_gt {Vg_on.shape}, V_canon {Vc_on.shape}, V_refined {Vr_on.shape}",
              flush=True)
        render_q3_per_subject(f"seq{i}", Vg_on, Vc_on, Vr_on, Vg_off, Vc_off, Vr_off,
                              os.path.join(OUT, f"Q3_filmstrip_subj{i}.png"))

    # ── Q2: OLD vs NEW on OCMR ──
    print("\n[Q2] Loading OLD ckpt (pre-refiner) ...")
    old_model = load_state(build_model(use_refiner=False), OLD_CKPT)

    print("\n[Q2] OCMR refiner-isolation: OLD V_canon | NEW V_canon | NEW V_refined ...")
    old_canons, new_canons, new_refineds, labs_oc = [], [], [], []
    for sub in OCMR_SUBJECTS:
        sd = os.path.join(_ROOT, "scratch/data/ocmr/recon", sub)
        if not os.path.isdir(sd):
            print(f"  skip {sub}"); continue
        b_old = ocmr_batch(sd)
        V_old_canon, _, _ = _forward(old_model, b_old)
        b_new = ocmr_batch(sd)
        V_new_canon, V_new_refined, _ = _forward(new_model, b_new)
        old_canons.append(V_old_canon)
        new_canons.append(V_new_canon)
        new_refineds.append(V_new_refined)
        labs_oc.append(sub)
    render_q2_refiner_isolation(old_canons, new_canons, new_refineds, labs_oc,
                                os.path.join(OUT, "Q2_OCMR_refiner_isolation.png"))

    print("\ndone.")


if __name__ == "__main__":
    main()
