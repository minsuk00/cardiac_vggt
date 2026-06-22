"""3-ckpt comparison across all 4 modes:

  Row 1: OLD ckpt V_canon       (218747856 — backbone only, no refiner; the _html/13 model)
  Row 2: bpblrai2 V_refined     (218246076 — backbone FROZEN at OLD's state; SSIM refiner trained on top)
  Row 3: NEW joint V_refined    (218349151 — backbone + refiner trained jointly with L1)

Outputs (one PNG per mode, 3 rows × N subjects, native 256x256, pct99.5):
  result/4way_refiner/three_ckpt_val_ON.png
  result/4way_refiner/three_ckpt_val_OFF.png
  result/4way_refiner/three_ckpt_OCMR.png
  result/4way_refiner/three_ckpt_Goettingen.png
"""
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from omegaconf import OmegaConf

from vggt.models.vggt import VGGT
from vggt.utils.splat import splat_predictions
from data.datasets.mri_dataset import MRIDataset
from tools.eval_ocmr_inference import load_cine, percentile_scale, assign_canonical_z, build_batch
from tools.diagnose_ood_clean_paradox import build_val_dataset

OLD_CKPT = "/home/minsukc/vggt/scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
SSIM_CKPT = "/home/minsukc/vggt/scratch/logs/218246076_mri_refiner_frozen_ssim/ckpts/checkpoint_last.pt"
NEW_CKPT = "/home/minsukc/vggt/scratch/logs/218349151_mri_refiner_joint/ckpts/checkpoint_last.pt"
OUT = os.path.join(_ROOT, "result", "4way_refiner")
DEV = torch.device("cuda")
GRID_SHAPE = (12, 256, 256)

OCMR_SUBJECTS = ["us_0084_1_5T", "us_0173_pt_1_5T", "us_0183_pt_1_5T",
                 "us_0197_pt_1_5T", "us_0169_pt_1_5T"]
GOTT_ROOT = "/home/minsukc/vggt/scratch/data/goettingen/canonical_subjects"
GOTT_SPLIT = GOTT_ROOT + "/gott_split.txt"
GOTT_SUBJECTS = ["vol0001_vis1", "vol0002_vis1", "vol0003_vis1",
                 "vol0009_vis1", "vol0023_vis1"]
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


def _forward(model, batch, target_t=-1.0, want="refined"):
    """want: 'canon' or 'refined'."""
    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), target_t, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    if want == "refined" and "V_refined" in preds:
        return preds["V_refined"][0].float().cpu().numpy()
    V = preds.get("V_canon")
    if V is None:
        wp = preds["world_points"].float()
        V, _cov = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    return V[0].float().cpu().numpy()


# ── batch builders (identical to other scripts; deterministic seeds) ──
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


def goettingen_batch(ds, subj):
    idx = next(i for i, s in enumerate(ds.subjects) if subj in s)
    data = ds.get_data(seq_index=idx, img_per_seq=12)
    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(DEV)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    return {"images": imgs, "scanner_coords": st("scanner_coords"),
            "z_indices": st("z_indices"), "t_indices": st("t_indices"),
            "slice_indices": torch.from_numpy(
                np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0).to(DEV)}


def build_gott_dataset():
    common_conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                                    "rescale_aug": False, "landscape_check": False,
                                    "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(common_conf, GOTT_ROOT, split="val", split_file=GOTT_SPLIT,
                      mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)


def window_pct(sl, V):
    hi = float(np.percentile(V, 99.5))
    lo = float(np.percentile(V, 1.0))
    return np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1)


def render_3row(old_canons, ssim_refs, new_refs, labels, title, path):
    """3 rows × N subjects, mid-z, pct99.5 per panel."""
    N = len(labels)
    fig, axes = plt.subplots(3, N, figsize=(N * 3.0, 9.2))
    if N == 1:
        axes = np.array(axes).reshape(3, 1)
    sources = [
        ("OLD ckpt\nV_canon\n(backbone only, no refiner)\n[_html/13 model]", old_canons),
        ("bpblrai2\nV_refined\n(backbone FROZEN, SSIM refiner trained on top)", ssim_refs),
        ("NEW joint ckpt\nV_refined\n(backbone + refiner trained jointly, L1)", new_refs),
    ]
    for r, (rlbl, vols) in enumerate(sources):
        for i, (lbl, V) in enumerate(zip(labels, vols)):
            mid = V.shape[0] // 2
            axes[r, i].imshow(window_pct(V[mid], V), cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
            if r == 0:
                axes[r, i].set_title(lbl, fontsize=8)
            if i == 0:
                axes[r, i].text(-0.22, 0.5, rlbl, transform=axes[r, i].transAxes,
                                fontsize=8, ha="right", va="center")
    fig.suptitle(f"{title}  (native 256x256, pct99.5 per volume)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def run_ckpt_across_modes(model, want, val_ds, rcfg, gott_ds):
    """Returns dict of per-mode lists of volumes."""
    out = {"val_ON": [], "val_OFF": [], "OCMR": [], "Goett": []}
    for i in VAL_SEQS:
        out["val_ON"].append(_forward(model, val_batch(val_ds, rcfg, i, True), want=want))
    for i in VAL_SEQS:
        out["val_OFF"].append(_forward(model, val_batch(val_ds, rcfg, i, False), want=want))
    for sub in OCMR_SUBJECTS:
        sd = os.path.join(_ROOT, "scratch/data/ocmr/recon", sub)
        if os.path.isdir(sd):
            out["OCMR"].append(_forward(model, ocmr_batch(sd), want=want))
    for sub in GOTT_SUBJECTS:
        out["Goett"].append(_forward(model, goettingen_batch(gott_ds, sub), want=want))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    gott_ds = build_gott_dataset()

    # OLD ckpt -> V_canon (the _html/13 backbone)
    print("\n[1/3] OLD ckpt (backbone-only) -> V_canon for all 4 modes ...")
    m = load_state(build_model(use_refiner=False), OLD_CKPT)
    old_outs = run_ckpt_across_modes(m, want="canon", val_ds=val_ds, rcfg=rcfg, gott_ds=gott_ds)
    del m
    torch.cuda.empty_cache()

    # bpblrai2 ckpt -> V_refined (frozen backbone + SSIM refiner)
    print("\n[2/3] bpblrai2 (frozen-bb + SSIM refiner) -> V_refined for all 4 modes ...")
    m = load_state(build_model(use_refiner=True), SSIM_CKPT)
    ssim_outs = run_ckpt_across_modes(m, want="refined", val_ds=val_ds, rcfg=rcfg, gott_ds=gott_ds)
    del m
    torch.cuda.empty_cache()

    # NEW joint ckpt -> V_refined
    print("\n[3/3] NEW joint ckpt -> V_refined for all 4 modes ...")
    m = load_state(build_model(use_refiner=True), NEW_CKPT)
    new_outs = run_ckpt_across_modes(m, want="refined", val_ds=val_ds, rcfg=rcfg, gott_ds=gott_ds)
    del m
    torch.cuda.empty_cache()

    # Render 4 PNGs
    val_labels = [f"seq{i}" for i in VAL_SEQS]
    render_3row(old_outs["val_ON"], ssim_outs["val_ON"], new_outs["val_ON"], val_labels,
                "val ON (in-dist, breathing sim)",
                os.path.join(OUT, "three_ckpt_val_ON.png"))
    render_3row(old_outs["val_OFF"], ssim_outs["val_OFF"], new_outs["val_OFF"], val_labels,
                "val OFF (in-dist, no breathing)",
                os.path.join(OUT, "three_ckpt_val_OFF.png"))
    render_3row(old_outs["OCMR"], ssim_outs["OCMR"], new_outs["OCMR"], OCMR_SUBJECTS,
                "OCMR (OOD real-time free-breathing)",
                os.path.join(OUT, "three_ckpt_OCMR.png"))
    render_3row(old_outs["Goett"], ssim_outs["Goett"], new_outs["Goett"], GOTT_SUBJECTS,
                "Göttingen (OOD radial RT free-breathing)",
                os.path.join(OUT, "three_ckpt_Goettingen.png"))


if __name__ == "__main__":
    main()
