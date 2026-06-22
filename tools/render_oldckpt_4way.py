"""Render OLD pre-refiner ckpt (218747856) on all 4 modes for direct comparison vs the
joint-refiner ckpt outputs. Same render style as the NEW ckpt 4-way (percentile-99.5,
native 256x256, larger panels).

Outputs:
  result/4way_refiner/outputs_oldckpt_val_ON.png
  result/4way_refiner/outputs_oldckpt_val_OFF.png
  result/4way_refiner/outputs_oldckpt_OCMR.png
  result/4way_refiner/outputs_oldckpt_Goettingen.png
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
from tools.eval_ocmr_inference import (
    load_cine, percentile_scale, assign_canonical_z, build_batch,
)
from tools.diagnose_ood_clean_paradox import build_val_dataset

OLD_CKPT = "/home/minsukc/vggt/scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
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


def load_old_model():
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
        enable_refiner=False,
    ).to(DEV).eval()
    ck = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  OLD ckpt loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    return model


def _forward_canon(model, batch, target_t=-1.0):
    """OLD model has no refiner -> return V_canon via splat."""
    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), target_t, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    V = preds.get("V_canon")
    if V is None:
        wp = preds["world_points"].float()
        V, _cov = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    return V[0].float().cpu().numpy()


def val_forward(model, ds, rcfg, seq_index, breathing):
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
    batch = gpu_augment_batch(batch, None, DEV,
                              respiratory_cfg=(rcfg if breathing else None), train=False)
    return _forward_canon(model, batch)


def ocmr_forward(model, subj_dir):
    cine, meta = load_cine(subj_dir)
    scale = percentile_scale(cine)
    z_map = assign_canonical_z(meta["slice_positions_mm"])
    rng = np.random.default_rng(0)
    batch, _S, _picks = build_batch(cine, meta, scale, z_map, rng, DEV)
    return _forward_canon(model, batch)


def goettingen_forward(model, ds, subj):
    idx = next(i for i, s in enumerate(ds.subjects) if subj in s)
    data = ds.get_data(seq_index=idx, img_per_seq=12)
    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(DEV)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    batch = {"images": imgs, "scanner_coords": st("scanner_coords"),
             "z_indices": st("z_indices"), "t_indices": st("t_indices"),
             "slice_indices": torch.from_numpy(
                 np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0).to(DEV)}
    return _forward_canon(model, batch)


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


def render_mode(volumes, labels, title, path):
    """1 x N mid-z, native 256x256, pct99.5 per panel."""
    N = len(volumes)
    fig, axes = plt.subplots(1, N, figsize=(N * 3.4, 3.8))
    if N == 1:
        axes = [axes]
    for ax, V, lbl in zip(axes, volumes, labels):
        mid = V.shape[0] // 2
        ax.imshow(window_pct(V[mid], V), cmap="gray", vmin=0, vmax=1)
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title}  —  OLD ckpt V_canon mid-z (ED)  (native 256x256, pct99.5)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_old_model()
    val_ds, rcfg = build_val_dataset()
    gott_ds = build_gott_dataset()

    print("\n=== val ON (OLD ckpt) ===")
    vols, labs = [], []
    for i in VAL_SEQS:
        print(f"  seq {i}", flush=True)
        vols.append(val_forward(model, val_ds, rcfg, i, breathing=True))
        labs.append(f"seq {i}")
    render_mode(vols, labs, "val ON (in-dist, breathing sim)",
                os.path.join(OUT, "outputs_oldckpt_val_ON.png"))

    print("\n=== val OFF (OLD ckpt) ===")
    vols, labs = [], []
    for i in VAL_SEQS:
        print(f"  seq {i}", flush=True)
        vols.append(val_forward(model, val_ds, rcfg, i, breathing=False))
        labs.append(f"seq {i}")
    render_mode(vols, labs, "val OFF (in-dist, no breathing)",
                os.path.join(OUT, "outputs_oldckpt_val_OFF.png"))

    print("\n=== OCMR (OLD ckpt) ===")
    vols, labs = [], []
    for sub in OCMR_SUBJECTS:
        sd = os.path.join(_ROOT, "scratch/data/ocmr/recon", sub)
        if not os.path.isdir(sd):
            print(f"  skip {sub}"); continue
        print(f"  {sub}", flush=True)
        vols.append(ocmr_forward(model, sd))
        labs.append(sub)
    render_mode(vols, labs, "OCMR (OOD real-time free-breathing)",
                os.path.join(OUT, "outputs_oldckpt_OCMR.png"))

    print("\n=== Göttingen (OLD ckpt) ===")
    vols, labs = [], []
    for sub in GOTT_SUBJECTS:
        print(f"  {sub}", flush=True)
        vols.append(goettingen_forward(model, gott_ds, sub))
        labs.append(sub)
    render_mode(vols, labs, "Göttingen (OOD radial RT free-breathing)",
                os.path.join(OUT, "outputs_oldckpt_Goettingen.png"))


if __name__ == "__main__":
    main()
