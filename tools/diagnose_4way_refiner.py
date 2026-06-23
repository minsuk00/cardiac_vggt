"""Multi-example 4-way inference comparison for the joint-refiner model (218349151) at ED.

For each of {val ON, val OFF, OCMR, Göttingen} runs N examples and writes ONE PNG per mode:
  result/4way_refiner/outputs_<mode>.png   1 x N grid of V_refined mid-z at ED

Run directly on a GPU node:
  PYTHONPATH=training:. micromamba run -n svr python tools/diagnose_4way_refiner.py
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


from vggt.models.vggt import VGGT
from tools.eval_ocmr_inference import (
    load_cine, percentile_scale, assign_canonical_z, build_batch,
)
from tools.diagnose_ood_clean_paradox import build_val_dataset
from eval.adapters.goettingen import GoettingenAdapter

CKPT = "/home/minsukc/vggt/scratch/logs/218349151_mri_refiner_joint/ckpts/checkpoint_last.pt"
OUT = os.path.join(_ROOT, "result", "4way_refiner")
DEV = torch.device("cuda")

OCMR_SUBJECTS = ["us_0084_1_5T", "us_0173_pt_1_5T", "us_0183_pt_1_5T",
                 "us_0197_pt_1_5T", "us_0169_pt_1_5T"]
GOTT_RECON = "/home/minsukc/vggt/scratch/data/goettingen/recon"  # native RT recons (direct adapter)
GOTT_SUBJECTS = ["vol0001_vis1", "vol0002_vis1", "vol0003_vis1",
                 "vol0009_vis1", "vol0023_vis1"]
VAL_SEQS = [0, 1, 2, 3, 4]


def load_refiner_model():
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
        enable_refiner=True, refiner_use_coverage=True, grid_shape=(12, 256, 256),
    )
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing
           if any(s in k for s in ("aggregator", "point_head", "refiner",
                                    "z_embedder", "target_t_embedder"))]
    if bad:
        raise RuntimeError(f"missing critical weights: {bad[:5]}")
    print(f"  refiner_joint loaded (missing={len(missing)}, unexpected={len(unexpected)})",
          flush=True)
    return model.to(DEV).eval()


def _run(model, batch, target_t=-1.0):
    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), target_t, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    V = preds.get("V_refined", preds.get("V_canon"))[0].float().cpu().numpy()
    return V


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
    return _run(model, batch, target_t=-1.0)


def ocmr_forward(model, subj_dir):
    cine, meta = load_cine(subj_dir)
    scale = percentile_scale(cine)
    z_map = assign_canonical_z(meta["slice_positions_mm"])
    rng = np.random.default_rng(0)
    batch, _S, _picks = build_batch(cine, meta, scale, z_map, rng, DEV)
    return _run(model, batch, target_t=-1.0)


def goettingen_forward(model, ds, subj):
    # Direct RTFB adapter on the native recon (real slices, no 6->8mm interp); `ds` unused.
    nii = os.path.join(GOTT_RECON, subj, subj + ".nii.gz")
    batch = GoettingenAdapter(nii).build_batch(np.random.default_rng(0), DEV)[0]
    return _run(model, batch, target_t=-1.0)


def build_gott_dataset():
    return None  # Göttingen now uses the direct adapter; no MRIDataset needed


def render_mode_png(volumes, labels, title, path):
    """1 x N grid of V_refined mid-z; per-panel percentile-99.5 windowing (matches
    goettingen_infer.py / eval_ocmr_inference.py) + bilinear 2x upscale via PIL."""
    from PIL import Image
    N = len(volumes)
    fig, axes = plt.subplots(1, N, figsize=(N * 3.2, 3.6))
    if N == 1:
        axes = [axes]
    for ax, V, lbl in zip(axes, volumes, labels):
        mid = V.shape[0] // 2
        sl = V[mid]
        hi = float(np.percentile(V, 99.5))
        lo = float(np.percentile(V, 1.0))
        a = np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1)
        a8 = (a * 255).astype(np.uint8)
        up = np.asarray(Image.fromarray(a8, "L").resize((512, 512), Image.BILINEAR))
        ax.imshow(up, cmap="gray", vmin=0, vmax=255)
        ax.set_title(lbl, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{title}  —  V_refined mid-z at ED (target_t=-1.0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_refiner_model()

    val_ds, rcfg = build_val_dataset()
    gott_ds = build_gott_dataset()

    # val ON
    print("\n=== val ON (N=5) ===")
    vols, labs = [], []
    for i in VAL_SEQS:
        print(f"  seq {i}", flush=True)
        vols.append(val_forward(model, val_ds, rcfg, i, breathing=True))
        labs.append(f"seq {i}")
    render_mode_png(vols, labs, "val ON (in-dist, breathing sim)",
                    os.path.join(OUT, "outputs_val_ON.png"))

    # val OFF
    print("\n=== val OFF (N=5) ===")
    vols, labs = [], []
    for i in VAL_SEQS:
        print(f"  seq {i}", flush=True)
        vols.append(val_forward(model, val_ds, rcfg, i, breathing=False))
        labs.append(f"seq {i}")
    render_mode_png(vols, labs, "val OFF (in-dist, no breathing)",
                    os.path.join(OUT, "outputs_val_OFF.png"))

    # OCMR
    print(f"\n=== OCMR (N={len(OCMR_SUBJECTS)}) ===")
    vols, labs = [], []
    for sub in OCMR_SUBJECTS:
        sd = os.path.join(_ROOT, "scratch/data/ocmr/recon", sub)
        if not os.path.isdir(sd):
            print(f"  skip {sub} (missing dir)"); continue
        print(f"  {sub}", flush=True)
        vols.append(ocmr_forward(model, sd))
        labs.append(sub)
    render_mode_png(vols, labs, "OCMR (OOD real-time free-breathing)",
                    os.path.join(OUT, "outputs_OCMR.png"))

    # Göttingen (only 3 available)
    print(f"\n=== Göttingen (N={len(GOTT_SUBJECTS)}, all available) ===")
    vols, labs = [], []
    for sub in GOTT_SUBJECTS:
        print(f"  {sub}", flush=True)
        vols.append(goettingen_forward(model, gott_ds, sub))
        labs.append(sub)
    render_mode_png(vols, labs, "Göttingen (OOD radial RT free-breathing)",
                    os.path.join(OUT, "outputs_Goettingen.png"))


if __name__ == "__main__":
    main()
