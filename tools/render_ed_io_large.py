"""Large INPUT|OUTPUT z-slice viz for the ED artifact check (style of
result/4way_refiner/refined_io_slices). Reads the per-head V_canon already cached by
render_ed_input_vs_pred_artifact.py (subj{ii}_volumes.npz) — NO 941M model reload — and only
rebuilds the (identical-across-heads) ED input volume from the dataset + val breathing.

Per (subject, head): one PNG = INPUT slices placed into the canonical cube by z (left, 3x4)
| that head's predicted V_canon (right, 3x4), red separator, large panels. Respiration ON.

Run: micromamba run -n svr python tools/render_ed_io_large.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from hydra import compose, initialize
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
OmegaConf.register_new_resolver(
    "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)

from hydra.utils import instantiate
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig

DEV = torch.device("cuda")
ART = os.path.join(_ROOT, "result", "ed_input_vs_pred_artifact")
OUT = os.path.join(ART, "io_large")
SUBJECTS = [0, 7]
HEADS = ["reference", "diffusion", "bspline"]
T_TARGET = 0
D = 12


def build_input_volume(mri_ds, subj_idx, resp_cfg, reference_slot):
    """Reproduce the ED (resp ON) input slices the models saw, placed into a (12,256,256) cube."""
    num_slices = mri_ds.num_slices
    data = mri_ds.get_data(seq_index=subj_idx, img_per_seq=num_slices)

    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(DEV)

    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    S = imgs.shape[1]
    batch = {
        "images": imgs,
        "scanner_coords": st("scanner_coords"),
        "z_indices": st("z_indices"),
        "t_indices": st("t_indices"),
        "phases": torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(DEV).unsqueeze(0),
        "timesteps": st("timesteps", np.int64),
        "slice_indices": st("slice_indices", np.int64),
        "seq_index": torch.tensor([[subj_idx]], dtype=torch.int64, device=DEV),
    }
    if reference_slot:
        batch["timesteps"][:, 0] = T_TARGET
    batch = gpu_augment_batch(batch, None, DEV, respiratory_cfg=resp_cfg, train=False)

    z = batch["slice_indices"][0].cpu().numpy()
    V = np.zeros((D, 256, 256), np.float32)
    for s in range(S):
        zi = int(z[s])
        if not (0 <= zi <= D - 1):
            continue
        sl = batch["images"][0, s, 0].float().cpu()
        sl256 = F.interpolate(sl[None, None], size=(256, 256),
                              mode="bilinear", align_corners=True)[0, 0].numpy()
        V[zi] = np.maximum(V[zi], sl256)  # if two slots share a z, keep the brighter
    return V


def window_pct(V):
    nz = V[V > 0]
    ref = nz if nz.size else V
    hi = float(np.percentile(ref, 99.5))
    lo = float(np.percentile(ref, 1.0))
    return np.clip((V - lo) / (hi - lo + 1e-9), 0, 1)


def render_io(Vin, Vout, title, path):
    Vin_w, Vout_w = window_pct(Vin), window_pct(Vout)
    fig = plt.figure(figsize=(8 * 2.6 + 0.5, 3 * 2.6))
    gs = gridspec.GridSpec(3, 9, figure=fig,
                           width_ratios=[1, 1, 1, 1, 0.12, 1, 1, 1, 1],
                           wspace=0.05, hspace=0.12)
    for Vw, c0, tag in [(Vin_w, 0, "in"), (Vout_w, 5, "out")]:
        for k in range(D):
            r, c = k // 4, c0 + (k % 4)
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(Vw[k], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{tag} z={k}", fontsize=8)
            ax.axis("off")
    sep = fig.add_subplot(gs[:, 4])
    sep.set_xlim(0, 1); sep.set_ylim(0, 1)
    sep.axvline(0.5, color="red", lw=3); sep.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _ROOT)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    with initialize(version_base=None, config_path=os.path.join("..", "training", "config")):
        cfg = compose(config_name="default")
    reference_slot = bool(cfg.get("reference_slot", False))
    resp_cfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.get("respiratory", None))
    val = cfg.data.val
    mri_ds = instantiate(val.dataset.dataset_configs[0],
                         common_conf=val.common_config, _recursive_=False)

    for subj in SUBJECTS:
        npz_path = os.path.join(ART, f"subj{subj:02d}_volumes.npz")
        if not os.path.exists(npz_path):
            print(f"  skip subj{subj}: no {npz_path}"); continue
        vols = np.load(npz_path)
        Vin = build_input_volume(mri_ds, subj, resp_cfg, reference_slot)
        for head in HEADS:
            if head not in vols:
                continue
            render_io(Vin, vols[head],
                      f"val subj {subj} — ED (t=0), respiration ON  —  "
                      f"INPUT slices (left)  |  {head} pred V_canon (right)",
                      os.path.join(OUT, f"subj{subj:02d}_{head}_io.png"))
        # GT reference panel too
        render_io(Vin, vols["gt"],
                  f"val subj {subj} — ED (t=0), respiration ON  —  "
                  f"INPUT slices (left)  |  GT (right)",
                  os.path.join(OUT, f"subj{subj:02d}_GT_io.png"))
    print(f"\ndone -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
