"""ED (t=0) reconstruction on MANY val subjects, respiration ON, for the 3 reference-conditioned
models. One figure per subject: rows = [input slices, reference, diffusion, bspline, GT], each a
12 z-plane strip — so all 3 heads can be judged side by side against GT at a glance.

Loads all 3 models once (kept resident), builds each val batch once (ED, deterministic val
breathing), forwards through all heads, renders. Reuses the val adapter + helpers from
five_row_compare / render_reference_5dataset_io.

Run: micromamba run -n svr python tools/render_ed_val_many.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from tools.five_row_compare import DEV, GRID_SHAPE, val_batch, build_val_dataset
from tools.render_reference_5dataset_io import input_volume, forward, MODELS
from vggt.models.vggt import VGGT

D = GRID_SHAPE[0]
OUT = os.path.join(_ROOT, "result", "ed_val_many_io")
N_SUBJECTS = 30  # all val subjects


def window_pct(V, ref=None):
    """Window V to [0,1] using 1/99.5 percentiles of ref (or V itself) over nonzero voxels."""
    src = ref if ref is not None else V
    nz = src[src > 0]
    base = nz if nz.size else src
    hi = float(np.percentile(base, 99.5)); lo = float(np.percentile(base, 1.0))
    return np.clip((V - lo) / (hi - lo + 1e-9), 0, 1)


def render_rows(rows, title, path):
    """rows = list of (label, volume, window_ref). One 12-z strip per row."""
    nr = len(rows)
    fig = plt.figure(figsize=(D * 1.7, nr * 1.7 + 0.4))
    gs = gridspec.GridSpec(nr, D, figure=fig, wspace=0.04, hspace=0.10)
    for r, (label, vol, wref) in enumerate(rows):
        Vw = window_pct(vol, wref)
        for k in range(D):
            ax = fig.add_subplot(gs[r, k])
            ax.imshow(Vw[k], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={k}", fontsize=8)
            if k == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _ROOT)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    val_ds.t_target_fixed = 0  # ED

    # Load all 3 models, keep resident.
    models = {}
    for name, ckpt, head in MODELS:
        m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=False,
                 use_target_t_pose_embedding=False, use_reference_token=True,
                 train_on_residual_dvf=True, warp_head_type=head).to(DEV).eval()
        ck = torch.load(os.path.join(_ROOT, ckpt), map_location="cpu", weights_only=False)
        miss, unexp = m.load_state_dict(ck["model"], strict=False)
        assert not miss and not unexp, f"{name}: missing={miss[:5]} unexpected={unexp[:5]}"
        print(f"loaded {name} ({head})", flush=True)
        models[name] = m

    for seq in range(N_SUBJECTS):
        try:
            batch = val_batch(val_ds, rcfg, seq, breathing=True)
        except Exception as e:
            print(f"  skip seq{seq}: {e}", flush=True); continue
        Vin = input_volume(batch)
        V_gt = batch["phases"][0, 0].float().cpu().numpy()  # ED phase, (D,H,W)
        rows = [("input", Vin, None)]
        for name in ("reference", "diffusion", "bspline"):
            V_can, _ = forward(models[name], batch)
            rows.append((name, V_can, V_gt))  # window pred rows by GT for fair intensity compare
        rows.append(("GT", V_gt, V_gt))
        render_rows(rows, f"val seq{seq} — ED (t=0), respiration ON  —  input / 3 heads / GT",
                    os.path.join(OUT, f"val_seq{seq:02d}_ED_resp_io.png"))
    print(f"\ndone -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
