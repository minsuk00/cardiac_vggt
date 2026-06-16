"""Qualitative panels for the 5-variant report (Protocol B = breathing val).

For chosen val seq_indices (default: an ED sample and an ES sample), runs identity-Δ + all
5 model checkpoints on the SAME deterministic breathing-corrupted inputs and renders:
  (1) mid-bbox-z comparison: columns [GT, identity, var1..var5], rows [image, signed diff]
  (2) per-z montage for the headline breathing comparison (var2 resp vs var4 no-resp)

Reuses the eval harness (build_dataset/build_batch/make_model/PROTOCOLS) so the pixels match
the metric numbers exactly. Outputs PNGs to result/variants_eval/panels/.

Run: micromamba run -n svr python tools/render_variants_panels.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_variants_matrix import (build_dataset, build_batch, make_model, eval_protocol,  # noqa: E402
                                  RUNS, LOGS, PROTOCOLS, GRID_SHAPE, NUM_SLICES)
sys.path.insert(0, os.path.join(REPO, "training"))
from loss import compute_volume_intensity_loss  # noqa: E402

OUT = os.path.join(REPO, "result", "variants_eval", "panels")
SEQS = [0, 7]   # seq 0 → subject0,t=0 (ED); seq 7 → subject7,t=7 (ES)
ERR = 0.10
LABELS = {0: "identity", 1: "var1 resp z+t", 2: "var2 resp", 3: "var3 resp+aug",
          4: "var4 no-resp", 5: "var5 no-resp+aug"}


def build_bases(ds, seqs, device):
    """Build breathing-protocol batches once per seq; return dict seq->(base, V_gt, t, bbox)."""
    from data.gpu_aug import gpu_augment_batch
    cfg = PROTOCOLS["breathing"]
    bases = {}
    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        base = build_batch(data, device, seq_index=seq)
        base = gpu_augment_batch(base, None, device, respiratory_cfg=cfg, train=False)
        t_target = int(np.asarray(data["t_target"]).flatten()[0])
        bbox = [int(v) for v in base["anatomy_bbox"][0].tolist()]
        out = compute_volume_intensity_loss({"world_points": base["scanner_coords"]}, base,
                                            grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float().cpu().numpy()
        ident_V = out["V_canon"][0].float().cpu().numpy()
        ident_p = float(out["metric_psnr_3d_bbox"])
        bases[seq] = dict(base=base, V_gt=V_gt, t=t_target, bbox=bbox,
                          results={0: (ident_V, ident_p)})
    return bases


def fill_models(bases, device):
    """Load each model ONCE, run on every seq's base, accumulate into bases[seq]['results']."""
    for run in RUNS:
        ckpt = os.path.join(LOGS, run["exp_dir"], "ckpts", "checkpoint_last.pt")
        model, _ = make_model(run["use_t"], ckpt, device)
        for seq, b in bases.items():
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = model(b["base"]["images"], batch=b["base"])
            out = compute_volume_intensity_loss(preds, b["base"], grid_shape=GRID_SHAPE, tv_weight=0.0)
            b["results"][run["var"]] = (out["V_canon"][0].float().cpu().numpy(),
                                        float(out["metric_psnr_3d_bbox"]))
        del model
        torch.cuda.empty_cache()
        print(f"  rendered var{run['var']}")


def render_midz(ds, seq, b):
    results, V_gt, t_target, bbox = b["results"], b["V_gt"], b["t"], b["bbox"]
    subj = os.path.basename(ds.subjects[seq % len(ds.subjects)])
    z0, z1 = bbox[0], bbox[1]
    zmid = (z0 + z1) // 2
    methods = [0, 1, 2, 3, 4, 5]
    ncol = 1 + len(methods)
    vmax = float(max(V_gt.max(), max(results[m][0].max() for m in methods), 1e-3))

    fig = plt.figure(figsize=(2.0 * ncol + 0.5, 4.4), dpi=140)
    gs = gridspec.GridSpec(2, ncol, wspace=0.06, hspace=0.18)
    # col0: GT
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(V_gt[zmid], cmap="gray", vmin=0, vmax=vmax)
    ax.set_title("V_gt", fontsize=10); ax.set_ylabel(f"image (z={zmid})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
    ax.text(0.5, 0.5, "(reference)", ha="center", va="center", fontsize=9, color="#888")
    for j, m in enumerate(methods):
        Vc, psnr = results[m]
        ax = fig.add_subplot(gs[0, j + 1]); ax.imshow(Vc[zmid], cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"{LABELS[m]}\n{psnr:.2f} dB", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j + 1])
        ax.imshow(Vc[zmid] - V_gt[zmid], cmap="RdBu_r", vmin=-ERR, vmax=ERR)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0: ax.set_ylabel(f"diff (±{ERR})", fontsize=9)
    fig.suptitle(f"VAL breathing — {subj}  t_target={t_target}  (mid-bbox z={zmid})  bbox-PSNR labeled",
                 fontsize=11, y=1.02)
    p = os.path.join(OUT, f"midz_seq{seq}_{subj}_t{t_target}.png")
    plt.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", p)
    return results, V_gt, t_target, bbox, subj


def render_perz(results, V_gt, t_target, bbox, subj, seq, focus=(0, 2, 4)):
    """Per-z montage comparing GT + a focused subset of methods across in-bbox z planes."""
    z0, z1 = bbox[0], bbox[1]
    zs = list(range(z0, z1))
    rows = [("V_gt", V_gt, None)] + [(LABELS[m], results[m][0], results[m][1]) for m in focus]
    vmax = float(max(V_gt.max(), max(results[m][0].max() for m in focus), 1e-3))
    nrow, ncol = len(rows), len(zs)
    fig = plt.figure(figsize=(1.5 * ncol + 1.0, 1.5 * nrow + 0.6), dpi=130)
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.04, hspace=0.06)
    for r, (label, vol, psnr) in enumerate(rows):
        for c, z in enumerate(zs):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(vol[z], cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(f"z={z}", fontsize=8)
            if c == 0:
                yl = label if psnr is None else f"{label}\n{psnr:.2f} dB"
                ax.set_ylabel(yl, fontsize=9)
    fig.suptitle(f"VAL breathing per-z — {subj}  t_target={t_target}", fontsize=11, y=1.01)
    p = os.path.join(OUT, f"perz_seq{seq}_{subj}_t{t_target}.png")
    plt.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("saved", p)


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset()
    bases = build_bases(ds, SEQS, device)
    fill_models(bases, device)
    for seq in SEQS:
        results, V_gt, t_target, bbox, subj = render_midz(ds, seq, bases[seq])
        # focus per-z: GT + identity + var2 (resp) + var4 (no-resp) → the breathing comparison
        render_perz(results, V_gt, t_target, bbox, subj, seq, focus=(0, 2, 4))


if __name__ == "__main__":
    main()
