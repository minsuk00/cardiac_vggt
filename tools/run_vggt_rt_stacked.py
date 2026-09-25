#!/usr/bin/env python
"""VGGT on a naive RT stack: frame f of EVERY slice (no correction) -> one 3D volume per (subject, f).

Same model path as evaluation/src/engine/run_vggt_rt.py (scaffold geometry, build_batch_rt, splat),
except every slot — reference and companions — takes frame f of its own plane, so the model input is
exactly the naive stack shown by tools/render_rt_stacked_frames.py. One forward pass per (subject, f).

Saves <out>/<subject>/frame_<f>/{input.nii.gz, ours.nii.gz} (canonical 1.4 mm grid, (X,Y,Z,1)) and
<out>/summary.png: one row per (subject, f), stacked input | ours, each as SAX + x-z + y-z cuts.

    PYTHONPATH=training:. python tools/run_vggt_rt_stacked.py --ckpt <ckpt> \
        --jobs MIITT_Patient_2024Jan04_Cardiomyopathy_AFib:0,30 MIITT_Volunteer1:60 --out <dir>
"""
import argparse
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_vggt_rt as rt                                  # noqa: E402
from render_rt_orthoviews import lv_centre                # noqa: E402

HALF = 45          # crop half-width, px (1.4 mm)


@torch.no_grad()
def reconstruct_stacked(model, ds, seq_index, bundle, f, device, splat_res):
    """(D, 256, 256) recon from frame f of every plane."""
    batch = rt.build_batch_rt(ds, seq_index, bundle, device)
    batch["timesteps"][:] = f                               # every slot: its own plane's frame f
    batch.pop("images", None)
    rt.gpu_augment_batch(batch, None, device, respiratory_cfg=None, train=False)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    D = bundle.shape[1]
    z_scale = float(batch["z_scale"].reshape(-1)[0])
    V, _ = rt._splat_preds_native({"world_points": preds["world_points"].float()}, batch,
                                  (D, 256, 256), z_scale, splat_res=splat_res)
    return V[0].float().cpu().numpy()


def cuts(v, x, y, z):
    """(D, H, W) -> SAX at z, x-z and y-z cuts through (x, y); base at top."""
    return [v[z, y - HALF:y + HALF, x - HALF:x + HALF], v[::-1, y, x - HALF:x + HALF],
            v[::-1, y - HALF:y + HALF, x]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jobs", nargs="+", required=True, help="SUBJECT:f1,f2,...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-summary", action="store_true", help="skip summary.png (many frames)")
    args = ap.parse_args()
    device = torch.device("cuda")
    model, cfg = rt.load_model_from_run(args.ckpt, device=device)
    splat_res = ((cfg.get("loss") or {}).get("volume") or {}).get("splat_res")

    rows = []                                               # (label, input cuts, ours cuts, vmax, zs)
    for job in args.jobs:
        subj, fs = job.split(":")
        sd = os.path.join(rt.RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax")
        rt_path = os.path.join(sd, "4d_recon.nii.gz")
        bundle = rt.build_rt_bundle(rt_path)                # (F, D, H, W)
        x, y, zc = lv_centre(os.path.join(sd, "heart_seg.nii.gz"))
        vmax = float(np.percentile(bundle[::10], 99.5))
        with tempfile.TemporaryDirectory() as td:
            ds = rt.rv.make_dataset(cfg, rt.make_rt_scaffold(rt_path, subj, td), "val", td)
            dz = float(np.asarray(ds.get_data(seq_index=0, img_per_seq=ds.num_slices)["dz_mm"]).reshape(-1)[0])
            for f in [int(s) for s in fs.split(",")]:
                inp = bundle[f]                             # (D, H, W): frame f of every plane
                ours = reconstruct_stacked(model, ds, rt.rv.name_seed("miitt", subj), bundle, f,
                                           device, splat_res)
                od = os.path.join(args.out, subj, f"frame_{f:03d}")
                os.makedirs(od, exist_ok=True)
                rt.save_cine_xyzt(os.path.join(od, "input.nii.gz"), inp[None], dz)
                rt.save_cine_xyzt(os.path.join(od, "ours.nii.gz"), ours[None], dz)
                print(f"{subj} frame {f}: -> {od}")
                rows.append((f"{subj.replace('MIITT_', '')}\nframe {f}", cuts(inp, x, y, zc),
                             cuts(ours, x, y, zc), vmax, dz / rt.rv.INPLANE_MM))

    if args.no_summary:
        return
    heads = ["stacked input\nSAX", "stacked input\nx-z cut", "stacked input\ny-z cut",
             "ours\nSAX", "ours\nx-z cut", "ours\ny-z cut"]
    fig, ax = plt.subplots(len(rows), 6, figsize=(6 * 2.0, len(rows) * 2.1), squeeze=False)
    for i, (lab, ci, co, vmax, zs) in enumerate(rows):
        for j, img in enumerate(ci + co):
            ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect=1 if j % 3 == 0 else zs,
                            interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(heads[j], fontsize=9)
        ax[i, 0].set_ylabel(lab, fontsize=8)
    plt.tight_layout()
    out = os.path.join(args.out, "summary.png")
    plt.savefig(out, dpi=100); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
