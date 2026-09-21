#!/usr/bin/env python
"""The trainer's wandb DVF panel (trainer_viz._log_volume_and_dvf_to_wandb), for two eval arms at once.

Same layout and the same colour conventions as the wandb panel -- one COLUMN per input slot (all of
them, slot 0 = the reference first), RdBu_r, in-plane +/-15 mm, through-plane +/-25 mm, no masking --
but with the two arms' rows interleaved so they can be compared slot by slot:

    input | A dx | B dx | A dy | B dy | A dz | B dz

The fields come from each arm's ed_dvf.npz (the point head's residual at the ED pass). Both arms saw
the identical input slots, so any difference is the model. Column titles carry frame, z and the TRUE
applied breathing dz for that slot's own frame.

Usage:
  PYTHONPATH=training:. python tools/render_dvf_wandb_style.py --arm af24 \
      --subjects cmrx2024:CMRx24_Test_P017 mnms:MNMs_K7L2Y6
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as _gs                                  # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
import nibabel as nib                                              # noqa: E402
import numpy as np                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
from build_af_bundle import DT, T_BREATH, breathing_model                 # noqa: E402
from data.respiratory import lujan_displacement                          # noqa: E402

EVAL = os.path.join(ROOT, "scratch", "eval")
IN_PLANE_MM, THROUGH_MM = (256 - 1) / 2.0 * 1.4, 90.0             # trainer_viz constants
IN_PLANE_R, THROUGH_R = 15.0, 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--subjects", nargs="+", required=True, help="source:subject")
    ap.add_argument("--a", default="vggt_final518_base_ep300")
    ap.add_argument("--b", default="vggt_final518_diff1000_ep300")
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "figs", "rhythm24", f"{args.arm}_dvf_wandb_style")
    os.makedirs(outdir, exist_ok=True)
    sa, sb = (x.replace("vggt_final518_", "").replace("_ep300", "") for x in (args.a, args.b))

    for item in args.subjects:
        src, subj = item.split(":")
        sd = os.path.join(EVAL, f"{src}_{args.arm}", "out", subj)
        man = json.load(open(os.path.join(sd, "manifest.json")))
        A, B = (np.load(os.path.join(sd, m, "ed_dvf.npz")) for m in (args.a, args.b))
        assert np.array_equal(A["slot_z"], B["slot_z"]) and np.array_equal(A["slot_t"], B["slot_t"]), "slot draws differ"
        dA, dB = (np.asarray(X["delta"], np.float32) for X in (A, B))               # (S,518,518,3) normalized
        S = dA.shape[0]
        u, amp, r0, n, _ = breathing_model(man)
        imgs, titles = [], []
        for s in range(S):
            z, j = int(round(float(A["slot_z"][s]))), int(A["slot_t"][s])
            im = np.asarray(nib.load(os.path.join(sd, "breath", f"stack_t{j:02d}.nii.gz")).dataobj, np.float32)[:, :, z].T
            imgs.append(im)
            dz = float((u * amp * float(lujan_displacement(float((r0[z] + j * DT / T_BREATH) % 1.0), 1.0, n=n)))[0])
            titles.append(f"{'REF ' if s == 0 else ''}t={j}, z={z}\ntrue dz={dz:+.1f}mm")
        vmax = float(np.percentile(np.concatenate([i[i > 0] for i in imgs]), 99.5))

        rows = [("input", imgs, "gray", 0, vmax)]
        for ax_i, (nm, mm, rng) in enumerate((("Δx", IN_PLANE_MM, IN_PLANE_R), ("Δy", IN_PLANE_MM, IN_PLANE_R),
                                              ("Δz", THROUGH_MM, THROUGH_R))):
            rows.append((f"{sa}\n{nm} (mm)", dA[..., ax_i] * mm, "RdBu_r", -rng, rng))
            rows.append((f"{sb}\n{nm} (mm)", dB[..., ax_i] * mm, "RdBu_r", -rng, rng))

        fig = plt.figure(figsize=(1.6 * S + 1.8, 1.75 * len(rows) + 0.8), dpi=95)
        g = _gs.GridSpec(len(rows), S + 1, width_ratios=[1.0] * S + [0.05], wspace=0.04, hspace=0.12)
        for r, (lbl, data, cmap, vmin, vmx) in enumerate(rows):
            last = None
            for s in range(S):
                ax = fig.add_subplot(g[r, s])
                last = ax.imshow(data[s], cmap=cmap, vmin=vmin, vmax=vmx)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(titles[s], fontsize=7)
                if s == 0:
                    ax.set_ylabel(lbl, fontsize=8)
            plt.colorbar(last, cax=fig.add_subplot(g[r, S]))
        fig.suptitle(f"DVF — {src}_{args.arm} / {subj}   ED pass   {sa} vs {sb}   (wandb panel layout, all {S} slots)",
                     fontsize=9)
        out = os.path.join(outdir, f"{subj}__{sa}_vs_{sb}.png")
        fig.savefig(out, facecolor="white", bbox_inches="tight"); plt.close(fig)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
