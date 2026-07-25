"""The OOD relocation figure: on real MIITT the model displaces the stack with NO breathing applied.

The clean arm of resp_diag applies exactly zero shift, so any predicted Dz is unrequested. In-distribution
that number is 0.14 mm. On real gated MIITT it is 4-8 mm and strongly structured in z -- a whole-field
relocation, not breathing. This is what makes MIITT's breathing slope/EPE uninterpretable as a breathing
result, and it is free to measure from artifacts every eval already writes.
"""
import glob
import json
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/minsukc/vggt/result/1frame_series"
FIG = f"{RES}/figs"
C = {"1f_gather05": "#111111", "1f_no_gather": "#d1495b", "1f_contz": "#7b2cbf",
     "1f_dino_ft": "#2a9d8f", "1f_aug_moderate": "#e07a1f", "1f_lowdiff100": "#457b9d",
     "gather05 (docs/42)": "#888888"}


def collect(ds, meth):
    CL, Z = [], []
    for f in sorted(glob.glob(f"/home/minsukc/vggt/scratch/eval/{ds}/out/*/{meth}/resp_diag.json")):
        d = json.load(open(f))
        cp = d["clean"]["pred_dz_mm"]
        try:
            z = np.load(f.replace("resp_diag.json", "ed_dvf.npz"))["slot_z"]
        except Exception:
            continue
        CL += list(cp); Z += list(z[:len(cp)])
    return np.array(Z), np.array(CL)


def main():
    os.makedirs(FIG, exist_ok=True)
    series = []
    for meth, lab in [("vggt_20260715_1f_gather05", "1f_gather05"),
                      ("vggt_20260715_1f_aug_moderate", "1f_aug_moderate"),
                      ("vggt_20260715_1f_no_gather", "1f_no_gather"),
                      ("vggt_20260715_1f_dino_ft", "1f_dino_ft"),
                      ("vggt_20260715_1f_contz_contz", "1f_contz"),
                      ("vggt_20260715_1f_lowdiff100", "1f_lowdiff100"),
                      ("vggt_20260713_gather05", "gather05 (docs/42)")]:
        z, cl = collect("miitt", meth)
        if len(cl) > 20:
            series.append((lab, z, cl))
    if not series:
        print("  (no MIITT clean data yet)"); return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for lab, z, cl in series:
        zz = sorted(set(z))
        mu = [cl[z == v].mean() for v in zz]
        ax.plot(zz, mu, "-o", color=C.get(lab, "#666"), lw=2, ms=4,
                ls="--" if "docs/42" in lab else "-", label=lab)
    ax.axhline(0, ls=":", color="#333", lw=1.2)
    ax.set_xlabel("canonical z-plane"); ax.set_ylabel("mean predicted Δz (mm)")
    ax.set_title("(a) On REAL OOD data (MIITT), with ZERO breathing applied\n"
                 "the model still displaces the stack — and it is structured in z", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    ax.text(0.02, 0.03, "applied shift is exactly 0 everywhere on this arm:\nany deviation from the dotted "
                        "line is unrequested motion", transform=ax.transAxes, fontsize=7.5, color="#555",
            va="bottom")

    ax = axes[1]
    labs, vals, cols = [], [], []
    for lab, z, cl in series:
        labs.append(lab); vals.append(np.abs(cl).mean()); cols.append(C.get(lab, "#666"))
    zi, cli = collect("cmrxrecon", "vggt_20260715_1f_gather05")
    if len(cli):
        labs.append("1f_gather05\n(IN-DIST, for scale)"); vals.append(np.abs(cli).mean()); cols.append("#059669")
    y = np.arange(len(labs))
    ax.barh(y, vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("mean |predicted Δz| on un-breathed input (mm)   ← lower is better, 0 is correct")
    ax.set_title("(b) The clean negative control:\nin-distribution ~0.1 mm, real OOD 4–8 mm", fontsize=10.5)
    ax.grid(alpha=0.25, axis="x")

    fig.suptitle("The most diagnostic OOD number we have — and it is free from every eval run",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/ood_relocation.png", dpi=125)
    plt.close(fig)
    print(f"  -> {FIG}/ood_relocation.png")
    for lab, z, cl in series:
        r, p = stats.pearsonr(z, cl)
        print(f"    {lab:22s} mean signed {cl.mean():+6.2f} mm  |mean| {np.abs(cl).mean():5.2f}  "
              f"vs z r={r:+.3f} p={p:.1e}  n={len(cl)}")


if __name__ == "__main__":
    main()
