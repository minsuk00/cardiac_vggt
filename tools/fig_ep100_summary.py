"""ep100 4-cohort summary figures for docs/46 / _html/46. Offline, reads scored summaries + resp_diag.

Panels:
  fig_intensity.png   — breath PSNR, 4 cohorts x 6 models (appearance-wall + cohort difficulty)
  fig_verdicts.png    — C1-C5 paired Δ vs hub: in-dist vs pooled-OOD, with 95% CI
  fig_breathing.png   — breathing slope + clean-arm relocation per cohort (hub)
"""
import json, glob, os, re
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "/home/minsukc/vggt/scratch/eval"
OUT = "/home/minsukc/vggt/result/1frame_ep100"
os.makedirs(OUT, exist_ok=True)
COH = ["cmrxrecon", "miitt", "ocmr", "acdc"]
CLBL = {"cmrxrecon": "CMRx (in-dist)", "miitt": "MIITT", "ocmr": "OCMR", "acdc": "ACDC"}
MODELS = ["gather05", "no_gather", "aug_moderate", "contz", "dino_ft", "lowdiff100"]
MC = {"gather05": "#111", "no_gather": "#d1495b", "aug_moderate": "#e07a1f",
      "contz": "#7b2cbf", "dino_ft": "#2a9d8f", "lowdiff100": "#457b9d"}


def summ(c, v):
    for suf in ("", "_contz"):
        f = glob.glob(f"{E}/{c}/out/vggt_20260719_1f_{v}_ep99{suf}_summary.json")
        if f:
            return json.load(open(f[0]))
    return None


def persub(c, v, key="breath_psnr"):
    d = summ(c, v)
    return {r["subject"]: r[key] for r in d["per_subject"]} if d else {}


def fig_intensity():
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(COH)); w = 0.13
    for i, m in enumerate(MODELS):
        ys = [summ(c, m)["all"]["breath_psnr"][0] for c in COH]
        ax.bar(x + (i - 2.5) * w, ys, w, label=m, color=MC[m])
    ax.set_xticks(x); ax.set_xticklabels([CLBL[c] for c in COH])
    ax.set_ylabel("breath PSNR (dB)"); ax.set_ylim(14, 23)
    ax.set_title("Breath-arm PSNR — 4 cohorts × 6 models (ep100)\nnote the ≤1.0 dB within-cohort spread = the appearance wall")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_intensity.png", dpi=130); plt.close(fig)


def fig_verdicts():
    comps = [("C1", "no_gather"), ("C2", "aug_moderate"), ("C3", "contz"),
             ("C4", "dino_ft"), ("C5", "lowdiff100")]
    hub = {c: persub(c, "gather05") for c in COH}
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(comps))
    for j, (cid, v) in enumerate(comps):
        # in-dist
        subs = [s for s in hub["cmrxrecon"] if s in persub("cmrxrecon", v)]
        din = np.array([persub("cmrxrecon", v)[s] - hub["cmrxrecon"][s] for s in subs])
        # pooled OOD
        od = []
        for c in ["miitt", "ocmr", "acdc"]:
            pv = persub(c, v)
            od += [pv[s] - hub[c][s] for s in hub[c] if s in pv]
        od = np.array(od)
        for dd, off, col, lab in [(din, +0.16, "#888", "in-dist"), (od, -0.16, "#c00", "OOD-pool")]:
            m, sem = dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd))
            ax.errorbar(m, y[j] + off, xerr=1.96 * sem, fmt="o", color=col, capsize=3,
                        label=(lab if j == 0 else None))
            p = stats.ttest_1samp(dd, 0).pvalue
            ax.text(m, y[j] + off + 0.06, ("*" if p < 0.05 else ""), ha="center", color=col, fontsize=13)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{c}: {v}−hub" for c, v in comps])
    ax.set_xlabel("Δ breath PSNR vs gather05 hub (dB)   [>0 favors variant; * = p<0.05]")
    ax.set_title("C1–C5 verdicts at ep100 — in-dist (n=30) vs pooled OOD (n=61)")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_verdicts.png", dpi=130); plt.close(fig)


def fig_breathing():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sl, rc = [], []
    for c in COH:
        fs = glob.glob(f"{E}/{c}/out/*/vggt_20260719_1f_gather05_ep99/resp_diag.json")
        s = [json.load(open(f))["breath"]["slope"] for f in fs if json.load(open(f))["breath"].get("slope") is not None]
        r = [json.load(open(f))["clean"]["epe_dz_mm"] for f in fs]
        sl.append(np.mean(s)); rc.append(np.mean(r))
    ax[0].bar([CLBL[c] for c in COH], sl, color="#2a9d8f"); ax[0].axhline(1, ls=":", color="#aaa")
    ax[0].set_ylabel("breathing slope →1"); ax[0].set_title("Breathing amplitude fidelity (hub)")
    ax[1].bar([CLBL[c] for c in COH], rc, color="#d1495b")
    ax[1].set_ylabel("clean-arm relocation (mm)"); ax[1].set_title("OOD relocation (Δz on un-breathed input)")
    for a in ax: a.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_breathing.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    fig_intensity(); fig_verdicts(); fig_breathing()
    print("wrote", OUT)
