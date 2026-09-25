"""Candidate subjects for the paper's AF + HRV LV volume-time-curve panels (render_vtc_2x2.py).

One subject is shown under both rhythms, so a candidate must be representative in BOTH arms:
Ours' VTC err. within p30-p70 of the cohort in af12 AND hrv12, and the af12 window must show an AF
signature (a short-R-R beat: contraction starting before full filling; vtc_af12_candidates.features).
Each candidate is drawn as an (AF | HRV) panel pair with every method, in the paper figure's style.

    python tools/vtc_pair_candidates.py --out temp/vtc_pair_candidates
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_paper_figures"))
from render_vtc_2x2 import curves  # noqa: E402
from render_vtc_figure import METHODS, smooth  # noqa: E402
from render_runtime_accuracy import INK  # noqa: E402
from vtc_af12_candidates import SRCS, METHOD, features  # noqa: E402


def ours(arm):
    out = {}
    for s in SRCS:
        for e in json.load(open(f"evaluation/metric_results/test/{s}_{arm}/ef/{METHOD}.json"))["per_subject"]:
            out[(s, e["subject"])] = e
    return out


def draw(ax, gt, cv, title):
    for m, label, c, dense in METHODS:
        o = label == "Ours"
        smooth(ax, cv[m], color=c, lw=1.8 if o else 0.9, ls="--" if dense else "-", alpha=1.0 if o else 0.5)
    smooth(ax, gt, color=INK, lw=1.8)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="temp/vtc_pair_candidates")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--band", type=float, default=30, help="keep Ours' VTC err. in [p_band, p_(100-band)]")
    ap.add_argument("--no-spike-filter", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    af, hrv = ours("af12"), ours("hrv12")
    a_nm = np.array([e["lv_curve_nmae_breath"] for e in af.values()])
    h_nm = np.array([e["lv_curve_nmae_breath"] for e in hrv.values()])
    alo, amed, ahi = np.percentile(a_nm, [a.band, 50, 100 - a.band])
    hlo, hmed, hhi = np.percentile(h_nm, [a.band, 50, 100 - a.band])
    print(f"af12 band {alo:.2f}-{ahi:.2f} (med {amed:.2f})   hrv12 band {hlo:.2f}-{hhi:.2f} (med {hmed:.2f})")

    rows = []
    for (s, subj), e in af.items():
        man = json.load(open(f"scratch/eval/{s}_af12/out/{subj}/manifest.json"))["rhythm"]
        f = features(man["ref_pos"], e["lv_curve_gt"])
        ea, eh = e["lv_curve_nmae_breath"], hrv[(s, subj)]["lv_curve_nmae_breath"]
        if (f["incomplete"] and (a.no_spike_filter or f["spike"] <= 0.2) and 40 <= e["ef_gt"] <= 70
                and alo <= ea <= ahi and hlo <= eh <= hhi):
            rows.append(dict(src=s, subject=subj, af=ea, hrv=eh, cycle=f["cycle"],
                             score=abs(ea - amed) / amed + abs(eh - hmed) / hmed))
    rows.sort(key=lambda r: r["score"])
    print(f"{len(rows)} candidates (short-R-R beat in the AF window, p{a.band:g}-p{100 - a.band:g} in both arms)")
    for i, r in enumerate(rows):
        print(f"{i + 1:2d} {r['src']:9s} {r['subject']:22s} AF {r['af']:5.2f}  HRV {r['hrv']:5.2f}  "
              f"full-cycle {r['cycle']}")
    json.dump(rows, open(os.path.join(a.out, "candidates.json"), "w"), indent=1)

    rows = rows[:a.n]
    per = 12                                              # candidates per page: 3 rows x 4 (AF|HRV) pairs
    for p in range(0, len(rows), per):
        page = rows[p:p + per]
        fig, axs = plt.subplots(3, 8, figsize=(22, 9), sharex=True)
        for ax in axs.flat[2 * len(page):]:
            ax.axis("off")
        for i, r in enumerate(page):
            k = p + i + 1
            for j, arm in enumerate(("af12", "hrv12")):
                gt, cv = curves(f"{r['src']}_{arm}", r["subject"], arm)
                ax = axs.flat[2 * i + j]
                draw(ax, gt, cv, f"#{k} {r['subject']}\n{'AF' if j == 0 else 'HRV'}  "
                                 f"Ours VTC err {r['af'] if j == 0 else r['hrv']:.1f}%")
        fig.suptitle(f"AF | HRV candidate pairs (black = GT, orange = Ours, dashed = dense-temporal). "
                     f"Cohort medians: AF {amed:.2f}%, HRV {hmed:.2f}%", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"pairs_p{p // per + 1}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
