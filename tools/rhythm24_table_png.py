#!/usr/bin/env python
"""Render a rhythm-arm results table (from tools/rhythm24_ef_table.py --json) as a PNG.

Usage:  python tools/rhythm24_table_png.py af24
        (reads temp/rhythm24_ef/<arm>_summary.json, writes figs/rhythm24/<arm>_results_table.png)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"fetal_cmr_4d": "Fetal CMR 4D (self-gated)", "vggt_final518_base_ep300": "VGGT base",
         "vggt_final518_diff1000_ep300": "VGGT diff1000", "vggt_final518_nogather_ep300": "VGGT nogather",
         "vggt_final518_hw0_ep300": "VGGT hw0"}
# (header, key, format, higher_is_better | None = no winner highlighted)
COLS = [("n", "n", "{:d}", None), ("PSNR\n(dB) ↑", "psnr", "{:.2f}", True),
        ("EF MAE\n(pp) ↓", "ef_mae", "{:.2f}", False), ("EF bias\n(pp)", "ef_bias", "{:+.2f}", None),
        ("EF r ↑", "r", "{:.2f}", True), ("EDV MAE\n(mL) ↓", "edv_mae", "{:.1f}", False),
        ("ESV MAE\n(mL) ↓", "esv_mae", "{:.1f}", False), ("Dice LV\nED / ES ↑", ("dice_lv_ed", "dice_lv_es"), None, True),
        ("Dice MYO\nED / ES ↑", ("dice_myo_ed", "dice_myo_es"), None, True),
        ("Motion EPE\n(mm) ↓", "motion_epe_mm", "{:.2f}", False)]


def main():
    arm = sys.argv[1]
    S = json.load(open(os.path.join(ROOT, "temp", "rhythm24_ef", f"{arm}_summary.json")))["methods"]
    order = [m for m in NAMES if m in S]
    cells, best = [], {}
    for ci, (_, key, fmt, hib) in enumerate(COLS):
        if hib is None:
            continue
        val = lambda m: (sum(S[m][k] for k in key) / 2 if isinstance(key, tuple) else S[m].get(key))   # noqa: E731
        cand = [(val(m), m) for m in order if val(m) is not None]
        best[ci] = (max if hib else min)(cand)[1]
    for m in order:
        row = [NAMES[m]]
        for _, key, fmt, _ in COLS:
            if isinstance(key, tuple):
                row.append(f"{S[m][key[0]]:.3f} / {S[m][key[1]]:.3f}")
            else:
                v = S[m].get(key)
                row.append("n/a" if v is None else fmt.format(v))
        cells.append(row)

    fig, ax = plt.subplots(figsize=(15.5, 0.62 * len(order) + 1.9), dpi=170)
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=["method"] + [c[0] for c in COLS], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10.5); tb.scale(1, 2.0)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        if r == 0:
            cell.set_facecolor("#2f3b52"); cell.set_text_props(color="white", weight="bold"); cell.set_height(cell.get_height() * 1.35)
        elif c == 0:
            cell.set_text_props(ha="left", weight="bold"); cell.set_facecolor("#f3f4f7")
        elif best.get(c - 1) == order[r - 1]:
            cell.set_facecolor("#d9f0dc"); cell.set_text_props(weight="bold")
    tb.auto_set_column_width(list(range(len(COLS) + 1)))
    ax.set_title(f"{arm}: 180 test subjects, 24 frames/slice  —  green = best per column", fontsize=12.5, weight="bold", pad=10)
    fig.text(0.5, 0.035, "EF bias = mean(pred − GT), signed; EF MAE = mean|pred − GT|.  EF r = corr(pred, GT).  "
             "Motion EPE = VGGT's predicted through-plane breathing shift vs the true applied shift "
             "(Fetal predicts none).  Fetal n<180: EF undefined where its LV vanishes in ≥1 frame.",
             ha="center", fontsize=8.6, color="#444444", wrap=True)
    out = os.path.join(ROOT, "figs", "rhythm24", f"{arm}_results_table.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
