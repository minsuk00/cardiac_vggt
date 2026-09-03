#!/usr/bin/env python
"""compare_burst_arms.py — 2x2 burst-sampling comparison from the git-tracked eval summaries.

Arms (train regime x eval regime), all `breath` variant, val split, pooled over the 7 sources:
    vggt_burst5noreg224_ep300      burst-k5 model @ 5 frames/slice   (its native regime)
    vggt_burst5noreg224_k1_ep300   burst-k5 model @ 1 frame/slice
    vggt_noreg224_ep300            k=1 model      @ 1 frame/slice    (baseline, its native regime)
    vggt_noreg224_k5_ep300         k=1 model      @ 5 frames/slice

Reads evaluation/metric_results/<ds>/<arm>.json (after eval_pooled_val.sh + eval_ef_dice.sh have
run) and reports per-subject PAIRED comparisons (docs/experiment_evaluation_instruction §6):
NCC primary (docs/88), PSNR/SSIM secondary, breathing slope/EPE, and the nnU-Net EF block
(EF bias, MAE, slope, Spearman, EDV/ESV error, SV ratio, LV Dice at ES). Wilcoxon signed-rank
p-values vs the baseline arm. Nothing is recomputed from volumes.

Run (from the repo root that holds the arms — the arm/burst5 worktree):
  PYTHONPATH=training:. python tools/compare_burst_arms.py [--out temp/burst_eval/compare.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import paths  # noqa: E402

ARMS = {
    "vggt_burst5noreg224_ep300": "burst5 @k5",
    "vggt_burst5noreg224_k1_ep300": "burst5 @k1",
    "vggt_noreg224_ep300": "noreg  @k1 (base)",
    "vggt_noreg224_k5_ep300": "noreg  @k5",
}
BASE = "vggt_noreg224_ep300"
SOURCES = ["cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms", "miitt", "ocmr"]

# per-subject scalar columns: (label, key, higher_is_better)
IMG_COLS = [
    ("NCC", "breath_ncc", True),
    ("PSNR", "breath_psnr", True),
    ("SSIM", "breath_ssim", True),
    ("resp slope", "resp_slope", True),
    ("resp EPE mm", "resp_epe_dz_mm", False),
]
EF_COLS = [
    ("EF err (pred-gt)", lambda r: r["ef_breath"] - r["ef_gt"], None),
    ("|EF err|", lambda r: abs(r["ef_breath"] - r["ef_gt"]), False),
    ("EDV err ml", lambda r: r["edv_breath"] - r["edv_gt"], None),
    ("ESV err ml", lambda r: r["esv_breath"] - r["esv_gt"], None),
    ("SV ratio", lambda r: (r["edv_breath"] - r["esv_breath"]) / (r["edv_gt"] - r["esv_gt"]), None),
    ("Dice LV ES", lambda r: r["dice_breath_LV_ES"], True),
    ("Dice LV ED", lambda r: r["dice_breath_LV_ED"], True),
]


def load(arm):
    rows = {}
    for ds in SOURCES:
        f = paths.RESULTS / ds / f"{arm}.json"
        if not f.is_file():
            print(f"  !! missing {f.relative_to(ROOT)}", file=sys.stderr)
            continue
        d = json.load(open(f))
        for r in d.get("per_subject", []):
            rows[(ds, r["subject"])] = r
    return rows


def val(r, col):
    try:
        v = col(r) if callable(col) else r.get(col)
        return np.nan if v is None else float(v)
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return np.nan


def paired(a_rows, b_rows, col):
    keys = sorted(set(a_rows) & set(b_rows))
    a = np.array([val(a_rows[k], col) for k in keys])
    b = np.array([val(b_rows[k], col) for k in keys])
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def fmt_p(p):
    return "—" if p is None else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--stat", default="median", choices=["mean", "median"])
    a = ap.parse_args()
    agg = np.nanmedian if a.stat == "median" else np.nanmean

    data = {arm: load(arm) for arm in ARMS}
    base = data[BASE]
    lines = [f"# burst-k5 2x2 — breath arm, val split, pooled {'+'.join(SOURCES)}", ""]
    lines.append(f"n subjects per arm: " + ", ".join(f"{ARMS[k]}={len(v)}" for k, v in data.items()))
    lines.append("")

    def block(title, cols):
        lines.append(f"## {title}  ({a.stat}; Δ vs baseline paired per subject; Wilcoxon p)")
        lines.append("")
        hdr = "| metric | " + " | ".join(ARMS.values()) + " |"
        lines.append(hdr)
        lines.append("|" + "---|" * (len(ARMS) + 1))
        for label, col, hib in cols:
            cells = []
            for arm, rows in data.items():
                if not rows:
                    cells.append("—")
                    continue
                x = np.array([val(r, col) for r in rows.values()])
                s = f"{agg(x):.3f}"
                if arm != BASE and base:
                    xa, xb = paired(rows, base, col)
                    if len(xa) >= 5:
                        d = agg(xa - xb)
                        try:
                            p = wilcoxon(xa, xb).pvalue if np.any(xa != xb) else 1.0
                        except ValueError:
                            p = None
                        arrow = ""
                        if hib is not None and p is not None and p < 0.05:
                            arrow = " ✓" if (d > 0) == hib else " ✗"
                        s += f" (Δ{d:+.3f}, p={fmt_p(p)}{arrow}, n={len(xa)})"
                cells.append(s)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    block("Image / breathing (per-subject)", IMG_COLS)
    block("EF / volumes (nnU-Net Task114, per-subject)", EF_COLS)

    # cohort-level EF regression stats (not per-subject; reported per arm, no p)
    lines.append("## EF cohort regression (pred vs GT, pooled)")
    lines.append("")
    lines.append("| arm | n | slope | Spearman r | bias (median) | MAE |")
    lines.append("|---|---|---|---|---|---|")
    for arm, rows in data.items():
        gt = np.array([val(r, "ef_gt") for r in rows.values()])
        pr = np.array([val(r, "ef_breath") for r in rows.values()])
        m = np.isfinite(gt) & np.isfinite(pr)
        if m.sum() < 5:
            lines.append(f"| {ARMS[arm]} | {m.sum()} | — | — | — | — |")
            continue
        slope = np.polyfit(gt[m], pr[m], 1)[0]
        rho = spearmanr(gt[m], pr[m]).correlation
        lines.append(f"| {ARMS[arm]} | {m.sum()} | {slope:.3f} | {rho:.3f} | "
                     f"{np.median(pr[m] - gt[m]):+.2f} | {np.mean(np.abs(pr[m] - gt[m])):.2f} |")
    lines.append("")

    # per-source NCC / EF-bias breakdown
    lines.append("## Per-source NCC (median) / EF bias (median)")
    lines.append("")
    lines.append("| source | " + " | ".join(ARMS.values()) + " |")
    lines.append("|" + "---|" * (len(ARMS) + 1))
    for ds in SOURCES:
        cells = []
        for arm, rows in data.items():
            sub = [r for (d, _), r in rows.items() if d == ds]
            if not sub:
                cells.append("—")
                continue
            ncc = np.nanmedian([val(r, "breath_ncc") for r in sub])
            efb = np.nanmedian([val(r, lambda r: r["ef_breath"] - r["ef_gt"]) for r in sub])
            cells.append(f"{ncc:.3f} / {efb:+.1f} (n={len(sub)})")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")

    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
