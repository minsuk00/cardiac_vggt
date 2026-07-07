#!/usr/bin/env python
"""Aggregate + visualize tools/exp_frames_sweep.py results into curves + an HTML report.

Reads result/frames_sweep/results.json (partial OK) and produces:
  frames_curves.png     metric vs frames_per_slice (subject-mean±sd), clean vs breathing
  frames_perphase.png   per-phase motion PSNR at fps=min vs fps=max (WHERE the frames help)
  frames_paired.png     per-subject paired delta (fps_max - fps_min) for each metric
  _html/25_frames_sweep.html   the write-up

  micromamba run -n svr python tools/analyze_frames_sweep.py
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def agg(rows, key, metric):
    """mean/std over subjects for each (mode, frames_per_slice) -> {(mode,fps): (mean,std,n)}."""
    buckets = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None and v == v:
            buckets[(r["mode"], r["frames_per_slice"])].append(v)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in buckets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="result/frames_sweep/results.json")
    ap.add_argument("--out", default="result/frames_sweep")
    ap.add_argument("--html", default="_html/25_frames_sweep.html")
    args = ap.parse_args()
    rows = load(args.results)
    modes = sorted({r["mode"] for r in rows})
    fpss = sorted({r["frames_per_slice"] for r in rows})
    subs = sorted({r["subject"] for r in rows})
    metrics = ["motion_mean", "bbox_mean", "full_mean", "ssim_mean", "coverage",
               "motion_ED", "motion_ES"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── curves: metric vs frames ──────────────────────────────────────────
    panel = ["motion_mean", "bbox_mean", "full_mean", "ssim_mean", "coverage", "motion_ED"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, met in zip(axes.ravel(), panel):
        a = agg(rows, None, met)
        for mode in modes:
            xs = [f for f in fpss if (mode, f) in a]
            ys = [a[(mode, f)][0] for f in xs]
            es = [a[(mode, f)][1] for f in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=mode)
        ax.set_title(met); ax.set_xlabel("frames per non-ref slice"); ax.grid(alpha=0.3)
        ax.set_xticks(fpss)
    axes.ravel()[0].legend()
    fig.suptitle(f"Frames-per-slice sweep, model 4wokxzov  (N={len(subs)} subjects: {subs})",
                 fontsize=11)
    fig.tight_layout()
    p_curves = os.path.join(args.out, "frames_curves.png")
    fig.savefig(p_curves, dpi=100); plt.close(fig)

    # ── per-phase motion PSNR: fps_min vs fps_max ─────────────────────────
    fmin, fmax = min(fpss), max(fpss)
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for f, style in ((fmin, "o--"), (fmax, "o-")):
            per = [r["motion_per_phase"] for r in rows
                   if r["mode"] == mode and r["frames_per_slice"] == f]
            if not per:
                continue
            arr = np.array(per)  # (n_subj, T)
            mu = arr.mean(0)
            ax.plot(range(len(mu)), mu, style, label=f"fps={f}")
        ax.set_title(f"per-phase motion PSNR ({mode})")
        ax.set_xlabel("cardiac phase t"); ax.set_ylabel("motion PSNR (dB)")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p_phase = os.path.join(args.out, "frames_perphase.png")
    fig.savefig(p_phase, dpi=100); plt.close(fig)

    # ── paired per-subject delta (fmax - fmin) ────────────────────────────
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 3.6), squeeze=False)
    for ax, met in zip(axes[0], metrics):
        for mode in modes:
            d = {}
            for r in rows:
                if r["mode"] == mode and r["frames_per_slice"] in (fmin, fmax):
                    d.setdefault(r["subject"], {})[r["frames_per_slice"]] = r.get(met)
            deltas = [d[s][fmax] - d[s][fmin] for s in d
                      if fmin in d[s] and fmax in d[s]
                      and d[s][fmin] is not None and d[s][fmax] is not None]
            if deltas:
                ax.scatter([mode] * len(deltas), deltas, alpha=0.6)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"Δ{met}\n(fps{fmax}-fps{fmin})", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Paired per-subject improvement from more frames", fontsize=11)
    fig.tight_layout()
    p_paired = os.path.join(args.out, "frames_paired.png")
    fig.savefig(p_paired, dpi=100); plt.close(fig)

    # ── HTML ──────────────────────────────────────────────────────────────
    def table():
        cols = ["motion_mean", "bbox_mean", "full_mean", "ssim_mean", "coverage",
                "motion_ED", "motion_ES"]
        h = "<table border=1 cellpadding=5 style='border-collapse:collapse'><tr>"
        h += "<th>mode</th><th>fps</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        for mode in modes:
            for f in fpss:
                h += f"<tr><td>{mode}</td><td>{f}</td>"
                for c in cols:
                    a = agg(rows, None, c).get((mode, f))
                    h += f"<td>{a[0]:.3f}±{a[1]:.2f}</td>" if a else "<td>-</td>"
                h += "</tr>"
        return h + "</table>"

    html = f"""<!doctype html><meta charset=utf-8>
<title>Frames-per-slice sweep (4wokxzov)</title>
<body style='font-family:system-ui;max-width:1100px;margin:2em auto;line-height:1.5'>
<h1>Does feeding more frames per slice help? (model 4wokxzov)</h1>
<p><b>Setup.</b> Reference-slot z-only model, CMRxRecon val, N={len(subs)} subjects {subs}.
The mid reference plane always contributes all 12 cardiac phases; each other in-bbox plane
contributes <code>frames_per_slice</code> consecutive phases (random-start burst). Metrics from
<code>compute_volume_intensity_loss</code> (identical to training). "coverage" = fraction of
in-bbox voxels the splat filled.</p>
<h2>Summary (subject-mean ± sd)</h2>
{table()}
<h2>Metric vs frames</h2>
<img src="../result/frames_sweep/frames_curves.png" width=100%>
<h2>Per-phase motion PSNR: few vs many frames</h2>
<p>Shows WHERE extra frames help across the cardiac cycle (ED = phase 0).</p>
<img src="../result/frames_sweep/frames_perphase.png" width=100%>
<h2>Paired per-subject improvement (fps{fmax} − fps{fmin})</h2>
<img src="../result/frames_sweep/frames_paired.png" width=100%>
</body>"""
    os.makedirs(os.path.dirname(args.html), exist_ok=True)
    with open(args.html, "w") as f:
        f.write(html)
    print(f"wrote {p_curves}\n{p_phase}\n{p_paired}\n{args.html}")


if __name__ == "__main__":
    main()
