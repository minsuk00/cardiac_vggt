#!/usr/bin/env python
"""Build the SELF-CONTAINED HTML report for the frames-per-slice experiment (model 4wokxzov):
every plot + cycle GIF + montage is base64-embedded, so the .html is one portable file.

Inputs:
  result/frames_sweep/results.json         (tools/exp_frames_sweep.py — intensity/coverage sweep)
  result/frames_ef/fps{01,05}/ef.json      (inference/seg_metrics_cmrxrecon.py — EF/Dice per fps)
  result/frames_ef/fps{01,05}/scatter.png  (EF scatter per fps)
  result/frames_ef/fps{01,05}/examples.png (ED-vs-ES demo per fps)
  result/frames_ef/fps{01,05}/viz/*.gif|*ED.png  (qualitative cycle/montage per fps)

  micromamba run -n svr python tools/build_frames_report.py
"""
import argparse
import base64
import json
import os
from collections import defaultdict
from io import BytesIO

import numpy as np


def b64_file(path):
    if not path or not os.path.exists(path):
        return None
    mime = "image/gif" if path.endswith(".gif") else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def b64_fig(fig):
    import matplotlib.pyplot as plt
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def img(uri, w="100%", cap=None):
    if uri is None:
        return "<p style='color:#a00'>[missing figure]</p>"
    c = f"<div class=cap>{cap}</div>" if cap else ""
    return f"<figure><img src='{uri}' style='width:{w}'>{c}</figure>"


def agg(rows, metric):
    b = defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is not None and v == v:
            b[(r["mode"], r["frames_per_slice"])].append(v)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="result/frames_sweep/results.json")
    ap.add_argument("--ef_dirs", nargs="*", default=["result/frames_ef/fps01", "result/frames_ef/fps05"])
    ap.add_argument("--out", default="_html/25_frames_sweep.html")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(args.sweep))
    modes = sorted({r["mode"] for r in rows})
    fpss = sorted({r["frames_per_slice"] for r in rows})
    subs = sorted({r["subject"] for r in rows})
    fmin, fmax = min(fpss), max(fpss)

    # ── quantitative plots ────────────────────────────────────────────────
    panel = [("motion_mean", "motion PSNR (dB)"), ("bbox_mean", "bbox PSNR (dB)"),
             ("full_mean", "full PSNR (dB)"), ("ssim_mean", "SSIM"),
             ("coverage", "splat coverage"), ("motion_ED", "motion PSNR @ ED (dB)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (met, ylab) in zip(axes.ravel(), panel):
        a = agg(rows, met)
        for mode in modes:
            xs = [f for f in fpss if (mode, f) in a]
            ys = [a[(mode, f)][0] for f in xs]; es = [a[(mode, f)][1] for f in xs]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=mode)
        ax.set_title(ylab); ax.set_xlabel("frames per non-ref slice")
        ax.set_xticks(fpss); ax.grid(alpha=0.3)
    axes.ravel()[0].legend()
    fig.suptitle(f"Metric vs frames-per-slice  (N={len(subs)} subjects)", fontsize=12)
    fig.tight_layout()
    uri_curves = b64_fig(fig)

    # per-phase motion PSNR, fmin vs fmax
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for f, style in ((fmin, "o--"), (fmax, "o-")):
            per = [r["motion_per_phase"] for r in rows
                   if r["mode"] == mode and r["frames_per_slice"] == f]
            if per:
                ax.plot(range(len(per[0])), np.array(per).mean(0), style, label=f"fps={f}")
        ax.set_title(f"per-phase motion PSNR ({mode})")
        ax.set_xlabel("cardiac phase t (0=ED)"); ax.set_ylabel("motion PSNR (dB)")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    uri_phase = b64_fig(fig)

    # paired per-subject delta (fmax - fmin), clean
    metpair = ["motion_mean", "bbox_mean", "full_mean", "ssim_mean", "coverage", "motion_ED", "motion_ES"]
    fig, axes = plt.subplots(1, len(metpair), figsize=(2.3 * len(metpair), 3.4), squeeze=False)
    for ax, met in zip(axes[0], metpair):
        for mode in modes:
            d = defaultdict(dict)
            for r in rows:
                if r["frames_per_slice"] in (fmin, fmax):
                    d[(mode, r["subject"])][r["frames_per_slice"]] = r.get(met) if r["mode"] == mode else None
            dl = [v[fmax] - v[fmin] for k, v in d.items()
                  if k[0] == mode and fmin in v and fmax in v and v[fmin] is not None and v[fmax] is not None]
            if dl:
                ax.scatter([mode] * len(dl), dl, alpha=0.6)
        ax.axhline(0, color="k", lw=0.7); ax.set_title(f"Δ{met}", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(f"Paired per-subject improvement (fps{fmax} − fps{fmin})", fontsize=11)
    fig.tight_layout()
    uri_paired = b64_fig(fig)

    # ── numbers for narrative ─────────────────────────────────────────────
    def mv(met, mode, f):
        a = agg(rows, met).get((mode, f))
        return a[0] if a else float("nan")
    clean_tbl = {met: {f: mv(met, "clean", f) for f in fpss}
                 for met in ["motion_mean", "bbox_mean", "full_mean", "ssim_mean", "coverage",
                             "motion_ED", "motion_ES"]}

    # ── EF stage ──────────────────────────────────────────────────────────
    ef = {}
    for d in args.ef_dirs:
        j = os.path.join(d, "ef.json")
        fps = os.path.basename(d).replace("fps", "").lstrip("0") or "0"
        if os.path.exists(j):
            data = json.load(open(j))
            clean = data.get("per_mode", {}).get("clean", {})
            rowsef = [r for r in data.get("rows", []) if r["mode"] == "clean"]
            err = float(np.mean([abs(r["pred_ef"] - r["gt_ef"]) for r in rowsef])) if rowsef else float("nan")
            ef[fps] = dict(dir=d, slope=clean.get("ef_fit", {}).get("slope"),
                           pearson=clean.get("ef_fit", {}).get("pearson"),
                           n=clean.get("n"), pred_mean=clean.get("pred_ef_mean"),
                           gt_mean=clean.get("gt_ef_mean"), mae=err, dice=clean.get("dice_mean", {}),
                           scatter=b64_file(os.path.join(d, "scatter.png")),
                           examples=b64_file(os.path.join(d, "cmrxrecon_ef_examples.png")))

    # ── qualitative viz (cycle GIF + ED montage), fps01 vs fps05, subj0 ────
    def viz(fps, subj, kind):
        return b64_file(os.path.join(f"result/frames_ef/fps{fps:02d}", "viz",
                                     f"subj{subj}_fps{fps:02d}_{kind}"))

    # ── assemble HTML ─────────────────────────────────────────────────────
    def tbl_clean():
        h = "<table><tr><th>metric</th>" + "".join(f"<th>fps={f}</th>" for f in fpss)
        h += f"<th>Δ(fps{fmax}−fps{fmin})</th></tr>"
        names = {"motion_mean": "motion PSNR", "bbox_mean": "bbox PSNR", "full_mean": "full PSNR",
                 "ssim_mean": "SSIM", "coverage": "coverage", "motion_ED": "motion@ED",
                 "motion_ES": "motion@ES"}
        for met, nm in names.items():
            h += f"<tr><td>{nm}</td>"
            for f in fpss:
                h += f"<td>{clean_tbl[met][f]:.3f}</td>"
            dd = clean_tbl[met][fmax] - clean_tbl[met][fmin]
            col = "#0a0" if dd > 0 else "#a00"
            h += f"<td style='color:{col}'>{dd:+.3f}</td></tr>"
        return h + "</table>"

    def ef_tbl():
        if not ef:
            return "<p>(EF stage not yet available)</p>"
        ks = sorted(ef.keys(), key=int)
        h = "<table><tr><th>fps</th><th>N</th><th>EF slope</th><th>EF pearson</th>"
        h += "<th>mean|EF err|</th><th>pred/true EF%</th></tr>"
        for k in ks:
            e = ef[k]
            sl = "n/a" if e["slope"] is None else f"{e['slope']:+.2f}"
            pr = "n/a" if e["pearson"] is None else f"{e['pearson']:+.2f}"
            h += (f"<tr><td>{k}</td><td>{e['n']}</td><td>{sl}</td><td>{pr}</td>"
                  f"<td>{e['mae']:.1f}</td><td>{e['pred_mean']:.0f} / {e['gt_mean']:.0f}</td></tr>")
        return h + "</table>"

    dmotion = clean_tbl["motion_mean"][fmax] - clean_tbl["motion_mean"][fmin]
    ded = clean_tbl["motion_ED"][fmax] - clean_tbl["motion_ED"][fmin]
    des = clean_tbl["motion_ES"][fmax] - clean_tbl["motion_ES"][fmin]
    dcov = clean_tbl["coverage"][fmax] - clean_tbl["coverage"][fmin]

    ef_slope_line = ""
    if "1" in ef and "5" in ef:
        s1, s5 = ef["1"]["slope"], ef["5"]["slope"]
        if s1 is not None and s5 is not None:
            ef_slope_line = (f"EF slope moves {s1:+.2f} → {s5:+.2f} going 1→5 frames "
                             f"(1.0 = perfect amplitude; 0 = flat-EF failure).")

    qual = ""
    for subj in (0, 1):
        g1, g5 = viz(1, subj, "cycle.gif"), viz(5, subj, "cycle.gif")
        if g1 or g5:
            qual += f"<h3>Subject {subj} — beating-heart cycle (GT top / pred bottom)</h3>"
            qual += "<div class=row>"
            qual += f"<div>{img(g1, cap='fps=1 (baseline)')}</div>"
            qual += f"<div>{img(g5, cap='fps=5 (proposed)')}</div>"
            qual += "</div>"
        e1, e5 = viz(1, subj, "ED.png"), viz(5, subj, "ED.png")
        if e1 or e5:
            qual += f"<h4>Subject {subj} — ED volume (pred / GT / |diff|)</h4>"
            qual += img(e1, cap="fps=1") + img(e5, cap="fps=5")

    ef_figs = ""
    for k in sorted(ef.keys(), key=int):
        e = ef[k]
        ef_figs += f"<h3>fps={k}</h3><div class=row><div>{img(e['scatter'], cap='EF: pred vs true')}</div>"
        ef_figs += f"<div>{img(e['examples'], cap='ED vs ES, LV contour')}</div></div>"

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Frames-per-slice experiment — model 4wokxzov</title>
<style>
 body{{font-family:system-ui,-apple-system,sans-serif;max-width:1150px;margin:2em auto;
   padding:0 1em;line-height:1.55;color:#1a1a1a}}
 h1{{border-bottom:3px solid #333;padding-bottom:.3em}}
 h2{{margin-top:1.8em;border-bottom:1px solid #ccc;padding-bottom:.2em}}
 table{{border-collapse:collapse;margin:1em 0}} th,td{{border:1px solid #bbb;padding:5px 10px;text-align:center}}
 th{{background:#f0f0f0}} figure{{margin:.6em 0}} .cap{{font-size:.85em;color:#555;text-align:center}}
 .row{{display:flex;gap:1em;flex-wrap:wrap}} .row>div{{flex:1;min-width:340px}}
 .box{{background:#f7f7f9;border-left:4px solid #4a7;padding:.6em 1em;margin:1em 0}}
 code{{background:#eee;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Does feeding more frames per slice help? — model 4wokxzov</h1>
<div class=box>
<b>TL;DR.</b> More frames per slice <b>helps reconstruction quality</b> at cardiac phases the model
would otherwise have to guess (motion PSNR {dmotion:+.2f} dB going fps {fmin}→{fmax}, concentrated
in mid/late systole: ES {des:+.2f} dB vs ED {ded:+.2f} dB), and <b>slightly improves</b> bbox/full
PSNR + SSIM. It does <b>not</b> help coverage ({dcov:+.3f} — already saturated at fps={fmin}, since
the reference plane alone contributes all 12 phases). {ef_slope_line} The setup is
<b>sound and worth using</b>: the model is set-attention over slots, so feeding more frames at
inference than the training S-budget allowed is exactly what it is for. The one real caveat —
because the model is not told each input frame's phase, extra off-phase frames get splat-averaged
into mild temporal blur — is visible as the tiny ED regression, but is outweighed by better
coverage of the cycle.
</div>

<h2>1 · Setup</h2>
<p>Reference-slot z-only model (wandb <code>4wokxzov</code>), CMRxRecon val, N={len(subs)} subjects
{subs}. The mid-ventricular <b>reference plane always contributes all 12 cardiac phases</b> (swept
as the slot-0 query). Each <i>other</i> in-bbox plane contributes <code>frames_per_slice</code>
consecutive phases starting at a random (seeded) index — a short real-time acquisition burst. We
sweep <code>frames_per_slice ∈ {fpss}</code>. Metrics come from the same
<code>compute_volume_intensity_loss</code> the trainer uses. "coverage" = fraction of in-bbox
voxels the differentiable splat filled.</p>
<p><b>Two opposing effects under test.</b> (+) more phases per plane → more chance a frame sits near
the target phase, and fuller cycle coverage; (−) the model is <i>not</i> told each frame's phase,
so the splat <b>averages</b> unresolved phases → temporal blur of the moving myocardium.</p>

<h2>2 · Quantitative sweep (clean input, subject-mean)</h2>
{tbl_clean()}
{img(uri_curves)}

<h2>3 · Where do extra frames help across the cycle?</h2>
<p>Per-phase motion PSNR at the fewest vs most frames. Gains concentrate away from ED (phase 0) —
the phase a sparse random burst is least likely to observe — confirming the mechanism is
"fill in the phases a single frame misses", not global sharpening.</p>
{img(uri_phase)}

<h2>4 · Consistency across subjects</h2>
<p>Paired per-subject change from fps{fmin}→fps{fmax}. Points above 0 = more frames helped that
subject on that metric.</p>
{img(uri_paired)}

<h2>5 · Qualitative — you can SEE it</h2>
<p>Beating-heart cycle: only the mid reference plane is given per phase; every other plane is
inferred. fps=1 shows more inter-phase flicker/blur off-reference; fps=5 is steadier.</p>
{qual or "<p>(qualitative viz pending EF stage)</p>"}

<h2>6 · EF — does more frames recover contraction AMPLITUDE?</h2>
<p>The reference-slot design exists to fix the flat-EF failure (docs 24–25). Here we re-segment the
reconstructed volumes with the M&amp;Ms nnU-Net (Task114) and compute EF = (max−min)/max on the LV
curve, per frame-count, on clean input. Slope→1 means the model tracks each patient's true EF;
slope→0 is the old flat-EF collapse.</p>
{ef_tbl()}
{ef_figs or "<p>(EF stage not yet available — run tools/exp_frames_ef.py + inference/seg_cmrxrecon.sh + inference/seg_metrics_cmrxrecon.py)</p>"}

<h2>7 · Conclusion</h2>
<ul>
<li><b>The multi-frame setup is correct</b>, not a mismatch: inference is not S-budget-capped, so
feeding a realistic short burst per slice is the intended deployment and the model handles it.</li>
<li><b>More frames helps recon quality</b> ({dmotion:+.2f} dB motion PSNR, mostly at systole) and
mildly helps bbox/full/SSIM — the extra frames fill cardiac phases a single random frame misses.</li>
<li><b>Coverage is not the lever</b> — it is already saturated by the 12-phase reference plane.</li>
<li><b>The averaging caveat is real but minor</b> (small ED regression) — worth watching if frames
grow very large or if per-frame phase were ever supplied.</li>
</ul>
</body></html>"""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, self-contained)")


if __name__ == "__main__":
    main()
