"""Build _html/11_refiner_full_analysis.html — thorough, motion-led refiner analysis.

Self-contained (matplotlib -> base64 PNGs embedded; no external assets).

Inputs:
  /tmp/refiner_trajectories.json       per-val-epoch series for all 3 runs (parsed from slurm logs)
  result/refiner_eval/frozen.json      eval_refiner on /tmp/an_frozen.pt (n=100)
  result/refiner_eval/joint.json       eval_refiner on /tmp/an_joint.pt  (n=100)
  result/refiner_eval/panels/frozen_seq{0,7}*.png, joint_seq{0,7}*.png

Run: micromamba run -n svr python _html/build_refiner_full_analysis.py
"""
import base64
import glob
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/minsukc/vggt"
OUT = os.path.join(REPO, "_html", "11_refiner_full_analysis.html")
TRAJ = "/tmp/refiner_trajectories.json"
FROZEN_J = os.path.join(REPO, "result", "refiner_eval", "frozen.json")
JOINT_J = os.path.join(REPO, "result", "refiner_eval", "joint.json")
PANELS = os.path.join(REPO, "result", "refiner_eval", "panels")

IDENT_BBOX, IDENT_MOTION = 23.23, 16.59  # breathing identity floor (docs/05)

# latest logged numbers (anchors, from slurm logs at snapshot 2026-06-17 21:05)
VAR2_MOTION_LATEST = 19.28   # step 84000 (fluctuates 19.28-19.55 over last ~6 epochs)
VAR2_MOTION_BEST = 19.55     # best of last ~6 val epochs (most-favorable var2 read)
VAR2_BBOX_LATEST = 26.73

traj = json.load(open(TRAJ))
frozen_ev = json.load(open(FROZEN_J))["summary"]
joint_ev = json.load(open(JOINT_J))["summary"]


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def png_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def series(run, key):
    pts = [(d["step"], d[key]) for d in traj[run] if key in d]
    return [p[0] for p in pts], [p[1] for p in pts]


def slope_last(run, key, k=10):
    x, y = series(run, key)
    if len(y) < 3:
        return None
    x = np.array(x[-k:], float)
    y = np.array(y[-k:], float)
    s = np.polyfit(x, y, 1)[0] * 1000  # per 1000 steps
    return s, y[0], y[-1], len(y)


# ---------- trajectory plots ----------
def fig_traj(canon_key, refined_key, title, floor):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = {"frozen": "#1f77b4", "joint": "#2ca02c", "var2": "#7f7f7f"}
    # frozen
    xs, ys = series("frozen", canon_key)
    ax.plot(xs, ys, "--", color=colors["frozen"], lw=1.3,
            label="frozen: V_canon (raw splat, geometry FROZEN)")
    xs, ys = series("frozen", refined_key)
    ax.plot(xs, ys, "-", color=colors["frozen"], lw=2.4, label="frozen: V_refined (refiner)")
    # joint
    xs, ys = series("joint", canon_key)
    ax.plot(xs, ys, "--", color=colors["joint"], lw=1.3, label="joint: V_canon")
    xs, ys = series("joint", refined_key)
    ax.plot(xs, ys, "-", color=colors["joint"], lw=2.4, label="joint: V_refined (refiner)")
    # var2 (no refiner) - canon only
    xs, ys = series("var2", canon_key)
    if ys:
        ax.plot(xs, ys, ":", color=colors["var2"], lw=2.0,
                label="var2: V_canon (no-refiner control)")
    ax.axhline(floor, color="#d62728", lw=1.0, ls=(0, (2, 2)), alpha=0.8,
               label=f"identity floor ({floor:.2f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    return fig_b64(fig)


traj_motion = fig_traj("val_motion", "val_motion_refined",
                       "Motion PSNR (dynamic/heart voxels) — PRIMARY METRIC", IDENT_MOTION)
traj_bbox = fig_traj("val_psnr_bbox", "val_psnr_bbox_refined",
                     "bbox PSNR (content region) — secondary", IDENT_BBOX)


# ---------- sharpness bar ----------
def fig_sharp():
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    runs = ["frozen", "joint"]
    canon = [frozen_ev["rel_canon"], joint_ev["rel_canon"]]
    refined = [frozen_ev["rel_refined"], joint_ev["rel_refined"]]
    x = np.arange(len(runs))
    w = 0.36
    b1 = ax.bar(x - w / 2, canon, w, label="V_canon (raw splat)", color="#9ecae1")
    b2 = ax.bar(x + w / 2, refined, w, label="V_refined (refiner)", color="#1f77b4")
    ax.axhline(1.0, color="#0a7d28", lw=1.2, ls="--", label="GT sharpness (1.0)")
    ax.axhline(0.668, color="#999", lw=0.9, ls=":", label="report-08 baseline 0.668")
    ax.set_xticks(x)
    ax.set_xticklabels([r + "\n(n=100)" for r in runs])
    ax.set_ylabel("in-plane gradient energy / GT")
    ax.set_ylim(0, 1.05)
    ax.set_title("Sharpness vs GT — V_refined is sharper, but the gain is MODEST")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2, axis="y")
    return fig_b64(fig)


sharp_fig = fig_sharp()

# ---------- panels ----------
panels = {}
for tag in ["frozen", "joint"]:
    for seq in [0, 7]:
        cand = glob.glob(os.path.join(PANELS, f"{tag}_seq{seq}_*.png"))
        if cand:
            panels[f"{tag}_{seq}"] = png_b64(cand[0])

# ---------- convergence numbers ----------
sl_frozen_m = slope_last("frozen", "val_motion_refined")
sl_joint_m = slope_last("joint", "val_motion_refined")
sl_var2_m = slope_last("var2", "val_motion")
sl_frozen_canon = slope_last("frozen", "val_motion")

n_frozen = len(traj["frozen"])
n_joint = len(traj["joint"])
n_var2 = len(traj["var2"])


def d(a, b):
    return f"{b - a:+.2f}"


HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>VGGT-MRI: refiner full analysis (motion-led)</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1040px;margin:0 auto;padding:32px 24px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:25px;margin-bottom:4px}} h2{{margin-top:32px;border-bottom:2px solid #eee;padding-bottom:6px;font-size:20px}}
.sub{{color:#666;margin-top:0}}
code,.mono{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:#f3f4f6;padding:1px 5px;border-radius:4px}}
img{{max-width:100%;border:1px solid #e2e2e2;border-radius:6px;margin:10px 0}}
.callout{{background:#f0f6ff;border-left:4px solid #1f77b4;padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0}}
.callout.key{{background:#eafaf0;border-color:#0a7d28}} .callout.warn{{background:#fff8e6;border-color:#d9a300}}
.note{{font-size:13px;color:#666}}
table{{border-collapse:collapse;margin:14px 0;font-size:13px;width:100%}}
th,td{{border:1px solid #e2e2e2;padding:5px 8px;text-align:left}} th{{background:#f7f7f7}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
ul{{margin-top:6px}} li{{margin:5px 0}}
.best{{background:#eafaf0;font-weight:600}}
blockquote{{background:#f7f9fc;border-left:4px solid #1f77b4;margin:16px 0;padding:12px 18px;border-radius:0 6px 6px 0}}
</style></head><body>

<h1>3D UNet refiner — full analysis (motion-led)</h1>
<p class="sub">Snapshot 2026-06-17 21:05. All three runs validate on the <b>same</b> breathing-corrupted
val task (deterministic, 30 subjects, multiphase) &rarr; their val metrics are <b>directly comparable</b>
(unlike report 07's cross-task issue) — but at <b>unequal epochs</b>. Last-checkpoint re-eval via
<span class="mono">tools/eval_refiner.py</span> (n=100); trajectories parsed from
<span class="mono">slurm_logs/*</span>. How-it-works: <span class="mono">09_unet_refiner.html</span>;
prior snapshot: <span class="mono">10_refiner_results.html</span>.</p>

<blockquote><b>TL;DR (motion-led).</b> The refiner <b>conclusively helps on the primary metric (motion PSNR)</b>
on a fixed geometry. On the <b>frozen</b> run (geometry pinned at the var2 ep-59 seed, only the refiner trains)
the refiner lifts motion PSNR <b>{frozen_ev['psnr_motion_canon']:.2f} &rarr; {frozen_ev['psnr_motion_refined']:.2f} dB
({d(frozen_ev['psnr_motion_canon'], frozen_ev['psnr_motion_refined'])})</b> and bbox
<b>{frozen_ev['psnr_bbox_canon']:.2f} &rarr; {frozen_ev['psnr_bbox_refined']:.2f}
({d(frozen_ev['psnr_bbox_canon'], frozen_ev['psnr_bbox_refined'])})</b> on n=100 held-out breathing-val subjects.
That frozen V_refined motion (<b>{frozen_ev['psnr_motion_refined']:.2f}</b>) <b>beats the best no-refiner baseline</b>
(var2, motion ~{VAR2_MOTION_LATEST:.2f} at a <i>more-trained</i> geometry) by ~<b>+0.8 dB motion</b> — i.e. deblurring
the splat beats training the geometry further. <b>Caveats:</b> frozen is decelerating but <b>still slowly rising</b>
(its number is a <i>lower bound</i>, not a plateau); <b>joint</b> is only epoch ~{n_joint}/200 (NOT converged); and the
sharpness gain is <b>modest</b> ({frozen_ev['rel_canon']:.3f}&rarr;{frozen_ev['rel_refined']:.3f}&times;GT) — most of the
gain is accuracy/coverage correction, not high-frequency recovery (expected under L1).</blockquote>

<div class="callout warn"><b>Honesty flags (read first).</b>
<ul>
<li><b>frozen V_canon is exactly flat</b> (geometry frozen) — slope {sl_frozen_canon[0]:+.4f}/1k-step. V_refined is
<b>decelerating but still rising</b> ({sl_frozen_m[1]:.2f}&rarr;{sl_frozen_m[2]:.2f} over last {sl_frozen_m[3]} epochs,
slope {sl_frozen_m[0]:+.4f}/1k-step) &rarr; the frozen number is a <b>lower bound</b>, NOT a converged plateau.</li>
<li><b>joint is NOT converged</b> (epoch ~{n_joint}/200, V_refined slope {sl_joint_m[0]:+.4f}/1k-step, ~{sl_joint_m[0]/sl_frozen_m[0]:.0f}&times;
faster than frozen). "frozen &gt; joint now" will likely <b>narrow</b> as joint's geometry catches up.</li>
<li><b>var2 fluctuates</b> (motion {VAR2_MOTION_LATEST:.2f}&ndash;19.55 over the last ~6 val epochs); it is roughly
plateaued (slope {sl_var2_m[0]:+.4f}/1k-step).</li>
<li><b>Sharpness gain is modest</b> — under L1 (mean-seeking) the refiner mostly fixes accuracy/coverage, not
high-frequency detail. A gradient/SSIM loss is the lever for more deblur.</li>
</ul></div>

<h2>1. The three runs (same val task, unequal epochs)</h2>
<table>
<tr><th>run</th><th>geometry (VGGT)</th><th>refiner</th><th>val epochs (snapshot)</th><th>role</th></tr>
<tr><td><b>frozen</b> (51876098)</td><td>FROZEN at var2 ep-59 (pinned)</td><td>trains</td><td class="num">~{n_frozen}</td><td>isolates the pure splat-deblur gain</td></tr>
<tr><td><b>joint</b> (51876099)</td><td>finetunes from VGGT-1B base</td><td>trains</td><td class="num">~{n_joint}/200</td><td>geometry + refiner co-adapt</td></tr>
<tr><td><b>var2</b> (51862799)</td><td>finetunes (resumed ep59&rarr;)</td><td>none</td><td class="num">~{60 + n_var2}</td><td>no-refiner control (V_canon only)</td></tr>
</table>
<p class="note">All three apply the identical deterministic breathing corruption at val (per-<span class="mono">seq_index</span>
seed), so V_refined(frozen) vs V_refined(joint) vs V_canon(var2) vs identity are on one yardstick.</p>

<h2>2. Last-checkpoint comparison — MOTION primary (n=100 held-out breathing val)</h2>
<table>
<tr><th>model (output)</th><th>motion PSNR &uarr;</th><th>&Delta; motion vs var2</th><th>bbox PSNR &uarr;</th><th>sharp/GT</th><th>status</th></tr>
<tr class="best"><td><b>frozen — V_refined</b></td><td class="num">{frozen_ev['psnr_motion_refined']:.2f}</td>
  <td class="num">{frozen_ev['psnr_motion_refined']-VAR2_MOTION_LATEST:+.2f}</td>
  <td class="num">{frozen_ev['psnr_bbox_refined']:.2f}</td><td class="num">{frozen_ev['rel_refined']:.3f}</td>
  <td>decelerating, still rising (lower bound)</td></tr>
<tr><td>joint — V_refined</td><td class="num">{joint_ev['psnr_motion_refined']:.2f}</td>
  <td class="num">{joint_ev['psnr_motion_refined']-VAR2_MOTION_LATEST:+.2f}</td>
  <td class="num">{joint_ev['psnr_bbox_refined']:.2f}</td><td class="num">{joint_ev['rel_refined']:.3f}</td>
  <td>NOT converged (ep ~{n_joint}/200)</td></tr>
<tr><td>var2 — V_canon (best no-refiner)</td><td class="num">{VAR2_MOTION_LATEST:.2f}</td>
  <td class="num">0.00</td><td class="num">{VAR2_BBOX_LATEST:.2f}</td><td class="num">&mdash;</td>
  <td>roughly plateaued (~ep {60 + n_var2})</td></tr>
<tr><td>frozen — V_canon (pinned geom)</td><td class="num">{frozen_ev['psnr_motion_canon']:.2f}</td>
  <td class="num">{frozen_ev['psnr_motion_canon']-VAR2_MOTION_LATEST:+.2f}</td>
  <td class="num">{frozen_ev['psnr_bbox_canon']:.2f}</td><td class="num">{frozen_ev['rel_canon']:.3f}</td>
  <td>frozen geometry (input to refiner)</td></tr>
<tr><td>joint — V_canon</td><td class="num">{joint_ev['psnr_motion_canon']:.2f}</td>
  <td class="num">{joint_ev['psnr_motion_canon']-VAR2_MOTION_LATEST:+.2f}</td>
  <td class="num">{joint_ev['psnr_bbox_canon']:.2f}</td><td class="num">{joint_ev['rel_canon']:.3f}</td><td>co-adapting</td></tr>
<tr><td>identity (do-nothing)</td><td class="num">{IDENT_MOTION:.2f}</td>
  <td class="num">{IDENT_MOTION-VAR2_MOTION_LATEST:+.2f}</td><td class="num">{IDENT_BBOX:.2f}</td><td class="num">&mdash;</td>
  <td>floor</td></tr>
</table>

<div class="callout key"><b>Lever-efficiency read.</b> frozen's V_canon (motion {frozen_ev['psnr_motion_canon']:.2f}) comes
from the <b>older, frozen</b> var2-ep59 geometry — yet a refiner on top of it reaches
<b>{frozen_ev['psnr_motion_refined']:.2f}</b> motion, <b>{frozen_ev['psnr_motion_refined']-VAR2_MOTION_LATEST:+.2f} dB
above</b> var2's V_canon ({VAR2_MOTION_LATEST:.2f}) — and var2 trained that geometry ~{n_var2} epochs <i>further</i> with
no refiner. So <b>deblurring the splat &gt; training the geometry more</b>: the splat (report 08), not the geometry, was
the motion bottleneck. (Caveat: the two geometries differ, so this is a lever-efficiency comparison, not a strict
head-to-head.)</div>

<h2>3. Trajectories — refiner (solid) vs raw splat (dashed)</h2>
<p class="note"><b>Motion (primary).</b> Frozen V_canon (blue dashed) is exactly flat (geometry frozen); V_refined
(blue solid) sits ~+1 dB above it — that gap IS the refiner. Joint (green) climbs steeply on both lines (young).
var2 (grey dotted) is the no-refiner control.</p>
<img src="data:image/png;base64,{traj_motion}">
<p class="note"><b>bbox (secondary).</b> Same structure; the refiner gap is larger in bbox (~+1.5 dB) than motion
(~+1 dB) — coverage/accuracy fixes help the whole content region more than the dynamic subset.</p>
<img src="data:image/png;base64,{traj_bbox}">

<h2>4. Is it a real deblur? Sharpness vs GT (V_canon vs V_refined &times; GT)</h2>
<p class="note">Sharpness = in-plane gradient energy &divide; GT (1.0 = as sharp as GT). If V_refined merely
<i>smoothed</i>, sharpness would <b>fall</b>; instead it <b>rises</b> in both runs &rarr; not blurring. But the rise is
<b>modest</b> (frozen {frozen_ev['rel_canon']:.3f}&rarr;{frozen_ev['rel_refined']:.3f},
joint {joint_ev['rel_canon']:.3f}&rarr;{joint_ev['rel_refined']:.3f}). Most of the PSNR gain is
accuracy/coverage-error correction, not high-frequency recovery — exactly what an L1 (mean-seeking) loss predicts.</p>
<img src="data:image/png;base64,{sharp_fig}">

<h3>Did it meet the original goal? (the splat made the recon blurry + left dark/under-covered spots)</h3>
<div class="callout key">The refiner was added to fix the two things report 08 blamed on the splat: <b>(1) dark / dim
under-covered voxels</b> and <b>(2) blurriness</b>. Scorecard:
<table>
<tr><th>original goal</th><th>fixed?</th><th>evidence</th></tr>
<tr><td><b>Dark / dim under-covered spots</b> (coverage-averaging artifacts)</td><td><b>✅ yes — substantially</b></td>
<td>This is where ~all of the +1.0 dB motion / +1.5 dB bbox comes from. V_refined fills &amp; brightens the
coverage-starved voxels and smooths seams (visible in the panels: V_refined is brighter/cleaner than V_canon).</td></tr>
<tr><td><b>Blurriness</b> (lost in-plane high frequency)</td><td><b>❌ mostly NOT</b></td>
<td>Only <b>~{round(100*(frozen_ev['rel_refined']-frozen_ev['rel_canon'])/(1-frozen_ev['rel_canon']))}% (frozen) / ~{round(100*(joint_ev['rel_refined']-joint_ev['rel_canon'])/(1-joint_ev['rel_canon']))}% (joint)</b>
of the blur-gap to GT is closed; V_refined is still only ~0.69&times; GT sharpness — about as soft as V_canon.</td></tr>
</table>
<b>Why:</b> the loss is <b>L1</b>, which is <i>mean-seeking</i> — it rewards getting the average intensity right
(so it fills dark spots) but does NOT reward sharp edges (so it won't deblur). So the refiner is currently a
<b>coverage / intensity corrector</b> more than a deblurrer.<br>
<b>The fix (to actually kill the blur):</b> add a <b>sharpness-aware supervised loss</b> on V_refined — a <b>2D SSIM</b>
term (per z-slice; the volume is too thin/anisotropic in Z for an isotropic 3D window) and/or a <b>gradient (edge-matching)
loss</b> <span class="mono">|&nabla;V_refined &minus; &nabla;V_gt|</span> (which is the in-plane gradient energy used as the
sharpness <i>metric</i>, turned into a loss). NB: this is NOT total variation — TV penalizes <i>any</i> gradient and would
<i>blur more</i>; a gradient/SSIM loss references GT and rewards matching its edges. Recipe: <span class="mono">L_post =
L1 + &lambda;&middot;(1&minus;SSIM&#95;2d)</span> (Zhao et&nbsp;al. 2017). <span class="mono">fused-ssim</span> (already in the env) provides a fast
differentiable 2D SSIM.</div>

<h2>5. Convergence assessment (per run)</h2>
<table>
<tr><th>run</th><th>metric</th><th>last-10 trend</th><th>slope /1k-step</th><th>verdict</th></tr>
<tr><td>frozen</td><td>V_canon motion</td><td class="num">{sl_frozen_canon[1]:.2f}&rarr;{sl_frozen_canon[2]:.2f}</td>
  <td class="num">{sl_frozen_canon[0]:+.4f}</td><td>PINNED (geometry frozen)</td></tr>
<tr><td>frozen</td><td>V_refined motion</td><td class="num">{sl_frozen_m[1]:.2f}&rarr;{sl_frozen_m[2]:.2f}</td>
  <td class="num">{sl_frozen_m[0]:+.4f}</td><td><b>decelerating, still slowly rising</b> &rarr; lower bound</td></tr>
<tr><td>joint</td><td>V_refined motion</td><td class="num">{sl_joint_m[1]:.2f}&rarr;{sl_joint_m[2]:.2f}</td>
  <td class="num">{sl_joint_m[0]:+.4f}</td><td><b>NOT converged</b> (ep ~{n_joint}/200, climbing fast)</td></tr>
<tr><td>var2</td><td>V_canon motion</td><td class="num">{sl_var2_m[1]:.2f}&rarr;{sl_var2_m[2]:.2f}</td>
  <td class="num">{sl_var2_m[0]:+.4f}</td><td>roughly plateaued (fluctuating)</td></tr>
</table>

<h2>6. Qualitative panels (breathing val) — hallucination check</h2>
<p>Rows: V_gt / V_canon (raw splat) / V_refined / (V_refined &minus; V_gt). Compare row 2 vs row 3; row 4 is the
residual error (red/blue = over/under). V_refined sharpens <b>real</b> structure (PSNR rises on held-out val) and the
residual shrinks toward zero — no fabricated anatomy is visible in these examples (verified, not assumed; n=2 panels —
a broader OOD audit is future work).</p>
<h3>frozen checkpoint</h3>
{("<img src='data:image/png;base64," + panels.get("frozen_0", "") + "'>") if "frozen_0" in panels else ""}
{("<img src='data:image/png;base64," + panels.get("frozen_7", "") + "'>") if "frozen_7" in panels else ""}
<h3>joint checkpoint</h3>
{("<img src='data:image/png;base64," + panels.get("joint_0", "") + "'>") if "joint_0" in panels else ""}
{("<img src='data:image/png;base64," + panels.get("joint_7", "") + "'>") if "joint_7" in panels else ""}

<h2>7. Conclusive vs not-yet-conclusive</h2>
<div class="callout key"><b>Conclusive.</b>
<ul>
<li>On a <b>fixed geometry</b> the refiner adds <b>{d(frozen_ev['psnr_motion_canon'], frozen_ev['psnr_motion_refined'])}
dB motion</b> / <b>{d(frozen_ev['psnr_bbox_canon'], frozen_ev['psnr_bbox_refined'])} dB bbox</b> on held-out
breathing val (n=100) — a clean, isolated splat-deblur gain.</li>
<li>frozen V_refined motion ({frozen_ev['psnr_motion_refined']:.2f}) <b>beats the best no-refiner baseline</b>
(var2 {VAR2_MOTION_LATEST:.2f}) by ~<b>{frozen_ev['psnr_motion_refined']-VAR2_MOTION_LATEST:+.2f} dB</b>, despite var2
having a more-trained geometry &rarr; deblurring the splat is the more efficient lever. Even against var2's
<b>best</b> recent val epoch ({VAR2_MOTION_BEST:.2f}), frozen V_refined still leads by
{frozen_ev['psnr_motion_refined']-VAR2_MOTION_BEST:+.2f} dB.</li>
<li>V_refined is <b>sharper, not smoother</b> (gradient energy rises toward GT in both runs) and shows
<b>no visible hallucination</b> in the inspected panels.</li>
</ul></div>
<div class="callout warn"><b>Not yet conclusive.</b>
<ul>
<li><b>Frozen's ceiling.</b> V_refined motion is still slowly rising (slope {sl_frozen_m[0]:+.4f}/1k-step) &rarr;
{frozen_ev['psnr_motion_refined']:.2f} is a <b>lower bound</b>, not a plateau.</li>
<li><b>Joint's final number.</b> Epoch ~{n_joint}/200, climbing ~{sl_joint_m[0]/sl_frozen_m[0]:.0f}&times; faster than
frozen — its current {joint_ev['psnr_motion_refined']:.2f} will rise; frozen-vs-joint will likely narrow / flip.</li>
<li><b>L1 ceiling on sharpness.</b> The modest sharpness gain suggests L1 caps high-frequency recovery; whether a
gradient/SSIM loss lifts it is untested.</li>
<li><b>Hallucination audit</b> is n=2 panels only — no systematic OOD / held-out-subject sweep yet.</li>
</ul></div>

<h2>8. Next steps</h2>
<ul>
<li><b>Let the joint run converge</b> (ep ~{n_joint}/200) and re-snapshot — expected to match or exceed frozen once its
geometry catches up.</li>
<li><b>Add a sharpness-aware loss</b> (gradient L1 / 1&minus;SSIM) — the modest sharpness gain + frozen's slow rise
suggest L1 caps the deblur; this is the lever to recover more high-frequency detail.</li>
<li><b>Broader hallucination / OOD audit</b> across more subjects and target phases before trusting V_refined clinically.</li>
</ul>

<p class="note">Reproduce: snapshot ckpts to <span class="mono">/tmp/an_frozen.pt</span>,
<span class="mono">/tmp/an_joint.pt</span>; <span class="mono">tools/eval_refiner.py --n 100</span> (both ckpts loaded with
0 missing / 0 unexpected keys, <span class="mono">has_refiner=True</span>); trajectories parsed from
<span class="mono">slurm_logs/*refiner*</span> + <span class="mono">*51862799*</span> &rarr;
<span class="mono">/tmp/refiner_trajectories.json</span>; this page:
<span class="mono">_html/build_refiner_full_analysis.py</span>.</p>
</body></html>"""

with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB)")
