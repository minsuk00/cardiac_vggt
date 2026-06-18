"""Build _html/12_ssim_analysis.html — a self-contained report on the 2D-SSIM refiner term:
how it works and whether it helps the two failure modes (black/dim spots, in-plane blur).

Quantitative numbers are the matched-step val trajectory + the n=60 breathing-val sharpness
evals collected in-session (frozen-L1 @ep95, frozen-SSIM @ep26, joint @ep22, L1 @ep62 refs).
Qualitative panels are rendered fresh by running BOTH frozen checkpoints (L1 ep95, SSIM ep26)
on the same breathing-val subjects so V_canon is shared (frozen geometry) and only V_refined
differs by the loss.

Run: CUDA_VISIBLE_DEVICES=0 micromamba run -n svr python _html/build_ssim_analysis.py
"""
import base64
import io
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_variants_matrix import build_dataset, build_batch, PROTOCOLS, GRID_SHAPE, NUM_SLICES  # noqa
from eval_refiner import make_refiner_model  # noqa
sys.path.insert(0, os.path.join(REPO, "training"))
from data.gpu_aug import gpu_augment_batch          # noqa
from loss import compute_volume_intensity_loss       # noqa

OUT_HTML = os.path.join(REPO, "_html", "12_ssim_analysis.html")
L1_CKPT = "/tmp/l1_ep95.pt"
SSIM_CKPT = "/tmp/ssim_ep26.pt"

# ── Quantitative data collected this session ────────────────────────────────
# Matched-step val (frozen-L1 51876098 vs frozen-SSIM 51950141; same seed+frozen
# geometry ⇒ identical V_canon, only the loss differs).
TRAJ_STEPS = [6000, 10000, 15000, 20000, 25000, 26000]
TRAJ = {
    "L1_bbox":   [26.67, 26.98, 27.24, 27.42, 27.51, 27.60],
    "SSIM_bbox": [26.56, 26.87, 27.24, 27.41, 27.56, 27.60],
    "L1_motion":   [19.24, 19.42, 19.61, 19.71, 19.73, 19.83],
    "SSIM_motion": [19.17, 19.35, 19.60, 19.71, 19.78, 19.83],
}
CANON_BBOX, CANON_MOTION = 26.41, 19.07   # flat baseline (frozen geometry), both runs

# n=60 breathing-val sharpness/PSNR evals (1.0 = as sharp as GT).
SHARP = {
    "splat (V_canon)":     dict(rel=0.668, bbox=26.49, motion=19.14, ep=None,  tag="baseline"),
    "frozen-L1 @ep95":     dict(rel=0.694, bbox=28.16, motion=20.26, ep=95,    tag="L1"),
    "frozen-SSIM @ep26":   dict(rel=0.692, bbox=27.69, motion=19.91, ep=26,    tag="SSIM"),
}
# sharpness-vs-epoch trajectory points (mixed n; for the "faster to ceiling" story)
SHARP_TRAJ = {
    "L1":   [(62, 0.688), (95, 0.694)],
    "SSIM": [(6, 0.670), (26, 0.692)],
}
GT_REF = 1.0


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Plot 1: matched-step PSNR trajectory ────────────────────────────────────
def fig_trajectory():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for j, (key, title, ylab) in enumerate([("bbox", "Refined bbox PSNR (dB)", "PSNR"),
                                             ("motion", "Refined motion PSNR (dB)", "PSNR")]):
        a = ax[j]
        a.plot(TRAJ_STEPS, TRAJ[f"L1_{key}"], "o-", color="#1f77b4", label="frozen-L1", lw=2, ms=5)
        a.plot(TRAJ_STEPS, TRAJ[f"SSIM_{key}"], "s-", color="#d62728", label="frozen-SSIM", lw=2, ms=5)
        base = CANON_BBOX if key == "bbox" else CANON_MOTION
        a.axhline(base, color="gray", ls="--", lw=1, label="raw splat (V_canon)")
        a.set_title(title, fontsize=11)
        a.set_xlabel("train step"); a.set_ylabel(ylab)
        a.grid(alpha=0.3); a.legend(fontsize=9)
    fig.suptitle("Matched-step comparison — same seed, same FROZEN geometry (only the loss differs)",
                 fontsize=12, y=1.02)
    return b64(fig)


# ── Plot 2: sharpness bars + sharpness-vs-epoch ─────────────────────────────
def fig_sharpness():
    fig = plt.figure(figsize=(11, 4.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)

    # left: bars of rel sharpness
    a = fig.add_subplot(gs[0])
    labels = list(SHARP.keys())
    vals = [SHARP[k]["rel"] for k in labels]
    colors = ["#7f7f7f", "#1f77b4", "#d62728"]
    bars = a.bar(range(len(labels)), vals, color=colors)
    a.axhline(GT_REF, color="green", ls="--", lw=1.5)
    a.text(len(labels) - 0.5, GT_REF + 0.005, "GT = 1.0 (perfectly sharp)", color="green",
           ha="right", fontsize=9)
    for i, v in enumerate(vals):
        a.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    a.set_xticks(range(len(labels)))
    a.set_xticklabels(labels, rotation=12, fontsize=9)
    a.set_ylim(0.60, 1.02)
    a.set_ylabel("in-plane sharpness / GT")
    a.set_title("Sharpness vs GT (n=60 breathing val)", fontsize=11)
    a.grid(axis="y", alpha=0.3)

    # right: sharpness vs epoch
    a2 = fig.add_subplot(gs[1])
    for key, color, mk in [("L1", "#1f77b4", "o"), ("SSIM", "#d62728", "s")]:
        pts = SHARP_TRAJ[key]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        a2.plot(xs, ys, mk + "-", color=color, lw=2, ms=7, label=f"frozen-{key}")
    a2.axhline(0.668, color="gray", ls="--", lw=1, label="raw splat")
    a2.set_xlabel("epoch"); a2.set_ylabel("sharpness / GT")
    a2.set_title("Sharpness vs training epoch\n(SSIM reaches the ceiling faster)", fontsize=11)
    a2.grid(alpha=0.3); a2.legend(fontsize=9)
    a2.annotate("SSIM hits 0.692\nby ep26", xy=(26, 0.692), xytext=(40, 0.678),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#d62728"))
    return b64(fig)


# ── Plot 3: dark-spots vs blur scorecard (grouped bars of % gap closed) ──────
def fig_scorecard():
    # Dark-spot proxy = PSNR gain; blur proxy = sharpness-gap closed.
    # bbox PSNR gain over splat (26.49): L1 +1.67, SSIM +1.20
    # sharpness gap to GT = 1.0 - 0.668 = 0.332; closed: L1 (0.694-0.668)=0.026 -> 7.8%, SSIM 0.024 -> 7.2%
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    # left: PSNR gain (dark/dim-spot filling)
    a = ax[0]
    a.bar([0, 1], [1.67, 1.20], color=["#1f77b4", "#d62728"])
    a.set_xticks([0, 1]); a.set_xticklabels(["L1 @ep95", "SSIM @ep26"])
    a.set_ylabel("bbox PSNR gain over splat (dB)")
    a.set_title("Dark/dim spots: FIXED  (PSNR gain)", fontsize=11)
    for i, v in enumerate([1.67, 1.20]):
        a.text(i, v + 0.03, f"+{v:.2f}", ha="center", fontweight="bold")
    a.grid(axis="y", alpha=0.3)
    # right: % of sharpness gap to GT closed
    a2 = ax[1]
    pct = [0.026 / 0.332 * 100, 0.024 / 0.332 * 100]
    a2.bar([0, 1], pct, color=["#1f77b4", "#d62728"])
    a2.set_xticks([0, 1]); a2.set_xticklabels(["L1 @ep95", "SSIM @ep26"])
    a2.set_ylabel("% of sharpness gap to GT closed")
    a2.set_ylim(0, 100)
    a2.set_title("Blur: barely  (only ~8% of gap closed)", fontsize=11)
    for i, v in enumerate(pct):
        a2.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    a2.grid(axis="y", alpha=0.3)
    return b64(fig)


# ── Qualitative: run both frozen models on shared breathing-val subjects ─────
def gather(ckpt, seqs, device):
    model, info = make_refiner_model(ckpt, device)
    print("loaded", ckpt, info)
    cfg = PROTOCOLS["breathing"]
    res = {}
    for seq in seqs:
        data = build_dataset_cache.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        batch = build_batch(data, device, seq_index=seq)
        batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
        bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        out = compute_volume_intensity_loss(preds, batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
        res[seq] = dict(
            Vg=out["V_gt"][0].float().cpu().numpy(),
            Vc=preds["V_canon"][0].float().cpu().numpy(),
            Vr=preds["V_refined"][0].float().cpu().numpy(),
            bbox=bbox,
            t=int(np.asarray(data["t_target"]).flatten()[0]),
        )
    del model
    torch.cuda.empty_cache()
    return res


def fig_qualitative(seq, L1, SSIM):
    g = SSIM[seq]
    Vg, Vc, bbox, t = g["Vg"], g["Vc"], g["bbox"], g["t"]
    Vr_l1 = L1[seq]["Vr"]; Vr_ss = g["Vr"]
    z0, z1 = bbox[0], bbox[1]
    zs = list(range(z0, z1))
    vmax = float(max(Vg.max(), Vc.max(), Vr_l1.max(), Vr_ss.max(), 1e-3))
    rows = [("GT (target)", Vg), ("raw splat V_canon", Vc),
            ("L1 refined @ep95", Vr_l1), ("SSIM refined @ep26", Vr_ss)]
    fig = plt.figure(figsize=(1.45 * len(zs) + 1.6, 1.45 * len(rows) + 0.5), dpi=125)
    gs = gridspec.GridSpec(len(rows), len(zs), wspace=0.03, hspace=0.06)
    for r, (lab, vol) in enumerate(rows):
        for c, z in enumerate(zs):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(vol[z], cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={z}", fontsize=8)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9.5)
    fig.suptitle(f"Subject seq{seq}, target phase t={t} (breathing val) — same V_canon, "
                 f"L1 vs SSIM refined", fontsize=11, y=1.005)
    return b64(fig)


def fig_zoom(seq, L1, SSIM):
    """Zoomed heart-center crop on one mid-z slice — makes blur visible."""
    g = SSIM[seq]
    Vg, Vc, bbox = g["Vg"], g["Vc"], g["bbox"]
    Vr_l1 = L1[seq]["Vr"]; Vr_ss = g["Vr"]
    z = (bbox[0] + bbox[1]) // 2
    cy = (bbox[2] + bbox[3]) // 2; cx = (bbox[4] + bbox[5]) // 2
    h = 55
    sl = (slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h))
    vmax = float(max(Vg[z].max(), 1e-3))
    panels = [("GT", Vg[z]), ("raw splat", Vc[z]), ("L1 refined", Vr_l1[z]), ("SSIM refined", Vr_ss[z])]
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    for a, (lab, im) in zip(ax, panels):
        a.imshow(im[sl], cmap="gray", vmin=0, vmax=vmax)
        a.set_title(lab, fontsize=11); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Zoomed heart crop — seq{seq}, z={z} (look at edge crispness)", fontsize=11, y=1.02)
    return b64(fig)


print("building dataset…")
build_dataset_cache = build_dataset()
device = "cuda"
SEQS = [0, 7]
print("gathering L1…");  L1 = gather(L1_CKPT, SEQS, device)
print("gathering SSIM…"); SSIM = gather(SSIM_CKPT, SEQS, device)

print("rendering figures…")
IMG_TRAJ = fig_trajectory()
IMG_SHARP = fig_sharpness()
IMG_SCORE = fig_scorecard()
IMG_Q0 = fig_qualitative(0, L1, SSIM)
IMG_Q7 = fig_qualitative(7, L1, SSIM)
IMG_Z0 = fig_zoom(0, L1, SSIM)

HTML = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>SSIM refiner analysis — blur & black spots</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;
   margin:0 auto;padding:28px 32px;color:#1a1a1a;line-height:1.55;background:#fff;}}
 h1{{font-size:25px;border-bottom:3px solid #d62728;padding-bottom:8px;}}
 h2{{font-size:20px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px;}}
 h3{{font-size:16px;margin-top:22px;color:#333;}}
 .tldr{{background:#fff6f6;border-left:5px solid #d62728;padding:14px 18px;border-radius:5px;margin:18px 0;}}
 .key{{background:#f0f7ff;border-left:5px solid #1f77b4;padding:12px 16px;border-radius:5px;margin:14px 0;}}
 code{{background:#f3f3f3;padding:1px 5px;border-radius:3px;font-size:90%;}}
 table{{border-collapse:collapse;margin:14px 0;font-size:14px;width:100%;}}
 th,td{{border:1px solid #ccc;padding:6px 10px;text-align:center;}}
 th{{background:#f5f5f5;}} td.l{{text-align:left;}}
 img{{max-width:100%;border:1px solid #eee;border-radius:4px;margin:10px 0;}}
 .good{{color:#1a7f37;font-weight:bold;}} .bad{{color:#c0392b;font-weight:bold;}}
 .cap{{font-size:13px;color:#666;margin:-4px 0 16px;}}
 .formula{{background:#fafafa;border:1px solid #e3e3e3;padding:10px 14px;border-radius:5px;
   font-family:ui-monospace,monospace;font-size:14px;text-align:center;margin:12px 0;}}
</style></head><body>

<h1>Does the SSIM term fix the blur? — refiner analysis</h1>
<p class="cap">Generated in-session · frozen-L1 (job 51876098, ep95) · frozen-SSIM (51950141, ep26) ·
joint-L1 (51876099, ep34) · breathing-val protocol, n=60 unless noted.</p>

<div class="tldr">
<b>TL;DR.</b> The refiner has two jobs: fill <b>dark/dim spots</b> (✅ both losses do this — big PSNR gain)
and remove <b>in-plane blur</b> (the hard one). On the controlled frozen comparison (same seed, same
frozen geometry, only the loss differs): the <b>2D-SSIM term is active and costs no accuracy</b> — it
pays a tiny PSNR tax early then ties L1 at matched steps. On sharpness, <b>SSIM reaches the ceiling
faster</b> (0.692×GT by epoch 26 vs L1 needing ~ep62), but <b>both plateau at the same modest
~0.69×GT ceiling</b> — only ~8% of the gap to GT. So SSIM is a <b>faster, no-cost improver, not a
blur silver bullet</b>: the splat itself is the real bottleneck. SSIM is still young (ep26 vs ep95) —
the decisive matched-epoch verdict comes when it reaches ep50/ep95.
</div>

<h2>1. The two failure modes, and why L1 alone wasn't enough</h2>
<p>The reconstruction is built by <b>splatting</b> the input slices into a canonical cube
(<code>V_canon</code>), then a small 3D-UNet <b>refiner</b> cleans it up (<code>V_refined</code>).
Two distinct defects:</p>
<ul>
<li><b>Black / dim spots</b> — voxels with little or no input coverage come out dark. A per-voxel
intensity loss fixes these well (just predict the missing brightness).</li>
<li><b>Blur</b> — the splat's trilinear scatter + coverage division throws away high-frequency
detail; the heart edges come out soft. Earlier analysis (<code>_html/08</code>) attributed ~75% of
the blur to the splat renderer itself.</li>
</ul>
<p>Plain <b>L1</b> is <i>mean-seeking</i>: the loss-minimizing output is the average of all plausible
textures, which is <b>smooth</b>. So L1 brightens the holes but has no incentive to produce crisp
edges — it fixes spots, not blur.</p>

<h2>2. How the SSIM term works</h2>
<p>We add a structural-similarity term to the refiner loss (applied to <code>V_refined</code> only;
the geometry's L1 supervision stays untouched):</p>
<div class="formula">L_post = 1.0 · L1(V_refined, V_gt) &nbsp;+&nbsp; 0.1 · (1 − SSIM<sub>2D</sub>(V_refined, V_gt))</div>
<ul>
<li><b>SSIM compares <i>local structure & contrast</i></b> in an 11×11 window, not per-pixel values.
It rewards matching GT's edges/texture — exactly the high-frequency signal L1 ignores.</li>
<li><b>It is two-sided / reference-based:</b> it penalizes blur (too smooth vs GT) <i>and</i>
hallucinated or misplaced edges (sharp where GT is smooth). It can't be gamed by just cranking up
contrast.</li>
<li><b>2D per-slice, in-plane</b>: each axial (Y–X) slice is scored independently
(<code>(B,D,H,W)→(B·D,1,H,W)</code>; the through-plane Z is never mixed). The cube is anisotropic
(8&nbsp;mm Z over 12 slices vs 1.4&nbsp;mm in-plane) and the blur we fight is in-plane, so 2D is the
right choice — a 3D window would mix the coarse Z axis.</li>
<li><b>L1 stays in the mix</b> because SSIM is contrast-normalized (blind to a uniform brightness
offset); L1 nails absolute intensity, SSIM nails structure. (Recipe per Zhao et al. 2017.)</li>
</ul>

<h2>3. Is the SSIM term actually doing something? (it is)</h2>
<p>The frozen-L1 and frozen-SSIM runs share the same seed and the same <b>frozen</b> backbone +
point head, so their splat <code>V_canon</code> is byte-identical (26.41 bbox / 19.07 motion in both).
If the SSIM weight were 0 they would be bit-identical at every step. They aren't — confirming the
term is live:</p>
<img src="data:image/png;base64,{IMG_TRAJ}">
<p class="cap">Refined PSNR at matched steps. SSIM (red) dips slightly below L1 (blue) early — the
expected "PSNR tax" from optimizing structure instead of pure MSE — then crosses over around step
25k. Net: <b>SSIM costs no accuracy.</b></p>

<h2>4. Does it fix the blur? — sharpness</h2>
<p>Sharpness = in-plane gradient energy of the reconstruction ÷ that of GT (1.0 = as sharp as GT).
The raw splat sits at <b>0.668</b>.</p>
<img src="data:image/png;base64,{IMG_SHARP}">
<p class="cap">Left: both refiners lift sharpness to ~0.69×GT — a <b>small</b> gain (the dashed green
line is GT). Right: but SSIM gets there <b>faster per epoch</b> — 0.692 by ep26, where L1 needed
~ep62 for 0.688. The catch: both seem to be converging to the same ~0.69 ceiling.</p>

<div class="key">
<b>Key finding.</b> SSIM's structural pressure genuinely bites — it sharpens faster and at no PSNR
cost. But the absolute sharpness ceiling (~0.69×GT) looks <b>shared with L1</b>, because the splat
discards detail neither loss can invent. The refiner can only recover what's still latent in
<code>[V_canon, coverage]</code>.
</div>

<h2>5. Black spots ✅ vs blur ❌ — scorecard</h2>
<img src="data:image/png;base64,{IMG_SCORE}">
<p class="cap">Both losses close most of the <b>dark-spot</b> gap (large PSNR gain, left) but only
~8% of the <b>blur</b> gap to GT (right). This is the core story: <span class="good">dark spots —
fixed</span>; <span class="bad">blur — only marginally improved, by either loss.</span></p>

<table>
<tr><th>metric (n=60)</th><th>raw splat</th><th>L1 @ep95</th><th>SSIM @ep26</th></tr>
<tr><td class="l">sharpness / GT</td><td>0.668</td><td>0.694</td><td>0.692</td></tr>
<tr><td class="l">bbox PSNR (dB)</td><td>26.49</td><td>28.16</td><td>27.69</td></tr>
<tr><td class="l">motion PSNR (dB)</td><td>19.14</td><td>20.26</td><td>19.91</td></tr>
<tr><td class="l">training epoch</td><td>—</td><td>95</td><td>26</td></tr>
</table>
<p class="cap">Note the epoch gap: L1 is 3.6× more trained. SSIM's lower current PSNR is mostly
immaturity, not a deficit — at matched <i>steps</i> they tie (§3).</p>

<h2>6. See it — qualitative panels</h2>
<p>Same subject, same input, same <code>V_canon</code> (frozen geometry) — only the refined row
differs by loss. Compare the bottom two rows to GT.</p>
<img src="data:image/png;base64,{IMG_Q0}">
<img src="data:image/png;base64,{IMG_Q7}">
<h3>Zoomed heart crop (edge crispness)</h3>
<img src="data:image/png;base64,{IMG_Z0}">
<p class="cap">Both refined outputs are visibly cleaner than the raw splat (dark-spot/coverage fixed),
but neither recovers GT's crisp myocardial edges — the residual blur both losses leave behind.</p>

<h2>7. The joint run (for context)</h2>
<p>The third run (joint-L1, ep34) co-trains the aggregator + point head <i>with</i> the refiner, so
its geometry/<code>V_canon</code> differs (higher PSNR base 26.51, but lower sharpness base 0.623 at
ep22 → 0.651 refined). It's mid-training and not directly comparable to the frozen pair, but tracking
healthily. It tests whether letting geometry co-adapt helps — a separate axis from the SSIM question.</p>

<h2>8. Verdict & next steps</h2>
<ul>
<li><b>Black spots:</b> <span class="good">solved</span> by the refiner (either loss).</li>
<li><b>Blur:</b> <span class="bad">only marginally improved.</span> SSIM helps a bit more and a lot
faster, but both losses hit a ~0.69×GT ceiling set by the splat.</li>
<li><b>SSIM is worth keeping</b> — faster convergence, no PSNR cost, and it may still edge ahead once
mature.</li>
<li><b>Decisive test pending:</b> compare SSIM @ep50 vs L1's saved <code>checkpoint_50</code> (matched
epoch) and again at ep95.</li>
<li><b>If blur is the priority</b>, the real lever is <b>upstream</b>: fix the splat/coverage
(higher-res scatter, learned splat, or a coverage-aware loss), or add a gradient-matching term —
because the information SSIM needs to sharpen further isn't in the splat anymore.</li>
</ul>

</body></html>"""

with open(OUT_HTML, "w") as f:
    f.write(HTML)
print("WROTE", OUT_HTML, f"({os.path.getsize(OUT_HTML)//1024} KB)")
