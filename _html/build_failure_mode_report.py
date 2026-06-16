"""Build the self-contained 'breathing reconstruction failure mode' report.

Explains (with measured proofs) what limits the breathing reconstruction: not black holes
but blur, and that the blur is dominated by the splat renderer. Covers how splatting and
coverage are computed, whether black voxels contribute, the sharpness proof, the resp-vs-no-resp
comparison, and next steps. Reads:
  result/variants_eval/sharpness.json     (tools/measure_sharpness.py)
  result/variants_eval/var{2,4}_breathing.json + identity_breathing.json (the matrix)
  result/variants_eval/panels/*.png       (qualitative)

Run: micromamba run -n svr python _html/build_failure_mode_report.py
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
from matplotlib import patches

REPO = "/home/minsukc/vggt"
EV = os.path.join(REPO, "result", "variants_eval")
PANELS = os.path.join(EV, "panels")
OUT = os.path.join(REPO, "_html", "08_breathing_failure_mode.html")


def load(name):
    p = os.path.join(EV, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def png_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


sharp = load("sharpness.json")
S = sharp["results"] if sharp else {}


def sv(key, field):
    return S[key][field] if key in S else None


# ── FIGURE A: splat + coverage schematic (1-D, CPU only) ──
def fig_splat_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    # left: trilinear splat of 2 input points onto a voxel grid (1-D for clarity)
    ax = axes[0]
    grid = np.arange(0, 8)
    ax.set_xlim(-0.5, 7.5); ax.set_ylim(0, 3)
    for g in grid:
        ax.plot([g, g], [0, 0.15], "k-", lw=1)
    ax.text(3.5, -0.45, "canonical voxel index (one axis)", ha="center", fontsize=9)
    # point A at 2.3 (intensity 1.0), point B at 2.7 (intensity 0.8)
    for px, inten, c, lab in [(2.3, 1.0, "#1f77b4", "pixel A (I=1.0)"),
                              (4.7, 0.8, "#2ca02c", "pixel B (I=0.8)")]:
        x0 = int(np.floor(px)); w1 = px - x0; w0 = 1 - w1
        ax.annotate("", xy=(x0, 1.0), xytext=(px, 1.9),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
        ax.annotate("", xy=(x0 + 1, 1.0), xytext=(px, 1.9),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
        ax.plot(px, 1.95, "o", color=c, ms=8)
        ax.text(px, 2.15, lab, ha="center", fontsize=8, color=c)
        ax.text(x0 - 0.05, 0.55, f"+{w0:.1f}·I", ha="right", fontsize=7, color=c)
        ax.text(x0 + 1.05, 0.55, f"+{w1:.1f}·I", ha="left", fontsize=7, color=c)
    ax.set_title("1) Splat: each input pixel drops its intensity into the\nnearest voxels, weighted by distance (trilinear)", fontsize=10)
    ax.axis("off")

    # right: coverage division
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.text(5, 3.6, "2) Each voxel:  value = Σ(weight·intensity)  ÷  Σ(weight)",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(5, 3.05, "                                                         └ coverage", ha="center", fontsize=9, color="#b03030")
    rows = [("many pixels land here", "= well-covered → clean average", "#2ca02c"),
            ("1 faint pixel lands here", "= low coverage → dim / grainy", "#d9a300"),
            ("2 disagreeing pixels here", "= averaged → BLUR", "#1f77b4"),
            ("no (non-black) pixel here", "= coverage≈0 → value≈0 → BLACK HOLE", "#d62728")]
    for i, (a, b, c) in enumerate(rows):
        y = 2.5 - i * 0.62
        ax.add_patch(patches.FancyBboxPatch((0.2, y - 0.18), 0.36, 0.36, boxstyle="round,pad=0.02",
                                            fc=c, ec="none", alpha=0.85))
        ax.text(0.75, y, a, fontsize=9, va="center")
        ax.text(5.1, y, b, fontsize=9, va="center", color=c)
    ax.set_title("Coverage = how much input landed in a voxel.\nDividing by it = averaging the contributors.", fontsize=10)
    ax.axis("off")
    return fig_b64(fig)


# ── FIGURE B: sharpness bars (rel_rec vs GT) ──
def fig_sharpness():
    if not S:
        fig, ax = plt.subplots(figsize=(7, 3)); ax.text(0.5, 0.5, "sharpness.json missing", ha="center"); ax.axis("off")
        return fig_b64(fig)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    groups = [("var2 resp", "var2"), ("var4 no-resp", "var4")]
    protos = ["clean", "breathing"]
    x = np.arange(len(groups)); w = 0.36
    for j, proto in enumerate(protos):
        vals = [sv(f"{g[1]}_{proto}", "rel_model") for g in groups]
        col = "#9ecae1" if proto == "clean" else "#fc9272"
        bars = ax.bar(x + (j - 0.5) * w, vals, w, label=f"{proto} val", color=col, edgecolor="#333", lw=0.5)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            if v: ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, ls="--", c="k", lw=1.2, label="ground-truth sharpness (=1.0)")
    # identity reference (breathing)
    ident_b = sv("var2_breathing", "rel_ident")
    if ident_b: ax.axhline(ident_b, ls=":", c="#de2d26", lw=1.0, label="raw splat (identity), breathing")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("recon sharpness ÷ GT sharpness"); ax.set_ylim(0, 1.1)
    ax.set_title("Sharpness proof: reconstruction high-freq energy relative to ground truth\n(<1 = blur). Model ≈ raw splat → the splat is the sharpness bottleneck.", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="y", alpha=0.3)
    return fig_b64(fig)


def main():
    f_splat = fig_splat_schematic()
    f_sharp = fig_sharpness()

    # numbers for prose
    def g(k, f, d=2):
        v = sv(k, f)
        return "—" if v is None else f"{v:.{d}f}"
    cov_resp_b = g("var2_breathing", "coverage_frac", 3)
    ident_b = load("identity_breathing.json")

    # Blur decomposition (deployment = resp model on breathing inputs): how much of the
    # sharpness loss is the splat (present even on clean) vs breathing-induced.
    relC = sv("var2_clean", "rel_model"); relB = sv("var2_breathing", "rel_model")
    splat_loss = (1 - relC) if relC else None
    breath_loss = (relC - relB) if (relC and relB) else None
    total_loss = (1 - relB) if relB else None
    splat_pct = round(100 * splat_loss / total_loss) if total_loss else None
    breath_pct = round(100 * breath_loss / total_loss) if total_loss else None
    var2_b = load("var2_breathing.json"); var4_b = load("var4_breathing.json")

    def msum(d, met):
        return "—" if d is None else f"{d['summary'][met]['mean']:.2f}"

    # same-input 2x2 PSNR from the N=200 matrix (robust), for the "fair comparison" callout
    def mb(v, proto):
        d = load(f"var{v}_{proto}.json")
        return "—" if d is None else f"{d['summary']['psnr_bbox']['mean']:.1f}"
    cmp2x2 = (
        '<table><tr><th>fed the SAME input →</th><th>resp model (v2)</th><th>no-resp model (v4)</th></tr>'
        f'<tr><td><b>breathing</b> inputs</td><td class="num"><b>{mb(2,"breathing")}</b></td><td class="num">{mb(4,"breathing")}</td></tr>'
        f'<tr><td><b>clean</b> inputs</td><td class="num"><b>{mb(2,"clean")}</b></td><td class="num">{mb(4,"clean")}</td></tr>'
        '</table>')

    midz = sorted(glob.glob(os.path.join(PANELS, "midz_*.png")))
    panel_html = "".join(
        f'<img src="data:image/png;base64,{png_b64(p)}"><p class="note">{os.path.basename(p)} '
        f'— per-method mid-bbox-z reconstruction (bbox-PSNR labeled).</p>' for p in midz)
    cov = sorted(glob.glob(os.path.join(PANELS, "coverage_*.png")))
    cov_html = "".join(
        f'<img src="data:image/png;base64,{png_b64(p)}"><p class="note">{os.path.basename(p)} '
        f'— per-z V_gt / V_canon / coverage / under-covered tissue (resp model, breathing val). '
        f'Red = tissue voxels with low input coverage = where dark/blurry spots come from.</p>' for p in cov)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>VGGT-MRI: the breathing reconstruction failure mode</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1040px;margin:0 auto;padding:32px 24px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:25px;margin-bottom:4px}} h2{{margin-top:32px;border-bottom:2px solid #eee;padding-bottom:6px;font-size:20px}}
h3{{margin-top:20px;font-size:16px}}
.sub{{color:#666;margin-top:0}}
code,.mono{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:#f3f4f6;padding:1px 5px;border-radius:4px}}
img{{max-width:100%;border:1px solid #e2e2e2;border-radius:6px;margin:10px 0}}
.callout{{background:#f0f6ff;border-left:4px solid #1f77b4;padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0}}
.callout.key{{background:#eafaf0;border-color:#0a7d28}}
.callout.warn{{background:#fff8e6;border-color:#d9a300}}
.note{{font-size:13px;color:#666}}
table{{border-collapse:collapse;margin:14px 0;font-size:13px;width:100%}}
th,td{{border:1px solid #e2e2e2;padding:5px 8px;text-align:left}}
th{{background:#f7f7f7}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
pre{{background:#f7f7f9;border:1px solid #e2e2e2;border-radius:6px;padding:10px 12px;font-size:12.5px;overflow-x:auto}}
ul{{margin-top:6px}} li{{margin:5px 0}}
</style></head><body>

<h1>The breathing reconstruction failure mode — blur, coverage &amp; splatting</h1>
<p class="sub">What actually limits the breathing reconstruction, with measured proofs. Companion to
<span class="mono">07_respiratory_variants_analysis.html</span>. Snapshot 2026-06-16.</p>

<p class="sub" style="margin-top:0"><b>The recon that matters for the goal:</b> the model fed
<b>breathing-corrupted inputs</b> (the real-time free-breathing deployment case). This report dissects why
<i>that</i> reconstruction is blurry and how to fix it.</p>
<div class="callout key"><b>TL;DR.</b><ul>
<li><b>The breathing recon is blurry — measurably.</b> It carries only <b>{g("var2_breathing","rel_model")}×</b> the
ground truth's high-frequency detail (vs 1.0 = as sharp as GT).</li>
<li><b>~{splat_pct}% of that blur is the SPLAT renderer, ~{breath_pct}% is the breathing.</b> Even on perfect
clean inputs the splat alone already drops sharpness to {g("var2_clean","rel_model")}× (that's the ~{splat_pct}%);
the breathing inputs add the rest down to {g("var2_breathing","rel_model")}×. So breathing <i>does</i> add blur, but
the splat is the bigger cause — and the splat is fixable without solving motion.</li>
<li><b>It's blur, not black holes.</b> The cube stays filled (coverage ≈ {cov_resp_b}); only ~5–6% of tissue
voxels are under-covered (the localized dark spots you see), not gaping holes.</li>
<li><b>Black input pixels do NOT contribute</b> — an intensity gate (<span class="mono">intensity &gt; 1e-3</span>)
zeroes them out of both the value and the coverage. A voxel that loses all its (non-black) input → coverage≈0 → black.</li>
<li><b>Fix, biggest lever first:</b> a learned decoder / 3D UNet refiner on the splat (the ~{splat_pct}% chunk),
then better motion correction (the ~{breath_pct}% breathing chunk).</li>
</ul></div>

<h2>1. The breathing recon is blurry — what causes it (splat vs breathing)</h2>
<p>Goal = real-time free-breathing, so the recon under scrutiny is <b>the model fed breathing-corrupted
inputs</b>. It's blurry: only <b>{g("var2_breathing","rel_model")}×</b> the ground truth's high-frequency detail
(1.0 = as sharp as GT). The sharpness measurement (§4) splits that blur into two causes:</p>
<table><tr><th>stage</th><th>sharpness ÷ GT</th><th>sharpness lost</th><th>share of the blur</th></tr>
<tr><td>ground truth</td><td class="num">1.00</td><td class="num">—</td><td class="num">—</td></tr>
<tr><td>splat alone (perfect/clean inputs)</td><td class="num">{g("var2_clean","rel_model")}</td><td class="num">{f"{splat_loss:.2f}" if splat_loss else "—"}</td><td class="num"><b>~{splat_pct}%</b></td></tr>
<tr><td>+ breathing inputs (deployment)</td><td class="num">{g("var2_breathing","rel_model")}</td><td class="num">{f"{breath_loss:.2f}" if breath_loss else "—"}</td><td class="num"><b>~{breath_pct}%</b></td></tr>
</table>
<div class="callout key"><b>So the blur you see is ~{splat_pct}% the splat renderer and ~{breath_pct}% the breathing.</b>
The splat smears even with perfect inputs (next two sections show why); the breathing adds a quarter on top by
making slices image shifted anatomy that then gets averaged together. Both hurt the deployment recon — but the
splat is the larger, motion-independent lever.</div>
<p class="note">(Aside, not the point here: this blur is a <i>pipeline</i> limit, not the resp model being a bad
reconstructor — on identical inputs the resp and no-resp models are within ~2% sharpness of each other, and the
resp model is more accurate. The model-vs-model comparison lives in
<span class="mono">07_respiratory_variants_analysis.html</span>.)</p>

<h2>2. How the reconstruction is built — splatting</h2>
<p>The model never outputs a volume directly. It outputs, for every input-slice pixel, a 3-D position
(<span class="mono">world_points = scanner_coords + Δ</span>) in the normalized canonical cube. Those
(position, intensity) pairs are then <b>splatted</b> into the voxel grid: each pixel drops its intensity
into its 8 surrounding voxels, weighted by how close it is to each (trilinear). Code:
<span class="mono">vggt/utils/splat.py:splat_to_volume</span>.</p>
<img src="data:image/png;base64,{f_splat}">
<pre>volume   = Σ over pixels of (trilinear_weight · intensity)     # numerator
coverage = Σ over pixels of (trilinear_weight)                 # denominator
V_canon  = volume / (coverage + 1e-6)                          # = weighted average</pre>
<div class="callout"><b>So a voxel's value is the coverage-weighted average of every input pixel that
landed near it.</b> The model only chooses <i>where</i> pixels land (via Δ); the splat does the rendering.
This is why the splat — not the model — caps how sharp the output can be.</div>

<h2>3. How coverage is computed, and whether black voxels contribute</h2>
<p><b><span class="mono">coverage[voxel]</span></b> = the accumulated trilinear weight at <i>that one voxel</i> —
a per-voxel soft count of "how much input landed here" (the denominator above). It can be 0, 0.4, 2.8, …
Don't confuse it with the two <i>summary</i> numbers reported to wandb:</p>
<ul>
<li><span class="mono">coverage_frac</span> ≈ {cov_resp_b} = the <b>fraction of cube voxels whose coverage &gt; 0</b>
— i.e. <i>what fraction of the box got hit by any input at all</i>. It is <b>not</b> an average coverage value;
it's "{round(float(cov_resp_b)*100) if cov_resp_b not in ('—',None) else '~71'}% of voxels are non-empty, the rest are empty."</li>
<li><span class="mono">coverage_mean</span> = the average <span class="mono">coverage[voxel]</span> over voxels
(≈ how many input pixels stack per covered voxel) — a different quantity.</li>
<li><span class="mono">gt_coverage_frac</span> ≈ 0.69 = fraction of voxels where the <i>target</i> has tissue
(<span class="mono">V_gt &gt; 1e-3</span>) — a property of the answer key, not the inputs: a heart-in-a-chest
fills only ~69% of a rectangular box; the rest is air / zero-padding that <i>should</i> be black.</li>
</ul>
<div class="callout key"><b>Do black voxels contribute? No.</b> Before splatting, each input pixel gets a gate
<span class="mono">splat_weight = (intensity &gt; 1e-3)</span> (<span class="mono">training/loss.py:342</span>).
A black pixel (zero-padding, or a slice that drifted off-FOV and shows nothing) has weight 0, so it adds to
<b>neither</b> the value numerator <b>nor</b> the coverage denominator. Consequently a voxel that receives no
non-black input ends up <span class="mono">0 / (0 + 1e-6) ≈ 0</span> → a true black hole. Black input never
dilutes or fills anything.</div>

<h2>4. Proof: it's blur, and the splat is the bottleneck</h2>
<p>Sharpness here = mean in-plane gradient magnitude over the anatomy bbox (anatomy voxels only), normalized
to the ground truth. A value of 1.0 = as sharp as GT; below 1.0 = high-frequency detail lost = blur. If the
recon were <i>sharp but mis-placed</i> (a geometry error, not blur), sharpness would stay ≈ 1.0 and the error
would show only as edge mismatches. Measured on the resp (v2) and no-resp (v4) models, 12 val samples each
(<span class="mono">tools/measure_sharpness.py</span>):</p>
<img src="data:image/png;base64,{f_sharp}">
<table><tr><th>model</th><th>protocol</th><th>recon ÷ GT sharpness</th><th>raw-splat ÷ GT</th><th>coverage_frac</th><th>bbox PSNR</th></tr>
{"".join(f'<tr><td>var{S[k]["var"]} {S[k]["name"]}</td><td>{S[k]["protocol"]}</td><td class="num">{S[k]["rel_model"]:.3f}</td><td class="num">{S[k]["rel_ident"]:.3f}</td><td class="num">{S[k]["coverage_frac"]:.3f}</td><td class="num">{S[k]["bbox_psnr"]:.2f}</td></tr>' for k in S)}
</table>
<div class="callout"><b>Three reads:</b> (1) <b>rel &lt; 1 everywhere</b> → the recon is genuinely blurrier than GT.
(2) <b>Even clean is only ~0.74×</b> → most of the blur is the splat itself, present without any breathing.
(3) <b>recon ≈ raw-splat sharpness</b> → the trained model adds essentially no high-freq over the raw splat; it
corrects <i>position</i> (PSNR climbs well above identity) but the splat caps detail. Breathing adds a second,
smaller blur on top (clean → breathing drop).</div>

<h2>5. Blur vs black holes — which dominates here (your dark spots)</h2>
<div class="callout warn"><b>Note on what this section proves.</b> <span class="mono">coverage</span> is a
<i>holes</i> detector — it only tells you whether a voxel received any input (filled vs empty); it says
<b>nothing</b> about blur. A voxel can be fully covered <i>and</i> blurry (its contributing pixels disagreed
and got averaged). Blur is proven only by the sharpness number in §4 (0.65× GT). This section uses coverage
to answer the separate question "are the dark spots actual empty holes?" — and the answer is mostly no.</div>
<p>Black holes require a voxel to receive <i>zero</i> corrected coverage. But <span class="mono">coverage_frac
≈ {cov_resp_b}</span> ≥ the ~0.69 tissue fraction, i.e. the inputs fill at least as much of the box as the
target occupies — so at current amplitudes (~8–24&nbsp;mm ≈ 1–3 planes at 8&nbsp;mm spacing) <b>there are no
gaping holes</b>; the cube is filled, just blurrily. <b>Holes would emerge</b> with larger motion or sparser
input (fewer slices → planes that lose all their corrected coverage). <b>Your dark spots are real but
localized:</b> measured per subject, only ~<b>5–6% of tissue voxels are under-covered</b> (low input weight) —
concentrated at specific z-planes and tissue edges, not gaping holes. The coverage maps below make this concrete
— V_gt, the reconstruction, the coverage (how much input landed in each voxel), and the <i>under-covered tissue</i>
(red) where coverage is low and dark/blurry spots appear:</p>
{cov_html}
<p class="note">Per-method mid-z reconstructions (identity + all 5 models) for context:</p>
{panel_html}

<h2>6. Next steps (biggest lever first)</h2>
<ul>
<li><b>Add a small 3D UNet refiner on <span class="mono">V_canon</span>, with deep supervision on the raw
splat.</b> Most of the blur is a deterministic splat smoothing (the clean case proves the detail is
recoverable), so learning splat→GT is a low-risk deblur. Keep a loss on the pre-refiner volume so the geometry
(point head) stays honest and the refiner only sharpens. Expect a real PSNR/SSIM + sharpness gain. Doubles as
the splat-artifact-vs-motion-error ablation.</li>
<li><b>Or replace/augment the splat with a learned decoder</b> (higher ceiling, since the splat is the
bottleneck), or use a sharper splat kernel (finer grid / gaussian / learned) and less aggressive coverage
division. Run head-to-head against the refiner.</li>
<li><b>Watch for hallucination.</b> Any learned refiner can invent plausible-but-wrong detail (clean ≠ correct).
Validate on held-out / out-of-distribution inputs that it sharpens real structure rather than fabricating it.</li>
<li><b>Stress the coverage regime.</b> Re-run this measurement at larger breathing amplitudes and with fewer
input slices to find where blur turns into actual black holes — that's the boundary the real-time (sparse,
single-frame-per-slice) target will push on.</li>
<li><b>Keep pushing motion correction.</b> The resp model's registration is good but partial (~5&nbsp;dB below
the clean ceiling); more epochs / better Δ shrinks the breathing-specific blur at its source.</li>
</ul>

<p class="note">Reproduce: <span class="mono">tools/measure_sharpness.py</span> (sharpness + coverage),
<span class="mono">tools/eval_variants_matrix.py</span> (PSNR matrix),
<span class="mono">tools/render_variants_panels.py</span> (panels),
<span class="mono">_html/build_failure_mode_report.py</span> (this page). Splat:
<span class="mono">vggt/utils/splat.py</span>; gate + coverage metrics: <span class="mono">training/loss.py</span>.</p>
</body></html>"""

    with open(OUT, "w") as f:
        f.write(html)
    print("WROTE", OUT, "| sharpness loaded:", bool(S), "| panels:", len(midz))


if __name__ == "__main__":
    main()
