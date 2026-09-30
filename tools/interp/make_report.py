"""Build a self-contained HTML report (figures inlined as base64) of the interpretability study.
Output: figs/interp/interp_report.html"""
import base64
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = os.path.join(ROOT, "figs/interp")


def img(name, alt, width="100%"):
    with open(os.path.join(F, name), "rb") as fh:
        b = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{alt}" style="width:{width}">'


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>How CardioStitch Works</title>
<style>
:root {{ --bg:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --muted:#52514e; --line:#e4e2dc;
        --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --warn:#fff6e8; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --card:#232322;
        --ink:#f4f4f2; --muted:#c3c2b7; --line:#3a3a38; --warn:#3a3020; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --card:#232322; --ink:#f4f4f2; --muted:#c3c2b7; --line:#3a3a38; --warn:#3a3020; }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
        margin:0; padding:0 16px; }}
main {{ max-width:880px; margin:0 auto; padding:32px 0 64px; }}
h1 {{ font-size:28px; margin:0 0 6px; }} h2 {{ font-size:21px; margin:40px 0 10px; border-top:1px solid var(--line); padding-top:24px; }}
h3 {{ font-size:17px; margin:22px 0 6px; }}
.sub {{ color:var(--muted); margin:0 0 24px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:14px 0; }}
.big {{ font-size:19px; line-height:1.5; }}
.fig {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:8px; margin:12px 0 4px; }}
.cap {{ color:var(--muted); font-size:14px; margin:4px 0 16px; }}
table {{ border-collapse:collapse; width:100%; font-size:15px; }}
td, th {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }}
.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.o {{ color:var(--orange); font-weight:600; }} .b {{ color:var(--blue); font-weight:600; }} .a {{ color:var(--aqua); font-weight:600; }}
.warn {{ background:var(--warn); border-radius:10px; padding:12px 16px; }}
ol li, ul li {{ margin:4px 0; }}
details summary {{ cursor:pointer; color:var(--muted); }}
code {{ font-size:14px; }}
</style></head><body><main>

<h1>How CardioStitch actually works</h1>
<p class="sub">Causal interpretability of the final model (<code>final518_diff1000</code>) · 180 test subjects + 2 real free-breathing scans · 2026-09-25</p>

<div class="card big">
<b>In one sentence:</b> the model is given no cardiac-phase label. <span class="o">One early layer copies the heartbeat
from the reference slice</span>, and <span class="a">the middle layers line the slices up against each other to
remove breathing</span>.
</div>

<h2>The picture</h2>
<div class="card">
<p>The model sees ~10 slices. Each slice was taken at a random heartbeat moment and a random breathing
depth. One slice, the <b>reference</b>, defines which heartbeat moment to reconstruct. Slices only
talk to each other through the <b>global attention</b> layers (24 of them). We asked which layers
carry what, by <b>cutting</b> and <b>transplanting</b> specific connections, not by just looking at
attention maps.</p>
<ol>
<li><b>Heartbeat (layer 3 of 24).</b> Every other slice looks at the <i>heart region</i> of the reference
slice, mostly near its own position, and takes on the reference's cardiac state from it. Four
attention heads do this.</li>
<li><b>Breathing (layers 12–17, helped by 6–11).</b> Each slice is compared with many other slices to find
where it really belongs. The reference plays no special role here.</li>
</ol>
</div>

<h2>Main figure (proposed for the paper)</h2>
<div class="fig">{img("fig_paper_main.png", "main figure")}</div>
<p class="cap">
<b>a</b> Give the other slices the reference "view" from a different heart phase in <i>one</i> layer only.
Layer 3 alone moves the whole reconstruction 97.5% of the way to that phase; every other layer does ~nothing.<br>
<b>b</b> Change only the reference's phase and look <i>inside another slice</i>. Its features don't change
before layer 3, then change right on the heart.<br>
<b>c–d</b> Cut the layer-3 reference read: the heartbeat is gone (2%) but breathing correction is untouched.
Cut slice-to-slice attention in layers 12–17: breathing correction is gone (error = naive stack) but most
of the heartbeat stays (85%).<br>
<b>e</b> Same cut on a <b>real</b> free-breathing AF patient scan: the other slices stop beating.
</p>

<h2>The evidence, ranked</h2>
<p>Two independent reviewer-agents (a clinical/MICCAI view and an ML-interpretability view) ranked the
findings by importance × simplicity × intuitiveness. They agreed on this order:</p>
<table>
<tr><th>#</th><th>Story</th><th>Key number</th><th>Where</th></tr>
<tr><td>1</td><td><b>Two jobs, two depths:</b> heartbeat from the reference early (L3), breathing by slice-vs-slice comparison mid (L12–17)</td>
<td class="num">2% vs 0.97 mm · 85% vs 4.77 mm</td><td>main (c,d)</td></tr>
<tr><td>2</td><td><b>Real data:</b> blocking layer 3 flattens the heartbeat of the non-reference slices</td>
<td class="num">97% of swing gone (both scans)</td><td>main (e)</td></tr>
<tr><td>3</td><td><b>One layer is enough:</b> transplanting only layer 3's read sets the heartbeat</td>
<td class="num">97.5% (min 92%/subject)</td><td>main (a)</td></tr>
<tr><td>4</td><td><b>Where it shows up:</b> other slices' features change only from layer 3, only on the heart</td>
<td class="num">43% heart vs 6% background</td><td>main (b)</td></tr>
<tr><td>5</td><td>Breathing correction is <b>relative</b> (a shift shared by all slices is invisible)</td>
<td class="num">—</td><td>Limitations</td></tr>
<tr><td>6</td><td>Details: 4 heads, heart tokens not background, bias toward nearby positions, attention maps</td>
<td class="num">93.8% · 4% vs 102%</td><td>appendix</td></tr>
<tr><td>—</td><td>PCA feature maps (VGGT-Ω style)</td><td class="num">not informative</td><td>dropped</td></tr>
</table>

<h2>Supporting figures (appendix)</h2>
<h3>Which part of the reference, and which heads</h3>
<div class="fig">{img("fig_mechanism.png", "mechanism")}</div>
<p class="cap"><b>b</b> Four of the 16 heads in layer 3 carry it (93.8% together, held-out subjects).
<b>c</b> The effect is strongest near the patched location and fades over ~30–40 mm: a bias, not a
pixel-exact lookup. <b>d,e</b> Hiding only the reference's <i>heart</i> tokens kills the transfer (4%);
hiding its background changes nothing (102%).</p>

<h3>Attention maps agree, but can mislead</h3>
<div class="fig">{img("fig_attention.png", "attention")}</div>
<p class="cap">Only layer 3 concentrates attention on the reference (up to ~7× uniform), and a query on the
LV lands on the same spot in the reference. <b>Caution:</b> heads h0 and h8 look at the reference just as much
but have <i>no</i> causal effect. Attention maps alone would have picked the wrong heads.</p>

<h3>What breathing correction needs</h3>
<div class="fig">{img("fig_breathing.png", "breathing")}</div>
<p class="cap"><b>g</b> Removing slice-to-slice attention in layers 12–17 kills breathing correction; 6–11 matter
too; 18–23 and the reference don't. Neighbouring slices alone are not enough. The model uses the whole stack.
<b>h</b> If every slice is shifted by the same amount, nothing is corrected: the correction is relative.</p>

<h3>In the paper's metric (nnU-Net LV volume, simulated, n=60)</h3>
<div class="fig">{img("fig_lv.png", "LV curves")}</div>
<p class="cap">Blocking layer 3 flattens the LV curve of the non-reference slices (swing 0.11 vs 0.81 of ground
truth; naive stack 0.05). Removing layers 12–17 keeps it (0.76).</p>

<h3>Real free-breathing scans (both subjects)</h3>
<div class="fig">{img("fig_real.png", "real data")}</div>
<p class="cap">LV volume with the reference slice excluded, over 4.5 s of real acquisition. Blocking the layer-3
read removes 97% of the beat in both subjects. Removing layers 12–17 leaves it intact.</p>

<h2>How sure are we?</h2>
<div class="card">
<ul>
<li><b>Causal, not correlational.</b> Every claim comes from cutting or transplanting a connection. Attention maps are supporting evidence only.</li>
<li><b>Consistent per subject.</b> Layer-3 cut: every one of 180 subjects is below 10% (max 7.6%). Layer-3 transplant: min 0.92. Breathing: every subject loses it without layers 12–17.</li>
<li><b>Checked twice.</b> A 4-agent code review found and fixed 8 bugs before final runs. A 3-agent debate then re-computed every number from the raw data and trimmed the wording until all three agreed.</li>
<li><b>Paper metric and real data agree</b> with the simulation.</li>
</ul>
</div>
<div class="warn">
<b>What we can't claim</b>
<ul>
<li>That the model "doesn't estimate phase". We only know it gets <i>no phase label</i>. <i>What</i> exactly is copied (shape vs. a phase code) is untested.</li>
<li>That the separation is perfect. Cutting the breathing layers also costs ~15% of the heartbeat.</li>
<li>Breathing on real data. There is no ground truth, so only the heartbeat pathway was tested on real scans (n=2).</li>
<li>Whether fine-tuning created the layer-3 read or reused it from pretrained VGGT (out of scope).</li>
<li>Breathing correction is relative and one-sided because the simulated breathing used in training only moves one way. This belongs in Limitations.</li>
</ul>
</div>

<h2>Text for the paper</h2>
<div class="card">
<p><i>What does CardioStitch learn?</i> We probed the model with causal interventions on its cross-slice
(global) attention, the only pathway between slices (simulated test set, n = 180). The model receives no phase
label. In global-attention layer 3 (of 24), four heads let every other slice read the reference slice's heart
region, preferentially near its own location. Blocking this read in that single layer removes 98% of the
cardiac-state transfer to the other slices (every subject below 10%). Transplanting only this read from a
reference at another phase moves the reconstruction 97.5% of the way to that phase. The same holds on real
free-breathing scans: blocking the layer-3 read removes 97% of the LV volume swing of the non-reference slices.
Breathing, in contrast, is aligned in the middle layers by comparing slices with each other. Removing
cross-slice attention in layers 12–17 abolishes the through-plane correction while keeping most of the
cardiac-state transfer (85%), and blocking the reference leaves it unchanged. Cardiac and respiratory motion
are thus handled at different depths of the network.</p>
<p><b>Limitations sentence:</b> <i>Because the respiratory correction is relative, a displacement shared by all
slices cannot be recovered; the absolute end-expiratory frame is inherited from the respiratory model used in
training.</i></p>
</div>

<details><summary>How it was done (one paragraph)</summary>
<p>We rebuilt the model's input pipeline so each slice's heart phase, breathing shift and position label could be
set exactly, and hooked all 24 global-attention layers to (i) block chosen slice→slice connections, (ii) swap the
reference's keys/values with those from a run at another phase ("activation patching"), and (iii) record
attention. Cardiac transfer = how much better the reconstruction matches the ground-truth volume at the target
phase than at other phases, measured on non-reference slices only. Breathing = error/slope of predicted vs applied
through-plane shift. Full record: <code>docs/126_what_the_model_does_interpretability.md</code>; code
<code>tools/interp/</code>.</p>
</details>

</main></body></html>
"""

if __name__ == "__main__":
    out = os.path.join(F, "interp_report.html")
    with open(out, "w") as fh:
        fh.write(HTML)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
