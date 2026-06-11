"""Build the self-contained respiratory-motion example report.

Embeds the PNGs from result/respiratory_examples/ as base64 (no external assets)
and writes _html/06_respiratory_motion_simulation_examples.html.

Run: micromamba run -n svr python _html/build_respiratory_report.py
"""
import base64
import os

RES = "/home/minsukc/vggt/result/respiratory_examples"
OUT = "/home/minsukc/vggt/_html/06_respiratory_motion_simulation_examples.html"


def b64(name):
    with open(os.path.join(RES, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


imgs = {k: b64(f"{k}.png") for k in
        ["lujan_curve", "reslice_sweep", "axial_sweep", "zmontage", "input_view", "combined"]}

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>VGGT-MRI: respiratory-motion simulation — examples</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1000px;margin:0 auto;padding:32px 24px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:26px;margin-bottom:4px}} h2{{margin-top:34px;border-bottom:2px solid #eee;padding-bottom:6px;font-size:20px}}
.sub{{color:#666;margin-top:0}}
code,.mono{{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:#f3f4f6;padding:1px 5px;border-radius:4px}}
img{{max-width:100%;border:1px solid #e2e2e2;border-radius:6px;margin:10px 0}}
.callout{{background:#f0f6ff;border-left:4px solid #1f77b4;padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0}}
.callout.key{{background:#eafaf0;border-color:#0a7d28}}
.note{{font-size:13px;color:#666}}
ul{{margin-top:6px}} li{{margin:5px 0}}
</style></head><body>

<h1>Respiratory-motion simulation — examples</h1>
<p class="sub">Cardiac 4D slice-to-volume on CMRxRecon2024. Visualization of the
<span class="mono">training/data/respiratory.py</span> augmentation (no model). Generated 2026-06-11.</p>

<div class="callout key"><b>Model.</b> A rigid translation per input slice, parameterized by ONE scalar
displacement <span class="mono">d</span> (mm): SI (superior-inferior, through-plane along the depth
axis <b>D</b>) plus a smaller AP (anterior-posterior, in-plane). <b>No LR, no rotation, no scaling</b>
(breathing translates the heart; it doesn't resize it — size is the cardiac clock). Magnitude follows
the Lujan waveform <span class="mono">d(r) = A·sin²ⁿ(πr)</span> with end-expiration as the rest
position (<span class="mono">r=0 → d=0</span>); <b>A ≈ 10–15 mm SI</b>, <b>AP ≈ 0.35·SI</b>.</div>

<div class="callout"><b>Deform-then-reslice.</b> To extract an input slice, the 3D volume is shifted by
<span class="mono">d(r)</span> and re-sampled at its <b>fixed canonical plane</b> — so the slice images
<i>different anatomy</i> as the heart moves (the physical through-plane content change). Geometry:
<b>D = SI</b> at <b>8 mm</b> spacing (so a 10–15 mm shift crosses &gt;1 slice), <b>H/W in-plane</b> at
<b>1.4 mm</b>. The reconstruction target and <span class="mono">scanner_coords</span> stay at the
unshifted end-expiration reference — the model learns to <b>correct</b> breathing (blind to
<span class="mono">r</span>; no embedder this round). Each input slice draws an <i>independent</i>
respiratory phase (decoupled from the cardiac phase — the scattered-acquisition regime).</div>

<h2>1. Respiratory waveform — and n=2 vs n=3</h2>
<img src="data:image/png;base64,{imgs['lujan_curve']}">
<p class="note">Left: SI displacement vs respiratory phase for n=1/2/3. The <span class="mono">sin²ⁿ</span>
form dwells at end-expiration (flat near r=0/1) and peaks at inspiration (r=0.5); dots mark the sweep
used below. Right: the same over 3 breaths (5 s period) — <b>higher n sits at exhale longer</b>.</p>
<div class="callout"><b>n=2 vs n=3?</b> n is the exhale-dwell knob: n=3 spends visibly more time near
rest (more realistic for shallow free-breathing), n=1 is a plain sinusoid (symmetric). <b>Default
n=2</b> (standard, moderate). It's a config knob (<span class="mono">cos2n</span>); for augmentation
we can also sample n∈{2,3} per breath for diversity. The exact n matters less than the amplitude
distribution.</div>

<h2>2. Coronal &amp; sagittal sweep, with difference vs rest</h2>
<img src="data:image/png;base64,{imgs['reslice_sweep']}">
<p class="note">One fixed cardiac phase (motion shown is purely respiratory). Depth axis D is vertical;
the heart slides along D as the displacement grows. The Δ rows (shifted − rest) make the growing motion
obvious — the d=0 column is blank by construction.
<b>On the coarseness:</b> these views are inherently blocky because D is only <b>12 planes at 8 mm</b>
(through-plane), shown here with nearest-neighbour display (no smoothing) so the real sampling is
honest, not faked. In-plane (1.4 mm) is sharp; depth is the resolution we have.</p>

<h2>3. Same scanner plane, different anatomy (the point)</h2>
<img src="data:image/png;base64,{imgs['axial_sweep']}">
<p class="note">A single fixed plane <span class="mono">z</span> across the sweep, with a Δ-vs-rest row.
Because the heart moves through-plane, the <i>same</i> scanner plane images a <i>different</i>
cross-section at each displacement — exactly what deform-then-reslice produces (and what a fixed-slice
free-breathing acquisition sees). This is the effect the model must learn to undo.</p>

<h2>4. Through-plane montage</h2>
<img src="data:image/png;base64,{imgs['zmontage']}">
<p class="note">Depth planes (columns) × displacement (rows). Reading down a column: as the heart
shifts, a fixed plane index reveals deeper anatomy — the cross-sections "scroll" through the stack.</p>

<h2>5. What the model actually sees</h2>
<img src="data:image/png;base64,{imgs['input_view']}">
<p class="note">The upsampled <b>518² input slice</b> for one slot (fixed t, z) across the sweep — exactly
what gets fed to the network — with a Δ-vs-rest row. Same geometric plane, shifted anatomy.</p>

<h2>6. Combined: beating + breathing</h2>
<img src="data:image/png;base64,{imgs['combined']}">
<p class="note">Secondary view — cardiac phase (rows) × respiratory displacement (columns) at a fixed
plane. Cardiac contraction and respiratory translation are independent (two clocks); in training they
combine, but the core panels above isolate the respiratory component.</p>

<h2>What training will see</h2>
<ul>
<li><b>Input slices</b> are resliced from the (respiratory-shifted) volume — the model's observations
carry breathing motion.</li>
<li><b>Target volume + <span class="mono">scanner_coords</span></b> stay at the unshifted
end-expiration reference — the supervision is the motion-free heart.</li>
<li>Per-slice independent respiratory phase; train-only (val never augments).</li>
</ul>
<p class="note"><b>Status:</b> simulation module + unit tests + this report implemented
(<span class="mono">training/data/respiratory.py</span>, <span class="mono">tests/test_respiratory.py</span>,
<span class="mono">tools/render_respiratory_examples.py</span>). Wiring into the GPU aug pipeline / config /
trainer is deferred (design in <span class="mono">docs/01_respiratory_motion_simulation.md</span> §5).</p>

</body></html>"""

with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({len(html)} bytes)")
