#!/usr/bin/env python3
"""Build the self-contained fetal_cmr_4d methodology + results report -> _html/27.

Embeds the V1 visualisation assets as base64 data URIs so the page is fully
self-contained (no external files). Diagrams are hand-authored inline SVG.
"""
import base64
import os

RD = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon/Volunteer1"
OUT = "/home/minsukc/vggt/_html/27_fetal_cmr_4d_methodology.html"


def datauri(path, mime):
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


gif = datauri(f"{RD}/vis_sax.gif", "image/gif")
montage = datauri(f"{RD}/vis_sax_montage.png", "image/png")
multislice = datauri(f"{RD}/vis_multislice.png", "image/png")
vs_gt = datauri(f"{RD}/vis_vs_gt.png", "image/png")
heartmask = datauri(f"{RD}/qc_heartmask.png", "image/png")

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fetal_cmr_4d — methodology &amp; MIITT baseline</title>
<style>
  :root{
    --bg:#f7f8fa; --card:#ffffff; --ink:#1a1d24; --muted:#5b6472; --line:#e4e7ec;
    --accent:#2f6fed; --accent2:#e8541e; --good:#1a9e6a; --warn:#d98a00;
    --code:#0f1320; --codeink:#e6e9f0; --shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0e1117; --card:#161a22; --ink:#e6e9f0; --muted:#9aa4b2; --line:#242a35;
    --accent:#6a9bff; --accent2:#ff8a5c; --good:#3fd39a; --warn:#f0b13a;
    --code:#0a0d14; --codeink:#e6e9f0; --shadow:0 1px 3px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }}
  :root[data-theme="dark"]{
    --bg:#0e1117; --card:#161a22; --ink:#e6e9f0; --muted:#9aa4b2; --line:#242a35;
    --accent:#6a9bff; --accent2:#ff8a5c; --good:#3fd39a; --warn:#f0b13a;
    --code:#0a0d14; --codeink:#e6e9f0;
  }
  :root[data-theme="light"]{
    --bg:#f7f8fa; --card:#ffffff; --ink:#1a1d24; --muted:#5b6472; --line:#e4e7ec;
    --accent:#2f6fed; --accent2:#e8541e; --good:#1a9e6a; --warn:#d98a00; --code:#0f1320; --codeink:#e6e9f0;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:920px;margin:0 auto;padding:40px 22px 90px}
  header h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
  header .sub{color:var(--muted);font-size:16px;margin:0 0 4px}
  header .meta{color:var(--muted);font-size:13px;margin-top:10px}
  h2{font-size:22px;margin:44px 0 10px;letter-spacing:-.01em}
  h3{font-size:17px;margin:26px 0 8px}
  p{margin:10px 0}
  a{color:var(--accent);text-decoration:none}
  a:hover{text-decoration:underline}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:20px 22px;margin:16px 0;box-shadow:var(--shadow)}
  .tl{border-left:4px solid var(--accent)}
  .eli5{background:linear-gradient(0deg,var(--card),var(--card));border:1px dashed var(--accent);border-radius:14px;padding:16px 20px;margin:14px 0}
  .eli5 .tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.08em;
    text-transform:uppercase;color:var(--accent);border:1px solid var(--accent);
    border-radius:20px;padding:2px 10px;margin-bottom:6px}
  .muted{color:var(--muted)}
  figure{margin:18px 0;text-align:center}
  figure img{max-width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#000}
  figcaption{color:var(--muted);font-size:13px;margin-top:8px}
  .svgwrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:10px;margin:16px 0}
  table{width:100%;border-collapse:collapse;margin:14px 0;font-size:14.5px}
  th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);vertical-align:top}
  th{color:var(--muted);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:.03em}
  code{background:rgba(127,127,127,.14);padding:1.5px 6px;border-radius:5px;font-size:13.5px;
    font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  .pill{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:20px}
  .pill.ok{color:var(--good);background:rgba(26,158,106,.13)}
  .pill.lim{color:var(--warn);background:rgba(217,138,0,.14)}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  @media(max-width:680px){.grid2{grid-template-columns:1fr}}
  .keyline{border:0;border-top:1px solid var(--line);margin:34px 0}
  .foot{color:var(--muted);font-size:13px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
  .toggle{position:fixed;top:14px;right:14px;background:var(--card);border:1px solid var(--line);
    color:var(--ink);border-radius:20px;padding:6px 12px;font-size:13px;cursor:pointer;box-shadow:var(--shadow)}
</style>
</head>
<body>
<button class="toggle" onclick="var r=document.documentElement;var d=(r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light'));r.setAttribute('data-theme',d==='dark'?'light':'dark')">◐ theme</button>
<div class="wrap">

<header>
  <h1>Turning a free-breathing, ungated heart movie into a clean 4D beating heart</h1>
  <p class="sub">The <code>fetal_cmr_4d</code> baseline — how it works, how I adapted it to our adult MIITT data, and the first result</p>
  <p class="meta">van Amerom et&nbsp;al., <em>Magnetic Resonance in Medicine</em> 2019 &nbsp;·&nbsp; <a href="https://github.com/mriphysics/fetal_cmr_4d">github.com/mriphysics/fetal_cmr_4d</a> &nbsp;·&nbsp; VGGT-MRI baseline roster (docs/31)</p>
</header>

<div class="card tl">
  <strong>TL;DR.</strong> A fetus can't hold its breath or wear an ECG, so you <em>can't</em> take a normal gated heart movie. This method takes a pile of <em>ungated, free-running 2D snapshots</em> and, purely by computer, figures out (1) <b>when</b> in the heartbeat each snapshot was taken and (2) <b>where</b> the heart was, then fuses them into one clean 4D (3D&nbsp;+&nbsp;time) beating heart. It's the <b>only baseline in our roster that recovers the heartbeat timing by itself</b> (no ECG). I adapted it from fetal / multi-camera-angle / Philips-raw data to our <b>adult, single-angle, already-reconstructed MIITT</b> real-time cine. <span class="pill ok">self-gating works</span> (recovered 55&nbsp;bpm) and it produces a <span class="pill ok">coherent beating heart</span> — but <span class="pill lim">blurry</span>, because we only have <em>one</em> camera angle where the method wants several. That blur is exactly the gap our learned method (VGGT) is meant to fill.
</div>

<h2>1 · The problem, in one picture</h2>
<div class="eli5">
  <span class="tag">Explain like I'm 5</span>
  <p>Imagine trying to photograph a <b>hummingbird's wings</b> while the hummingbird is also <b>flying around the room</b>. Each photo is sharp, but every photo catches the wings in a different position <em>and</em> the bird in a different spot. If you want one smooth slow-motion movie of just the wings, you have to sort the photos: "this one is wings-up, this one is wings-down…" and also "the bird was <em>here</em> in this photo, <em>there</em> in that one." Do that for hundreds of photos and you can rebuild the movie.</p>
  <p>The fetal heart is the hummingbird. It beats ~2×/second, and it drifts around because the baby and mother move and breathe. You get lots of sharp 2D snapshots at random beat-moments and random positions. Nobody tells you the beat-moment (no ECG) or the position. The computer has to figure both out.</p>
</div>

<div class="svgwrap">__SVG_PROBLEM__</div>

<h2>2 · The big idea</h2>
<p>Take <b>many</b> 2D real-time snapshots ("frames") covering the heart, then <b>retrospectively</b> — after the scan, in software — solve two puzzles and fuse everything:</p>
<div class="grid2">
  <div class="card"><b>Puzzle 1 — WHEN? (cardiac phase)</b><br><span class="muted">Which moment of the heartbeat is each frame? Recovered from the images themselves, with no ECG. This is <em>self-gating</em>.</span></div>
  <div class="card"><b>Puzzle 2 — WHERE? (motion)</b><br><span class="muted">Where exactly was the heart in each frame? Recovered by <em>registering</em> (aligning) the frames to a common 3D reference.</span></div>
</div>
<p>Once every frame has a known <b>(when, where)</b>, a <b>super-resolution reconstruction</b> stacks them all into a single 4D volume: a 3D heart you can scroll through, that beats.</p>

<h2>3 · How it actually works — the pipeline</h2>
<p>The paper's framework is five stages (A→E). Everything is automatic except drawing a rough box around the heart and chest.</p>
<div class="svgwrap">__SVG_PIPELINE__</div>

<h3>The clever bit: self-gating (stage C)</h3>
<div class="eli5">
  <span class="tag">Explain like I'm 5</span>
  <p>Watch <b>one pixel</b> over a slice's frames — sitting on the heart wall, it gets brighter/darker as the heart beats. That's a wave. Do a <b>Fourier transform</b> (a math prism that splits a wiggle into its rhythms) and the heartbeat shows up as a <b>spike</b> at ~1–3 beats/sec. The spike's location = the heart rate. Once you know the rate, you know the beat-moment of every frame → you can sort them "early-beat" to "late-beat."</p>
  <p>Then, because neighbouring slices overlap in space, you <b>slide each slice's clock</b> until the overlapping regions beat in sync — now "phase&nbsp;0" means end-diastole <em>everywhere</em>.</p>
</div>
<div class="svgwrap">__SVG_GATING__</div>

<h2>4 · The catch — why you need several camera angles</h2>
<div class="eli5">
  <span class="tag">Explain like I'm 5</span>
  <p>A 2D MRI slice is sharp <b>in-plane</b> but thick <b>through-plane</b> (like a fat slice of bread). One stack of parallel slices leaves <b>gaps between the slices</b> that nothing ever photographed. To fill the gaps sharply, the method wants slices from <b>other directions</b> (top-down, side-on, front-on) — like photographing a sculpture from 3 sides instead of 1. The paper prescribes the first three stacks roughly <b>mutually orthogonal</b>.</p>
  <p>With only <b>one</b> direction, there's no extra information to fill the gaps, so the reconstruction can only <b>smoothly interpolate</b> across them → the result is <b>blurry through-plane</b>. This is a property of the <em>data</em>, not the tool.</p>
</div>
<div class="svgwrap">__SVG_ORIENT__</div>
<p class="muted">This is the single most important fact for reading our result below, and it's exactly the gap the VGGT-MRI project targets: classical SVR buys through-plane sharpness with <em>extra acquisition</em> (more angles); VGGT buys it with a <em>learned prior</em> from one angle (docs/31).</p>

<h2>5 · What I did — adapting it to our data</h2>
<p>The method was built for a very specific setup. Our MIITT data is different on every axis, so the adaptation is a set of <b>bridges</b> — all additive; the authors' code is byte-for-byte untouched except one shadowed file with 3 small edits.</p>
<table>
  <tr><th>Dimension</th><th>Paper (fetal_cmr_4d)</th><th>Our MIITT data</th><th>Bridge I built</th></tr>
  <tr><td>Subject</td><td>Fetus (HR 115–180)</td><td>Adult (HR 45–110)</td><td>Retune the heart-rate search band</td></tr>
  <tr><td>Camera angles</td><td>Multiple (≥3 orthogonal)</td><td>One (short-axis only)</td><td>— (accept the limit; it's the point)</td></tr>
  <tr><td>Raw data</td><td>Philips k-space + ReconFrame</td><td>Already-reconstructed NIfTI</td><td>Inject at the post-recon entry; skip ReconFrame</td></tr>
  <tr><td>Heart/chest masks</td><td>Hand-drawn in MITK</td><td>—</td><td>Auto-generate from cardiac-band power</td></tr>
  <tr><td>Recon engine</td><td>Compiled SVRTK, native build</td><td>—</td><td>Run SVRTK via a Singularity container + a <code>mirtk</code> shim</td></tr>
  <tr><td>Setup struct</td><td>Reads Philips <code>PARAM</code></td><td>—</td><td>Rebuild it from MIITT's known timing</td></tr>
</table>
<p>The auto-masking is the neatest bridge: the beating heart oscillates at ~1&nbsp;Hz, so its <b>power in the cardiac frequency band</b> lights it up while slow respiratory body-motion stays dark — a free, automatic heart finder (no manual segmentation).</p>
<figure>
  <img src="__HEARTMASK__" alt="auto heart mask">
  <figcaption>Auto-generated heart ROI (red) per slice, from cardiac-band spectral power — replaces the paper's hand-drawn MITK masks.</figcaption>
</figure>

<h2>6 · The result — Volunteer 1</h2>
<p>End-to-end on one MIITT volunteer's real-time free-breathing short-axis cine (128×128, 13 slices, 180 frames/slice), self-gated and reconstructed into a 4D cine.</p>

<div class="grid2">
  <div class="card"><b>Self-gating</b> <span class="pill ok">works</span><br>Recovered <b>55&nbsp;bpm</b> (median, range 48–60), consistent across all 13 slices — a healthy resting adult rate, not the fetal band. The unique capability transfers.</div>
  <div class="card"><b>Beating</b> <span class="pill ok">yes</span><br>Heart-region temporal contrast <b>0.19</b> vs the gated GT's 0.38 (~50%), with a comparable per-phase signal swing (7.4% vs 8.3%). It genuinely contracts.</div>
</div>

<figure>
  <img src="__GIF__" alt="beating heart recon" style="max-width:340px">
  <figcaption>The reconstructed 4D cine — mid-ventricular short-axis slice, looping through the 25 cardiac phases. Self-gated from ungated free-breathing frames.</figcaption>
</figure>
<figure>
  <img src="__MONTAGE__" alt="25-phase montage">
  <figcaption>Same slice, all 25 phases laid out. The LV blood pool (bright centre) fills and contracts across the cycle.</figcaption>
</figure>
<figure>
  <img src="__VSGT__" alt="recon vs gated GT">
  <figcaption><b>Qualitative</b> comparison — recon (top) vs the volunteer's gated breath-hold cine (bottom). These are <em>separate scans</em> with different FOV and heart position, so pixel metrics (PSNR/SSIM) are meaningless; this is a visual sanity check that the beating anatomy is recovered. Quantitative comparison will be <b>ejection fraction</b> (segmentation-based, alignment-free).</figcaption>
</figure>

<div class="card">
  <b>Honest read.</b> The recon is a coherent, correctly-located, beating heart — but <span class="pill lim">blurry</span>. That is <em>not</em> a bug: MIITT is single-orientation (short-axis only), and single-orientation SVR can only interpolate through-plane (§4). The paper itself notes blurring even with full multi-orientation data, "due to the low spatio-temporal resolution." Our blur is that plus the missing camera angles.
</div>

<h2>7 · Why this matters for the project</h2>
<p>This baseline delivers two things for the write-up:</p>
<ul>
  <li><b>Classical self-gating is achievable</b> — you can recover cardiac timing from free-breathing images with no ECG. (The other baselines can't.)</li>
  <li><b>Even a purpose-built classical 4D pipeline blurs on single-orientation data.</b> That's the precise setting VGGT-MRI targets: recover through-plane structure from <em>one</em> orientation using a learned prior, instead of buying it with extra acquisition.</li>
</ul>

<h2>8 · What's next</h2>
<ul>
  <li><b>Ejection fraction</b> via the nnU-Net M&amp;Ms segmentation model (GPU) — segment the LV in the recon and in the gated GT independently, compare EDV/ESV/EF. Alignment-free, clinically meaningful, and it plugs into the project's EF story (docs 24/25).</li>
  <li><b>Production-quality recon</b> (more iterations, finer resolution) for sharper output.</li>
  <li><b>Scale to all 13 subjects</b> — 10 volunteers + 3 patients, including one with <b>atrial fibrillation</b> (an irregular-rhythm stress test for the self-gating).</li>
</ul>

<hr class="keyline">
<div class="foot">
  <p><b>Code:</b> <code>baselines/fetal_cmr_4d/</code> (export + masking, MATLAB gating adaptation, <code>mirtk</code> shim, visualize). <b>Results:</b> <code>baselines/fetal_cmr_4d/results/</code> → GPFS scratch. <b>Container:</b> <code>fetalsvrtk/svrtk</code> (Singularity). Everything is CPU-only.</p>
  <p><b>Paper:</b> J.F.P. van Amerom et al., "Fetal whole-heart 4D imaging using motion-corrected multi-planar real-time MRI," <em>Magn Reson Med</em> 82:1055–1072, 2019. This is the <em>magnitude</em> 4D reconstruction our pipeline uses (a companion paper covers 4D flow).</p>
</div>

</div>
</body>
</html>"""

# ---- inline SVG diagrams ----
SVG_PROBLEM = r"""<svg viewBox="0 0 880 200" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="inherit">
<style>.t{fill:var(--ink);font-size:13px}.m{fill:var(--muted);font-size:12px}.b{fill:var(--card);stroke:var(--line)}</style>
<text class="t" x="10" y="24" font-weight="700">Hundreds of sharp 2D frames, each at an unknown beat-moment AND unknown position</text>
<g>
<!-- scattered frames -->
<g transform="translate(20,50)"><rect class="b" width="70" height="70" rx="6"/><circle cx="35" cy="35" r="15" fill="var(--accent2)" opacity=".55"/><text class="m" x="6" y="86">t=0.2s</text></g>
<g transform="translate(120,60)"><rect class="b" width="70" height="70" rx="6"/><circle cx="35" cy="35" r="9" fill="var(--accent2)" opacity=".55"/><text class="m" x="6" y="86">t=0.9s</text></g>
<g transform="translate(220,48)"><rect class="b" width="70" height="70" rx="6"/><circle cx="40" cy="30" r="15" fill="var(--accent2)" opacity=".55"/><text class="m" x="6" y="86">t=1.4s</text></g>
<g transform="translate(320,64)"><rect class="b" width="70" height="70" rx="6"/><circle cx="28" cy="42" r="10" fill="var(--accent2)" opacity=".55"/><text class="m" x="6" y="86">t=2.1s</text></g>
<text x="430" y="95" font-size="34" fill="var(--muted)">→</text>
<!-- sorted result -->
<g transform="translate(480,42)">
<rect class="b" width="380" height="96" rx="10"/>
<text class="t" x="16" y="24" font-weight="700">Computer sorts them: WHEN + WHERE</text>
<circle cx="60" cy="62" r="18" fill="var(--good)" opacity=".7"/><circle cx="130" cy="62" r="13" fill="var(--good)" opacity=".7"/><circle cx="200" cy="62" r="8" fill="var(--good)" opacity=".7"/><circle cx="270" cy="62" r="13" fill="var(--good)" opacity=".7"/><circle cx="340" cy="62" r="18" fill="var(--good)" opacity=".7"/>
<text class="m" x="40" y="90">ED</text><text class="m" x="185" y="90">ES</text><text class="m" x="322" y="90">ED</text>
</g>
</g>
</svg>"""

SVG_PIPELINE = r"""<svg viewBox="0 0 880 250" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="inherit">
<style>.h{fill:var(--ink);font-size:13px;font-weight:700}.d{fill:var(--muted);font-size:11.5px}.box{fill:var(--card);stroke:var(--line);stroke-width:1.5}.ar{stroke:var(--accent);stroke-width:2;fill:none;marker-end:url(#a)}</style>
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6" fill="var(--accent)"/></marker></defs>
<g>
<g transform="translate(6,30)"><rect class="box" width="158" height="80" rx="9"/><text class="h" x="12" y="22">A · Acquire</text><text class="d" x="12" y="42">Multi-planar real-time</text><text class="d" x="12" y="58">2D bSSFP slices,</text><text class="d" x="12" y="74">many frames/slice</text></g>
<path class="ar" d="M168,70 L184,70"/>
<g transform="translate(186,30)"><rect class="box" width="158" height="80" rx="9"/><text class="h" x="12" y="22">B · Static MC</text><text class="d" x="12" y="42">Temporal-mean images →</text><text class="d" x="12" y="58">stack-stack + slice-vol</text><text class="d" x="12" y="74">registration</text></g>
<path class="ar" d="M348,70 L364,70"/>
<g transform="translate(366,30)"><rect class="box" width="158" height="80" rx="9"/><text class="h" x="12" y="22">C · Cardiac sync</text><text class="d" x="12" y="42">Heart-rate estimation +</text><text class="d" x="12" y="58">slice-slice cycle</text><text class="d" x="12" y="74">alignment (self-gating)</text></g>
<path class="ar" d="M526,70 L542,70"/>
<g transform="translate(544,30)"><rect class="box" width="158" height="80" rx="9"/><text class="h" x="12" y="22">D · Dynamic MC</text><text class="d" x="12" y="42">Frame-volume</text><text class="d" x="12" y="58">registration, interleaved</text><text class="d" x="12" y="74">with 4D recon</text></g>
<path class="ar" d="M704,70 L720,70"/>
<g transform="translate(722,30)"><rect class="box" width="152" height="80" rx="9"/><text class="h" x="12" y="22">E · 4D SR recon</text><text class="d" x="12" y="42">Super-resolution +</text><text class="d" x="12" y="58">outlier rejection →</text><text class="d" x="12" y="74">beating 4D cine</text></g>
<!-- our position -->
<g transform="translate(6,150)">
<rect x="0" y="0" width="868" height="78" rx="10" fill="none" stroke="var(--accent2)" stroke-dasharray="5 4"/>
<text class="h" x="14" y="24" fill="var(--accent2)">Where our MIITT adaptation plugs in</text>
<text class="d" x="14" y="44">A: skipped — we inject already-reconstructed NIfTIs.  Masks: auto (cardiac-band power).  C: patched to read NIfTI + adult HR.</text>
<text class="d" x="14" y="62">B/D/E: authors' SVRTK engine, unmodified, in a container.  Limitation: single orientation → E can only interpolate through-plane.</text>
</g>
</g>
</svg>"""

SVG_GATING = r"""<svg viewBox="0 0 880 210" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="inherit">
<style>.h{fill:var(--ink);font-size:13px;font-weight:700}.d{fill:var(--muted);font-size:11.5px}.ax{stroke:var(--line);stroke-width:1}.w{stroke:var(--accent);stroke-width:2;fill:none}.pk{stroke:var(--accent2);stroke-width:2.5;fill:none}</style>
<!-- pixel wave -->
<text class="h" x="6" y="20">1 · One pixel's brightness over time (it beats)</text>
<line class="ax" x1="20" y1="120" x2="300" y2="120"/><line class="ax" x1="20" y1="50" x2="20" y2="120"/>
<path class="w" d="M20,90 C50,50 70,50 100,90 C130,130 150,130 180,90 C210,50 230,50 260,90 C280,115 290,118 300,110"/>
<text class="d" x="120" y="140">time →</text>
<text x="315" y="90" font-size="26" fill="var(--muted)">→</text>
<text class="d" x="312" y="112">Fourier</text>
<!-- spectrum -->
<text class="h" x="360" y="20">2 · Its rhythm spectrum → spike at the heart rate</text>
<line class="ax" x1="370" y1="120" x2="640" y2="120"/><line class="ax" x1="370" y1="40" x2="370" y2="120"/>
<path class="pk" d="M370,120 L430,116 L470,60 L478,44 L486,60 L540,112 L640,118"/>
<line x1="478" y1="44" x2="478" y2="120" stroke="var(--accent2)" stroke-dasharray="3 3"/>
<text class="d" x="452" y="138">heart rate (55 bpm)</text>
<text class="d" x="380" y="52" fill="var(--accent2)">peak</text>
<text x="655" y="90" font-size="26" fill="var(--muted)">→</text>
<!-- phase bins -->
<text class="h" x="690" y="20">3 · Sort frames</text>
<g transform="translate(690,36)">
<circle cx="24" cy="34" r="20" fill="var(--good)" opacity=".8"/><text class="d" x="14" y="70">ED</text>
<circle cx="86" cy="34" r="10" fill="var(--good)" opacity=".8"/><text class="d" x="78" y="70">ES</text>
<circle cx="148" cy="34" r="20" fill="var(--good)" opacity=".8"/><text class="d" x="138" y="70">ED</text>
</g>
<text class="d" x="690" y="128">→ then slide each slice's clock so slices beat in sync (interslice)</text>
</svg>"""

SVG_ORIENT = r"""<svg viewBox="0 0 880 210" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="inherit">
<style>.h{fill:var(--ink);font-size:13px;font-weight:700}.d{fill:var(--muted);font-size:12px}.sl{fill:var(--accent);opacity:.28}.slh{fill:var(--accent2);opacity:.3}.g{fill:var(--warn);opacity:.25}</style>
<!-- single orientation -->
<text class="h" x="6" y="20">One orientation (our MIITT) — gaps between thick slices</text>
<g transform="translate(30,36)">
<rect class="sl" x="0" y="0" width="150" height="16" rx="3"/><rect class="g" x="0" y="16" width="150" height="14"/>
<rect class="sl" x="0" y="30" width="150" height="16" rx="3"/><rect class="g" x="0" y="46" width="150" height="14"/>
<rect class="sl" x="0" y="60" width="150" height="16" rx="3"/><rect class="g" x="0" y="76" width="150" height="14"/>
<rect class="sl" x="0" y="90" width="150" height="16" rx="3"/>
<text class="d" x="168" y="20">← sharp slices</text>
<text class="d" x="168" y="58" fill="var(--warn)">← gaps nothing sampled</text>
<text class="d" x="0" y="128" fill="var(--warn)">⇒ can only interpolate the gaps → blurry through-plane</text>
</g>
<!-- multi orientation -->
<text class="h" x="470" y="20">Three orientations (paper) — coverage from every side</text>
<g transform="translate(500,36)">
<rect class="sl" x="0" y="0" width="150" height="16" rx="3"/><rect class="sl" x="0" y="30" width="150" height="16" rx="3"/><rect class="sl" x="0" y="60" width="150" height="16" rx="3"/><rect class="sl" x="0" y="90" width="150" height="16" rx="3"/>
<rect class="slh" x="10" y="-6" width="16" height="118" rx="3"/><rect class="slh" x="46" y="-6" width="16" height="118" rx="3"/><rect class="slh" x="82" y="-6" width="16" height="118" rx="3"/><rect class="slh" x="118" y="-6" width="16" height="118" rx="3"/>
<text class="d" x="0" y="128" fill="var(--good)">⇒ orthogonal slices fill the gaps → sharp 3D</text>
</g>
</svg>"""

html = (HTML
        .replace("__SVG_PROBLEM__", SVG_PROBLEM)
        .replace("__SVG_PIPELINE__", SVG_PIPELINE)
        .replace("__SVG_GATING__", SVG_GATING)
        .replace("__SVG_ORIENT__", SVG_ORIENT)
        .replace("__HEARTMASK__", heartmask)
        .replace("__GIF__", gif)
        .replace("__MONTAGE__", montage)
        .replace("__MULTISLICE__", multislice)
        .replace("__VSGT__", vs_gt))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(html)
print("wrote", OUT, f"({len(html)//1024} KB)")
