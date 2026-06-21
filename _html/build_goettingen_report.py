#!/usr/bin/env python
"""Build the self-contained HTML report for the Göttingen radial RT recon + model inference."""
import base64, os

OUT = "/home/minsukc/vggt/_html/17_goettingen_radial_recon_inference.html"
B = "/home/minsukc/vggt/scratch/data/goettingen"


def img(path, mime=None):
    if not os.path.exists(path):
        return f"<p style='color:red'>[missing: {os.path.basename(path)}]</p>"
    mime = mime or ("image/gif" if path.endswith(".gif") else "image/png")
    data = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:{mime};base64,{data}" style="max-width:100%;border:1px solid #ccc;border-radius:4px"/>'


S = """<!doctype html><html><head><meta charset=utf-8>
<title>Göttingen radial RT — recon + model inference</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1080px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.55}
h1{font-size:25px;border-bottom:3px solid #2c3e50;padding-bottom:8px}
h2{font-size:20px;color:#2c3e50;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:16px;color:#34495e;margin-top:22px}
.tldr{background:#eef5fb;border-left:5px solid #2980b9;padding:14px 18px;border-radius:5px;margin:16px 0}
.warn{background:#fdf3e7;border-left:5px solid #e67e22;padding:12px 16px;border-radius:5px;margin:14px 0}
.good{background:#eafaf1;border-left:5px solid #27ae60;padding:12px 16px;border-radius:5px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#f4f6f7}
code{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:13px}
.fig{margin:18px 0;text-align:center}.cap{font-size:13px;color:#555;margin-top:6px}
.grid{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}.grid>div{flex:1;min-width:300px}
small{color:#666}
</style></head><body>

<h1>Göttingen radial real-time free-breathing bSSFP — reconstruction + trained-model inference</h1>
<p><small>VGGT-MRI project · 2026-06-21 · data <code>scratch/data/goettingen/</code> · tools <code>tools/goettingen_recon/</code> · companion doc <code>docs/16</code></small></p>

<div class=tldr>
<b>TL;DR.</b> We added a second real-time free-breathing short-axis (SAX) eval source — the
<b>Göttingen radial RT bSSFP cine</b> (Blumenthal/Uecker, NLINV-Net, MRM 2024) — as a <b>radial</b>
complement to OCMR. <b>(1)</b> Downloaded all <b>68 volumes</b> (209 GB raw k-space) and reconstructed
each with <b>classical RT-NLINV</b> (the authors' own BART pipeline; <b>no neural net</b>, to avoid a
learned-prior confound). <b>(2)</b> Measured respiratory motion directly (the authors don't report it):
across all 68 subjects the heart's in-plane (≈AP) drift is <b>gentle, median 1.16 mm</b> — real but
shallow (the big SI motion is through-plane for SAX). <b>(3)</b> Ran the trained <b>joint-L1 model</b>
(z-only inputs + target_t query, refiner, resp-trained) on the real RT data: it produces
<b>anatomically coherent beating-heart reconstructions</b>, but the cardiac <b>motion across target_t
is weak</b> (localized to the heart, small magnitude) — a clear <b>domain-gap</b> signal on out-of-distribution
real data. The pipeline works end-to-end; quality is reconstruction-limited as expected.
</div>

<h2>1 · The dataset</h2>
<p>Radial, real-time, free-breathing, bSSFP cine (Siemens Skyra 3T). Each SAX slice is an
<b>independent ~4.26 s acquisition</b> (127 frames, ~33 ms/frame, 13 spokes/frame ≈ 19× undersampled),
so slices share <b>no cardiac or respiratory clock</b> — the genuine "scattered acquisition" the project
simulates. 5 Zenodo parts; 68 volumes; 20–27 slices each; 160×160 / 1.6 mm / 6 mm.</p>
<table><tr><th>Property</th><th>Value</th></tr>
<tr><td>Volumes / slices / frames</td><td>68 vols · 1,630 SAX slices · 207,010 2D frames</td></tr>
<tr><td>Acquisition</td><td>radial bSSFP, real-time, free-breathing, ungated; TR/TE/flip 2.58/1.29 ms/23°</td></tr>
<tr><td>Geometry</td><td>FOV 256 mm, matrix 160² (1.6 mm), 6 mm slice; through-plane gap undocumented (assumed contiguous)</td></tr>
<tr><td>Raw size</td><td>209 GB BART .cfl/.hdr radial k-space (CC-BY-4.0)</td></tr></table>

<h2>2 · Reconstruction — classical RT-NLINV (no neural net)</h2>
<p>We use the authors' BART pipeline <code>nlinv-net/10_reco_rt.sh -R -S13 -V</code>: 5-turn radial
trajectory + gradient-delay correction + ROVir coil compression → <code>bart nlinv --real-time</code>
(iteratively-regularized Gauss–Newton, joint image+coil, <b>temporal ℓ2 regularization</b>) → temporal
median filter → magnitude, center-crop 240→160 → NIfTI <code>(160,160,Z,127)</code>. ~31 min/volume on
an A40. We <b>skip nlmeans</b> (cosmetic denoise, 1.37% effect) and <b>NLINV-Net</b> (the trained recon —
a learned prior would confound a downstream model eval). BART was built from source with CUDA.</p>
<div class=grid>
<div class=fig>{recon_stack}<div class=cap>vol0001: all 24 SAX slices at one frame (apex→base), classical RT-NLINV.</div></div>
<div class=fig>{recon_gif}<div class=cap>vol0001 mid-slice cine — a real beating heart with breathing drift (~4 s).</div></div>
</div>

<h2>3 · Respiratory motion — measured across all 68 subjects</h2>
<p>The authors don't quantify respiratory motion, so we measured it on the recon: per-slice subpixel
translation tracking → frequency-split into respiratory (0.1–0.5 Hz) and cardiac (0.7–2.0 Hz) bands;
PCA → ≈AP axis; heart localized by cardiac-band power. Controls: cardiac band recovers the heart rate
(method valid); frame-shuffle collapses the drift (physiology, not noise).</p>
<div class=good>
<b>Result (68 subjects):</b> heart-ROI AP respiratory drift = <b>median 1.16 mm, mean 1.45 mm</b>
(quartiles 0.48 / 0.80 / 1.16 / 1.69, max 4.89). Only 6% of subjects exceed 3 mm. <b>Real but gentle</b>
free-breathing — AP is intrinsically the small respiratory component; the dominant SI motion is
<b>through-plane for SAX</b> (shows as slice-content change, not in-plane displacement; implied ~4 mm).
A heart-ROI refinement <i>overturned</i> an initial guess: the heart's AP is <i>smaller</i> than the
whole-FOV estimate (the chest wall moves more than the heart).
</div>
<div class=grid>
<div class=fig>{breath_all}<div class=cap>Per-subject AP drift + distribution across all 68 — gentle cohort with a small high-breathing tail.</div></div>
<div class=fig>{breath_heart}<div class=cap>vol0001 heart-ROI method: cardiac-power localizer, AP drift, controls.</div></div>
</div>

<h2>4 · Running the trained model on the real RT data</h2>
<p>We ran the <b>joint-L1 model</b> (<code>218349151_mri_refiner_joint</code>): <b>z-only inputs</b>
(<code>use_t_pose_embedding=false</code> — blind to input cardiac phase), <b>target_t query</b>, a
coverage-refiner, trained <b>with respiratory augmentation</b>. Because it's blind to input phase, the
data's unknown frame phases are a non-issue — we feed <b>scattered real Göttingen slices (z only)</b> and
sweep <code>target_t</code> 0..11 to render the cardiac cycle. The Göttingen recon is resampled to the
canonical 256×256×12 cube via the <i>exact training preprocess</i> (so geometry matches training).</p>
<div class=warn>
<b>Why no self-gating / why your t-concern is moot:</b> the model never sees input cardiac phase, so we
do <b>not</b> need to estimate the phase of each input frame. Only the <i>target</i> phase is queried,
and we sweep it. (An earlier draft self-gated the inputs — unnecessary once we read the actual ckpt
config.)
</div>
<div class=grid>
<div class=fig>{infer_gif1}<div class=cap>vol0001: reconstructed mid-slice across target_t (the model's beating heart, from scattered z-only real inputs).</div></div>
<div class=fig>{infer_gif2}<div class=cap>vol0023 (a higher-breathing subject): same.</div></div>
</div>
<div class=fig>{infer_montage}<div class=cap>vol0001: V_refined at 6 target phases × 3 depths — coherent heart anatomy; the LV cavity responds to target_t.</div></div>

<h3>Honest assessment — it runs, but the phase response is weak (domain gap)</h3>
<div class=warn>
The model produces <b>anatomically coherent</b> heart volumes from the real RT data, and what cardiac
motion it renders is correctly <b>localized to the heart</b> (4× higher temporal-std in the central
heart region than the full volume; peak at z≈8). <b>But the magnitude is small</b> — only ~0.1–0.4% of
voxels show meaningful variation across <code>target_t</code>. So on this out-of-distribution real data
the model reconstructs well but <b>does not produce a strong beat</b> as the queried phase sweeps.
This is consistent with the known limitations (<code>docs/13</code>): the geometry/motion is the
bottleneck, and the splat caps sharpness. It is the expected <b>gated-cine → real-RT domain gap</b>.
</div>
<div class=fig>{infer_motion}<div class=cap>Where the model puts cardiac motion (temporal std across target_t): localized at the heart — anatomically valid, just weak.</div></div>

<h2>5 · Evaluation — the decision</h2>
<p>There is <b>no clean 3-D ground-truth volume</b> (slices are independently acquired at uncorrelated
cardiac+respiratory phases), and the model's <code>target_t=k</code> need not equal training's absolute
ED-anchored phase k. So <b>absolute-phase metrics are unreliable</b>. The honest quantitative eval is
<b>leave-one-slice-out self-consistency</b>: reconstruct from S−1 scattered slices, predict the held-out
slice at its own phase, compare to the acquired slice — this only needs <i>internal</i> consistency, so
it survives the t-mismatch. <b>Decision:</b> lead with the qualitative demo here (feasibility + the
domain-gap finding); implement leave-one-out self-consistency next for a number. A gated-reference
(Tier-B) PSNR is possible later via the dataset's self-gating indices, but that reference is itself an
approximate derived recon.</p>

<h2>6 · Limitations &amp; next steps</h2>
<ul>
<li><b>Domain gap is the headline:</b> trained on gated breath-hold CMRxRecon cine, the model
under-responds to <code>target_t</code> on real radial RT data. Closing it likely needs RT-like training
data (contrast/undersampling/real breathing) — the project's headline direction.</li>
<li><b>Reconstruction is splat-blurred</b> (known, <code>docs/10/13</code>) — independent of the domain gap.</li>
<li><b>Gentle breathing cohort:</b> these volunteers breathed shallowly (~1.2 mm AP); a stress test of
large respiratory motion would need deeper-breathing data.</li>
<li><b>Next:</b> (a) leave-one-out self-consistency number; (b) run the full 68; (c) compare against the
no-refiner / no-resp variants to isolate what helps on real data.</li>
</ul>

<h2>7 · Reproduce</h2>
<p>Data + outputs under <code>scratch/data/goettingen/{radial,recon,analysis,inference}/</code> (+ data-side
<code>README.md</code>). Tools in <code>tools/goettingen_recon/</code>: <code>download_goettingen.sh</code>,
<code>recon_all.sbatch</code>, <code>cfl_to_nifti.py</code>, <code>measure_respiratory_motion*.py</code>,
<code>breathing_all_subjects.py</code>, <code>goettingen_to_canonical.py</code>, <code>goettingen_infer.py</code>.
Model: <code>scratch/logs/218349151_mri_refiner_joint</code>. BART build: memory <code>reference_bart_build_greatlakes</code>.</p>

</body></html>"""

html = S
for key, path in [
    ("{recon_stack}", f"{B}/recon/vol0001_vis1/sax_stack.png"),
    ("{recon_gif}", f"{B}/recon/vol0001_vis1/sax_multislice_cine.gif"),
    ("{breath_all}", f"{B}/analysis/breathing_all_subjects.png"),
    ("{breath_heart}", f"{B}/analysis/vol0001_heartroi.png"),
    ("{infer_gif1}", f"{B}/inference/vol0001_vis1_beating.gif"),
    ("{infer_gif2}", f"{B}/inference/vol0023_vis1_beating.gif"),
    ("{infer_montage}", f"{B}/inference/vol0001_vis1_phase_montage.png"),
    ("{infer_motion}", f"{B}/inference/vol0001_motion_map.png"),
]:
    html = html.replace(key, img(path))
open(OUT, "w").write(html)
print(f"wrote {OUT}  ({len(html)/1e6:.1f} MB)")
