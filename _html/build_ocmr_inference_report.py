#!/usr/bin/env python
"""Build _html/13_ocmr_inference_results.html — a self-contained report of running the
trained VGGT-MRI model on REAL OCMR real-time free-breathing SAX cines.

Embeds every GIF / PNG produced by tools/eval_ocmr_inference.py as base64 so the file
opens standalone. Facts come from /tmp/ocmr_report_facts.json (subject geometry) and the
recon meta. Run AFTER tools/eval_ocmr_inference.py has populated result/ocmr_eval/.

Usage:  micromamba run -n svr python _html/build_ocmr_inference_report.py
"""
import base64
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL = os.path.join(ROOT, "result", "ocmr_eval")
FACTS = json.load(open("/tmp/ocmr_report_facts.json"))
OUT = os.path.join(ROOT, "_html", "13_ocmr_inference_results.html")


def b64(path, mime):
    if not os.path.exists(path):
        return None
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()


def gif(subj, draw):
    return b64(os.path.join(EVAL, subj, f"draw{draw}_cycle.gif"), "image/gif")


def png(subj, name):
    return b64(os.path.join(EVAL, subj, name), "image/png")


CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;
  max-width:1080px;margin:0 auto;padding:28px 22px 80px;background:#fafafa}
h1{font-size:26px;margin:0 0 4px} h2{font-size:20px;margin:34px 0 8px;border-bottom:2px solid #e0e0e0;padding-bottom:4px}
h3{font-size:16px;margin:22px 0 6px}
.sub{color:#666;margin:0 0 18px}
.tldr{background:#eef5ff;border-left:4px solid #2f6fdb;padding:14px 18px;border-radius:6px;margin:18px 0}
.caveat{background:#fff6e9;border-left:4px solid #e08a17;padding:12px 16px;border-radius:6px;margin:14px 0}
code{background:#eee;padding:1px 5px;border-radius:4px;font-size:13px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid #ddd;padding:5px 9px;text-align:left} th{background:#f0f0f0}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:16px 18px;margin:20px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.badge{display:inline-block;background:#eef;border-radius:10px;padding:1px 9px;font-size:12px;margin-right:6px;color:#335}
.gifrow{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0}
.gifrow figure{margin:0;text-align:center}
.gifrow img{height:150px;image-rendering:pixelated;border:1px solid #ccc;border-radius:4px;background:#000}
figure{margin:10px 0} figcaption{font-size:12px;color:#666;margin-top:3px}
.panel img{width:100%;border:1px solid #ccc;border-radius:4px;background:#000}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
"""

H = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
     "<title>OCMR real-time inference — VGGT-MRI</title>",
     f"<style>{CSS}</style></head><body>"]
A = H.append

A("<h1>Running the model on real real-time free-breathing data (OCMR)</h1>")
A("<p class='sub'>2026-06-18 · qualitative gated→real-time transfer test · "
  "<span class='mono'>tools/eval_ocmr_inference.py</span> · companion to <span class='mono'>docs/06</span></p>")

A("<div class='tldr'><b>TL;DR.</b> The model trains entirely on <i>simulated</i> sparse sampling "
  "of gated breath-hold cine; its headline risk is whether that transfers to <b>real</b> real-time "
  "free-breathing acquisition. We ran the trained <b>z-only</b> model on <b>11 real OCMR real-time "
  "SAX cines</b> (reconstructed from R≈9 k-space; see <span class='mono'>docs/06</span>). From <b>one "
  "fixed scattered single-frame-per-slice input</b> the model synthesizes a coherent beating heart "
  "across 12 queried cardiac phases — including canonical depth planes that had <i>no</i> input slice. "
  "This is the <b>optimistic</b> read (inputs are k-t-cleaned, not true single-shot) and is "
  "<b>qualitative only</b> (OCMR is prospectively undersampled → no ground-truth volume to score).</div>")

# ── what / how ──
A("<h2>What this is</h2>")
A("<p>OCMR <span class='mono'>us_*</span> series are genuine real-time, non-gated, free-breathing "
  "SAX stacks. We previously reconstructed 11 of them to image-domain cines "
  "(<span class='mono'>scratch/data/ocmr/recon/&lt;subj&gt;/sax_cine.nii.gz</span>, "
  "k-t CS-SENSE, λ=0.02 — full story in <span class='mono'>docs/06</span>). Here we feed those through "
  "the trained model. There is <b>no paired clean 3-D reference</b>, so this is a visual check of "
  "anatomy, depth coherence, and plausible cardiac motion — not a PSNR/SSIM number.</p>")

A("<h2>How the model is run (the adapter)</h2>")
A("<p>A single standalone script, <span class='mono'>tools/eval_ocmr_inference.py</span> — no "
  "training-code changes. Per subject:</p>")
A("<table>"
  "<tr><th>Step</th><th>What</th><th>Why it matches training</th></tr>"
  "<tr><td>1. Pick frames</td><td><b>One random frame per slice</b> (3 independent draws/subject)</td>"
  "<td>= the scattered single-frame-per-slice regime; each slice = a random (cardiac, respiratory) state</td></tr>"
  "<tr><td>2. In-plane</td><td>Resample each slice to <b>1.4 mm</b> + center crop/pad to <b>256×256</b></td>"
  "<td>same canonical cube as <span class='mono'>preprocess.py</span> (Spacingd + ResizeWithPadOrCropd)</td></tr>"
  "<tr><td>3. Depth (z)</td><td>Place each slice on the canonical 8-mm grid using its <b>true ~10 mm "
  "center-to-center spacing</b> (read from <span class='mono'>meta.json</span> positions, NOT the 8 mm "
  "slab thickness in the header)</td>"
  "<td>physically-correct placement; 10&gt;8 mm ⇒ outer slices crop, a few interior planes left empty — "
  "exactly the &lt;12-real-plane case training already handles</td></tr>"
  "<tr><td>4. Intensity</td><td>One per-subject scale from the <b>0.5/99.9 percentiles over the whole "
  "cine's nonzero voxels</b> → clip to [0,1]</td>"
  "<td>matches <span class='mono'>ScaleIntensityByT0PercentilesD</span>; computed over the full cine so "
  "different random draws share one scale (comparable)</td></tr>"
  "<tr><td>5. Model</td><td><b>z-only</b> checkpoint "
  "(<span class='mono'>use_t_pose_embedding=false</span>); query each of 12 <span class='mono'>target_t</span> "
  "phases</td>"
  "<td>input cardiac phase is unknown for OCMR — the z-only model never consumes it, so nothing is faked "
  "(the <span class='mono'>docs/04</span> blind-input stance)</td></tr>"
  "<tr><td>6. Splat</td><td><span class='mono'>world_points</span> → "
  "<span class='mono'>splat_predictions</span> → <b>V_canon</b> (12×256×256)</td>"
  "<td>identical splat the training loss uses; no <span class='mono'>V_gt</span> needed</td></tr>"
  "</table>")

A("<div class='caveat'><b>Two honest caveats (carried from docs/06).</b> "
  "<b>(1) Optimistic inputs.</b> Each input frame is k-t-reconstructed using <i>many</i> frames/slice — "
  "the very temporal information the one-frame-per-slice pitch aims to avoid. A <i>faithful</i> single-shot "
  "test (degraded, no temporal recon) is still unbuilt. <b>(2) No reference.</b> Real-time OCMR is "
  "prospectively undersampled, so no clean volume exists to score against — everything here is qualitative.</div>")

# ── per-subject table ──
A("<h2>Subjects (11)</h2>")
A("<table><tr><th>Subject</th><th>Type</th><th>Slices</th><th>Frames</th><th>In-plane</th>"
  "<th>z-spacing</th><th>Canonical planes (of 12)</th></tr>")
for f in FACTS:
    typ = "volunteer" if f["vol"] == "vol" else "patient"
    A(f"<tr><td class='mono'>{f['subject']}</td><td>{typ}</td><td>{f['slices']}</td>"
      f"<td>{f['frames']}</td><td>{f['inplane']} mm</td><td>{f['spacing']} mm</td>"
      f"<td class='mono'>{f['planes']}: [{f['plane_list']}]</td></tr>")
A("</table>")
A("<p class='sub'>Every stack spans 106–125 mm &gt; the 96 mm canonical z-extent, so each maps to "
  "<b>9–10</b> of the 12 planes (ends cropped, 1–3 interior planes empty). The model fills the empty "
  "planes from neighbours via cross-slice attention + the splat.</p>")

# ── how to read ──
A("<h2>How to read each card</h2>")
A("<ul>"
  "<li><b>Inputs</b> — the 9–10 scattered slices actually fed (label <span class='mono'>z=canonical "
  "plane, s=physical slice, f=frame</span>). Different (cardiac, respiratory) state per slice.</li>"
  "<li><b>V_canon @ ED</b> — the reconstructed volume across all 12 canonical depth planes at target "
  "phase t=0. Empty planes here are the ones with no input slice.</li>"
  "<li><b>Beating-heart GIFs (×3 draws)</b> — mid-depth slice of V_canon swept over the 12 queried "
  "cardiac phases, from one <i>fixed</i> input. The 3 draws use different random frames → shows "
  "sensitivity to <i>which</i> states were sampled.</li></ul>")

# ── cards ──
A("<h2>Per-subject results</h2>")
for f in FACTS:
    s = f["subject"]
    typ = "volunteer" if f["vol"] == "vol" else "patient"
    A("<div class='card'>")
    A(f"<h3 class='mono'>{s}</h3>")
    A(f"<div><span class='badge'>{typ}</span><span class='badge'>{f['slices']} slices · {f['frames']} frames</span>"
      f"<span class='badge'>{f['spacing']} mm spacing</span><span class='badge'>{f['planes']}/12 planes</span></div>")
    inp = png(s, "draw0_inputs.png")
    vol = png(s, "draw0_volume_t0.png")
    if inp:
        A(f"<figure class='panel'><img src='{inp}'><figcaption>Inputs (draw 0): scattered single-frame "
          f"slices fed to the model</figcaption></figure>")
    if vol:
        A(f"<figure class='panel'><img src='{vol}'><figcaption>V_canon at ED across all 12 canonical "
          f"depth planes (empty planes = no input slice)</figcaption></figure>")
    A("<div class='gifrow'>")
    for d in range(3):
        g = gif(s, d)
        if g:
            A(f"<figure><img src='{g}'><figcaption>draw {d}: beating heart (12 phases)</figcaption></figure>")
    A("</div></div>")

A("<h2>Reading of the result</h2>")
A("<ul>"
  "<li><b>Transfer happens.</b> Across the volunteer + all 10 patients the model produces a coherent SAX "
  "heart with a clear LV blood pool and plausible contraction over the queried phases — from real "
  "free-breathing data it never saw in training.</li>"
  "<li><b>Depth in-fill works.</b> Canonical planes with no input slice are reconstructed from neighbours, "
  "the intended behaviour of the slice-to-volume design.</li>"
  "<li><b>Quality tracks input richness.</b> The 128-frame volunteer (us_0084) is cleanest; the 29-frame "
  "patients (us_0197) are grainier but still clearly anatomical.</li>"
  "<li><b>Draw sensitivity is modest.</b> The 3 random draws give similar reconstructions, i.e. the result "
  "isn't hostage to one lucky frame selection.</li></ul>")

A("<div class='caveat'><b>Do not over-read.</b> This is the optimistic, k-t-clean-input, no-reference "
  "path. It demonstrates anatomy/contrast transfer and cross-slice in-fill on real data; it does <b>not</b> "
  "measure true single-shot fidelity or give a metric. Next steps (docs/06 §7): the faithful single-shot "
  "degradation test, and a Tier-B reconstructed reference for quantitative scoring.</div>")

A("<p class='sub'>Generated by <span class='mono'>_html/build_ocmr_inference_report.py</span> from "
  "<span class='mono'>result/ocmr_eval/</span>. Recon provenance: <span class='mono'>docs/06</span>, "
  "<span class='mono'>scratch/data/ocmr/ocmr_recon_ktcs.py</span>.</p>")
A("</body></html>")

open(OUT, "w").write("".join(H))
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
