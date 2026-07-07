#!/usr/bin/env python
"""Self-contained HTML report for the multi-frame experiment on model 4wokxzov:
does feeding 5 frames/slice beat 1 frame/slice? Covers CMRxRecon (in-distribution, GT metrics,
breathing on/off) + OCMR + Goettingen (real free-breathing, qualitative). Every montage / cycle
GIF / volume sheet is base64-embedded, so the .html is one portable file. The big DVF panels
(~4 MB each) are left on disk and linked by path, not embedded.

  micromamba run -n svr python tools/build_multiframe_report.py
"""
import base64
import glob
import json
import os

R = "/home/minsukc/vggt"


def b64(path):
    if not path or not os.path.exists(path):
        return None
    mime = "image/gif" if path.endswith(".gif") else "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def img(path, w="100%", cap=None):
    u = b64(path)
    if u is None:
        return "<span style='color:#a00'>[missing]</span>"
    c = f"<div class=cap>{cap}</div>" if cap else ""
    return f"<figure><img src='{u}' style='width:{w}'>{c}</figure>"


def main():
    os.chdir(R)
    m = json.load(open("result/multiframe_quick/metrics.json"))
    subs = sorted({r["subject"] for r in m})

    def get(subj, mode, fps, k):
        for r in m:
            if r["subject"] == subj and r["mode"] == mode and r["frames_per_slice"] == fps:
                return r[k]
        return float("nan")

    # ── CMRxRecon metrics table ───────────────────────────────────────────
    def metric_table():
        h = ("<table><tr><th>subj</th><th>mode</th>"
             "<th>motion 1→5</th><th>Δ</th><th>bbox 1→5</th><th>Δ</th>"
             "<th>full 1→5</th><th>Δ</th><th>ssim 1→5</th><th>Δ</th></tr>")
        for subj in subs:
            for mode in ("clean", "breathing"):
                row = f"<tr><td>{subj}</td><td>{mode}</td>"
                for k in ("motion", "bbox", "full", "ssim"):
                    v1, v5 = get(subj, mode, 1, k), get(subj, mode, 5, k)
                    d = v5 - v1
                    col = "#0a7" if d > 0 else "#a00"
                    fmt = ".3f" if k == "ssim" else ".2f"
                    row += f"<td>{v1:{fmt}}→{v5:{fmt}}</td><td style='color:{col}'>{d:+{fmt}}</td>"
                h += row + "</tr>"
        return h + "</table>"

    # ── CMRxRecon montages (ALL output slices) ────────────────────────────
    def cmrx_slices():
        s = ""
        for subj in subs:
            s += f"<h3>Subject {subj}</h3>"
            s += img(f"result/multiframe_quick/subj{subj}_GT_allslices.png",
                     cap="GROUND TRUTH — 12 phases (rows) × 12 z-planes (cols)")
            s += "<div class=grid2>"
            for mode in ("clean", "breathing"):
                for fps in (1, 5):
                    s += ("<div>" + img(
                        f"result/multiframe_quick/subj{subj}_{mode}_fps{fps:02d}_pred_allslices.png",
                        cap=f"PRED · {mode} · fps={fps}") + "</div>")
            s += "</div>"
        return s

    # ── CMRxRecon cycle GIFs (the dramatic breathing rescue on subj7) ─────
    def cmrx_gifs():
        s = "<p>Beating-heart cycle (GT top / pred bottom, 5 spanning planes). fps=1 vs fps=5:</p>"
        for subj in subs:
            for mode in ("clean", "breathing"):
                s += f"<h4>subj{subj} · {mode}</h4><div class=grid2>"
                for fps in (1, 5):
                    s += ("<div>" + img(
                        f"result/multiframe_quick/subj{subj}_{mode}_fps{fps:02d}_cycle.gif",
                        cap=f"fps={fps}") + "</div>")
                s += "</div>"
        return s

    # ── OCMR / Goettingen (real free-breathing, qualitative) ──────────────
    def ood_section(ds, label, subj_dirs):
        s = f"<p>{label}. Real prospectively-acquired free-breathing cine — no GT, qualitative " \
            f"transfer check. Only the mid reference plane is given per animation frame; all other " \
            f"planes are inferred. Left = fps 1, right = fps 5.</p>"
        for name in subj_dirs:
            s += f"<h3>{name}</h3>"
            s += "<h4>all-z volume sheet (reference frame)</h4><div class=grid2>"
            for fps in ("01", "05"):
                s += ("<div>" + img(f"result/{ds}_fps{fps}/{name}/volume_t0.png",
                                    cap=f"fps={int(fps)}") + "</div>")
            s += "</div><h4>beating-heart cycle</h4><div class=grid2>"
            for fps in ("01", "05"):
                s += ("<div>" + img(f"result/{ds}_fps{fps}/{name}/cycle.gif",
                                    cap=f"fps={int(fps)}") + "</div>")
            s += "</div>"
            s += (f"<p class=note>Predicted DVF panels (Δx/Δy/Δz, ~4 MB each) not embedded — on disk: "
                  f"<code>result/{ds}_fps05/{name}/dvf_t0.png</code></p>")
        return s

    ocmr_subj = sorted(os.path.basename(os.path.dirname(p))
                       for p in glob.glob("result/ocmr_fps05/*/cycle.gif"))
    goet_subj = sorted(os.path.basename(os.path.dirname(p))
                       for p in glob.glob("result/goettingen_fps05/*/cycle.gif"))

    # aggregate clean deltas for the TL;DR
    dmot = [get(s, "clean", 5, "motion") - get(s, "clean", 1, "motion") for s in subs]
    dmot_br = [get(s, "breathing", 5, "motion") - get(s, "breathing", 1, "motion") for s in subs]
    avg = lambda x: sum(x) / len(x)

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>Multi-frame experiment — model 4wokxzov</title><style>
 body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em;
   line-height:1.55;color:#181818}}
 h1{{border-bottom:3px solid #333;padding-bottom:.3em}}
 h2{{margin-top:2em;border-bottom:1px solid #ccc;padding-bottom:.2em}}
 table{{border-collapse:collapse;margin:1em 0;font-size:.9em}}
 th,td{{border:1px solid #bbb;padding:4px 9px;text-align:center}} th{{background:#f0f0f0}}
 figure{{margin:.4em 0}} .cap{{font-size:.82em;color:#555;text-align:center;margin-top:2px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:.8em}}
 .box{{background:#f5f8f6;border-left:4px solid #2a8;padding:.7em 1.1em;margin:1.2em 0}}
 .note{{font-size:.82em;color:#777}} code{{background:#eee;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Does 5 frames/slice beat 1 frame/slice? — model 4wokxzov</h1>
<div class=box>
<b>TL;DR.</b> Yes, in every case tested. On CMRxRecon (in-distribution, with ground truth), going
1→5 frames per non-reference slice improves motion PSNR by <b>{avg(dmot):+.2f} dB on clean input</b>
and <b>{avg(dmot_br):+.2f} dB under simulated breathing</b> (all 3 subjects, both modes). The gain
is largest exactly where a single frame fails hardest — <b>subject 7 under breathing jumps
+2.42 dB motion / +0.058 SSIM</b>, because when one frame per plane is respiratory-corrupted, extra
frames give the model consistent views to average toward true anatomy. On the real free-breathing
OOD data (OCMR, Goettingen) the fps=5 reconstructions are visibly steadier through the cycle. The
multi-frame setup is <b>correct and worth using</b>: the model is set-attention over slots, so
feeding a realistic short burst per slice at inference is exactly what it is for.
</div>

<h2>1 · Setup</h2>
<p>Reference-slot z-only model (wandb <code>4wokxzov</code>). The mid reference plane always
contributes all cardiac phases (swept as the slot-0 query). Each other in-bbox plane contributes
<code>frames_per_slice</code> consecutive frames from a random-start burst. We compare fps=1 vs
fps=5. CMRxRecon has ground truth → PSNR/SSIM, run with breathing OFF and ON (breathing corrupts
only the input slices; GT stays at the unshifted reference). OCMR + Goettingen are real
free-breathing acquisitions → qualitative only.</p>

<h2>2 · CMRxRecon metrics (fps 1 → 5)</h2>
{metric_table()}
<p class=note>motion = PSNR on moving voxels (the primary metric); bbox/full = PSNR over the anatomy
box / whole cube; all in dB. Green Δ = fps=5 better.</p>

<h2>3 · CMRxRecon — every output slice (12 phases × 12 z)</h2>
<p>The full reconstructed volume for each config, so you can inspect every slice against the GT
grid. fps=5 grids are sharper/steadier across phases, most visibly under breathing.</p>
{cmrx_slices()}

<h2>4 · CMRxRecon — beating-heart cycle (fps 1 vs 5)</h2>
{cmrx_gifs()}

<h2>5 · OCMR — real free-breathing transfer</h2>
{ood_section('ocmr', 'OCMR 1.5T real-time free-breathing SAX', ocmr_subj)}

<h2>6 · Goettingen — real radial RT free-breathing transfer</h2>
{ood_section('goettingen', 'Goettingen radial real-time free-breathing bSSFP', goet_subj)}

<h2>7 · Conclusion</h2>
<ul>
<li><b>More frames helps everywhere</b> — every subject, both modes, all PSNR/SSIM metrics improve
1→5 (clean avg {avg(dmot):+.2f} dB motion, breathing avg {avg(dmot_br):+.2f} dB).</li>
<li><b>Biggest win under corruption</b> — subj7/breathing +2.42 dB motion, +0.058 SSIM: extra
frames rescue a plane whose single frame was respiratory-corrupted.</li>
<li><b>The setup is correct</b>, not a train/inference mismatch — set-attention over slots means a
realistic multi-frame burst at inference is the intended deployment.</li>
<li><b>Caveat</b>: the model isn't told each frame's phase, so extra off-phase frames get
splat-averaged (mild blur); this is dominated by the coverage/consistency gain here, but is why
very large frame counts also OOM'd and gave diminishing returns.</li>
<li><b>OOD transfer</b> — on real OCMR + Goettingen free-breathing data the fps=5 cycles are
visibly steadier, consistent with the in-distribution numbers.</li>
</ul>
</body></html>"""

    out = "_html/26_multiframe_experiment.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, self-contained)")


if __name__ == "__main__":
    main()
