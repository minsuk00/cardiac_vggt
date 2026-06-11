import base64
def b64(p): return base64.b64encode(open(p,'rb').read()).decode()
imgs={k:b64(f'_html/assets/{k}.png') for k in
      ['motion_curves','motion_mask','learning_curves',
       'cmp_allphases_P055','cmp_allphases_P048','cmp_t0t7_P055','cmp_t0t7_P048',
       'cycle_aggft','nozt_coronal','nozt_sagittal','nozt_perz','nozt_montage']}

# motion(primary), bbox, full, ssim ; t0/t7 per metric
runs=[
 dict(id='warrwlv8',name='t0t7_aggft',color='#1f77b4',state='running',steps=173575,
      phases='{0,7}',zt='on',scope='aggregator+head',
      mot=23.72,motb=20.5, m_t0=23.38,m_t7=24.06,
      bbox=32.66,full=34.03,ssim=0.975, b_t0=32.70,b_t7=32.62),
 dict(id='fc8d065g',name='allphases_aggft',color='#d62728',state='running',steps=173580,
      phases='all 12',zt='on',scope='aggregator+head',
      mot=23.44,motb=20.7, m_t0=23.14,m_t7=24.27,
      bbox=32.63,full=33.95,ssim=0.976, b_t0=32.86,b_t7=32.88),
 dict(id='vry47r4f',name='t0t7_aggft_no_zt',color='#2ca02c',state='running',steps=171510,
      phases='{0,7}',zt='OFF (ablation)',scope='aggregator+head',
      mot=22.92,motb=20.5, m_t0=22.56,m_t7=23.28,
      bbox=32.01,full=30.87,ssim=0.955, b_t0=32.00,b_t7=32.02),
 dict(id='bnwfjav6',name='allphases_headonly',color='#888888',state='finished',steps=200000,
      phases='all 12',zt='on',scope='point_head only',
      mot=21.29,motb=20.7, m_t0=19.70,m_t7=21.94,
      bbox=30.64,full=30.07,ssim=0.957, b_t0=29.61,b_t7=30.45),
]
def d(v,b):
    x=v-b;c='pos' if x>=0 else 'neg';s='+' if x>=0 else ''
    return f'{v:.2f}<span class="delta {c}">{s}{x:.2f}</span>'

# primary motion table
rmot=""
for r in runs:
    rmot+=f"""<tr>
      <td><span class="dot" style="background:{r['color']}"></span><b>{r['name']}</b><br><span class="mono">{r['id']}</span></td>
      <td>{r['phases']}</td><td>{r['scope']}</td><td>{r['zt']}</td>
      <td>{r['steps']:,} <span class="badge {'run' if r['state']=='running' else 'fin'}">{r['state']}</span></td>
      <td class="num bigm">{d(r['mot'],r['motb'])}</td>
      <td class="num">{r['bbox']:.2f}</td><td class="num">{r['full']:.2f}</td><td class="num">{r['ssim']:.3f}</td></tr>"""

# comparable t0/t7 across metrics
rt=""
IDM={'t0':19.65,'t7':21.2}; IDB={'t0':29.80,'t7':30.68}
for r in runs:
    rt+=f"""<tr><td><span class="dot" style="background:{r['color']}"></span>{r['name']}</td>
      <td class="num bigm">{r['m_t0']:.2f}</td><td class="num bigm">{r['m_t7']:.2f}</td>
      <td class="num">{r['b_t0']:.2f}</td><td class="num">{r['b_t7']:.2f}</td></tr>"""

html=f"""<!doctype html><html><head><meta charset="utf-8"><title>VGGT-MRI: 4-run comparison</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;padding:32px 24px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:26px;margin-bottom:4px}} h2{{margin-top:38px;border-bottom:2px solid #eee;padding-bottom:6px}} h3{{margin-top:24px;color:#333}}
.sub{{color:#666;margin-top:0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:14px 0}}
th,td{{border:1px solid #e2e2e2;padding:7px 9px;text-align:left;vertical-align:top}}
th{{background:#f6f7f9;font-weight:600}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.bigm{{background:#fffaf0}}
.mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#666}}
.delta{{font-size:11px;margin-left:6px;padding:1px 4px;border-radius:4px}}
.delta.pos{{color:#0a7d28;background:#e6f6ea}} .delta.neg{{color:#b00020;background:#fdeaea}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}}
.badge{{font-size:10px;padding:1px 6px;border-radius:8px}} .badge.run{{background:#fff3cd;color:#856404}} .badge.fin{{background:#d4edda;color:#155724}}
img{{max-width:100%;border:1px solid #e2e2e2;border-radius:6px;margin:8px 0}}
.callout{{background:#f0f6ff;border-left:4px solid #1f77b4;padding:12px 16px;margin:16px 0;border-radius:0 6px 6px 0}}
.callout.key{{background:#eafaf0;border-color:#0a7d28}} .callout.warn{{background:#fff7e6;border-color:#e0a800}}
ul{{margin-top:6px}} li{{margin:4px 0}} .note{{font-size:12.5px;color:#777}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media(max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
</style></head><body>

<h1>VGGT-MRI — comparison of four multi-phase runs</h1>
<p class="sub">Cardiac 4D slice-to-volume on CMRxRecon2024. All runs warm-start from the 4-day baseline and train the canonical-grid unsupervised intensity pipeline. Generated 2026-06-10.</p>

<div class="callout key"><b>Primary metric = <span style="color:#0a7d28">motion PSNR</span></b> (the <span class="mono">val_motion</span> panel): PSNR over <b>only the dynamic voxels</b> — the ~5–8% of the cube that actually moves across the cardiac cycle (the beating heart). Static tissue is excluded, so this is the honest measure of whether the model corrects motion <i>where it matters</i>. bbox/full PSNR are dominated by stationary anatomy and overstate everything. Identity (Δ=0, no motion correction) motion baseline ≈ 20.5–20.7 dB.</div>

<div class="callout key"><b>Bottom line (on motion PSNR).</b>
<b>Finetuning the aggregator is what corrects motion</b> — the two aggft runs reach +2.7 to +3.2 dB over the no-correction baseline; <b>head-only adds only +0.6 dB and is at/below the baseline at ED</b> (t0), i.e. it barely moves the heart at all. <b>All-12-phases ties the {{0,7}} specialist</b> on t0/t7 (one model covers the whole cycle for free). Dropping the <b>z/t embeddings</b> costs ~0.8 dB of motion PSNR — a real penalty on the dynamic region (larger than it looked on bbox).</div>

<h2>1. Runs &amp; headline numbers</h2>
<table>
<tr><th>Run</th><th>Target phases</th><th>Trained</th><th>z/t embed</th><th>Steps</th>
<th>★ MOTION PSNR<br><span class="note">Δ vs ~20.6 baseline</span></th>
<th>bbox PSNR</th><th>full PSNR</th><th>SSIM</th></tr>
{rmot}
</table>
<p class="note">All dB. ★ motion = primary. Means for the two t0/t7 runs average only phases 0 &amp; 7; the all-phases runs average all 12 — so the <b>mean</b> column is not comparable across the phase split. See §3 for the apples-to-apples t0/t7 view. (bbox/full/SSIM included for continuity but are static-tissue-dominated.)</p>

<h2>2. Motion learning curves (the metric that matters)</h2>
<img src="data:image/png;base64,{imgs['motion_curves']}">
<ul>
<li><b>At ED (t0, left):</b> <span style="color:#888"><b>head-only (gray)</b></span> sits <b>on the identity baseline</b> (~19.7) for all of training — it does essentially zero motion correction at ED. The three aggregator-finetunes climb to ~23 dB.</li>
<li><b>At ~ES (t7, right):</b> same separation; head-only reaches ~22 (modest), aggft variants ~24, <span style="color:#2ca02c"><b>no_zt</b></span> trails the aggft pair by ~0.8 dB.</li>
<li><span style="color:#1f77b4"><b>t0t7_aggft</b></span> and <span style="color:#d62728"><b>allphases_aggft</b></span> are interleaved throughout — phase coverage is free.</li>
</ul>

<h2>3. Apples-to-apples: phases 0 (ED) and 7 (~ES)</h2>
<p>The only phases all four runs are scored on — removes the averaging-window mismatch.</p>
<table>
<tr><th>Run</th><th>★ motion t0</th><th>★ motion t7</th><th>bbox t0</th><th>bbox t7</th></tr>
{rt}
<tr style="background:#fafafa"><td><i>identity baseline</i></td><td class="num bigm">{IDM['t0']:.2f}</td><td class="num bigm">{IDM['t7']:.2f}</td><td class="num">{IDB['t0']:.2f}</td><td class="num">{IDB['t7']:.2f}</td></tr>
</table>
<div class="callout"><b>Reading it (motion):</b>
<ul>
<li><b>head-only at ED = 19.70 dB ≈ identity 19.65</b> — statistically no motion correction at ED, and only +0.7 at t7. The bbox view (29.6) hides this because static tissue carries it.</li>
<li><b>aggregator-finetune = +3.5 dB</b> over head-only at ED (23.1–23.4 vs 19.7). This is the decisive factor.</li>
<li><b>allphases_aggft ≈ t0t7_aggft</b> on motion (23.1/24.3 vs 23.4/24.1) — tied within noise.</li>
<li><b>no_zt</b> = 22.6/23.3, i.e. ~0.8 dB below the aggft pair on motion (vs only ~0.6 dB on bbox) — the embeddings help the dynamic region more than the global average suggested.</li>
</ul></div>

<h2>4. What the motion metric covers</h2>
<img src="data:image/png;base64,{imgs['motion_mask']}">
<p class="note">Val subject 0, mid-bbox z. Left: GT at ED. Middle: per-voxel motion magnitude (max−min over the 12 phases). Right: the binary motion mask (τ=0.05) overlaid in red — <b>7.9% of the bbox</b>, tightly localized on the myocardium/blood-pool. The motion PSNR is computed over exactly these red voxels.</p>

<h2>5. Secondary metrics (bbox / full / SSIM)</h2>
<p>Static-tissue-dominated, kept for completeness. Same ranking, compressed magnitudes; <b>full</b> is additionally distorted by zero-padded planes (prefer bbox) — its one real use here is exposing the no_zt spatial-placement failure (full drops ~3 dB).</p>
<img src="data:image/png;base64,{imgs['learning_curves']}">

<h2>6. Qualitative (validation subjects)</h2>
<p>Mid-(highest-energy) short-axis slice. Top: GT then each prediction; bottom: absolute error (magma). Pairs share subject + target phase (identical GT within a panel).</p>
<h3>6a. Aggregator-finetune vs head-only (all-phases pair)</h3>
<div class="grid2"><div><img src="data:image/png;base64,{imgs['cmp_allphases_P055']}"></div>
<div><img src="data:image/png;base64,{imgs['cmp_allphases_P048']}"></div></div>
<p class="note">Head-only error is brighter on the moving myocardium/blood-pool ring — the exact region the motion metric scores.</p>
<h3>6b. Aggregator-finetune vs z/t-embedding ablation, in-plane (t0/t7 pair)</h3>
<div class="grid2"><div><img src="data:image/png;base64,{imgs['cmp_t0t7_P055']}"></div>
<div><img src="data:image/png;base64,{imgs['cmp_t0t7_P048']}"></div></div>
<p class="note">Nearly identical at this in-plane slice — the no_zt penalty is <b>not</b> in-plane. §7 shows where it actually is.</p>
<h3>6c. Cardiac-cycle tracking (aggft, val subj 0)</h3>
<img src="data:image/png;base64,{imgs['cycle_aggft']}">
<p class="note">GT (top) vs V_canon (bottom) across all 12 phases. The heart tracks through the cycle, but the prediction is visibly smoother than GT (fine wall/texture detail is lost) — consistent with the ~23 dB motion PSNR being "partial correction", not solved.</p>

<h2>7. Effect of the z/t embeddings (no_zt ablation)</h2>
<p><b>Setup:</b> <span class="mono">vry47r4f</span> turns off <b>both</b> the z (per-slice depth) and t (per-slice input-phase) Fourier embedders, while <b>keeping</b> the target-phase query embedder; otherwise identical to <span class="mono">warrwlv8</span> (t0/t7, aggregator-finetune). Same val subjects &amp; target phases. ⚠️ <b>This removes z and input-t together, so it measures their <i>joint</i> effect — it cannot attribute anything to z alone.</b> Headline: <b>bbox/motion barely move (~0.5–0.8 dB), but full PSNR drops ~3 dB.</b> Below is where that ~3 dB lives.</p>

<div class="callout key"><b>Finding (observation): without the z/t conditioning, no_zt bleeds content into the out-of-FOV (zero-padded) z-planes</b> that should be empty. The in-FOV content reconstructs almost as well as aggft (~0.5 dB worse); the ~3 dB damage is concentrated entirely in the padding.</div>

<div class="callout warn"><b>Caveat — not a z-specific effect.</b> An earlier version of this section claimed the <i>z-embedder</i> was responsible ("its job is to tell the model where the volume ends in depth"). That is <b>not supported</b>: no_zt drops z and input-t jointly. A direct isolation in the head-only / ED-only setting (<span class="mono">fixedED_no_z</span> vs <span class="mono">fixedED_no_t</span> vs <span class="mono">fixedED_no_zt</span>) gives <b>identical</b> results — full ≈ 30.0–30.2, bbox ≈ 33.4–33.5, same tiny empty-plane bleed — i.e. dropping z alone, t alone, or both is indistinguishable. So there is <b>no evidence the bleed is caused by z specifically</b> (if anything, z and t look interchangeable). The bleed is a property of removing the conditioning + the splat's padding sensitivity, not of the depth embedder per se. (Those isolation runs are a different regime — head-only, ED-only — so they don't perfectly transfer to the aggft run, but no evidence favors the z-specific story either way.)</div>

<h3>7a. Through-plane (coronal) reslice — the tell</h3>
<img src="data:image/png;base64,{imgs['nozt_coronal']}">
<p class="note">P054 @t07, coronal cut (vertical axis = depth z, stretched ×6 for visibility). Rightmost = no_zt error: the bright bands at the <b>top and bottom edges</b> are content bleeding into z-planes outside the heart's extent. aggft (4th panel) keeps those planes clean. The central in-FOV region is comparable between the two.</p>

<h3>7b. Per-z PSNR including padded planes</h3>
<p>The gap is invisible in the content planes and catastrophic in the empty ones:</p>
<table>
<tr><th>z-plane type</th><th>aggft PSNR</th><th>no_zt PSNR</th><th>gap</th></tr>
<tr><td>In-FOV content planes (mean)</td><td class="num">~37 dB</td><td class="num">~36.5 dB</td><td class="num">~0.5 dB</td></tr>
<tr><td>Out-of-FOV padded planes (should be 0)</td><td class="num">64–120 dB (≈ exact 0)</td><td class="num">~17–25 dB</td><td class="num">≫ 40 dB</td></tr>
</table>
<p class="note">Examples (P055 @t07): padded z11 = aggft 64.2 dB vs no_zt 20.2 dB. P054 @t07: padded z0 = aggft 120 (perfect) vs no_zt 18.0 dB. Because full PSNR averages MSE over the whole cube, a few badly-filled empty planes dominate it — which is the entire ~3 dB full-PSNR deficit.</p>

<h3>7c. In-FOV per-z PSNR (content only)</h3>
<img src="data:image/png;base64,{imgs['nozt_perz']}">
<p class="note">Restricted to content voxels, aggft (blue) leads no_zt (green) by only a small, fairly uniform margin across depth — the genuine reconstruction penalty of dropping the embeddings is mild (~0.4–0.6 dB), matching the motion-metric gap.</p>

<h3>7d. Per-z axial montage (content planes)</h3>
<img src="data:image/png;base64,{imgs['nozt_montage']}">
<p class="note">P054 @t07, every in-FOV z-plane: GT / aggft / no_zt. Within the FOV the two predictions are nearly indistinguishable — confirming no_zt's loss is the out-of-FOV bleed (7a/7b), not in-plane quality.</p>
<div class="callout"><b>Takeaway on no_zt:</b> dropping the z/t embeddings costs only ~0.5–0.8 dB on the anatomy that matters (content + motion region); the big ~3 dB full-PSNR hit is concentrated in out-of-FOV padding (a splat-sensitivity artifact, not worse heart reconstruction) and is <b>not</b> attributable to the z-embedder specifically (7-caveat). The embeddings are cheap (~28K params) and add no compute, so they're worth keeping for the small consistent gain — but the model clearly recovers most of the reconstruction without explicit z/t conditioning, since the geometric <span class="mono">scanner_coords</span> input already carries depth into the splat.</div>

<h2>8. Caveats</h2>
<ul>
<li>Three runs are <b>still running</b> (~172k steps); head-only is finished (200k). Curves have plateaued so rankings are stable, but live numbers may tick up slightly.</li>
<li>Canonical-grid series — <b>not comparable to the pre-refactor 31 dB "4-day baseline"</b>.</li>
<li>The §1 <b>mean</b> averages different phase sets across the coverage split; use §3 for comparison.</li>
<li>Motion baseline differs marginally between val configs (20.5 for t0t7 vs 20.7 for allphases) because it averages different phase sets; the t0/t7 reference in §3 uses ~19.65/21.2.</li>
<li><b>No LV/myocardium-specific metric.</b> There is no cardiac-segmentation tool or chamber label available in this repo/env (only a whole-body elastix mask), so chamber-level fidelity (LV cavity vs the harder LV myocardium) is not quantified here. The honest signal remains the motion mask (§4), which covers the whole dynamic heart, not the LV alone.</li>
</ul>
</body></html>"""
open('_html/01_four_run_comparison.html','w').write(html)
print('wrote', len(html),'bytes')
