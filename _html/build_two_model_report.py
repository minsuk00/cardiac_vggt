import base64, json, numpy as np
def b64(p): return base64.b64encode(open(p,'rb').read()).decode()
A='_html/assets'
imgs={k:b64(f'{A}/{k}.png') for k in
      ['two_model_curves','m1_allphases_perphase','m1_cycle_subj0','m1_cycle_subj1',
       'm2_nozt_perphase','m2_subj0_t0','m2_subj0_t7','m2_subj1_t0','m2_subj1_t7',
       'inputs_subj0','inputs_subj1']}
d=json.load(open(f'{A}/analysis_metrics.json'))

def stats(mk):
    per=d[mk]; ts=sorted(set(r['t_target'] for r in per)); nsub=len(set(r['subj_idx'] for r in per))
    out={'ts':ts,'nsub':nsub}
    for m in ['motion','bbox','full']:
        v=[r[m] for r in per]; out[m]=(np.mean(v),np.min(v),np.max(v))
    pm={t:np.mean([r['motion'] for r in per if r['t_target']==t]) for t in ts}
    out['best']=max(pm,key=pm.get); out['worst']=min(pm,key=pm.get); out['pm']=pm
    return out
s1=stats('m1_allphases'); s2=stats('m2_nozt')

def row(mk,nm,phases,zt,sx):
    return f"""<tr><td><b>{nm}</b><br><span class=mono>{mk}</span></td><td>{phases}</td><td>{zt}</td>
    <td class=num>{sx['motion'][0]:.2f}<br><span class=note>[{sx['motion'][1]:.1f}–{sx['motion'][2]:.1f}]</span></td>
    <td class=num>{sx['bbox'][0]:.2f}<br><span class=note>[{sx['bbox'][1]:.1f}–{sx['bbox'][2]:.1f}]</span></td>
    <td class=num>{sx['full'][0]:.2f}<br><span class=note>[{sx['full'][1]:.1f}–{sx['full'][2]:.1f}]</span></td></tr>"""

html=f"""<!doctype html><html><head><meta charset=utf-8><title>Two-model analysis</title><style>
*{{box-sizing:border-box}} body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;padding:32px 24px;color:#1a1a1a;line-height:1.55}}
h1{{font-size:25px;margin-bottom:2px}} h2{{margin-top:36px;border-bottom:2px solid #eee;padding-bottom:6px}} h3{{margin-top:22px;color:#333}}
.sub{{color:#666;margin-top:0}} table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}}
th,td{{border:1px solid #e2e2e2;padding:7px 9px;text-align:left;vertical-align:top}} th{{background:#f6f7f9;font-weight:600}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} .mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#666}}
img{{max-width:100%;border:1px solid #e2e2e2;border-radius:6px;margin:8px 0}} .note{{font-size:11.5px;color:#888}}
.callout{{background:#f0f6ff;border-left:4px solid #1f77b4;padding:11px 15px;margin:14px 0;border-radius:0 6px 6px 0}}
.callout.key{{background:#eafaf0;border-color:#0a7d28}} .callout.warn{{background:#fff7e6;border-color:#e0a800}}
ul{{margin-top:6px}} li{{margin:4px 0}}
</style></head><body>

<h1>Per-target-phase analysis — two trained models</h1>
<p class=sub>Independent characterization (not a head-to-head). Fresh inference on the val set with each model's <code>checkpoint_last.pt</code>, swept across target phases. Generated 2026-06-10.</p>

<div class=callout><b>The two models</b><br>
<b>M1 = allphases_aggft (<span class=mono>fc8d065g</span>)</b> — aggregator-finetune, z+t+target_t embeddings all on, trained on all 12 target phases. Evaluated here at <b>every phase 0–11</b>.<br>
<b>M2 = no_zt (<span class=mono>vry47r4f</span>)</b> — aggregator-finetune, z &amp; input-t embeddings <b>off</b> (target_t on), trained on target phases {{0,7}}. Evaluated here at <b>t=0 and t=7</b>.</div>

<h2>0. Inputs — what the models were given</h2>
<p>Each subject's reconstruction uses a <b>pseudo-random scattered set of single-frame slices</b> — S slices, each at a different (cardiac-phase <i>t</i>, depth <i>z</i>) drawn from inside the anatomy bbox (the simulated sparse acquisition). The val draw is deterministic per subject (seeded by <code>seq_index</code>), and — critically — <b>the input set is held fixed while the target query phase is swept</b> (verified identical across t_target=0…11). So the per-phase montages below are a clean "same inputs, different requested phase" demonstration.</p>
<img src="data:image/png;base64,{imgs['inputs_subj0']}">
<img src="data:image/png;base64,{imgs['inputs_subj1']}">
<p class=note>The 9 inputs for subj0 span phases t∈{{0,4,6,7,8}} and depths z∈{{1..9}} — note the model is <b>never given a slice at most target phases</b>; it must synthesize the queried phase from slices acquired at other phases. Both M1 and M2 receive the identical input set per subject (the sampler is shared); they differ only in what conditioning the network attaches to each slice.</p>

<h2>1. Summary metrics (my inference, {s1['nsub']} val subjects)</h2>
<p>PSNR over three voxel sets: <b>motion</b> = dynamic heart voxels (the honest signal), <b>bbox</b> = in-FOV anatomy, <b>full</b> = whole 12×256×256 cube. Mean [min–max] across subjects×phases.</p>
<table><tr><th>Model</th><th>Phases</th><th>z/t emb</th><th>motion PSNR</th><th>bbox PSNR</th><th>full PSNR</th></tr>
{row('m1_allphases','M1 allphases',  '0–11','on',  s1)}
{row('m2_nozt','M2 no_zt','0, 7','OFF', s2)}
</table>
<p class=note>These are computed at the latest checkpoint over a fixed set of {s1['nsub']} val subjects (slightly different sampling/averaging than the wandb panels, so numbers differ by tenths of a dB but agree on the picture). SSIM omitted (the GPU SSIM kernel was unavailable in this run).</p>

<h2>2. Training curves (both models)</h2>
<img src="data:image/png;base64,{imgs['two_model_curves']}">
<p class=note>Left: validation motion PSNR (M1 = 12-phase mean, M2 = {{0,7}} mean). Right: bbox (solid) vs full (dashed). Note M2's full sits <i>below</i> its bbox — the out-of-FOV padding penalty discussed in the four-run report; M1's full sits above bbox (clean empty planes).</p>

<h2>3. M1 — allphases: performance across the whole cardiac cycle</h2>
<img src="data:image/png;base64,{imgs['m1_allphases_perphase']}">
<div class=callout key><b>Finding — a clear mid-systole dip.</b> Motion PSNR is highest at <b>ED (t0/t1 ≈ {s1['pm'][0]:.1f} dB)</b> and dips to its minimum at <b>peak contraction (t{s1['worst']} ≈ {s1['pm'][s1['worst']]:.1f} dB)</b>, a swing of ~{s1['pm'][s1['best']]-s1['pm'][s1['worst']]:.1f} dB, recovering toward end-systole (t7/t8). The model reconstructs every phase, but is hardest-pressed where the heart is furthest from its cycle-average shape — exactly where displacement is largest. bbox and full track the same U-shape. So "works on all phases" is true, but quality is <b>phase-dependent</b>: best at rest, weakest at peak systole.</div>

<h3>3a. Reconstruction across all 12 target phases — val subj0 (P053)</h3>
<img src="data:image/png;base64,{imgs['m1_cycle_subj0']}">
<h3>3b. Reconstruction across all 12 target phases — val subj1 (P055)</h3>
<img src="data:image/png;base64,{imgs['m1_cycle_subj1']}">
<p class=note>Rows: V_gt / V_canon / |error| at the highest-energy z-slice; columns = the 12 target phases (per-column GT bbox PSNR in title). The heart contracts and relaxes across the row and the prediction follows; error (bottom) concentrates on the moving myocardium/blood-pool and grows through systole — the visual counterpart of the §3 dip.</p>

<h2>4. M2 — no_zt: performance at t=0 (ED) and t=7 (~ES)</h2>
<img src="data:image/png;base64,{imgs['m2_nozt_perphase']}">
<div class=callout key><b>Finding — works on its two phases, with the no_zt signature.</b> Motion PSNR ≈ {s2['pm'][0]:.1f} dB at t0 and {s2['pm'][7]:.1f} dB at t7 — flat across its two trained phases, and on the anatomy (bbox ≈ {s2['bbox'][0]:.1f}) it reconstructs well. The distinctive mark: <b>full ({s2['full'][0]:.1f}) sits below bbox ({s2['bbox'][0]:.1f})</b>, the out-of-FOV padding bleed (the model lost depth/phase conditioning). It is a working 2-phase reconstructor; it simply leaves dirt in the empty planes.</div>

<div class=callout warn><b>Why this matters for the real-time free-breathing goal.</b> In a true real-time free-breathing acquisition we would <b>not know each input slice's cardiac phase t</b> (no ECG gating) — nor its respiratory phase r (an embedder we haven't built). The no_zt ablation removes exactly that per-slice <i>input-phase</i> conditioning (it keeps only the <i>target</i> query, which stays available — we always choose which phase to render). That it still reconstructs the heart at ~{s2['motion'][0]:.1f} dB motion is <b>encouraging evidence the approach can tolerate not knowing input-slice phase</b> — the network appears to infer enough phase/geometry from the images themselves via attention. <b>Caveats:</b> (i) no_zt also dropped the depth embedder z, which in deployment we <i>do</i> still have (slice z is known), so the realistic "unknown-t" model would keep z and likely do a bit better; (ii) it was trained on clean gated cine, so this doesn't yet test the real domain shift (bSSFP transient, single-shot artifacts, respiratory motion); (iii) the padding-bleed wrinkle remains. So: a promising signal that the information contract is viable, not proof of real-time transfer. See <code>docs/04_inference_information_contract.md</code>.</div>

<h3>4a. Per-z reconstruction — val subj0 (P053)</h3>
<div><img src="data:image/png;base64,{imgs['m2_subj0_t0']}"></div>
<div><img src="data:image/png;base64,{imgs['m2_subj0_t7']}"></div>
<h3>4b. Per-z reconstruction — val subj1 (P055)</h3>
<div><img src="data:image/png;base64,{imgs['m2_subj1_t0']}"></div>
<div><img src="data:image/png;base64,{imgs['m2_subj1_t7']}"></div>
<p class=note>Rows: V_gt / V_canon / |error| across all in-FOV z-planes, for t=0 (top pair) and t=7 (bottom pair). In-FOV anatomy reconstructs cleanly; the model handles both ED and ES at this subject.</p>

<h2>5. Takeaways</h2>
<ul>
<li><b>M1 (allphases) is a true full-cycle reconstructor</b> — every phase 0–11 reconstructs, motion PSNR ~{s1['motion'][0]:.1f} dB on average (range {s1['motion'][1]:.1f}–{s1['motion'][2]:.1f}), with a ~{s1['pm'][s1['best']]-s1['pm'][s1['worst']]:.1f} dB best-to-worst swing: easiest at ED, hardest at peak systole (t{s1['worst']}).</li>
<li><b>M2 (no_zt) works on its trained phases {{0,7}}</b> — flat ~{s2['motion'][0]:.1f} dB motion, good in-FOV anatomy, but carries the no_zt fingerprint of full&lt;bbox from out-of-FOV bleed.</li>
<li>Both are characterized on the honest <b>motion</b> metric; both land in the low-20s dB on the dynamic region — solid partial motion correction, with M1 generalizing across the entire cycle.</li>
</ul>
<p class=note><b>Method:</b> for each (subject, target phase) the val sampler draws input slices from <code>seq_index</code> (identical inputs as target phase varies), the model predicts per-pixel Δ, slices are splatted to V_canon, and PSNR is computed vs the on-disk phase volume over each voxel set. Motion mask = voxels whose intensity swings &gt;0.05 across the 12 phases. Script: <code>tools/analyze_two_models.py</code>.</p>
</body></html>"""
open('_html/02_two_model_analysis.html','w').write(html)
print('wrote _html/02_two_model_analysis.html', len(html), 'bytes')
