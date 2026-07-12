"""Build the self-contained HTML report for the 4wokxzov analysis (embeds figures as base64).
Beginner-friendly, step-by-step, honest post-debate numbers. Run: micromamba run -n svr python tools/build_4wok_report.py"""
import base64, json, os, glob
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(REPO, "result", "analysis_4wok", "figs")
OUT = os.path.join(REPO, "_html", "33_4wok_conclusive_analysis.html")


def img(name, cap):
    p = os.path.join(FIG, name)
    if not os.path.exists(p):
        return f'<div class="cap">[missing {name}]</div>'
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}"/><figcaption>{cap}</figcaption></figure>'


def load(path, default=None):
    p = os.path.join(REPO, path)
    return json.load(open(p)) if os.path.exists(p) else default


def main():
    ef = load("result/analysis_4wok/ef_honest.json", {})
    cmp = load("result/analysis_4wok/comparison_3way.json", {})
    p95 = load("result/analysis_4wok/p95_dvf.json", {})
    # head EF (may be pending)
    def head_ef(path):
        d = load(path)
        if not d:
            return None
        import numpy as np
        r = d["rows"]; pe = np.array([x["pred_ef"] for x in r]); ge = np.array([x["gt_ef"] for x in r])
        return {"slope": round(float(np.polyfit(ge, pe, 1)[0]), 3), "corr": round(float(np.corrcoef(ge, pe)[0, 1]), 3)}
    ref_ef = head_ef("scratch/analysis/phase_analysis/ref_vols/ef_ref.json")
    bsp_ef = head_ef("scratch/analysis/phase_analysis/bsp_vols/ef_bsp.json")

    css = """
    :root{--bg:#fff;--fg:#1a1a2e;--mut:#5a5a72;--card:#f6f7fb;--bd:#e2e4ee;--acc:#3a6ea5;--good:#1a7f4b;--bad:#b0341d;--warn:#9a6a00}
    @media (prefers-color-scheme:dark){:root{--bg:#12131a;--fg:#e6e7ee;--mut:#a0a2b4;--card:#1c1e27;--bd:#2c2f3c;--acc:#7fb0e0;--good:#5fd39a;--bad:#f08a72;--warn:#e0b84a}}
    :root[data-theme=dark]{--bg:#12131a;--fg:#e6e7ee;--mut:#a0a2b4;--card:#1c1e27;--bd:#2c2f3c;--acc:#7fb0e0;--good:#5fd39a;--bad:#f08a72;--warn:#e0b84a}
    :root[data-theme=light]{--bg:#fff;--fg:#1a1a2e;--mut:#5a5a72;--card:#f6f7fb;--bd:#e2e4ee;--acc:#3a6ea5;--good:#1a7f4b;--bad:#b0341d;--warn:#9a6a00}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
    .wrap{max-width:920px;margin:0 auto;padding:32px 20px 80px}
    h1{font-size:30px;line-height:1.2;margin:0 0 6px}h2{font-size:23px;margin:44px 0 12px;padding-top:14px;border-top:2px solid var(--bd)}
    h3{font-size:18px;margin:26px 0 8px;color:var(--acc)}
    p,li{color:var(--fg)}.mut{color:var(--mut)}
    .tldr{background:var(--card);border:1px solid var(--bd);border-left:4px solid var(--acc);border-radius:10px;padding:18px 22px;margin:20px 0}
    .tldr b{color:var(--acc)}
    figure{margin:18px 0;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px}
    figure img{width:100%;border-radius:6px;display:block}figcaption{font-size:13.5px;color:var(--mut);margin-top:8px}
    table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14.5px;overflow-x:auto;display:block}
    th,td{border:1px solid var(--bd);padding:7px 10px;text-align:left}th{background:var(--card)}
    code{background:var(--card);padding:1px 6px;border-radius:4px;font-size:13.5px}
    .good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}.warn{color:var(--warn);font-weight:600}
    .box{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 18px;margin:14px 0}
    .step{border-left:3px solid var(--acc);padding-left:16px;margin:16px 0}
    ul{padding-left:22px}.big{font-size:19px}
    """

    def T(rows, head):
        h = "".join(f"<th>{c}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"

    headef_row = f"<b>reference (L1-TV):</b> {'slope '+str(ref_ef['slope'])+', corr '+str(ref_ef['corr']) if ref_ef else '<span class=mut>(measuring…)</span>'} &nbsp;·&nbsp; <b>bspline:</b> {'slope '+str(bsp_ef['slope'])+', corr '+str(bsp_ef['corr']) if bsp_ef else '<span class=mut>(measuring…)</span>'}"

    def g(d, k, sub):
        return str((d.get(k, {}) or {}).get(sub, "?"))
    tbl_variants = T([
        ['4wok (smoothness)', g(cmp, '4wok_diffusion', 'breathing_slope'), g(cmp, '4wok_diffusion', 'cardiac_inplane_mm_mean_range'), g(p95, '4wok_diffusion', 'through_p95_mm') + ' mm', g(cmp, '4wok_diffusion', 'model_psnr_range')],
        ['reference (L1-TV)', g(cmp, 'reference_L1TV', 'breathing_slope'), g(cmp, 'reference_L1TV', 'cardiac_inplane_mm_mean_range'), g(p95, 'reference_L1TV', 'through_p95_mm') + ' mm', g(cmp, 'reference_L1TV', 'model_psnr_range')],
        ['bspline', g(cmp, 'bspline', 'breathing_slope'), g(cmp, 'bspline', 'cardiac_inplane_mm_mean_range'), g(p95, 'bspline', 'through_p95_mm') + ' mm', g(cmp, 'bspline', 'model_psnr_range')],
    ], ['variant', 'breathing slope', 'in-plane mean (mm)', 'through-plane p95', 'recon PSNR (dB)'])
    tbl_limits = T([
        ['1. EF under-squeezes (bias + rank)', 'Recovers ~½ the spread (Spearman 0.55) but under-predicts by ~9 points', '<span class=warn>Fixable</span> — the 9-pt <i>bias</i> is a free post-hoc calibration (spread already matches truth); the <i>rank</i> (0.55) is the genuinely soft part.'],
        ['2. Appearance gap (~7–9 dB)', 'Biggest raw error; the model can&#39;t paint target-phase looks it never saw', '<span class=bad>Accepted limit</span> — proven information-limited (fancier decoders add +0.03 dB). Dominant number, least worth chasing.'],
        ['3. Deep-breath under-correction', '~1.9 dB in a ~12% tail; ~54% of deep breaths ignored', '<span class=warn>Half-fixable</span> — the renderer suppresses part of it (fixable by retraining); the rest is blind-breathing information limit.'],
        ['4. Through-plane cardiac ≈ 0', 'p95 0.49 mm', '<span class=mut>Accepted (cosmetic)</span> — bspline applies 3× more and changes nothing; EF is in-plane-dominated.'],
        ['5. Trained @20 slices, tested @12', 'Absolute numbers mildly off-distribution', '<span class=mut>Analysis artifact</span> — re-evaluate at 20; free.'],
        ['6. EF via segmenter on blurry recon, n=30', 'Correlations sensitive to individual patients', '<span class=mut>Analysis artifact</span> — report robust rank correlation + confidence interval.'],
    ], ['# Current limitation', 'What it is', 'Verdict'])
    tbl_roadmap = T([
        ['<b>0a</b> Re-evaluate at 20 slices', 'Match the training regime', 'Locks the honest numbers', '<span class=good>free (no training)</span>'],
        ['<b>0b</b> Report robust metrics + calibrate EF', 'Rank-correlation + confidence interval; linear EF calibration', 'Removes the 9-pt bias; honest headline', '<span class=good>free</span>'],
        ['<b>0c</b> Proximity sampling', 'Query the phase near the target', '+0.8 dB, no extra frames', '<span class=good>free eval</span>'],
        ['<b>1</b> Coverage-free renderer <i>re-train</i> ⭐', 'Replace the splat that penalizes across-slice motion; retrain', 'Banks the fixable half of breathing (~0.5–1 dB); may sharpen recon. Won&#39;t break the appearance wall.', '<span class=warn>short retrain</span>'],
        ['<b>2</b> Confirm multi-frame-per-slice', 'Evaluate the already-built S=20 cine regime (doc 28)', 'Could lift EF + shallow-breath from content, no phase labels', '<span class=warn>train / eval</span>'],
        ['<b>3</b> A few more reference planes', 'k=2–3 target-phase references', 'EF slope 0.77 → ~0.9 (prior ablation → 1 at k=6)', '<span class=warn>short retrain, mild goal cost</span>'],
        ['<b>defer</b> Multi-orientation (add LAX views)', 'The <i>only</i> principled fix for through-plane + appearance (a breathing shift is in-plane for a long-axis view)', 'Large — but partly breaks the fast 1-frame goal; EF already works without it', '<span class=bad>new data + full retrain</span>'],
    ], ['Priority · Action', 'What', 'Expected payoff', 'Cost'])

    html = f"""<style>{css}</style>
<div class="wrap">
<h1>What can (and can't) the slice-to-volume heart model do?</h1>
<p class="mut">A conclusive analysis of model <code>4wokxzov</code> — reference-slot, 1-frame-per-slice, smoothness-regularized. Measured on 30 held-out patients. Written for a reader new to the project.</p>

<div class="tldr">
<b>TL;DR.</b> The model reconstructs a beating 3-D heart from single 2-D slices. We measured it on four abilities.
<b class="good">It does the clinically important thing well:</b> it recovers each patient's <b>ejection fraction</b> (how much blood the heart pumps) — about <b>half the patient-to-patient spread</b> (Spearman ≈ 0.55), a big jump from the older "everyone looks average" model, though it slightly under-squeezes. Its <b>in-plane</b> (within-slice) heart motion and <b>timing</b> are real and correct.
<b class="warn">Its real weaknesses are narrow:</b> it barely corrects deep <b>breathing</b> shifts (ignores ~half of the big ones) and applies almost no <b>through-plane</b> (across-slice) motion — but the latter barely matters for the volume. The biggest raw error is <b>appearance it was never shown</b> (a fundamental limit of the fast 1-frame acquisition, not a bug). Four independent AI reviewers stress-tested every claim and caught two of my own mistakes, now fixed.
</div>

<h2>1. What is the model doing? (the big picture)</h2>
<p>Imagine photographing a loaf of bread by slicing it and taking one flat photo of each slice, then stacking the photos to rebuild the 3-D loaf. This model does that for a <b>beating heart</b>: it takes a few flat 2-D MRI slices and rebuilds the full 3-D heart at a chosen moment of the heartbeat. The goal is <b>speed</b> — one quick photo per depth instead of the slow standard method.</p>
<p>Two things make it hard: (1) the patient is <b>breathing</b>, which slides the heart up/down between photos so they don't stack cleanly; (2) the heart is <b>beating</b>, so different photos catch it at different moments. The model's job is to place every slice's content correctly in 3-D at one target moment. It's told the target moment by being handed <b>one real reference slice</b> at that moment (slot 0) — this is how it knows what to aim for.</p>

<h2>2. What we measured, and how (step by step)</h2>
<div class="step"><b>Step 1 — Pick the model.</b> <code>4wokxzov</code> = run 217720691: uses the reference-slot design, a "DPT" motion head, one frame per slice, and was trained with an extra <b>smoothness penalty</b> on the predicted motion (called the "diffusion" loss — this is <i>not</i> a generative diffusion model; it just discourages jagged motion fields, added to fix checkerboard artifacts an earlier penalty caused).</div>
<div class="step"><b>Step 2 — Read the motion directly, not just the score.</b> For every input slice we recover the model's predicted displacement (how far, in millimeters, it moves each pixel) and compare to the truth. We separate <b>breathing</b> (turn the simulated breathing on) from <b>heartbeat</b> motion (turn breathing off).</div>
<div class="step"><b>Step 3 — Measure ejection fraction (EF).</b> We reconstruct each patient's heart at all 12 phases, run an independent AI segmenter (nnU-Net) to measure the blood-pool volume, and compare predicted EF to true EF. EF = the fraction of blood ejected per beat — the number cardiologists care about.</div>
<div class="step"><b>Step 4 — Compare three model variants</b> (the smoothness penalty vs the original vs a "bspline" motion head) to see if the design choices matter.</div>
<div class="step"><b>Step 5 — Have 4 AI agents attack every conclusion</b> to catch confounds, leaks, and bugs before believing anything.</div>

<div class="box"><b>Two bugs the process caught (and I fixed):</b>
<ul>
<li><span class="bad">Wrong input:</span> my first run forgot to feed the model its reference slice, so it looked like it did no motion at all. Fixing it made the model jump +2 dB and its motion vary correctly with the target phase.</li>
<li><span class="bad">A broken statistic and a mislabel:</span> a "137% recovery" number used inconsistent baselines (real value: it removes ~31% of the breathing error), and I'd called a "do-nothing floor" a "ceiling." Both corrected below.</li>
<li><span class="bad">A silent tool failure:</span> the EF segmenter failed ~4 times silently because of a wrong model name — fixed once we read the actual error.</li>
</ul></div>

<h2>3. Ability 1 — Ejection fraction (the clinical headline): <span class="good">recovered</span></h2>
<p>This is the most important result and it was a <b>surprise</b>. Earlier in the project the model reconstructed a flat ~48% EF for <i>everyone</i> regardless of their true EF. This model does <b>not</b>: predicted EF tracks true EF with <b>Spearman ≈ 0.55</b> (rank correlation), recovering roughly <b>half the patient-to-patient spread</b>. It still slightly under-squeezes (predicts 54% when the truth averages 63%).</p>
<p><b>Is this real or a trick?</b> The reconstruction literally contains one true slice (the reference), so a skeptic worried the EF "recovery" was just copying that one slice. The decisive control: <b>remove</b> the reference slice from the EF calculation entirely — the correlation barely drops (0.77→0.68), and the apical/basal slices the model <i>never</i> observed still carry the per-patient amplitude. So the recovery <b>generalizes</b> beyond the copied plane — it's learned, not a copy. Verdict: <span class="good">real, though I initially overstated the exact number.</span> (An earlier "the reference model is flat" argument turned out to be an <i>undertrained checkpoint</i>; at its final checkpoint that model also recovers EF — see §7.)</p>
{img('cardiac_cycle_s0.png','A patient&#39;s heartbeat: top row = ground truth at each of 12 phases, bottom = model reconstruction. The chamber visibly shrinks and grows — the model captures the contraction (that&#39;s why EF is recovered), though the squeeze is slightly gentler than truth.')}

<h2>4. Ability 2 — In-plane heart motion (within each slice): <span class="good">real</span></h2>
<p>The <i>average</i> predicted motion looks tiny (~0.5 mm), which first made it seem like the model does nothing. But the average is misleading — it's diluted by the many still pixels (chest wall, background). Looking at the <b>top 5%</b> of pixels (the moving heart muscle), the motion is <b>2.9 mm (up to 6.7 mm)</b> — real, meaningful contraction, concentrated exactly where the heart moves. This is why EF works.</p>
{img('recon_s0_ES.png','At peak contraction (ES): a scattered input slice, the ground-truth heart, the do-nothing baseline, the model&#39;s reconstruction, and the error. The model beats the do-nothing baseline and captures the contracted chamber.')}

<h2>5. Ability 3 — Through-plane heart motion (across slices): <span class="warn">minimal, but barely matters</span></h2>
<p>As the heart beats it also shortens along its long axis, which is the <b>through-plane</b> (stacking) direction for these slices. The model applies almost none of this — even the top 5% is only <b>0.5 mm</b>. However, this barely affects the result: the chamber's <b>volume</b> is dominated by the in-plane area change (which the model <i>does</i> get), so EF still recovers despite near-zero through-plane motion. A minor limitation, not the headline.</p>
{img('dvf_s0_ES.png','The predicted displacement field for one slice. The through-plane component (Δz, right) is essentially flat — the model applies almost no across-slice cardiac motion.')}

<h2>6. Ability 4 — Breathing correction: <span class="warn">partial; deep breaths under-corrected</span></h2>
<p>Breathing slides the heart through-plane; the model must undo it. It handles <b>small/typical</b> breaths well but <b>under-corrects deep ones</b>: for a ~17 mm breath it applies only ~5.7 mm, and about <b>half of the deepest breaths are essentially ignored</b>. Overall breathing leaves ~1.9 dB of error the model can't remove.</p>
<p><b>Why?</b> Two reasons, and the debate showed it's a mix: (1) a <b>rendering</b> issue — the way slices are accumulated into the volume actively penalizes moving a slice across planes (it creates a gap), so the model learns to under-move; (2) an <b>information</b> limit — from a single cropped slice with no breathing signal, the model genuinely can't tell a 12 mm breath from an 18 mm one. Importantly, the model <i>does</i> extract partial breathing information (correlation 0.52, not zero), so it's not hopeless — but the deep tail is partly unfixable without changing the acquisition.</p>
{img('breath_s0.png','Simulated breathing shifts a slice&#39;s anatomy through-plane (here by a large amount). The model corrects small shifts well but under-corrects large ones.')}

<h2>7. Do the design choices matter? (three variants)</h2>
<p>We compared the smoothness-penalty model (4wok) against the original (L1-TV) and a bspline motion head. On reconstruction quality they are <b>essentially identical</b> (~23 dB). The bspline head applies ~3× more through-plane motion but doesn't improve the score — so the motion head is a <b>second-order</b> choice. And crucially for the clinical number: <b>the L1-TV model recovers EF just as well (slope 0.79 ≈ 4wok's 0.77) at its final checkpoint</b> — so it's the <b>reference-slot design</b> that drives EF recovery, not the smoothness penalty. (The "smoothness helps EF" idea was a red herring — the earlier flat L1-TV number was simply an undertrained checkpoint.)</p>
{tbl_variants}
<p class="mut">EF by head (settles whether the design changes the clinical number): {headef_row}</p>

<h2>8. Round 1 debate — checking the findings</h2>
<p>Four independent reviewers attacked the conclusions against the raw data and code. Outcome:</p>
<ul>
<li><b>EF recovery: real but I overstated it.</b> The honest number is Spearman ≈ 0.55 (half the spread), not 0.77 — the higher number leaned on one lucky patient and a small leak. Still a genuine, large improvement over flat.</li>
<li><b>Two of my numbers were wrong</b> (a "137% recovery" statistic and a "ceiling" mislabel) — corrected.</li>
<li><b>Breathing is not the top problem</b> given EF is solved; its residual is small (~1.9 dB) and lives in a ~12% deep-breath tail.</li>
<li><b>The biggest raw error</b> (~7–9 dB) is <b>appearance the model was never shown</b> — a fundamental consequence of the fast 1-frame acquisition, not a fixable bug.</li>
</ul>

<h2>9. Round 2 debate — the current limitations, ranked</h2>
<p>A second four-agent debate then argued specifically about <b>what is actually limiting the model and what to do about it</b> — a "ship it, nothing's broken" minimalist vs a "the renderer is the fixable root" champion vs a "we must change the acquisition" champion, refereed by an adjudicator. They converged on this ranked picture. The key colour code: <span class="warn">fixable</span>, <span class="bad">accepted information-limit</span> (can't fix without changing the acquisition), <span class="mut">artifact of this analysis / cosmetic</span>.</p>
{tbl_limits}
<div class="box"><b>The three disputes they settled:</b>
<ul>
<li><b>Is the deep-breath tail fixable?</b> <span class="warn">Half.</span> The rendering step secretly penalizes across-slice motion, so retraining with a better renderer recovers the shallow/medium part. But the <i>deepest</i> breaths are genuinely unobservable from one cropped slice (the model can't see how deep the breath was) — that part is a hard information limit. So: a bounded ~0.5–1 dB win, not a full fix.</li>
<li><b>Should we chase through-plane motion?</b> <span class="mut">No.</span> The bspline variant already applies 3× more of it and changes <i>nothing</i> (same score, same EF) — it's cosmetic because the heart's volume is set by the in-plane squeeze, which the model already gets.</li>
<li><b>Should we add other views (multi-orientation)?</b> <span class="bad">Not now — but it's the only real lever for the hard limits.</span> A breathing shift that's invisible across-plane in a short-axis view is <i>directly visible</i> in a long-axis view. Adding a few long-axis frames is the one principled way to fix through-plane motion and the appearance gap — but it partly breaks the project's "one fast frame per slice" goal, and EF (the clinical deliverable) already works without it. So: document it as a deliberate speed↔quality upgrade, don't build it yet.</li>
</ul></div>

<h2>10. What we need to do — the roadmap</h2>
<p>Ordered by payoff per unit effort. The headline: <b>this is a good model whose biggest errors are information-limits of the fast acquisition, not bugs.</b> So the plan is mostly cheap "confirm and bank," not a redesign.</p>
{tbl_roadmap}
<div class="box big"><b>#1 next action:</b> re-evaluate at the training slot count (20) and report the robust rank-correlation with a confidence interval + a calibrated EF — this is free and locks the honest headline. <b>Then</b> the one worthwhile retrain: swap in a coverage-free renderer to bank the fixable half of the breathing error. Everything past that (more reference planes, multi-frame, multi-orientation) is optional and trades against the fast-acquisition goal — pursue only if a through-plane clinical endpoint (e.g. regional wall-motion / strain) becomes the target.</div>

<h2>11. Honest limitations of this analysis</h2>
<ul>
<li>The model was trained with ~20 slices/patient but evaluated with 12, so absolute numbers are mildly off-distribution (relative comparisons are robust) — fixed by roadmap item 0a.</li>
<li>EF is measured through an AI segmenter on blurry reconstructions; the blur biases the numbers, and n=30 makes correlations sensitive to individual patients — hence we report the robust rank correlation.</li>
<li>The "does the smoothness penalty help EF" question is now <b>settled: no</b> — the L1-TV model recovers EF equally at its final checkpoint (slope 0.79), so the reference-slot design, not the penalty, drives it.</li>
</ul>

<p class="mut" style="margin-top:40px">Data: <code>result/analysis_4wok/</code>, <code>scratch/analysis/phase_analysis/{{4wok,ref,bsp}}_vols/ef_*.json</code>. Scripts: <code>tools/exp_4wok_analysis.py</code>, <code>exp_4wok_p95.py</code>, <code>render_4wok_qualitative.py</code>, <code>build_4wok_report.py</code>. Companion doc: <code>docs/33</code>. Two four-agent debates informed §8–10.</p>
</div>"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(html)
    print("wrote", OUT, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
