"""Build the self-contained HTML report: limitations + proven improvements.

Reads result/limits_eval/*.json and embeds result/limits_eval/*.png as base64.
Output: _html/14_limitations_and_improvements.html
"""
import base64, json, os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(REPO, "result", "limits_eval")
OUT = os.path.join(REPO, "_html", "14_limitations_and_improvements.html")

dec = json.load(open(os.path.join(D, "decomposition.json")))
imp = json.load(open(os.path.join(D, "improvements.json")))
ks = json.load(open(os.path.join(D, "kspace_singleshot.json")))
N = dec["n"]


def b64(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return ""
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def img(name, cap=""):
    src = b64(name)
    if not src:
        return f"<p style='color:#c00'>[missing {name}]</p>"
    c = f"<div class='cap'>{cap}</div>" if cap else ""
    return f"<figure><img src='{src}'/>{c}</figure>"


# pull key numbers
id_m = dec["identity"]["psnr_motion"]
mc_m = dec["model_canon"]["psnr_motion"]
mr_m = dec["model_refined"]["psnr_motion"]
or_m = dec["oracle_perfect"]["psnr_motion"]
head = or_m - mr_m
or_sh = dec["oracle_perfect"]["sharp_rel"]
nat_sh = dec["oracle_native256"]["sharp_rel"]
mr_sh = dec["model_refined"]["sharp_rel"]
dc_model = dec["data_consistency_psnr"]["model_refined"]
ood = dec["ood_breathing_motion_psnr"]
ind = dec["ind_breathing_motion_psnr"]
gain_ind = ind["model_refined"] - ind["identity"]
gain_ood = ood["model_refined"] - ood["identity"]
ks_drop = ks.get("model_R8", {}).get("drop", float("nan"))
ks_clean = ks.get("model_R8", {}).get("motion_clean", float("nan"))
ks_alias = ks.get("model_R8", {}).get("motion_aliased", float("nan"))

ns = imp["native_splat"]
nat_model_before = ns["canon_518"]["sharp_rel"]
nat_model_after = ns["canon_native"]["sharp_rel"]
Kgrid = sorted({int(k.split("K")[1]) for k in imp["multidraw"] if k.startswith("refined_K")})
md0 = imp["multidraw"][f"refined_K{Kgrid[0]}"]["psnr_motion"]
mdK = imp["multidraw"][f"refined_K{Kgrid[-1]}"]["psnr_motion"]
md_gain = mdK - md0

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>VGGT-MRI — limitations & proven improvements</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;
margin:0 auto;padding:28px 30px;color:#202124;line-height:1.55;font-size:15.5px}}
h1{{font-size:25px;border-bottom:3px solid #1a73e8;padding-bottom:8px}}
h2{{font-size:20px;margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:5px}}
h3{{font-size:16.5px;margin-top:22px;color:#174ea6}}
.tldr{{background:#e8f0fe;border-left:5px solid #1a73e8;padding:14px 18px;border-radius:5px;margin:18px 0}}
.tldr b{{color:#174ea6}}
figure{{margin:16px 0;text-align:center}} img{{max-width:100%;border:1px solid #e0e0e0;border-radius:5px}}
.cap{{font-size:13px;color:#5f6368;margin-top:5px;text-align:center}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
th,td{{border:1px solid #dadce0;padding:7px 10px;text-align:center}}
th{{background:#f1f3f4}} td.l,th.l{{text-align:left}}
.good{{color:#188038;font-weight:bold}} .bad{{color:#c5221f;font-weight:bold}} .mid{{color:#e8710a;font-weight:bold}}
.box{{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;padding:12px 16px;margin:14px 0}}
.win{{background:#e6f4ea;border-left:5px solid #188038}}
.kill{{background:#fce8e6;border-left:5px solid #c5221f}}
code{{background:#f1f3f4;padding:1px 5px;border-radius:3px;font-size:13.5px}}
small{{color:#5f6368}}
</style></head><body>

<h1>VGGT + refiner: where the limits are, and what actually moves the needle</h1>
<p><small>End-to-end, training-free decomposition on the current best model
(<b>joint refiner, epoch ~76</b>, <code>218349151_mri_refiner_joint/checkpoint_last.pt</code>),
breathing-val protocol, n={N}. MOTION PSNR (dynamic heart voxels) is the headline metric throughout.
Generated in-session; reproduce via <code>tools/limits_decomposition.py</code>,
<code>tools/improvements_test.py</code>, <code>tools/kspace_singleshot_toy.py</code>,
<code>tools/make_limits_figures.py</code>.</small></p>

<div class="tldr">
<b>TL;DR (human-facing).</b> I stress-tested the VGGT+refiner pipeline with a ladder of
training-free "oracle" reconstructions and a 4-expert debate, and the picture is clear and
actionable — it splits into <b>two independent levers</b> plus one transfer risk:
<ul>
<li><b>Accuracy (MOTION PSNR) is GEOMETRY-limited, not renderer-limited.</b> With perfect motion/placement
the same splat reaches <b>{or_m:.1f} dB</b> motion; the model sits at <b>{mr_m:.1f} dB</b> — a
<b class="bad">~{head:.0f} dB</b> gap that lives in <b>motion estimation</b>, not the splat or the refiner.
(Floor = {id_m:.1f}.) Most of the project's effort — refiner, SSIM — has been polishing the wrong lever.</li>
<li><b>Sharpness IS renderer-limited, but the culprit is the 256→518→256 RESIZE, not the splat kernel.</b>
A native-resolution splat (no resize) reaches <b class="good">{nat_sh:.3f}× GT</b> sharpness vs the current
pipeline's <b class="mid">{or_sh:.3f}×</b> — the trilinear scatter itself is fine. <b>No loss function can
break this; the fix is a renderer change.</b></li>
<li><b>The real binding constraint for the stated goal (real-time transfer) is the DOMAIN GAP.</b> Feeding
true single-shot–aliased input (R=8) — what real-time actually delivers — drops the model
<b class="bad">{ks_drop:.1f} dB</b> ({ks_clean:.1f}→{ks_alias:.1f}), back near the do-nothing floor. The model has
never seen acquisition artifacts. And every number here is on the model's own simulator — <b>there is no
real-data metric.</b></li>
</ul>
<b>Proven improvements (ranked):</b> (1) <b>native-resolution splat</b> — recovers sharpness, training-free
evidence; (2) <b>multi-draw test-time ensemble</b> — <b class="good">+{md_gain:.1f} dB</b> motion, free;
(3) <b>degraded-input augmentation</b> — proven necessary (the {ks_drop:.0f} dB drop) and plausible (the
respiratory-sim precedent generalized). <b>Ranked DOWN by evidence:</b> fancier losses (sharpness is
splat-limited, not loss-limited), a data-consistency loss (already ~{dc_model:.0f} dB satisfied), unfreezing
the backbone (grad-norm ~0.014), and k-space input (incompatible with the frozen RGB backbone).
The biggest gap is not a model change at all: <b>a real-data, motion-region metric.</b>
</div>

<h2>1. Method — how this was measured</h2>
<p>The pipeline transports input pixel intensities: <code>world_points = scanner_coords + Δ</code> →
trilinear <b>splat</b> into a 12×256×256 cube → <code>V_canon</code> → 3D-UNet <b>refiner</b> →
<code>V_refined</code>, scored against the target-phase volume. To find <i>where</i> the error is, I built a
<b>ladder of reconstructors</b> that each removes one source of error, all scored identically:</p>
<ul>
<li><b>identity</b> — Δ=0 splat of the (breathing-corrupted) input slices. The do-nothing floor.</li>
<li><b>model V_canon / V_refined</b> — the trained joint model. The operating point.</li>
<li><b>oracle (perfect placement)</b> — splat the <i>true target-phase planes</i> at the model's z's with Δ=0.
This is an <b>upper bound</b>: it hands the model perfect motion correction AND perfect appearance.
(Honest caveat in §3.1: a pure-warp model can't fully reach it.)</li>
<li><b>oracle native-256 / nearest / super-2×</b> — the same perfect-placement splat but varying the
<i>renderer</i> (skip the 518 resize / drop the trilinear tent / use a 512 grid) to attribute the sharpness loss.</li>
</ul>
<p>Plus three diagnostics: <b>data-consistency</b> (re-slice the recon at its input geometry vs the input
pixels), an <b>out-of-distribution breathing</b> test (re-score under a respiratory waveform the model never
trained on — a sim-overfit check), and a <b>single-shot k-space</b> test (feed R=8-aliased input). I also ran a
<b>4-expert debate</b> (MRI physicist · neural-rendering architect · pragmatic experimentalist · red-team
skeptic); their convergence and disagreements shaped the experiments and are summarized in §4.</p>

<h2>2. The operating point</h2>
<table>
<tr><th class="l">reconstructor</th><th>MOTION PSNR ↑</th><th>bbox PSNR</th><th>sharp/GT</th></tr>
<tr><td class="l">identity (floor)</td><td>{id_m:.2f}</td><td>{dec['identity']['psnr_bbox']:.2f}</td><td>{dec['identity']['sharp_rel']:.3f}</td></tr>
<tr><td class="l">model V_canon</td><td>{mc_m:.2f}</td><td>{dec['model_canon']['psnr_bbox']:.2f}</td><td>{dec['model_canon']['sharp_rel']:.3f}</td></tr>
<tr><td class="l">model V_refined <small>(best)</small></td><td><b>{mr_m:.2f}</b></td><td>{dec['model_refined']['psnr_bbox']:.2f}</td><td>{mr_sh:.3f}</td></tr>
<tr style="background:#fff4e5"><td class="l">oracle (perfect placement)</td><td class="mid">{or_m:.2f}</td><td>{dec['oracle_perfect']['psnr_bbox']:.2f}</td><td>{or_sh:.3f}</td></tr>
</table>
<p>The refiner adds {mr_m-mc_m:+.2f} dB motion over the raw splat — consistent with reports 10–12. But the
headline is the last row: <b>perfect placement is {head:.0f} dB above the model.</b></p>

<h2>3. Limitations (with the toy experiment that proves each)</h2>
{img("qual_panel.png", "One subject, mid-heart planes. Row 2 (model) is blurry AND displaced; row 3 (perfect placement, current splat) is correct but resize-blurred; row 4 (native-256 splat) is crisp. The two rows below the model isolate the two levers.")}

<h3>3.1 — The dominant gap is MOTION ESTIMATION, not the renderer (~{head:.0f} dB)</h3>
{img("fig_ladder.png", "Motion-PSNR ladder, n="+str(N)+". The jump from the model to the perfect-placement oracle is the recoverable motion-estimation gap.")}
<p>With the <i>same splat and no refiner</i>, perfect motion/placement reaches <b>{or_m:.1f} dB</b> motion —
so the renderer can represent the dynamic heart far better than the model currently extracts. The model's
{mr_m:.1f} dB is bottlenecked by <b>getting the displacement field right</b>, which matches doc 07's finding
that the model <i>under-corrects</i> (DVF fit slope 0.42, plateaus at 7–8 mm vs ~24 mm applied).</p>
<div class="box"><b>Honest caveat (red-team).</b> The oracle feeds the <i>true target-phase</i> planes, so it
also hands the model appearance a pure-warp transport can't synthesize (ES blood-pool, through-plane
disocclusion). So the full {head:.0f} dB is <b>not</b> all reachable by better Δ — part is the warp-only wall
and part is the information the model is contractually denied (no input cardiac t, no respiratory r → forced
regression-to-mean shrinkage). The <i>achievable</i> ceiling sits between {mr_m:.0f} and {or_m:.0f} dB. But the
direction is unambiguous: <b>the lever is motion/geometry, and the model is well below even a conservative
ceiling.</b></p></div>

<h3>3.2 — The sharpness ceiling is the RESIZE, not the kernel or the loss</h3>
{img("fig_sharpness.png", "Sharpness attribution at perfect placement. Skipping the 256→518→256 resize (native-256 splat) recovers almost all the lost detail; the trilinear kernel and grid resolution are not the problem.")}
<p>Reports 08/12 attributed the blur to "the splat" and showed neither L1 nor SSIM breaks ~0.69× GT. This
decomposition pinpoints <i>why</i>: at perfect placement, the current 518→256 splat gives <b
class="mid">{or_sh:.3f}×</b>, but splatting at <b>native 256 (no resize)</b> gives <b class="good">{nat_sh:.3f}×</b>
— near-perfect. Nearest-neighbour matches native trilinear ({dec['oracle_nearest']['sharp_rel']:.3f}), so the
trilinear tent is innocent; a 512 grid only partly helps ({dec['oracle_super2x']['sharp_rel']:.3f}). <b>The
256→518→256 resampling round-trip is where the high-frequency detail dies.</b> This is the strongest single
result in the report: it converts "the splat is lossy, live with it" into a concrete, fixable renderer bug
(see §5, Improvement A).</p>

<h3>3.3 — Coverage caps at 12 planes, but it's not the motion bottleneck</h3>
<p>The splat fills only ~{dec['coverage']['identity']*100:.0f}% of cube voxels, and <code>S</code> is hard-capped
at <code>min(T=12, bbox_z, img_per_seq)</code> ⇒ you cannot feed &gt;12 input planes without enlarging the
canonical grid. <b>But</b> the dynamic heart lives in well-covered central planes, and the perfect-placement
oracle reaches {or_m:.0f} dB <i>at the same {dec['coverage']['oracle_perfect']*100:.0f}% coverage</i> — so coverage
is not what caps motion PSNR. Multi-draw ensembling (§5-B) confirms: unioning coverage over draws buys only
modest motion PSNR. Coverage is a <i>full/bbox</i> problem, not a <i>motion</i> problem.</p>

<h3>3.4 — The real binding constraint: the domain gap (and no real-data metric)</h3>
{img("fig_domaingap.png", "Single-shot realism: feeding R=8-aliased input (what real-time delivers) collapses the model toward the floor.")}
{img("kspace_singleshot.png", "Toy: clean gated cine (trained on) vs simulated single-shot R={4,8,12} input. The model has never seen this.")}
<p>Everything above is measured on <i>clean simulated</i> data. The project's goal is gated→real-time transfer,
and real-time gives <b>single-shot, prospectively-undersampled</b> frames. Feeding R=8-aliased input drops the
model <b class="bad">{ks_drop:.1f} dB</b> motion ({ks_clean:.1f}→{ks_alias:.1f}) — within ~1 dB of the do-nothing
floor. The model is <b>brittle to the artifacts that define its target domain.</b> Combined with the fact that
<b>no clean 3-D real-data reference exists</b> (OCMR, report 13, is k-t-reconstructed and unscored), this — not
the splat — is the project's true risk.</p>
<div class="box win"><b>Reassuring counter-evidence (sim-overfit is NOT happening, within the motion family).</b>
A real worry was that the model just memorized the training respiratory waveform. It did not: under an
<b>out-of-distribution</b> respiratory model (different waveform shape, larger amplitude, more direction
jitter), the model's gain over identity <b>survives and grows</b> ({gain_ind:+.1f} dB in-distribution →
{gain_ood:+.1f} dB OOD). It learned to correct motion, not to replay a simulator. The untested gap is the
<i>non-rigid single-shot</i> physics, not the breathing parameters.</p></div>

<h3>3.5 — Two more limits worth noting (and what they rule OUT)</h3>
<p><b>Data consistency is already satisfied.</b> Re-slicing the model's recon at its own input geometry gives
<b>{dc_model:.0f} dB</b> — the splat round-trips its own pixels well, so a data-consistency loss (a popular
suggestion) has little gradient to give. <b>The frozen backbone is not the bottleneck</b>: even in the joint
run the aggregator's grad-norm is ~0.014 vs the point head's ~0.6 — unfreezing 605M params (2.8× slower) barely
moves them. Both are evidence-based <i>kills</i>, not endorsements.</p>

<h2>4. The expert debate (4 agents, independent)</h2>
<p>I ran four adversarial subagents. Their <b>convergence</b> was striking: all three "builder" agents
independently nominated a <i>training-free decomposition</i> as the highest-value next step (the physicist's
data-consistency probe, the architect's resize-vs-kernel split, the pragmatist's coverage/ensemble sweep) —
which is exactly what this report executes. Key positions:</p>
<table>
<tr><th class="l">Agent</th><th class="l">Top pick</th><th class="l">Strongest kill</th></tr>
<tr><td class="l">MRI physicist</td><td class="l">Data-consistency loss (r/t-blind-safe)</td><td class="l">k-space input — kills own field's idea (frozen RGB backbone)</td></tr>
<tr><td class="l">Rendering architect</td><td class="l">Feature-splat + tiny decoder (break warp-only wall)</td><td class="l">Unfreezing backbone (0.014 grad); splat-free decode = ablation not product</td></tr>
<tr><td class="l">Pragmatist</td><td class="l">More slices / coverage + multi-draw ensemble</td><td class="l">Fancy/perceptual losses — "a trap; sharpness is splat-limited"</td></tr>
<tr><td class="l">Red-team</td><td class="l">(critique) — the oracle bbox number is static-tissue-inflated; report MOTION</td><td class="l">Refiner/SSIM will disappoint most; biggest miss = no real-data metric</td></tr>
</table>
<p>The experiments <b>adjudicated</b> the disagreements: the red-team was right that motion PSNR (not bbox) is
the honest number — I led with it, and the gap survived ({head:.0f} dB). The physicist's DC loss is real but
low-headroom here ({dc_model:.0f} dB already). The architect's resize hypothesis was confirmed
({or_sh:.3f}→{nat_sh:.3f}). The pragmatist's "fancy losses are a trap" is corroborated (sharpness is the
renderer, not the loss).</p>

<h2>5. Proven improvement directions (ranked by evidence)</h2>
{img("fig_improvements.png", "Two improvements tested on the trained model, training-free. Left: native-res splat with the model's own geometry. Right: multi-draw test-time ensemble.")}

<div class="box win"><h3>A — Native-resolution splat (attacks sharpness) — PROVEN at oracle + model level</h3>
The fix for §3.2: splat the <b>native-256 intensity at native-resolution positions</b> instead of the
518-upsampled intensity. With the model's <i>own (imperfect) geometry, no retraining</i>, sharpness rises
<b class="good">{nat_model_before:.3f}→{nat_model_after:.3f}</b> (and the oracle ceiling jumps to {nat_sh:.3f}).
Cost: decouple the splat resolution from the DINOv2 518 input + a head re-fit (the head was trained against the
518 splat). <b>Counter:</b> sharpness is the <i>secondary</i> metric — this buys crispness, not the {head:.0f} dB
of motion PSNR. Ship it, but it's not the headline lever.</div>

<div class="box win"><h3>B — Multi-draw test-time ensemble (attacks motion variance) — PROVEN, free</h3>
Average <code>V_refined</code> over K independent scattered draws of the same subject/phase:
<b class="good">{md0:.2f}→{mdK:.2f} dB</b> motion at K={Kgrid[-1]} ({md_gain:+.2f}). Zero training, pure inference.
<b>Caveats:</b> it blurs (averaging lowers sharpness), it costs K× latency (a tension with "real-time"), and
the modest size confirms coverage/variance is <i>not</i> the main gap. Use it as a free baseline-raiser and as
proof that the residual error is genuine motion estimation, not variance.</div>

<div class="box win"><h3>C — Degraded-input augmentation (attacks the domain gap) — PROVEN necessary</h3>
§3.4 shows a {ks_drop:.0f} dB cliff on single-shot input. The respiratory-sim precedent (doc 05) shows the model
<i>can</i> learn to correct a simulated corruption and even generalize it (§3.4 OOD result). So adding
single-shot k-space undersampling + transient/SNR to the training augmentation is the highest-leverage
<i>transfer</i> fix — same recipe that made breathing "decisive." <b>This is the one that matters for the
stated goal.</b></div>

<div class="box kill"><h3>Ranked DOWN by evidence (don't fund yet)</h3>
<ul>
<li><b>Fancier / perceptual / adversarial losses</b> — sharpness is renderer-limited (§3.2); no objective breaks
a ceiling the resize sets. Fix the renderer first.</li>
<li><b>Data-consistency loss</b> — already ~{dc_model:.0f} dB satisfied (§3.5); little gradient to give.</li>
<li><b>Unfreeze the aggregator</b> — grad-norm 0.014; 2.8× cost for ~no movement.</li>
<li><b>k-space / complex input</b> — incompatible with the frozen RGB DINOv2 patch-embed; high cost, overlaps C.</li>
<li><b>More input planes (S&gt;12)</b> — capped by the canonical grid; coverage isn't the motion bottleneck (§3.3).</li>
</ul></div>

<h2>6. The single most important thing NOT being measured</h2>
<p>(Red-team's call, and I agree.) Every metric here — including mine — is on the model's <b>own simulator</b>,
and bbox/full PSNR is dominated by static tissue. The project cannot currently distinguish "a working
slice-to-volume reconstructor" from "a plausible-GIF generator." The missing instrument: <b>a real-data,
motion-region metric</b> — reconstruct a held-out multi-frame reference per slice from OCMR (or any real RT
cine), feed the held-out single frames, and score <code>psnr_motion</code> at matched phases against a trivial
baseline (nearest-acquired-frame, conditional-mean). Build this before chasing more in-distribution dB.</p>

<h2>7. Recommended sequence</h2>
<ol>
<li><b>Build the real-data motion metric</b> (§6) — without it, every dB is suspect.</li>
<li><b>Degraded-input augmentation</b> (Improvement C) — the transfer lever; cheap, precedented.</li>
<li><b>Attack motion estimation</b> (the {head:.0f} dB lever): low-rank/basis-DVF motion subspace on the point
head (architect's regularizer for the under-constrained through-plane Δ), and/or content-inferred cardiac-phase
conditioning. This is the dominant accuracy lever and is still largely untried.</li>
<li><b>Native-resolution splat</b> (Improvement A) — ship the sharpness win with a head re-fit.</li>
<li><b>Multi-draw ensemble</b> (Improvement B) — free baseline raiser.</li>
</ol>
<p><small>Future training experiments this report scoped but did not run (shared-cluster time): a
feature-splat + tiny 3-D decoder overfit (architect's test of breaking the warp-only wall), and a
degraded-input fine-tune to confirm C recovers the {ks_drop:.0f} dB cliff.</small></p>

<p style="margin-top:30px;color:#9aa0a6;font-size:12.5px">Reproduce: <code>tools/limits_decomposition.py</code>
(ladder + OOD + DC), <code>tools/improvements_test.py</code> (native splat + multi-draw),
<code>tools/kspace_singleshot_toy.py --model</code> (domain gap), <code>tools/make_limits_figures.py</code>
(figures), <code>_html/build_limits_report.py</code> (this page). All numbers n={N}, breathing-val,
joint refiner ckpt epoch ~76.</small></p>
</body></html>"""

open(OUT, "w").write(HTML)
print("Wrote", OUT, f"({len(HTML)//1024} KB before image embedding ~ done)")
