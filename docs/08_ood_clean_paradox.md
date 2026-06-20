# 08 — Why OOD OCMR reconstructions look "cleaner" than in-distribution val

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Observation that triggered this:** running the trained model on *real* OOD OCMR real-time data
> (`docs/06`, `_html/13`) produced beating-heart volumes that *look cleaner* than the model's own
> **in-distribution** CMRxRecon validation visuals (which show blur / speckle / "black spots"). OOD
> looking better than in-dist is a red flag — it usually means the eval setup, not generalization,
> is doing the work.
>
> **Answer (reconciled with [[10_breathing_failure_mode]] / [[11_unet_refiner_results]]): the BLUR is
> mostly the SPLAT renderer — NOT breathing — and that splat blur is just more *visible* on val than on
> OCMR.** Three things, in order of weight:
> 0. **Baseline blur = the splat (the bulk, common to both).** `docs/10` measured it: the
>    coverage-averaging splat caps sharpness at **0.74× GT even on perfectly clean inputs** — ~75% of
>    the blur — and the trained model adds ~no high-frequency over the raw splat. Both val *and* OCMR go
>    through this same splat, so it does **not** explain the val-vs-OCMR difference; it's the common
>    floor. (Black "spots" are minor at full input — ~5–6% under-covered, "blur not holes" — and the
>    refiner already fixes them, `docs/11`.)
> 1. **Resolution makes the splat blur VISIBLE on val, invisible on OCMR.** The splat discards a roughly
>    *fixed* fraction (~26%) of each input's detail. On high-res CMRxRecon (~1.3 mm) that smears visible
>    fine structure → looks blurry; on OCMR (2.16–2.84 mm, upsampled, **input slices 2.4× smoother** —
>    Laplacian 0.0058 vs 0.0142) there is little fine detail to lose → the same splat looks clean.
>    A **matched render shows breathing-OFF val is the *sharpest*** of the three; OCMR is clean-but-soft.
> 2. **Breathing adds val's *extra* ~25% blur + dark spots; OCMR's real breathing is much milder.** The
>    val pipeline shifts the *input* slices 16±8 mm and scores against the *unshifted* reference; the
>    model under-corrects. Toggling breathing ON→OFF on the *same* val subjects costs **−2.31 dB motion
>    PSNR** (the primary metric; ON 19.3 *below* the ~20.6 dB identity floor, OFF 21.6 above) — but per
>    `docs/10` this is only ~25% of the absolute blur, not its cause. **OCMR is free-breathing, so it
>    does carry *real* respiratory motion** (§3.4, measured: ~1–3 mm in-plane heart translation, SI of
>    order a few mm; one volunteer up to ~11 mm) — but that is **3–50× smaller** than the sim *and*
>    **coherent** (a bulk SI slide) where the sim is **per-slice incoherent** (`per_slot`, 30° jitter),
>    so it scatters the stack far less. "OCMR has no breathing" (an earlier phrasing) is wrong; it has
>    *small, coherent* breathing.
>
> **Two corrections recorded honestly (this analysis was wrong twice before landing here):**
> (a) My *first* answer said OCMR looks cleaner because its slices stay coplanar → *dense coverage*.
> Wrong — a `/prove-it` pass + a deep-hole metric showed OCMR coverage is actually the *holeyest*
> (deep-hole frac ~0.90 vs val-ON ~0.6 vs val-OFF ~0.05); coverage density is not the reason.
> (b) My *second* answer (after that) said the blur/spots are *entirely breathing* and resolution is a
> red herring. Also wrong — `docs/10` shows the blur is ~75% the splat, and resolution is what makes
> that splat blur visible-or-not. The headline is the **splat**, modulated by **resolution**, with
> **breathing** the smaller val-specific add-on.
>
> **Status (2026-06-20):** Reconciled with the measured blur decomposition in `docs/10`/`docs/11`.
> Breathing's val penalty is real and verified (+2.31 dB motion when removed), but it is the *minor*
> (~25%) blur term; the dominant blur is the splat, made visible by resolution. Scripts + artifacts:
> `tools/diagnose_*`, `result/{ood_paradox,ocmr_cleaner}/`.

**Date:** 2026-06-20
**Related:** [[10_breathing_failure_mode]] (**the splat is ~75% of the blur** — the key reconciliation),
[[11_unet_refiner_results]] (refiner solves the black spots, blur plateaus at the splat's ~0.69× ceiling),
[[05_respiratory_variants_results]] (breathing is decisive), [[06_ocmr_realtime_eval_data]] (the OCMR
data + optimistic-eval caveats), [[07_predicted_dvf_analysis]] (the model under-corrects breathing — slope 0.42).

---

## 1. The puzzle

The model trains 100% on simulated sparse sampling of gated CMRxRecon cine. Running it on **real**
OCMR real-time free-breathing SAX (`docs/06`) gave qualitatively coherent beating hearts (`_html/13`)
that looked **cleaner** than the model's in-distribution CMRxRecon val visuals, which carry blur,
speckle, and dark "black-spot" patches. Since OCMR is genuinely out-of-distribution (1.5 T patients,
R≈9 k-t recon texture, different vendor/contrast), it should — if anything — look *worse*. So the
clean OCMR look had to come from the eval setup, not from the model generalizing unusually well.

The model under test is the **breathing-sim, z-only, aggregator-finetuned** checkpoint
(`218747856_mri_volume_resp_allphases_aggft_z_no_t`, the `docs/05` best recipe), **no refiner**.

## 2. The wrong first answer (and how it was caught)

My initial conclusion was: *OCMR has no synthetic breathing → its slices stay coplanar at their true
z → dense, uniform splat coverage → clean; the breathing-corrupted val slices get scattered →
speckled coverage (the "black spots") → looks worse.* I asserted this with high confidence.

A **`/prove-it`** pass (4 independent reviewers + adversarial verification + a GPU re-run) refuted the
OCMR half of it:

- The coverage metric I used (`splat_cleanliness.hole_frac`) was **non-monotonic and blind to deep
  holes**: a voxel that *should* be covered but has near-zero coverage falls below the body threshold
  and is *excluded* before it can be counted as a hole. It also degenerates when faint trilinear
  bleed dominates the positives. So its absolute numbers are unreliable.
- A **deep-hole-sensitive** re-measure (bbox including zeros, central populated plane) showed OCMR
  coverage is the **holeyest** of the three, not the cleanest:

  | condition | deep-hole frac (in-plane) |
  |---|---|
  | val breathing OFF | 0.002 – 0.06 |
  | val breathing ON | 0.43 – 0.99 |
  | **OCMR (ED)** | **0.89 – 0.92** |

So "OCMR is clean because dense coverage" is **false**. (`/prove-it` also found a real bug in the
diagnostic — OCMR was queried at `target_t=0.0`, which normalizes to *phase 6*, not ED=`-1.0`; fixed.
The bug did not change the qualitative coverage picture, and the user-facing OCMR report
`_html/13`/`eval_ocmr_inference.py` always used the correct phase.)

**Methodology note for future agents:** threshold-on-coverage cleanliness metrics are treacherous
here — the splat coverage field has bright central peaks and empty gap-planes that distort any
`x*max` or `x*median` threshold. Prefer (a) the **primary motion PSNR** against a reference, and (b)
**image-domain sharpness**, over coverage-field heuristics.

## 3. The correct answer — splat blur (dominant), made visible by resolution, plus a breathing add-on

### 3.0 The baseline: the splat is the dominant blur (common to both) — see [[10_breathing_failure_mode]]

Before the val-vs-OCMR difference, note the *absolute* blur source. `docs/10` measured (resp model,
`tools/measure_sharpness.py`) that the coverage-averaging splat caps sharpness at **0.74× GT even on
perfectly clean inputs** (~75% of the blur), and the trained model adds **~no** high-frequency over the
raw splat — the point head fixes *position*, the splat caps *detail*. **Both val and OCMR use this same
splat**, so it is the common floor, not the differentiator. It also means the absolute "blur" is mostly
the renderer, *not* breathing — breathing is only the ~25% on top (0.74→0.65× GT). The remaining two
sections are about why that common splat blur *looks* worse on val.

### 3.1 Breathing adds val's extra blur + dark spots (primary metric)

The val pipeline applies the respiratory sim **in val** (deterministic per `seq_index`): a per-slice
SI+AP shift, **amplitude 16 ± 8 mm** (8–24 mm), `per_slot=true`, 30° direction jitter, applied by
deform-then-reslice to the **input slices only** while the reference/`phases` stay unshifted. The
model must *undo* this shift but **under-corrects** (predicted ‖Δz‖ 1.96 mm ON vs 0.24 mm OFF; cf.
`docs/07` fit slope 0.42).

Controlled test — **same val subjects, only breathing toggled** (so OOD/everything else is held
constant), scored on the **project-primary motion PSNR** (dynamic voxels, swing > 0.05 across phases):

| metric | breathing ON | breathing OFF | cost of breathing |
|---|---|---|---|
| **motion PSNR** (primary) | **19.31 dB** | **21.62 dB** | **−2.31 dB** |
| full PSNR | 28.09 dB | 32.50 dB | −4.41 dB |
| full L1 (`V_gt>0.05`) | 0.045 | 0.026 | +73% error |
| predicted ‖Δz‖ | 1.96 mm | 0.24 mm | — |

Per-subject motion-PSNR drops: +1.78, +1.53, +3.98, +3.41, +0.86 dB — **always positive**. Context:
the motion *identity* baseline is ~20.6 dB ([[feedback]]/`baseline_identity.json` convention), so
**breathing pushes the model below do-nothing on the voxels that matter** (ON 19.3 < 20.6 < 21.6 OFF).
The val visuals the user compared against had breathing ON; OCMR's *real* breathing is much milder and
coherent (§3.4), so OCMR largely skips this penalty.

This confirms both user hypotheses, now on the primary metric: *"breathing sim is applied to val"* and
*"a breathing-sim model on non-breathing input does better."* (The latter is the `docs/05` "≈free on
the clean task" result, here quantified as +2.31 dB motion.)

### 3.2 Resolution makes the splat blur visible on val, invisible on OCMR

This is **not** an independent "OCMR is smoother" cosmetic point — it is *why the §3.0 splat blur looks
worse on val.* The splat discards a roughly fixed fraction (~26%) of each input's detail (§3.0). On
high-res CMRxRecon (~1.3 mm) that smears *visible* fine structure → reads as blur; on low-res OCMR
(2.16–2.84 mm, upsampled) there is little fine structure to lose → the same splat looks clean.

Note: "resampled to 1.4 mm" is a **grid** choice, not a resolution gain — upsampling 2.16 mm data onto
a 1.4 mm grid adds no information; the image stays smooth on a finer grid. So OCMR carries detail only
to ~2.16 mm. Image-domain sharpness (mean Laplacian magnitude over anatomy) confirms the gap:

| | input-slice sharpness | V_canon sharpness |
|---|---|---|
| val (CMRxRecon, ~1.3 mm) | **0.0142** | 0.038 (OFF) |
| OCMR (2.16 mm → upsampled) | **0.0058** | 0.032 |

**OCMR inputs are 2.4× smoother.** With little high-frequency content to begin with, the splat's
fixed detail-loss is imperceptible → output *reads as clean*. This is a property of the data + the
splat interaction, not superior reconstruction.

### 3.3 The matched render (controls presentation)

Rendering all three identically (mid-z `V_canon`, per-image window, **no V_gt/diff panel**;
`result/ocmr_cleaner/matched_render.png`):

- **val breathing ON** — grainiest: speckle + streak artifacts (the breathing penalty, visible).
- **val breathing OFF** — clean **and the most detailed** of the three.
- **OCMR** — clean but **soft/low-detail** (the resolution effect).

So when the user compared OCMR against the breathing-corrupted val they had seen, OCMR won on *both*
axes — but **val-OFF is actually the sharpest**, i.e. OCMR is not "better reconstructed," it is
**mildly-corrupted** (small, coherent real breathing — §3.4) **and** smoother (lower res). The original
"val looks bad" impression also had a **presentation** component (val shown all-z + signed-diff, which highlights every error and the
low-coverage edge planes; OCMR shown as a mid-z GIF with no reference) — real but secondary; the
matched render removes it and val-OFF still looks clean.

### 3.4 How much REAL breathing does OCMR have? (measured — `tools/measure_ocmr_breathing.py`)

OCMR *is* free-breathing real-time data, so it carries **real** respiratory motion — the earlier
"OCMR has no breathing" only ever meant "no *synthetic* sim applied." Measured directly: per slice's
real-time cine, rigid-register each frame to the temporal median (skimage phase cross-correlation,
subpixel — **validated to recover known shifts to 0.0 px**), giving in-plane displacement over time;
separate the **slow respiratory drift** from the fast cardiac oscillation by a one-cardiac-cycle
low-pass, on a heart-centred crop. (In-plane is a **lower bound** on true SI amplitude — SAX is tilted
~20–45° off SI, so SI ≈ in-plane / sin(tilt) ≈ 1.4–3× larger. Cardiac registers as ~0.2 mm *translation*
because cardiac motion is local *deformation*, not translation — a correctness check, not a failure.)

| subject | in-plane resp ptp (median / max) | ≈ true SI |
|---|---|---|
| **us_0084** (volunteer, 128 fr, 7.4 cardiac cyc) | **3.0 / 11.5 mm** | ~4–23 mm |
| us_0173 / 0174 / 0183 (patients, 61–63 fr) | 0.7–0.9 / 1.6–2.3 mm | ~1–7 mm |
| 6 short-cine patients (30 fr, <1 full breath) | 0.1–0.6 mm | <~2 mm (under-sampled) |

**Reads:** (1) OCMR breathing is **real and subject-dependent** — the *volunteer* breathes meaningfully
(~3 mm in-plane median, up to ~11 mm), the 9 *patients* breathe little (<1 mm median; likely shallow /
coached-still). (2) For 9/10 subjects it is **3–50× smaller than the 16±8 mm sim.** (3) Crucially the sim
is **per-slice incoherent** (`per_slot=true` + 30° direction jitter → each slice an independent random
breath), whereas real breathing is a **coherent bulk SI slide** — at equal amplitude, incoherent scatter
distorts a *stacked* volume far more than coherent motion, which plausibly explains why even us_0084's
real ~3 mm breathing looks cleaner than the sim's. *(The amplitude is measured; the coherence claim is
grounded in the sim config + physics but not separately quantified — an open item.)*

So the honest correction to the headline: OCMR is **not** breathing-free; it has *small, coherent* real
breathing, well below the aggressive incoherent sim — which keeps the §3.1 conclusion (breathing is the
smaller, val-specific term) intact, just no longer overstated as "OCMR has none."

## 4. What is and isn't established

- **Established:** the dominant blur is the **splat renderer** (`docs/10`: ~75%, caps at 0.74× GT on
  clean inputs), common to val and OCMR; **breathing** adds val's extra ~25% blur + dark spots
  (−2.31 dB motion when toggled OFF, primary metric); **resolution** makes that splat blur visible on
  val (2.4× sharper inputs) and invisible on OCMR; val-OFF is the **sharpest** reconstruction; the
  coverage-density mechanism is **false**; black spots are minor at full input and refiner-solved
  (`docs/10`/`docs/11`); **OCMR carries real but small breathing** (§3.4: ~1–3 mm in-plane, 3–50× below
  the sim, and coherent vs the sim's per-slice-incoherent shifts).
- **Not claimed:** that breathing is the main cause of the blur (it is ~25%, `docs/10`); that OCMR is
  breathing-*free* (it is not — §3.4); that the **coherence** difference's *magnitude* is quantified
  (grounded, not measured); that OCMR is *quantitatively* cleaner than val — there is **no shared
  reference** (OCMR is prospectively undersampled, no V_gt), so OCMR cannot be scored on PSNR at all.
  "Cleaner look" ≠ "better reconstruction"; here it means *mildly-corrupted (small coherent breathing)
  + smoother (lower-res hides splat blur)*.
- **Caveat carried from `docs/06`:** the OCMR inputs are themselves k-t-reconstructed (optimistic,
  many-frames-per-slice), not faithful single-shot. That is a separate open item, orthogonal to this.

## 5. Provenance & reproduce

- `tools/measure_ocmr_breathing.py` — measures OCMR's **real** respiratory amplitude (§3.4): per-slice
  frame registration (validated to 0.0 px), cardiac/respiratory frequency separation, heart-crop.
  Artifacts → `result/ocmr_cleaner/breathing_amplitude.json`.
- `tools/diagnose_ood_clean_paradox.py` — builds the val dataset standalone (process-group + Hydra
  compose, `respiratory.enable=true`), `val_forward(seq, breathing)` toggles the sim, computes motion/
  full PSNR + coverage metrics + the predicted-DVF magnitude. Artifacts → `result/ood_paradox/`.
- `tools/diagnose_ocmr_cleaner.py` — the Part-2 settle: motion PSNR (ON/OFF), Laplacian sharpness
  (val vs OCMR, input + V_canon), and the matched render. Artifacts → `result/ocmr_cleaner/`
  (`summary.json`, `matched_render.png`).
- Run (GPU): `PYTHONPATH=training:. micromamba run -n svr python tools/diagnose_ocmr_cleaner.py`
  (load the checkpoint from local `/tmp` — GPFS load is ~20 min, `/tmp` ~1 min).
- Model: `218747856_mri_volume_resp_allphases_aggft_z_no_t` (resp, z-only, aggft, no refiner).
