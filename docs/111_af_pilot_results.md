# 111 — Arrhythmia: pilot (n=4) then full cohort (n=38) — results and the coverage-not-timing finding

> **TL;DR & takeaway**
>
> ⚠️ **CORRECTION (docs/112, measured — read it before quoting any EF number below).** The EF
> headline as written is **not supported by a significance test**. Fetal 4D's EF advantage over
> VGGT is significant under `regular_frozen` (+3.61 pp, 95 % CI [+0.17, +6.87], p = 0.007) but
> **indistinguishable from zero under `af`** (+1.01 pp, CI [−2.13, +4.01], p = 0.50), and the
> *collapse* itself is **not significant** (−2.61 pp, CI [−6.79, +1.70], p = 0.22). So "survives
> AF but collapses 3.6×, still wins barely" must be read as "**significant under a regular rhythm,
> no longer detectable under `af`**" — the two cohort estimates differ in significance, which is
> not the same as differing from each other. The **PSNR** mechanism results below are unaffected
> and are strongly significant (af oracle gap −0.815 dB, CI [−1.14, −0.50], p = 2.8e−05; the
> reversal −1.106 dB, p = 2.6e−06; coverage-balanced flips it to +0.166 dB, p = 0.005).
> Separately, docs/112 §1 **withdraws** the bin-imbalance correlation of §3a (null within `af`:
> r = −0.152, p = 0.40, with 4.5× imbalance variation available) — the coverage mechanism now
> rests solely on the §3d/§10 intervention, which is enough. docs/112 §3 closes the
> degenerate-subject question: chance beat draws, **not** the window-open-phase bug and not a
> subject trait.
>
> **Full-cohort update (§10): the pilot's findings replicate and strengthen at n=33 (paired), not
> just n=3.** Giving Fetal CMR 4D the TRUE per-frame AF timing makes it WORSE, not better — PSNR
> gap −0.81 dB (pilot: −0.63), EF gap +4.43 pp (worse than self-gated's own +2.68 pp under `af`).
> Fixing bucket COVERAGE while keeping true timing flips this positive on both metrics — EF
> 7.91 pp, beating self-gated (8.94) outright. **The real answer this project was built to find:
> Fetal 4D's EF advantage over VGGT survives AF but collapses from 3.61 pp to 1.01 pp** — still
> wins, barely. Visually confirmed (§10.4, GIFs): oracle-raw shows visible speckle where buckets
> are under-filled; self-gated/oracle-balanced are clean. The elevated degenerate-subject rate
> (5/38, not ~1/38 predicted) is real model behavior, confirmed by 5000-trial Monte Carlo (15.4%
> vs measured 13.2%) — the design-time estimate was wrong, not the simulator.
>
> **Original pilot (n=4 subjects, below) — same finding, first discovered here.** 4 cohorts × 4
> subjects × 3 arms (`fetal_cmr_4d`, `fetal_cmr_4d_oracle`, VGGT `final518_diff1000_ep300`), 32
> fetal recons + 16 VGGT runs. **Gating collapses as designed** — the structural (within-slice) θ
> error goes 0.000 → 0.000 → 0.486 → 1.328 cine frames and **62–75 %** of input images land in the
> wrong reconstruction bin (the range is admissible-offset dependent, §2a). **But the damage barely
> reaches the image metrics** (self-gated −0.60 dB under `af`) and **the oracle arm REVERSES**:
> under `af`, handing Fetal 4D the TRUE per-frame timing makes it **0.63 dB WORSE**, on all 3
> non-degenerate subjects independently. That reversal is the robust result — it survives including
> P004 (at 0.46 dB instead of 1.09 dB) and is **not** explained by window span. **Its cause was a
> confounded hypothesis — now settled by a direct intervention (§3d), and reconfirmed at full scale
> in §10.** The correlational bin-imbalance measure was near-totally confounded with cohort identity
> (a plain "is this `af`" dummy predicted the gap *better*, r = −0.891 vs −0.832). So a fourth gate
> arm was built: true per-frame θ, but rank-quantized per slice to force the same one-image-per-bin
> coverage the self-gated method gets for free. Restoring coverage **flips the sign** — raw-oracle
> trails self-gated by −0.63 dB, balanced-oracle **leads** it by **+0.26 dB**, positive or flat on
> all 3 subjects. **Coverage, not timing accuracy, is the mechanism**: true timing never hurt
> reconstruction — the reversal was an artifact of uneven phase-bin occupancy that came bundled with
> representing that timing honestly.
> VGGT is flat across rhythm (+0.03 dB under `af`), though `af` differs from `regular` in **more than
> rhythm** (§5). **EF is uninterpretable at n=3** — resolved at full scale in §10.3. Pipeline
> correctness verified four ways with seven fault injections (`tools/verify_af_pipeline.py`), then
> re-audited by 3 independent reviewers whose findings are folded in below — including two claims of
> an earlier draft that are now **withdrawn** — and re-reviewed again before the full-cohort launch
> (§9.1).

**Provenance.** Branch `exp/af-sim`, commits `66e6b6e` (code) → `bb45672` (docs/110) → `4899364`
(pilot tooling) → `c0e958d` (verification). Recons: SLURM `61545022` (6 tasks), `61545023`,
`61545328`, all COMPLETED, 0 failed, 54–71 min each at 4 CPUs. Report: `tools/af_pilot_report.py`;
raw blob `temp/af_pilot_report.json`.

---

## 1. Correctness before results

`tools/verify_af_pipeline.py` — every check paired with fault injections that must fire, because a
checker nobody has broken on purpose can pass by being vacuous.

| check | result | faults fired |
|---|---|---|
| A. oracle gates carry the truth (θ = 2π·pos/T + one global const, θ₀₀ at the 710 residue) | 16/16 | 4/4 |
| B. `ref_phase` readout selects the right volume (heredoc extracted from the SHIPPED script, driven on a synthetic cine whose phase k holds value k) | 32/32 | 3/3 |
| C. `regular_frozen` replicates the live bundle | 4/4, input max\|diff\| = **0.000e+00**, GT byte-identical permutation | — |
| D. analysis decomposition math on synthetic θ | 5/5, incl. 200-trial uniform-ramp ⇒ residual 0 | — |

Also verified: all 32 recon stamps carry `readout: ref_phase` with the correct `gate_arm`
(16 self-gated / 16 oracle); all 16 VGGT runs are fresh at commit `bb45672`; the §11.6 slot-seeding
fix works (`seq_index_used` identical across all 5 cohorts, and equal to `name_seed("cmrx2024", …)`).

## 2. The gate degrades exactly as designed

Self-gated θ measured against the **oracle** gate, global circular mean removed (a constant shared
by every image only relabels which bin is "phase 0" and the readout undoes it). Units: cine frames.

| cohort | mean \|err\| | rms | per-slice const | **within-slice residual** | misbinned | collisions |
|---|---|---|---|---|---|---|
| `regular_frozen` | 0.910 | 1.533 | 1.533 | **0.000** | 31.4 % | 3.8 % |
| `regular` | 0.863 | 1.467 | 1.467 | **0.000** | 31.4 % | 3.9 % |
| `hrv` | 1.166 | 1.745 | 1.694 | **0.486** | 54.9 % | 5.7 % |
| `af` | 1.739 | 2.329 | 2.273 | **1.328** | **74.6 %** | 6.4 % |

Two distinct failure modes, and only the second is structural:

- **Per-slice constant** = the nnU-Net ED-anchor error. Present already at 1.47 frames on a
  *periodic* rhythm, so it is a pre-existing limitation, not an AF effect. Not harmless: it
  mis-registers slices against each other — precisely what the paper's inter-slice sync fixes and
  our single-orientation SAX adaptation cannot (docs/34/35).
- **Within-slice residual** = the cost of the constant-heart-rate assumption itself. **Exactly 0**
  for a rhythm periodic **at the assumed rate** (12 frames/beat) — both θ are then uniform ramps, so
  their difference is a per-slice constant by construction; the `regular*` rows therefore self-check
  the decomposition, and 200 random-anchor trials confirm it to 8.7e-07. ⚠️ **Not** zero for a
  rhythm periodic at a *different* constant rate: synthetic `rr ≡ 1.2` gives 0.575 frames, larger
  than `hrv`'s measured 0.486. So the residual conflates beat-to-beat variation with a wrong
  constant R-R estimate. Harmless here (all arms have mean beat 1.0) but material in a real
  deployment, where docs/110 §3 measures the x-f estimator at 3–8 % off.

⚠️ **CORRECTION — an earlier draft of this doc claimed the error is "not concentrated at beat
boundaries (1.25× under `af`, 0.98× under `hrv`), i.e. no concentration at all". That claim is
WITHDRAWN.** It was measured on the **total** error, which is dominated by the per-slice anchor
constant — a quantity that is uniform across frames *by definition* and therefore dilutes any
boundary effect. Re-measured on the **within-slice residual**, which is the component §2 is about:

| cohort | total-error ratio | **residual ratio** |
|---|---|---|
| `regular_frozen` / `regular` | 1.07 / 1.09 | 0.68 / 0.60 |
| `hrv` | 0.98 | 1.01 |
| `af` | 1.25 | **1.98** |

Boundary frames under `af` carry ~2× the structural error of interior frames. The concentration is
real; the original test simply could not see it.

### 2a. The two headline percentages are offset-dependent

`af_pilot_report.gate_stats` removes the **global** circular mean, justified because a global
constant cancels in the `ref_phase` readout — but that also makes the constant unidentifiable, so
the statistic depends on which admissible constant is chosen. Sweeping them: mis-binned under `af`
is **74.6 % as reported but 61.7 % at the best-aligned offset** (`hrv` 54.9 % vs 51.5 %; `regular`
31.4 % either way). The mean resultant length under `af` is only 0.502, so the circular mean is a
weak summary of a broadly spread error. **Quote 62–75 %, not 75 %.** The ordering across cohorts is
unaffected. Likewise the decomposition is **not additive** — `res = ang(e − off)` re-wraps, so
`const² + resid² ≠ rms²` (ratio up to 1.88 on `af` subjects); read the two parts as diagnostics, not
as orthogonal components.

## 3. …but the damage barely reaches the images, and the oracle REVERSES

Paired `psnr_unit_peak` over the 3 non-degenerate subjects (P004 dropped everywhere — §5).

| cohort | self-gated | oracle | **oracle gap** | VGGT |
|---|---|---|---|---|
| `regular_frozen` | 24.18 | 24.64 | **+0.46** | 25.04 |
| `regular` | 23.96 | 24.35 | **+0.39** | 24.77 |
| `hrv` | 24.02 | 24.40 | **+0.37** | 25.09 |
| `af` | 23.58 | **22.95** | **−0.63** | 25.07 |

Δ vs `regular_frozen`: self-gated −0.22 / −0.15 / **−0.60**; oracle −0.29 / −0.24 / **−1.69**;
VGGT −0.27 / +0.05 / **+0.03**.

Per subject under `af`, the oracle is worse on **all three independently**: P016 −0.40,
P017 −1.07, P046 −0.42. Not one subject's fluke.

### 3a. Why: bin occupancy, measured

The engine pools images into 12 phase bins by θ. A uniform θ ramp puts **exactly D images in every
bin, by construction**. True AF θ does not — holds pile images up:

```
af / P017   self-gated: [12 12 12 12 12 12 12 12 12 12 12 12]   std/mean 0.000
af / P017   oracle    : [12 10  9 25  9  8 11 12 13 11 11 13]   std/mean 0.350
```

| cohort | oracle bin imbalance (std/mean) | oracle gap |
|---|---|---|
| `regular_frozen` / `regular` | 0.000 | +0.46 / +0.39 |
| `hrv` | 0.139 | +0.37 |
| `af` | 0.281 | −0.63 |

`r(bin imbalance, oracle gap) = −0.832` (n = 12). Fit: `gap = −3.51 × imbalance + 0.52 dB`.

### 3a-bis. Why the correlation alone was NOT sufficient — before §3d settled it

Four independent problems with the correlational argument, all found by adversarial review and each
verified by recomputation. §3d then resolved the question directly with an intervention rather than
a correlation — but the problems below are why that follow-up was necessary in the first place:

1. **A plain cohort dummy predicts better than the mechanism does.** `r(af_dummy, gap) = −0.891`
   vs imbalance's **−0.832**. Fitting both: intercept +0.458, af-dummy −0.773, imbalance **−1.123**
   (down from −3.51). Imbalance carries almost no information beyond "is this the `af` cohort".
   The confound is not partial — in this design it is near-total.
2. **The only out-of-range test of the fit falsifies it.** P004/`af` has imbalance **1.044** (3.7×
   the af mean, occupancy `[40 6 6 6 5 5 5 6 7 8 8 6]`). The fit predicts **−3.15 dB**; the observed
   gap is **+0.87 dB** — a 4.0 dB miss in the wrong direction. It is excluded for independent
   reasons (degenerate GT), but it is the single most extreme datum available and it disagrees.
3. **"Holds within `af` alone" was one subject.** The three af points are P016 (0.238, −0.40),
   P046 (0.254, −0.42), P017 (0.350, −1.07). P016 and P046 nearly coincide, so the within-af trend
   is a two-point comparison driven entirely by P017, and adding P004 reverses it.
4. **"Predicted −0.47 vs observed −0.63" is a fit residual, not a validation** — the af points are
   in the regression. 0.16 dB is 25 % of the effect.

### 3b. The competing explanation is NOT refuted — but it is not supported either

⚠️ **CORRECTION.** An earlier draft claimed `hrv` "refutes" the readout-blending hypothesis because
its mean blend weight matches `af` (0.184 vs 0.180) while its gap stays positive. **That argument is
withdrawn**: the `af` mean averages a bimodal set — P016 0.200, P017 **0.009**, P046 0.331 — versus
`hrv`'s homogeneous (0.171, 0.225, 0.155). Cohort-mean equality with n=3 per cell controls nothing.

The *correct* version is stronger and points the same way. **Within `af`, the least-blended subject
has the worst gap**: P017 blend 0.009 → −1.07, P016 0.200 → −0.40, P046 0.331 → −0.42. Blending
hypothesis predicts the opposite ordering. So: **not supported, and the within-`af` ordering runs
against it** — but not "refuted".

### 3c. Restating the mechanism at the precision it was actually measured

Two corrections to how §3a was phrased:

- **The engine does not hard-bin.** `ReconstructionCardiac4D.cc:567-586` applies a Gaussian temporal
  PSF (σ = dtrad/2.355, FWHM ≈ 1 cine frame); every image contributes to every output phase with a
  smooth weight. Recomputing imbalance from the engine's own weight sums leaves the picture
  unchanged and slightly stronger (`r = −0.895`), so the conclusion is safe — but the mechanism is
  about **weight**, not counts.
- **The sharper quantity is per-(slice, phase) coverage, not a global histogram.** Under self-gating
  every (slice, output-phase) cell holds exactly one image at full weight — verified: min soft
  weight 1.125, **0.0 % empty cells, every cohort**. Under the `af` oracle, **9.8–18.1 % of cells are
  empty** with min weight 0.000, so some slices have no data at some phases and the engine must fill
  them from regularisation. That is *coverage holes*, which is actionable; `r(empty-cell fraction,
  gap) = −0.815`, statistically indistinguishable from imbalance's −0.832 at n=12. **The pilot
  cannot separate the two.**

### 3d. The follow-up that settles it — RUN, and it points at coverage

Built a third gate arm, `fetal_cmr_4d_oracle_balanced` (`fetal4d_gate.py --oracle --balanced`,
`rank_quantize`): per slice, the 12 true positions are mapped onto a **bijection** with the 12
output bins by rank order — preserving the true relative ordering of the frames but forcing exactly
one image per bin, the same coverage guarantee the self-gated method gets for free from its uniform
rate assumption. Verified before spending any compute: hard-bin coverage matches self-gated exactly
(std/mean 0.000, 0 empty cells, vs the raw oracle's 0.238–0.350) on all 3 subjects; the on-disk gate
round-trips to 4.9e-07 against an independent recomputation; the plain self-gated and plain-oracle
code paths hash byte-identical before and after the change. Ran on the 3 pilot `af` subjects
(SLURM, `jjparkcv98`, sharded one job per subject, all COMPLETED, 770–947 s each).

| | self-gated | oracle (raw, uneven coverage) | **oracle (balanced coverage)** |
|---|---|---|---|
| P016 | 23.47 | 23.06 | 23.42 |
| P017 | 22.04 | 20.97 | 22.31 |
| P046 | 25.25 | 24.82 | 25.80 |
| **mean** | **23.58** | **22.95** | **23.84** |

Restoring coverage while keeping true timing **flips the sign**: raw-oracle trails self-gated by
−0.63 dB (the reversal); balanced-oracle **leads** self-gated by **+0.26 dB**, and per-subject is
flat-to-positive on all three (−0.05 / +0.27 / +0.56) — no subject where accurate timing still hurts
once coverage is fixed. Fixing coverage alone, holding timing accuracy at "true", is worth **+0.89 dB**
— more than enough to explain the original −0.63 dB gap and reverse it.

**Conclusion: coverage, not timing accuracy, is the mechanism.** True timing does not hurt Fetal 4D
under AF — it was the accidental byproduct (uneven bin occupancy) of representing that true timing
without also respecting the constant-rate assumption's coverage guarantee. This is a single
disambiguating run (n=3, one balanced variant), not a large-n confirmation, but it directly settles
the question §3a-bis could only leave open: the correlational design was confounded, this
intervention is not.

## 4. VGGT is flat, as pre-registered

Paired: 25.04 / 24.77 / 25.09 / 25.07. The only real cost is **frame-wise breathing (−0.27 dB)**;
`hrv` and `af` are within noise of `regular_frozen`. Expected — VGGT is *told* the target instant
(`timesteps[0,0] = t`), so it never infers timing and rhythm irregularity cannot damage that step.

## 5. Traps this pilot confirmed in real data

**`af` does NOT open its window at the same stored phase as the other arms — docs/110 §11.5's
invariant is FALSE.** `simulate`'s `t0` (`build_af_bundle.py:228`) is a *time* offset
`frac·rr[BURN]` into beat 3. Under `pos_compress` (uniform stretch) that maps to phase `frac·T`
exactly, so `regular*`/`hrv` open at `(-roll[z]) % T` with deviation **0.00**. Under `pos_physio`
phase advances at a fixed rate, so the open phase is `min(p_start[BURN] + frac·rr[BURN]·T, T)` —
measured deviation up to **5.52 phases** on `af`. Consequence: **16.4 % of `af` frames sit parked at
the full-ED hold (0 % in every other arm)**, so `af` differs from `regular` in **window-open phase
as well as rhythm** and cross-cohort deltas are not rhythm-only. The behaviour itself is defensible
(a free-running camera opens at a uniformly random *time* inside a beat), but the asserted control
is not. **This does NOT touch §3's oracle gap**, which is within-cohort on identical input —
verified: the two arms' `stack4d` differ only by `assemble()`'s slice-0 frame roll, and undoing it
gives max|diff| = 0.000e+00 with all non-slice-0 planes bit-identical.

**P004 under `af` is degenerate and must never be averaged in.** **55 of 66 GT pairs are
byte-identical**, position span 0.97 frames (vs 9.44–11.00 for everyone else), GT EF collapses
70.6 % → **14.4 %**. ⚠️ Its *cause* was mis-attributed: docs/110 §11.8 and an earlier draft here
blamed "a long effective beat with the camera window opening inside its diastasis". The real cause
is the `frac·rr[BURN]` conversion above — with `roll[4] = 8 ⇒ frac = 1/3`, every other arm opens at
phase 4 while `af` opens at phase **12.0** and stays there. Signature: `rhythm.duplicate_targets`;
threshold `dup_pairs > 5` is **not fitted** — measured values are 55, 1, and 0 elsewhere, so any
threshold in [2, 54] gives the identical exclusion.

⚠️ **CORRECTION — two supporting statistics in an earlier draft were VGGT-only and reverse on the
fetal arms:**
- *"its `af` PSNR is 30.46, the highest of its four cohorts"* — that is **VGGT**. Per-cohort
  `[rf, regular, hrv, af]`: fetal self `[30.15, 29.05, 28.99, 28.12]` → `af` is the **LOWEST**;
  fetal oracle `[29.59, 28.49, 29.04, 29.00]` → 2nd. For the two arms the headline is about, the
  claimed PSNR inflation **does not occur**. The exclusion remains justified on the GT evidence.
- *"window-span confound r = −0.445"* — also **VGGT only**. Per arm: VGGT −0.445, fetal self
  **+0.567** (opposite sign to the stated "less span ⇒ easier" mechanism), fetal oracle +0.067.
  It is not a property of the data.

**The span confound does NOT undermine §3.** The oracle gap is a within-(cohort, subject) paired
difference between two arms sharing input, GT and engine, so span cancels: over the 12 non-degenerate
rows `r(span, gap) = +0.384`, and OLS `gap ~ imbalance + span` leaves imbalance at −3.57 with span at
−0.03. Decisively, **af/P017 — the worst gap (−1.07) — has the maximum 11.00-frame span**, the
opposite of what the span mechanism predicts. The span warning applies to cross-cohort *absolute*
PSNR claims only.

**GT EF is otherwise unmoved** (64.1 → 64.1 → 63.3 → 63.7 over the 3 non-degenerate subjects), so the
target does not shift under the method's feet and the comparison stays apples-to-apples. Note this
partly undercuts docs/110 §7's motivation: the GT-protocol change is still *correct*, but it only
bites for the rare degenerate window (1/38 per the design-time cohort pass).

**P046: a live instance of the documented 7/38 readout disagreement.** Its nnU-Net ED anchor for the
reference slice is off by exactly one frame (true ED at input frame 7, anchor says 8) because its LV
area at stored frames 11 and 0 differs by only 4.1 % — argmax is ambiguous. `gt_ed_roll` would have
corrected this for free; `ref_phase` does not. The equivalence `ed_ref ≡ (gt_ed + roll_ref) mod T`
holds exactly for the other 3/4 subjects. This is docs/110 §11.2 behaviour, **not a bug** — and P046
is the same ED≠0 subject behind `pos_physio`'s disclosed limitation.

## 6. EF: no interpretable signal at n=3

`regular_frozen` vs `regular` differ only in frozen vs frame-wise breathing, so their EF gap bounds
how much EF moves under a perturbation that is *not* rhythm: **mean 4.3–6.2 pp, max 15.0 pp**
(fetal self-gated, P046: 28.5 → 43.5), on the same paired n=3 as the table below.

⚠️ Called a "pure noise floor" in an earlier draft — **wrong in kind**. §4 establishes frame-wise
breathing as a genuine −0.27 dB systematic effect, so this gap mixes a real effect with noise. It is
a **perturbation-sensitivity bound**, not a noise floor. (The §6 conclusion is unaffected, and if
anything strengthened: even a bound that generous swamps the rhythm signal.)

Paired EF MAE (pp) over the same 3 subjects:

| cohort | fetal self | fetal oracle | VGGT |
|---|---|---|---|
| `regular_frozen` | 15.70 | 9.71 | 14.47 |
| `regular` | 9.60 | 10.91 | 18.73 |
| `hrv` | **3.37** | 9.11 | 13.54 |
| `af` | 15.04 | **6.33** | **11.51** |

Every arm's best cohort is a *different* one and none is monotone in irregularity — the signature of
noise. Spreads nominally exceed the noise floor, but that floor is itself an n=3 estimate and this is
not a significance test. **No EF conclusion is supportable from this pilot.** EF was the original
motivation for the whole experiment (docs/110 §1), so the full cohort is required before anything can
be said about it.

## 7. Defects and risks found

1. **`run_vggt.py:485` records a wrong provenance label.** `seq_index_basis` is written as
   `sha256('{ds_name}/{subject}')` using the cohort directory, while the value comes from
   `name_seed(man["source"], …)`. For `cmrx2024_af`/P016 the label names an input that would give
   1814439323 while the recorded value is 1165133435 (the base). **Provenance only, no result
   impact** — the §11.6 fix updated the computation but not its label. One-line fix; deliberately
   NOT applied during the pilot so every run shares one binary.
2. **`ef_dice score` merges into the LIVE campaign `_ef/<arm>.json`.** Verified additive and
   non-destructive — the published rows survive exactly (5.65/6.23/7.07/7.86/6.48, n=180, weighted
   **6.62**) and `per_subject` grew 180 → 196. But campaign files now carry pilot rows; the full run
   will add more. Consider a `--out` override for rhythm cohorts.
3. **`af_pilot_report.py` read the wrong EF keys** on first write (`per_cohort`/`lv_ef_mae` vs the
   actual `aggregate`/`breath_ef_mae_pct`), silently printing "no data". Fixed to read `per_subject`
   so the degenerate-subject exclusion applies consistently.
4. **n=3 is too small for EF** and marginal for the PSNR mechanism claims.
5. **`af_pilot_report.py` tables 4/5 were UNPAIRED** — they dropped degenerate rows per
   (cohort, subject), so `af` averaged n=3 while the others averaged n=4, and P004 scores 28–30 dB
   against everyone else's 21–26. The tool therefore printed `af − regular_frozen` as **−2.09 dB**
   (fetal self) and **−1.11 dB** (VGGT) against the hand-computed paired −0.60 and **+0.03** — a
   **sign flip on VGGT**, and it flipped the fetal EF direction too. The doc's §3/§6 were always the
   paired hand-computation and are correct, but a reader re-running the cited tool got different
   numbers. **FIXED**: the tool now intersects the usable subject set across all cohorts first and
   prints it. This would have recurred at 38 subjects, where ~1 degenerate subject again lands in
   `af` only.
6. **Reviewer findings in the shipped code.** ⚠️ **The `GATE_ARM`/`METHOD` guard is now FIXED**
   (`run_baselines.py:100`, see §9) — kept here for the record of what the pilot ran with. It used
   to admit a mis-attributed `_oracle` arm when `METHOD` was set and `GATE_ARM` was not — the pilot
   itself was unaffected because `af_pilot_fetal.sh` always set both and all 32 stamps verify;
   `resp_diag` reporting the **frozen**
   displacement on frame-wise cohorts (max **6.74 mm**, `regular_frozen` exactly 0.00) — diagnostic
   only, not used in any conclusion here, but it corrupts the `metric_resp_*` family; an
   exact-bin tolerance (1e-4) smaller than the oracle's own token residue (1.1514e-4) so the
   bit-exact branch is dead on **0/192** oracle frames (blend weight 1.15e-4, numerically inert);
   one unguarded `$( )` at `run_fetal4d.sh:53`; and an unconditional `scatter.ref_plane` read that
   would break legacy bundles (no live cohort lacks it).
7. **`verify_af_pipeline.py` weaknesses** (reviewer-found): check C's GT half is **vacuous** — it
   reads `gt_map` from the manifest, picks the source file by it, and asserts equality with a
   destination that `build_af_bundle.py:310` created by `copyfile` from that same index, so it
   cannot fail. (The property *is* verified independently by `build_af_bundle.check` #2, which
   derives `k` from the source manifest's `roll_per_plane`.) Check A validates the gate against
   `pos_per_plane` — both written by the same generator — so it proves the gate is a faithful
   transform of the *recorded* positions, not that those positions describe the rendered pixels;
   that link is checked by `build_af_bundle.check` #4, which **skips `af`**. Check D's
   "per-slice const" case reuses the same circular-mean estimator it is testing. Also `:184`
   hardcodes `% 9` where D = 11, and `got != base` counts a crash as a detected fault.

## 8. Scoping the full run

- **Keep all 4 arms.** `regular_frozen` earned its place: it is the only reason we can attribute
  the −0.60 dB to rhythm rather than to the pipeline, and it caught the P046 readout disagreement.
- **Keep the oracle arm.** It produced the single most interesting result and halving the campaign
  by dropping it would have hidden the reversal entirely.
- **38 cmrx2024 subjects** ⇒ 4 cohorts × **3** fetal arms (self-gated, oracle, oracle-balanced) × 38
  ≈ **456 fetal runs**. At ~850 s each and 7 concurrent 4-CPU tasks, ≈ 15 h wall. VGGT adds 152 runs
  at ~1 min. The balanced arm is cheap relative to what it answers and should ship in the full run,
  not stay a 3-subject spot-check.
- **Report per-subject, not just cohort means**, and exclude degenerate windows by the
  `duplicate_targets` rule — at 38 subjects expect ~1 (design-time pass: 1/38 below 30 % excursion).
- **Control for window span** (§5) in any cross-cohort *absolute* PSNR claim — per arm, since the
  sign differs between VGGT and the fetal arms.
- ~~Run the §3d rebalanced-coverage gate arm first~~ **DONE.** It settled the mechanism (coverage,
  not timing accuracy) on 3 subjects. Worth re-confirming at 38 for the effect size, but the
  qualitative question that would have made the campaign ambiguous is answered.
- **Fix before scale-up**: the one-sided `GATE_ARM` guard, `resp_diag`'s frozen displacement, the
  `seq_index_basis` label, the exact-bin tolerance, and `verify_af_pipeline.py`'s `% 9` — none
  affected the pilot, all are cheap, and the first two would silently corrupt a 38-subject run's
  arm attribution and breathing metrics respectively.
- **Decide whether pilot rows belong in the live campaign `_ef/*.json`** (verified additive here,
  published 6.62 intact) or should be redirected with a `--out` override.

## 9. Full-cohort launch (38 subjects) — pre-launch review + what changed

**Scope.** 38 cmrx2024 test subjects × 4 rhythm cohorts × 3 fetal arms, **minus** a provable
optimization: `fetal_cmr_4d_oracle_balanced` is skipped for `regular_frozen`/`regular`, because
`rank_quantize` on an already-integer, already-uniformly-spaced rhythm is the identity permutation
— verified once on P016 (§3d) and then, before launch, on **all 76 manifests** in both cohorts
(§9.2). **380 total reconstructions** (not 456), submitted as SLURM job **61553947**
(`sbatch/af_full_fetal.sh`, 30 array tasks = 10 (cohort, arm) combos × 3 subject-shards each, using
`run_baselines.py`'s own `--shard I N` — the same mechanism the production campaign uses — rather
than the pilot's per-pair concurrency scheme). VGGT (152 runs) launched in parallel on the GPU;
doesn't depend on the fetal jobs.

**The 30-vs-CPU-budget correction.** Early planning conflated "30 CPUs" (this session's original
budget) with "30 concurrent jobs" — the user corrected this mid-session. 30 concurrent 4-CPU jobs
is 120 CPUs, not 30; wall-clock estimate corrected from ~15h to ~3.2h accordingly.

### 9.1 Second prove-it pass, before submission (3 reviewers, nothing yet queued)

Scoped to everything added since the pilot's own prove-it pass (commit `e6c1811` onward): the
`--balanced` gate code now going into production, the skip-decision's universal validity, and the
full-scale launch script + bundle correctness at 38 subjects. Did **not** re-review the pilot's own
PSNR/EF conclusions (already reviewed).

- **R1 (balanced-gate code):** clean on all four sub-questions — non-regression with `--balanced`
  OFF verified against the literal pre-commit diff line; `rank_quantize`'s bijection property holds
  **unconditionally** for any input (`argsort` always returns a valid permutation; `kind="stable"`
  affects only tie-break determinism, not correctness — confirmed with an all-identical-values
  probe, a mixed-tie probe, and an unstable-sort comparison, all still bijective); the constant-offset
  math is unaffected by quantized vs. raw input; the suffix-order question from the pilot's review
  was itself a false alarm — `"_oracle_balanced"` and `"_oracle"` end in different characters, so
  neither can ever be a suffix of the other, and no ordering in the strip tuple can matter.
- **R2 (skip-decision validity):** re-verified the P016 spot-check on **all 76 manifests** (38
  subjects × 2 cohorts) using the actual imported `rank_quantize`, not a reimplementation. Max
  deviation from integer: 7.1e-15. Every slice: a valid complete permutation of {0..11}, zero ties.
  Max `|theta_raw − theta_balanced|` across all 76: 5.3e-15 — identical noise floor to the original
  one-subject check, now universal. **CONFIRMED SAFE at full scale**, not just for P016.
- **R3 (launch readiness):** sharding math, `COMBO_IDX`/`SHARD_IDX` arithmetic, and `GATE_ARM`
  pairing all verified correct against `run_baselines.py`'s real `--shard` semantics (38 subjects
  over 3 shards → sizes 13/13/12, exact union, no gaps/overlaps). **Found the actual launch blocker**
  (§9.3) and **a new edge-case count** (§9.4) — both closed before submission.

### 9.2 Gate-build sequence actually run

Self-gated needs a GPU nnU-Net pass (dump 1824 frames → seg → assemble, ~35 min); oracle and
oracle-balanced are CPU-only (`assemble --oracle [--balanced]`, seconds). Order used:
`dump` (all 4 cohorts) → `seg` (GPU) → `assemble` self-gated → `assemble --oracle` (all 4 cohorts)
→ `assemble --oracle --balanced` (hrv + af only). **Completeness check before touching `sbatch`:**
all 10 (cohort, arm) combos confirmed at 38/38 gate files (380/380 total) by direct `ls` count, then
`sbatch/af_full_fetal.sh --dry-run` exercised at 4 array indices spanning both boundaries and the
middle of the index space (0, 9, 19, 29) — all four resolved to the correct (cohort, arm, shard)
and returned rc=0.

### 9.3 The launch blocker R3 found (closed before submission, not a code bug)

`run_baselines.py:130-138` raises an unguarded `FileNotFoundError` if **any** subject in a cohort's
full 38-subject list lacks its gate file — and this check runs **before** the `--shard` filter is
applied. So a missing gate for even one subject crashes the **entire** array task (all subjects in
that task's shard), not just the one subject missing it — the opposite of the "driver skips
subjects that lack them" behavior the script's own header comment implies. R3 checked this against
the actual state at review time (only the pilot's 4 subjects + 3 `af`/oracle-balanced pilot
subjects had gates) and correctly flagged that submitting then would have crashed all 30 tasks with
zero of 380 reconstructions completed. Not a logic bug — the gate-build step in §9.2 simply hadn't
been run yet at review time. Resolved by finishing §9.2 and re-verifying completeness (380/380)
before submission. **Worth fixing later**: make the precondition check per-shard, so a single
missing gate degrades to "skip one subject" instead of "crash the whole task."

### 9.4 New finding: 5/38 (not ~1/38) AF subjects are degenerate

The design-time cohort pass (docs/110 §9 item 4) predicted ~1/38 subjects with excursion < 30 %.
R3's full-cohort spot-check found **5/38 (13 %)** meet the `duplicate_targets` degeneracy signature:
the known **P004** (dup_pairs = 55, the pilot's case) plus **CMRx24_Test_P013** (6),
**CMRx24_Test_P023** (10), **CMRx24_Train_P038** (10), **CMRx24_Val_P026** (10) — all less extreme
than P004 (span 5.5–7.8 frames vs. P004's 0.97) but all exceed the `DEGEN_DUP_PAIRS = 5` exclusion
threshold. P023 verified at the pixel level, not just the manifest flag: `gt_t01`..`gt_t05` are
byte-identical (max|diff| = 0.0) across all 10 pairs. All 5 must be excluded from cohort PSNR/EF
means the same way P004 was in the pilot — `tools/af_pilot_report.py`'s exclusion logic already
handles this generically (`dup_pairs > DEGEN_DUP_PAIRS`), but its subject-list default was hardcoded
to the 4 pilot subjects and would have silently missed 4 of these 5. **Fixed** (§9.5): the default
now resolves the full test-split subject list dynamically via `paths.filter_by_split`.

### 9.5 Code changes made for the full-cohort launch

- `evaluation/src/engine/run_baselines.py` — the `GATE_ARM`/`METHOD` guard (previously one-sided,
  §7 item 6) is now symmetric: also refuses when `METHOD` names a `fetal_cmr_4d` variant and
  `GATE_ARM` is left unset (the exact silent-mispairing bug the original guard existed to prevent,
  just from the other direction). Verified: existing self-gated calls (no `GATE_ARM`) and existing
  correctly-paired oracle calls both still work unchanged; the bug scenario now hard-refuses; the
  new condition is syntactically scoped to `fetal_cmr_4d*` methods only and can never fire for
  svrtk3d/nesvor/other baselines. `pytest tests/` → 372 passed after the change.
- `sbatch/af_full_fetal.sh` (new) — the full-cohort launch script, §9 design above.
- `tools/af_pilot_report.py` — `SUBJECTS` default now resolves the live test-split subject list
  (38, not 4) via `paths.filter_by_split`, falling back to the pilot's 4-subject list only if that
  resolution fails; `FETAL` now includes `fetal_cmr_4d_oracle_balanced` so EF gets computed for the
  coverage-balanced arm too, not just self-gated/oracle/VGGT (closing the gap the user flagged: the
  §3d disambiguation was only ever measured in PSNR, and PSNR is known to be a poor instrument for
  exactly this failure mode — LV phase-mixing barely moves whole-volume PSNR but directly corrupts
  the (max−min)/max computation EF is; the coverage-vs-timing effect size may be much larger in EF
  than the 0.3–0.9 dB PSNR swings found in §3d).

## 10. Full-cohort RESULTS (38 subjects) — the pilot findings replicate and strengthen

Execution: fetal array `61553947` (30 tasks, 380 reconstructions) — **all COMPLETED, zero
failures**, ~3.6 h wall. VGGT (152 runs) ran in parallel on GPU. EF chain ran as 3 parallel SLURM
jobs (`61562349/350/351`, one per fetal arm) — all COMPLETED. Paired subject set: **n = 33**
(38 minus the 5 degenerate subjects confirmed in §10.2, excluded from every cohort so every arm
is compared on the identical subject set).

### 10.1 PSNR — the reversal replicates and gets SHARPER, not weaker

| cohort | self-gated | oracle-raw | VGGT | oracle gap |
|---|---|---|---|---|
| regular_frozen | 24.41 | 24.70 | 26.21 | +0.29 |
| regular | 24.63 | 25.00 | 26.11 | +0.37 |
| hrv | 24.72 | 24.96 | 26.43 | +0.24 |
| **af** | 24.21 | **23.40** | 26.33 | **−0.81** |

The pilot (n=3) measured the af oracle gap at −0.63 dB. At n=33 it is **−0.81 dB** — larger, not
smaller. Δ vs regular_frozen: self −0.20 / oracle **−1.31** / VGGT +0.12 under `af`, the same
signature as the pilot (self mild, oracle large, VGGT flat).

**Oracle-balanced replicates the fix, too** (`hrv`/`af` only, same paired set):

| cohort | self-gated | oracle-raw | oracle-**balanced** |
|---|---|---|---|
| hrv | 24.72 | 24.96 (+0.24) | 24.80 (+0.09) |
| af | 24.21 | 23.40 (−0.81) | **24.38 (+0.17)** |

Restoring bucket coverage flips the sign again, exactly as at n=3 (there: −0.63 → +0.26).

### 10.2 The degenerate-subject rate is real model behavior, not a bug — Monte Carlo confirms it

R3's full-scale review (§9.4) found 5/38 af subjects degenerate — **P004, P013, P023, P038,
P026** — vs the design-time ~1/38 estimate. Ran 5000 independent Monte Carlo draws of the
**actual, unmodified** `rr_sequence`/`pos_physio` mechanism (not a re-derivation): **15.4%**
degenerate rate, matching the measured **13.2%** (5/38) closely — well within sampling noise at
n=38. **The design-time estimate was wrong because it was based on too small a sample; the
simulator is behaving exactly as its own physiology implies** — AF beat lengths have a floor
(0.45×) but no ceiling, so occasional very long beats (and the holds they produce) are an
expected, not anomalous, consequence of that distribution. Still open: whether the disclosed
window-open-phase limitation (§11.5) inflates degeneracy for any of the 4 non-P004 cases beyond
P004 itself — not yet checked.

### 10.3 EF — the real headline. The advantage survives AF but drops by two-thirds

| cohort | self-gated | oracle-raw | VGGT |
|---|---|---|---|
| regular_frozen | 6.26 | 5.17 | 9.87 |
| regular | 6.31 | 6.03 | 10.84 |
| hrv | 6.28 | 4.78 | 9.41 |
| **af** | **8.94** | **9.60** | 9.95 |

**Sanity check — replication held.** `regular_frozen` self-gated EF MAE (6.26) matches the
already-published cmrx2024-specific number (6.23) closely, and `regular_frozen`'s input stacks
are still byte-identical to the live campaign's `rolled/` bundle. The full-cohort pipeline
reproduces the known result before this section's new numbers.

**Does Fetal 4D's EF advantage over VGGT survive AF?** Yes, but it collapses by roughly
two-thirds: **3.61 pp** (regular_frozen: 9.87 − 6.26) → **1.01 pp** (af: 9.95 − 8.94). Self-gated's
own EF error rises **+2.68 pp** under af (vs +0.05/+0.02 pp under regular/hrv — essentially flat
there); VGGT's is flat everywhere (+0.08 to +0.97 pp, no rhythm-driven trend).

**The coverage-vs-timing mechanism is far more visible in EF than PSNR — as predicted.** Oracle-raw
degrades **harder** than self-gated under af (+4.43 pp vs self-gated's own +2.68 pp — a ~65% larger
absolute degradation), and oracle-balanced doesn't just recover, it **beats self-gated outright**:

| cohort | self-gated | oracle-raw | oracle-**balanced** |
|---|---|---|---|
| hrv | 6.28 | 4.78 | 5.10 |
| af | 8.94 | 9.60 | **7.91** |

This is the sharpest version of the §3d finding yet: fixing bucket coverage while keeping true
timing doesn't just avoid the reversal, it produces the best EF of any fetal arm under af.

### 10.4 Visual confirmation (docs/111 had none until this session)

Built `tools/render_af_bucket_comparison.py` and `tools/render_af_bucket_with_curve.py`. **First
attempt had a real bug**, not a quality finding: it sliced every arm's volume at the same raw
Z-index, but `fetal_cmr_4d`'s native reconstruction grid is a completely different shape
((100,85,103) vs GT's (256,256,12)) — comparing same-index slices across incompatible grids is
meaningless and produced a misleadingly bad-looking result. **Fixed** by reusing
`image_metrics.py`'s own `load_blurred → fit_rigid → resample` pipeline directly, so what's
rendered is exactly what gets scored.

Confirmed with the corrected renders (subject P017, `af` cohort):
- `oracle-raw` shows visible speckle/mottled texture at z=2/z=6, f=6, where self-gated and
  oracle-balanced are visually smooth — the coverage failure is directly visible, not just a
  metric artifact.
- `VGGT` reconstructs the entire canonical FOV (chest wall, liver, etc.); the fetal_cmr_4d family
  is confined to a tight heart-only crop by design — this, not a quality gap, is why VGGT looks
  far more "complete" despite losing on EF. Not previously stated plainly in this doc.
- `vol_t{f}.nii.gz` confirmed by reading `run_fetal4d.sh:192` to be the **AF-order readout**
  (already resampled to match GT's target-f indexing), not the engine's raw uniform-phase-bin
  output — settles a question raised mid-session.

**P004 visually and numerically confirmed frozen**, independent of the manifest flag: real
`seg_gt` segmentation gives LV voxel counts `[5329]×11, 4563` — 11 of 12 frames pixel-identical,
EF = 14.4% (vs a true ~70%). **P017 rendered as the healthy contrast case**: real segmentation,
smooth LV curve, EF = 63.6%.

**Correction to an earlier in-session explanation.** Initially reasoned that `hrv` avoids
degradation because its bucket misplacements are small ("adjacent bucket" blur). **Measured, and
wrong**: `hrv` and `af` have similarly large misplacement-magnitude distributions when a
misplacement occurs (43.2% vs 45.3% off by 4+ buckets — nearly identical). The real reason `hrv`
doesn't hurt: (1) self-gated is *always* a per-slice bijection regardless of accuracy — it never
creates true coverage gaps under either rhythm — and (2) nearby cardiac phases look almost
identical (measured on real GT: only ~0.67% intensity difference at 1 bucket apart, ~1.06% even at
6 buckets/half a cycle apart), so a mislabeled-but-real frame doesn't look visibly wrong. `af`'s
hold mechanic is a qualitatively different failure — a genuine information gap (a phase nobody
captured) — which no amount of "nearby frames look similar" can paper over.

### 10.5 Not yet done

- Statistical re-check of the pilot's confounded correlation (§3a-bis) at full n — does the
  bin-imbalance-vs-cohort-identity confound resolve with ~130 data points instead of 12?
- Formal significance testing on the EF/PSNR deltas.
- Whether the window-open-phase bug (§11.5) contributes to P013/P023/P038/P026's degeneracy.
- Extending beyond cmrx2024 to the other 6 sources — not started, no bundles built.
