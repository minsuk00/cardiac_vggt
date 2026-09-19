# 111 — Arrhythmia pilot: results, and why the pre-registered outcome did NOT happen

> **TL;DR & takeaway**
> First execution of the docs/110 rhythm ladder: 4 cohorts × 4 subjects × 3 arms
> (`fetal_cmr_4d`, `fetal_cmr_4d_oracle`, VGGT `final518_diff1000_ep300`), 32 fetal recons + 16 VGGT
> runs, all scored through the standard chain. **Gating collapses as designed** — the structural
> (within-slice) θ error goes 0.000 → 0.000 → 0.486 → 1.328 cine frames and **62–75 %** of input
> images land in the wrong reconstruction bin (the range is admissible-offset dependent, §2a).
> **But the damage barely reaches the image metrics** (self-gated −0.60 dB under `af`) and **the
> oracle arm REVERSES**: under `af`, handing Fetal 4D the TRUE per-frame timing makes it **0.63 dB
> WORSE**, on all 3 non-degenerate subjects independently. That reversal is the robust result — it
> survives including P004 (at 0.46 dB instead of 1.09 dB) and is **not** explained by window span.
> **Its cause is NOT established.** The leading hypothesis is phase-coverage: the constant-rate
> assumption gives every (slice, phase) cell exactly one image **by construction**, while the `af`
> oracle leaves **9.8–18.1 % of cells empty**. But with 4 cohorts × 3 subjects the coverage measure
> is **near-totally confounded with cohort identity** — a plain "is this `af`" dummy predicts the gap
> *better* (r = −0.891) than imbalance does (−0.832), and fitting both collapses imbalance from
> −3.51 to −1.12. Treat it as a hypothesis with a designed follow-up (§3d), not a measured cause.
> VGGT is flat across rhythm (+0.03 dB under `af`), though `af` differs from `regular` in **more than
> rhythm** (§5). **EF is uninterpretable at n=3.** Pipeline correctness verified four ways with seven
> fault injections (`tools/verify_af_pipeline.py`), then re-audited by 3 independent reviewers whose
> findings are folded in below — including two claims of an earlier draft that are now **withdrawn**.

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

### 3a-bis. Why this is a HYPOTHESIS, not a measured cause

Four independent problems, all found by adversarial review and each verified by recomputation:

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

### 3d. The follow-up that would settle it

The correlational design cannot escape the cohort confound. The decisive experiment is a **third
gate arm**: true per-frame θ (as the oracle) but **quantised/rebalanced so every (slice, phase) cell
is occupied**, run on the same `af` bundles. If the reversal disappears, coverage is causal; if it
persists, the cause is timing accuracy itself and the whole story changes. ~4 recons × ~14 min. This
does not modify any core logic — it adds a gate variant alongside the existing two.

Until then the honest statement is: **the reversal is real and robust; its cause is a hypothesis.**

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
6. **Reviewer findings in the shipped code, not fixed during the pilot** (one binary for all runs):
   a one-sided `GATE_ARM`/`METHOD` guard (`run_baselines.py:100`) that still admits a mis-attributed
   `_oracle` arm when `METHOD` is set and `GATE_ARM` is not — the pilot is unaffected because
   `af_pilot_fetal.sh` always sets both and all 32 stamps verify; `resp_diag` reporting the **frozen**
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
- **38 cmrx2024 subjects** ⇒ 4 cohorts × 2 fetal arms × 38 = **304 fetal runs**. At ~850 s each and
  7 concurrent 4-CPU tasks, ≈ 10 h wall. VGGT adds 152 runs at ~1 min.
- **Report per-subject, not just cohort means**, and exclude degenerate windows by the
  `duplicate_targets` rule — at 38 subjects expect ~1 (design-time pass: 1/38 below 30 % excursion).
- **Control for window span** (§5) in any cross-cohort *absolute* PSNR claim — per arm, since the
  sign differs between VGGT and the fetal arms.
- **Run the §3d rebalanced-coverage gate arm first**, on the pilot subjects. It costs ~1 h and it
  decides whether the full campaign is measuring a coverage effect or a timing effect. Scaling up a
  correlational design that is already near-totally confounded with cohort identity would buy
  precision on a number that cannot answer the question.
- **Fix before scale-up**: the one-sided `GATE_ARM` guard, `resp_diag`'s frozen displacement, the
  `seq_index_basis` label, the exact-bin tolerance, and `verify_af_pipeline.py`'s `% 9` — none
  affected the pilot, all are cheap, and the first two would silently corrupt a 38-subject run's
  arm attribution and breathing metrics respectively.
- **Decide whether pilot rows belong in the live campaign `_ef/*.json`** (verified additive here,
  published 6.62 intact) or should be redirected with a `--out` override.
