# 111 — Arrhythmia pilot: results, and why the pre-registered outcome did NOT happen

> **TL;DR & takeaway**
> First execution of the docs/110 rhythm ladder: 4 cohorts × 4 subjects × 3 arms
> (`fetal_cmr_4d`, `fetal_cmr_4d_oracle`, VGGT `final518_diff1000_ep300`), 32 fetal recons + 16 VGGT
> runs, all scored through the standard chain. **Gating collapses exactly as designed** — the
> structural (within-slice) θ error goes 0.000 → 0.000 → 0.486 → 1.328 cine frames and 31 % → 31 %
> → 55 % → **75 %** of input images land in the wrong reconstruction bin. **But the damage barely
> reaches the image metrics** (self-gated −0.60 dB under `af`) and **the oracle arm REVERSES**: under
> `af`, handing Fetal 4D the TRUE per-frame timing makes it **0.63 dB WORSE**, consistently on all 3
> non-degenerate subjects. Measured cause: the constant-rate assumption puts *exactly* D images in
> every output bin **by construction**, while true AF timing gives 7–25 per bin; bin imbalance
> predicts the oracle gap at **r = −0.832**, and the fit says correct timing is worth **+0.52 dB**
> while AF's imbalance costs **−0.99 dB**. The competing explanation (fractional readout blending)
> is refuted by `hrv` as an internal control. **So the method's apparent AF-robustness is not
> robustness — the wrong assumption buys balanced phase coverage for free, and that is worth more
> than the timing error costs.** VGGT is flat across rhythm (+0.03 dB under `af`), as predicted,
> because it is told the target instant. **EF is uninterpretable at n=3** (frozen-vs-framewise noise
> floor alone reaches 15 pp) and needs the full cohort. Pipeline correctness verified four ways with
> seven fault injections (`tools/verify_af_pipeline.py`).

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
  for any periodic rhythm (both θ are then uniform ramps, so their difference is a per-slice
  constant by construction — the `regular*` rows therefore self-check the decomposition, and 200
  random-anchor trials confirm it to 8.7e-07). No choice of ED anchor can reduce it.

**Measured, and it refutes the obvious story:** the error is **not** concentrated at beat
boundaries. Using the manifest's own `beat_per_plane`, boundary frames carry only 1.25× the interior
error under `af` and **0.98×** under `hrv` — i.e. no concentration at all. The damage is spread
across the window, because the dominant term is a per-slice anchor offset that shifts all 12 frames
together.

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

**r(bin imbalance, oracle gap) = −0.832** (n = 12, t = −4.74). Fit:
`oracle gap = −3.51 × imbalance + 0.52 dB`, i.e. **correct timing is worth +0.52 dB at zero
imbalance, and AF's mean imbalance of 0.281 costs −0.99 dB** — net −0.47 dB, against the −0.63
observed.

⚠️ The 12 points are **not independent** (3 subjects × 4 cohorts, cohorts sharing subjects) and
imbalance is 0 by construction for `regular*`, so the correlation partly re-encodes cohort identity.
The effect also holds *within* `af` alone, where the worst-imbalance subject (P017, 0.350) has the
worst gap (−1.07). Treat r as descriptive; the full cohort is needed for a real test.

### 3b. The competing explanation, refuted

H2: the oracle readout **blends** two bins (fractional index) while the self-gated readout takes an
exact bin — maybe blending is what hurts. **`hrv` is the internal control**: its oracle readout is
fractional on all 12 frames with essentially the same mean blend weight as `af` (0.184 vs 0.180),
but its bin imbalance is low (0.139) — **and its oracle gap stays positive (+0.37)**. Blending alone
does not cause the loss. Bin imbalance does.

### 3c. What this means

The self-gated method's apparent AF-robustness (−0.60 dB) is **not robustness**. The constant-rate
assumption is wrong about *when* each image was taken but accidentally right about *how to spread
images over the cycle*, and on this data the second is worth more than the first costs. Any future
"better gate" that recovers true timing without also controlling bin occupancy would therefore make
reconstruction **worse**, not better — a concrete, testable prediction from this pilot.

## 4. VGGT is flat, as pre-registered

Paired: 25.04 / 24.77 / 25.09 / 25.07. The only real cost is **frame-wise breathing (−0.27 dB)**;
`hrv` and `af` are within noise of `regular_frozen`. Expected — VGGT is *told* the target instant
(`timesteps[0,0] = t`), so it never infers timing and rhythm irregularity cannot damage that step.

## 5. Traps this pilot confirmed in real data

**P004 under `af` is degenerate and must never be averaged in.** Its window holds at full ED, so
**55 of 66 GT pairs are byte-identical**, position span 0.97 frames (vs 9.5–11.0 for everyone else),
and GT EF collapses 70.6 % → **14.4 %**. Live confirmation that it distorts in *both* directions: its
`af` PSNR is **30.46, the highest of its four cohorts**, because predicting a near-constant volume is
easy. Signature: `rhythm.duplicate_targets` in the manifest, flagged automatically by
`af_pilot_report.py` (`dup_pairs > 5`).

**Window-span confound.** Under `af` a window can cover less of the cycle, which makes
reconstruction *easier*. r(span, ΔPSNR) = **−0.445** over all 16 rows. Any cross-cohort PSNR claim
in the full run must control for span.

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

`regular_frozen` vs `regular` differ **only** in frozen vs frame-wise breathing, so their EF gap is a
pure noise floor: **mean 3.7–5.8 pp, max 15.0 pp** (fetal self-gated, P046: 28.5 → 43.5).

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

## 8. Scoping the full run

- **Keep all 4 arms.** `regular_frozen` earned its place: it is the only reason we can attribute
  the −0.60 dB to rhythm rather than to the pipeline, and it caught the P046 readout disagreement.
- **Keep the oracle arm.** It produced the single most interesting result and halving the campaign
  by dropping it would have hidden the reversal entirely.
- **38 cmrx2024 subjects** ⇒ 4 cohorts × 2 fetal arms × 38 = **304 fetal runs**. At ~850 s each and
  7 concurrent 4-CPU tasks, ≈ 10 h wall. VGGT adds 152 runs at ~1 min.
- **Report per-subject, not just cohort means**, and exclude degenerate windows by the
  `duplicate_targets` rule — at 38 subjects expect ~1 (design-time pass: 1/38 below 30 % excursion).
- **Control for window span** (§5) in any cross-cohort PSNR claim.
- The bin-occupancy result (§3a) is the headline and deserves the full cohort to firm up r.
