# 112 — AF campaign follow-up: the three open statistical questions, answered

> **TL;DR & takeaway**
> Closes the three analyses docs/111 left open. All three came back **negative for the claim they
> were testing**, and two of them change what docs/111 may say.
> **(1) The bin-imbalance correlation does NOT survive.** At n=132 (not the pilot's 12) the cohort
> dummy *still* out-predicts it (r = −0.588 vs −0.529), and the decisive test — the correlation
> *within* the `af` cohort, where the dummy is constant — is **null**: r = −0.152 (p = 0.40) for
> imbalance, −0.061 for the PSF-weighted version, −0.227 (p = 0.20) for empty-cell fraction. In a
> cohort-fixed-effects fit imbalance carries **p = 0.32** while the `af` dummy keeps p = 0.007.
> This is *not* a power failure: imbalance varies 4.5× within `af` (0.21–0.94) and n = 33 resolves
> |r| ≥ 0.344. **Withdraw the correlational argument; the mechanism rests on the §3d/§10
> intervention, which is unaffected and is independently significant (below).**
> **(2) The PSNR mechanism is significant; the EF headline is NOT.** The oracle reversal is rock
> solid — af oracle gap **−0.815 dB, 95 % CI [−1.14, −0.50]**, p = 2.8e−05; the reversal vs
> `regular_frozen` **−1.106 dB [−1.48, −0.73]**, p = 2.6e−06; coverage-balancing flips it positive
> **+0.166 dB [+0.04, +0.29]**, p = 0.005. But EF cannot carry its headline: Fetal's advantage over
> VGGT is significant under `regular_frozen` (**+3.61 pp [+0.17, +6.87]**, p = 0.007) and
> **indistinguishable from zero under `af`** (**+1.01 pp [−2.13, +4.01]**, p = 0.50), while the
> *collapse* itself (−2.61 pp [−6.79, +1.70], p = 0.22) is **not significant**. So the supportable
> sentence is "**the advantage is no longer detectable under AF**", NOT "it survives, barely" and
> NOT "AF significantly shrinks it".
> **(3) Degeneracy is a beat-draw lottery — not the window-open bug, and not a subject trait.**
> Only 2 of the 5 degenerate subjects open inside a hold (Fisher p = 0.076), and all 5 stay
> degenerate when the window is moved elsewhere in the beat (P = 0.35–0.81, and *higher* when hold
> openings are excluded). Had windows never opened in a hold the rate would be **0.212, above the
> measured 0.132** — the disclosed bug slightly *reduces* degeneracy here. Re-drawing the beats
> instead: P(degenerate) is 0.163 for the 5 vs 0.136 for the other 33 — the same. Cohort mean over
> re-draws **0.139**, matching the measured 0.132 and docs/111 §10.2's Monte Carlo 15.4 %.
> **The exclusion is legitimate: those subjects drew a rhythm that genuinely froze the reference
> plane. It is simulated AF physiology, not an artifact.**

**Written:** 2026-09-19. **Branch:** `exp/af-sim` (worktree `/home/minsukc/vggt-afsim`).
**Predecessor:** docs/111 (the campaign), docs/110 (the simulator design).
**Tools:** `tools/af_confound_recheck.py`, `tools/af_significance.py`,
`tools/af_degeneracy_cause.py`. All three are read-only analyses over the already-scored full
cohort — no reconstructions were re-run and nothing under `scratch/eval/` was written.

---

## 0. Why these three

docs/111 §11 listed them as the open items. They share a property: each tests whether a claim
already *written* in docs/111 is actually supported by the data behind it. None of them can
discover a new effect; they can only confirm or withdraw.

**Scope note.** Everything below is on the 33-subject paired set (the 5 degenerate windows dropped
from every cohort, exactly as every other docs/111 table does). §3 is the one section that uses all
38, since degeneracy is its subject.

## 1. The bin-imbalance correlation does not survive (docs/111 §3a, §3a-bis)

### 1.1 What was claimed and why it was doubted

The pilot explained the oracle reversal by bin imbalance: true AF timing fills the engine's 12
phase bins unevenly, so some output phases are starved. Evidence was
`r(bin imbalance, oracle gap) = −0.832` at n = 12. §3a-bis already recorded the problem — a plain
"is this the `af` cohort" dummy predicted the same gap *better* (−0.891), and fitting both dropped
imbalance's coefficient from −3.51 to −1.123. At n = 12 the two regressors are near-collinear by
construction (imbalance is exactly 0 for both `regular` arms, small for `hrv`, large for `af`), so
the design could not separate them.

### 1.2 The test

`tools/af_confound_recheck.py`. Per (cohort, subject), n = 132 rows:

| quantity | definition |
|---|---|
| `gap` | `psnr_unit_peak(fetal_cmr_4d_oracle) − psnr_unit_peak(fetal_cmr_4d)`. Within-cohort, same input, GT and engine — only the gate differs. |
| `imbalance` | std/mean of the 12 bins' **image counts** under the oracle gate (the §3a definition). |
| `imbalance_w` | the same on the engine's actual **Gaussian temporal PSF weights** (§3c's refinement — the engine does not hard-bin; σ = dtrad/2.355 with dtrad = 2π/T, from `ReconstructionCardiac4D.cc:567-586`). |
| `empty_frac` | fraction of (slice, output-phase) cells receiving total weight < 1e−2 — §3c argued this is the physically sharper quantity. |

**Unit checks (both reproduce docs/111 exactly):** af/P017 oracle occupancy
`[12 10 9 25 9 8 11 12 13 11 11 13]`, std/mean **0.350** — as published in §3a. Self-gated minimum
cell weight **1.125** with **0 %** of cells empty in all 132 rows — as published in §3c.

### 1.3 Result: the confound does not resolve, it persists

Cohort means (n = 33 each):

| cohort | imbalance | imbalance_w | empty % | oracle gap (dB) |
|---|---|---|---|---|
| `regular_frozen` | 0.000 | 0.000 | 0.0 | +0.29 |
| `regular` | 0.000 | 0.000 | 0.0 | +0.37 |
| `hrv` | 0.137 | 0.092 | 0.2 | +0.24 |
| `af` | 0.374 | 0.297 | 8.0 | −0.81 |

**Pooled across cohorts (the pilot's design, now at n = 132):**

| regressor | r | p |
|---|---|---|
| imbalance | −0.529 | 6.9e−11 |
| imbalance_w | −0.514 | 3.0e−10 |
| empty_frac | −0.587 | 1.4e−13 |
| **af dummy** | **−0.588** | 1.2e−13 |

The dummy still ties or beats every mechanism regressor. Ten times the data did not help, because
the collinearity is structural, not a sampling artifact.

**Within each cohort (the decisive test — the dummy is constant here):**

| cohort | imbalance | imbalance_w | empty_frac |
|---|---|---|---|
| `regular_frozen` | (no variance) | +0.057 (p = 0.75) | (no variance) |
| `regular` | (no variance) | +0.113 (p = 0.53) | (no variance) |
| `hrv` | **+0.310** (p = 0.080) | +0.259 (p = 0.15) | −0.123 (p = 0.50) |
| `af` | **−0.152** (p = 0.40) | −0.061 (p = 0.73) | −0.227 (p = 0.20) |

Null within `af`, and within `hrv` the sign is *positive* — the opposite of the hypothesis.

**Cohort fixed-effects fit** (`gap ~ regressor + cohort dummies`, `regular_frozen` = reference):

| regressor | coefficient | p | af dummy's p |
|---|---|---|---|
| imbalance | −0.700 | 0.32 | 0.007 |
| imbalance_w | −0.299 | 0.73 | 0.001 |
| empty_frac | −6.330 | **0.059** | 0.058 |

Imbalance is not significant once cohort is controlled; the `af` dummy absorbs the effect.
`empty_frac` is the only one that comes close and it is still above 0.05 — consistent with §3c's
position that empty cells are the better-motivated quantity, but it does not reach significance
either, and at that point it and the af dummy are statistically interchangeable (p = 0.059 vs
0.058), which is the same confound again.

### 1.4 The null is informative, not underpowered

A within-cohort null only means something if the regressor varies and the outcome is measurable
there. Within `af`:

- `imbalance` spans **0.210 → 0.941** (sd 0.165) — a 4.5× range.
- `empty_frac` spans 0.015 → 0.189 (sd 0.035).
- `gap` spans −3.09 → +0.69 dB (sd 0.949) — plenty of outcome variance to explain.
- n = 33 resolves **|r| ≥ 0.344** at α = 0.05. Measured |r| = 0.152.

So the true within-`af` association is probably below 0.344, not merely undetected. Had the pilot's
−0.832 reflected a within-cohort mechanism, this design would have seen it.

### 1.5 What to do about it

**Withdraw the correlational argument from the paper.** The mechanism claim — coverage, not timing
accuracy — is established by the §3d/§10 **intervention**: build a gate that keeps true timing but
forces bijective coverage, and the reversal flips. That is causal evidence, it does not depend on
the correlation, and §2.4 below shows it is independently significant at p = 0.005. The correlation
was always the weaker leg; it is now measured to be carrying nothing.

## 2. Confidence intervals and paired tests (docs/111 §10, all of it)

`tools/af_significance.py`. Paired on the same 33 subjects; each contrast is a one-sample test on
per-subject differences with a 95 % percentile bootstrap CI (20 000 resamples), Wilcoxon
signed-rank, and a paired t as cross-check. **Every point estimate reproduces docs/111 §10 exactly**
(3.612 / 1.006 / +2.683 / −0.815 / +0.657 / −1.032), so this adds intervals to unchanged numbers.

### 2.1 The EF headline — significant under regular rhythm, NOT under AF

| contrast (EF: VGGT error − Fetal error; >0 = Fetal better) | mean | 95 % CI | Wilcoxon p |
|---|---|---|---|
| `regular_frozen` | **+3.612** | [+0.173, +6.866] | **0.007** |
| `regular` | +4.537 | [+1.332, +7.917] | **0.025** |
| `hrv` | +3.132 | [+0.189, +6.191] | 0.068 |
| `af` | **+1.006** | **[−2.133, +4.010]** | **0.502** |
| **collapse (af − regular_frozen)** | **−2.606** | **[−6.792, +1.701]** | **0.221** |

Two consequences, both binding on how docs/111 may be phrased:

1. **"Still wins, barely" is not supported.** Under `af` the advantage's CI straddles zero by a
   wide margin. The supportable statement is *"Fetal CMR 4D's EF advantage is significant under a
   regular rhythm and is no longer detectable under `af`."*
2. **"Collapses 3.6×" is also not supported as an inference.** The interaction test is null
   (p = 0.22). The two cohort estimates differ in significance, which is not the same as differing
   from each other. We can report the point estimates and that one is significant and the other is
   not; we cannot claim AF measurably shrank the advantage.

EF at n = 33 has CIs roughly ±3–4 pp. It is simply too noisy an instrument for a 2.6 pp interaction.

### 2.2 Whose EF moves under AF

| contrast | mean | 95 % CI | Wilcoxon p |
|---|---|---|---|
| Fetal self-gated, `regular` − `regular_frozen` | +0.045 | [−2.44, +2.45] | 0.66 |
| Fetal self-gated, `hrv` − `regular_frozen` | +0.020 | [−2.98, +2.65] | 0.40 |
| Fetal self-gated, **`af` − `regular_frozen`** | **+2.683** | [−0.137, +5.671] | **0.065** |
| VGGT, `regular` − `regular_frozen` | +0.971 | [−1.53, +3.63] | 0.61 |
| VGGT, `hrv` − `regular_frozen` | −0.460 | [−3.46, +2.31] | 0.61 |
| VGGT, `af` − `regular_frozen` | +0.077 | [−3.37, +3.71] | 0.79 |

The qualitative picture in docs/111 §10 holds — Fetal's EF error moves only under `af`, VGGT's is
flat everywhere — but Fetal's +2.68 pp is **marginal (p = 0.065), not established**. VGGT's
flatness is genuinely flat (all three CIs centred near zero), though with CIs this wide "flat" here
means "no detectable change", not "proven unchanged".

### 2.3 The EF mechanism arms are directionally right but individually null

| contrast (EF |err| under `af`; negative = better than self-gated) | mean | 95 % CI | Wilcoxon p |
|---|---|---|---|
| oracle-raw − self-gated | +0.657 | [−2.96, +4.54] | 0.82 |
| oracle-balanced − self-gated | −1.032 | [−2.92, +0.76] | 0.34 |
| oracle-balanced − oracle-raw | −1.690 | [−5.49, +1.76] | 0.57 |

All three point estimates order exactly as the coverage mechanism predicts, and none is
significant **in EF**. Report them as consistent-with, not as evidence. The same three contrasts in
PSNR (next) are where the evidence actually lives.

### 2.4 PSNR — the mechanism is significant, strongly

| contrast (PSNR, dB) | mean | 95 % CI | Wilcoxon p |
|---|---|---|---|
| oracle gap, `regular_frozen` | +0.291 | [+0.119, +0.488] | 0.0006 |
| oracle gap, `regular` | +0.367 | [+0.178, +0.585] | 0.0002 |
| oracle gap, `hrv` | +0.238 | [+0.085, +0.393] | 0.009 |
| **oracle gap, `af`** | **−0.815** | **[−1.144, −0.500]** | **2.8e−05** |
| **reversal, `af` − `regular_frozen`** | **−1.106** | **[−1.480, −0.732]** | **2.6e−06** |
| **oracle-BALANCED gap, `af`** | **+0.166** | **[+0.039, +0.285]** | **0.005** |

This is the paper's real evidence and it is unambiguous. The oracle gap is positive and significant
in all three periodic/near-periodic cohorts, significantly **negative** under `af`, the reversal
itself is significant at p = 2.6e−06, and restoring coverage while keeping true timing flips the
sign back to significantly positive. **Coverage, not timing accuracy** — established by
intervention, with intervals, without needing §1's correlation.

### 2.5 AF's own PSNR cost is not significant for either method

| contrast | mean | 95 % CI | Wilcoxon p |
|---|---|---|---|
| Fetal self-gated, `af` − `regular_frozen` | −0.203 | [−0.721, +0.329] | 0.30 |
| VGGT, `af` − `regular_frozen` | +0.123 | [−0.105, +0.352] | 0.37 |

Consistent with docs/111's "the damage barely reaches the image metrics": self-gating absorbs AF
almost completely in PSNR. Note this is the *self-gated* arm — the large, significant effects in
§2.4 are all *within*-cohort comparisons between gates, which is why they survive while this
cross-cohort comparison does not.

## 3. The 5 degenerate subjects: chance, not the window bug (docs/111 §9.4, §10.2)

### 3.1 What was suspected

GT for frame *f* is the clean volume at the **reference plane's** true position for frame *f*, so
duplicate targets appear when that plane's trace repeats — i.e. the heart is parked at the full-ED
hold. 5/38 `af` subjects are degenerate (P004 55 dup pairs, P013 6, P023/P038/P026 10 each). Only
P004's cause was traced, to docs/110 §11.5's disclosed window-open-phase limitation: `simulate`'s
`t0` is a **time** offset, and under `pos_physio` that can map to a stored phase already at the
hold — something no other arm can do. The suspicion was that the same bug explains the other four.

### 3.2 The test

`tools/af_degeneracy_cause.py` reproduces each subject's reference-plane trace from the manifest's
own seed and roll — **asserted equal to the stored `rhythm.ref_pos` for all 38 subjects**, so the
recomputation is exact. Fault-injected: shifting `t0` by one frame or using a neighbouring plane's
RNG stream both make the assert fire. (One real bug was caught this way: the LV volume curve must
be read from the **source** cohort, since the `af` bundle's own `seg_gt/` is a `gt_map`-permuted
copy.) Then two counterfactuals, both keeping the subject fixed:

- **Move the window:** same beats, sweep the open time across all of beat `BURN` (200 samples).
- **Re-draw the beats:** same LV curve and open rule, 200 alternative RNG streams.

### 3.3 Result — the bug is not the cause

**Q1, contingency (n = 38).** Only 2 of the 5 degenerate subjects open inside a hold:

|  | degenerate | not |
|---|---|---|
| opens in hold | 2 | 2 |
| opens normally | 3 | 31 |

P(degenerate | opens in hold) = 0.50 vs 0.088 otherwise — suggestive, but **Fisher p = 0.076**, not
significant, and it leaves 3 of 5 unexplained.

**Q2, move the window.** Every degenerate subject stays degenerate at most other open times, and
*more so* when hold openings are excluded:

| subject | dup/66 | opens in hold | P(deg \| any open time) | P(deg \| non-hold open) | P(deg \| re-drawn beats) |
|---|---|---|---|---|---|
| Test_P004 | 55 | yes | 0.805 | **1.000** | 0.325 |
| Test_P023 | 10 | no | 0.600 | 0.664 | 0.235 |
| Train_P038 | 10 | no | 0.345 | 0.345 | 0.025 |
| Val_P026 | 10 | no | 0.750 | **1.000** | 0.055 |
| Test_P013 | 6 | yes | 0.730 | 0.940 | 0.175 |

**Q3, attributable fraction.** Cohort-wide, if windows never opened in a hold the degenerate rate
would be **0.212** — *higher* than the measured **0.132**. The disclosed bug slightly **reduces**
degeneracy in this cohort. It cannot be its cause.

### 3.4 Result — nor is it a subject trait

Re-drawing the beats (200 seeds per subject, own open rule):

- cohort mean P(degenerate) = **0.139**, against the measured 0.132 and docs/111 §10.2's 5000-trial
  Monte Carlo 15.4 %. Three independent estimates of the same quantity agree.
- P(deg | re-draw) among the 5 degenerate subjects = **0.163**, among the other 33 = **0.136**.
  Essentially identical — being degenerate this time says almost nothing about the subject.
- **0/38** subjects are degenerate under more than half of their re-draws; only **1/38** is immune
  under all of them.

The high P(deg | sweep) with **own** beats (0.35–0.81) alongside low P(deg | re-drawn beats)
(0.03–0.33) is the coherent story: **given an unlucky beat sequence almost any window into it is
degenerate, but unlucky beat sequences are uncommon.** Degeneracy is a per-(subject, seed) lottery
of the AF rhythm itself.

### 3.5 What to say to a reviewer

"We exclude 5 of 38 subjects (13 %) whose simulated AF rhythm froze the reference plane, producing
duplicate ground-truth targets. The rate matches the mechanism's analytic expectation (15.4 % by
Monte Carlo, 13.9 % by beat re-draw, 13.2 % measured), it is not attributable to the known
window-open-phase limitation (which, if removed, would *raise* it to 21 %), and it is not a
property of particular subjects (P(degenerate) 0.163 vs 0.136 for the rest under re-drawn beats).
These are genuine unreconstructable windows of the simulated physiology, not artifacts."

## 4. Consequences for docs/111

1. **§3a / §3a-bis:** mark the bin-imbalance correlation **withdrawn** (§1 here). §3c's empty-cell
   variant is the best of them and still does not reach significance.
2. **§10 TL;DR:** "survives AF but collapses from 3.61 pp to 1.01 pp — still wins, barely" must
   become "significant under a regular rhythm; **no longer detectable** under `af`; the collapse
   itself is not statistically significant (p = 0.22)".
3. **§10.1 / §10.3:** all EF arm-comparisons are directionally consistent and individually null —
   the load-bearing evidence is the PSNR set in §2.4 here.
4. **§9.4 / §10.2:** degeneracy is explained (§3 here); the window-open-phase suspicion is closed,
   negative.
5. Nothing in docs/110 changes. The simulator is not implicated in any of the three.

## 5. Open after this

- The **`hrv` positive-sign correlation** (+0.310, p = 0.080) is unexplained. Probably noise at
  n = 33 with a tiny imbalance range (0.091–0.200), but it is the one unexplained sign in §1.
- EF is underpowered for every contrast that matters here. If the EF claims need to be load-bearing
  in the paper, that argues for **more subjects** (the other 6 sources, docs/111 §6 item 4 — still
  undecided) rather than more analysis of these 33.
- Neither `empty_frac` nor `imbalance` being significant leaves the *quantitative* size of the
  coverage effect unestimated. The intervention establishes the direction and that it is real; it
  does not give a dose-response.
