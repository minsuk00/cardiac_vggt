# 113 — AF: cut-off vs hold — which mechanism costs EF? (null result)

> **TL;DR & takeaway**
> The shipped `af` arm mixes two mechanics — short beats are **cut off**, long beats **hold** —
> and they are anti-correlated (r = −0.50), so `af` alone cannot say which one hurts EF. Two arms
> isolated them on cmrx2024 (12-frame, n = 30 paired): `af_rvr` (all short beats, only cut-offs)
> and `af_pause` (all long beats, only holds). **Result: null.** `af_pause − af_rvr` = **+0.24 pp
> EF MAE, CI [−2.56, +3.42], p = 0.52**. Either mechanism alone reproduces the AF effect; neither
> is necessary; they cannot be ranked. `af_rvr` is additionally **confounded** (its GT EF drops
> 6.8 pts), so it must not be reported as a clean cut-off effect. Pooled over the three AF arms,
> AF does damage Fetal 4D's EF (+1.71 pp, Wilcoxon p = 0.016) while VGGT is flat.
>
> ⚠️ **Provenance:** the campaign ran in the 2026-09-20 session; the numbers below are
> **transcribed from that session's handoff (`_agent/handoff-2026-09-20-1537.md`) and were NOT
> re-derived when this doc was written (2026-09-21)**. All of it is on the **12-frame, pre-hold-fix
> position model** (docs/114) and is superseded for absolute values by the 24-frame campaign
> (docs/115). Bundles: `scratch/eval/_archive_af_disambig_20260920/`.

## 1. Design

| arm | R-R draw (`rr_sequence`) | measured cut-offs / slice | hold_frac |
|---|---|---|---|
| `af` | N(1.0, 0.25), floor 0.45 | ≈0.44 (derived: the builder comment gives `af_rvr` = 2.5× `af`) | 0.131 |
| `af_rvr` | N(0.60, 0.10), ≤ 0.85 — rapid ventricular response | 1.10–1.12 | 0.054 |
| `af_pause` | N(1.20, 0.15), ≥ 1.03 — slow response / pauses | 0.00 | 0.177 |

Tuning notes (in `tools/build_af_bundle.py` next to the draws): short beats **cannot** lower phase
coverage (a short beat starts a new beat and keeps sampling new phases; only a hold re-samples the
same one), so `af_rvr` can only test "do cut-offs cost EF at constant coverage". `af_pause` at
μ = 1.30 made 14/38 subjects degenerate; μ = 1.20 keeps the degenerate rate at `af`'s own 5/38.

Fetal CMR 4D recons 76/76, VGGT 76/76, PSNR + EF scored.

## 2. Results (EF MAE, pp, paired n = 30)

| cohort | EF MAE |
|---|---|
| `regular_frozen` | 6.23 |
| `af_rvr` | 7.52 |
| `af_pause` | 7.75 |
| `af` | 8.52 |

- **`af_pause − af_rvr` = +0.24 pp, CI [−2.56, +3.42], p = 0.52** — indistinguishable.
- Adding the second mechanism to either arm does nothing (p = 0.95, p = 0.61); no interaction
  (p = 0.36).
- **Pooled AF effect:** Fetal +1.71 pp (Wilcoxon p = 0.016); VGGT flat (p = 0.50);
  difference-in-differences +2.91 pp (p = 0.036); the `hrv` control is clean (p = 0.75).
  ⚠️ The pooled Fetal effect's bootstrap mean CI straddles zero ([−0.91, +3.95]) — heavy-tailed;
  the rank test is the appropriate one.
- Fetal's EF advantage over VGGT: significant under `regular_frozen` (+3.84, p = 0.008) and `hrv`
  (+3.58, p = 0.043); **not significant under any AF arm** (+0.40 / +0.63 / +1.76).

## 3. The `af_rvr` confound

Short beats cannot fill, so `af_rvr`'s **GT** changes: GT EF −6.8 pts (EDV −11 %). VGGT predicts a
near-constant 51–55 EF regardless of cohort, so its error **shrinks mechanically** when the target
moves toward that constant. `af_pause` is the uncontaminated arm (GT EF 61.4 vs clean 62.3).

## 4. What this licenses

- "AF degrades Fetal 4D's EF; the degradation does not depend on which of the two mechanics is
  present." Not: "cut-offs are harmless" or "holds are the cause".
- The question is closed for the 12-frame design. The measured strong lever is frames-per-slice,
  not rhythm parameters (docs/115) — so neither arm is carried into the 24-frame campaign.
