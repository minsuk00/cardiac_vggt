# 45 — One-frame ablation at ep50–60: what held, what didn't, from re-running docs/44

> **TL;DR & takeaway** (2026-07-16, prove-it-audited — numbers and wording corrected). Re-ran the docs/44
> ablation on the **resumed ep44–60 checkpoints** (was ep25–39) to test which findings replicate with more
> training. **The mechanistic findings HELD and sharpened; the marginal small-n statistical ship-decisions did
> NOT.** Specifically: **(1) the OOD-relocation prediction is SUPPORTED** — the MIITT clean-arm relocation shrank
> **8.9 mm (ep39) → 7.6 mm (ep60)** within the same series (paired p=0.015, 9/13 subjects), same z-shape,
> consistent with the docs/42 ep100 value (4.2 mm, a *separate* run). The shrink is real and consistent with a
> training-maturity effect; **whether the residual is a domain-shift artifact or the model partially correcting
> real MIITT breath-hold drift remains UNTESTED** (docs/44 §8 candidate 2 — the disambiguating experiment has
> not been run). **(2) C1's mechanism sharpened** — without the gather loss, breathing estimation *degrades with
> training* (no_gather EPE 2.4→3.4 mm, slope 0.72→0.41) while the hub keeps improving, so the placement
> separation GREW from 1.8× to **2.9×**; the gather loss prevents the model from abandoning breathing as it
> optimizes the breathing-insensitive reconstruction loss. **(3) The appearance wall persists** — in-distribution
> PSNR is still flat across all six despite the breathing divergence (**90% shared error**, direct-MSE). **(4) EF
> is still pure noise** at n=29. **BUT (5) all THREE marginal ep39 OOD "wins" (C1, C2, C5) failed to replicate at
> ep60.** They rested on marginal (p≈0.008–0.02) MIITT wins at **n=13** that never survived multiple-comparison
> correction (~10 tests). At ep60 the models are near-tied on real OOD data: **C1** OOD +0.27 dB (p=0.14, 95% CI
> **[−0.10, +0.65]** — *direction replicated, underpowered to confirm or refute*, NOT "effect gone"); **C2**
> reversed sign (−0.36 dB, aug regressed on OOD); **C5** collapsed to null (+0.53 p=0.008 → +0.06 p=0.58). The
> *mechanism* still holds (OOD has **26% placement-driven unique error vs 10% in-dist**, direct-MSE), but the
> effect is below what n=13 detects (paired SEM ≈ **0.17 dB**, power ≈0.31; need n≈35). **(6) C4 (dino) is the
> cautionary tale** — its in-dist win STRENGTHENED (+0.161 → **+0.261 dB, p=0.0001, robust**) while OOD stayed
> flat (p=0.75): more training widened the in-dist/OOD gap = overfits training geometry. **Net: robust
> in-distribution effects (n=30: C4 win, C3 loss) replicate; every marginal n=13 OOD win does not. docs/44's
> "C1/C2 SHIP on a significant OOD win" was the fragile call — downgraded to "mechanism real, ship-decision
> unproven at current OOD power (need n≫13 or convergence)."**

Companion to docs/44 (the ep25–39 read this revises), docs/43 (the design), docs/42 (the OOD harness + the
docs/42-ep100 relocation datapoint). Checkpoints: `scratch/checkpoints/20260716_1frame_*` (ep44–60), kept
alongside the `20260715_*` ep25–39 set for the progression comparison. wandb: `minsuk-choi/vggt-mri`.

---

## 1. What was re-run, and why

docs/44 was explicitly **mid-training** (ep25–39). Several of its conclusions were flagged for re-check at
ep100 — most importantly (a) my falsifiable prediction that the OOD relocation shrinks with training, and
(b) whether the marginal n=13 OOD ship-wins (C1, C2) would hold. The six runs resumed and reached **ep44–60**;
this doc re-runs the identical eval/analysis on those checkpoints and compares to docs/44.

| variant | ep39 ckpt | **ep60 ckpt (now)** |
|---|---|---|
| gather05 (hub) | 39 | **60** |
| no_gather | 37 | **57** |
| contz | 39 | **59** |
| dino_ft | 33 | **50** |
| aug_moderate | 39 | **59** |
| lowdiff100 | 25 | **44** |

Epoch spread is still uneven (44–60); lowdiff (44) and dino (50) lag, so their comparisons keep an epoch
confound (smaller than docs/44's, but present).

## 2. SUPPORTED: the OOD relocation shrinks with training (my falsifiable prediction)

docs/44 §8 predicted the MIITT clean-arm relocation (predicted Δz on un-breathed input) shrinks with training.
Measured on the same 13 MIITT subjects, hub model:

| training epoch | mean \|clean Δz\| | vs z-plane |
|---|---|---|
| ep39 (this series) | **8.85 mm** | r=−0.73 |
| **ep60 (this series, NEW)** | **7.61 mm** | r=−0.77 |
| ep100 (docs/42, **separate run**) | 4.23 mm | r=−0.46 |

Within the same series it dropped **8.85 → 7.61 mm** over 21 epochs — and this is **paired-significant**
(per-subject ep39→ep60: t-test p=0.015, Wilcoxon p=0.027, 9/13 subjects dropped), stronger than the two bare
aggregate numbers suggest. Same z-shape (peak at z2–3, ≈0 beyond z6). **The prediction is supported: the
relocation is real, z-structured, and shrinks with training.**

**Two honest caveats (prove-it):**
- **The ep100 = 4.23 mm point is a DIFFERENT training run** (docs/42's `vggt_20260713_gather05`), not this one
  continued. It corroborates cross-series but must not be read as the third point of one trajectory (the
  figure's connecting line is illustrative only), and 4.2 mm is a floor, not zero.
- **"Shrinks with training" (measured) ≠ "it's a domain-shift artifact" (the CAUSE, NOT established).** docs/44
  §8 candidate 2 — MIITT gated is a *real* acquisition whose slices are minutes apart, so the nonzero clean-arm
  Δz could be the model *correctly* fixing real inter-slice drift, in which case the shrink would read as the
  model getting *worse* at that (opposite sign). The disambiguating experiment (does the recon become more or
  less anatomically coherent than the input stack) **still has not run**, so the cause stays open. The one real
  argument *for* artifact — the +13 to +22 mm at z2–3 far exceeds plausible breath-hold drift — is suggestive,
  not conclusive. Figure: `result/1frame_series/figs_ep60/reloc_vs_epoch.png`.

This matters for §5: because the ep39 OOD PSNR differences were partly separated by different per-model
relocation magnitudes, shrinking the relocation makes the models converge on OOD.

## 3. CONFIRMED + SHARPENED: C1 mechanism, the appearance wall, EF noise

**C1 breathing separation GREW** (wandb, last-epoch each): without the gather loss, breathing *degrades* with
training.

| run | breathing slope (ep39→now) | EPE mm (ep39→now) |
|---|---|---|
| gather05 (hub) | 0.852 → **0.905** | 1.32 → **1.14** (improving) |
| no_gather | 0.715 → **0.406** | 2.41 → **3.35** (*degrading*) |
| aug_moderate | 0.882 → **0.947** | 1.03 → **0.94** (best) |
| contz | 0.651 → **0.856** | 2.28 → **1.67** (recovered — was undertrained, docs/44 §5) |
| dino_ft | 0.768 → 0.754 | flat (never learns breathing) |
| lowdiff100 | 0.837 → 0.824 | flat (ep45) |

The hub-vs-no_gather EPE ratio went 1.8× → **2.9×**. The gather loss doesn't just boost breathing once — it
**prevents the model from abandoning breathing correction** as it optimizes the (breathing-insensitive)
reconstruction loss. This is a cleaner C1 mechanism than docs/44 had.

**The appearance wall persists.** In-distribution `recov`/`bbox`/`motion` PSNR are still clustered across all
six at ep60 (bbox 27.2–29.0), and no_gather sits *at/above* the hub on appearance (bbox 28.85) despite its
breathing collapsing to slope 0.41. The 89%-shared-error finding is confirmed with more training.

**EF is still pure noise** at n=29: hub range 0.08–0.63, all runs overlapping (lowdiff now dips to −0.23). No
between-run signal emerged — power-limited, not undertrained, exactly as docs/44 predicted. nnU-Net EF/Dice
remains not worth running.

## 4. C4 (dino) — the cautionary tale, confirmed and stronger

dino, paired per-subject on breath PSNR (dino ep50 vs hub ep60):

| cohort | ep39 | **ep60** |
|---|---|---|
| in-dist (n=30) | +0.161 dB, p=0.004, 22/30 | **+0.261 dB, p=0.0001, 24/30** |
| real OOD (n=13) | +0.067 dB, p=0.71 | +0.060 dB, **p=0.75, 6/13** |

More training **widened** the in-dist/OOD divergence: the in-distribution win got *more* convincing while OOD
stayed a coin-flip. dino overfits the training geometry. **Judging C4 on in-distribution PSNR alone — as
docs/43 §2 specified — would ship a change that provably does not transfer**, and the in-dist signal is now
even more tempting (p=0.0001). This is the single clearest "trust the OOD arm" lesson in the series.

## 5. THE MARGINAL OOD SHIP-WINS DID NOT HOLD (C1, C2, C5)

This is the main revision of docs/44. Paired per-subject, breath PSNR:

| comparison | ep39 | **ep60** | reading |
|---|---|---|---|
| **C1** gather − no_gather, OOD | +0.402, **p=0.020**, 10/13 | +0.274, **p=0.14**, 9/13 | direction held, underpowered |
| **C2** aug − gather, OOD | +0.632, **p=0.016**, 10/13 | **−0.355**, p=0.27, 6/13 | reversed sign |
| **C5** lowdiff − gather, OOD | +0.525, **p=0.008**, 12/13 | +0.063, **p=0.58**, 8/13 | collapsed to null |

**All three marginal ep39 OOD "wins" failed to hold at ep60.** By contrast the two effects from the n=30
in-distribution cohort — **C4** (dino overfit, p=0.0001) and **C3** (contz worse, p=0.0007) — replicated and
strengthened. The pattern: **robust n=30 in-dist effects replicate; marginal n=13 OOD wins do not.**

**On the power, corrected (prove-it).** The framing must use the *paired-difference* SEM, not the marginal
per-model SEM. For C1 OOD ep60 the paired SEM is **0.17 dB** (not the ~0.5 dB an earlier draft cited), giving a
95% CI of **[−0.10, +0.65] dB** — which includes both 0 *and* the ep39 value (+0.40). So the honest statement
is **not** "the effect is gone": C1's direction *replicated* (+0.40→+0.27, same sign, 9/13) and the re-test is
simply **underpowered** (power ≈0.31 for a 0.27-dB effect; you'd need **n≈35**). C2 is genuinely different — it
*reversed sign* — and C5 *collapsed to null* (and was epoch-confounded, being the least-trained model). Only
for C2/C5 is "did not hold" a positive claim; for C1 it is absence of evidence.

**Multiplicity (prove-it).** At ep39 there were ~5 factors × 2 cohorts = ~10 primary tests; neither C1 (0.020)
nor C2 (0.016) nor C5 (0.008) survives a Bonferroni correction (α=0.005). **docs/44's "SHIP on p≈0.02" was
never robust to multiplicity** — the fragile call was the *original* ship, and ep60's non-replication
retroactively vindicates the skepticism. ep39 and ep60 are two correlated estimates of one quantity and should
be *pooled*, not played "ship then non-replication."

**The mechanism still holds (direct-MSE):** OOD reconstruction has **26% placement-driven unique error vs 10%
in-distribution** (~2.6×), and the direction still favors gather (+0.25 dB). What does not hold is that this
converts to a *statistically robust* OOD PSNR win at n=13.

**The convergence is measured; its cause is a hypothesis.** The models do converge on OOD with training
(16.15/15.75 → 16.52/16.25), but attributing that to the shrinking relocation is **untested** — a simpler
sufficient explanation is that more training improves OOD for everyone (no_gather, 3 epochs behind, improved
*more*, closing C1). Convergence = fact; relocation-drives-convergence = plausible, not established.

| model | MIITT breath PSNR (ep39 → ep60) |
|---|---|
| gather05 | 16.15 → 16.52 (+0.37) |
| no_gather | 15.75 → **16.25 (+0.50)** — improved *more* than the hub, closing C1 |
| aug_moderate | 16.79 → **16.17 (−0.62)** — the only model that *regressed* on OOD, flipping C2 |
| dino_ft | 16.09 → 16.58 (+0.49) |

**Revision:** docs/44's **C1/C2 SHIP** (justified by a "significant OOD win") are downgraded to **"mechanism
real, ship-decision unproven at current OOD power."** Deciding them needs a larger OOD cohort (n≫13) or
convergence, not a marginal n=13 p-value. aug additionally shows a directional *regression* on OOD with
training (16.79→16.17) worth watching — possible over-augmentation
hurting real-data transfer late in training.

## 6. C3 and C5 — completed

Both evals finished (30 CMRx / 13 MIITT each). Paired per-subject, breath PSNR:

| comparison | in-dist (n=30) | real OOD (n=13) | verdict |
|---|---|---|---|
| **C3** contz − hub | **−0.307 dB, p=0.0007** (worse) | **−0.507 dB, p=0.036** (worse) | **NO-SHIP** — worse on both |
| **C5** lowdiff − hub | −0.071, p=0.086 (null) | +0.063, **p=0.58** (null) | **NULL** — ep39 OOD win did not hold |

- **C3 (contz):** significantly worse on *both* cohorts, including off-grid MIITT where continuous-z was
  supposed to have its advantage. NO-SHIP. **Caveat (docs/44 §5):** contz's *oracle* is ~3 dB handicapped
  (2-plane z-blended inputs destroyed before the model sees them + off-grid double-resample), so part of the
  OOD gap is structural, not generalization — but it loses even where it should win, so the verdict is clear.
- **C5 (lowdiff100):** NULL on both, and its ep39 OOD "win" (+0.525, p=0.008) **collapsed to +0.063, p=0.58** —
  confirming it was the epoch confound (lowdiff was the least-trained model). It is now the third marginal
  ep39 OOD win to evaporate (§5).

## 7. Status / open items

- **The meta-lesson:** re-running at a new epoch is the cheapest way to catch marginal findings. Every
  *mechanistic* claim in docs/44 replicated and sharpened; every claim that rested on a marginal n=13
  p-value did not. Trust effect sizes and mechanisms over small-n significance — and trust the OOD arm over
  the seductive in-distribution one (C4).
- Everything is still **mid-training** (ep44–60 of 100). The clean test of C1/C2 remains a larger OOD cohort
  or ep100.

Tools (all offline, date-parametrized): `tools/{pull_1frame_series,fig_reloc_vs_epoch,compare_recons_1frame,
gather_benefit_vs_depth}.py`; the ep60 verdicts recompute from `scratch/eval/*/out/*/vggt_20260716_1f_*`.
