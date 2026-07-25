# 46 — One-frame ablation at ep100: final 4-cohort eval (in-dist + 3 OOD gated)

> **TL;DR & takeaway** (2026-07-21). Re-ran the docs/44 (ep39) / docs/45 (ep60) ablation on the
> **converged ep100** checkpoints and, for the first time, across **four cohorts**: in-distribution
> **CMRx (30-subject in-dist set)** + three OOD gated cohorts **MIITT (13) / OCMR (8) / ACDC (40, pathology)** — all
> pushed through the SAME frozen-breathing head-to-head harness (`run_vggt.py`), so every number is
> apples-to-apples. **The bigger OOD sample (pooled n=61) resolves the underpowered questions docs/45
> left open, and it flips docs/45's headline: (C2) data augmentation is a ROBUST OOD WINNER** — pooled
> OOD **+0.41 dB breath PSNR, p<0.001** (MIITT +0.48 p=0.04, ACDC +0.47 p=0.002), at a small in-dist
> cost (−0.13 dB). docs/45 had called C2 a non-replication *because it only had the n=13 MIITT cohort*;
> with MIITT+OCMR+ACDC the effect is unambiguous. **(C4) unfreezing DINO is a mild UNIFORM winner**
> (+0.19 dB pooled OOD, p=0.04; +0.21 in-dist) — positive on every cohort (revises docs/45's "overfit"
> call), but the pooled p=0.04 is not correction-robust: direction-robust, significance fragile (§9). **(C1) the gather loss stays a keep** (helps in-dist −0.15 dB & ACDC, neutral
> pooled OOD; and it remains decisive for *breathing* quality per docs/45). **(C3) continuous-z is
> NO-SHIP** (worse both arms: in-dist −0.40, pooled OOD −0.34, both p<0.05; partly its ~structural
> oracle handicap). **(C5) less diffusion reg is NULL** (pooled OOD +0.08, p=0.19). **The appearance
> wall persists at convergence** — within every cohort all six models sit within **~1 dB** breath
> PSNR (0.27 OCMR / 0.61 CMRx / 0.81 ACDC / 1.04 MIITT; prove-it-corrected from an earlier "≤0.7 dB"
> that MIITT+ACDC exceed), so *in-distribution* placement is still not the reconstruction bottleneck
> (in-dist breathing-slope ⊥ PSNR, r=−0.09; on OOD it is not identifiable — a coexisting placement axis
> co-exists but is dwarfed by the cohort floor — §9). **Cohort difficulty:** CMRx 21.0 > ACDC 19.1 > OCMR 18.7 > MIITT 16.0 dB. **OOD
> relocation is real and cohort-dependent** (clean-arm predicted Δz: in-dist 0.18 mm vs MIITT 10.3,
> ACDC 4.2, OCMR 4.1 mm). **OOD EF/Dice is now RUN** (nnU-Net Task114, hub only, §7): the clinical
> readout splits in two. **Absolute EF is not usable** — under-predicted by **19.7–29.2 points on
> every cohort** (ACDC 45.6→25.9, MIITT 62.6→36.1, OCMR 59.8→38.9), the 1-frame under-contraction
> ceiling showing up exactly as in-distribution. **Relative ranking survives only where EF actually
> varies**: ACDC (pathology, GT EF sd 19.0, range 11.7–74.2) gives **slope 0.52 / Spearman +0.74**
> clean, while MIITT and OCMR (sd ~6.7–6.9, healthy/narrow) collapse to ρ +0.47 / +0.07 — consistent
> with restriction of range, so **OCMR's ~0 ρ is NOT evidence the model fails there**. Segmentation
> transfers respectably (ACDC Dice ED LV 0.84 / MYO 0.65 / RV 0.74 clean). Breathing costs a modest,
> consistent amount on the one well-powered cohort (ACDC ρ 0.74→0.64, Dice LV 0.835→0.800).
> In-distribution EF (wandb, slope ~0.4–0.5, n=29, power-limited) is the 1-frame regime's lower
> ceiling vs the multiframe 0.77 (docs/33). **vs classical SVR baselines (SVRTK/NeSVoR, cmrx+miitt,
> §10): VGGT wins the breathing task on PSNR/SSIM/NCC** (cmrx +3–4 dB) — SVR reconstructs a *clean*
> stack better (SVRTK 28 dB) but has no breathing model, so it collapses under respiratory corruption
> (−10.6 dB) while VGGT barely moves (−1.4 dB). **Post-hoc 3-agent audit (§9): every number reproduces
> exactly, C2-SHIP robust; "placement is not the bottleneck" scoped to in-distribution; verdict
> yes-with-caveats.**

Companion to docs/44 (ep39), docs/45 (ep60), docs/43 (design), docs/42 (frozen OOD harness).
Checkpoints: `scratch/checkpoints/20260719_1frame_*_ep99.pt` (prev_epoch=99 = the 100th epoch).
wandb `minsuk-choi/vggt-mri`. All six runs finished training at ep100 (`state=finished`).

---

## 1. What was run

The 6 one-frame runs (hub `gather05` + C1–C5) at their **final ep100** checkpoints, evaluated on 4
cohorts × {clean, breath} through the frozen-bundle harness (`run_vggt.py --regime onef`), scored by
the shared `assemble_and_gif.py` (PSNR/SSIM/NCC over the heart&FOV ROI) + `resp_diag.json`
(predicted Δz vs applied). **New this round:** OCMR-gated and ACDC were brought INTO the harness
(they weren't before) — see §5. Every cohort therefore uses the identical breathing recipe, target,
ROI, and metric as the docs/44/45 CMRx+MIITT numbers.

## 2. Intensity — the 4-cohort table (breath PSNR, dB)

| model | CMRx (30) | MIITT (13) | OCMR (8) | ACDC (40) |
|---|---|---|---|---|
| gather05 (hub) | 21.01 | 15.98 | 18.74 | 19.12 |
| no_gather | 20.86 | 16.05 | 18.74 | 18.92 |
| **aug_moderate** | 20.89 | **16.45** | 18.78 | **19.59** |
| contz | 20.61 | 15.42 | 18.77 | 18.78 |
| **dino_ft** | **21.22** | 16.16 | **19.00** | 19.29 |
| lowdiff100 | 20.94 | 16.31 | 18.73 | 19.13 |

**The appearance wall, at convergence:** within each cohort the six models span **≤1.0 dB** breath
PSNR (max−min: OCMR 0.27, CMRx 0.61, ACDC 0.81, **MIITT 1.04**). This is the docs/44/45 thesis,
confirmed at ep100 **in-distribution** — where breathing-estimation slope is uncorrelated with recon
PSNR (across-model r=−0.09) and placement error is ~0.1–0.4 mm, so placement quality varies while
breath PSNR barely moves. **On OOD the thesis is not identifiable** (breathing slope, relocation and
PSNR are collinear across the six models); a genuine coexisting placement axis is present there but
explains only the ≤1 dB band, dwarfed by the 2–5 dB cohort floor — see §9. Cohort difficulty: **CMRx 21.0 > ACDC 19.1 > OCMR 18.7 > MIITT 16.0**. (MIITT shows a
*negative* breathing "cost" — breath>clean — the known MIITT clean-arm oracle handicap, docs/45.)

**SSIM/NCC corroborate the PSNR ablation** (breath, heart ROI): the same ≤1 dB within-cohort wall and
the same aug-best-OOD / dino-best-in-dist ranking hold on both — e.g. aug leads MIITT SSIM 0.718 /
NCC 0.724 and ACDC 0.790 / 0.794; contz is worst; dino best in-dist (SSIM 0.881, NCC 0.887). So the
C1–C5 verdicts are not a single-metric artifact (full baseline-inclusive PSNR/SSIM/NCC table in §10).

## 3. The C1–C5 verdicts — paired per-subject, in-dist vs pooled OOD

Paired (by subject) Δ breath PSNR vs the hub; **pooled OOD = MIITT+OCMR+ACDC (n=61)** — the power
docs/45 lacked. `fig_verdicts.png`.

| comp | change | in-dist (n=30) | pooled OOD (n=61) | verdict |
|---|---|---|---|---|
| **C1** | no_gather − hub | −0.15 (p=0.013) | −0.12 (p=0.11) | **keep gather** — helps in-dist & ACDC, neutral pooled; decisive for breathing (docs/45) |
| **C2** | aug − hub | −0.13 (p<0.001) | **+0.41 (p<0.001)** | **SHIP** — robust OOD winner (MIITT+ACDC both sig), small in-dist cost |
| **C3** | contz − hub | −0.40 (p<0.001) | −0.34 (p=0.013) | **NO-SHIP** — worse both arms (partly structural handicap) |
| **C4** | dino − hub | +0.21 (p<0.001) | +0.19 (p=0.042) | **mild, direction-robust (significance fragile)** — positive everywhere (revises docs/45 "overfits") but pooled p=0.04 fails correction (§9) |
| **C5** | lowdiff − hub | −0.08 (p=0.004) | +0.08 (p=0.19) | **NULL** — MIITT-only blip (+0.33), not replicated on OCMR/ACDC |

**The meta-lesson realized.** docs/45 predicted these marginal OOD calls needed "n≫13 or
convergence." Both arrived: **C2 flips from ep60 non-replication to a decisive multi-cohort ship**,
and **C4's OOD gain becomes significant** — precisely because pooling three OOD cohorts gives real
power. The single-cohort n=13 verdicts of docs/45 were, as it warned, underpowered — not wrong in
direction (C2/C4 were always directionally positive) but unresolvable at that n.

**Stats caveat (prove-it).** The pooled-OOD test is a 1-sample t-test on the 61 per-subject paired
diffs — it treats subjects across MIITT/OCMR/ACDC as exchangeable, ignoring cohort clustering, and is
weighted toward ACDC (n=40). This *inflates* the pooled variance (p-values are if anything
conservative, so it won't manufacture a false positive), but it is an unweighted-across-cohort
estimand. **C2 survives regardless** — independently significant on MIITT (p=0.04) AND ACDC (p=0.002),
not a pooling artifact. **C4 leans more on the pool** (no single OOD cohort reaches p<0.05 alone); the
3-agent audit (§9) sharpened this — C4's pooled p=0.042 dies (p=0.091) when ACDC's 66% weight is
removed and fails Bonferroni/BH, so its direction is robust but "significant" is not.

## 4. Breathing & OOD relocation (hub, `resp_diag.json`)

| cohort | breathing slope →1 | clean-arm relocation (mm) |
|---|---|---|
| CMRx (in-dist) | 0.83 | 0.18 |
| MIITT | 0.78 | **10.3** |
| OCMR | 0.70 | 4.1 |
| ACDC (pathology) | **0.58** | 4.2 |

Breathing amplitude fidelity degrades on OOD, worst on the pathology cohort (ACDC 0.58). The
clean-arm relocation (predicted Δz on *un-breathed* input — the domain-shift displacement, docs/45
§2) is ~0 in-dist and **~4–10 mm on OOD, largest on MIITT**. (Caveats, prove-it: **the OCMR mean 4.1
is outlier-driven — median 1.48 mm, one subject at 11.5** — so *typical* OCMR relocation is small,
the smallest of the three OOD cohorts; and do not generalize from any single subject either way.)
EPE/relocation numbers on
OOD are relocation-contaminated; the *slope* is the offset-robust read (docs/45).

## 5. Method — bringing OCMR/ACDC into the frozen harness

`inference/run_gated_ood.py` (which supports ocmr/acdc) is **not comparable** to the frozen CMRx/MIITT
numbers: it re-samples breathing via `gpu_augment_batch` (different realization), computes
training-space PSNR (no NCC, no `resp_diag`), and its layout is incompatible with the scorer
(`docs/42:34`). So OCMR/ACDC were built into the frozen harness instead:
- **`scratch/eval/ocmr/build_inputs.py`** (8 SAX exams from `scratch/data/ocmr/recon/gated/`, per-subject
  spacing from `meta.json`) + **`scratch/eval/acdc/build_inputs.py`** (40 patients, 8/pathology-class;
  ACDCGatedAdapter reorients cine to LPS, and **`heart_roi`/`heart_seg` are reoriented the SAME way** or
  the masks misalign — the ACDC LPS trap, CLAUDE.md). Both mirror `miitt/build_inputs.py` (name-hash
  seed, one per-subject breath, VGGT-identical canonical GT bundle).
- **`run_vggt.py`**: added `ocmr`/`acdc` datasets + a shared `_prep_gated_native` (native slices →
  canonical placement via the adapter — identical machinery to `prep_miitt`).
- Eval driver `run_1frame_series_v3.sh`; scoring `score_1frame_series.sh` (extended to 4 cohorts).

**Validation** (per verification plan): OCMR gather05 → PSNR 19.5/18.5, resp slope 1.11, EPE 0.44 mm;
ACDC p001 (DCM) renders a correct LV donut with a co-located mask (LPS trap handled); in-dist
gather05 Test_P012 → slope 0.98, EPE 0.68, clean relocation 0.06 mm (negative control passes).

## 6. In-distribution EF (wandb `val/ef/*`, n=29, mean of last ~8 val-EF epochs)

| model | EF slope | Spearman | MAE% |
|---|---|---|---|
| dino_ft | 0.54 | 0.36 | 11.5 |
| lowdiff100 | 0.49 | 0.32 | 12.7 |
| no_gather | 0.45 | 0.32 | 12.2 |
| gather05 | 0.40 | 0.28 | 11.4 |
| aug_moderate | 0.39 | 0.27 | 11.3 |
| contz | **0.20** | 0.17 | 14.1 |

EF slope ~0.4–0.5 — the **1-frame regime's lower ceiling** (multiframe reference was ~0.77, docs/33),
exactly as docs/43 predicted. **Power-limited at n=29** (hub's own epoch-to-epoch range swallows most
between-model gaps, docs/45) → EF is a capability readout, **not** a C1–C5 discriminator. The one
clear signal: **contz halves EF slope** (0.20), consistent with its structural handicap. **These
digits are the only numbers in this doc not reproducible from on-disk artifacts** (wandb pull,
smoothing/epoch-window dependent, §9) — only the ordering and the "contz collapses" reading are robust.

## 7. OOD EF / Dice (nnU-Net Task114, hub `gather05` only, n=61)

SLURM job 54241019 (spgpu/A40, 1 h 42 m). Per-phase recon + GT volumes → nnU-Net Task114 2d
(`-tr nnUNetTrainerV2_MMS`) → EF = (LV_max−LV_min)/LV_max. **EF is a ratio, so the 12 mm-pitch
voxel-volume caveat (docs/39) cancels** — no bias. Absolute EDV/ESV would need the pitch and are
not reported. Hub only: EF is power-limited at these n and is a *capability readout, not* a
C1–C5 discriminator (same call as §6).

| cohort | n | arm | EF slope | Spearman ρ | EF MAE% | Dice ED LV / MYO / RV |
|---|---|---|---|---|---|---|
| MIITT | 13 | clean | 0.647 | +0.467 | 26.5 | 0.749 / 0.526 / 0.727 |
| | | breath | 0.849 | +0.346 | 29.2 | 0.808 / 0.571 / 0.761 |
| OCMR | 8 | clean | 0.127 | +0.071 | 20.9 | 0.810 / 0.677 / 0.794 |
| | | breath | 0.184 | +0.310 | 22.7 | 0.831 / 0.637 / 0.794 |
| **ACDC** | 40 | clean | **0.523** | **+0.739** | 19.7 | 0.835 / 0.652 / 0.741 |
| | | breath | 0.451 | +0.642 | 21.4 | 0.800 / 0.577 / 0.724 |

![OOD EF](../result/1frame_ep100/fig_ood_ef.png)

**Two separate findings, don't conflate them.**

**(a) Absolute EF is not clinically usable.** Every cohort is under-predicted by a large, consistent
margin — GT vs predicted (clean): ACDC 45.6→25.9 (**−19.7**), MIITT 62.6→36.1 (**−26.5**), OCMR
59.8→38.9 (**−20.9**). This is the same 1-frame under-contraction as in-distribution (§6 slope
0.4–0.5), not an OOD artifact: the model recovers *timing* but compresses *amplitude* toward the
cohort mean (the long-running finding of docs/24/25/33, here at the one-frame extreme).

**(b) Per-patient ranking holds only where GT EF actually varies.** Measured GT spread:

| cohort | GT EF mean | sd | range | → Spearman (clean) |
|---|---|---|---|---|
| ACDC | 45.6 | **19.0** | 11.7–74.2 | **+0.739** |
| MIITT | 62.6 | 6.7 | 48.6–71.8 | +0.467 |
| OCMR | 59.8 | 6.9 | 50.8–70.6 | +0.071 |

The two narrow-spread cohorts are exactly the two with weak ρ, and the wide-spread pathology cohort
is the one that ranks well. That is textbook **restriction of range**, and it matches the known
cohort composition (ACDC pathology-labeled; OCMR volunteers; MIITT mixed — see
`reference_dataset_cohort_composition`). **Caveat: this is a pattern across only 3 cohorts —
well-supported, not proven.** The operational consequence is the important part: **OCMR's ρ≈0 is NOT
evidence the model fails on OCMR.** At sd 6.9 with n=8 there is almost nothing to rank, and OCMR's
*intensity* PSNR is mid-pack (18.7 dB, §3). Do not cite OCMR EF as a negative result.

**Breathing cost.** On the only well-powered cohort (ACDC): ρ 0.739→0.642, slope 0.523→0.451, Dice
LV_ED 0.835→0.800 — a real but modest degradation, consistent with the intensity arm. MIITT's breath
slope *exceeding* clean (0.849 vs 0.647) alongside a *lower* ρ is noise at n=13, not an improvement.

**Segmentation transfers.** ACDC Dice ED LV 0.84 / RV 0.74 on reconstructions the network never saw
in training is a meaningful sanity check that the recon is anatomically well-formed even where EF
amplitude is compressed — i.e. the failure is contraction magnitude, not structure.

## 8. Status / open items

- **DONE:** 4-cohort intensity (PSNR/SSIM/NCC) + breathing + relocation + paired C1–C5 verdicts +
  in-dist EF (§6) + **OOD EF/Dice for the hub (§7)**.
- **Not run (deliberate):** OOD EF/Dice for the other five models. EF is power-limited at these n
  (§6/§7), so a per-model OOD EF table would not discriminate C1–C5 — the intensity verdicts (§4)
  remain the ship-decision evidence. Run only if a specific model's clinical readout is in question:
  `METHOD=vggt_20260719_1f_dino_ft_ep99 sbatch --account=jjparkcv0 sbatch/ef_dice_ood.sh`.
- Tools (offline, reproducible): `tools/{paired_verdicts_1frame,fig_ep100_summary,fig_ood_ef}.py`,
  `tools/ef_dice_1frame.py` (+ `sbatch/ef_dice_ood.sh`); bundles
  `scratch/eval/{ocmr,acdc}/build_inputs.py`; figures `result/1frame_ep100/`.
- Per-slice qualitative panels (all 12 canonical planes / all input slots, GT-vs-recon cycle GIF +
  predicted Δz): `scratch/eval/engine/analysis/fig_slice_panels.py` → `result/slice_panels_final/`.

## 9. Post-hoc audit — 3-agent debate (2026-07-23)

A three-auditor debate (statistical-rigor / causal-logic / data-reproduction lenses) re-derived every
§2–§7 number from the raw on-disk artifacts and stress-tested the conclusions. **All numbers reproduced
exactly; no code bugs; C2/aug SHIP is statistically robust** (survives Bonferroni; independently
significant on MIITT and ACDC). Four claims were tightened — none overturn a decision:

1. **"Placement is not the bottleneck" is an in-distribution result.** In-dist: breathing-slope ⊥ PSNR
   (across-model r=−0.09), relocation ~0.1–0.4 mm — the wall is real. On OOD: breathing slope,
   relocation and PSNR are **collinear** across the six models, so neither "placement irrelevant" nor
   "placement dominant" is identifiable. A **coexisting placement axis** is genuinely present on OOD
   (clean-arm relocation vs recon PSNR, ACDC within-subject r=−0.76, n=40, p<1e-7) but explains only
   the ≤1 dB band, dwarfed by the 2–5 dB cohort floor. *(An initial "the wall reverses on OOD" reading
   was withdrawn: the strong n=6 across-model relocation↔PSNR r≈−0.9 is a `contz` leverage artifact —
   `contz`'s low PSNR is appearance-side from its z-blend — and collapses to n.s. on MIITT/OCMR once
   `contz` is dropped; and relocation is a model output, so the correlation is subject-difficulty-
   confounded, not causal.)*
2. **The C2 win is real but its mechanism is unidentified** — aug improves both appearance robustness
   and slice placement (it also cuts ACDC relocation, 2.28 vs hub 4.19 mm), so don't credit the gain to
   appearance alone.
3. **C4's pooled significance is not correction-robust** — direction uniformly positive, but p=0.042
   dies (p=0.091) without ACDC's 66% pool weight and fails Bonferroni/BH. Direction-robust,
   significance fragile.
4. **Framing:** §6 in-dist EF digits are wandb-only (not disk-reproducible); restriction-of-range (§7b)
   is confounded with n (larger-sd cohorts are also larger-n) and unfalsifiable at n=3 —
   well-supported, not proven; "in-distribution CMRx" is a 17-train / 10-test / 3-val mixed split, so
   absolute in-dist numbers (§2, §6) are optimistic (paired §3 verdicts immune).

**Verdict: yes, with caveats** — numbers exact, C2-SHIP / C3-NO-SHIP and the in-dist findings
conclusive; the caveats scope the OOD generalization and soften two secondary statistical claims. See
`project_doc46_audit_appearance_wall_scoping` (memory) and `_html/46_oneframe_ep100_final.html` §9.

## 10. External comparison — VGGT vs classical SVR baselines (SVRTK / NeSVoR)

Head-to-head against two classical slice-to-volume reconstruction baselines through the SAME frozen
harness (identical breathing, target, ROI, scorer). Baselines exist on **cmrx + miitt only** (SVRTK /
NeSVoR are per-subject optimizers with no training set; run on the in-dist + one OOD cohort). All
three intensity metrics, over the heart ROI:

| cohort | method | breath PSNR | breath SSIM | breath NCC | clean PSNR |
|---|---|---|---|---|---|
| CMRx | SVRTK | 17.63 | 0.754 | 0.754 | **28.23** |
| | NeSVoR | 16.73 | 0.737 | 0.764 | 22.27 |
| | **VGGT hub** | **21.01** | **0.874** | **0.881** | 22.39 |
| | VGGT aug | 20.89 | 0.870 | 0.877 | 22.28 |
| | VGGT dino | **21.22** | **0.881** | **0.887** | 22.79 |
| MIITT | SVRTK | 14.84 | 0.681 | 0.686 | 21.23 |
| | NeSVoR | 14.94 | 0.672 | 0.680 | 18.11 |
| | VGGT hub | 15.98 | 0.697 | 0.703 | 15.32 |
| | **VGGT aug** | **16.45** | **0.718** | **0.724** | 15.52 |

**The breathing-robustness gap is the story.** On *clean* input classical SVR is excellent — SVRTK
CMRx **28.2 dB / SSIM 0.98**, well *above* VGGT (22.4). But SVR has **no breathing model**, so under
respiratory corruption it collapses: SVRTK CMRx **28.2 → 17.6 dB (−10.6)**, while VGGT drops only
**22.4 → 21.0 (−1.4)**. On the realistic (breath) task VGGT therefore wins on **all three metrics,
both cohorts** (CMRx +3–4 dB PSNR, +0.12 SSIM/NCC; MIITT +1–1.5 dB). **SSIM and NCC track PSNR
throughout**, corroborating the §2/§3 intensity verdicts (aug the OOD winner, dino best in-dist). This
is precisely the paper's thesis: gating-free reconstruction from breathing-corrupted, phase-scattered
single frames — the regime where classical gated-stack SVR is not applicable.

NCC is now stored in the aggregate summaries alongside PSNR/SSIM (`evaluation/engine/aggregate.py`);
figure `tools/fig_baseline_compare.py` → `result/1frame_ep100/fig_baseline_compare.png`. Baselines are
absent on OCMR/ACDC (not run).

![VGGT vs SVR baselines](../result/1frame_ep100/fig_baseline_compare.png)
