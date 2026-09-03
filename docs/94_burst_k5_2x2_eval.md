# 94 — Burst sampling (5 sequential frames/slice): 2×2 train-regime × eval-regime result

> **TL;DR & takeaway** (2026-09-03). Trained one arm identical to the paper model (`noreg224`,
> `212682843_noreg224_pooled1337`) except the sampler feeds **`burst_k=5` temporally sequential
> frames per z-plane** (random start phase per plane, no phase labels, one shared breathing draw
> per burst) → `212280371_burst5noreg224_pooled1337` (branch `arm/burst5`, 300 ep, clean finish).
> Scored both checkpoints through the frozen-bundle harness (`evaluation/`, breath arm, val split,
> 7 sources, n=144, nnU-Net Task114 EF) under BOTH input regimes (a `--burst-k` override in
> `run_vggt.py`). **Burst training wins on every axis when fed bursts:** NCC 0.831→0.870
> (+0.037), PSNR +1.0 dB, breathing slope 0.907→0.957, EF bias −13.0→−10.6, SV ratio 0.71→0.77,
> LV Dice ED/ES +0.02 (all paired Wilcoxon p≤0.004). **The image gain is NOT phase averaging:**
> the k=1 model fed the same 5 frames gains only +0.012 NCC and *collapses* EF (bias −31, SV
> ratio 0.37, slope 0.48) — the temp-probe washout, now confirmed under breathing on the full
> cohort. **Surprise:** the burst-trained model fed only ONE frame per slice (the sparse contract
> input) nearly halves the EF bias (−13.0→−6.8, SV ratio 0.83, MAE 12.9→9.3, p<1e-8) at a small
> fidelity cost (NCC −0.014). Per-patient EF correlation holds in every burst cell (ρ 0.65–0.67
> vs 0.69). Verdict per docs/38 rule: **burst5@5 frames is a ship-quality win; burst5@1 frame is
> the EF-bias lever we did not have.** Open: why does burst training help even at 1 frame (better
> phase-appearance prior vs. regularization from the 5× larger slot set), and is the residual
> +1 dB partly a within-burst averaging component (ES Dice ↑ says mostly not).

## 1. What was compared

| | trained 1 frame/slice | trained 5 frames/slice (burst) |
|---|---|---|
| **fed 1 frame/slice** | `vggt_noreg224_ep300` (baseline, paper model) | `vggt_burst5noreg224_k1_ep300` |
| **fed 5 frames/slice** | `vggt_noreg224_k5_ep300` | `vggt_burst5noreg224_ep300` |

- Both checkpoints: `checkpoint_last.pt` (ep 299). Config diff (from the two `run_meta.jsonl`):
  `burst_k` 5 vs unset, `num_slices`/`img_nums` 106 vs 20 (slot budget for S = 1 + 5·D),
  `gradient_checkpointing` true vs false. Everything else identical (224 px, aggressive aug,
  respiratory on, no smoothness reg, gather 0.5, heart 0.5, LR 5e-5, seed 42, pooled1337 split).
- Eval regime comes from each checkpoint's own `run_meta.jsonl`; the off-diagonal cells use the
  new `run_vggt.py --burst-k` override (metadata records `burst_k_trained` / `burst_k_eval`).
  With `burst_k=1` the sampler takes the pre-existing one-frame path, so burst5@k1 sees exactly
  the baseline's input distribution.
- Inputs: the frozen breathing bundles (`evaluation/volumes/<src>/out/<subj>/breath/`), byte-
  identical across arms. In the 5-frame regime the 5 frames of one plane come from that plane's
  single frozen breathing draw (same as training: `group_by_burst`).
- Scorer: `evaluation/src/score/` unchanged (docs/93 content-keyed provenance ported to the
  branch), EF via `eval_ef_dice.sh` (nnU-Net Task114). Numbers: `tools/compare_burst_arms.py`
  → `temp/burst_eval/compare.md`; scatter: `tools/plot_burst_ef_scatter.py` →
  `temp/burst_eval/ef_scatter.png`.

## 2. Results (pooled 7 sources, n=144, median; Δ paired vs baseline, Wilcoxon)

### Image / breathing

| metric | burst5 @5 fr | burst5 @1 fr | **noreg @1 fr (base)** | noreg @5 fr |
|---|---|---|---|---|
| NCC (primary, docs/88) | **0.870** (+0.037, p=5e-25) | 0.809 (−0.014, p=2e-7) | 0.831 | 0.844 (+0.012, p=2e-17) |
| PSNR dB | **20.92** (+1.01, p=5e-25) | 19.34 (−0.40, p=1e-8) | 20.02 | 20.41 (+0.34, p=2e-17) |
| SSIM | **0.867** (+0.032) | 0.811 (−0.011) | 0.834 | 0.843 (+0.008) |
| resp slope | **0.957** (+0.037, p=3e-6) | 0.935 (+0.024, p=0.016) | 0.907 | 0.914 (+0.007) |
| resp EPE mm | 0.823 (−0.010, ns) | 0.787 (−0.054, p=0.009) | 0.805 | 0.786 (−0.042) |

### EF / volumes (nnU-Net Task114, breath arm)

| metric | burst5 @5 fr | burst5 @1 fr | **noreg @1 fr (base)** | noreg @5 fr |
|---|---|---|---|---|
| EF bias (pred−GT, median) | −10.6 (+2.0, p=0.003) | **−6.8** (+5.0, p=3e-14) | −13.0 | −31.1 (−18.3, p=2e-25) |
| EF MAE | 11.2 | **9.3** | 12.9 | 30.8 |
| EF slope / Spearman ρ | 0.78 / 0.67 | 0.84 / 0.65 | 0.83 / 0.69 | 0.48 / 0.54 |
| EDV err ml | −7.6 (+4.1) | **−6.1** (+7.0) | −13.4 | −31.5 |
| ESV err ml | 10.5 (−0.2, ns) | **7.1** (−2.8, p=2e-5) | 12.0 | 21.1 |
| SV ratio | 0.767 (+0.064, p=2e-6) | **0.834** (+0.117, p=7e-14) | 0.710 | 0.366 |
| Dice LV ES | **0.843** (+0.016, p=8e-6) | 0.825 (ns) | 0.826 | 0.787 (−0.026) |
| Dice LV ED | **0.913** (+0.019, p=4e-11) | 0.888 (+0.007, p=0.04) | 0.891 | 0.845 (−0.038) |

Per-source: burst5@5 fr beats the baseline on NCC in all 7 sources (Δ +0.03 to +0.05); EF bias
improves in 6/7 (miitt n=3 is the exception). burst5@1 fr improves EF bias in 7/7 (ocmr −8.8→−1.4,
cmrx2024 −13.9→−6.2) while NCC drops in 6/7.

## 3. Reading

1. **Training on bursts, evaluating on bursts (the intended protocol) is a clean win**: fidelity,
   breathing correction, EF bias, SV ratio and Dice all improve, none of the docs/38 vetoes fire
   (in-run `hole_frac_heart` 0.08 vs 0.26 for the baseline; ES Dice ↑ rules out "blur to the
   temporal mean" as the mechanism).
2. **The control cell settles the averaging question.** A 1-frame model fed 5 frames gets the
   "free" +0.3 dB from coverage averaging and pays for it with a 2.4× larger EF bias and slope
   0.48 — under breathing, on 144 subjects, matching the clean-input 5-subject probe of Aug 24
   (`temp/twophase_multiframe_probe/`). The burst-trained model gets 3× that NCC gain and
   improves EF, so it learned to use the temporal derivative rather than to average it away.
3. **burst5@1 frame is the unexpected result.** Same input as the paper model, EF bias halved,
   EDV error halved, ESV error down 40%, ρ held. That is the largest EF-bias movement of any
   in-contract lever tried so far (docs/91–92 arms bought ≤2 EF points). It comes with a small,
   significant fidelity cost (−0.014 NCC), i.e. the burst model is slightly burst-dependent for
   appearance but not for cardiac function. Hypotheses, untested: (a) bursts teach a sharper
   phase-appearance prior (the docs/92 "ES wall band" artifact is what shrank — ESV err 12.0→7.1),
   (b) the 5× larger slot set acts as a regularizer on the aggregator.
4. **What burst5@5 fr does NOT fix:** ESV error is unchanged vs baseline (10.5 vs 12.0, ns); the
   EF-bias gain at 5 frames is all EDV (−13.4→−7.6). So at 5 frames the docs/92 ES-surplus artifact
   is still there, and burst5@1 fr addresses it better than burst5@5 fr does — the two cells win
   on different halves of the EF error.

## 3b. Is the NCC gain a coverage effect? No. (`tools/burst_coverage_check.py`, n=144)

Hole fraction inside the scoring ROI and NCC re-scored on the voxels covered by ALL four arms
(`ncc_common`) — same subjects, same voxels:

| arm | hole_frac | NCC (ROI) | NCC (common voxels) | NCC (own covered) |
|---|---|---|---|---|
| burst5 @5 fr | 0.000 | 0.870 | **0.876** | 0.871 |
| burst5 @1 fr | 0.015 | 0.809 | 0.840 | 0.841 |
| noreg @1 fr (base) | 0.000 | 0.831 | 0.843 | 0.839 |
| noreg @5 fr | 0.000 | 0.844 | 0.854 | 0.846 |

The burst5@5 fr gain over the baseline is unchanged on equal-coverage voxels (+0.033 vs +0.037),
so it lives in the rendered intensities, not in fewer holes. The one coverage story is burst5@1 fr:
its 1.5% holes explain most of its ROI-NCC deficit (0.809 → 0.840 on common voxels, i.e. at parity
with the baseline's 0.843) — at 1 frame the burst model tears small coverage holes, which the
in-run `hole_frac_heart` (0.08 at 5 frames) never saw because training always fed 5 frames.

## 3c. Predicted displacement magnitude (ED DVFs, cmrx2024 val n=29, mm, medians)

| arm | whole-plane shift | heart local Δxy | tissue local Δxy | background Δxy | heart local Δz |
|---|---|---|---|---|---|
| noreg @1 fr | 2.44 | 1.18 | 0.28 | 0.46 | 1.36 |
| burst5 @5 fr | 2.57 | **1.75** | 0.41 | 10.6 | **1.96** |
| burst5 @1 fr | 2.52 | 1.80 | 0.39 | 10.6 | 1.78 |
| noreg @5 fr | 2.53 | 0.85 | 0.22 | 0.46 | 1.01 |

Whole-plane shift = breathing correction, identical across arms (slope vs applied AP shift
1.0–1.1 for all). Inside the heart the burst model predicts ~1.5× more local in-plane and ~1.4×
more through-plane motion than the baseline; the 1-frame model fed 5 frames predicts LESS (0.85),
i.e. it averages instead of moving. The burst model also emits ~10 mm garbage Δ on background
pixels (gated out of the splat by `intensity > 1e-3`; harmless, but it makes the raw
`panel_dvf.png` and whole-image means misleading — restrict to tissue when quoting Δ).

## 4. Provenance / files

- Checkpoints: `scratch/logs/212280371_burst5noreg224_pooled1337/ckpts/checkpoint_last.pt`
  (burst; job 59054212, resumed once from ep 83 after a GPFS stale-handle storm, docs in
  `_agent/handoff-2026-09-03-0220.md`), `scratch/logs/212682843_noreg224_pooled1337/ckpts/checkpoint_last.pt`.
- Recon jobs 59970577/578/580/581, EF jobs 59970583/584/586/587 (jjparkcv98/spgpu; owned1 was
  GPU-saturated, jjparkcv0 billing-blocked). A first submission (5996791x) died on
  `SKIP_DVF: unbound variable` under `set -u` in `sbatch/eval_pooled_val.sh` — fixed on the branch
  (c3806d8); **main's copy still has the bug.**
- Results: `evaluation/metric_results/{<src>,_ef}/vggt_{burst5noreg224,burst5noreg224_k1,noreg224,noreg224_k5}_ep300.json`,
  `evaluation/models.json` rebuilt. Per-subject recons + GIFs under
  `evaluation/volumes/<src>/out/<subj>/vggt_<arm>/`. Checkpoint copies under
  `evaluation/checkpoints/` NOT made (2×8.9 GB; decide before the originals go away).
- Branch: `arm/burst5` (worktree `vggt-burst`), not merged — the burst sampler (`burst_k`,
  default 1 = old behaviour) lives only there.

## 5. Next

- Decide whether `burst_k` merges to main (inert at the default) so the arms can be scored from
  main's harness in future.
- Test hypotheses 3(a)/(b): burst-trained model + 1-frame eval on the clean arm; a `burst_k=5`
  arm trained with `num_slices` matched to k=1 (drop planes) to separate "more slots" from
  "sequential frames".
- Within-burst Δt / frame-index embedding (the model currently gets the 5 frames as an unordered
  set) — the natural architecture follow-up now that the sampler-only change is a confirmed win.
