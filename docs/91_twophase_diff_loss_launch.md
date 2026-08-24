# 91 — Two-phase difference loss: design, port, verification, launch protocol

> **TL;DR & takeaway** — The CorSeg-Dice sweep (docs/90 arms, evaluated 2026-08-24) **refuted region/boundary losses as the EF-amplitude fix**: dice is neutral at w=0.1 and monotonically worse with weight (SV ratio 0.711→0.580, r 0.792→0.744 at ep300), while no-reg (diffusion=0) delivered the only real win — per-patient ranking r 0.792→**0.837** (bootstrap 95% CI [+0.005, +0.082]) with bias/SV-ratio unchanged (Wilcoxon p=0.67). Diagnosis: every loss so far is **single-phase** — under phase uncertainty its optimum hedges the ED−ES swing toward the temporal mean, and no gradient path ever penalizes "the difference between two phases is too small," which is exactly the measured failure (DC perfect at mass ratio 0.996, swing ×0.71). This arm adds the first **between-phase** term: per train step, render the same subject at a second target phase t2 = (t1 + T//2) % T — second reference slice in slot 0, companion slots shared bit-identically, same breath — and add heart-ROI L1 on the swing error (V_canon(t1)−V_canon(t2)) − (V_gt(t1)−V_gt(t2)), stacked on diffusion=0. **Win rule: SV-ratio ↑ AND r held (~0.837)** — SV↑ with r↓ is the cohort-mean cheat the α-sweep showed is free, reject it. Expect partial recovery at best: amplitude information decays 0.87→~0.3 within ±5 planes of the reference, and ~27% of true SV is longitudinal motion the forward splat cannot express — a null result would bound the loss-induced share of the deficit at ≈0 and point at the renderer.

Worktree: `/home/minsukc/vggt-twophase`, branch `arm/twophase-diff` (off `arm/corseg-dice`). Commit: `73796a3`.

## 1. Why this loss (and why Dice failed)

Evidence chain (details: docs/90 + the 2026-08-24 eval of its arms, `temp/dice_ef_sweep/mirror_full/results/_ef/`):

| arm (ep300, n≈144) | EF bias | SV ratio | r | EF MAE |
|---|---|---|---|---|
| baseline (awrobewn) | −12.6 | 0.711 | 0.792 | 13.02 |
| no-reg (diffusion=0) | −12.6 | 0.712 | **0.837** | 12.89 |
| dice 0.1 | −12.5 | 0.716 | 0.784 | 13.28 |
| dice 0.5 | −16.1 | 0.650 | 0.767 | 16.42 |
| dice 2.0 | −19.5 | 0.580 | 0.744 | 19.59 |

Same monotone pattern at ep200 — not a training-stage artifact. Dice sharpened the *per-phase* boundary gradient and it didn't help, because the deficit is not per-phase: EDV/ESV errors are antisymmetric (−10.9/+10.85 ml, midpoint −0.03 ml), i.e. the model knows the mean and hedges the swing. A per-phase loss of any kind — L1, heart-L1, Dice — has a hedged optimum under phase uncertainty; only a term computed **across two phases** puts gradient directly on the swing.

Caveats accepted in advance (2-agent adversarial debate, 2026-08-24): (a) L1-on-the-difference still shrinks toward the conditional median where phase is genuinely unrecoverable — this loss recovers only the *identifiable* share of amplitude; (b) at info-starved planes the cheapest gradient descent direction is the cohort-typical swing, which would improve bias while eroding r — hence the win rule; (c) ~27% of true SV is through-plane motion structurally outside the forward splat's reach. The arm doubles as the measurement that separates loss-induced from info/renderer-limited deficit: SV→~0.85 with r held = mechanism confirmed; SV unmoved = loss-blindness bounded at ≈0, next lever is the gather renderer.

## 2. Implementation (commit 73796a3, ~90 lines + tests)

Key simplification: everything for render 2 derives from **post-aug GPU state**, so there are zero dataset / `composed_dataset.py` (collate-whitelist) changes — the trap that bit the corseg port (a8ff511) is structurally avoided.

- `training/data/gpu_aug.py` — `build_twophase_inputs(batch, device)`: t2 = (t1+T//2)%T; re-extracts ONLY slot 0 at t2 from the already-warped `phases` (one affine, drawn once), applying the SAME `resp_disp_mm[:, :1]` so both renders see one breath; emits `images_2`/`images_splat_2` (slots 1..S−1 are the *same tensors* as render 1 — identical by construction), `gt_target_volume_2 = phases[b, t2]`, `t_target_2`.
- `training/loss.py` — `MultitaskLoss.forward(..., predictions_2=None)`: splats render 2 via the standard `_splat_preds_native` with `images_splat` swapped for `images_splat_2`; `loss_diff = mean_ROI |dC − dG| · diff_weight` with `roi = heart_roi_canonical` (phase-independent, co-warped; same raise-if-missing policy as heart_weight), dG in float32. No plain L1 on t2 — the arm is baseline + exactly one term.
- `training/trainer.py` — train_epoch builds the second inputs (gated `diff_weight>0`); `_step` runs the second forward inside the same autocast, gated `phase=="train" and "images_2" in batch`; one backward, clip/step untouched; the NaN-skip guard covers `loss_diff` automatically.
- `training/config/default.yaml` — `loss.volume.diff_weight: 0.0` (inert default) + `loss_diff` in the train scalar list only.
- `sbatch/train_twophase_w05.sh` / `_w20.sh` — awrobewn recipe verbatim (img224, aggressive aug, 300 ep, LR 5e-5), spgpu2/L40S/`jjparkcv_owned1`, overrides `diffusion_weight=0 diff_weight={0.5,2.0}`. Weights bracket the useful range: heart-L1 history says <0.5 may not bite; dice showed high weight can distort.

## 3. Verification (all on the interactive A40, 2026-08-24)

1. `tests/test_twophase.py` (8 tests, all green; full suite 371 green): slots 1..S−1 bit-identical across renders; slot 0 matches a direct t2 extraction, incl. the shared-breath case; `gt_target_volume_2` exact; analytic `loss_diff` value; gradient flows through BOTH renders' `world_points`; `predictions_2=None` bit-identical objective (the val / diff_weight=0 guarantee at the loss level); missing `images_splat_2` raises (the model-res splat fallback must never fire for render 2 alone).
2. **1-epoch smoke** (`diff_weight=0.1`, img224, no checkpointing): `loss_diff` finite and nonzero (0.0176 / 0.0042 / 0.0069 at the logged steps → raw term ~0.04–0.18, same order as heart-L1), per-phase losses at baseline magnitudes, no diff key in val. **Peak GPU memory 28.4 GiB / 46** — two full graphs fit on A40=L40S without gradient checkpointing; the recipe's `gradient_checkpointing=false` stands.
3. **Inertness control**: identical 1-epoch runs, twophase tree at `diff_weight=0` vs the unmodified `vggt-dice` tree — **step 0 bit-identical on every scalar** (same data, init, forward, loss). Later steps differ, but a same-tree rerun mismatches at the same rate (148/265 vs 146–147/265 cross-tree): that is the pipeline's inherent GPU-atomic nondeterminism after the first backward, not the arm. Combined with test 1's exact loss-level identity, the weight-0 path is as verified-inert as this pipeline permits.

## 4. Launch + eval protocol

- Submit `sbatch/train_twophase_w05.sh` and `_w20.sh` (self-submitting, requeue-safe) **only on explicit user go**.
- Judge at ep300 (and a matched intermediate ckpt) vs **no-reg ep300** via the frozen `temp/dice_ef_sweep/` redirect harness (writes only to `temp/`, never `evaluation/`): `CKPT=… MODEL_NAME=twophase{05,20}_ep300 sbatch temp/dice_ef_sweep/sweep_ef_dice_redirected.sh`, add the arms to `compare_ef.py` ARMS. nnU-Net Task114 scoring (decoupled from anything in the training signal).
- Decision rule: **win** = SV ratio ↑ from 0.712 with r held ≈0.837 (paired test + bootstrap CI, as in the dice comparison); **cohort-mean cheat** = SV↑, r↓ → reject, loss path closed; **null** = SV unmoved → deficit is info/renderer-limited, escalate to the gather/backward-warp renderer or reframe the headline to ranking/timing/OOD.
