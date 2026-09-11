# 91 — Two-phase difference loss: design, port, verification, launch protocol

> **TL;DR & takeaway** — The CorSeg-Dice sweep (docs/90 arms, evaluated 2026-08-24) **refuted region/boundary losses as the EF-amplitude fix**: dice is neutral at w=0.1 and monotonically worse with weight (SV ratio 0.711→0.580, r 0.792→0.744 at ep300), while no-reg (diffusion=0) delivered the only real win — per-patient ranking r 0.792→**0.837** (bootstrap 95% CI [+0.005, +0.082]) with bias/SV-ratio unchanged (Wilcoxon p=0.67). Diagnosis: every loss so far is **single-phase** — under phase uncertainty its optimum hedges the ED−ES swing toward the temporal mean, and no gradient path ever penalizes "the difference between two phases is too small," which is exactly the measured failure (DC perfect at mass ratio 0.996, swing ×0.71). This arm adds the first **between-phase** term: per train step, render the same subject at a second target phase t2 = (t1 + T//2) % T — second reference slice in slot 0, companion slots shared bit-identically, same breath — and add heart-ROI L1 on the swing error (V_canon(t1)−V_canon(t2)) − (V_gt(t1)−V_gt(t2)), stacked on diffusion=0. **Win rule: SV-ratio ↑ AND r held (~0.837)** — SV↑ with r↓ is the cohort-mean cheat the α-sweep showed is free, reject it. Expect partial recovery at best: amplitude information decays 0.87→~0.3 within ±5 planes of the reference, and ~27% of true SV is longitudinal motion the forward splat cannot express — a null result would bound the loss-induced share of the deficit at ≈0 and point at the renderer. **Update 2026-08-25**: `diff_weight≥1.0` collapses the point head from a fresh start (displacement runaway → dead ReLU, docs/64 signature) within a few epochs — only `w=0.5` survived; see §5. **Final verdict (2026-08-28, §6): w=0.5 at ep300 improves EF bias (−12.6→−10.8), MAE (12.89→11.80, p=0.0014) and SV-ratio (0.712→0.743, p=0.014) but drops per-patient r (0.837→0.779) — FAILS the pre-registered "SV↑ AND r held" rule.** The r drop is not a broad cohort-mean shrink (jackknife: ~5 large-overshoot subjects carry it), but the user's decision is to **ship no-reg as the paper model**; the TD loss is closed as a partial-win-not-adopted. Root-cause work on the residual EF bias (shared by ALL arms) moved to docs/92.

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

## 5. Launch outcome: w≥1.0 is unstable from a fresh start; w=0.5 is the only surviving arm

Both `w20` (`diff_weight=2.0`, job 58567123) and a subsequent `w10` (`diff_weight=1.0`, job 58600101, `sbatch/train_twophase_w10.sh` — not committed, deleted after cancellation) **collapsed and were cancelled**; only `w05` (job 58567122) survived and is the sole twophase arm going forward. Root cause is the same in both: early in training, before the point head has learned anything, the diff term's gradient (computed from two still-garbage renders) pushes `world_points` outside the volume — `mean_disp_norm` runs away (w20: 0.07→~10 mid-epoch-17; w10: 0.83→~8 by epoch 2, i.e. faster at the higher weight) — `coverage_frac`/`coverage_mean` collapse to 0, the coverage-division splat then passes zero gradient back, and the point head dies (docs/64 dead-ReLU signature: `Grad/aggregator` and `Grad/point` both pin at exactly 0.0, confirmed via `probe_aggft_collapse.py` returning `### COLLAPSED` on w20's `checkpoint_last.pt`). **This is a weight-magnitude threshold, not an unlucky seed or a late-training instability** — w10 failed even faster than w20, and w05 has now run 175+ epochs with `mean_disp_norm` stable (~0.06–0.07) and zero alarms. Practical ceiling for this recipe from a fresh VGGT-1B start: **diff_weight ≤ 0.5**. A warmup ramp (0→target over the first ~10 epochs) was proposed as the fix for testing higher weights later but not built — deprioritized once w05 alone showed a positive signal (below).

**In-progress result (w05, epoch 173/300, NOT the final ep300 nnU-Net verdict — that still requires the §4 harness once training completes):** matched-epoch comparison of w05 against no-reg's own training curve at its epoch 173 (`tools/load_run.py` on both log dirs — `scratch/logs/212413171_twophase05_pooled1337` vs `scratch/logs/212682843_noreg224_pooled1337`):

| metric (val, epoch 173) | twophase w05 | no-reg | direction |
|---|---|---|---|
| motion PSNR | 20.83 dB | 20.64 dB | ↑ |
| bbox PSNR | 24.87 dB | 24.75 dB | ↑ |
| recov_frac_heart | 0.486 | 0.456 | ↑ (more contraction recovered) |
| hole_frac_heart | 0.241 | 0.253 | ↓ (fewer coverage holes) |

No-reg itself plateaus by ~epoch 125–150 and is essentially flat out to its own ep299 (recov_frac_heart 0.475, motion PSNR 20.74, bbox 24.85) — w05 is **already at no-reg's final plateau at epoch 173**, ~125 epochs earlier, and still climbing slightly. This is directionally consistent with the mechanism argument in §1, but these are voxelwise training-time proxies, not the clinical EF/SV metric the arm targets — the decision-rule verdict is in §6.

## 6. Final result (ep300, 2026-08-28): partial win, FAILS the pre-registered rule; no-reg ships

w05 trained to ep300 clean (job 58567122, COMPLETED, ~50.4h, zero collapse alarms). Scored via the §4 redirect harness (job 58756454) → `temp/dice_ef_sweep/mirror_full/results/_ef/vggt_twophase05_ep300.json` (144/144 subjects; baseline drops one — `CMRx24_Test_P044`, GT EF 91.2%, where baseline's recon yields an empty nnU-Net LV at ES).

| metric (n=144 paired) | no-reg ep300 | twophase05 ep300 | test |
|---|---|---|---|
| SV-ratio | 0.712 | **0.743** | Wilcoxon p=0.014 |
| EF bias | −12.6 | **−10.8** | — |
| EF MAE | 12.89 | **11.80** | Wilcoxon p=0.0014 |
| per-patient r | **0.837** | 0.779 | Δ=−0.057, bootstrap 95% CI [−0.138, +0.003] |
| slope | 0.831 | 0.802 | — |

**Verdict: FAILS the §4 clean-win rule** — SV↑ real, but r dropped (borderline-significant). Jackknife on the Δr shows it is **not a broad cohort-mean shrink**: ~5 subjects where twophase *overshoots* EF by +13 to +25 points (no-reg undershoots the same subjects) carry the gap; excluding the 5 most influential, Δr=−0.013 (0.830 vs 0.817, tied). For the bulk of the cohort twophase matches no-reg's ranking while beating its bias/MAE — the failure mode is a handful of large overshoots on hard subjects, a different pathology from the α-sweep's free-bias cheat, but still an r regression.

**Decision (user, 2026-08-28): no-reg ships as the paper model.** Rationale: best per-patient ranking (the honest headline), passes the rule, and the residual EF bias it shares with every arm was root-caused the same day to mechanisms the TD loss does not address — the bias is only ~⅓ genuine amplitude (bounded ~2–4 EF points, consistent with TD's 1.8-point gain) and the rest is ED under-expansion + a mis-rendered ES wall band. Full root-cause chain, refuted hypotheses, and per-lever ceilings: **docs/92**. The TD arm is closed; if reopened for w>0.5, build the §5 warmup ramp first.

Figures from the eval + root-cause session (gitignored, on GPFS): `temp/dice_ef_sweep/ef_scatter_*.png` (GT-vs-pred EF, 3 arms; final version colored by ESV/EDV relative error), `temp/ef_fix/` (docs/92 walkthrough).
