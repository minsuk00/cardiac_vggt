# 108 — final518 five-arm sweep: real test-set results, `motion10_hw2` leads

> **TL;DR & takeaway** (2026-09-16). Five 518-input/518-splat no-reg arms — `base`, `diff1000`
> (+diffusion=1000), `hw2` (+heart_weight=2.0), `motion10` (+motion_l1_weight=10), `motion10_hw2`
> (both) — trained to convergence (300 epochs, curated-v2 pooled cohort) and scored on the real
> held-out **test** split (180 subjects, CMRx23/24/25+ACDC+MNMs) with real nnU-Net EF/Dice and the
> full image-metric suite (widened `mask_heart_pad10` ROI, docs/107). **`hw2` alone has the best EF
> MAE (10.19 pp) but a catastrophic breathing-correction failure** — mean EPE 5.25 mm driven by a
> small subset of subjects where predicted z-displacement runs opposite the true direction
> (`resp_slope` down to −30; median EPE is a completely normal 0.96 mm, so most subjects are fine).
> Demeaning does **not** fix it (mean EPE gets worse, 8.49 mm) — the error isn't a constant bias.
> **`motion10_hw2` is the best overall pick**: EF MAE 10.65 pp (within 0.5 pt of `hw2`), same image
> quality (PSNR/SSIM/NCC), and breathing correction stays largely intact (EPE 1.29 mm, `resp_corr`
> 0.91 vs `hw2`'s 0.73). `diff1000` is the fallback if breathing fidelity matters more than the EF
> gain (EPE 0.93 mm, matching the clean baseline arms, but weaker EF at 10.92 pp). `base` and
> `motion10` alone are both weak on EF (~13.1–13.5 pp) and not competitive. Independently confirms
> and extends docs/106's `hw2` breathing-instability finding (that doc found it via training-time
> `val/resp/epe_dz_mm`; this doc confirms it on the held-out test split with the real eval harness).
> **Three more arms from the same sweep — `hw0`, `nogather`, `diff1000_nogather` — are still
> training as of this writing and are NOT in this doc**; they complete the heart-weight and
> gather-loss ablations and will need a follow-up once scored (docs/64's dead-ReLU collapse hit
> `nogather` twice at the identical step before a fresh retry survived past it — that mechanism is
> its own open finding, not written up anywhere yet).

## 1. What was run

Base recipe (`sbatch/train_final_518.sh`): `img_size=518`, `loss.volume.splat_res=518` (dense
training-time splat, docs/65's `splat_res` knob), no-reg (`diffusion_weight=0.0`), `gather_weight=0.5`,
`heart_weight=0.5`, cosine LR 5e-5→0 over 300 epochs (5% warmup), aggressive aug, respiratory sim on,
`pooled_curated_v2.txt` (628 train / 90 val / 180 test). Every arm below differs from `base` by
exactly the listed override — verified by Hydra-composing each arm and diffing against `base`:

| arm | override | exp dir (`scratch/logs/`) |
|---|---|---|
| `base` | — | `210823094_final518_base_curated898` |
| `diff1000` | `loss.volume.diffusion_weight=1000.0` | `210823094_final518_diff1000_curated898` |
| `hw2` | `loss.volume.heart_weight=2.0` | `210823094_final518_hw2_curated898` |
| `motion10` | `loss.volume.motion_l1_weight=10.0` | `210823094_final518_motion10_curated898` |
| `motion10_hw2` | `loss.volume.motion_l1_weight=10.0` + `heart_weight=2.0` | `210823094_final518_motion10_hw2_curated898` |

All five reached epoch 300 (`train/loss/objective` epoch max = 299) and `sacct` shows `COMPLETED`
for all five SLURM jobs. `motion10` and `motion10_hw2` crashed once mid-training from a GPFS
file-count quota exhaustion (`OSError: Disk quota exceeded`, unrelated to the recipe — the lab
shared fileset hit its inode cap) and were resumed from their intact `checkpoint_last.pt`
(verified: `torch.load` round-trip clean, `prev_epoch` matched the last successful save) — their
`scratch/logs/` dir names above are unchanged from the original launch.

Scored with `sbatch/eval_pooled_val.sh` (`SPLIT=test`, `SKIP_GIF=1`) + `sbatch/eval_ef_dice.sh`
(`SPLIT=test`) against each arm's own `checkpoint_last.pt`, on `jjparkcv98`/`spgpu` (A40) and
`jjparkcv_owned1`/`spgpu2` (L40S). Image-metric scoring uses the widened `mask_heart_pad10.nii.gz`
ROI as of docs/107 (re-scored after that mask default changed mid-session; the rescore reused the
existing reconstructed volumes — no re-reconstruction needed since only the scoring ROI changed).

## 2. Results (real, not the in-training proxy)

n=180 test subjects for both image metrics and EF (nnU-Net Task114), re-verified fresh from disk
with `assert len(df)==180` and `assert split=='test'` before writing this table:

| arm | EF MAE (pp) | PSNR (unit-peak) | SSIM | NCC | EPE mean (mm) | EPE median (mm) | resp slope | resp corr |
|---|---|---|---|---|---|---|---|---|
| base | 13.47 | 24.21 | 0.695 | 0.876 | 0.94 | 0.81 | 0.92 | 0.99 |
| diff1000 | 10.92 | 24.23 | 0.700 | 0.876 | 0.93 | 0.81 | 0.88 | 0.98 |
| **hw2** | **10.19** | 24.67 | 0.714 | 0.889 | **5.25** | 0.96 | **−0.21** | **0.73** |
| motion10 | 13.12 | 24.39 | 0.702 | 0.882 | 0.93 | 0.81 | 0.92 | 0.99 |
| **motion10_hw2** | 10.65 | 24.65 | 0.712 | 0.889 | 1.29 | 0.89 | 0.78 | 0.91 |

PSNR is `psnr_unit_peak` (the harness's HEADLINE convention per docs/106/console output, peak=1.0,
field-standard) — NOT the legacy `breath_psnr` ROI-peak key.

## 3. The `hw2` breathing failure, in detail

`hw2`'s EPE **median** (0.96 mm) is normal — indistinguishable from every other arm. The mean
(5.25 mm) is entirely driven by a small subset of subjects: the worst 10 of 180 have EPE 22–100 mm
and `resp_slope` between −4 and −30 (predicted z-displacement moving **opposite** the true applied
direction, not just noisy). ACDC and MNMs are overrepresented among the worst 10 (5/10, vs 11.7%/
27.2% of the 180-subject pool) — a real lead, not yet confirmed as the root cause. **Demeaning does
not fix it**: `resp_epe_dz_demeaned_mm` mean is 8.49 mm (worse than raw's 5.25 mm), because the
failure isn't a constant per-subject bias that demeaning would cancel — it's the model predicting
displacement in the wrong direction for specific subjects, so subtracting its own (also wrong) mean
amplifies rather than removes the error. This independently confirms docs/106's finding (which used
the in-training `val/resp/epe_dz_mm`, jumping 1.03→5.95→8.36 mm after epoch ~200) on the completely
separate held-out test-set eval harness — two independent measurements, same conclusion.

## 4. Where the raw results live

- Per-subject image metrics: `evaluation/metric_results/test/{cmrx2023,cmrx2024,cmrx2025,acdc,mnms}/vggt_final518_<arm>_ep300.json`
- Per-subject real EF/Dice: `evaluation/metric_results/test/_ef/vggt_final518_<arm>_ep300.json`
- Reconstructed volumes (reused across the mask rescore): `evaluation/volumes/{source}/out/{subject}/vggt_final518_<arm>_ep300/`
- Training logs / checkpoints: `scratch/logs/<exp_dir>/` (table in §1), `ckpts/checkpoint_last.pt`
- Sweep launch script: `sbatch/train_final_518.sh` (git `986c2e4` at the time of this sweep)

## 5. Not yet resolved

- `hw0`, `nogather`, `diff1000_nogather` (heart-weight and gather-loss ablation arms, same recipe
  family) are still training — not included here.
- `nogather`'s dead-ReLU collapse (docs/64 signature: `grad_aggregator` dead, exact same step
  13210/epoch 21 on two independent attempts, same seed) is a real, reproducible finding this
  session surfaced but has not been written up in its own doc yet.
- `hw2`'s breathing-failure root cause (why those specific subjects) is not confirmed, only the
  ACDC/MNMs-overrepresentation lead above.
