# 86 — Seven-arm config sweep: full eval (image + motion + EF/Dice) and knob conclusions

> **TL;DR & takeaway**
> All 7 pooled-1337 ep300 checkpoints (base 224 / hw2 / hw2+miitt / p2p98 / 336 / 518 / dinov3-256)
> were pushed through the full docs/85 pipeline — recon + PSNR/SSIM/NCC + breathing EPE + the
> **first-ever real-data run of the EF/Dice chain** — on the frozen breath bundles, val split, all
> 7 sources. **The EF chain is validated: pooled recon-EF-vs-GT-EF slopes are 0.74–0.83 for every
> arm, reproducing the docs/33 anchor (~0.77–0.79).** Knob verdicts (each isolated, paired
> per-subject Wilcoxon over 144 shared subjects): **518 wins image fidelity** (+0.26 dB vs 224,
> 83% of subjects, p=2e-12, no EPE cost) **and EF MAE** (~30–40% better, e.g. 9.2 vs 13.4% on
> cmrx2024) but has the weakest EF *slope* (0.74 — regression-to-mean flavor); **p2p98 is a free
> +0.16 dB** with the best breathing slope; **MIITT-in-training** adds +0.11 dB paired and the best
> pooled EF slope (0.83); **hw2 trades a small PSNR gain for significantly worse EPE**
> (0.79→0.93 mm, p=7e-6); **dinov3-256 loses or ties everywhere** (PSNR n.s. vs 224, worst EPE and
> worst EF MAE) — drop it. EF slope and Dice are essentially config-invariant. **Recommended
> combo run: img 518 + MIITT-pooled + hw 0.5 + DINOv2, keeping percentiles [0.5,99.9]** (p2p98's
> +0.16 dB is real but not worth a second gauge convention — user decision 2026-08-20, §3).
> Report with all
> tables/plots: `_html/47_seven_arm_sweep_full_metrics.html`.

## 1. What was run (2026-08-20)

The 7 checkpoints, mapped from wandb ids via each run's `run_meta.jsonl` (all pooled-1337 splits,
300 epochs, `checkpoint_last.pt`; configs differ in exactly the listed knobs, verified by flattening
and diffing the launch records):

| arm slug (`vggt_<slug>`) | wandb | log dir | differs by |
|---|---|---|---|
| `augaggr224_ep300` | awrobewn | 213340611_…augaggr224… | base: img 224, heart_weight 0.5, percentiles [0.5,99.9] |
| `augaggr224hw2_ep300` | cfvoed6b | 213338187_augaggr224hw2… | heart_weight 2.0 (already scored pre-session) |
| `augaggr224hw2_miitt_ep300` | 1gor8rcs | 213110090_…miitt… | hw 2.0 + `pooled_miitt.txt` (MIITT in training) |
| `augaggr224p2p98_ep300` | dy3uq82a | 213331641_…p2p98_nogc… | intensity_percentiles [2,98] |
| `augaggr336_ep300` | pxpv7in5 | 213321784_augaggr336… | img 336 |
| `augaggr518_ep300` | jyctv2mm | 213415771_…augaggr… | img 518 |
| `dinov3_256_cont_ep300` | 209usq6p | 213106973_dinov3_256_cont… | DINOv3-vitl16 backbone, 256/patch16, strict warm-start from `dinov3_256_ep299_weights_only.pt` |

Pipeline: `sbatch/eval_pooled_val.sh` per arm (spgpu2/jjparkcv_owned1) = frozen-bundle build (no-op,
all idempotent-skipped) → `run_vggt.py` (protocol from the ckpt's own run_meta — this is what makes
the dinov3/518/336 variants "just work") → `score/image_metrics.py` → `analysis/viz.py` GIFs →
`score/aggregate.py`. Then `sbatch/eval_ef_dice.sh` (NEW this session) per arm = `ef_dice.py dump`
→ `run_seg.sh` (nnU-Net Task114, nnunet env) → `ef_dice.py score` → re-`aggregate` all 7 sources,
folding the `ef` block into every citable JSON. Breath arm only (`ARMS=breath`).

**Wall-clock for reference:** image-metric jobs 35.5 min (224-family/256) / 39.5 (336) / 53 (518)
end-to-end for all 7 sources — GIF rendering dominates (~12 s/subject; recon itself is 2.5→9 s per
12-phase cine, 93→~600 ms/phase for 224→518; `SKIP_GIF=1` for metrics-only). EF/Dice jobs 74–91 min
(dump ~7 min for ~3.5k volumes, nnU-Net ~95 vols/min, score+aggregate minutes); the hw2 arm took
108 min because it is the only arm with clean-arm cines on disk (earlier campaign scored
`clean breath`), so its dump carried ~1.6k extra volumes — bonus: hw2 is the only arm with
clean-arm EF/Dice numbers.

## 2. Results (val, breath arm, heart∩FOV ROI; full tables in `_html/47`)

Cohort-mean over the 7 sources:

| arm | PSNR dB | SSIM | EPE mm (demeaned) | resp slope | pooled EF slope | EF r | EF MAE % | Dice LV ES |
|---|---|---|---|---|---|---|---|---|
| 224 | 19.41 | 0.818 | **0.79** | 0.92 | 0.785 | 0.79 | 12.1 | 0.804 |
| 224hw2 | 19.48 | 0.821 | 0.93 | 0.91 | 0.791 | 0.80 | 11.4 | 0.815 |
| 224hw2+miitt | 19.69* | 0.830* | 0.89 | 0.91 | **0.831** | **0.82** | 10.5 | 0.823 |
| 224p2p98 | 19.54 | 0.824 | 0.82 | **0.94** | 0.774 | 0.80 | 11.7 | 0.810 |
| 336 | 19.52 | 0.822 | 0.99 | 0.90 | 0.795 | 0.78 | 11.7 | 0.798 |
| 518 | **19.64** | 0.827 | 0.83 | 0.92 | 0.744 | 0.73 | **9.6** | **0.820** |
| dinov3-256 | 19.44 | 0.819 | 1.00 | 0.93 | 0.778 | 0.81 | 14.2 | 0.797 |

\* confounded by extra training data (MIITT in the pool), not a pure-knob number.

Paired per-subject Wilcoxon vs base 224 (144 shared subjects), breath PSNR:
hw2 **+0.12** (67%, p=8e-5) · p2p98 **+0.16** (70%, 2e-6) · 336 **+0.11** (69%, 3e-6) ·
518 **+0.26** (83%, 2e-12) · dinov3 +0.05 (47%, **n.s.**) · hw2→hw2+miitt +0.11 (61%, 2e-3) ·
336→518 +0.15 (74%, 1e-7). EPE deltas: hw2 +0.12 mm worse (7e-6), 336 +0.19 worse (3e-3),
dinov3 +0.19 worse (3e-6); p2p98/518 n.s.

## 3. Conclusions

- **EF chain first real run = VALID.** Pooled slopes 0.74–0.83 bracket the docs/33 anchor; no
  EF=100% pathologies; per-cohort slopes are noisy at n≤37 (healthy cohorts have narrow EF range —
  read the pooled number, not cmrx2023's 0.2 or miitt's n=3 negatives).
- **The metric axes dissociate** (as docs/24/33 predicted): PSNR ranking ≠ EPE ranking ≠ EF-MAE
  ranking. Image/Dice/EF-slope barely move with knobs; **EPE (~25% relative) and EF MAE (~30–40%
  relative) move a lot — in opposite directions** (224/p2p98 best EPE; 518 best EF MAE).
- **518's EF profile:** best MAE everywhere but weakest slope (0.74) — part of its low average
  error is predicting nearer the cohort mean, i.e. slightly under-tracking per-patient contraction
  differences. Report both numbers; don't sell MAE alone.
- **dinov3-256 is droppable** on current evidence: no PSNR gain, significantly worse EPE, worst EF
  MAE. (Continuation run — a longer/differently-tuned schedule could still change this.)
- **p2p98 gauge fairness (question raised, resolved in code):** the frozen bundles are built by
  `MRIDataset` at the default [0.5,99.9] gauge and `run_vggt` swaps `batch["phases"]` with bundle
  pixels, so the p2p98 arm's recon AND the GT are both in bundle gauge — the comparison is fair.
  The residual caveat is a train/eval input-normalization mismatch: the p2p98 model trained on
  [2,98]-normalized inputs but was fed [0.5,99.9]-normalized bundle pixels — so we measured the
  arm UNDER a test-time input shift, not the config as it would deploy (inputs in its own gauge).
  SSIM/NCC (gauge-insensitive) also rank it above 224, so the win is not a gauge artifact.
  **Native-gauge probe RUN 2026-08-20 (`temp/native_gauge_eval.py`, results in
  `temp/p2p98_native_gauge/`):** inputs re-normalized per subject to the [2,98] percentiles of
  bundle `gt_t00` (nonzero voxels, mirroring `ScaleIntensityByT0PercentilesD`), recon inverse-mapped
  to bundle gauge, scored with the standing `image_metrics` functions against untouched GT; a
  `--null` run (no renorm) reproduced stored scores exactly (max |dPSNR| = 0). Result: **mean
  dPSNR = −0.037 dB** (n=144; per-source −0.11…+0.03) — noise-level, slightly negative. The
  mismatch was NOT suppressing p2p98; the stored +0.16 dB is the config's real margin, not a
  lower bound.
- **Dice DOES show the correction — once the right identity floor is used.** Two geometric floors
  (pure numpy on the GT `heart_seg.nii.gz` + frozen manifest `disp_dhw_mm` + the RECORDED per-slice
  cardiac-phase draw from `ed_dvf.npz` `slot_t`; no nnU-Net re-run; 144 val subjects):
  the two references that matter (user-selected; resp-only and the hw2-clean "ceiling" are less
  meaningful — the former assumes slices magically at the target phase, the latter is a one-arm
  recon result): **cardiac-only floor** (phase mix, no breathing — the breath-hold do-nothing)
  **0.827 ED / 0.746 ES**, and the **full floor** (cardiac + breathing — the real do-nothing
  input) **0.732 ED / 0.664 ES** (`metric_results/_floors/dice_identity.json`, generator
  `evaluation/src/score/dice_floors.py`). Arms score 0.87–0.89 / 0.80–0.84 ⇒ **the model
  recovers +0.14 ED / +0.16 ES Dice over its uncorrected input, cardiac-phase correction the
  larger component** (largest at ES, where the LV is smallest and most slices come from other
  phases). An intermediate "Dice is blunt" conclusion used the resp-only floor — which grants
  cardiac correction for free — and is RETRACTED. (Stricter nnU-Net-on-corrupted-volume floor
  would be lower still; not run, not needed.) **prove-it review of `dice_floors.py` (1 reviewer
  + runtime verification) found and fixed 2 real bugs before these numbers were finalized:**
  (1) the in-plane breathing roll had a MIRRORED sign vs the builder's sampling convention —
  settled by probing real breathed bundle pixels, 15/15 subjects favor the negative shift;
  (2) the z_mid reference slice was scored at phase 0 although the real input re-extracts it at
  the target phase — fixing this raised the ES floors ~0.05 (earlier values 0.735/0.636 for the
  full floor superseded). Plus sorted-glob + fractional-slot_z guard hardening (all 144 subjects
  probed: every arm's draw identical and integral).
- **Recommended combined config** (each knob tested in isolation, no observed conflicts):
  **img 518 + MIITT-pooled + heart_weight 0.5 + DINOv2**, keeping `intensity_percentiles`
  at the default **[0.5, 99.9]** — to be confirmed by one training run, not assumed.
  **[2,98] deliberately EXCLUDED (user decision 2026-08-20):** the native-gauge probe (above)
  confirmed its +0.16 dB is real, but it is the smallest lever in the sweep, changes no
  EF/Dice conclusion, and adopting it would fork the intensity-gauge convention across
  training cache, frozen eval bundles, and all historical checkpoints. Simplicity — one
  gauge everywhere — is worth more than ~0.16 dB.

## 4. Fixes & gotchas hit along the way

- **`sbatch` spools the script**, so `eval_pooled_val.sh`'s `REPO=$(dirname BASH_SOURCE)/..`
  resolved to `/var/spool/slurmd.spool` and the first 6 submissions died at t=0. Fixed:
  `REPO=${REPO:-${SLURM_SUBMIT_DIR:-…}}`. Any future self-locating sbatch script needs the same.
- **Off-split bundles tripped the strict exit code:** miitt's `out/` holds train/test bundles;
  `run_vggt` skips them (no recon) but the driver's scoring glob tried to score them →
  4 spurious "failures"/exit 1 per job (aggregates were always correct — split is enforced at
  aggregate time). Fixed: the loop now skips subjects with no arm dir.
- **GIF speed:** `viz.py` default fps 3→6 (12-phase cycle 4 s→2 s); all 865 existing subject-arm
  GIFs re-rendered in place via `tools/render_all_gifs.sh` (CPU job, 8 workers, 0 failures).
- **"GT segmentation" means Task114-on-GT** (docs/39), re-derived from `cine_gt.nii.gz` at score
  time so both sides of EF/Dice share grid+gauge+segmenter. The bundle's `heart_seg.nii.gz`
  sibling is the same segmenter on the same images; ACDC's manual masks remain the only truly
  independent reference (unused so far — candidate paper check).
- New standing script: **`sbatch/eval_ef_dice.sh`** (MODEL_NAME-parameterized, spgpu2, node-local
  /tmp work dirs — sidx-keyed dump REQUIRES a fresh dir per run, which the per-job-id path gives).

## 5. Still open (unchanged from docs/85)

`pose_psf.py` (3-DOF + PSF, docs/83) → score the 144 SVRTK/NeSVoR baseline volumes; OCMR intensity
gauge (~13 dB, docs/84); `resp_corr` into the cohort `all` block; stats at paper time.
