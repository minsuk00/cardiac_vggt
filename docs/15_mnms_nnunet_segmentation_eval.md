# 15 — M&Ms nnU-Net cardiac segmentation as a recon-quality metric

> **TL;DR & takeaway** (2026-06-20). We can use the **pretrained M&Ms nnU-Net** (Zenodo
> `Task114_heart_mnms`, Full et al. STACOM 2020) to segment LV / MYO / RV on our SAX-stack
> output — it is a clean, anatomically meaningful downstream eval to complement PSNR. It is a
> **nnU-Net v1** model (NOT v2 — koalai's `nnunetv2` can't load it). Installed cleanly into a
> **fresh isolated `nnunet` micromamba env** (nnunet 1.7.1 + torch 2.3.1+cu121); **`svr` was not
> touched and remains bit-identical**. Model + env live on GPFS. It runs end-to-end (2d, 3d_fullres,
> ensemble) on both GT and VGGT-predicted volumes and produces **clean, plausible LV/MYO/RV**.
> First read on 6 ED (t00) subjects: **Dice( seg(pred_vol), seg(gt_vol) ) ≈ LV 0.94 / MYO 0.81 /
> RV 0.93** (ensemble) — i.e. our reconstruction preserves segmentable cardiac anatomy at ED.
> Status: **validated and ready to use**; not yet run across phases / on the val set at scale.
>
> **Validated against human GT (2026-06-20):** ran the model on the **ACDC test set** (50 patients,
> 100 ED+ES frames, real human segmentations) — a *cross-dataset* test (M&Ms-trained, never saw
> ACDC). Ensemble **mean Dice 0.902** (LV 0.928 / MYO 0.875 / RV 0.903), on par with in-domain
> ACDC-trained methods (~0.93/0.89/0.90). **The model is genuinely good, not just self-consistent.**

## What it is / why
The M&Ms 2020 challenge winner (Full, Isensee, Jäger, Maier-Hein) released a pretrained nnU-Net
ensemble (five 2d + five 3d_fullres folds, trainer `nnUNetTrainerV2_MMS`) that segments short-axis
cine MRI into **1=LV blood pool, 2=LV myocardium, 3=RV blood pool**. Our pipeline outputs exactly
that domain (a 3-D SAX stack at one cardiac phase), so the model gives an **anatomical** quality
signal on top of intensity PSNR:
- run it on `seg(GT volume)` and `seg(pred volume)` and compute **Dice between the two** → does the
  recon yield the same heart the GT does (geometry/anatomy fidelity)?
- downstream it also enables clinical metrics (LV/RV volume, EF, myocardial mass) per phase.

## Key compatibility finding: it is nnU-Net **v1**
The Zenodo install/predict commands (`nnUNet_download_pretrained_model`, `nnUNet_predict -t 114 -m
2d -tr nnUNetTrainerV2_MMS`) and the on-disk format (`fold_*/model_final_checkpoint.model` +
`.pkl`, `plans.pkl`) are **nnU-Net v1**. The `koalai` env has **nnunetv2 2.5** — a different
framework with an incompatible checkpoint format; it **cannot** load this model, and it is on slow
GPFS anyway. So we need the original `nnunet` (v1) package, in its own env.

## Isolation (svr protected)
- Fresh env: `micromamba create -n nnunet python=3.10`, then `pip install "numpy<2"
  torch==2.3.1 --index-url .../cu121` and `pip install nnunet` → **nnunet 1.7.1**.
- **Never ran pip in `svr`.** Verified before/after: `svr` = torch 2.3.1+cu121 / monai 1.4.0 /
  numpy 1.26.4, unchanged.
- torch 2.3.1 loads the 2020-era v1 checkpoints with no pickle/`weights_only` issues; CUDA visible.

## Where things live (all GPFS: `/gpfs/.../minsukc/vggt` = `scratch/`)
```
scratch/data/nnunet_mnms/
  TASK114_heart_mnms.zip          # 2.2 GB, md5 e6613f33… (matches Zenodo) — source, deletable
  results/nnUNet/{2d,3d_fullres}/Task114_heart_MNMs/...   # extracted model, RESULTS_FOLDER layout
  test_inputs/                    # converted SAX inputs (*_0000.nii.gz)
  seg_{3d,2d,3d_npz,ensemble}/    # segmentations
```
Env `nnunet` is ~6 GB under `~/micromamba/envs` (home, same place as all other envs — `fiss-recon`
etc.; the *model/data* is on GPFS as requested).

## How to run
```bash
# inputs: VGGT val_volumes (Z,Y,X identity-affine) -> (X,Y,Z) @ (1.4,1.4,8.0) mm, named *_0000.nii.gz
micromamba run -n nnunet python tools/nnunet_mnms_eval/prep_inputs.py \
  --val_dir <log>/val_volumes --out_dir scratch/data/nnunet_mnms/test_inputs --n 6
# predict (env.sh sets RESULTS_FOLDER + the two raw/preprocessed vars nnU-Net v1 requires at import)
micromamba run -n nnunet bash -c 'source tools/nnunet_mnms_eval/env.sh && \
  nnUNet_predict -i .../test_inputs -o .../seg_3d -t 114 -m 3d_fullres -tr nnUNetTrainerV2_MMS'
#   -m 2d for the 2d ensemble; add --save_npz to both and nnUNet_ensemble -f seg_2d seg_3d_npz -o seg_ensemble
micromamba run -n nnunet python tools/nnunet_mnms_eval/analyze_segs.py \
  --seg_dir .../seg_ensemble --input_dir .../test_inputs --out_png_dir result/nnunet_mnms_overlays
```
**Geometry gotcha:** our `val_volumes` NIfTIs are splat-order `(Z,Y,X)` with an identity affine.
nnU-Net resamples by header spacing, so `prep_inputs.py` transposes to nibabel `(X,Y,Z)` and writes
the true canonical spacing `(1.4, 1.4, 8.0)` mm. nnU-Net z-scores intensities internally, so our
percentile-normalized `[-1,1]` inputs pass through fine.

## Report
Self-contained HTML write-up (provenance, method, ACDC + CMRxRecon results, figures, 2d/3d):
**`_html/15_mnms_segmentation_eval.html`** (built by `tools/nnunet_mnms_eval/build_report.py`).

## Applied to our reconstruction — live joint-refiner run (across phases)
Segmented the current best model's output (`218349151_mri_refiner_joint`, ep~88; pred = `V_canon`)
for all 30 val subjects, stratified over t0…t11. Dice(seg(pred),seg(gt)), ensemble:
**ED (t00) LV 0.93 / MYO 0.81 / RV 0.88** (reproduces the ED-only run below) → **all-12-phases
LV 0.848 / MYO 0.739 / RV 0.828**. The phase drop is the point — it penalizes systole (hard motion)
and the V_canon splat blur, so the metric is phase-resolved and recon-sensitive. One genuine
recon-failure outlier at t06. Figure: `result/cmrx_joint_seg_panel.png`.

## First results (6 subjects, ED / t00, run `220360425_…dynamic_axial`)
All three modes ran cleanly and gave clean LV/MYO/RV on both GT and predicted volumes.
**Dice( seg(pred_vol), seg(gt_vol) ) — mean over 6 subjects:**

| mode      | LV    | MYO   | RV    |
|-----------|-------|-------|-------|
| 3d_fullres| 0.939 | 0.812 | 0.926 |
| 2d        | 0.934 | 0.804 | 0.905 |
| ensemble  | 0.941 | 0.811 | 0.931 |

## 2D vs 3D vs ensemble — which to use (the winner flips with the data)
| mode | ACDC vs human GT (mean) | our recon, all phases (mean) |
|---|---|---|
| 2d | 0.896 | 0.799 (LV .847 / MYO .744 / RV .805) |
| **3d_fullres** | 0.893 | **0.820** (LV .876 / MYO .738 / RV .847) |
| ensemble | **0.902** | 0.805 (LV .848 / MYO .739 / RV .828) |

On **ACDC** (sharp clinical images) the **ensemble wins** (2d≈3d). On **our reconstructions** the picture
**inverts: `3d_fullres` alone is best** (+0.04 RV, +0.03 LV over 2d; MYO a wash) and the **ensemble is
*worse* than 3d alone** — averaging in the weaker 2d softmax drags it down. Our recon is blurry (splat)
and anisotropic; 2d leans on sharp in-plane edges we lack and has no cross-slice context (the weak link),
while 3d's volumetric context is robust to that blur (esp. RV, which spans slices). **Recommendation for
this project: use `3d_fullres` alone** on our recon — most accurate AND ~2× cheaper than the ensemble.
Reserve the full ensemble for clean clinical images (ACDC/raw cine).

Note the figures just below are the 6-subject ED-only first run where ensemble edged 3d marginally;
the all-phases n=30 comparison above is the decisive one. MYO lowest as expected (thin structure). Overlays:
`result/nnunet_mnms_overlays_3d/` (red=LV, yellow=MYO, cyan=RV) — anatomically correct rings/crescents.

## Validation vs human GT — ACDC test set (the "is it actually good" test)
ACDC ships human GT at ED+ES. We ran the model on the **50-patient official test split** (100 frames,
`scratch/data/ACDC/testing`) and Dice'd against GT. **This is cross-dataset** (the model was trained
on M&Ms, never on ACDC), so it's a real generalization test, not a home-field one.

**Label-convention gotcha (critical):** ACDC GT is `1=RV, 2=MYO, 3=LV`; the Task114 model outputs
`1=LV, 2=MYO, 3=RV` (confirmed empirically from overlays). `eval_acdc.py` remaps per structure
(LV: pred==1 vs gt==3 · MYO: 2 vs 2 · RV: pred==3 vs gt==1). Forget this and LV/RV Dice collapse.

**Dice vs human GT (n=100 frames):**

| mode      | LV (ED/ES/all)        | MYO (all) | RV (all) | mean3 |
|-----------|-----------------------|-----------|----------|-------|
| 3d_fullres| 0.947 / 0.901 / 0.924 | 0.869     | 0.888    | 0.893 |
| 2d        | 0.949 / 0.892 / 0.921 | 0.868     | 0.899    | 0.896 |
| **ensemble** | **0.950 / 0.906 / 0.928** | **0.875** | **0.903** | **0.902** |

For reference, in-domain ACDC-trained nnU-Net scores ~0.93 LV / 0.89 MYO / 0.90 RV — so a M&Ms model
hitting **0.93/0.88/0.90 with zero ACDC training** is strong and confirms it's trustworthy as an eval
metric. ES-LV is the weakest cell (small contracted cavity, expected). Overlays (pred vs GT, label
colors unified): `result/acdc_mnms_pred_vs_gt.png`. Tools: `prep_acdc.py`, `eval_acdc.py`.

## Caveats / not-yet-done
- This is **ED only (t00)**, where our model is strongest; segmentation Dice will likely drop at
  systole / for OOD OCMR recons — that's the point (it becomes a discriminative metric).
- Dice here is **seg(pred) vs seg(gt)**, an internal-consistency / anatomy-preservation measure, not
  vs a human label. There is no GT segmentation for our subjects — but that's fine for a relative
  recon-quality metric.
- Not yet wired into the eval harness or run across phases / full val set. `tools/nnunet_mnms_eval/`
  has the three scripts to do so.
