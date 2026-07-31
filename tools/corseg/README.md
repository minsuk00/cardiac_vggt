# CorSeg-CineSAX — headless runner + head-to-head bench vs our nnU-Net

Candidate replacement for the M&Ms nnU-Net (Task114) we currently use as the segmentation-based
recon metric. Findings and the ship decision live in `docs/57_corseg_segmentation_evaluation.md`.

- **Upstream code**: `CorSeg/` (cloned from https://github.com/RunhaoXu2003/CorSeg) + `CorSeg/corseg.pdf`
  (medRxiv preprint, doi 10.64898/2026.04.01.26349955).
- **Weights**: `scratch/data/corseg/ModelWeight-CorSeg-CineSAX_MedNextL.pth` (741 MB, MedNeXt-L,
  61.7 M params, `best_test_dice` 0.890 @ epoch 270). Extracted from `scratch/CorSeg-ModelWeights.zip`;
  the other 6 GB in that zip is two Windows `.exe` bundles and is not needed on Linux.
- **Env**: runs in **`svr` as-is — no installs**. monai 1.6.0 already ships `create_mednext`; torch,
  nibabel, scipy, matplotlib are present. Only PyQt6/pydicom are missing and both are GUI/DICOM-only.

## Labels (differ from ours — easy to get wrong)

| convention | LV cavity | myocardium | RV |
|---|---|---|---|
| CorSeg      | **2** | **1** | 3 |
| nnU-Net Task114 | **1** | **2** | 3 |
| ACDC GT     | 3 | 2 | 1 |

## Why not just use the upstream script

The release ships only a PyQt6 GUI, and its inference path contradicts the paper:

- `load_image` collapses **any** volume to a single 2D slice (`np.take(..., shape[argmin]//2)`),
  so it can never segment a stack.
- `_infer_one` resizes the whole slice to 224x224 with `F.interpolate`, **ignoring voxel spacing**.
  The paper (Methods 2.3) resamples to **1.25 mm** in-plane and *then* center-crops/zero-pads to
  224. On our canonical 256x256 @ 1.4 mm grid the GUI path puts the heart at ~1.6 mm/px — ~22%
  too small — and measurably degrades the result (see docs/57).

`corseg_infer.py` implements the paper-faithful path and loops over every z-slice and phase.
`--mode gui` reproduces the shipped behaviour and exists only as an ablation.

## Usage

```bash
# stage the checkpoint node-local first: GPFS torch.load of 741 MB is the slow part
cp scratch/data/corseg/ModelWeight-CorSeg-CineSAX_MedNextL.pth /tmp/corseg_mednextl.pth

# segment a directory of 3D/4D SAX NIfTIs
micromamba run -n svr python tools/corseg/corseg_infer.py \
    --input <dir-or-file> --out <seg_dir> --mode paper \
    --ckpt /tmp/corseg_mednextl.pth --device cuda [--postproc]

# all-z-slice visual comparison against our nnU-Net (never judge from the mid slice alone)
micromamba run -n svr python tools/corseg/render_corseg_panels.py \
    --subject_dir scratch/eval/cmrxrecon/out/<subj> \
    --corseg_paper <seg_dir> [--corseg_gui <seg_dir>] --t 0 --zoom \
    --out result/corseg/panel_<subj>_t00.png

# ACDC head-to-head against REAL human GT (the only honest accuracy comparison here)
micromamba run -n svr python tools/corseg/bench_acdc.py stage \
    --img_dir scratch/data/nnunet_mnms/acdc/inputs --gt_dir scratch/data/nnunet_mnms/acdc/gt \
    --out_img /tmp/corseg_acdc/roi_inputs --out_gt /tmp/corseg_acdc/roi_gt
micromamba run -n svr python tools/corseg/bench_acdc.py score \
    --pred_dir <preds> --gt_dir <gt> --pred_conv corseg|t114 --tag "..." --out <json>
```

## Files

| file | what |
|---|---|
| `corseg_infer.py` | paper-faithful headless inference; 3D/4D; batched over z; `--mode paper\|gui` |
| `corseg_postproc.py` | the 3 anatomical post-processing steps, **extracted verbatim** from the upstream GUI (which can't be imported headlessly because PyQt6 is a top-level import) |
| `bench_acdc.py` | `stage` ROI crops + `score` Dice vs human GT, with the label conventions above |
| `render_corseg_panels.py` | all-z-slice panels on our canonical volumes: image vs CorSeg(paper) vs CorSeg(gui) vs nnU-Net |
| `render_acdc_3way.py` | all-z-slice panels on ACDC: **human GT** vs CorSeg vs nnU-Net, with per-case Dice. Works for full-FOV *or* ROI-cropped inputs — just point `--img_dir/--gt_dir/--corseg_dir/--nnunet_dir` at the ROI dirs |
| `render_val_grid.py` | multi-subject CorSeg-vs-nnU-Net grid on the CMRxRecon canonical cube at a chosen phase |

## Durable outputs: `scratch/data/corseg/bench/`

**Do not stage intermediates in bare `/tmp`** — it is SLURM node-local and *was wiped mid-session*
here, destroying a full ROI benchmark. Regenerated outputs live on GPFS:

```
scratch/data/corseg/bench/
  roi_inputs/ roi_gt/      # ROI-cropped ACDC images + cropped human GT (bench_acdc.py stage)
  corseg_full/ corseg_roi/ # CorSeg predictions, full FOV and ROI-cropped
  nnunet_roi/              # nnU-Net Task114 2d on the same ROI crops
  res_*.json               # scored results
```
(nnU-Net *full-FOV* predictions already live durably at `scratch/data/nnunet_mnms/acdc/seg_2d`.)
Only the 741 MB checkpoint should be copied to `/tmp` — it is a pure read cache and cheap to redo.
The ROI score **reproduced exactly (0.4126)** after regeneration, which is itself a determinism check.

## Verifications done (don't re-litigate)

- `center_pad_crop` matches MONAI `ResizeWithPadOrCrop` **exactly** on 5 shape cases (crop, pad,
  mixed, odd sizes), and its inverse round-trips.
- `bench_acdc.py score` reproduces the existing `tools/nnunet_mnms_eval/eval_acdc.py` numbers to
  3 decimals on nnU-Net 2d (0.896) and the ensemble (0.9018, matching docs/15).
- Fault-injected: scoring CorSeg with the wrong (unswapped) convention collapses LV to 0.049 and
  MYO to 0.061 while RV stays 0.899 — so the scorer is sensitive to the mapping and the mapping is right.
- **ACDC affine gotcha**: ACDC's affine 3x3 is `diag(-1,-1,1)` with the real spacing only in
  `pixdim`. Copying `im.affine` into a new `Nifti1Image` silently stamps 1.0 mm and destroys any
  spacing-aware preprocessing. `stage` builds a clean diagonal affine from `header.get_zooms()`.
