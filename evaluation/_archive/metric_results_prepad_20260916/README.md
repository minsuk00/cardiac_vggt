# `metric_results_prepad_20260916/` — every v2 result scored under the TIGHT `mask_heart` segmentation crop

Moved here 2026-09-16, **not deleted**. This is the full `evaluation/metric_results/{val,test}/`
tree as it stood after docs/103 (v2 baselines, gated diagnostic, NiftyMIC rerun, `final518_hw2`
ep250, Fetal CMR 4D pilot).

## Why these numbers are superseded

docs/104 §4 measured that cropping every cine to the tight `mask_heart` (seg union + 6 mm
in-plane) before nnU-Net inflates a reconstruction's ES LV volume far more than GT's: **+5.4 pp EF
MAE paired (p = 0.002)** on the 19 cmrx2024 val subjects shared with the pre-v2 archive. A further
**+10 mm in-plane pad** (`mask_heart_pad10.nii.gz`, z-extent unchanged) removes it (padded-crop EF
within ~0.5 pp of full-FOV). The harness now segments under the padded crop and the classical
baselines reconstruct inside the padded mask (docs/107), so:

- every `_ef/*.json` here is ~2–5 pp pessimistic (arm-dependent: less at 518 px) and not comparable
  to anything scored afterwards;
- image metrics (`<cohort>/*.json`) for VGGT and Dangi arms are unaffected by the crop and would
  reproduce, but they are archived with the rest so no split tree mixes the two protocols;
- the mask-dependent baseline recons (`svrtk3d*`, `nesvor*`, `niftymic*`, `fetal_cmr_4d*`) and every
  `seg_gt/` cache that produced these numbers are in `scratch/eval/_archive_prepad_20260916/`.

The headline PSNR key in these files is the ROI-peak `breath_psnr`; the paper switched to
`breath_psnr_unit_peak` (docs/106), which is also stored in every file here.
