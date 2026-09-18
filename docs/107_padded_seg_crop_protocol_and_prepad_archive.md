# 107 — Padded segmentation crop (`mask_heart_pad10`) adopted; every tight-crop result archived

> **TL;DR & takeaway** (2026-09-16). Implemented the docs/104 decision. Each eval bundle now
> carries `mask_heart_pad10.nii.gz` = `mask_heart` dilated **+7 px (≈10 mm at 1.4 mm) in-plane
> only**, z-extent unchanged, clamped to the FOV (`tools/build_padded_heart_mask.py`, 291/291
> subjects, z-extent and containment asserted). `ef_dice.py` segments under it (ROI-sha keyed, so
> `seg_gt/` caches self-invalidate) and `run_baselines.py` passes it as every shell runner's
> `MASK_FILE`; Dangi and VGGT take no mask and need no regeneration. **Image metrics score under
> the padded mask too** (`image_metrics.py` scoring ROI = `mask_heart_pad10` ∩ FOV — one ROI
> everywhere; an earlier draft of this doc said the tight mask stayed, corrected 2026-09-16 after
> the code was checked: on CMRx24_Test_P012 the wider ROI lifts VGGT's `breath_psnr` 22.18 → 23.27
> dB, so archived tight-ROI image numbers are NOT comparable to re-scored ones). **All old-crop
> results moved, not deleted**: `evaluation/metric_results/{val,test}` →
> `evaluation/_archive/metric_results_prepad_20260916/`; 2041 dirs (`svrtk3d*`, `nesvor*`,
> `niftymic*`, `fetal_cmr_4d*`, `seg_gt`, gated + scatter, all 7 cohorts, 484 GB) →
> `scratch/eval/_archive_prepad_20260916/`. `metric_results/` is now empty; the next campaign is
> test / scatter only (SVRTK, NeSVoR, NiftyMIC regenerated; every arm re-scored) and every 518
> arm at ep300 afterwards.

## 1. Why the pad, in one paragraph

docs/104 §4: the tight crop (seg union + 6 mm in-plane + z±1, `tools/nnunet_mnms_eval/build_heart_roi.py`)
makes nnU-Net over-draw the ES cavity of a *reconstruction* much more than of GT (+5.4 pp paired
EF MAE, p = 0.002, 19 subjects). Padding the crop by a further 10 mm in-plane brings EF within
~0.5 pp of full-FOV segmentation (14.3 vs 14.8 vs 16.8 tight), verified on the full 111-subject
val set for both a 224 px arm (21.6 → 18.4 pp) and `final518_hw2` ep250 (11.2 → 9.8 pp). Restricting
z does nothing (the false-positive LV lives on planes GT has LV at ED), so z-extent is left alone.
Total in-plane margin from the seg union is now 6 + 10 = 16 mm.

## 2. What changed

| file | change |
|---|---|
| `tools/build_padded_heart_mask.py` | new; `binary_dilation(roi, ones((15,15,1))) & fov` → `mask_heart_pad10.nii.gz`; asserts z-extent unchanged and tight ⊆ padded |
| `evaluation/paths.py` | `HEART_MASK_PAD`, `heart_mask_pad()`; `heart_mask()` = the tight ROI, now only the seed the padded mask is built from |
| `evaluation/src/score/image_metrics.py` | scoring ROI `heart_mask` → `heart_mask_pad` (∩ FOV); headline key → `psnr_unit_peak` (docs/106), `psnr` kept for continuity |
| `evaluation/src/score/ef_dice.py` | `_heart_roi` / `_roi_sha` read the padded mask; `seg_gt/src.json` `crop` field records the file name |
| `evaluation/src/engine/run_baselines.py` | `MASK_FILE = paths.HEART_MASK_PAD` (svrtk3d / nesvor / niftymic_v2 / fetal4d shells already honour `MASK_FILE`) |
| `evaluation/README.md` | crop paragraph + quick-ref comment |
| `evaluation/_archive/metric_results_prepad_20260916/README.md` | why the archived numbers are superseded |

Not changed: `run_dangi.py` (no mask), the VGGT runner, the tight `mask_heart.nii.gz` files.
`tests/` 372 passed after the edits.

The FOV clamp is the one deviation from the scratch experiment that produced the docs/104 numbers
(which dilated without clamping). It cannot change a segmentation result — GT and every recon are
zero outside the FOV — it only stops the baselines from reconstructing empty pixels.

## 3. Archive layout

```
evaluation/_archive/metric_results_prepad_20260916/{val,test}/<cohort>/<arm>.json, _ef/<arm>.json
scratch/eval/_archive_prepad_20260916/<cohort>/out/<subject>/{svrtk3d,nesvor,niftymic,fetal_cmr_4d}[_scatter|_norobust]/
scratch/eval/_archive_prepad_20260916/<cohort>/out/<subject>/seg_gt/
```
Bundles (`gt/ clean/ breath/ scatter/ rolled/ mask*.nii.gz manifest.json cine_gt.nii.gz`), VGGT
arms and `dangi*` arms stay in place. `scratch/eval/_smoke_20260913_cmrx2024/` untouched.

## 4. Next (decided)

1. Regenerate `svrtk3d_scatter`, `nesvor_scatter`, `niftymic_scatter` on the **test** split
   (`run_baselines.py --split test --input scatter`, 3 shells).
2. Re-score every test arm: image metrics (numbers CHANGE for every arm incl. VGGT/Dangi — the
   scoring ROI is now the padded mask, see TL;DR) and `ef_dice` (new crop) — headline PSNR key →
   `psnr_unit_peak` (docs/106).
3. Every 518 arm at ep300 through the harness at the end; pick the reporting arm (docs/106 §3).
