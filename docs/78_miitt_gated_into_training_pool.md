# 78 — MIITT gated cine added to the training pool (`pooled_miitt`)

> **TL;DR & takeaway**
> The MIITT gated arm (U-Michigan paired gated/real-time cine, J. Hamilton — 10 volunteers +
> 3 patients) is now a first-class training source. `tools/convert_miitt_to_12phase.py` picks 12
> of its 30 gated phases (ED-anchored, no interpolation) into `scratch/data/MIITT_sax/`, mirroring
> `ACDC_sax`/`MNMs_sax`; the source tree is never written to. Cohort 935/133/269 → **940/136/274**
> via `training/splits/pooled_miitt.txt` (= `pooled.txt` verbatim + 13 lines). Epoch cost **+0.5%**.
> **Three traps found and handled: (1) all 13 subjects resolve to `subj_id="gated"` under the
> source layout, (2) `heart_weight>0` HARD-RAISES without `heart_roi_canonical`, (3) `default.yaml`
> hardcodes `limit_train_batches=935` / `limit_val_batches=266` and truncating val to 266 silently
> deletes exactly the ES entries of the new val subjects, killing their EF with no error.**
> Verified end-to-end by a 1-epoch MIITT-only run: `loss_heart` nonzero, `[ef] n=3 (skipped 0)`,
> `source:MIITT` strata, 367/367 tests pass.

## 1. Why

MIITT is the only dataset here with **paired** ECG-gated and real-time free-breathing SAX cine of
the same subject, so it sits closest to the research goal's target domain. It had been eval-only.
Putting part of it in train tests whether a small in-domain slice helps; the rest stays held out.

## 2. What was built

`tools/convert_miitt_to_12phase.py` — reads `MIITT/nifti/<subj>/gated/sax/` **read-only**, writes:

```
scratch/data/MIITT_sax/MIITT_<subj>/sax/3d_recon/sax_frame_{00..11}.nii.gz
                                       /sax/heart_seg.nii.gz            (X,Y,Z,12)
                                       /sax/heart_roi.nii.gz            (X,Y,Z) native union
                                       /sax/heart_seg_canonical.nii.gz  (256,256,D,12)
                                       /sax/heart_roi_canonical.nii.gz  (256,256,D)
                                       /sax/convert_meta.json
```

The converter is thin because the data was already 95% ready — **verified, not assumed**:

| property | value | how checked |
|---|---|---|
| on-disk layout | already the CMRx `3d_recon/sax_frame_NN` layout | `convert_miitt_to_nifti.py` wrote it that way on purpose |
| orientation | LPS, clean axis-aligned diagonal affine | asserted per subject in `check_geometry` (raises, never warns) |
| spacing | 1.5 × 1.5 in-plane, **dz = 10 mm** (8 mm thickness + 2 mm gap) | real protocol values from the data author, not placeholders |
| slice order | **apex-at-z0, 13/13**, f1/f2 agree, 0 undetermined | `tools/render_slice_order_check.py` on the CONVERTED tree |
| canonical preprocess | `(12,256,256,D) fp16`, `dz_mm=10.0`, in-FOV frac 0.71 | ran the real monai transform before writing any code |

So the only real work is **which 12 of 30**: `pick_frames` is imported verbatim from
`convert_to_sax_layout.py` — `native_idx(j) = (ed + round(j*T/12)) % T`, nearest native frame, NO
temporal interpolation (blended targets teach blur). ED comes from `cardiac_phase.csv` and is
never assumed 0: **10 of 13 subjects have ED = 28 or 29**, and the `% T` wrap is what makes that
work. Images are byte-copied (`shutil.copy2`) since the source frames already carry the exact
affine/dtype we want.

Slice order was **not** flipped. The check ran on the converted tree deliberately: source frame 0
is not ED for most subjects, so the seg-based features would have been read off the wrong phase.

## 3. The three traps

### 3.1 `subj_id` collides at `"gated"` for every subject

`preprocess.build_data_dicts` derives `subj_id = basename(dirname(sax_dir))`. Under the source
layout `.../Volunteer1/gated/sax` that is **`"gated"` for all 13**, which would collide in
`ef_tmp/pred/*.nii.gz` and `val_per_subject.csv`. Hence the `MIITT_sax/MIITT_<subj>/sax/` tree:
directory names are source-prefixed and globally unique, exactly like `ACDC_patient001` /
`MNMs_A0S9V9`, so `unit == subj` and everything downstream keys correctly.

### 3.2 `heart_weight > 0` raises without `heart_roi_canonical`

`training/loss.py` raises `RuntimeError` — deliberately, not a silent 0.0 — when a sample has no
`heart_roi_canonical`. Every shipped pooled config sets `heart_weight`, and the target run uses
**2.0**, so the canonical ROI is a hard requirement, not a nice-to-have.

Free to satisfy: MIITT already ships the persisted native nnU-Net (Task114) `heart_seg.nii.gz` on
the same affine as the images, so `assemble_whs.build_canonical_siblings` rebuilds both canonical
files with no GPU and no nnU-Net rerun. The native `heart_roi.nii.gz` is **rebuilt from the 12
selected frames** rather than copied from the source (whose union is over all 30), so `roi_vox`
means the same thing across sources.

### 3.3 `limit_val_batches` truncation deletes the new subjects' ES volumes

`default.yaml` hardcodes `limit_train_batches: 935`, `limit_val_batches: 266`,
`logging.log_visual_frequency.train: 935`. With 940/136 these truncate:

- train 935 of 940 → a seed-dependent 5-subject dropout every epoch.
- val 266 of 272 is the nasty one. `ef_val_sweep` enumerates **all 136 ED entries first, then all
  136 ES entries**, so cutting 272 → 266 removes the last 6 **ES** entries specifically — the ES
  volume of the 3 MIITT val subjects and 3 others. `EF = (EDV − ESV)/EDV`, so those subjects would
  drop out of the EF metric with no error and no warning. This is the same failure the
  `mri_dataset.py` `len_train` comment documents for a different cause.

All three are overridden in the sbatch recipe. **Re-derive them whenever the split changes.**

## 4. Cohort and split

`training/splits/pooled_miitt.txt` is `pooled.txt` **verbatim** plus 13 lines
(`diff` confirms: 13 additions, zero other changes). Hand-assembled on purpose —
re-running `build_pooled_split.py` would reshuffle the 359 CMRxRecon2025 subjects, since 2025
absorbs the whole-pool 7:1:2 residual and that residual changes the moment a source joins.

| split | count | MIITT members |
|---|---|---|
| train | 935 → **940** | Volunteer 1–5 |
| val | 133 → **136** | Volunteer 6–8 |
| test | 269 → **274** | Volunteer 9, 10 + ARVC / HCM / cardiomyopathy-AFib |

The 8 train+val subjects' **real-time** arm is no longer subject-unseen; the 5 test subjects keep
both arms clean for OOD evaluation. All 13 subjects are paired, so this costs 8 of 13 possible
RT-eval subjects, not a specific pre-committed eval set.

## 5. Metadata plumbing

Regenerated, never hand-edited, and each swap verified by diff (zero pre-existing rows changed):

- `scratch/data/whs/rows/MIITT_*.csv` (13) — written by the converter, byte-compatible with
  `assemble_whs.py`'s row format including its `flag = "low" if mean_lab < 4.0 else "ok"` rule.
  All 13 land at **`ok`** (mean labeled planes/frame 7.6–9.5).
- `scratch/data/whs/whs_manifest.csv` — 1343 → 1356 via the documented
  `cat rows/*.csv` collation.
- `scratch/data/whs/worklist.txt` — 13 `miitt_sax gated <abs sax dir>` lines.
- `scratch/data/whs/cardiac_phase.csv` — 1514 → 1527 rows by re-running
  `compute_cardiac_phase.py`. **ED came out 0 for all 13**, confirming the anchoring; ES lands on
  the advisory 12-grid index; EF drifts ≤ 1.2 pts from the native-30 values (the expected
  ES-quantization cost, consistent with the ≈0.24 pt median measured for ACDC).
- `training/splits/manifest.csv` — 13 rows appended surgically. **Not** regenerated:
  `build_manifest.py` writes `split` blank by design, so re-running it would wipe the split column
  for all 1343 subjects.

Two one-word code changes let the converted tree flow through the existing tooling:
`assemble_whs.unit_id` and `compute_cardiac_phase.converted_labels` now accept `miitt_sax`
alongside `acdc_sax`/`mnms_sax`.

`group` uses the existing vocabulary (`NOR` for the 10 volunteers, `ARV`/`HCM`/`Other` for the
patients) because `build_manifest.py` maps `pathology_label = "healthy" if group == "NOR" else
"diseased"` — a novel string there would silently label a healthy volunteer diseased.
**`vendor` is left EMPTY**: MIITT ships no scanner metadata anywhere in the repo, so these
subjects drop out of vendor-stratified analyses rather than polluting them with a guess.

## 6. Verification

1. **Canonical preprocess**, real monai transform, before writing the converter —
   `(12,256,256,13) fp16`, `dz_mm=10.0`, intensities in [0,1].
2. **`MRIDataset.get_data`** on a converted subject — S=13 slots (one per plane, `one_frame_per_slice`),
   `heart_roi_canonical` present and shape-matched to `gt_target_volume`, ROI frac 4.3%.
3. **Split + EF sweep** on `pooled_miitt.txt` — 940 train / 136 val (272 sweep entries), no
   `KeyError`; MIITT val entries resolve to ED=0 and ES=5/5/4 matching `cardiac_phase.csv`.
4. **1-epoch MIITT-only training smoke** (5 train / 6 val steps, hw2 recipe, `heart_weight=2.0`,
   aggressive aug, offline wandb), exit 0:
   - `Loss/train_loss_heart: 0.7030` — nonzero, so the ROI reached the loss and §3.2 does not fire.
   - `[ef] epoch 0: ... n=3 (skipped 0)` — all 3 MIITT val subjects reach the EF metric; the
     `seg_flag="ok"` plumbing works (a blank flag would have silently skipped all 3).
   - `[val strata] source:MIITT` — the `training/splits/manifest.csv` join works.
   - `psnr_3d_heartseg`, per-phase panels, per-subject CSV, checkpoint all produced.
   - Identity-Δ baseline on MIITT: full 19.83 / bbox 18.37 / motion 15.35 dB.
   - Metric VALUES are meaningless here (base VGGT-1B after 5 steps); only the plumbing is proven.
5. **`pytest tests/` — 367 passed.**

## 7. How to run it

`sbatch/train_pooled1337_dpt_augaggressive_224_hw2_miitt.sh` — identical to the `augaggr224hw2`
arm (job 57366221) except `VARIANT_TAG` and four overrides: `split_file`, `limit_train_batches=940`,
`limit_val_batches=272`, `logging.log_visual_frequency.train=940`.

**Reading the A/B against hw2:** the pooled val mean moves simply because 3 subjects joined val.
For the honest head-to-head, recompute over the original 133 val ids from `val_per_subject.csv`
(`tools/load_run.py`) instead of comparing headline scalars.

## 8. Open

- **Dose.** 5 of 940 train subjects = **0.5% of gradient steps**. There is no per-source sampler
  weighting, so if this proves too weak the crude lever is duplicating the MIITT lines in the
  split file. Shipped at ×1 to keep the run a minimal delta from hw2.
- **Vendor** unknown (see §5) — would need to come from the data author.
- The 3 MIITT **patients** are in `test` and have never been run through anything.
