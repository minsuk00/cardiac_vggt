# 98 — The v2 baseline-comparison eval pipeline: registration wired in, crop-everyone segmentation, same-input scatter bundle, native NeSVoR, GT-seg cache

> **TL;DR & takeaway** (2026-09-13). The `evaluation/` harness was rebuilt end-to-end on
> `pooled_curated_v2.txt` (val 111 / test 180 bundles across 7 sources, every pre-v2 result and
> volume archived, nothing deleted) and five protocol changes shipped, all prove-it-verified
> (3 reviewers, 13 candidate defects, 6 confirmed and fixed):
> **(1)** docs/83's 6-DOF NCC registration + PSF is now **inside `image_metrics.score_subject`**
> — every arm reports a registered headline (`{var}_*_mean`) AND an unregistered raw score
> (`{var}_*_raw_mean`) from ONE shared intensity gauge, so the two are comparable;
> **(2)** SSIM is now the standard windowed skimage SSIM per SAX slice (old global statistic was
> not SSIM; every prior SSIM number is superseded);
> **(3)** the shared nnU-Net segmenter is measurably worse on a heart-ROI crop than on full FOV
> (ACDC expert labels, n=40: LV/MYO/RV ED Dice 0.952/0.871/0.918 → 0.936/0.818/0.868 with
> catastrophic tails), and SVRTK/NeSVoR only ever reconstruct inside that ROI by protocol —
> so **every arm, GT and VGGT included, is cropped to `mask_heart` before segmentation**.
> Uniform noise, no bias. Looser ROIs and CorSeg were measured and rejected;
> **(4)** a **`scatter/`** bundle variant freezes VGGT's own one-frame-per-slice random-phase
> draw into `manifest["scatter"]` so SVRTK/NeSVoR get **byte-identical input** to VGGT
> (verified to 6e-8 on D=6..20 subjects, val and test) — the paper's same-input headline;
> **(5)** NeSVoR runs natively (docs/90 env), 174 s/phase on A40 vs the container's 459 s.
> Plus: GT is segmented **once per subject** (`seg_gt/` cache, keyed on GT+ROI sha256) and can be
> pre-warmed before any arm exists (`ef_dice.py dump --gt-only`). A self-norm gauge bug
> (NeSVoR's anchor computed from the PSF-blurred read, inflating scale 1.6×, 19.9→15.2 dB) was
> caught and fixed in the same pass. **Generation campaign launched 2026-09-13** (SVRTK
> 61046556, NeSVoR 61046557, VGGT curation arms 61052860-3 + 61053923-4).

---

## 1. Why this doc exists

docs/83 designed the pose-gauge + PSF protocol and docs/89 recorded that `pose_psf.py` was
built but **not wired into the real scorer**. docs/97 shipped the curated v2 split but every
eval bundle on disk was still from the old split (silent contamination, flagged in the
2026-09-11 handoff). This doc records the session that closed both gaps and the protocol
decisions made along the way. The 2026-09-12 handoff (registration/PSF build) and the
2026-09-13 handoff (this session) are the agent-level records; this is the durable summary.

## 2. Archival (nothing deleted)

- `scratch/eval/{_ef_ood,acdc,cmrx2023,cmrx2024,cmrx2025,miitt,miitt_rt,mnms,ocmr}` →
  `scratch/eval/_archive_prev2_20260913/` (plain `mv`, ~300 GB).
- `evaluation/metric_results/{_ef,_floors,acdc,…,ocmr}` → `evaluation/_archive/metric_results_prev2_20260913/`
  (`git mv`, 121 renames).

Rule: nothing pre-v2 may be averaged into a v2 number. Post-2026-09-13 results are a fresh
series; do not compare PSNR/SSIM/NCC across this boundary (SSIM is a different statistic now,
registration changes every baseline number).

**Cohort summaries are split-keyed:** `metric_results/<split>/<ds>/<arm>.json` and
`metric_results/<split>/_ef/<arm>.json` (`paths.summary`/`ef_summary` take `split`, no default).
Fixed 2026-09-13 after the val and test eval jobs of one arm were found overwriting the same
summary file; per-subject `metrics.json` were never affected (each subject is in one split), so
the fix needed only a re-aggregate, no rescoring. The same fix made `eval_pooled_val.sh` score only
its own split's subjects and skip the val-only miitt/ocmr on a test run (`eval_ef_dice.sh` too).

## 3. Registration + PSF wired into `image_metrics.score_subject`

Per variant (`clean`/`breath`), per phase `t`:

```python
vol_t, vaff = pose_psf.load_blurred(rp, thick, inplane_mm)   # PSF only for PSF_METHODS
pose        = pose_psf.fit_rigid(vol_t, vaff, gt[t], mask, aff)  # 6-DOF, NCC-max, ±8 mm/±8°
rec.append(pose_psf.resample(vol_t, vaff, shape_xyz, aff, pose))
raw.append(load_canon(rp, shape_xyz, aff))                   # unregistered, unblurred
poses.append(pose.record())                                  # -> metrics.json["poses"][var]
```

Both `rec` and `raw` are scored through the same `prep_recon → psnr/ssim/ncc` path. The
**self-norm gauge (`norm_map`) is computed once from `raw`** and shared. This is the fix for
the gauge bug found mid-session: computing the map from the blurred volume let the blur
spread NeSVoR's apparent coverage into brighter non-heart stack voxels, inflating its scale
~1.6× (PSNR 19.9 → 15.2 dB on one subject while pose-only sat at 21.7 dB).

Registration is real signal, not fitter noise: the fitter's self-fit error is 0.001 mm and it
recovers exact integer-mm injected shifts to 1e-3 mm (prove-it measurements, `temp/reg/proveit/`).
VGGT's registration gain is small because VGGT sits at ~0 mm already (docs/83).

PSF applies only to deconvolvers (`PSF_METHODS = svrtk3d, nesvor, niftymic, fetal_cmr_4d, cinevol`);
`pose_psf.base_method()` strips `_scatter`/`_debug` so same-input arms inherit their base
method's PSF/gauge treatment.

## 4. SSIM is now real SSIM

Old: a homegrown global (whole-ROI mean/variance) statistic. New: skimage
`structural_similarity`, 11×11 Gaussian window, σ=1.5, `data_range=1.0`, computed **per 2D SAX
slice** (a 3D window would span ~80 mm in z at 12 mm pitch) and averaged over ROI voxels.
Every previously reported SSIM is superseded.

## 5. Segmentation fairness: crop everyone

**How it surfaced.** After registration, SVRTK's EDV MAE roughly doubled (16 → 31 mL). Rendered
panels (`temp/reg/figs/edv_example_*.png`) showed nnU-Net's LV contour flipping between planes
under sub-plane pose shifts — a segmentation instability at the black ROI boundary, not a
registration artifact.

**Root cause, measured.** SVRTK/NeSVoR reconstruct only inside the heart ROI (their published
protocol; SVRTK cardiac mode and Fetal CMR 4D require it). nnU-Net Task114 was trained on
full-FOV images. On 40 ACDC subjects against ACDC's own expert labels:

| input | LV | MYO | RV |
|---|---|---|---|
| full FOV, ED Dice | 0.952 | 0.871 | 0.918 |
| heart-ROI crop, ED Dice | 0.936 | 0.818 | 0.868 |

with per-subject tails (one LV 0.93 → 0.62, basal planes lost). Scoring GT/VGGT full-FOV
would hand them a segmenter advantage unrelated to reconstruction.

**Rejected alternatives (measured).**
- Rectangular bbox +30 mm or 20 mm-dilated ROI: halves the mean gap but worst-case Dice 0.32,
  EF errors to 85 points. A looser black boundary confuses nnU-Net differently, not less.
- CorSeg-CineSAX (docs/57): 0.89 → 0.41 Dice on crops (fixed 224² canvas, crop fills ~17%);
  magnifying the crop makes it worse (0.30) by breaking its 1.25 mm voxel prior.
- Giving the baselines a full-FOV reconstruction mask: **wrong** — changes what they
  reconstruct and violates their protocol. The fix must live in the scoring layer.

**Decision (user's):** `ef_dice._dump_cine` multiplies every cine — GT, VGGT, baselines — by
`mask_heart` before writing per-phase files for nnU-Net. Everyone sees the same input
distribution; the ~0.02–0.05 Dice cost is uniform noise. Absolute Dice/EF are therefore below
full-FOV literature numbers for every method, by design.

## 6. GT-seg cache and `--gt-only`

GT is method-independent, so it is segmented once per subject into
`<subject>/seg_gt/{seg_t00..NN.nii.gz, src.json}`, keyed on `sha256(gt_t00)` **and**
`sha256(mask_heart.nii.gz)` (a regenerated ROI under an unchanged GT must not reuse old segs).
`ef_dice.py dump` skips GT for cached subjects; `score` copies cached segs into the sidx-named
seg dir (atomic tmp+replace).

`ef_dice.py dump <dir> --gt-only` (added 2026-09-13) walks every split subject **without needing
any arm**, writes `cine_gt` itself (`image_metrics.ensure_cine_gt`, factored out of
`score_subject`), dumps only the cropped GT for uncached subjects, and `score` on that manifest
just fills the cache. Verified end-to-end on ocmr (8 subjects; re-dump skips all 8; segs carry
LV/MYO/RV). Run for val+test on 2026-09-13 so the first scored arm pays no GT cost.

## 7. Same-input `scatter/` bundle

`pooled.draw_scatter(rel_path, split, seed, D, T)` builds a one-subject `MRIDataset` exactly as
`run_vggt.make_dataset` does (`reference_slot=True, one_frame_per_slice=True,
continuous_z=False, seq_index=seed`) and reads back the slot draw. `add_scatter` assembles

```
scatter/stack_t{k}[z]   = breath/stack_t{phase_per_plane[z]}[z]   for z != ref
scatter/stack_t{k}[ref] = breath/stack_t{k}[ref]                  (VGGT's reference slot)
```

and freezes `phase_per_plane`/`ref_plane`/`draw` into `manifest["scatter"]`. `run_vggt.pin_scatter`
re-pins the model's companion-slot phases to that draw (defensive; in every measured case the
model's own draw already matched). Baselines read it via `INPUT=scatter` in
`run_svrtk3d.sh`/`run_nesvor.sh` (`run_baselines.py --input scatter`, arm `<method>_scatter`).

**Verified byte-level** on 4 subjects (D=6..20, val and test): VGGT's actual `images_splat` equals
the `scatter/stack_t{k}` plane to 5.96e-8; 0 slots ever re-pinned. 291/291 live bundles carry
`scatter/`. Retrofit onto existing bundles with `pooled.py --add-scatter` (touches nothing else).
`--variant clean --input scatter` is refused: scatter is built from `breath/` only.

The scatter arm's self-norm gauge anchors to the `scatter/` stacks
(`stack_kind = "scatter" if method.endswith("_scatter") else var`); `"scatter"` was added to
`paths.BUNDLE_DIRS`/`_STACK_PREFIX` for that.

## 8. NeSVoR native

`run_nesvor.sh` calls `$NESVOR_BIN` from the `nesvor-t2` env (docs/90) instead of
`singularity exec`; the stamp records `engine_id` (git commit + working-tree diff hash of
`baselines/nesvor/NeSVoR`). 174 s/phase on A40 vs 459 s in the container on the same GPU.
Timing fairness = same GPU class for every timed method: **A40/spgpu for NeSVoR and every VGGT arm**.

## 9. Prove-it review: 6 confirmed defects, all fixed

1. `stamp.json["input_stack"]` recorded the raw variant, so a subject's clean+breath stamps
   differed and `check_variant_stamps` raised for every baseline. Now records the scheme
   (`gated`/`scatter`).
2. Scatter-arm gauge anchored to `breath/` stacks → §7 fix.
3. `run_vggt` created an empty `recon_breath/` before failing on a scatter-less bundle → check
   moved above `os.makedirs`.
4. `run_baselines.py` clobbered a user-supplied `METHOD` env (broke the documented
   `METHOD=svrtk3d_debug` `.dof`-extraction workflow).
5. `--variant clean --input scatter` unguarded → explicit exit.
6. GT-seg cache key omitted the ROI content → `_roi_sha()` added.

Seven other candidates were refuted with measurements (fitter precision above; PSF
extrapolation at padded boundaries bounded; scatter gauge invariance).

## 10. Campaign as launched (2026-09-13)

| job | what | account/partition |
|---|---|---|
| 61046556 (×8) | SVRTK 3D, val+test × gated+scatter, breath | jjparkcv98 / standard |
| 61046557 (×8) | NeSVoR native, same | jjparkcv98 / spgpu (A40) |
| 61052860-3 | VGGT `curation_{curated,uncurated}_ep300`, val+test | jjparkcv98 / spgpu |
| 61053923-4 | VGGT `curation_uncurated_ep177` (matched steps), val+test | jjparkcv98 / spgpu |
| (this node) | `ef_dice.py dump --gt-only` val+test → `seg_gt/` cache | interactive A40 |

**Curation ablation arms.** Both curation runs are 224 px, warm-up + constant 5e-5 LR
(`exp_curation_ablation.yaml`), **not** the paper's cosine default — appendix rows, not the
headline. Curated = 628 subjects × 300 ep = 188,400 steps; uncurated = 1066 × 300 = 319,800.
Uncurated `checkpoint_177.pt` (177 × 1066 ≈ 188,700 steps) is the **matched-steps** arm, so the
comparison exists at both same-epoch and same-steps with no retraining. Val/test of
`pooled_uncurated_ablation.txt` are byte-identical to v2's.

**Headline VGGT arms** are the five `210823094_final518_*` runs (base / diff1000 / hw2 /
motion10 / motion10_hw2), epoch ~123/300 at launch — eval when they finish
(`CKPT=… MODEL_NAME=… SPLIT=… sbatch --account=jjparkcv98 sbatch/eval_pooled_val.sh`).

## 11. Still open

- Baselines still to wire: NiftyMIC runner (in `PSF_METHODS`, no engine shell), FC-SVR (old-split
  Stage-1 ckpt exists; needs v2 retrain + runner), Dangi (reimplement), Fetal CMR 4D (needs the
  docs/35 self-gate), CiNeVol (cite-vs-run undecided).
- Baseline breathing-EPE (`.dof` / NeSVoR pose affines → `resp_diag.json`): scoped, not built.
- NeSVoR masked-boundary behaviour checked on one subject only (docs/89 §6 step 4 still open).
- Nothing from this session is committed until at least one real sbatch batch completes.
