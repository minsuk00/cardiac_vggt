# 57 — CorSeg-CineSAX as a replacement for our M&Ms nnU-Net segmenter

> **TL;DR & takeaway** (2026-07-30). Evaluated **CorSeg-CineSAX** (MedNeXt-L, 61.7 M params,
> medRxiv 2026) as a drop-in replacement for the **M&Ms nnU-Net Task114** we currently use for
> Dice/EF recon metrics. **It runs in `svr` with ZERO installs** (monai 1.6.0 already ships
> `create_mednext`) — no new env needed, nothing in `svr` touched. On **ACDC (the only cohort here
> with real human GT, zero-shot for both models)** the answer **splits by metric**. On boundary
> **Dice** it is slightly worse than what we already run: **0.889** vs nnU-Net-2d **0.896** vs
> ensemble **0.902**, the whole gap being **myocardium** (0.844 vs 0.868). But on **volumes and EF —
> which is what `ef_dice.py` actually reports — it is clearly BETTER**: **EF MAE 2.51 vs 4.67 pp**
> (paired Wilcoxon p=3e-4, better on 35/50 subjects, r 0.987 vs 0.962), LV volume MAE **4.4 vs
> 8.9 mL**, and near-zero volume bias on all three structures against nnU-Net's significant
> −3.4% / +2.8% / −4.5%. No paradox: CorSeg's errors are boundary-localised and **unbiased**,
> nnU-Net's are smaller but **systematic** — Dice punishes the first, volume/EF the second.
> **Verdict: keep nnU-Net for Dice, use CorSeg for EF/volumes** (or run both and treat divergence as
> a QA flag); do not switch wholesale, since that re-baselines all of `evaluation/results/`. CorSeg is
> also **8.5× cheaper per volume** — one in-env model instead of a 5/10-model cross-env `nnunet` hop —
> and unlike Task114 (ED/ES labels only) it was trained on **all ~25 cardiac phases**, making it the
> better candidate for our 10 mid-cycle target phases: plausible but **NOT measured** (no cohort here
> has mid-cycle human GT).
>
> **Two traps found, both worth remembering.** (1) **The shipped GUI's inference contradicts the
> paper**: it resizes each slice to 224² ignoring voxel spacing, whereas training resamples to
> **1.25 mm** then center-crops/pads. Our runner implements the paper path; the GUI path costs
> ~0.02 agreement Dice on our canonical grid and visibly drops apical RV. Also `load_image`
> collapses any volume to **one** slice, so the GUI can never segment a stack. (2) **CorSeg
> collapses on heart-ROI-cropped input: 0.889 → 0.413**, while **nnU-Net is untouched by the same
> crop (0.896 → 0.890)** — so *on ROI-cropped images nnU-Net wins overwhelmingly*. 44% of CorSeg
> cases fall under 0.3 and many sit at exactly **0.00**. Cause is **measured**: its input is a fixed
> 224² canvas, full-FOV frames fill **99.7%** of it but ROI crops fill only **17%**, and Dice tracks
> occupancy (r=0.625, p=3.7e-12); nnU-Net escapes because it is patch-based sliding-window. My
> predicted fix — magnify the crop to fill the canvas — **made it worse (0.296)**, because that
> breaks the 1.25 mm scale prior; there is no preprocessing trick, the fix is to not crop.
> **And this is NOT hypothetical for us** — an earlier claim in this doc that "our baselines are not
> ROI-cropped" is **RETRACTED**: I had checked the assembled `cine_clean.nii.gz` instead of
> `recon_clean/vol_tXX.nii.gz`, which is what `ef_dice.py dump` actually feeds the segmenter. The
> `gt` and VGGT arms are indeed the full canonical cube, but **SVRTK3D and NeSVoR emit small ~1.4 mm
> isotropic heart-ROI volumes** (e.g. (72,87,86) and (91,106,98)) — exactly the failing regime. On
> the real SVR outputs (30 subjects): CorSeg leaves **8/30 NeSVoR volumes blank** (median 269 voxels)
> and on SVRTK segments only a **mid-ventricular band**, missing base and apex. ⇒ **if CorSeg is used
> for EF, restrict it to the canonical-cube arms and keep nnU-Net for the SVR baselines**, or it will
> silently mangle the very baselines it is being compared against.
>
> Status: **evaluated, not adopted.** Tooling is git-tracked in `tools/corseg/` and is reusable.

## Why we looked

Our standing segmentation metric is **nnU-Net v1 Task114 (M&Ms 2020 winner)**, run via
`evaluation/engine/run_seg.sh` — which has to hop into a **separate `nnunet` micromamba env**
because it is nnU-Net *v1* (docs/15). It is a 2020-era model trained on ~345 M&Ms subjects with
**ED/ES annotations only**. CorSeg advertises exactly the things that would matter to us: a much
larger multi-centre training set, **full-cardiac-cycle** labels, and full-FOV 2D operation.

## What CorSeg is

From the preprint (`CorSeg/corseg.pdf`, doi 10.64898/2026.04.01.26349955):

- **MedNeXt-L**, 2D, `kernel=5`, 4 output channels; 61.7 M params; MONAI `create_mednext`.
- **Training data**: 1,555 subjects / **12 Chinese centres** / 319,175 labelled 2D images, five
  diseases (NC, HCM, DCM, HHD, CA), **all slices AND all ~25 cardiac phases** — 1,245 train /
  310 internal test. Acquired on **3.0 T Siemens** (Trio/Vida) bSSFP — a good domain match for
  CMRxRecon.
- **External validation** (never trained on): ACDC 0.900, M&Ms1 0.912, M&Ms2 0.914; internal 0.913.
  Cross-domain gap 0.002.
- **Labels: 1=LV myocardium, 2=LV cavity, 3=RV** — labels 1 and 2 are **swapped** relative to
  Task114 (1=LV cavity, 2=myocardium). Easy to get silently wrong.
- **Anatomical post-processing** (3 deterministic steps: largest-component, LV-cavity containment,
  gap filling). Paper: negligible Dice effect (0.912→0.912) but HD95 drops a lot and containment/gap
  violations go to exactly 0%.
- Checkpoint metadata: `epoch 270`, `best_test_dice 0.8903`, `pixdim (1.25,1.25)`, `img_size (224,224)`.

## Environment: no new env, nothing touched

`svr` already satisfies every non-GUI requirement — **verified by loading the model and running a
forward pass, not by reading the requirements table**:

```
monai 1.6.0 | torch 2.13.0+cu130 | nibabel 5.3.3 | numpy 2.2.6 | scipy 1.15.3
create_mednext OK ; load_state_dict(strict=True) -> All keys matched ; fwd (2,1,224,224)->(2,4,224,224)
```

Only **PyQt6** (GUI) and **pydicom** (DICOM input) are absent, and we need neither. MedNeXt landed in
monai 1.3 and we are on 1.6, so the constraint that forced a separate env for nnU-Net v1 does not
apply here. **No `pip install` was run in `svr`.**

## Trap 1 — the shipped GUI does not implement the paper

The release ships only `CorSeg-CineSAX_{en,ch}.py`, a PyQt6 GUI. Two defects make it unusable for us:

| | paper (Methods 2.3, what training did) | shipped GUI (`_infer_one`) |
|---|---|---|
| spacing | resample in-plane to **1.25 mm** | **ignored** |
| to 224² | center **crop / zero-pad** | `F.interpolate` **resize** of the whole slice |
| 3D input | slice-by-slice over the stack | `load_image` reduces the volume to **ONE** slice (`np.take(..., shape[argmin]//2)`) |

On our canonical grid (256 px @ 1.4 mm = 358.4 mm FOV) the GUI path yields an effective 1.6 mm/px —
the heart renders ~22% too small. Measured cost on a val subject (agreement Dice vs nnU-Net,
12 phases): **paper-prep 0.901 vs GUI-prep 0.882**, the loss concentrated in **RV (0.926 → 0.883)**,
and the panel figure shows GUI-prep dropping the apical RV entirely at z=7.

`tools/corseg/corseg_infer.py` implements the paper path (`--mode paper`, default) and keeps
`--mode gui` only as this ablation. Geometry was **verified against MONAI**: `center_pad_crop`
matches `ResizeWithPadOrCrop` exactly on 5 shape cases (pure crop, pure pad, mixed, odd sizes) and
its inverse round-trips.

## Head-to-head on ACDC (real human GT)

ACDC is the honest comparison: 50 test patients × ED/ES = 100 cases with **human** segmentations,
and it is **zero-shot for both** models (Task114 trained on M&Ms; CorSeg's private set excludes it).

**Scorer validated first**: `bench_acdc.py score` reproduces the existing
`tools/nnunet_mnms_eval/eval_acdc.py` to 3 decimals (nnU-Net 2d 0.896; ensemble 0.9018, matching
docs/15's 0.902). **Fault-injected**: scoring CorSeg with the wrong (unswapped) label convention
collapses LV to 0.049 and MYO to 0.061 while RV stays 0.899 — so the scorer is sensitive to the
mapping, and the mapping is right.

### Full field of view (the condition that matches our pipeline)

| method | models | LV cavity | myocardium | RV | **mean** |
|---|---|---|---|---|---|
| nnU-Net Task114 **ensemble** (2d+3d, 5 folds each) | 10 | 0.928 | **0.875** | **0.903** | **0.9018** |
| nnU-Net Task114 **2d** — *what `run_seg.sh` runs today* | 5 | 0.921 | 0.868 | 0.899 | **0.8961** |
| **CorSeg** MedNeXt-L, paper-prep | 1 | **0.925** | 0.844 | 0.899 | **0.8892** |

Per phase (LV / MYO / RV):

| method | ED | ES |
|---|---|---|
| nnU-Net 2d | 0.949 / 0.857 / 0.924 | 0.892 / 0.880 / 0.875 |
| CorSeg | 0.946 / **0.822** / **0.930** | **0.903** / 0.866 / 0.869 |

Reading: **CorSeg's entire deficit is myocardium** (−0.024 vs nnU-Net-2d, −0.031 vs ensemble). It is
slightly *better* at **ES LV cavity** (0.903 vs 0.892) and **ED RV** (0.930 vs 0.924).

### …but on volumes and EF, CorSeg is clearly BETTER — and that is what we actually report

Dice measures boundary overlap. `evaluation/analysis/ef_dice.py` reports **EF from the LV-cavity
volume curve**, so volume accuracy is the metric that matters for us. Same 100 ACDC cases, same
human GT:

| structure | GT mean | CorSeg bias | nnU-Net 2d bias |
|---|---|---|---|
| LV cavity | 137.6 mL | **+1.8 mL (+1.3%)** | −4.7 mL (−3.4%) |
| myocardium | 133.1 mL | **+0.8 mL (+0.6%, n.s.** p=0.39**)** | +3.7 mL (+2.8%, p=6e-3) |
| RV | 132.9 mL | **+1.9 mL (+1.4%, n.s.** p=0.14**)** | −6.0 mL (−4.5%, p=4e-4) |

CorSeg is **near-unbiased on all three** (two of three not significantly different from GT);
nnU-Net is significantly biased on all three. And on **EF** (50 subjects with ED/ES pairs):

| | mean EF | bias | **MAE** | Pearson r |
|---|---|---|---|---|
| human GT | 50.7% | — | — | — |
| **CorSeg** | 49.5% | −1.2 pp | **2.51 pp** | **0.987** |
| nnU-Net 2d | 48.6% | −2.1 pp | **4.67 pp** | 0.962 |

**CorSeg's EF error is roughly half nnU-Net's** — paired Wilcoxon **p=3.0e-4** (paired t p=2.4e-4),
better on **35/50** subjects, median |error| 1.52 vs 3.04 pp, worst case 14.9 vs 18.4 pp. LV volume
MAE **4.4 vs 8.9 mL** (r 0.9972 vs 0.9942).

So the two views genuinely disagree, and the disagreement is not a paradox: CorSeg's myocardial
errors are **boundary-localised and unbiased** (low Dice, ~zero volume bias), whereas nnU-Net's are
**smaller but systematic** (higher Dice, consistent volume offset). Dice punishes the former; volume
and EF punish the latter.

**Correction to an earlier visual read of mine:** from the `patient101` panel I described CorSeg's
myocardial ring as "systematically thinner" than GT. That is **wrong** — measured myocardial volume
bias is **+0.6% and not significant**, and the myo/(myo+cavity) volume fraction is 0.521 for CorSeg
vs 0.523 for GT (nnU-Net 0.536). It was a single-case appearance, not a systematic bias.

We reproduce the paper's ACDC RV exactly (0.899) and land ~0.014 below its LV/MYO (0.925 vs 0.939,
0.844 vs 0.861); the paper scored all 150 ACDC subjects while we score the 50-patient test split,
so a small subset difference is expected. Close enough to consider our implementation faithful.

## Trap 2 — CorSeg collapses on heart-ROI-cropped input

Motivating question: our baselines were believed to emit heart-ROI-cropped volumes. **They do not** —
see "Premise correction" below. But the hypothetical was worth measuring, because a fixed-input 2D
network is exactly the kind of model that breaks on it.

Condition: crop each ACDC case to the bbox of our own ROI recipe
(`tools/nnunet_mnms_eval/build_heart_roi.build_roi`, project defaults `in_mm=6, z_extend=1`) derived
from the **human GT**, so the crop is identical for both models and neither is favoured. The crop
keeps **11.5%** of voxels.

| condition | CorSeg (paper-prep) | nnU-Net 2d |
|---|---|---|
| full FOV | 0.889 | 0.896 |
| **ROI-cropped** | **0.413** | **0.890** |
| Δ | **−0.476** | **−0.006** |

**nnU-Net is essentially immune to the crop** (0.896 → 0.890; ED LV even *improves* to 0.955) because
it does its own spacing-aware preprocessing and runs a **patch-based sliding window**, so input size
is not a constraint. CorSeg's fixed 224² input is.

CorSeg's ROI failure is **bimodal, not a uniform degradation**: median 0.487, **44% of cases below
0.3**, and many ES cases at exactly **0.00** (no output at all), while the best cases reach 0.888 —
full-FOV quality. ED 0.677 vs ES 0.296.

**So for the ROI-cropped question specifically: nnU-Net wins overwhelmingly (0.890 vs 0.413).**

### The failure is SILENT and total — an operational hazard, not just a low score

On the ROI crops CorSeg returns a **completely empty mask (0 foreground voxels)** for **23 of 95**
cases — and **every one of them is an `_ES` case**, i.e. exactly the small-crop end of the
distribution. It raises no exception, logs nothing, and writes a valid, well-formed, all-zero NIfTI.
nnU-Net scores **0.75–0.83** on those same cases. Visual proof: `result/corseg/roi_3way.png`
(a working case, a partial dropout, a total blank) and `result/corseg/roi_3way_silent_failures.png`
(three total blanks with nnU-Net alongside).

Consequences if this ever ran unattended: an empty LV curve makes EF `0/0` → NaN or a divide-by-zero,
and a Dice of 0 is indistinguishable from "the reconstruction is bad" rather than "the segmenter
declined". **Any adoption of CorSeg must assert non-empty foreground per slice/volume and fail loudly**
— `lv_curve` in `ef_dice.py` already returns `None` when a curve is all-zero, which would silently
drop such subjects from the cohort rather than flag them.

**Mechanism, measured rather than reasoned:** the network input is a fixed 224² canvas. Full-FOV
frames fill **99.7%** of it (min 0.911); ROI crops fill only **17.1%** (range 0.091–0.257) because a
~57×60 resampled crop gets zero-padded into 224². Dice tracks that occupancy directly —
**Pearson r=0.625 (p=3.7e-12)**, Spearman 0.656:

| input occupancy | n | mean Dice | fraction total failures |
|---|---|---|---|
| 0.0–0.1 | 1 | 0.000 | 1.00 |
| 0.1–0.2 | 74 | 0.344 | 0.32 |
| 0.2–0.3 | 25 | 0.632 | 0.00 |

ES crops are smaller than ED (occupancy 0.151 vs 0.191), which is exactly why ES is worse. So this
is a **framing/scale** OOD failure, not a myocardial-accuracy failure — consistent with the paper's
own design claim that it is a *full-FOV* model. The claim cuts both ways: it **requires** full FOV.

### My predicted fix was WRONG — recorded because it is the informative part

From the occupancy correlation I predicted that **filling the canvas** would recover performance:
resize the crop up to 224² (i.e. `--mode gui`) instead of zero-padding it. Measured result:

| ROI-crop preprocessing | canvas occupancy | mean Dice |
|---|---|---|
| paper-prep — zero-pad at true 1.25 mm scale | 0.17 | **0.413** |
| gui-prep — magnify to fill 224² | 1.00 | **0.296** (worse) |

Filling the frame made it **worse**, so occupancy is not the whole causal story: magnifying breaks
the **1.25 mm physical-scale prior** the model was trained under. CorSeg needs *both* the trained
physical scale *and* a frame filled with real anatomy — and only genuine full FOV supplies both.
Zero-padding gives correct scale with an empty frame; magnifying gives a full frame at wrong scale;
both fail. **There is no preprocessing trick that rescues ROI crops here — the fix is to not crop.**
(The occupancy correlation r=0.625 remains valid *within* the zero-padded condition; it just does not
license the intervention I inferred from it.)

### A methodological warning: an affine gotcha nearly produced a fake result

The first ROI run scored **0.061**, which I nearly reported as a model property. It was **my bug**.
ACDC's NIfTI affine 3×3 is `diag(-1,-1,1)` with the true spacing living **only in `pixdim`**.
Copying `im.affine` into a fresh `Nifti1Image` therefore silently stamps **1.0 mm** spacing, which
wrecks any spacing-aware preprocessing (CorSeg's 1.25 mm resample *and* nnU-Net's own). Fixed by
building a clean diagonal affine from `header.get_zooms()`; the corrected number is 0.413. The
full-FOV runs were never affected (they read the originals). **Lesson: a catastrophic number is a
prompt to audit your own harness first** — the tell was that all three structures collapsed
together, whereas a label-convention error spares RV (as the fault-injection above shows).

### The same affine gotcha bit my output writer too

`corseg_infer.py` originally saved labels as `Nifti1Image(lab, im.affine)`. For ACDC that stamps
**1.0 mm** zooms on the output, so every physical volume computed from a CorSeg label map came out
~**−95%** — which is exactly how I first "discovered" a nonsensical uniform volume deficit across all
three structures. **Dice was never affected** (voxel counts on a shared grid; re-running after the
fix reproduced 0.8892 exactly), but LV/RV volume and LV mass in mL would have been silently wrong,
and EF only survives because it is a ratio. Fixed by propagating the input **header** as well as the
affine (`Nifti1Image(lab, affine, header=hdr)`), verified to round-trip through disk and to leave
well-formed files (our canonical 1.4/1.4/12.0 grid) untouched.

Generalisable tell: **a uniform error across all structures is a geometry/metadata bug; a
structure-specific one is a model or label-convention issue.**

## ⚠️ RETRACTED: "our baselines are NOT ROI-cropped" — they ARE, for the SVR arms

**This doc previously claimed the opposite. That claim was wrong and is withdrawn.** I checked
`svrtk3d/cine_clean.nii.gz` — the *assembled* canonical 4D cine — when the file that actually reaches
the segmenter is `<method>/recon_clean/vol_tXX.nii.gz`, which `ef_dice.py dump` copies. Those are
**not** on the canonical cube for the classical SVR baselines:

| arm | file fed to the segmenter | shape | spacing | non-zero |
|---|---|---|---|---|
| `gt` | `gt/gt_tXX.nii.gz` | (256,256,12) | 1.4 × 1.4 × **12.0** | 0.85 |
| VGGT | `recon_clean/vol_tXX.nii.gz` | (256,256,12) | 1.4 × 1.4 × **12.0** | 0.86 |
| **SVRTK3D** | `recon_clean/vol_tXX.nii.gz` | **(72,87,86)** | **1.4 isotropic** | 1.00 |
| **NeSVoR** | `recon_clean/vol_tXX.nii.gz` | **(91,106,98)** | **1.4 isotropic** | 0.33 |

(`Test_P012`; shapes vary per subject.) So **SVRTK and NeSVoR emit small ~1.4 mm isotropic
heart-centred ROI volumes**, which is precisely the regime where CorSeg breaks. They are also **RAS**,
whereas our canonical frame is LPS — both segmenters see the same array, so the comparison is fair,
but it is a separate orientation inconsistency worth noting. Slice axis is **axis 2** (verified by
adjacent-slice correlation, 0.983/0.986 vs 0.92–0.96 for the other two axes).

### Measured on the real SVR outputs (30 subjects × 2 methods, t=0)

| baseline | 224² occupancy | blank outputs | median foreground voxels | LV-cavity agreement |
|---|---|---|---|---|
| | | CorSeg / nnU-Net | CorSeg / nnU-Net | CorSeg vs nnU-Net |
| SVRTK3D | 0.150 | **0/30** / 0/30 | 131,262 / 187,919 | mean 0.748, median 0.800 |
| NeSVoR | 0.234 | **8/30** / **0/30** | **269** / 185,512 | **mean 0.000, median 0.000 — 30/30 below 0.3** |

**nnU-Net has zero blanks on both and segments the full stack; CorSeg does not.** Both baselines sit
in the low-occupancy failure band measured on ACDC. Qualitatively
(`result/corseg/svr_roi_3way.png`, `svr_roi_3way_b.png`): on **SVRTK** CorSeg covers only a
**mid-ventricular band** (z≈23–46 of 86), missing base and apex, where nnU-Net segments z=0→85; on
**NeSVoR** CorSeg collapses to nothing (23 voxels on `Test_P012`) while nnU-Net produces clean
LV/MYO/RV across z=8→77. The NeSVoR agreement of **exactly 0.000 on all 30 subjects** means CorSeg
and nnU-Net share **no overlapping LV voxel anywhere** on that arm.

**Consequence for the ship decision:** the ROI failure is **not** hypothetical for us. Any use of
CorSeg for EF/volumes must be restricted to the **canonical-cube arms (`gt`, VGGT)**; the SVR baseline
arms must keep using nnU-Net, or CorSeg's EF advantage would be bought at the cost of silently
mangling exactly the baselines it is being compared against. `heart_roi`/`mask_heart` remain metric
*masks*, and the canonical arms are genuinely full-FOV — that part of the original claim stands.

## Sample val volume (visual check)

`Val_P048`, all 12 canonical z-planes, all 12 phases — figures in `result/corseg/`:

- `panel_Val_P048_t00_zoom.png` — image / CorSeg-paper / CorSeg-GUI / nnU-Net contours per z-plane.
  All three agree closely at z=1–7; GUI-prep loses apical RV at z=7.
- CorSeg-vs-nnU-Net agreement over 12 phases: LV 0.922 / MYO 0.856 / RV 0.926, **mean 0.901**.
- EF from LV-cavity curve: **CorSeg 58.8% vs nnU-Net 62.1%**; both give a clean monotone
  contraction–relaxation curve (ED 140 mL → ES 58 mL for CorSeg).

### Provenance of these val figures: CMRxRecon2024, **v1 recon**, our `val` split

Worth stating exactly, because two staleness traps overlap here.

**Dataset / split.** CMRxRecon2024 `Cine_combined`, subjects `CMRx24_Val_P048` / `P054` / `P055`.
All three are genuinely in **our `val` split** of `training/splits/random_8_1_1.txt` (verified by
parsing the `[train]/[val]/[test]` sections, not by trusting the directory name). Note the
`Train_`/`Val_`/`Test_` prefix is the **challenge's own** set naming and is **decoupled** from our
split: the 30-subject eval cohort is **30/30 in our val split** while its prefixes read 17 `Train_` /
10 `Test_` / 3 `Val_`. The split file itself is **DEPRECATED (2026-07-25)** — 7 duplicate subjects
were archived and 2 of its 30 val subjects were copies of trained-on subjects.

**Recon version = v1, not v2.** `evaluation/engine/build_inputs/cmrxrecon.py` reads
`DATA_ROOT = scratch/data/CMRxRecon2024/Cine_combined` — i.e. *not* the
`CMRxRecon2024_recon_v1_espirit_imagedomain/` path — **but that directory's contents changed under
it.** The bundle was written **2026-07-12**; the current `Cine_combined` volumes were rewritten
**2026-07-27** by the docs/54 v2 re-reconstruction. So the figures carry **v1** content (the
ESPIRiT-image-domain-bug recon, now preserved in the `_recon_v1_*` directory). Confirmed on content,
not just mtimes, for `Val_P048`:

- `corr(v1, v2) = 0.391` on the native frame — the two reconstructions differ materially.
- v1 adjacent-slice correlation **collapses at the last link** (0.254 vs ~0.70 elsewhere) = the
  docs/56 odd-Z **slice roll**; v2 shows no such break (min 0.584) = roll fixed.
- The bundle renders exactly that rolled layout (most basal slice at z=9), matching v1.

So `result/corseg/panel_Val_*`, `val_grid_*` and `Val_P048_basal_planes.png` are **v1, pre-roll-fix**.
The live source is correct now (`result/corseg/P048_live_source_slices.png` runs base→apex), and
regenerating `scratch/eval/` is already on the owed list in docs/54/56.

### ⚠️ The whole `scratch/eval` bundle is DEPRECATED and awaiting regeneration

**User-confirmed (2026-07-30): these volumes are deprecated and will be rebuilt.** That matches the
owed-regeneration item in docs/54/56. Everything in this doc that reads
`scratch/eval/cmrxrecon/out/**` (= `evaluation/volumes/**`) is therefore measured on **superseded
data** and must be re-run after the rebuild. Concretely that is:

- the val panels — `panel_Val_*`, `val_grid_t00/t06`, `Val_P048_basal_planes` — and the CorSeg↔nnU-Net
  agreement figures on that cohort (mean 0.901 at ED, EF 58.8% vs 62.1%);
- **the SVR heart-ROI experiment** — `svrtk3d`/`nesvor` `recon_clean/vol_tXX.nii.gz` come from the
  same bundle, so the exact counts (8/30 NeSVoR blank, median 269 vs 185,512 voxels, agreement 0.000)
  are all on deprecated inputs.

**Which conclusions survive the rebuild, and why:**

| finding | survives? | reasoning |
|---|---|---|
| ACDC accuracy (Dice 0.889 vs 0.896/0.902; EF MAE 2.51 vs 4.67 pp; ROI 0.413 vs 0.890) | **yes** | ACDC is a different dataset, untouched by CMRxRecon recon versions |
| SVR arms emit small ~1.4 mm isotropic heart-ROI volumes, not the canonical cube | **yes** | a property of the SVR *pipeline geometry*, not of the recon version |
| CorSeg degrades on that grid (mid-band only / blanks) | **yes, direction** | driven by 224²-canvas occupancy (0.15–0.23), which the grid sets; independently reproduced on ACDC |
| the exact SVR counts and agreement numbers | **NO — must re-measure** | image content changes (v2 ESPIRiT + roll fix), so per-case outcomes will shift |
| paper-prep > GUI-prep on the canonical grid (0.901 vs 0.882) | **direction only** | same volume for both, but re-measure the magnitude |

Also note the CMRx figures were only ever **qualitative** (no human GT on that cohort), and in each
one both segmenters consumed the **identical** volume — so staleness never biased CorSeg *against*
nnU-Net or vice versa.

One asymmetry the staleness itself exposed, visible in `val_grid_t00.png`: on `Val_P054` z=9 — the
misplaced basal slice — **CorSeg segments an LV that nnU-Net ignores**, which would inflate CorSeg's
LV volume on rolled data. That specific effect should *disappear* after the rebuild; if it does not,
it was not the roll.

## Is nnU-Net itself actually fine on the SVR heart-ROI volumes? Partly — and unverifiable

nnU-Net was **also never trained on this** — Task114 saw full-FOV M&Ms SAX cine, never 1.4 mm
isotropic SVR reconstructions, heart-ROI crops, or NeSVoR's masked support (its non-zero region is a
**single connected blob covering 32.6%** — a genuine mask; SVRTK by contrast is 99.87% non-zero, i.e.
*cropped but not masked*). So both models are out of distribution here. The difference is
**graceful vs catastrophic degradation**, not in-domain vs out-of-domain.

**Where nnU-Net is genuinely solid — ACDC, with human GT.** It never fails badly:

| condition | mean | median | min | p5 | cases <0.80 | <0.70 |
|---|---|---|---|---|---|---|
| full FOV | 0.896 | 0.904 | **0.809** | 0.834 | **0** | 0 |
| ROI-cropped | 0.890 | 0.901 | **0.749** | 0.817 | 4 | **0** |

No catastrophic case in 200 evaluations; the worst case is still a usable segmentation.

**Where it is degraded — the SVR arms, no GT available.** Two GT-free probes, restricted to the
**middle 50% of each segmented z-extent** so the base/apex slice mix cannot confound the comparison
(SVR stacks carry ~75 thin slices per subject vs ~8 canonical):

| input | fragmented slices | containment violations |
|---|---|---|
| canonical GT (in-domain) | **0.0%** | **0.0%** |
| SVRTK heart-ROI | **22.8%** | 8.5% |
| NeSVoR heart-ROI | 3.0% | 0.2% |

and a physical cross-check — LV cavity volume at t=0, SVR arm vs the canonical GT arm, same subject:

| arm | mean LV | bias vs GT arm | MAE | r |
|---|---|---|---|---|
| GT (canonical) | 149.0 mL | — | — | — |
| SVRTK | 181.4 mL | **+32.4 mL (+21.7%)** | 32.6 | 0.882 |
| NeSVoR | 172.1 mL | **+23.1 mL (+15.5%)** | 23.1 | 0.966 |

### Robustness audit: nnU-Net never went blank or unusable in 260 volumes

Checked explicitly, because "never fails" is the kind of claim that hides in a mean.

**Volume-level blanks — 0 out of 260**, across every nnU-Net run in this study:

| run | n | blanks | min foreground voxels | p1 |
|---|---|---|---|---|
| ACDC full FOV | 100 | **0** | 6,532 | 7,998 |
| ACDC ROI-cropped | 100 | **0** | 5,179 | 6,991 |
| SVR heart-ROI | 60 | **0** | 79,773 | 86,027 |

**Per-structure collapse on ACDC — none.** A healthy mean can hide one dead structure, so per label:

| condition | LV cav min | myo min | RV min | #structures <0.5 | #<0.3 | structures entirely missed |
|---|---|---|---|---|---|---|
| full FOV | 0.692 | 0.767 | 0.655 | **0** | **0** | **0** |
| ROI-cropped | 0.604 | 0.716 | 0.669 | **0** | **0** | **0** |

Worst *single structure* across 600 structure-evaluations is **0.604** — degraded but usable, never
absent. Compare CorSeg on ROI crops: 23/95 volumes entirely blank.

**Why the difference is structural, not luck.** nnU-Net v1 resamples to its plans' target spacing
from the NIfTI header and runs a **patch-based sliding window**, so the heart is presented at trained
scale regardless of image size. CorSeg squeezes the whole slice into a fixed 224² canvas, so framing
and scale are set by the input FOV — which is exactly what the occupancy correlation (r=0.625)
measures. This is a design difference that is consistent with every observation here; the causal
claim is supported for CorSeg (measured) and inferred for nnU-Net (not separately ablated).

**Scope — what this does NOT cover.** The "never unusably bad" claim is verified only where human GT
exists (**ACDC**, 200 volumes). On the 60 SVR volumes I verified **non-blank + anatomical
plausibility only** — and the 22.8% fragment rate shows it *is* degraded there, just not broken.
Untested entirely: the VGGT recon arms, breathing-corrupted arms, and other cohorts (OCMR, MIITT,
M&Ms, Göttingen). Nothing here makes a blank output architecturally impossible — argmax over an
all-background prediction is a valid outcome — only that it did not occur in 260 volumes.

**What the SVR-arm probes do and do not establish.** They establish that nnU-Net stays *functional*
(0 blanks, full-stack coverage) but is **not pristine** on the SVR grids — one in five mid-ventricular SVRTK
slices carries a fragmented label where the in-domain rate is zero. It does **not** establish accuracy:
there is no human GT on this cohort, and the +15–22% LV offset confounds three things that cannot be
separated here — (1) segmentation error on OOD input, (2) genuine differences between the SVR
reconstruction and the GT volume, and (3) **partial-volume**, since the canonical grid's 12 mm slices
average through the cavity while the SVR grid resolves it at 1.4 mm, so the GT arm may legitimately
*under*-read. **Do not quote the +21.7% as a segmentation error.**

**Mitigation worth testing (not measured):** EF is a *ratio*, so a volume bias that is consistent
between ED and ES largely cancels. Only t=0 was run here. Measuring the same bias at ES would show
whether the SVR arms' EF is trustworthy despite the volume offset — **the single highest-value check
to run after the `scratch/eval` rebuild**, and cheap (one extra phase).

⚠️ All SVR-arm numbers in this section are on the **deprecated** bundle and are provisional.

## Decision

### ⚠️ Revised: use nnU-Net for EVERYTHING, plus CorSeg's post-processing. Do not finetune.

**An earlier version of this section recommended splitting by metric — nnU-Net for Dice, CorSeg for
EF. That was wrong for our actual use and is superseded.** The decisive point it missed:

> **A method comparison requires the SAME segmenter on every arm.** If the VGGT arms were scored
> with CorSeg and the SVR arms with nnU-Net, any EF gap between VGGT and SVRTK/NeSVoR would be
> partly a *segmenter* difference, not a reconstruction difference. And CorSeg **cannot** run on the
> SVR arms (8/30 blank, mid-band only). Only nnU-Net runs on all of them, so only nnU-Net can
> produce a valid cross-method table.

CorSeg's EF advantage (2.51 vs 4.67 pp MAE) remains real, but it is usable only for a *different*
claim: **absolute** EF accuracy against clinical truth on full-FOV data. It cannot be used to rank
reconstruction methods against each other.

**The free upgrade — CorSeg's post-processing on nnU-Net's output.** The three anatomical steps are
already extracted and verified in `tools/corseg/corseg_postproc.py` and require no training:

| input | slices with fragment/containment violation | Dice effect |
|---|---|---|
| ACDC full FOV (in-domain) | 5.6% → **3.0%** | mean3 0.8961 → 0.8960 (**neutral**, ≤0.0005 per structure) |
| SVRTK heart-ROI (the degraded arm) | 24.7% → **6.0%** (**4× fewer**) | — |

So it removes ~4/5 of the visible anatomical implausibility on exactly the arm where nnU-Net was
weakest, at zero Dice cost and zero training cost. **Recommended: adopt it.**

**Do NOT finetune either model.** Reasons, in order of weight:

1. **The segmenter is a measurement instrument, not a contribution.** For a reconstruction paper, an
   off-the-shelf, published, citable segmenter is far more defensible than one we tuned ourselves —
   a tuned metric invites exactly the criticism you cannot answer.
2. **We have no ground truth to finetune on** for the target domain. There are no human labels on
   SVR-reconstructed heart-ROI volumes. Training on nnU-Net's own predictions is circular: it
   distils nnU-Net, inherits its −3.4%/+2.8%/−4.5% volume bias, and caps at its accuracy.
3. **The residual error is largely a shared bias.** Applied identically to every arm, a common
   offset mostly cancels in the *relative* comparisons that are the headline claims.
4. **The cheap fix already captures most of the gain** (row above).

**When finetuning WOULD be justified:** if the claim became *absolute* clinical volume/EF accuracy on
the SVR arms rather than relative ranking, **and** real human GT on that domain existed. (Update
2026-07-30: expert labels on **full-FOV cine**, ~30–150 subjects, DO exist — so a finetune is now
well-founded; see the risk below.)

### ⚠️ The finetune puts CorSeg's own headline advantage at risk

Non-obvious and worth stating explicitly: **CorSeg's EF win comes from calibration, not from sharper
boundaries.** Its Dice is *worse* than nnU-Net's (0.889 vs 0.896) while its volumes are near-unbiased
(+1.3% / +0.6% n.s. / +1.4% n.s.) — i.e. its boundary sits, on average, exactly where the ACDC
annotators put it, whereas nnU-Net's is consistently offset.

That calibration is a property of **whose labels it was trained on**. Finetuning on a different
annotator's conventions — e.g. whether papillary muscles and trabeculae go in the cavity or the
myocardium, which the CorSeg paper itself names as a known source of disagreement — will
**re-calibrate the boundary to that annotator**. The 2.51 pp EF MAE was measured with the *original*
weights against ACDC GT; it is **not guaranteed to survive**, and could move in either direction.

**Therefore the finetune gate must re-measure EF, not assume it.** Required checks, on data untouched
by training:

| gate | requirement | why |
|---|---|---|
| ACDC full-FOV Dice | ≥ 0.889 | no catastrophic forgetting (external, independent of the expert set) |
| ACDC ROI-crop Dice | 0.413 → ≥ 0.85 | the actual objective |
| blanks (ACDC ROI + real SVR volumes) | 0 | the disqualifier must be gone |
| **EF MAE vs ACDC GT** | **≤ 4.67 pp, ideally ~2.51** | **the advantage being bought — re-verify, do not assume** |

If EF MAE degrades past nnU-Net's 4.67 pp, the finetune has bought robustness at the cost of the
only thing CorSeg was better at, and the honest conclusion is to stay on nnU-Net.

**Against a wholesale switch to CorSeg:** it re-baselines every number in `evaluation/results/`, and
it cannot score the SVR arms at all.

CorSeg remains the better model in two respects, relevant only on full-FOV data:

1. **Single model, in-env, 8.5× cheaper per volume.** No `nnunet` env hop, no `RESULTS_FOLDER`
   plumbing, 1 forward pass vs 5 (2d) or 10 (ensemble). **Measured** against **nnU-Net-2d** — the
   config `run_seg.sh` actually runs — by timing 1 vs 12 volumes to separate fixed from marginal cost
   (canonical `(256,256,12)` volumes, A40, checkpoint pre-staged to `/tmp` for both):

   | | fixed startup | marginal per volume | wall @ 12 volumes |
   |---|---|---|---|
   | CorSeg | ~13.5 s | **0.165 s** | **15.5 s** |
   | nnU-Net 2d (5 folds) | ~34.1 s | **1.40 s** | **50.9 s** |

   So **3.3×** on a single 12-phase subject (startup-dominated) but **8.5× per volume** once
   amortized — the number that matters for a real sweep. A full cohort pass (~30 subjects × 12
   phases × 3 arms ≈ 1080 volumes) is ~3 min vs ~25 min. The ensemble is dearer still: it adds a
   3d_fullres pass on top of the five 2d folds.

   **Caveat — most of that gap is the ensemble, not the architecture.** `nnUNet_predict -m 2d`
   defaults to **all 5 folds**; CorSeg is 1 model. Measured on the same 12 volumes end-to-end:
   **CorSeg 15.5 s / nnU-Net 1 fold (`-f 0`) 32.0 s / nnU-Net 5 folds 50.9 s** — so CorSeg is 3.3×
   faster than the config we run but only **~2× faster than a single fold**. (The single-fold
   *marginal* per-volume cost was not separately measured; only the end-to-end total.) Quote 8.5×
   only against the 5-fold default, not as an architectural claim.
2. **Trained on all ~25 cardiac phases**, whereas Task114 saw only ED/ES. Our pipeline evaluates 12
   target phases, so nnU-Net is out of its label distribution for 10 of them. This is the strongest
   argument for CorSeg — and it is **NOT measured here**, because no cohort we have carries
   mid-cycle human GT. Testing it would need either mid-cycle annotations or an indirect proxy
   (e.g. temporal smoothness of the LV-volume curve, or inter-model agreement stratified by phase).

**If it is ever adopted**, hard requirements: use `--mode paper` (never the GUI path), feed
**full-FOV** volumes only, ensure NIfTI headers carry true spacing, and remember the **1↔2 label
swap** against Task114.

## Is nnU-Net-on-OOD-input defensible in the paper? Yes, if scoped to RELATIVE claims

The concern a reviewer will raise: *the segmenter was never trained on your reconstructed volumes.*
Three defences, all backed by numbers measured here rather than asserted:

1. **It is standard practice.** Downstream-task evaluation with a pretrained segmenter is the norm in
   the reconstruction literature; the segmenter is always OOD w.r.t. its training set. Training one
   on your own reconstruction outputs would be circular.
2. **The metric ranks methods, and the same segmenter is applied to every arm**, so its bias is
   **common-mode** and largely cancels in relative comparisons. This is the reason to prefer *one*
   segmenter everywhere over the *best* segmenter per arm — and it is why the earlier
   "CorSeg for EF, nnU-Net for Dice" split was wrong.
3. **Robustness is measured**: 0 blanks in **260 volumes**; worst single-structure Dice **0.604** in
   600 structure-evaluations with no structure ever missed; and documented cross-domain
   generalisation (zero-shot **0.896–0.902 on ACDC** vs human GT, on par with ACDC-trained methods).

**Hard limit — do NOT make absolute clinical claims from the SVR arms.** The +15–22% LV offset vs the
canonical GT arm confounds segmentation error, genuine recon differences, and partial-volume
(12 mm vs 1.4 mm slices) and cannot be attributed to any one of them. Relative comparisons only.

**Report the degradation, don't hide it.** A reviewer who finds the 22.8% fragment rate unaided is a
far worse outcome than disclosing it — especially since post-processing mitigates it to 6.0%, so the
number you report can be the *mitigated* one. Suggested wording:

> Segmentation used the pretrained M&Ms nnU-Net (Task114), applied identically to all reconstruction
> arms; it is out-of-distribution for all arms equally, so its bias is common-mode and does not
> affect relative comparisons. It produced non-empty, anatomically plausible segmentations for all
> N volumes; on the SVR arms, whose heart-ROI grids differ most from the training domain, X% of
> slices required anatomical post-processing.

**A second-segmenter sensitivity check is NOT available — tested and refuted.** If the method
*ranking* were unchanged under an independent segmenter, the conclusion would be segmenter-independent.
CorSeg cannot read the raw SVR ROI grids, so I tested the obvious workaround: the **assembled**
`cine_clean.nii.gz`, which *is* on the full canonical 358 mm grid (frame occupancy 1.000). Hypothesis
was that putting the heart at correct scale and position inside a proper 280 mm window would work.

**It made things worse.** CorSeg on the assembled grid: **SVRTK 23/30 blank** (vs **0/30** on the raw
ROI grid), NeSVoR 7/30 blank. The assembled volumes are only **3.8–4.6% non-zero** — the heart is a
tiny island in a sea of zeros. So CorSeg cannot score the SVR arms in *either* representation, and
the two-segmenter sensitivity check is unavailable.

**Correction — my occupancy mechanism was over-generalised.** The r=0.625 occupancy↔Dice correlation
is real and solid *within* the ACDC ROI-crop condition (n=100, p=3.7e-12), but it does **not**
generalise across representations. Neither frame occupancy nor an anatomy-weighted version predicts
outcome across all six conditions tested (anatomy-fraction vs blank rate: r=−0.526, **p=0.284**, n=6):

| condition | frame occupancy | non-zero | anatomy fraction | CorSeg blank rate |
|---|---|---|---|---|
| ACDC full FOV | 1.000 | 1.000 | 1.000 | **0%** |
| ACDC ROI crop | 0.166 | 1.000 | 0.166 | 30% |
| SVR raw ROI (SVRTK) | 0.150 | 0.998 | 0.150 | **0%** |
| SVR raw ROI (NeSVoR) | 0.234 | 0.341 | 0.080 | 27% |
| SVR assembled (NeSVoR) | 1.000 | 0.046 | 0.046 | 23% |
| SVR assembled (SVRTK) | 1.000 | 0.038 | 0.038 | **77%** |

SVRTK-raw (0.150) gives 0% blanks while ACDC-ROI (0.166) gives 30% — a *higher* anatomy fraction
doing worse. **The defensible statement is the empirical one: CorSeg is reliable only on genuinely
full-FOV input, and erratic below that.** Implication for any finetune: augment over the *actual*
target representations, not a synthetic occupancy sweep, since no single scalar parameterises the
failure.

**nnU-Net control on the same sparse grid: 0 blanks in all 60** (SVRTK min 5,617 / NeSVoR min 12,387
voxels), i.e. it survives a representation where CorSeg is 77% blank. That brings the standing
robustness tally to **0 blanks in 320 volumes across four distinct representations**:

| representation | n | nnU-Net blanks | CorSeg blanks |
|---|---|---|---|
| ACDC full FOV | 100 | **0** | 0 |
| ACDC ROI-cropped | 100 | **0** | 23 |
| SVR raw heart-ROI (1.4 mm iso) | 60 | **0** | 8 |
| SVR assembled canonical (~4% non-zero) | 60 | **0** | 30 |
| **total** | **320** | **0** | **61** |

This is the single most citable robustness fact for the paper's OOD concern.

**Sensitivity check that IS available:** perturb nnU-Net itself — single fold vs 5-fold vs
2d+3d ensemble — and show the method *ranking* is unchanged. Weaker than an independent architecture,
but honest, cheap, and it directly addresses "is the conclusion an artefact of the segmenter?".

## Artifacts

- Code: `tools/corseg/{corseg_infer.py,corseg_postproc.py,bench_acdc.py,render_corseg_panels.py}` + `README.md`
- Upstream: `CorSeg/` (clone) + `CorSeg/corseg.pdf`
- Weights: `scratch/data/corseg/ModelWeight-CorSeg-CineSAX_MedNextL.pth` (741 MB, extracted from
  `scratch/CorSeg-ModelWeights.zip`; the zip's other 6 GB is two Windows `.exe` bundles)
- Figures: `result/corseg/`
- Results JSON: `/tmp/corseg_acdc/res_*.json` (scratch — rerun `bench_acdc.py` to regenerate)
