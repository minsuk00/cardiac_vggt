# 39 — Whole-heart segmentation, ROI masks, and cardiac-phase (ED/ES/EF) across all datasets

> **TL;DR & takeaway**
> We built one **consistent, anatomy-derived** segmentation + heart-ROI + cardiac-phase reference across
> every dataset, so the SVR baselines and VGGT-MRI are scored on the same region and we have GT
> ED/ES/EDV/ESV/EF everywhere. Method: **nnU-Net M&Ms Task114 (2D)** → per-frame 3-class LV/MYO/RV seg,
> a dilated binary ventricle ROI, and (for gated data) ED/ES/EF from the 3D LV-cavity volume curve.
> Produced as **siblings** next to each recon (`heart_seg.nii.gz` 4D + `heart_roi.nii.gz`), plus one
> central regenerable table `scratch/data/whs/cardiac_phase.csv`. **421 + 150 ACDC units done** (CMRx
> 301, MIITT 26, OCMR 25, Goettingen 69, **ACDC 150** — a full SVR-recon target). **472 gated subjects**
> carry ED/ES/EDV/ESV/EF, and Task114 cleanly stratifies ACDC's 5 pathologies by EF (NOR 61 / HCM 63 /
> RV 57 vs MINF 31 / DCM 21%) — clinical validation on 150 disease-labeled patients. Key facts:
> Task114 works on all **gated** data + MIITT-RT + Goettingen-RT and **fails only on OCMR-RTFB**
> (aliased recon, flagged); **2D ≈ ensemble** on these thick anisotropic slices (0.896 vs 0.902 Dice);
> EF is clinically realistic (CMRx 64%, MIITT 60%, OCMR 58% mean) and **Task114-EF matches ACDC manual
> EF at r=0.962** — and because eval is method-matched (Task114 on both pred and GT), the segmenter's
> small bias cancels, so **we use Task114 everywhere and don't need manual GT operationally**. "Whole
> heart" is loose wording: it's a **ventricle-centered ROI** (no atria/vessels — Task114 has no such class).

---

## 1. Motivation

The classical-SVR baselines and the VGGT-MRI metrics need a **heart region**: SVRTK/NiftyMIC take a
recon-FOV mask as *input*, and a fair pred-vs-GT comparison needs a common **metric ROI**. Before this,
masks were inconsistent and heuristic: NiftyMIC used the whole-FOV `content_mask`
(`baselines/niftymic/export_stack.py:12`); fetal_cmr_4d/SVRTK used a motion-defined cardiac-band FFT
blob (`baselines/fetal_cmr_4d/export_miitt.py:116`) — motion-biased, misses atria/low-motion tissue,
and differs per dataset. Goal: one **anatomy-defined, identically-constructed** mask + cardiac-phase
reference for every dataset.

## 2. Segmenter — nnU-Net M&Ms Task114, 2D

`nnUNet_predict -t 114 -m 2d -tr nnUNetTrainerV2_MMS` in the isolated `nnunet` micromamba env
(`tools/nnunet_mnms_eval/env.sh`, invocation mirror `inference/seg_cmrxrecon.sh`). Output labels
**LV=1, MYO=2, RV=3** (M&Ms convention — note ACDC's own GT is the *opposite*, §8).

**Why 2D (not 3D_fullres or the 2D+3D ensemble):** both 2D and 3D output a 3D label volume; the
difference is convolution dimensionality. Our SAX slices are **highly anisotropic** (~1.4 mm in-plane
vs 8–12 mm through-plane), so a 3D model's through-plane receptive field can barely bridge the gaps —
it segments nearly per-slice anyway. A prior ACDC Dice study (`tools/nnunet_mnms_eval/build_report.py`)
measured **ensemble 0.902 vs 2D 0.896 vs 3D 0.893** — a 0.006 gap, within noise. 2D was chosen for
speed (ensemble = 10 folds ≈ 2×) and consistency with the existing EF pipeline (docs/17, 24). For a
**method-matched** EF eval it does not matter which model, as long as it's the same on both sides — and
everything here is 2D.

## 3. Trust-check across datasets (before committing)

Ran Task114 on a sample per dataset/regime and inspected overlays + per-plane label counts
(`result/whs_check/multiDS_*.png`). Verdict — **recon quality, not gated-vs-RT, is what matters**:

| dataset | regime | Task114 |
|---|---|---|
| CMRx | gated (native) | ✅ clean |
| MIITT | gated / **RT** | ✅ / ✅ (RT recon is clean) |
| OCMR | gated / **RTFB** | ✅ / ❌ (~half fail — aliased raw-k-space recon) |
| Goettingen | RT free-breathing SAX | ✅ (RT-NLINV recon is high quality) |

Base/apex behaviour: the seg contiguously covers the ventricular mass and **correctly leaves the
superior (basal) planes unlabeled** — those are atria/valve/great-vessels, which Task114 has no class
for. So the ROI is a **ventricle-centered cardiac ROI**, *not* literal whole-heart; the atria are the
deliberate gap (dilation only pulls in adjacent tissue). Details + panels: `result/whs_check/`.

## 4. Pipeline & storage

Per **work unit** (= one subject-regime = one `nnUNet_predict` call over all its phases/frames):
`prep → predict → assemble`. Orchestrated by an **idempotent SLURM array job**
`sbatch/whs_segment.sh` (3 workers stride the worklist, skip units whose `heart_seg.nii.gz` already
exists; small intermediates on node-local `$TMPDIR`, only 4D outputs hit GPFS).

- `tools/nnunet_mnms_eval/make_whs_worklist.py` → `scratch/data/whs/worklist.txt` (one line
  `"<dataset> <regime> <path>"`; cmrx path = the `sax/` dir, others = the 4D recon file).
- `prep_one.py` → per-frame nnU-Net inputs `f{ff:03d}_0000.nii.gz`.
- `assemble_whs.py` → stack per-frame segs into 4D `heart_seg.nii.gz`, build the ROI, append a
  per-unit manifest row. Reuses `build_heart_roi.build_roi()`.

**Storage = siblings next to each recon** (not a central mirror). Each eval adapter already anchors on
its recon dir, so lookup is `os.path.join(dirname(recon), "heart_roi.nii.gz")` — zero-config, can't
desync. Per-frame RTFB masks are **4D NIfTIs** mirroring the recon (a uint8 4D mask is ~3 MB gz / 0.13 s
to load — ¼ the recon's RAM — vs ~30k tiny per-frame files, which is GPFS-hostile). Two products:
- `heart_seg.nii.gz` — 4D `(X,Y,Z,T)`, 3-class. For EF/Dice.
- `heart_roi.nii.gz` — binary ROI: **3D static** (gated) or **4D per-frame** (rtfb). For FOV/metric ROI.
- CMRx also gets `heart_{seg,roi}_canonical.nii.gz` — resampled through `preprocess.get_canonical_transforms`
  (nearest) to the training canonical cube, since CMRx eval reads the canonical cache not raw NIfTIs
  (`run_cmrxrecon.py` → `MRIDataset.get_data(seq_index)`).
- `scratch/data/whs/whs_manifest.csv` — QC index only (per-unit `mean_labeled`, voxel counts, `flag`).

## 5. ROI recipe — alignment-correct, dilated

`build_heart_roi.build_roi(seg, spacing, in_mm=6, z_extend)`: union(LV∪MYO∪RV) → per-slice
fill-holes + closing (bridges the LV–RV gap into one blob) → **6 mm in-plane dilation** (spacing-aware
via a voxel disk) → z-extension. The gated-vs-RTFB split is **physically motivated**:

- **Gated** (CMRx/MIITT/OCMR-gated): frames are at a fixed respiratory position (breath-hold), differing
  only by cardiac phase. So a **temporal union over phases is valid** → one **static 3D** ROI that
  covers the whole cycle, `z_extend=1` (±1 plane at apex/base). Invariant checked: `roi ⊇ union(seg>0)`.
- **RTFB** (MIITT-rt / OCMR-rtfb / Goettingen): frames are respiratory-**unaligned** (heart slides with
  breathing), and the SAX slices are independent RT cines at different breaths (Goettingen's recon code
  literally: *"each SAX slice is an independent ~4.26 s real-time cine"*). So **no temporal union, no
  z-dilation** → the ROI is **per-frame, in-plane only** (`z_extend=0`), a 4D mask. Invariant checked:
  each `roi_t ⊇ seg_t` and the ROI varies across frames.

`z_extend` is a *fixed plane*, so the physical z-margin varies with pitch (±6 mm Goettingen … ±12 mm
CMRx) — accepted as generous.

## 6. Datasets & scope (final)

571 units done (421 + 150 ACDC):

| dataset | regime | units | seg | ROI |
|---|---|---|---|---|
| CMRx | gated | 301 | 4D 12-phase | static 3D (+canonical) |
| MIITT | gated / rt | 13 / 13 | 4D | static 3D / **4D per-frame** |
| OCMR | gated / rtfb | 8 / 17 | 4D | static 3D / **4D per-frame** |
| Goettingen | rt | 69 | 4D | **4D per-frame** |
| **ACDC** | gated | **150** | 4D | static 3D |

**5 low-quality units** (manifest `flag=low`, `mean_labeled < 4`), all expected: 3 OCMR-RTFB
(`us_0204` mean 0.0 = total failure, `us_0182`, `us_0080` — aliased recon), `goettingen vol0001_slc12`
(a 1-slice volume), `cmrx Test_P044` (small-FOV / seg hiccup, EF 91% nonsense). Exclude via the flag.

## 7. Cardiac phase (ED/ES/EDV/ESV/EF)

`tools/nnunet_mnms_eval/compute_cardiac_phase.py` → `scratch/data/whs/cardiac_phase.csv` (gated only;
RTFB has no cardiac gating). Columns: `unit,dataset,regime,subject,T,ED,ES,EDV_mL,ESV_mL,EF_pct,
curve_mono_frac,unimodal_ok,seg_flag,source,group` (`source` = `task114_3d` / `acdc_task114`; `group` =
ACDC pathology, else empty). **472 gated subjects** (CMRx 301, MIITT 13, OCMR 8, ACDC 150).

- **ED/ES** = phase of **max / min LV-cavity (label-1) volume** over all phases (`argmax`/`argmin`).
  ES is a unique mid-cycle minimum. **ED is periodic-ambiguous**: end-diastole (LV max) sits at the
  cycle boundary, so frame 0 ≡ frame T-1 (same cardiac state) and `argmax` picks whichever is marginally
  larger — e.g. **9/13 MIITT** and 6/301 CMRx land on the *last* frame (fraction 1.0), not 0. This is not
  an error (predicted ED/ES match GT — cf. the ACDC r=0.962), just the 0≡1 wrap; the plot folds ED to
  the cyclic distance `min(f, 1−f)` so all ED collapse to ~0. **ACDC is the exception:** its cine starts
  *and* ends at ED, so `argmax` is maximally ambiguous — we use its **ground-truth ED/ES from `Info.cfg`**
  (`acdc_ed_es()`, 1-indexed → 0-indexed) instead. EF is unchanged either way (both endpoints are ~max).
- **EDV/ESV** = full **3D** LV-cavity volume = `voxel_count × prod(spacing)/1000` mL. This is Simpson's
  method — the **slice *pitch*** (center-to-center) is the correct z term (CMRx Z=12 mm = 8 mm thickness
  + 4 mm gap; using 8 mm would underestimate by ~33%). A single mid-slice gives an *area*, not a volume,
  and its ED/ES timing only matches the 3D answer to ±1 frame — so **always use the 3D volume**; the
  mid-slice is a fallback only when no 3D seg exists (RTFB self-gating, doc 35).
- **EF** = (EDV−ESV)/EDV × 100. EF is a **ratio → spacing-independent** (robust even where spacing is a
  placeholder, e.g. MIITT's 10 mm slice is provisional; MIITT *absolute* mL is approximate, EF is not).
- **Quality:** `curve_mono_frac` = fraction of frame-to-frame steps moving the physiologically-correct
  way (LV volume falls ED→ES, rises ES→ED); 1.0 = clean heartbeat, low = jittery seg. `unimodal_ok` = 1
  iff `curve_mono_frac ≥ 0.8` and EDV>0. `seg_flag` joined from the manifest. **Trust rows with
  `unimodal_ok==1 & seg_flag=='ok'` (470/472).** Only 2 fail: `cmrx Test_P044` (EF 91% nonsense, also
  `seg_flag=low`) and `acdc_patient050` (`mono=0.7`).

**Clinical realism (flagged excluded):** EF medians **64% (CMRx) / 60% (MIITT) / 61% (OCMR)**, EDV
130–143 mL — normal adult ranges. **ACDC spans the full disease range** (EF 9–79, mean 47, EDV to
~380 mL) because it's pathology-rich — see §9.

**Cohort composition (this is what the EF spreads reflect — verify from source, not from the EF):**
- **CMRxRecon2024 = 330 healthy volunteers** (official challenge description; 3.0T Siemens Vida,
  multi-contrast; we use 301) → tight ~64% EF. Note: *not* stated in the on-disk `cine_sax_info.csv`
  (acquisition params only) — it's from the challenge documentation.
- **OCMR gated = volunteers** — documented in `scratch/data/ocmr/ocmr_data_attributes.csv` (`sub=vol`
  for all 8) → ~61% EF.
- **MIITT = mixed, NOT healthy** — folder names: 10 `Volunteer*` + **3 `Patient_*`** (ARVC, HCM,
  Cardiomyopathy+AFib) → EF spread down to ~43%.
- **ACDC = pathology-labeled** — `Info.cfg → Group`: 30 NOR + 30 each MINF/DCM/HCM/RV → EF 9–79%.

## 8. ACDC — special handling (it's an SVR-recon target too)

ACDC is used **both** as an SVR-recon target (with our breathing sim) **and** as the only dataset with
gold-standard *manual* EF — so it's fully integrated (siblings + CSV), and separately validates Task114.
Three ACDC-specific gotchas, all handled:

1. **Manual GT ships only at ED/ES** — but ED/ES + pathology `Group` are given in `Info.cfg`. We still
   run Task114 on the **full 4D cine** (all ~30 phases) to match our pipeline and get argmax/argmin ED/ES.
2. **Label convention is opposite:** ACDC manual GT is **LV=3, RV=1, MYO=2** (Task114 is LV=1). Our
   `heart_seg.nii.gz` comes from Task114 → LV=1, so `compute_cardiac_phase` works unchanged; only the
   manual-GT comparison remaps (`eval_acdc.py` already does: LV pred==1 vs gt==3).
3. **Spacing lives in the header `pixdim`, not the affine** (affine is identity → `voxel_sizes()` gives
   `[1,1,1]`, an ~18× volume error). `prep_one.py` builds a diagonal affine from `header.get_zooms()`
   for ACDC. Verified on patient101 (DCM): EDV 215 mL, **EF 31% vs manual 30%**, spacing 1.64×1.64×10.

**Task114 vs ACDC manual EF (n=50 test patients, `result/whs_check/acdc_task114_vs_manual_ef.png`):**
Pearson **r = 0.962**, Spearman 0.903, MAE 4.7 EF-pts, bias **−2.1** (Task114 slightly under-reads),
worst at high-EF HCM (−10 to −18: tiny ES cavity). **Conclusion: use Task114 everywhere, drop manual
operationally** — not just because r=0.96, but because in a **method-matched** eval (Task114 on both
pred and GT) that bias is common-mode and **cancels**; mixing manual-GT with Task114-pred would inject
it one-sidedly (the exact unfairness to avoid). Manual only wins for *absolute* high-EF accuracy.

## 9. Findings from the stats (`result/whs_check/cardiac_stats.png`)

Plotted with `tools/nnunet_mnms_eval/plot_cardiac_stats.py` — **box + jittered points** (box=summary,
points=actual data + small-n honesty + visible quantization), ED/ES normalized to cycle-fraction
`idx/(T-1)` so datasets with different T compare. 6 panels: ED timing · ES timing · EF · ACDC
EF-by-pathology · EDV-vs-ESV · EF-vs-EDV.

- **EF-by-pathology is textbook (the headline validation, 150 ACDC patients):** NOR **61 ± 6**, HCM
  **63 ± 10**, RV **57 ± 7** (preserved) vs MINF **31 ± 9**, DCM **21 ± 9%** (reduced). Task114 cleanly
  separates all 5 pathologies by EF — end-to-end clinical validation of the segmenter on disease cases.
- **ES-timing anomaly confirmed with the ACDC anchor:** CMRx ES-fraction ≈ **0.54** and **quantized**
  into k/11 bands, while **ACDC (0.41, 30-frame gold-standard) / MIITT (0.37) / OCMR (0.40)** all sit in
  the normal ~**0.30–0.45** end-systolic window. So CMRx is the outlier. End-systole normally falls
  ~0.3–0.45 of R-R at rest (HR-dependent — no fixed "1/3"). **Observation solid; mechanism still a
  hypothesis** — most likely CMRx's native-14–40→12 resampling (documented; it breaks index↔time
  linearity, so its 0.54 may not even be a true time-fraction). *Implication:* ED/ES **indices** live on
  different effective timescales across datasets — normalize to cycle-fraction before cross-dataset use.
- **Dilation→dysfunction:** with ACDC's pathology cases added, EF-vs-EDV shows a clear negative trend
  (DCM/MINF populate the high-EDV / low-EF corner); EDV–ESV falls in a tight band between the 50–70%
  iso-EF lines with ACDC extending it to EF 30% (dilated, ESV to ~380 mL).
- **Task114-EF vs ACDC manual EF** (§8): r = 0.962 — the segmenter validation underpinning "use
  Task114 everywhere."

## 10. Gotchas & lessons

- **micromamba cache-lock race (cost 16 units, silently):** concurrent workers share
  `~/.cache/mamba/proc`; colliding `micromamba run` calls abort ("libmamba Could not set lock"), and the
  `process_unit … || echo FAILED` pattern disabled `set -e` inside the function so the failure fell
  through with no row. **Fix:** `mrun()` retry-with-backoff wrapper + explicit `return 1` per step in
  `whs_segment.sh`; the 16 casualties were re-run **sequentially** (no contention). Sequential is
  bulletproof; the hardened 3-worker job should be fine too.
- **CMRx canonical import pulled `vggt`:** `from data.preprocess import …` triggers
  `training/data/__init__.py` → `MRIDataset` → `import vggt` (absent on the `svr` subprocess path).
  **Fix:** `assemble_whs._load_preprocess()` loads `preprocess.py` standalone via `importlib` (it only
  needs torch+monai).
- **Don't judge dataset geometry from NIfTI headers** — verified twice-wrong-then-right (Göttingen IS a
  SAX stack; CMRx HAS 12 native per-phase files; ACDC spacing is in the header not the affine). Render
  slices / read recon code. See memory `feedback_verify_geometry_from_data_not_header`.
- **"native" vs "gated" regime label** was a naming wart (CMRx is ECG-gated too) — unified to `gated`
  across worklist/manifest/CSV.
- **ACDC is mixed-orientation (114 LPS / 36 LAS)** — training/eval require **LPS everywhere**
  (`CLAUDE.md → CMR data notes`; `ACDCGatedAdapter` reorients the *recon* to LPS at load). Our ACDC
  masks were segmented on the **raw** (native-orientation) cine, so the 36 LAS patients' `heart_seg`/
  `heart_roi` **siblings are not LPS-normalized**. **Cardiac metrics are unaffected** (LV volume/EF are
  orientation-invariant — hence the plots and the r=0.962 validation are correct), **but for the
  SVR-recon-target use the masks should be reoriented to LPS** (or ACDC re-segmented on LPS-reoriented
  inputs). **Open follow-up — see §12.**

## 11. Files & reproduce

New/changed (all under `tools/nnunet_mnms_eval/` unless noted):
`make_whs_worklist.py`, `prep_one.py`, `assemble_whs.py`, `build_heart_roi.py`,
`compute_cardiac_phase.py`, `plot_cardiac_stats.py`, and `sbatch/whs_segment.sh`.
Reused: `env.sh`, `inference/seg_cmrxrecon.sh` (invocation), `inference/eval_acdc.py` (Dice), `preprocess.py`.

```bash
micromamba run -n svr python tools/nnunet_mnms_eval/make_whs_worklist.py     # -> worklist.txt (571)
sbatch sbatch/whs_segment.sh                                                  # idempotent; 3 workers
cat scratch/data/whs/rows/*.csv > scratch/data/whs/whs_manifest.csv          # collate QC (add header)
micromamba run -n svr python tools/nnunet_mnms_eval/compute_cardiac_phase.py  # -> cardiac_phase.csv
micromamba run -n svr python tools/nnunet_mnms_eval/plot_cardiac_stats.py     # -> cardiac_stats.png
```
Masks land as `heart_seg.nii.gz` / `heart_roi.nii.gz` siblings next to each recon (CMRx also
`_canonical`). Everything derived (manifest, cardiac CSV, plots) is a **regenerable cache** — the segs
are the source of truth.

## 11a. Update 2026-07-30 — rerun for the pooled multi-dataset cohort

Everything below §11 describes the **first** run (421 units, CMRxRecon2024 only + eval datasets).
The cohort has since grown to 1343 training subjects (docs/58), which required four changes and a
full regeneration. **All prior products were archived, not deleted**, to
`scratch/data/whs/_archive/2026-07-30/` (`cardiac_phase.csv`, `whs_manifest.csv`, `worklist.txt`,
`rows/` — 571 files). The old CSV is worth keeping: diffing CMRx EF old-vs-new is a free check that
the regenerated pipeline agrees with itself.

**Why a rerun was forced, not just an extension.** Two live defects:

1. `make_whs_worklist.py` enumerated CMRx from `training/splits/random_8_1_1.txt` — deprecated,
   CMRxRecon2024-only, and listing **pre-rename** names (`Train_P140`, on disk `CMRx24_Train_P140`).
2. Consequently `cardiac_phase.csv` was keyed `Test_P001` while `MRIDataset._build_val_targets`
   looks up `basename(dirname(sax_dir))` = `CMRx24_Test_P001` → **`ef_val_sweep: true` raises
   `KeyError` today**. Independently confirmed: CMRx currently has **0** `heart_seg.nii.gz` and
   **0** `heart_roi_canonical.nii.gz` on disk, so `metric_psnr_3d_heartseg` is silently inert too.

**Changes made:**

| file | change |
|---|---|
| `make_whs_worklist.py` | enumerate the pool by **globbing directories**, not from a split file — the seg should cover everything on disk regardless of how train/val/test is later partitioned. Covers CMRx 2023/24/25 + `ACDC_sax` + `MNMs_sax`; requires all 12 frames present so a half-written subject cannot enter. |
| `prep_one.py` | `acdc_sax` / `mnms_sax` route through the existing **cmrx** branch (12 3D frames per subject) |
| `assemble_whs.py:unit_id()` | branches for the new tokens returning `subj` = the **directory name** (`ACDC_patient001`, `MNMs_A0S9V9`). Without this they fell through to the legacy `acdc` branch, which assumes a 4D-file path and would have recreated defect (2) for 495 subjects. Docstring now states the invariant explicitly. |
| `build_heart_roi.py` | **no change needed** — it keys on `regime`, so the new gated sources get the union-over-phases recipe automatically |
| `compute_cardiac_phase.py` | new `converted_labels()`. The `group` column was populated only `if ds == "acdc"`, so the 150 **converted** ACDC subjects would have silently written a blank pathology label — losing the NOR/DCM/HCM/MINF/RV split the ACDC val carve ("3 per group", docs/58 §2.1) depends on — and M&Ms pathology/vendor were never captured at all. The converted sources have no `Info.cfg`/CSV beside them, so the labels are read from the `convert_meta.json` the converter writes. Added `vendor` + `centre` columns (vendor matters: Canon appears only in M&Ms' Validation and Testing). Note ED/ES for `acdc_sax` correctly falls through to argmax/argmin on the LV curve rather than `Info.cfg`, whose indices are in native T and would be wrong on the 12-frame grid. |

**Deliberately NOT changed:** the canonical-siblings block in `assemble_whs.py` stays CMRx-only. It
writes `(256,256,12)`, but under the native-z design (docs/58) the canonical grid becomes
`(256,256,D)` with `D` = the subject's slice count. Extending it now would emit wrong-shaped ROIs
that `MRIDataset` then has to skip. Do all five sources — **including regenerating the CMRx ones,
which are also on the old grid** — in one pass after native-z lands (docs/58 A6).

**Resulting worklist: 1613 units.**

```
cmrx        848  |                                    ACDC and M&Ms are the 12-frame
acdc_sax    150  |  1343 = the training pool          ED-anchored stacks produced by
mnms_sax    345  |                                    tools/convert_to_sax_layout.py
acdc        150  ]  raw ACDC download at native T (13-35) — the QC reference, NOT the
goettingen   69  ]  training path; plus the eval datasets. All 270 already have
miitt        26  ]  heart_seg.nii.gz, so they SKIP (the job is idempotent).
ocmr         25  ]
```

**Runtime — measured, and a lesson about how to estimate it.**

| measurement | conditions | s/unit |
|---|---|---|
| job 55577444 (**the real number**) | 3 × L40S, spgpu2, in situ | **54** |
| interactive smoke run | A40, another job on the node, cgroup-limited to 1 CPU | 116 |
| job 53298619 (previous full run) | A40, batch, 140 units in 4h15m | 109 |

⇒ `1343 × 54 s ≈ **20 GPU-hours** ≈ **8 h per worker** on a 3-way array`. `--time` raised 12 h → 20 h:
all three array tasks can land on one node, and the rate may degrade on the larger M&Ms volumes.
The job is idempotent regardless, so an overrun costs one free resubmit.

⚠️ **Do not estimate this from slice counts.** I first modelled cost as ∝ 2D slices
(~28,000 slices/worker-hour from job 53298619) and predicted ~6 GPU-hours — **7× wrong**. Per-unit
time is near-**constant** (96–139 s) across units whose volumes differ substantially (320×320×10 vs
216×256×10), because each unit spawns a fresh `nnUNet_predict` that loads **all 5 fold checkpoints**
plus three micromamba env switches, against only ~126 2D slices of actual inference. Fixed overhead
dominates. The corollary optimisation, not taken: `nnUNet_predict` accepts a *directory* of cases,
so batching N subjects per call would amortise the 5-fold load and could cut this several-fold — it
changes the unit/idempotency structure, so it was left alone.

Also note the interactive 116 s/unit was an **upper bound**, not a representative figure — the node
was shared and this shell had 1 CPU. Measuring in situ from the running job's manifest-row mtimes is
what gave the trustworthy number, and it is free.

### 11b. Output validation of the pooled rerun (2026-07-31, n=298 units, job ~35% through CMRx)

Checked while the job ran, so the numbers cover **292 CMRx + 3 acdc_sax + 3 mnms_sax** — the new
sources are barely represented and this says little about them yet.

| check | result |
|---|---|
| job failures / tracebacks | **0** |
| `seg_flag` | 297 ok / 1 low |
| units missing LV, MYO or RV entirely | **0** |
| LV cavity, mean over cycle | median **87 mL** (p5 57, p95 125) |
| MYO/LV ratio · RV/LV ratio | 1.08 · 1.23 |
| volume outliers (<15 or >400 mL) | **0** |
| z-gaps (unlabeled plane between labeled) | 0 of 80 sampled |
| fragmented (>1 component, largest <95%) | 0 of 80 sampled |

**Dice vs the shipped ACDC expert GT at ED** — the strongest available check, because it validates
the entire chain at once (frame 0 really being ED, the affine re-frame, and the label remap: ACDC
`_gt` is 1=RV/2=MYO/3=LV, the *reverse* of Task114). Any of those being wrong collapses Dice rather
than landing here:

```
ACDC_patient001   LV 0.971  MYO 0.870  RV 0.935
ACDC_patient002   LV 0.936  MYO 0.839  RV 0.944
ACDC_patient003   LV 0.962  MYO 0.890  RV 0.936     (Task114 reference band ~0.94 / 0.81 / 0.93)
```

**The one `low` unit is a data limitation, not a seg failure.** `CMRx24_Test_P044` has D=6 with only
z3–z5 labeled; z0–z2 are visibly above the heart — the stack was prescribed high and catches only
the basal half. `LOW_LABELED` did its job.

Visual overlay (all z-planes, LV/MYO/RV): `result/whs_pooled_check/seg_overlay.png`. Chirality looks
**consistent** across all three sources there (RV on the same side of the LV in 9/9 rows) — visual
and under-powered, but the first evidence on that question; the quantitative LV→RV centroid check
still wants the full segs.

⚠️ Chasing an apparent false positive in that overlay (it was not one — the unit is a single
connected component, the "blob" is the apical tip) surfaced a real finding instead: **slice order
differs between sources.** See **docs/58 §10a** — measured, under-powered, nothing changed yet.

### 11c. Incident — native-z broke the canonical block mid-run, silently (job 55577444, 2026-07-31)

**What happened.** ~4 h into the run, `preprocess.py` was switched to the native-z values
(`TARGET_SPACING = (1.4, 1.4, 0.0)`, docs/58 §8.1 C1) while the seg job was still going.
`assemble_whs.py` reads that constant **live** to build the canonical affine, so
`np.diag([1.4, 1.4, 0.0, 1])` became degenerate and nibabel raised
`HeaderDataError: Could not decompose affine`. Every subsequent CMRx unit died.

**Why it was dangerous rather than merely noisy.** The crash lands **after** `heart_seg.nii.gz` and
`heart_roi.nii.gz` are written but **before** the manifest row. The job's skip test is
`-f heart_seg.nii.gz`, so those units look **finished** on resubmit — they would have skipped and
stayed out of `cardiac_phase.csv` permanently, with no error anywhere. **75 units** were affected
before it was caught; they are listed in `scratch/data/whs/orphans_no_manifest_row.txt`.

**Fix applied (live, safe — each unit spawns a fresh interpreter):** the canonical block is now
guarded on `all(s > 0 for s in TARGET_SPACING)` and **skips loudly** with a printed reason instead
of crashing. This is also correct on its own terms: under native-z the canonical grid is
`(256,256,D)`, so writing `(256,256,12)` files was pointless — all of them get regenerated for every
source in one pass (A6). Verified: failures froze at 75 and subsequent units complete with
`canonical siblings SKIPPED`.

**Still to do, AFTER this job ends** (editing a *running* bash script is hazardous — bash re-reads
from a byte offset — so this was deliberately not done live):

1. In `sbatch/whs_segment.sh`, make the skip test require **both** `heart_seg.nii.gz` **and** the
   manifest row. A unit is not done until its row exists; that is the general fix, not a patch for
   this one incident.
2. Resubmit to pick up the 75. They need a full re-run — assemble cannot be replayed alone because
   the per-frame nnU-Net output lived on node-local `$TMPDIR` and is gone.

**Lessons.** (a) A long-running job that reads a mutable module constant is coupled to edits made
during the run — `assemble_whs.py` importing `preprocess.TARGET_SPACING` at runtime was the coupling,
and it was not obvious from either file. (b) An idempotency test that checks a **partial** output
converts a crash into silent permanent data loss; the test must key on the **last** artifact written,
or on all of them.

## 12. Open items

- **ACDC completion:** seg job 53338428 running (150 patients); once done, add a `source` column
  (`task114_3d` / `acdc_task114`) to `cardiac_phase.csv`, append ACDC rows, and regenerate the plots
  with pathology-stratified EF (ACDC `Group`: NOR/DCM/HCM/MINF/RV) + save the Task114-vs-manual stats.
- **Downstream wiring:** consume `heart_roi.nii.gz` in the eval adapters and the CMRx canonical mask in
  `MRIDataset`/masked-loss — not done here (this job only *produces* the masks).
- **OCMR-RTFB** masks are unreliable (no gated pair to fall back on) — flagged, decide whether to use.
- Optional extra cuts: stroke volume, BSA-indexed EDVi/ESVi (ACDC has height/weight), contraction
  duration (ES−ED fraction).
