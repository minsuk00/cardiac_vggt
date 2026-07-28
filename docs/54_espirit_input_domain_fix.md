# 54 — The ESPIRiT input-domain bug, the fix, and the full v1→v2 re-reconstruction

> **TL;DR & takeaway**
>
> Every CMRxRecon SAX volume this project has ever used was reconstructed with `EspiritCalib` fed an
> **image** where its API expects **k-space**. The coil maps that came out were nearly *uniform in
> magnitude*, which collapsed the SENSE combine into a phase-aligned coil sum — anatomy intact, but the
> receive-coil (B1−) shading never divided out. Measured on 2024 `Train_P154`, `corr(SENSE, RSS)` fell
> from **0.984 at the centre to 0.651 at the periphery**; with correct maps it is 1.000 → 0.981. Two
> 2025 UIH subjects had peripheral correlation of **0.03 and −0.10**.
>
> **The fix is one line** (pass `ref_kspace_gpu` instead of `sp.ifft(ref_kspace_gpu)`). It was applied
> 2026-07-27 and **all three years were re-reconstructed**: the recon run produced **851** volumes
> (2023 196 / 2024 295 / 2025 360); the **live cohort is 850** (2023 196 / 2024 **294** / 2025 360)
> because `CMRx24_Train_P002` has only 2 z-slices and was moved back out to the archive (§1b of the
> 2024 README). The v1 data is preserved at
> `scratch/data/CMRxRecon{2023,2024,2025}_recon_v1_espirit_imagedomain/`.
>
> **The Z-pitch relabel and a full verification are now DONE too** (2026-07-27, §10). All 850 live
> subjects / 11,050 files were fully decompressed and checked: **0 problems** — correct slice pitch,
> LPS everywhere, no all-zero or duplicated frames, every 3d frame bit-identical to its 4d slice, and
> every subject confirmed to differ from its archived v1 (i.e. genuinely re-reconstructed, not a
> stale leftover).
>
> **2025 slice pitch was MEASURED, not assumed** (§10c): `Center006/Prisma` is **10 mm, not the
> assumed 12** — relabeled (60 subjects, a 17% z correction); `Center001/Vida` confirmed at 12;
> `Center004/Aera` is **unresolved** between 12 and 14 and left at 12 (only 4 of its 44 subjects have
> a long-axis series, all used). The +4 mm gap rule holds everywhere — only the *thickness* varies by
> centre — which **refutes** this session's earlier "contiguous ~6 mm" hypothesis.
>
> **Duplicates are settled for all three years** (§11): 2023 **68**, 2024 **7 pairs**, 2025 **1 pair**
> — none of them consumable, all excluded at the *data* level rather than merely flagged in splits.
> 2025's was found and archived on 2026-07-27, taking the live cohort to **849** (196 + 294 + 359).
> All pairs visually confirmed as the same scan released twice (corr 1.000000).
>
> **Philips spacing is no longer an open item here — see `docs/55`** (it was `pixel_x`, not `FOVy`;
> fixed on disk, exclusion lifted).
>
> **Status: DONE for reconstruction; downstream regeneration still OWED.** `heart_seg*`, `heart_roi*`,
> `dvf_*` were derived from v1 and have been moved into the archive — they must be regenerated. The
> ~270 GB `scratch/eval` baseline harness and **every number in `evaluation/results/`** were also built
> on v1 and are superseded. Do not quote any existing baseline PSNR/Dice/EF as current.
>
> **The earlier "keep the bug for cross-year consistency" decision (recon_code README) is OVERTURNED** —
> the defect's severity varied by year, so preserving it produced three *different* appearance
> signatures rather than one consistent one.

---

## 1. The bug

`scratch/data/CMRxRecon2024/recon_code/batch_reconstruct_cmrxrecon2024.py` (canonical copy; the repo's
`_archive/` path is now a symlink to it) did:

```python
ref_kspace_gpu = cp.array(ref_kspace)
ref_image_gpu  = sp.ifft(ref_kspace_gpu, axes=[-2, -1])          # -> IMAGE domain
smap_gpu = mr.app.EspiritCalib(ref_image_gpu, crop=0.80, ...)    # API wants K-SPACE
```

Verified against the sigpy 0.1.27 source, not inferred:

- the parameter is named **`ksp`**, documented *"k-space array"*;
- `__init__` does `calib = sp.resize(ksp, calib_shape)` — it grabs the **centre block**, which is the
  ACS calibration region *only if the input is k-space*. Fed an image, it calibrates on a central
  32×32 patch of anatomy;
- `_output()` does `mps *= max_eig > self.crop` — a **hard binary mask** at eigenvalue > 0.80.

## 2. Why it still "worked" — the part that was initially explained wrong

The first explanation offered (and it was wrong) was that a central image patch still encodes inter-coil
sensitivity ratios. Measurement refuted that. What actually happens:

**The image-fed maps degenerate to approximately uniform magnitude.** With `Σ_c|S_c|² ≈ 1` and no spatial
structure, the SENSE combine `Σ conj(S_c)·I_c / (Σ|S_c|² + ε)` reduces to a **phase-aligned sum of
coils** — a perfectly valid, if suboptimal, reconstruction. That is why the anatomy was always fine. It
was not accidentally recovering sensitivities; it was silently falling back to a simpler combiner.

What is lost is the **receive-coil bias correction**. sigpy's power method normalises the maps to unit
norm per voxel, so with *correct* maps `I_c = S_c·M` gives `Σ|S_c|²·M = M` exactly — the coil profile
cancels algebraically. With ≈uniform maps you instead get `M · [(1/√nc)·Σ e^{−iφ_c}·S_c^true]`, and that
bracket is a residual, spatially-varying bias field.

Measured on 2024 `Train_P154` (frame 0, mid slice), correlation against the map-free RSS reference by
radius:

| radius | image-fed (v1) | k-space-fed (v2) |
|---|---|---|
| 0.0–0.2 (heart) | 0.984 | 1.000 |
| 0.2–0.4 | 0.858 | 1.000 |
| 0.4–0.6 | 0.844 | 0.999 |
| 0.6–0.8 | 0.848 | 0.995 |
| 0.8–1.2 (periphery) | **0.651** | 0.981 |

Eigenvalue-mask support, same subject: overall 0.845 (v1) vs 0.533 (v2), but **inside the body** 0.937
(v1) vs **0.992** (v2). The overall number is misleading because it counts air — the corrected maps
correctly mask *air* while keeping ~5.5% more body voxels that v1 was hard-zeroing.

**A clean independent confirmation:** after the fix, `corr(SENSE, RSS) = 1.000` in the heart for all 9
A/B subjects. That is an algebraic identity — unit-norm maps make `S_true = α(r)·S_est`, so SENSE gives
`α·M` and RSS gives `α·|M|`, identical in magnitude. Recovering it to three decimals on real data says
the maps are now right. (It also means that for *fully-sampled* data ESPIRiT was never buying anything
over plain RSS; parallel-imaging coil maps exist to unfold **undersampled** acquisitions, and
`Cine_combined` is the fully-sampled ground truth.)

## 3. Why it mattered more than "the heart is fine either way"

Severity varied by year, because our own ky zero-fill changes how much invented data the (mis-)calibration
sees: map support 95% (2024, no fill) → 80.9% (Philips native) → **54.8%** (Philips after fill). So
keeping the bug did **not** buy cross-year consistency — it produced three different severities of the
same defect. That is what overturned the earlier decision.

The stronger, forward-looking argument: preprocessing normalises each subject against phase_00's
0.5/99.9 percentiles over non-zero FOV voxels. A residual receive-coil bias field that differs by scanner
and coil configuration feeds straight into those percentiles — and the cohort is going from one Siemens
scanner to **11 scanner models across 3 vendors**. (Measured impact of the fix on that statistic was
small — zero-fraction +3.4%, scale-free dark-floor/bright-tail ratio shift ≤0.0008 — so this was a
risk-reduction argument, not a measured problem.)

## 4. The fix, and where it lives

```python
ref_kspace_gpu = cp.array(ref_kspace)
smap_gpu = mr.app.EspiritCalib(ref_kspace_gpu, crop=0.80, thresh=0.01, calib_width=32, ...)
```

**Provenance hardening done at the same time.** The recon script previously existed only at
`_archive/batch_reconstruct_cmrxrecon2024.py`, which is **gitignored** — the script that generated the
entire training set was not under version control. Now:

- `scratch/data/CMRxRecon2024/recon_code/batch_reconstruct_cmrxrecon2024.py` is the **canonical** file
  (md5 `69f2ffb74449eb3d6e739e6a1f045f88`), parked next to the data;
- `_archive/batch_reconstruct_cmrxrecon2024.py` is a **symlink** to it — one file, cannot drift;
- the pre-fix script is preserved verbatim as `batch_reconstruct_cmrxrecon2024_v1_ORIGINAL_espirit_image_domain.py`
  (md5 `65eb34a61ca77392e85b642abff8be76`).

All three drivers (`tools/reconstruct_cmrx{2023,2024,2025}.py`) import `reconstruct_subject` through
that symlink, so the fix applies uniformly.

## 5. Validation before the re-run

A/B on **3 subjects per year through the real `reconstruct_subject`** (not a reimplementation):

| year | new~old global | heart ROI | periphery vs RSS: v1 → v2 |
|---|---|---|---|
| 2023 ×3 | 0.742–0.810 | 0.948–0.999 | 0.662–0.864 → 0.999–1.000 |
| 2024 ×3 | 0.676–0.698 | 0.956–0.996 | 0.692–0.883 → 0.998–0.999 |
| 2025 ×3 | 0.360–0.897 | 0.907–0.999 | **−0.101, 0.032**, 0.807 → 0.968–0.999 |

Periphery improved **9/9** (mean +0.394); heart preserved everywhere (min 0.907). Figures:
`result/cmrx2025_recon_check/espirit_fix_before_after_physical.png` (true physical aspect) and
`espirit_domain_explained.png` (the near-uniform vs correctly-falling-off coil maps).

Independently, `/prove-it` ran 6 reviewers + 6 adversarial refuters over the three drivers plus the
shared recon (790 lines) and the full test suite (**215 passed**). Every high/critical correctness claim
was refuted, several by scanning all existing volumes. Two real defects were found and fixed: the 2025
driver had **no resume guard**, and its **report was clobbered on every run and never recorded failures**
(already proven on disk — 360 dirs vs a 348-entry report).

## 6. The re-reconstruction

851 volumes across 7 GPU tasks. **2023 + 2025 ran on A40 (spgpu); 2024 on L40S (spgpu2)** because
spgpu went into cluster-wide emergency maintenance mid-run.

| year | subjects | verified |
|---|---|---|
| 2023 | 196 | 0 missing, 0 shape-mismatch, 0 zero-plane |
| 2024 | 295 (294 v1 cohort + `Train_P002`) | " |
| 2025 | 360 | " |

Verification report: `scratch/data/recon_v2_verification.json`. Merged 2025 report:
`scratch/data/CMRxRecon2025/recon_report.json` (360 rows, 0 errors); v1 reports kept as `v1_*`.

**Cross-architecture note (A40 sm_86 vs L40S sm_89):** `EspiritCalib` is deterministic (power method
initialised with `xp.ones`, no RNG), so the only difference is float32 reduction order in
cuFFT/cuSOLVER. The documented same-arch noise floor is 135–137 dB PSNR / max abs diff 1.8e-08 / corr
0.999999999998. Judged negligible against the fix itself (which moved global correlation to 0.68–0.90)
and accepted without a dedicated cross-arch experiment.

## 7. Two operational bugs found the hard way

**`shutil.SameFileError` (2024).** `reconstruct_subject` packages its inputs with
`shutil.copy2(mat_file, <out>/sax/cine_sax.mat)`. In the live 2024 tree that destination was **already a
symlink to the same source** (from `tools/symlink_cmrx_mat_copies.py`), so copy-onto-itself raised and
killed the shard. The 2023/2025 drivers were immune because they patch `shutil.copy2` with a symlinking
stub; the 2024 run used the archive's own `main()`, which does not. Fixed by writing
`tools/reconstruct_cmrx2024.py` with the same wrapper.

**377 GB of duplicated k-space (2025).** That same `copy2` had written a full ~1 GB *staged* `.mat` into
every one of 360 output dirs. Patched to symlink the raw source (subject dir 1.3 GB → 62 MB). The
existing copies were deleted only after proving regenerability: 360/360 raw sources present, 0 orphans,
and 3 sampled subjects re-generated **bit-identical** through `normalize()`.

## 8. The silent all-zero-slice failure mode — REAL, now GUARDED

`batch_reconstruct_cmrxrecon2024.py`'s per-slice loop is `try: … except Exception: continue`, and the
output array is pre-allocated with `np.zeros`. **A failed slice stays zero, the volume is still written,
and it still prints "Processing Complete!"** — no exception, no flag.

A `/prove-it` refuter declared this unreachable based on scanning existing outputs. It then fired for
real: a smoke test accidentally run on a **login node with no GPU** produced `Train_P003` as 11/11 zero
planes, reported as a success. It would have fired again at scale — a V100 probe (`gpu` partition) died
with `CURAND_STATUS_LAUNCH_FAILURE` because **CUDA 13 dropped Volta**, and submitting 294 subjects there
would have written 294 all-zero volumes, each reporting success.

Lessons, both now guarded by memory `reference_gpu_arch_requirements`:
- the `svr` env needs **sm_75+**; V100 (sm_70) and P40 (sm_61) cannot run a single kernel;
- **`torch.cuda.is_available()` returns `True` on a V100** and then every launch fails — it is not a safe
  guard. Check `get_device_properties` against `torch.cuda.get_arch_list()`, or run a 1-line matmul.

`Train_P003` was re-reconstructed on an L40S and is clean (0/11 zero planes). **The guard is now
implemented and tested** (`recon_code/batch_reconstruct_cmrxrecon2024.py`: `failed_slices` collected at
:90/:143, the `raise` at :152-154). Forcing every slice to throw now raises
`11/11 slices failed … refusing to write a volume with all-zero planes` and writes **no** 4D and **no**
3D files, instead of the old swallow-and-continue that left `np.zeros`-preallocated planes in place and
still printed "Processing Complete!".

## 9. What is still owed

1. Regenerate `heart_seg*`, `heart_roi*` (nnU-Net Task114 → `tools/nnunet_mnms_eval/assemble_whs.py` +
   `build_heart_roi.py`) and, if still wanted, the `dvf_*` volumes. `heart_roi_canonical` is read by
   `mri_dataset.py` / `composed_dataset.py` / `loss.py` but guarded by `os.path.exists`, so training
   degrades gracefully meanwhile.
2. Regenerate `scratch/eval` (~270 GB) and every `evaluation/results/` number — see memory
   `project_eval_harness_scratch_eval`.
3. Re-render the 50 QC PNGs in `result/cmrx2025_recon_check/`, which were all made from v1.
4. Re-baseline the identity-Δ floor once 2023+2025 are wired into `MRIDataset` and 3-year splits exist.

*(The failed-slice guard and the Z-pitch relabel, previously listed here, are DONE — §8 and §10.)*

## 10. The Z-pitch relabel and the full verification (2026-07-27)

**The relabel.** 2023/2024 drivers bake `SliceThickness` (8/6 mm) into the affine, not the true pitch,
so a fresh recon carries a ~33 % Z error until a separate pass fixes it. The queued SLURM job never got
a slot (`standard` was in cluster-wide maintenance), so it was cancelled and run inline on the
interactive node with 4 workers: `RELABEL_WORKERS=4 python tools/relabel_slice_spacing_parallel.py`
→ **4,331 changed + 2,039 skipped = 6,370 files** (= 294×13 + 196×13, i.e. every 2023/2024 file),
**0 errors, 0 stray `.relabeltmp`**. The skips were an earlier partial run; they were scattered, not a
clean prefix. Use the *parallel* script — it writes to a tmp file and `os.replace`s, so a kill cannot
truncate a NIfTI (the single-threaded one is not atomic and produced a 0-byte frame once).

**The verification** — `tools/verify_recon_v2.py`, report at
`scratch/data/recon_v2_verification_full.json`. Every file is **fully decompressed**; there is no
sampling and no header-only shortcut. Checks: structure (missing/zero-byte/extra files, ≠12 frames,
leftover tmp); data (truncated gzip, dtype, T=12, NaN/Inf, **any all-zero (z,t) plane**, **duplicated
adjacent phases**); consistency (each 3d frame's voxels equal `4d[...,k]`, matching shape/dtype/affine);
geometry (axcodes LPS, slice pitch vs an authoritative per-year source, in-plane vs `FOV/ReconMatrix`,
grid vs `ReconMatrix`/`SliceNum`, affine finite/non-singular, sform/qform set); and provenance
(4d not bit-identical to the archived v1).

| year | subjects | files | pitch | axcodes | zero planes / dup phases | 3d≡4d | differs from v1 |
|---|---|---|---|---|---|---|---|
| 2023 | 196 | 2,548 | 193@12.0 + **3@10.0** | LPS 196 | 0 / 0 | max diff 0.0 | 196/196 |
| 2024 | 294 | 3,822 | 294@12.0 | LPS 294 | 0 / 0 | max diff 0.0 | 294/294 |
| 2025 | 360 | 4,680 | 323@12.0 + 37@10.0 | LPS 360 | 0 / 0 | max diff 0.0 | 360/360 |

**TOTAL PROBLEMS: 0** over 850 subjects / 11,050 files. 2023's pitch histogram matches
`SUBJECT_MANIFEST.csv` exactly (the three 6 mm subjects at 10.0); 2025's matches `recon_report.json`.
The least-changed subject still differs from its v1 by ≥43 % of peak intensity, so no subject is a
stale v1 leftover.

**Expected pitch comes from an authoritative per-year source, never a guess:** 2023 →
`SUBJECT_MANIFEST.csv` `pitch_mm`; 2024 → 12.0 for all; 2025 → `recon_report.json` `pitch_mm`.

### 10a. The verifier was fault-injected before it was trusted

A "0 problems" report is worthless until each check is shown to *fire*. 13 deliberately broken copies of
one real subject (throwaway tree, real data untouched) were run through the verifier, asserting each
variant trips its intended check and that a pristine control trips nothing: truncated gzip, zero-byte
file, all-zero plane, NaN, wrong pitch, missing frame, 3d/4d voxel mismatch, LAS orientation, wrong
in-plane spacing, leftover tmp file, duplicated phase, v1-identical leftover, and a genuinely-new
control. All 13 pass.

**This caught a real bug in the verifier itself.** The 2024 expected-pitch map was built as a
`collections.defaultdict`, but `main()` read it via `.get(s, default)` and `s not in map` — **neither
triggers a defaultdict's factory**. Every one of the 294 subjects would have received
`expected_pitch=None`, degrading the pitch check into a vacuous "no authoritative expected pitch"
message while still looking like it ran — i.e. the *exact* check whose absence caused this open item in
the first place. Reading the code did not reveal it; only injection did. Lookup maps in verification
code should be **plain dicts**, so a missing key is loud. Recorded as memory
`feedback-fault-inject-verifiers`.

### 10b. What this does NOT prove

The checks validate the data against its **stated** metadata. They cannot tell you the source metadata
is itself correct, so these remain genuinely open and are *not* settled by the 0-problem result:
- the **114 provisional 2025 pitches** (blank `SliceThickness` → assumed 12 mm) pass because they are
  checked against the assumption;
- ~~the **Philips `FOVy=299`** question — the in-plane check compares against the source CSV, which is
  the very thing under suspicion.~~ **SUPERSEDED 2026-07-27 → `docs/55`. It was never `FOVy`.** The
  defective quantity is **`pixel_x`**: `reconstruct_subject` derives spacing as
  `FOVx / ReconMatrix_X`, but `ReconMatrix_X` is an **output grid size**, not an acquired sample count.
  Philips acquires `nx = 304` with `ReadOutOversample = 2` (⇒ base = **152**) and reconstructs onto
  `rx = 256`, so the stamped 1.168 mm under-scaled the readout axis by `256/152 = 1.684`; every Siemens
  subject escaped because `nx = 2·rx` makes `base = rx`. `pixel_y = FOVy/ry` was correct all along —
  ky zero-filling preserves the FOV while readout cropping preserves the pixel. **Fixed on disk**
  (`tools/fix_philips_pixel_x.py`, 12 subjects / 156 files + staged CSV + `recon_report.json`,
  reversible), and the Philips exclusion is lifted. Note the in-plane check reads the **staged** CSV,
  which `normalize()` derives (UIH carries 540 where its source says 720) — so it had to be updated
  in step or all 12 would have failed `inplane`. Re-verified: 2025 **0 problems / 359 subjects**.
  ⚠️ Still owed: nnU-Net confirmation that 1.967 (not `299/304 = 0.983`) is right — that choice rests
  on an anatomical prior, though *that 1.168 is wrong* does not. See `docs/55` §4b.

The 114 provisional pitches are addressed in §10c; the Philips spacing in `docs/55`. Both would also be
served by the same nnU-Net LV-diameter measurement.

### 10c. The 2025 slice pitch was MEASURED (2026-07-27) — Prisma relabeled 12 → 10 mm

**The problem.** 114 subjects at three Siemens centres ship an **empty** `SliceThickness`
(`SliceThickness,` — present key, no value; verified in the raw source, and the `.mat` carries no
geometry at all: one `kspace` dataset, no attributes, no slice positions, and there are no DICOMs).
Their pitch was therefore defaulted to 12 mm. A second, independent assumption — the CMRxRecon**2024**
"+4 mm gap" rule — sits under **all 359**.

| centre / scanner | n | thickness | note |
|---|---|---|---|
| Center006 / Siemens_30T_Prisma | 60 | EMPTY | median Z=13, max 18 |
| Center004 / Siemens_15T_Aera | 44 | EMPTY | median Z=10 |
| Center001 / Siemens_30T_Vida | 10 | EMPTY | median Z=13 |

Every UIH scanner and Philips populate the field. Blank thickness does **not** predict trouble —
Aera is blank yet perfectly plausible; **high slice count** is the predictor.

**The method** (`tools/render_pitch_measurement_panels.py` → human reading →
`tools/analyze_pitch_measurements.py`). Pitch is a **per-centre protocol property**, so this is 3
unknowns, not 114. For a handful of subjects per centre: measure LV length `L` in mm on the 4ch
long-axis (in-plane spacing is KNOWN there; the quick RSS render was validated to reproduce
`reconstruct_subject`'s grid and FOV *exactly*), count the SAX slices `N` containing LV, then
`pitch = L/N`. Calibrated on **control subjects with a documented thickness**, whose known 12.0 mm
was reproduced to **+2.1%**.

| centre | n | mean-of-ratios | pooled slope | verdict |
|---|---|---|---|---|
| **Center006 / Prisma** | 6 | **10.25 ± 0.55** | **10.05** | **10 mm — CONCLUSIVE** (8 and 12 excluded at 2SE) |
| Center001 / Vida | 2 | 12.22 ± 0.58 | 12.27 | 12 mm — correct, no change |
| Center004 / Aera | 4 | 13.53 ± 1.09 | 13.11 | **12 or 14 — UNRESOLVED**, left at 12 |

**Action taken:** `tools/relabel_cmrx2025_pitch.py` relabeled **Center006/Prisma 12 → 10 mm**
(60 subjects × 13 files = 780, atomic, 0 skipped) and updated `recon_report.json` in step —
necessary because `tools/verify_recon_v2.py` takes its expected 2025 pitch from that report. The
superseded value is preserved per subject as `pitch_mm_assumed`, with
`pitch_source: measured_lax_lv_length_2026-07-27`; full backup at `recon_report.json.bak_preprisma`.

**Aera is left at 12 mm and is genuinely unresolved.** Separating 12 from 14 at its observed scatter
(σ ≈ 2.2 mm) needs n ≈ 19, and **only 4 of its 44 subjects ship any long-axis series — all 4 were
used.** This method cannot do better there. Residual risk: ≤15% z error on 44 subjects. The
slice-to-slice decorrelation route (calibrate image similarity vs known through-plane distance on
documented-pitch centres) needs no LAX and would work on all 44 if it ever matters.

**What this overturns.** An earlier hypothesis in this session — that those centres acquire
*contiguous* slices, making the true pitch ~6 mm — is **REFUTED**. Every measured centre lands at
10–13.5 mm, fully consistent with the +4 mm gap rule; only the **thickness** varies by centre
(Prisma 6 mm at 3T, Vida-C001 8 mm at 3T, Aera 8–10 mm at 1.5T, where thicker slices for SNR are
routine). Prisma's "impossible" 216 mm stacks are likewise explained: at 10 mm that is 180 mm, of
which the LV occupies ~120 mm — **genuine over-coverage, not a geometry error**.

**Two analysis bugs the controls caught**, both mine:
1. I first specified `pitch = L/(N−1)` (slice *intervals*). The controls showed that biases **+14.3%**;
   `L/N` is right, because each slice covers a slab ~one pitch thick so the LV extends about half a
   slice beyond the outermost slice centres at each end. Calibrated bias for `L/N`: **+2.1%**.
2. My candidate-pitch list stopped at 12 mm, which silently snapped Aera's 13.5 down to "conclusively
   12". Extended to 16 mm, Aera is correctly reported as inconclusive.
Also: the first per-centre error bars used the *control* scatter (±7%) for every centre, which faked
precision for centres whose own subjects scatter 25–33%. Now the larger of (control-derived,
observed) SE is used, so n=2 cannot masquerade as conclusive.

## 11. Duplicate subjects across all three years (2026-07-27)

Duplicate detection is a **separate** concern from §10 — the verification checks each subject in
isolation and has no notion of two subjects being the same person. Records live next to the data in
each year's `DUPLICATES.txt`.

| year | scanned | unique | redundant | cross-split? | live tree |
|---|---|---|---|---|---|
| 2023 | 265 | 197 | **68** (all test-side) | yes (train/val ↔ test) | already excluded — the 68 carry `reconstruct=0` and were **never reconstructed** |
| 2024 | 301 | 294 | **7 pairs** (`Train_P192..P200`) | yes (train ↔ test/val) | already excluded — v1 recons only, moved to `_archive/` on 2026-07-24 |
| 2025 | 405 | 404 | **1 pair** (`TaskR2/ValidationSet` P005↔P006) | **no** — same split *and* set | archived 2026-07-27, cohort **360 → 359** |

**Live cohort after dedup: 196 + 294 + 359 = 849** (the §10 verification ran on 850, before the 2025
duplicate was archived). Pooled *unique* subject count across the three years: 196 + 294 + 404 = 894,
of which 849 are reconstructed (2025 reconstructed 360 of its 405 sources).

**Method** (same three-part design each year; full evidence in each `DUPLICATES.txt`): hash a
*canonicalized data array*, never file bytes — a re-release in a different container, compression or
header would otherwise defeat it. 2023 hashed a raw k-space plane; 2024 hashed the reconstructed
voxel array then re-confirmed on raw k-space; 2025 hashed raw k-space in **chunk-aligned blocks**
(one chunk in ky/kx, full in `(t,z,c)`, ~46 KB) because the gzip chunking is `(nt,nz,nc,2,1)` and a
naive plane read decompresses the whole file (73 s/plane → ~17 h). Group on one block set, **confirm
on a second independent one**; h1-only agreement is reported *suspicious*, not duplicate. Controls
that make it evidence: a shape negative-control (2024: 103 same-shape subjects, zero collisions;
2025: largest shared-shape group 16, zero collisions), a zero-norm degeneracy check, and an
inode/`nlink` check ruling out hardlinks.

### 11a. Visual confirmation (2026-07-27)

Both members of a pair must come from the **same recon version**, or the ESPIRiT fix itself shows up
as the difference. 2024's redundant copies were v1, so they were re-reconstructed; 2023's had never
been reconstructed at all and were built from raw k-space for the first time (the stronger test —
nothing about them derives from the kept subject). Figures in `result/`, script
`tools/render_cmrx_duplicate_pairs.py`:

| year | pairs shown | corr | max\|diff\| |
|---|---|---|---|
| 2024 | 7 of 7 | 1.000000 | **0** (exact) |
| 2023 | 5 of 68 | 1.000000 | 9.3e-10 – 7.7e-8 |
| 2025 | 1 of 1 | 1.000000 | **0** (exact) |

**The 2023 residual is GPU numerics, and this was measured, not assumed.** Control: re-reconstructing
a *kept* 2023 subject (`CMRx23_Val_P041`, whose live volume came from the A40 run) on an L40S gives
`max|diff| = 2.24e-08`, relative 7.97e-06, corr 1.0000000000 — same magnitude as the pair residuals,
same person, same raw k-space. So exact-zero appears where both sides ran on the same architecture
(2024, 2025) and ~1e-8 where they did not (2023: live A40 vs re-run L40S). This also finally puts a
number on the A40-vs-L40S float32 difference accepted without measurement in §6: **~8e-6 relative**.

### 11b. 🔴 The 2025 scan had a silent key bug

`tools/scan_cmrx2025_duplicates.py` keyed subjects on `parts[-5]` — which is `FullSample_TaskR1`, not
the Set level (`parts[-6]`) its own comment claimed to include. TaskR1 ships the same
Center/Scanner/P### in **both** TestSet and ValidationSet (4 cases), so those 4 TestSet subjects were
silently overwritten and **never compared**: 405 files produced only 401 keys. The dropped 4 were
precisely the highest-risk cross-split candidates. Fixed; an assert now fails loudly on any key
collision, and "cross-split" counts crossing the **Set** level too, not just the task. After the fix
all 405 are compared and those 4 are confirmed to be **different people**.

Same lesson as §10a and memory `feedback-fault-inject-verifiers`: the first result *looked* clean
(1 pair, 0 cross-split) and was arithmetically identical to the final answer — the bug was visible
only in the subject count, 401 vs 405. **Always check that a scan compared everything it was given.**

## Sources

- sigpy 0.1.27 `sigpy/mri/app.py` — `EspiritCalib.__init__` / `_output` (read directly, not cited from docs).
- Uecker et al., *ESPIRiT — An Eigenvalue Approach to Autocalibrating Parallel MRI*, MRM 71:990–1001 (2014).
- `scratch/data/CMRxRecon2024/recon_code/README.md` — provenance, the reproduction check (135–137 dB), and the struck-through prior decision.
- `docs/27` (Z-pitch relabel); `docs/55` (the Philips `pixel_x` fix, which **supersedes** the Philips
  `FOVy` framing in §10b and lifts the exclusion); `scratch/data/CMRxRecon2025/README.md` §"Cohort
  decisions".
