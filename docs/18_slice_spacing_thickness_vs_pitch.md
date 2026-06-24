# 18 — Slice spacing: thickness vs. true pitch (canonical Z = 12 mm)

> **TL;DR & takeaway**
> Our CMRx (and OCMR) recon NIfTIs were stamped with Z spacing = **8 mm**, but
> that's the slice *thickness*, not the center-to-center *pitch*. CMRxRecon2024's
> protocol is 8 mm thickness **+ 4 mm gap = 12 mm pitch**, so the canonical cube was
> **Z-squished to 67 %** of true anatomy. **Fix: relabel the affine on disk** (header
> only, voxels untouched, reversible) to each dataset's *true* pitch — **CMRx → 12 mm**,
> **OCMR → its measured 9.6/10 mm** (from `meta.json` slice positions) — and set the
> canonical `TARGET_SPACING` Z (training) and `CANON_Z_SPACING_MM` (eval) to **12 mm**.
> Because CMRx's affine now genuinely says 12, the `Spacingd→12` resample is a
> **Z-identity**, so the cached training tensor is byte-identical to before → the
> existing checkpoint warm-starts unchanged (**no retrain**). The script is
> `tools/relabel_slice_spacing.py` (`--dry-run`, `--revert`). ACDC (5/10 mm) and
> Göttingen (6 mm) affines are already honest and were left alone. Per-dataset true
> pitch: **CMRx 12** (protocol), **OCMR 9.6–10** (measured), **ACDC 5/10** (honest),
> **Göttingen 6** (assumed-contiguous, unverified).

---

## 1. The bug

`MRIDataset` resampled every subject to a canonical cube whose Z spacing was `8.0 mm`.
That 8 mm came from our k-space→NIfTI recon, which set Z spacing = `SliceThickness`
straight from `cine_sax_info.csv`. But the CSV ships **only** `SliceThickness=8` and
`SliceNum` — **no `SpacingBetweenSlices`, no gap field** (the raw `cine_sax.mat` holds
only `kspace_full`, no slice positions). So the 4 mm inter-slice gap was silently dropped.
The CMRxRecon2024 protocol is **8 mm thickness + 4 mm gap = 12 mm pitch**; stacking
12-mm-apart slices at 8 mm **compresses the heart through-plane to 8/12 = 67 %**.

## 2. Why it matters (and where)

The model works in normalized `[-1,1]` coords, so within CMRx the squish is
self-consistent and the absolute mm normalizes out. It bites at exactly two places:

1. **Respiratory simulation** (`respiratory.py`) — the one spot a physical-mm amplitude
   is divided by the Z spacing (`_norm_delta`). At 8 mm a 16 mm SI breath injected
   `16/8 = 2` voxels; at the true 12 mm it's `16/12 = 1.33`. The 8 mm value
   **over-injected SI motion by 1.5×**.
2. **Cross-dataset eval transfer** — the model learned "≈12 mm of anatomy per canonical
   slice." An eval recon gridded at a different real mm-per-slice makes it mis-scale the
   through-plane (`dz`) reconstruction (z-distortion, `dz` motion under/overshoot,
   through-plane blur, dim base/apex planes). In-plane transfers fine.

## 3. Per-dataset true pitch + evidence

| Dataset | Header was | **True pitch** | Basis | Confidence |
|---|---|---|---|---|
| **CMRx** | 8 (thickness) | **12 mm** | CMRxRecon2024 protocol (8 + 4 gap); no positions in our data | Their documentation |
| **OCMR** | 8 (thickness) | **9.6 / 10.0** (per-subject) | **Measured** from `meta.json slice_positions_mm` | Certain (measured) |
| **ACDC** | 5 / 10 | **= header** | Affine already = pitch (127/150 at 10, 21 at 5; thickness "generally 5 mm" so 10≠thickness) | High (inferred) |
| **Göttingen** | 6 | **6 mm (assumed)** | `cfl_to_nifti.py` hardcodes 6; README states pitch is *assumed* contiguous (no positions in raw `.cfl`) | Low — unverified |

Coverage sanity (`nslices × pitch`): ACDC median 90 mm (textbook — and proves its affine
= pitch, since thickness 5 mm would give an absurd ~45 mm); OCMR 120 mm (measured);
Göttingen 144 mm (plausible whole-heart; a gap → implausible 160–216 mm); CMRx @12 = 120
median (generous but within protocol). **Option B** (reconstructed the 2024 `cine_lax.mat`
long-axis and measured base→apex) independently confirmed the heart exceeds the @8 mm SAX
coverage; it can't cleanly separate 10 vs 12, so we adopt **12 from the protocol** (OCMR/ACDC
~10 are different scanners — context, not an override). Verify imgs: `result/slice_spacing_verify/`.

## 4. The fix — on disk + two constants

- **On disk** (`tools/relabel_slice_spacing.py`, in place, header-only, reversible):
  rewrites only the slice-axis affine column to the true pitch.
  - CMRx `8→12` on `…/sax/3d_recon/sax_frame_*` + `…/sax/4d_recon` (301 subj)
  - OCMR `8→measured 9.6/10` on `…/recon/*/sax_cine.nii.gz` (from `meta.json` positions)
  - ACDC / Göttingen: untouched (already honest)
  - Voxel data byte-untouched (via `set_sform`/`set_qform`; verified md5-identical). These
    are *our* recons (regenerable from k-space), not pristine source. `--revert` sets Z→8.
- **Code (constants only):** `preprocess.TARGET_SPACING` Z 8→12, `HALF_EXTENT` 48→72;
  `eval/adapters/base.CANON_Z_SPACING_MM` 8→12 + `MM_PER_NORM` Z 44→66 (single source of
  truth — covers `tools/eval_ocmr_inference.py` and `diagnose_ood_clean_paradox.py` via
  re-export); `respiratory.SPACING_MM` Z→12; diagnostic mm labels (`trainer.py`,
  `dump_predicted_dvf.py`, `render_respiratory_examples.py`) and the nnU-Net eval header
  (`nnunet_mnms_eval/prep_inputs.py`) 8→12.

**Notably NOT changed:** no in-pipeline relabel transform / param — the data is honest on
disk, so the pipeline just resamples (CMRx 12→12 identity, ACDC 5/10→12, Göttingen 6→12).
The eval adapters' *native* spacings stay (Göttingen `SLICE_SPACING_MM=6`, MIITT `8`
placeholder, OCMR via meta positions) — those are dataset properties, binned to the new 12 mm
canonical by `assign_canonical_z`. **OCMR's disk relabel is honesty-only**: the OCMR eval path
reads `meta.json` positions directly and never the NIfTI affine, so it has no functional effect.

## 5. No-retrain guarantee

CMRx affine now genuinely says 12 → `Spacingd→12` is a true Z-identity (X/Y resample
unchanged, pad/crop centering unchanged) → cached tensor byte-identical to the old 8mm/8mm
pipeline. Verified (prove-it review of the earlier in-memory variant: **max|Δ| = 0.0** for
CMRx's diagonal affine across Z slice counts 6–14). The existing checkpoint warm-starts
unchanged; only respiratory injection (future runs) and eval gridding change.

## 6. Verification done

- `set_sform/set_qform` preserves voxel data **byte-exactly** (md5 identical before/after) —
  tested on synthetic + a real CMRx file round-trip (relabel→12, revert→8, data unchanged).
- CMRx affines are pure-diagonal, slice axis is the unique column with norm ≈ 8 (the script
  asserts exactly one such axis before writing).
- Full pytest suite green (183 passed), incl. `test_eval_ocmr_equivalence.py` (the frozen
  OCMR refactor guard `_legacy_ocmr.py` tracks the 12 mm canonical so it stays valid).

## 7. Open / follow-up

- **Göttingen 6 mm is assumed, not measured** — verify vs Blumenthal et al. MRM 2024 / the
  Zenodo record before fully trusting it.
- **CMRx 10 vs 12**: anatomy (OCMR/ACDC ~10, coverage, LAX) leans ~10; we trust the
  protocol's 12. Low-stakes for the model (a 20 % global scale, identical mechanics) but it
  sets how much eval resamples. Resolvable only from raw CMRx slice positions, which our data
  lacks.
- nnU-Net seg eval (`docs/15`): re-run its `prep_inputs` on `val_volumes` from the 12 mm
  pipeline (its header is now 12, so a stale 8 mm run would distort the heart and bias Dice).
