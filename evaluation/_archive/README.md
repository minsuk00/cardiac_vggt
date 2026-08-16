# `evaluation/_archive/` — pre-native-z harness results and builders

Moved here 2026-08-16, **not deleted**. These describe real past runs and stay readable rather
than being rewritten out of history.

## Contents

| item | what it is |
|---|---|
| `results/{acdc,cmrxrecon,miitt,ocmr}/` | 19 scored VGGT arms + `svrtk3d.json` / `nesvor.json`, from the 2026-07 harness. |
| `MODELS.md`, `models.json` | The model card / registry for those arms. |
| `build_inputs/{acdc,miitt,ocmr,geom}.py` | Per-dataset bundle builders, replaced by one `pooled.py`. |

## Why these numbers cannot be reused

Three independent reasons, any one of which is sufficient:

1. **Geometry.** They were built on the fixed 12-plane cube retired by docs/58. Every subject now
   keeps its native `D` and `dz`, and the splat grid is `(D, 256, 256)` with
   `z_scale = Z_HALF_MM / dz`.
2. **The cohort itself changed.** On 2026-07-31 the sources were re-flipped to apex-at-z0,
   renamed (`Test_P012` → `CMRx24_Test_P012`) and re-laid-out (`sax/3d_recon/sax_frame_*.nii.gz`).
   The frozen input bundles these scored against were built 2026-07-12, before all of it.
3. **Every scored checkpoint is pre-refactor.** Not comparable to anything trained since.

The frozen input bundles themselves (264 GB) moved to
`scratch/eval/_archive_prenativez_20260712/`.

## One caveat when reading the ACDC numbers

`build_inputs/acdc.py` sampled 40 subjects from `ACDC/training/` (patient 001–008, 021–028,
041–048, 061–068, 081–088) — a stratified 8-per-pathology-group draw chosen back when **no ACDC
was in the training pool**. It has no relationship to any split. Now that ACDC is pooled,
**36 of those 40 are in pooled `train`**, so the archived ACDC scores are largely on seen
subjects. ACDC's official `testing/` (patient101–150) is 100% in pooled `test` and is the clean
ACDC cohort if you want one.
