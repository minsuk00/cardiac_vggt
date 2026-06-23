# 23 — OOD RTFB inference adapters: the `eval/` package refactor

> **TL;DR & takeaway** (2026-06-23, done, prove-it-reviewed). The OOD real-time-free-breathing
> (RTFB) inference code — OCMR, Göttingen, and now MIITT — was scattered as duplicated, ad-hoc
> helpers inside `tools/` scratch scripts, with each dataset adapted *differently* (OCMR a clean
> in-memory adapter; Göttingen faking a 12-phase CMRxRecon layout to reuse `MRIDataset`). This
> refactor consolidates all of it into a **permanent `eval/` package**: a `BaseRTFBAdapter` with 3
> per-dataset seams (`load` / `inplane_mm` / `slice_positions_mm`) over one shared canonical
> pipeline, plus shared `inference.py` (`load_rtfb_model` / `forward` / `phase_sweep`), `render.py`,
> and a single `run_rtfb.py` runner (`--dataset {ocmr,goettingen,miitt}`). **OCMR is provably
> bit-identical** (the lift is pure parameter-threading, guarded by a bit-exact test); **Göttingen
> is intentionally better** (real native slices, dropping the old 6→8 mm through-plane interpolation);
> **MIITT is new** (placeholder spacing, qualitative only). Adding a future RTFB dataset = one new
> `BaseRTFBAdapter` subclass, zero changes elsewhere. No training code or `MRIDataset` was touched.

This is an engineering/infrastructure write-up, not a research finding — recorded per the
`CLAUDE.md → Docs` convention because it's a non-trivial structural decision future agents will
build on.

## Why

The project's headline direction is gated→real-time transfer, validated qualitatively on OOD RTFB
cine (OCMR `docs/06`, Göttingen `docs/16`, MIITT — U-Michigan paired gated+RT, see project memory).
The "data → model batch" adapters and "batch → reconstructed volume" inference loops had grown as
copy-pasted code across ~8 `tools/` diagnostic scripts, and the two existing datasets were handled
asymmetrically for purely historical reasons:

- **OCMR** — a clean direct in-memory adapter (`tools/eval_ocmr_inference.py`): one random frame per
  slice from the full continuous cine → canonical `(256,256,12)@(1.4,1.4,8.0)` mm batch. This is the
  clinically-faithful scattered-single-frame regime.
- **Göttingen** — faked a 12-phase CMRxRecon layout (`goettingen_to_canonical.py` wrote placeholder
  `sax_frame_{00..11}.nii.gz`) purely to reuse `MRIDataset`. The "12 phases" were meaningless
  placeholders (the model is z-only for inputs, `use_t_pose_embedding=false`, so input phase is
  ignored), it threw away ~91% of frames, and `MRIDataset`'s `Spacingd` resampled Z **6→8 mm** —
  making each input slice a through-plane-interpolated blend rather than a real acquired slice.

The asymmetry was pragmatic, not principled. With MIITT arriving as a third RTFB dataset, a shared
base class crossed the "earns its keep" bar.

## What was built

New permanent package (a real package with `__init__.py`, imported as a library — unlike `tools/`):

```
eval/
  adapters/
    base.py        # BaseRTFBAdapter + the shared canonical pipeline (lifted VERBATIM from OCMR)
    ocmr.py        # OCMRAdapter        (geometry from meta.json)
    goettingen.py  # GoettingenAdapter  (geometry from the NIfTI affine; native recon, no interp)
    miitt.py       # MIITTAdapter       (geometry from affine; PLACEHOLDER spacing)
  inference.py     # load_rtfb_model / forward / phase_sweep   (dataset-agnostic)
  render.py        # save_cycle_gif / save_dvf_png / save_inputs_png / save_volume_png
  run_rtfb.py      # single-subject CLI runner: --dataset {ocmr,goettingen,miitt}
```

**The 3 seams.** Per-dataset variation is exactly: how you `load()` the 4-D cine `(frame,slice,H,W)`,
the in-plane mm, and the per-slice positions. Everything downstream (percentile-normalize, in-plane
resample to 1.4 mm + crop/pad to 256², one-random-frame-per-canonical-z sampling, 518 upsample,
`scanner_coords`) is shared in `base.py`. OCMR reads the seams from `meta.json`; Göttingen/MIITT
derive them from the NIfTI affine and synthesize an evenly-spaced slice stack.

**Inference layering** (the three modules people conflate):
- `inference.py` = batch → volume (the model math; `forward` = one `target_t` query, `phase_sweep` =
  loop over the 12 phases). Dataset-blind.
- `render.py` = volume → figures (GIF/PNG). No model.
- `run_rtfb.py` = the CLI glue calling adapter → inference → render. The diagnose/compare scripts skip
  it and import `inference`/`render` directly to draw their own figures.

**Migration (low-churn):**
- `tools/eval_ocmr_inference.py` became a **thin back-compat shim** re-exporting from `eval/`, so the
  8 consumer scripts' imports keep working unchanged.
- The 5 Göttingen comparison scripts (`diagnose_4way_refiner`, `three_ckpt_compare`,
  `bpblrai2_compare`, `five_row_compare`, `render_oldckpt_4way`) were rewired to build Göttingen via
  `GoettingenAdapter` instead of `MRIDataset` (their now-orphaned `OmegaConf`/`MRIDataset` imports
  removed). `five_row_compare` also gained a **MIITT panel** (5th OOD mode).
- Deleted: `scratch/data/goettingen/canonical_subjects/` (129 MB placeholder NIfTIs),
  `goettingen_to_canonical.py` (obsolete generator), `goettingen_infer.py` (superseded by
  `run_rtfb`). The real recon at `scratch/data/goettingen/recon/` is untouched.

## Key consequences / what changed numerically

- **OCMR: bit-identical.** The lift is pure parameter-threading — the only diff vs the original
  `build_batch` is `inplane` passed as an argument instead of `meta["inplane_mm"]`. Guarded by
  `tests/test_eval_ocmr_equivalence.py` (real-subject batch bit-exact, `atol=0`, against a frozen
  snapshot in `tests/_legacy_ocmr.py`) and an end-to-end check (`phase_sweep` produces `world_points`
  identical to the old `reconstruct_cycle`; the ~5e-7 volume delta is GPU scatter nondeterminism in
  the splat, identical when the *same* path is run twice).
- **Göttingen: intentionally different (better).** The direct adapter feeds the real native slices and
  drops the 6→8 mm through-plane interpolation, so its panels are sharper than the old placeholder
  path. Real-data sweep: 24 native 6 mm slices → 12 canonical planes (intended subsample). The
  re-rendered `five_row_Goett.png` reflects this.
- **MIITT: new, qualitative only.** Spacing is a PLACEHOLDER (2.6 mm in-plane, 8 mm slice — see
  `tools/convert_miitt_to_nifti.py`); `run_rtfb`/the adapter surface a loud warning. RT is low-res
  128², so expect softer images regardless of model. The MIITT 5-row panel **cross-validates doc 18**
  on a third independent dataset: separate-refiner (Row 3) preserves OOD, joint-refiner (Rows 4-5)
  shows the grid/checkerboard degradation — same pattern as OCMR/Göttingen.

## Verification

- `tests/` → **170 passed** (incl. the bit-exact OCMR guard).
- Batch-build sweep over **all 20 real subjects** (10 OCMR / 5 Göttingen / 5 MIITT): all build,
  S=9–12, float32, correct shapes — no z-map collapse, empty-stack, or 0-px edge cases on real data.
- `run_rtfb` produced beating-heart GIFs for all three datasets; full 5-row render (3 ckpts × 5 modes)
  completed.
- **prove-it review (4 reviewers + adversarial verification):** zero refactor-introduced bugs. OCMR
  bit-identity confirmed by AST diff + numeric probe; axis-order (`(X,Y,Z,T)→(T,Z,Y,X)`, `H=Y/sy`,
  `W=X/sx`) consistent; shim import-complete; rewires clean. One dead-code wart found and removed
  (`forward`'s `V_gt` branch hardcoded phase 0 and was unreachable). A few latent edge-case crash
  paths (empty `z_map`, sub-pixel slice) are **inherited verbatim from the original** and unreachable
  by any of the three datasets — left unchanged.

## Pointers

- Code: `eval/` (package), `tools/eval_ocmr_inference.py` (shim), the 5 rewired comparison scripts.
- Tests: `tests/test_eval_ocmr_equivalence.py`, `tests/_legacy_ocmr.py`.
- Figures: `result/4way_refiner/five_row_{val_ON,val_OFF,OCMR,Goett,MIITT}.png`.
- Data docs: OCMR `docs/06`, Göttingen `docs/16`; MIITT in project memory. The OOD-generalization
  finding these panels test is `docs/18`.
