# ES-surplus root-cause diagnostics (docs/92 repro)

Scripts behind the docs/92 finding (EF bias = real ED under-expansion + a mostly-artifactual
ES surplus from the half-rendered wall band). All run from the repo root with the `svr` env,
read-only against `temp/dice_ef_sweep/mirror_full/`, and write only new files under
`temp/dice_ef_sweep/` / `temp/ef_fix/`.

- `es_diag_intensity.py` — pixel-space ES tests (annulus brightness, boundary sharpness,
  threshold-swept pseudo-ESV, effective contraction fraction) → `es_diag_noreg.csv`.
- `es_diag_alignment_edv.py` — pseudo-EDV + alignment controls (blood-pool Dice, CoM shift)
  → `es_diag2_noreg.csv`.
- `blur_test_dump.py` — dumps the blur-causal-test nnU-Net inputs (ctrl / pred-matched σ /
  σ=1.0 GT ES frames); segment with `evaluation/src/engine/run_seg.sh`, then score LV volumes.
- `render_walkthrough_figs.py` — the numbered evidence figures 01–06 → `temp/ef_fix/`.

The pred-frame segmentation, per-voxel surplus decomposition, sharpening oracle, and 07 example
panels were run as session heredocs (recipes in docs/92 §3–§5); their outputs live in
`temp/dice_ef_sweep/es_blurtest/`.
