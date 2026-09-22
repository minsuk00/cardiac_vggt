# CiNeVol baseline — what is the paper's, what is ours

Vogt et al., *Cardiac neural representations for ECG-guided slice-to-volume reconstruction*,
Med Biol Eng Comput 64:1505–1516 (2026), doi:10.1007/s11517-025-03488-7 (`paper/`).

`cinevol/` + `tests/` are a **reimplementation from the paper + supplement** (upstream
`github.com/sin021004/CiNeVol` @ `e30aecc`), copied verbatim — the authors' own code was not
available. Audit against the supplement (2026-09-17): architecture, all loss terms and weights, PSF
sampling, optimizer and the 500-step budget match; choices the paper leaves open (softplus output,
Adam betas/eps, Jacobian at pixel centres, 48-input correction MLP per supp. Table 1) are logged as
assumptions in `IMPLEMENTATION_PLAN.md` §7. **Nothing under `cinevol/` is modified here.**

## Kept exactly as published
Seven hash grids (Grid4D decomposition), cardiac + respiratory deformation MLPs, canonical intensity
MLP, slice scale / bias / correction heads, MAE + TV + bias + correction + two Jacobian losses with
the **in-vivo** weights (λ_Corr 0.2, λ_Jac^φ 1e-4), 500 AdamW steps, 2^15 pixels × 16 PSF samples,
lr 5e-3, MLP weight decay 1e-2, 16 PSF samples at inference, bias/correction heads dropped at inference.

## Deviations (all disclosed; the first three change what the model is fitted on or asked)
| # | Deviation | Why |
|---|---|---|
| 1 | **Single SAX stack.** The paper fits SAX + VLAX + HLAX. | Our cohorts have one orientation. Through-plane information is therefore weak (12 mm pitch). |
| 2 | **Training pixels = the bundle's padded heart mask** (`mask_heart_pad10`), not the paper's multi-stack intersection mask. | A single stack has no intersection; this is the ROI every other baseline reconstructs in, and the closest analogue. Done in `tools/cinevol_prepare.py` by bbox-crop + NaN outside (cinevol.prepare drops non-finite pixels). Outside the mask the exported volume is unconstrained extrapolation; the scorer's ROI ignores it. |
| 3 | **States come from a simulator.** Cardiac φ = the paper's Eq. 16 (Feinstein bilinear phase) applied to the simulated **R-peak times**; respiratory ψ = the true breath level in [0,1]. The heart's true cardiac position is never read. | The method requires both states as inputs (paper: ECG + belt; its own phantoms used ground-truth states). **ψ is nearly the answer here:** the simulator's displacement is `direction × amplitude × level` with direction and amplitude fixed per subject, so ψ fixes it up to one 3-vector per subject. Do not present CiNeVol's breathing correction as like-for-like with methods that get no breathing signal. |
| 4 | **Export = 24 queried volumes**, one per GT frame f: φ = the reference plane's own label at frame f, ψ = 0 (end-expiration), 1.4 mm isotropic over the mask bbox. The paper exported 20 evenly spaced phases. | Scoring needs each output aligned to one GT volume. Same labelling rule as the fit — a label from another rule addresses a different point on the learned phase axis. |
| 5 | **torch 2.13 / CUDA 13** (paper: 1.13.1 / 11.7), run in the project's `svr` env, nothing installed. | Evidence the math is unchanged: `tests/test_cuda.py` matches the Grid4D CUDA encoder to the pure-torch reference in values, first and second gradients. |
| 6 | **Grid4D `a8992a9` `hashencoder/` vendored with 3 one-line patches** (`third_party/Grid4D/`): `-std=c++14` → `c++17` (torch ≥ 2.1 headers), unused `from cachetools import cached` removed, build dir from `$GRID4D_BUILD_DIR`. CUDA source untouched. | Same commit the supplement cites. Only the 76 KB encoder is copied; no Gaussian-splatting code. |
| 7 | `--microbatch 8192` (upstream default 1024). | Not a hyperparameter: exact gradient accumulation of the same 32 768-pixel batch (`tests/test_core.py::test_chunking_preserves_global_bias_loss_and_gradients`); identical loss measured at 1024 and 8192, ~4× faster. |

## Known, measured, not fixed
- **Label ≠ true cardiac position** (`tools/cinevol_verify_labels.py`, all 180 test subjects, in gated
  phases of 12): `regular24` 0.000; `hrv24` mean 0.21 / p95 0.66 (the simulator stretches the whole
  beat, Eq. 16 assumes near-fixed systole — two heuristics disagreeing, not arrhythmia); `af24` mean
  1.23 / p95 4.10 / max 6.0. This gap is what the AF arm measures; it is not repaired with the truth.
- Phase 0 and 1 are not tied together (no periodic wrap) — as in the paper.
