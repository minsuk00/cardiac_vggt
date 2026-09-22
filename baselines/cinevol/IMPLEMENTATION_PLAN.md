# CiNeVol baseline implementation plan

> **ACDC update (2026-09-15):** The user approved a single-SAX ACDC adaptation with supplied simulated respiratory states. See [ACDC_RUN_COMMANDS.md](ACDC_RUN_COMMANDS.md). This supersedes the native-only input restriction in the historical sections below. Both motion branches and published fitting settings remain active. Native multistack preparation remains supported.


Updated 2026-09-14. The implementation is now present; see README.md, RUN_COMMANDS.md, and HANDOFF.md for verified status. This document records the original implementation specification and paper assumptions.

## 1. Scope and deliverable

Implement **CiNeVol, the “Proposed” model without segmentation supervision**, from Vogt et al., *Cardiac neural representations for ECG-guided slice-to-volume reconstruction*. The user requests only the implementation needed for their research comparison, using published parameters. Deliver a reproducible baseline under `/home/sincheol/CMRI/CiNeVol`, with a central `CiNeVol_Workbench.ipynb` and reusable Python modules/CLI scripts.

CiNeVol fits one dynamic neural representation **per subject**, using that subject's observed slice intensities. It does not train a population model to regress LV centres, and it does not need a paired high-resolution target for its reconstruction objective. The paper's optional “Proposed+seg” variant adds segmentation supervision; it is unnecessary for the initial baseline. There is no Dangi-style atlas segmentation stage to implement.

Native inputs are multiple SAX/VLAX/HLAX cine acquisitions, spatial acquisition geometry, and cardiac and respiratory states for each acquired frame. Native output is a continuous dynamic 3D intensity representation, queried at chosen cardiac and respiratory states. Physiology states are provided or derived before fitting; the method does not infer them from a single ungated stack by itself.

This is a reimplementation based on the article, its four-page supplement, NeSVoR, and the cited Grid4D encoder. The article says author code will be made available; no author CiNeVol repository was located during this review. Do not describe this implementation as released author code or claim numerical reproduction before validation.

## 2. Fixed scope for our research paper

The supplied introduction explicitly marks CiNeVol with a dagger: it receives dense temporal input and is evaluated using its native full-cine reconstruction. Follow that protocol. Our method receives one asynchronous SAX frame per slice and a mid-ventricular reference, while CiNeVol receives the full-cine observations and physiological states needed by its formulation. This difference must remain visible in the comparison.

**Implement one primary baseline: full CiNeVol without segmentation supervision.** Preserve all seven grids, both cardiac/respiratory deformation branches, intensity/bias/correction heads, the acquisition PSF, and published objective. Do not build a sparse one-frame-per-slice CiNeVol variant or a cardiac-only variant. Do not add phase estimation from images, a reference-conditioned network, population pretraining, extra rigid-registration optimization, or a new loss. The reference identifies a query state for evaluation; it is not an additional CiNeVol network input.

For each benchmark subject:

1. Obtain that subject's permitted **motion-corrupted full-cine observations**, native multiplanar acquisition geometry where available, and cardiac/respiratory states. Clean source volumes used by a simulator are not reconstruction inputs.
2. Fit CiNeVol independently for 500 optimizer steps using the published settings in §3. The target image and its segmentation are evaluation-only.
3. Reconstruct the full cardiac cycle at fixed end-expiration: 20 equally spaced cardiac states, following the paper's full-cine output convention.
4. Also query **the exact cardiac state corresponding to our selected mid-ventricular reference**, at the same end-expiration target. This gives a matched 3D output for the before/output/ground-truth comparison. If the state lies between the 20 exported phases, query the continuous model directly rather than selecting the best-looking or closest-performing frame.
5. Report end-to-end per-subject runtime for the full-cine pipeline as required by the table. Record the additional matched-state query separately and disclose whether it is included in the total.

For the simulated benchmark, supply the simulator's cardiac and respiratory states to CiNeVol and record this additional information. This follows the paper's use of known states for simulated data. Our method continues to operate without supplied states. Map the reference's simulator state to CiNeVol's cardiac coordinate without inspecting the ground-truth volume. Map end-expiration explicitly; the article's in-vivo ψ=0.5 does not establish a universal end-expiration convention.

**Data dependency, not an alternative model:** the table resolves full-cine versus sparse input, but the attached introduction does not identify the full-cine benchmark files, their views, or state metadata. ACDC exists at `/home/sincheol/CMRI/Dangi/ACDC/database`; it provides gated SAX cine, not native multiplanar free-breathing acquisitions or measured respiration. Resliced LAX views from thick SAX data are not independent acquisitions. If the benchmark supplies only SAX cine, document the acquisition adaptation before reporting results; do not call that native multiplanar reproduction.

The supplement uses geometric coverage by more than one distinct acquisition stack to select training pixels. Native preprocessing will enforce this. A single SAX stack cannot satisfy it, and its repeated frames must not be counted as separate stacks. Replacing this mask with single-stack support would be an explicit benchmark deviation, not a hidden default. The geometric intersection is compatible with the table's “no external mask” claim because it does not require anatomical segmentation supervision.

Out of scope: reproducing all CiNeVol experiments or tables, XCAT/MRXCAT dataset recreation, volunteer acquisition, Proposed+seg, the simplified INR comparator, additional NeSVoR/SVRTK baselines, and research ablation studies. Small mathematical fixtures remain necessary to verify this implementation.

## 3. Published architecture and optimization settings

These settings are specified in Supplemental Material 1, §1.1.1 and Table 1. They are also recorded in `configs/paper_defaults.json`. All published values are fixed: no learning-rate search, increased fitting budget, reduced effective batch, added scheduler, or altered regularization weights. Select the published phantom profile for phantom images and the in-vivo profile for real MRI intensities. Using the in-vivo profile on synthetically corrupted real MRI is our documented dataset-to-profile choice, not a separately specified experiment in the article.

| Component | Specification |
|---|---|
| Spatial encoding | One xyz hash grid; 16 levels; 2 features/level; base 16³; finest 2048³; maximum hash table size 2¹⁹ per level |
| Cardiac encoding | Three grids: xyφ, xzφ, yzφ |
| Respiratory encoding | Three grids: xyψ, xzψ, yzψ |
| Each spatiotemporal grid | 32 levels; 2 features/level; base 8×8×8; finest **32×32×16**; maximum table size 2¹⁹ per level |
| Cardiac and respiratory MLPs | Each 224 → 64 ReLU → 3 |
| Intensity MLP | 32 → 64 ReLU → 17: intensity plus 16 intermediate features |
| Slice/frame embeddings | 16 features each |
| Bias MLP | 20 → 64 ReLU → 1; four spatial features from the two coarsest levels plus slice embedding |
| Additive correction MLP | 48 → 64 ReLU → 1; intermediate features plus slice/frame embeddings; see §7 |
| Optimization | **500 AdamW steps per subject**, batch **32,768 observed pixels**, learning rate **0.005** |
| Weight decay | **0.01 for MLP parameters** |
| PSF sampling | **16 samples per pixel** in fitting and inference |
| Phantom loss weights | TV 0.01; bias 0.1; correction 0.5; cardiac Jacobian 0.01; respiratory Jacobian 0.1 |
| In-vivo loss changes | Correction 0.2; cardiac Jacobian 0.0001; other weights unchanged |
| No-segmentation scope | Segmentation and segmentation-based motion penalties disabled |
| Breath-hold export | 20 cardiac states; fixed respiration; phantom ψ=0 and in-vivo ψ=0.5 in the paper |

One full optimizer step evaluates 524,288 PSF samples. The published budget is 16,384,000 observed-pixel presentations per subject, not 500 dataset epochs. There is no Dangi-style “steps per epoch” requirement. Do not accidentally inherit NeSVoR's comparator settings of 1,000 steps and 128 PSF samples.

The supplement reports Python 3.10.14, PyTorch 1.13.1, and an A100 40 GB GPU with CUDA driver version 12.2. The paper reports roughly 500 seconds per reconstruction in its experiments. This is a reference measurement, not a runtime guarantee for our code or GPUs.

## 4. Model and forward acquisition operator

Use physical coordinates in millimetres as the external geometry contract, with one explicitly stored normalization for hash inputs.

For a pixel at x, cardiac state φ, and respiratory state ψ:

1. Encode x using the shared spatial grid, and encode (xyφ, xzφ, yzφ) and (xyψ, xzψ, yzψ) separately.
2. Predict δφ(x,φ) and δψ(x,ψ), each from 32 spatial plus 192 state-dependent features.
3. Compute **x′ = x + δφ + δψ**. Both branches see the original x; the paper adds their outputs rather than composing sequential warps.
4. Reuse the same spatial hash grid to encode x′. The intensity MLP predicts uncorrected intensity î and features f.
5. Predict log-bias b from the coarsest spatial features at x′ and the slice embedding. Predict additive correction c from f and slice/frame embeddings. Fit one intensity scale s per physical slice acquisition, shared across its frames.
6. Form the sample prediction **s × exp(b) × î + c**.
7. Average predictions over the 16 spatial PSF samples, then compare with the observed pixel intensity.

Training PSF: a Gaussian with local FWHM `(1.2 × pixel_spacing_x, 1.2 × pixel_spacing_y, slice_thickness)`. Convert FWHM to standard deviation using `FWHM / sqrt(8 log(2))`, rotate local samples into world space using the acquisition orientation, and then evaluate the motion model. Use slice thickness, not automatically slice-centre spacing. Keep the same physiological states for all samples of one pixel.

At inference, query the full motion-plus-intensity model at the requested φ and ψ, using 16 samples from an isotropic Gaussian. Return the uncorrected intensity average; omit acquisition scale, bias and correction. Preserve both motion branches in the exported model. Calling only NeSVoR's static INR would lose the dynamic reconstruction.

The learned map goes from a queried state to canonical coordinates. Dense inference uses this map directly; do not negate predicted offsets as an assumed inverse warp. The paper does not specify that any particular physiological state must have exactly zero displacement.

## 5. Loss implementation

Implement Eqs. (10)–(15), logging every unweighted and weighted contribution. Let B be the observed-pixel batch, U=16, and pair samples u with u+U/2.

- **Data:** mean absolute error between the observed pixel and the PSF-averaged corrected prediction. Upstream NeSVoR's uncertainty-weighted squared error is not the CiNeVol objective.
- **Intensity TV:** sum `abs(î_u − î_pair) / ||x′_u − x′_pair||` over B pixels and U/2 pairs, divided by **B×U**. Use uncorrected sample intensities. An ordinary mean over the half-pairs doubles the specified weight.
- **Correction:** mean absolute correction, with the sample aggregation choice recorded in §7.
- **Bias:** squared mean absolute log-bias plus the corresponding half-pair spatial smoothness term. Preserve `(mean(abs(b)))²`; NeSVoR's `(mean(b))²` is different.
- **Motion:** separate cardiac and respiratory Jacobian penalties, `mean(||JᵀJ − I||²_F)`, at fixed physiological state. Interpret J as the derivative of the full branch map `x + δ(x,state)` with respect to physical x, consistent with the rigidity penalty and NeSVoR implementation. A zero-displacement field must have zero penalty. This interpretation and sampling convention will be documented because the article does not spell them out fully.

Autodifferentiation must include the spatial dependence of both the xyz encoding and state-dependent encodings. The Jacobian loss must backpropagate to motion MLPs and hash tables. Test the encoder's higher-order derivatives rather than assuming a successful forward pass is sufficient. Keep geometry and Jacobian calculations in float32 initially; fail visibly on non-finite losses.

Memory chunking must preserve the mathematical batch. In particular, `(mean(abs(b)))²` couples chunks; separately squaring each chunk mean changes the loss. Use a differentiable global reduction, recomputation, or an equivalent verified gradient strategy. Maintain U/2 pairs within each pixel.

## 6. Data and physiological states

Create a subject manifest containing image paths, acquisition/stack IDs, slice IDs, frame IDs, per-frame timestamps, voxel-to-world transforms, pixel spacing, slice thickness, observed intensities, valid/coverage masks, φ, ψ, and provenance of both state variables. Permit different frame counts across slice acquisitions. Store output-grid geometry and target states separately from observations and evaluation references.

Geometry checks must include oblique planes, axis ordering, RAS/LPS conventions, qform/sform consistency, and slice thickness versus spacing. Preserve acquisition geometry before any resampling. Do not reuse Dangi's 192×192 preprocessing by default.

The initial benchmark adapter consumes supplied simulation states directly. Raw ECG/belt processing is outside the required implementation unless the chosen benchmark actually supplies these signals. For reference, the paper's bilinear cardiac normalization is as follows. With current RR interval r, `HR=60/r`, systolic duration `s=(546−2.1×HR)/1000` seconds, diastolic duration `d=r−s`, and elapsed time τ since the current QRS:

- During systole: `φ = (τ/s) × (mean(s)/mean(r))`.
- During diastole: `φ = ((τ−s)/d) × (mean(d)/mean(r)) + mean(s)/mean(r)`.

If a recorded-signal adapter becomes necessary, synchronize acquisition and physiological timestamps, validate intervals, and document the averaging scope. ACDC's frame index divided by frame count is only a gated-cine phase proxy; it does not reproduce the variable-RR procedure.

For recorded respiration, the paper uses a Butterworth high-pass at 0.1 Hz and low-pass at 3 Hz. ECG-derived respiration uses 0.15 Hz high-pass and 0.25–0.5 Hz low-pass filtering. Respiration input is **normalized signal amplitude**, not a cyclic angle. Filter order, selected ECG low-pass cutoff, and edge handling require explicit choices if this adapter is used.

For simulations, use the generator's known states and record that oracle state information is supplied. End-expiration must be mapped using the signal/generator convention. Do not assume ψ=0.5 universally means end-expiration just because it was used for the paper's in-vivo export.

## 7. Assumptions and source differences to record

These are implementation decisions, not additional claims about the paper.

| Issue | Planned default / resolution |
|---|---|
| Correction input | Main Eq. (8) also lists spatial encoding, implying 80 inputs; supplement Table 1 specifies 48, consistent with f+slice+frame. Use **48** as the primary configuration and record this discrepancy. |
| Encoder interpolation | Use the cited Grid4D implementation's smoothstep behavior and anisotropic level schedule. Do not substitute an isotropic grid or trilinear interpolation silently. |
| Coordinate scaling | Store an isotropic physical normalization and invert it for offsets/Jacobians. Map normalized encoder inputs deliberately: Grid4D's public wrapper expects coordinates in `[-size,size]` and maps internally to `[0,1]`. Do not normalize twice or leave phases in a different range accidentally. |
| Optimizer extras | Start with NeSVoR's betas (0.9,0.99) and epsilon 1e-15, explicitly marked as inherited. Set MLP decay 0.01 and other parameter-group decay explicitly to 0; non-MLP decay is unspecified in CiNeVol. |
| LR schedule | Constant 0.005 unless author evidence supplies a schedule; do not silently inherit NeSVoR's MultiStepLR. |
| Activations and initialization | Provisional NeSVoR-style softplus intensity, signed linear motion/bias/correction, small initial motion, and positive mean-one slice scales. Exact output activations and initialization are not supplied by CiNeVol. Record the resolved choices. |
| Intensity normalization | Fit a documented normalization using observed input only; no normalization using the target volume. Keep published regularization weights fixed and record this normalization. |
| Bias/correction aggregation | Provisional mean absolute value over all PSF samples for zero-centering penalties; the equations omit the sample dimension here. Preserve half-pair normalization for spatial terms. |
| Jacobian evaluation | Provisional evaluation at observed pixel centres for each branch, with full-map physical derivatives; sample count/location are not specified. |
| Frame embeddings | Use local temporal index t shared across slice acquisitions, with separate unique slice IDs; handle variable T explicitly. Do not replace frame index with ECG phase or unique slice-frame IDs silently. |
| Boundary behavior | Record bbox padding, out-of-domain encoding, denominator epsilon and inference PSF width. Provisional isotropic inference FWHM equals output voxel spacing; update only if author evidence resolves this choice. |
| Cardiac cycle export | Use 20 uniformly spaced phases with endpoint excluded to avoid duplicate 0/1 frames; endpoint inclusion is unspecified. |

Unspecified details above cannot be represented as published parameters. Use the cited implementation where it resolves a choice; otherwise record the minimal necessary assumption. Do not add ablation runs to resolve these choices as part of this task. Save all resolved settings, input adaptations and dependency revisions with each run. If author CiNeVol code becomes available, compare these choices before claiming close numerical reproduction.

## 8. Reuse NeSVoR and Grid4D selectively

Reviewed NeSVoR **v0.5.0**, commit `730ddaa3711a2304386de34193ea4b957892fe7b`, and Grid4D commit **`a8992a9bd18b1828d2890a190421bca8bf0d2e80`**, as cited by the supplement.

| Source component | Implementation action |
|---|---|
| NeSVoR image/transform/PSF utilities | Reuse or adapt selected geometry and sampling concepts, with CiNeVol's exact PSF factor and explicit tests |
| NeSVoR `inr/models.py` | Reuse small network/scaling patterns; implement a new CiNeVol model |
| NeSVoR `sigma_net` | It predicts log variance, not additive correction. Add a new correction head and objective. |
| NeSVoR deformation and rigid registration | Replace slice-conditioned deformation with two physiological-state branches; do not enable trainable rigid slice transforms by default |
| NeSVoR `inr/data.py`, `train.py` | Extend the data contract and write a per-subject fitting loop; save the complete dynamic model rather than only `model.inr` |
| NeSVoR `inr/sample.py` | Adapt chunked dense sampling to explicit cardiac/respiratory state queries |
| Grid4D `hashencoder/` | Use the cited anisotropic encoder through an adapter; its code includes custom second backward support, which still needs numerical validation |
| Grid4D scene/Gaussian rendering | Unnecessary; CiNeVol uses intensity fields and an MRI acquisition operator |

NeSVoR's scalar-resolution PyTorch fallback cannot directly reproduce the required 32×32×16 time grids. Build a small independently differentiable reference encoder for correctness tests if needed; the CUDA encoder should be checked against it, including smoothstep interpolation and dense-table allocation at low resolutions.

Keep NeSVoR's MIT notices for reused code. No top-level license for the pinned Grid4D encoder was located in the inspected checkout; initially reference it as an external pinned dependency and resolve its provenance before redistributing its source. Avoid importing the unrelated Gaussian-rendering dependency tree.

Use a dedicated `cinevol` Python 3.10 environment. Start compatibility work with PyTorch 1.13.1 and an appropriate CUDA toolkit (NeSVoR's environment uses CUDA 11.7). A driver reporting 12.2 is not a requirement to compile with toolkit 12.2. Validate the selected combination on the allocated GPU before freezing dependencies. Keep the existing Dangi environment intact.

GPU jobs use account `jjparkcv98`; prefer the available `spgpu` allocation for initial memory profiling:

```bash
salloc --account=jjparkcv98 --partition=spgpu --gres=gpu:1 --mem=48G --cpus-per-task=4 --time=01:30:00
```

Determine available CUDA/compiler modules on the compute node before writing the final setup recipe. GPU build, derivative tests, and a full-size optimizer step were verified on a Tesla V100 16 GB with PyTorch 1.13.1+cu117 and CUDA toolkit 11.7.1. This is software verification, not a patient benchmark result.

## 9. Repository layout and central notebook

```text
CiNeVol/
  CiNeVol.pdf
  IMPLEMENTATION_PLAN.md
  CiNeVol_Workbench.ipynb
  README.md
  pyproject.toml
  environment.yml
  configs/
    paper_defaults.json  # published common settings and phantom/in-vivo weights
    benchmark.json       # paths, acquisition metadata, state mapping, output grid
  cinevol/
    data.py           # subject manifest, acquisition records, pixel sampling
    geometry.py       # physical coordinates, coverage, output grids
    states.py         # supplied benchmark states and reference/end-expiration mapping
    encodings.py      # seven grids and backend adapter
    model.py          # two deformations, intensity, bias, correction
    psf.py
    losses.py
    fit.py
    reconstruct.py
    evaluate.py
    workbench.py      # notebook orchestration and plotting
  scripts/
    prepare_data.py
    fit_subject.py
    reconstruct.py
    evaluate.py
    fit.slurm
    notebook.sh
  tests/
  runs/               # ignored: checkpoints, logs, visualizations, NIfTI
```

The notebook should contain only important controls and outputs: environment/kernel check; benchmark subject selection; geometry/phase preview; configuration summary; launch/resume fitting; total and component loss curves; reconstruction at chosen states; matched SAX and two LAX before/output/reference views; cine animations; metrics; artifact links. Heavy fitting and reusable plotting remain in scripts/modules. Provide a dedicated `Python (CiNeVol)` kernel.

Before/after/reference images must share a physical grid, orientation, target state, crop and display range. Label the input rendering as an interpolation of the observed stack when appropriate. Show a ground-truth panel only when an actual target exists; otherwise show a clearly named reference. Native real-time data do not automatically provide motion-free voxelwise ground truth.

## 10. Implementation order and acceptance checks

1. **Freeze the benchmark interface and paper defaults.** Keep published parameters in the versioned configuration. Require subject observation paths, actual acquisition geometry, supplied state provenance, reference cardiac state, end-expiration mapping and the shared evaluation grid. Verify non-empty native multistack coverage. A missing benchmark manifest does not prevent model implementation, but it prevents a valid benchmark run.
2. **Establish the environment and encoder backend.** Check anisotropic resolution endpoints and feature dimensions. Compare tiny CPU-reference/CUDA forward values, coordinate gradients and Jacobian-loss gradients to MLP/hash parameters. Confirm gradients through the spatial encoder evaluated at deformed coordinates before fitting a full case.
3. **Implement the model, acquisition operator and losses.** Test identity, translation, rotation and scaling maps; oriented PSFs; physical-unit invariance; exact reductions; and gradients through both branches. Verify inference omits nuisance intensity terms while retaining dynamic motion. Use a small analytic fixture as a correctness check; no separate phantom research pipeline is required.
4. **Fit one benchmark subject at the published budget.** Use 500 steps, batch 32,768 and 16 PSF samples. Profile memory/runtime; chunk computation only with equivalent batch loss and gradients. Save final-step weights, configuration, dependency commits, input hashes, normalization, geometry, states, optimizer/scaler and RNG state. Verify checkpoint reload reproduces the output within the numerical tolerance.
5. **Complete the notebook and exports.** Provide total/component fitting losses, the 20-frame end-expiration cine, the exact reference-state 3D output, matched SAX/LAX before/output/ground-truth panels, animations and metric summaries. Export NIfTI plus state metadata; use `(X,Y,Z,cardiac_time)` for the full cine and a separate 3D NIfTI for the reference-state result.
6. **Run the paper comparison.** Fit each subject independently from its permitted observations with frozen published settings. Report step-500 outputs. Do not select checkpoints using target PSNR or add an early-stopping protocol. Use the existing benchmark metrics and subject split; do not build an unrelated evaluation suite.

Per-subject fitting curves are not the same as Dangi population train/validation curves. Log fitting losses by optimization step; a held-out validation split is not required by the published procedure and will not be introduced into the primary run. Separate setup, preprocessing, optimization and export time, and record peak GPU memory.

## 11. Evaluation and interpretation

For synthetic data with known targets, report PSNR, SSIM and NMSE per 3D frame and average across the selected states. Specify metric formulas/data range, physical grid, intensity convention, masks and aggregation. Include before/output/target and error maps at the target state.

For our single-target comparison, evaluate the exact reference-state output on the same target grid, mask and intensity convention used for the other methods. Do not add target-driven registration or a new SVRTK-derived evaluation mask to this implementation. The original CiNeVol paper used temporal-mean rigid registration and an SVRTK-derived ROI; our benchmark evaluation is a different, explicitly documented protocol. If the shared benchmark itself prescribes a registration step, apply that same rule to all methods.

Reconstruction fitting must not consume the clean target volume, target segmentation or ground-truth deformation. Optional downstream LV-volume/segmentation metrics should use the same evaluation segmenter for every method and should not enable CiNeVol segmentation supervision by accident. Dense canonical offsets are not directly comparable to an injected motion field without reconciling reference space and map direction.

The paper's quantitative phantom experiments used XCAT/MRXCAT and its real-time experiments used three volunteers with physiological recordings. Neither dataset was established as available in this directory. ACDC adaptation or an analytic phantom validates an implementation but cannot by itself reproduce the paper's reported metrics.

The implementation scope is settled: **full-cine CiNeVol without segmentation supervision, using published parameters**. The remaining integration input is the benchmark manifest that supplies its actual acquisitions and physiological-state metadata. The introduction's dagger already establishes that CiNeVol receives dense temporal input; it does not authorize inventing missing acquisition views or state measurements.

## Sources reviewed

- Local main article: `CiNeVol.pdf`, all 12 pages, including architecture and equation figures. [Publisher article](https://link.springer.com/article/10.1007/s11517-025-03488-7).
- [Supplemental Material 1](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs11517-025-03488-7/MediaObjects/11517_2025_3488_MOESM1_ESM.pdf), all four pages; exact model/configuration details and dependency citations.
- [NeSVoR v0.5.0](https://github.com/daviddmc/NeSVoR/tree/v0.5.0), especially `nesvor/inr/{models,train,data,sample,hash_grid_torch}.py`, PSF utilities, environment and license.
- [Grid4D at the cited commit](https://github.com/JiaweiXu8/Grid4D/tree/a8992a9bd18b1828d2890a190421bca8bf0d2e80), especially `hashencoder/hashgrid.py`, CUDA backend, and `scene/network.py`.
- Existing Dangi handoff and research context for data availability, notebook workflow and GPU account; historical handoff statements are not evidence of current CiNeVol implementation status.

Research scope source: user-supplied introduction at `/home/sincheol/.codex/attachments/98690601-a290-45ef-84f5-8c9ae662d5fe/pasted-text.txt`, especially Table 1 and its full-cine dagger footnote.
