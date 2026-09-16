# Handoff: Dangi et al. (2018) baseline implementation

> Implementation update (2026-09-06): The user clarified that the deliverable is
> Dangi's Stage A code/model using the documented assumptions, for integration
> into their own comparison. `dangi/`, `scripts/train.slurm`, and `tests/` now
> implement this scope. See `README.md` for usage and explicit choices. The
> original planning notes below are historical; simulation and Stage B are not
> implemented. CPU and GPU verification passed; `runs/smoke` contains only
> smoke-test weights, not a trained comparison baseline. Full training has not
> been launched. GPU jobs must use account `jjparkcv98`.
> The central user workflow is now `Dangi_Workbench.ipynb`; notebook orchestration
> and plotting helpers are in `dangi/workbench.py`. Start it with
> `bash scripts/notebook.sh` or select the project `.venv` in an existing notebook UI.

## 1. Context

We are implementing **Dangi, Linte & Yaniv, "Cine Cardiac MRI Slice Misalignment
Correction Towards Full 3D Left Ventricle Segmentation," SPIE 2018** (PDF in this
directory) as a **baseline** for an ICLR 2027 submission on *reference-conditioned
cardiac volume reconstruction* from free-breathing, ungated short-axis CMR.

Our paper recovers a coherent 3D cardiac volume from an asynchronous SAX stack
corrupted by (a) respiratory motion and (b) cardiac-state mismatch, with no ECG
gating and no retrospective phase synchronization. Dangi is the classical
slice-alignment baseline we compare against.

Nothing has been implemented yet. This directory contains only the PDF and the data.

## 2. What the Dangi method is

Two independent stages.

**Stage A — slice misalignment correction (this is the baseline we need).**
A per-slice CNN regresses the LV blood-pool centre `(x, y)` from each 2D SAX slice.
Each slice is then translated so its predicted centre lands on the image centre,
making all centres collinear on a vertical axis. It is a rigid **in-plane
translation only** — one 2-vector per slice.

**Stage B — 3D atlas + graph-cut LV segmentation (probably out of scope).**
Runs on the corrected volume: affine atlas registration with exhaustive NCC
initialisation (Alg. 1), 3D 6-connected graph-cut with atlas-prior energy
(Eq. 2-3), IoU per-slice cleanup (Alg. 2), iterative re-registration, and
Stochastic Outlier Selection slice pruning (Alg. 3).

### Recommendation on scope
Build **Stage A only**. Our paper's comparison is about *motion correction*;
Stage B is downstream segmentation that does not bear on our claims and would
produce no table entry. Stage A is ~400 lines with no new dependencies. Stage B is
~1200 lines, needs SimpleITK + scikit-image + PyMaxflow, and the paper
underspecifies exactly the parts you must get exact (sign convention and t-link
mapping in Eq. 2, the GMM refit bootstrap, the masked distance-map registration
mask in Sec. 2.5.5). Add it later only if a reviewer asks for segmentation numbers.

## 3. The CNN architecture (Fig. 1, reverse-engineered and verified)

Two 3x3 conv+ReLU per resolution, max-pool 2x2 between levels, and at each of the
first four levels a **1-channel side conv** that is flattened straight into the
concatenation layer.

| Level | Res  | Channels | Flattened side branch |
|-------|------|----------|-----------------------|
| 1     | 192² | 4, 4     | 36 864                |
| 2     | 96²  | 8, 8     | 9 216                 |
| 3     | 48²  | 12, 12   | 2 304                 |
| 4     | 24²  | 16, 16   | 576                   |
| 5     | 12²  | 20, 20   | 2 880 (full 20-ch flatten, no side conv) |

36864 + 9216 + 2304 + 576 + 2880 = **51 840**, which matches the "FC1 / Concatenate
51840" label in Fig. 1 — this confirms the reading. Then Dense 256 (`FC2`) ->
Dense 2 = `(x, y)`. About 13.3 M params, essentially all in the first dense layer.

Training per the paper: sum-of-squared-difference loss to the GT centre,
Xavier-uniform init, 100 epochs, keep the best-validation checkpoint.
The optimizer is **not stated** in the paper — use Adam 1e-3 and mark it as an
assumption in a comment.

## 4. Data: ACDC

Located at `/home/sincheol/CMRI/Dangi/ACDC/database/`.

```
training/patient001 ... patient100     (100 patients)
testing/patient101 ... patient150      (50 patients)
```

Per patient:
```
Info.cfg                       # ED:, ES:, Group:, Height:, NbFrame:, Weight:
patientXXX_4d.nii.gz           # (X, Y, Z, T) full cine, all ~30 frames, NO labels
patientXXX_frameNN.nii.gz      # ED frame image
patientXXX_frameNN_gt.nii.gz   # ED frame labels
patientXXX_frameMM.nii.gz      # ES frame image
patientXXX_frameMM_gt.nii.gz   # ES frame labels
```

**Labels: 0 = background, 1 = RV cavity, 2 = LV myocardium, 3 = LV blood pool.**
Both `training/` and `testing/` include ground truth.

Geometry survey over the 100 training patients:
- In-plane spacing 0.7031-1.9196 mm, **mode = 1.5625 mm** (27 cases).
  This matches the paper's stated range (0.7031-2.0833) and its resample target
  almost exactly, so the paper's preprocessing transfers directly.
- Slice thickness: 10 mm (85 cases), 5 mm (12), plus a few at 6.5/7 mm.
- 6-18 slices per stack (mode 10). Typical matrix 216 x 256.

### Two consequences that simplify or complicate the work

**Simplification: skip Sec. 2.1.1 entirely.** Dangi had only myocardium masks, so
they had to recover the blood pool by inverting the myocardium, morphological
opening, and picking the smaller connected component. ACDC gives the LV blood pool
directly as label 3. The GT centre is just the centroid of `label == 3` per slice.
That removes a whole fiddly component and a class of bug.

**Complication: ACDC labels only 2 frames per patient (ED and ES).** Dangi trained
on all cardiac phases (~24k slices). Here labelled slices are only
~100 patients x 2 frames x ~10 slices = **~2000**. With a 13.3 M-param dense layer
this is a real overfitting risk. Mitigations: rely on the 189-fold augmentation,
optionally add dropout before FC2, and watch the validation curve closely. If it
overfits badly, note it in the paper — it is a fair limitation of the baseline
under this data regime, not something to hide.

## 5. How to actually solve the project problem

**Critical point: ACDC is breath-held and ECG-gated, so there is no real
misalignment in it to correct.** Dangi's own evaluation defines "initial
misalignment" as the distance of each *true* centre to the *mean of true centres*,
which measures LV-axis obliquity, not acquisition motion. That metric is
self-referential and is not what our paper is about.

So the baseline must be run on **simulated corruption**, matching our paper's
"simulated free-breathing, ungated conditions". The 4D volume is exactly what makes
this possible:

1. **Clean target volume**: take the ED (or chosen reference) frame stack.
2. **Cardiac-state mismatch**: for each slice `z`, sample a *different* frame
   `t(z)` from `patientXXX_4d.nii.gz`. Now each slice shows a different phase.
3. **Respiratory misalignment**: apply a known random in-plane shift (and
   optionally a through-plane shift) to each slice. **Record these shifts.**
4. Run Dangi Stage A on the corrupted stack.
5. **Evaluate two ways:**
   - *Paper-faithful metric*, to reproduce Table 1 (mean/std/median in pixels,
     KS test, Figs. 4-5) and show we implemented it correctly.
   - *Our metric*: recovered shift vs. the known injected shift. This is the number
     that goes in our comparison table.

**Expected result, and why it is fair.** Dangi applies a rigid in-plane translation
per slice. It structurally *cannot* address cardiac-state mismatch or through-plane
motion — two of the three corruptions in our Figure 1. It should therefore partially
correct (2) and fail on (3)/(4). That is an informative baseline, not a strawman,
and the write-up should say so explicitly rather than letting the numbers imply the
method is bad in general.

## 6. Implementation plan

```
dangi/
  config.py       all paper constants in one dataclass
  data.py         ACDC loader -> resample 1.5625mm -> crop/pad 192² -> GT centres
  corrupt.py      simulate cardiac-state mismatch + respiratory shifts (Sec. 5)
  augment.py      the 189-combination grid (9 trans x 7 rot x 3 scale)
  model.py        the CNN in Sec. 3   (~60 lines)
  train.py        SSD loss, Xavier init, 100 epochs, best-val checkpoint
  align.py        predict centres -> translate to image centre -> corrected volume
  evaluate.py     paper metrics (Table 1, Figs. 4-5) + injected-shift recovery
```

Suggested build order: `model.py` -> `data.py` -> `augment.py` -> `train.py` ->
`corrupt.py` -> `align.py` -> `evaluate.py`.

Details fixed by the paper:
- **Preprocess** (Sec. 2.2): resample to 1.5625 mm in-plane, centre-crop/zero-pad to
  192x192, rescale intensity to [0, 1].
- **Augmentation**: 9 translations (+/-10% in x and y), 7 rotations
  (0, +/-10, +/-20, +/-30 deg), 3 scalings (0.9, 1.0, 1.1) = 189 combinations.
  Do this **on the fly** — the paper's 2.8 GB pre-materialised shuffled-file scheme
  exists only to hide disk I/O and is not worth reproducing.
- **Alignment**: shift slice `i` by `(96, 96) - c_hat_i`, zero-fill.
- **Split**: ACDC gives 100 train / 50 test. Dangi used 79/9/9 of 97. Mirror the
  proportions: split the 100 training patients 80/10/10, and optionally report on
  the 50-patient ACDC test set as a held-out check.

## 7. Pitfalls that will silently corrupt results

1. **Transform the GT centre by the same affine as the image** during augmentation.
   Easy to forget; poisons training with no visible error.
2. **Mid-slice centre propagation** (Sec. 2.2): apical/basal slices with partial or
   absent blood pool have no valid centroid. Dangi propagates the centre from the
   nearest mid-slice. Skip this and the training targets are garbage at exactly the
   slices the paper is about. On ACDC this means slices where `label == 3` is empty
   or tiny.
3. **Sign/axis convention** on the shift, and NIfTI (X, Y) vs numpy (row, col) order.
   Cheap to get wrong, cheap to unit-test — write the test.
4. When simulating, **store the injected shifts**; without them the evaluation in
   Sec. 5 is impossible to do after the fact.

## 8. Environment

- Python 3.13.5 (conda base), `torch 2.9.0+cu128`, `numpy 2.3.4`, `scipy 1.16.2`.
- CUDA reported unavailable from the login node ("Can't initialize NVML") — needs a
  GPU allocation via Slurm. This is a UMich HPC cluster; `/scratch/engin_root/engin1/sincheol`
  and `/scratch/jjparkcv_root/jjparkcv98/sincheol` exist and are empty.
- **GPU access (must use `--account=jjparkcv98`):**
  - GPU access: salloc --account=jjparkcv98 --partition=gpu --gres=gpu:1 --mem=48G --cpus-per-task=4 --time=01:00:00
  - GPU access: salloc --account=jjparkcv98 --partition=spgpu --gres=gpu:1 --mem=48G --cpus-per-task=4 --time=01:30:00
  - Check GPU in use: my_account_resources -v jjparkcv98
- **Not installed**: SimpleITK, scikit-image, nibabel, PyMaxflow.
- Stage A needs no new dependencies: resampling and the affine augmentations can go
  through `scipy.ndimage`. A minimal NIfTI reader is also possible via `gzip` +
  `numpy.frombuffer` (used during this survey), but installing `nibabel` is cleaner
  if permitted.
- Compute is not a concern. Dangi's 46 min/epoch x 100 epochs (~77 GPU-hours on two
  Titan Xps) is an artifact of their file-caching scheme. On-the-fly augmentation
  over ~2000 labelled slices at 192² through a small conv net is a couple of minutes
  per epoch at most.

## 9. Notes on what is NOT here

- **Our ICLR paper's codebase and experiments section are not on this machine.**
  Only the introduction text (quoted from the user) is known. Decision 2 below
  cannot be answered from anything in this directory.
- A filesystem-wide search turned up `/tmp/acdc*.png` and `/tmp/acdc_heatmap_layout2.log`.
  These are **owned by a different user (`umcolin`)**, dated Aug 24-25, and are the
  output of an unrelated LaTeX build on this shared cluster. Coincidental naming.
  Not our work — do not chase them.

## 10. Open decisions for the new session

1. Confirm **Stage A only** (recommended) vs. also Stage B.
2. Confirm the **corruption simulation parameters** — respiratory shift magnitude and
   distribution, whether through-plane shift is simulated, and how frames are sampled
   for cardiac-state mismatch. These should match whatever our paper's experiments
   section already specifies; check that document first.
3. Whether `nibabel` may be pip-installed, or whether to use the dependency-free
   NIfTI reader.
