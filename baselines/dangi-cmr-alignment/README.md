# Dangi et al. (2018): Stage A baseline

PyTorch reimplementation of the LV centre-regression CNN and in-plane slice
translation in **Cine Cardiac MRI Slice Misalignment Correction Towards Full 3D
Left Ventricle Segmentation**, Dangi, Linte & Yaniv, SPIE 2018.
[Paper](https://doi.org/10.1117/12.2294936).

This provides the Dangi baseline for comparison with another reconstruction
model. It does not include Stage B atlas/graph-cut segmentation. No pretrained
author weights are supplied; train on ACDC to obtain `best.pt`.

## Setup

From this directory (the existing environment supplies CUDA-enabled PyTorch):

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

The project virtual environment contains the added dependencies without modifying
the base environment. Commands below run from the repository root.
To import this baseline from another project's environment, install it there
with `python -m pip install -e /home/sincheol/CMRI/Dangi`.

## Central notebook

Open [Dangi_Workbench.ipynb](Dangi_Workbench.ipynb) for the complete workflow in
one place: configuration, data preview, model summary, Slurm/local training,
resume, job logs, learning curves, evaluation, interactive slice alignment and
NIfTI export. Implementation remains in the Python modules.

Section **6b: Ground truth vs output** adds matched SAX/XZ/YZ views of the clean
reference, input, Dangi output and absolute error, with interactive slice controls.
Set `REFERENCE_PATH` to a clean intensity volume and enable `RUN_ALIGNMENT`.
The default matches the example ACDC ED input; use your clean target when viewing
corrupted inputs. Labelled inputs also show true and predicted LV centres.

Notebook dependencies are already installed in the project virtual environment.
To recreate them and start JupyterLab:

```bash
.venv/bin/python -m pip install -e '.[notebook]'
bash scripts/notebook.sh
```

Select **Python (Dangi)**, edit the configuration cell, and run the notebook.
The initial run previews data and commands. Enable the action switches when
ready to train, evaluate or align. Use a GPU Jupyter session for local training,
or keep `TRAIN_MODE='slurm'` to submit through account `jjparkcv98`. Slurm uses
the notebook's exact data directory, run directory and training settings.
From an existing Jupyter/VS Code session, select `.venv/bin/python` as the kernel.

## Train on ACDC from the command line

```bash
sbatch scripts/train.slurm
```

The launcher uses **account `jjparkcv98`**, partition `spgpu`, one GPU, 48 GB RAM,
four CPUs and 90 minutes. It automatically resumes `runs/dangi/last.pt` when
resubmitted. `best.pt` is selected by minimum validation SSD; `last.pt` includes
optimizer state for epoch-boundary resume. If the allocation expires during an
epoch, resubmission repeats that incomplete epoch. Benchmark epoch duration and
request a longer allocation if needed:

```bash
sbatch --time=04:00:00 scripts/train.slurm
```

Alternatively, after acquiring an interactive GPU allocation:

```bash
salloc --account=jjparkcv98 --partition=spgpu --gres=gpu:1 --mem=48G --cpus-per-task=4 --time=01:30:00
.venv/bin/python -m dangi.train --data ACDC/database --output runs/dangi
```

The default is 100 epochs, each visiting **every training slice × 189 transforms**
in shuffled order. Original, preprocessed slices are cached in RAM; augmentation
is performed on demand. Validation is unaugmented. A seeded, diagnosis-stratified
80/10/10 patient split of the 100 ACDC training subjects is saved as `split.json`.
All ED/ES frames from each patient remain together. The 50 ACDC testing patients
are separate. Outputs include `config.json` and `history.jsonl`.

## Apply the trained model

```bash
.venv/bin/python -m dangi.align \
  --checkpoint runs/dangi/best.pt \
  --input ACDC/database/testing/patient101/patient101_frame01.nii.gz \
  --output runs/dangi/patient101_aligned.nii.gz --device cuda
```

Accepts a 3D volume or 4D cine (each frame is processed independently). Output
uses the **1.5625 mm, 192×192 preprocessed grid**, preserves intensity units, and
retains slice spacing and orientation. A companion `.nii.gz.json` records
predicted centres, applied translations, configuration, and input/output affines.
Arrays in JSON are `[frame][slice][xy]`, with translations in preprocessed pixels.
The affine describes the output sampling grid; individual content translations
are stored in JSON because a single NIfTI affine cannot encode per-slice motion.

For integration with your model's comparison pipeline:

```python
from dangi.align import load_model, align_stack, align_nifti
from dangi.data import preprocess
import nibabel as nib

model, config = load_model("runs/dangi/best.pt", device="cuda")
volume = nib.load("input.nii.gz")
corrected_volume, transforms = align_nifti(model, volume, config)

# Or supply preprocessed float arrays in (slice, row=y, column=x) order:
slices, output_affine = preprocess(volume, config)  # 3D input
corrected, centers_xy, applied_shifts_xy = align_stack(model, slices, config)
```

The model sees normalized intensities and predicts `(x,y)` centres in pixels.
Each slice receives `(96,96) - predicted_center`, using linear interpolation
and zero fill. No target/reference image or segmentation is required at inference.

## Paper-style evaluation

```bash
.venv/bin/python -m dangi.evaluate --checkpoint runs/dangi/best.pt --device cuda
.venv/bin/python -m dangi.evaluate --checkpoint runs/dangi/best.pt \
  --subset acdc-test --output runs/dangi/acdc_test.json --device cuda
```

Reports mean/std/median initial distance to the stack's mean true centre and
residual predicted-versus-true centre error, in pixels, with the paper-style KS
statistic. Raw per-slice values and measured-versus-propagated target flags are
saved. This is an ACDC implementation of those metrics, not a claim to reproduce
the original STACOM table values. Your corruption simulation and reconstruction
comparison remain external to this baseline.

## Method and explicit assumptions

| Component | Implementation |
|---|---|
| CNN | Two 3×3 conv+ReLU at each level, channels 4/8/12/16/20; four 2×2 max pools |
| Side branches | First four levels: 3×3, one-channel conv+ReLU, same padding, then flatten |
| Concatenation | 36,864 + 9,216 + 2,304 + 576 + 2,880 = 51,840 features |
| Dense layers | 51,840 → 256 with ReLU → 2 linear outputs; no dropout |
| Initialization | Xavier uniform for convolution/dense weights; zero biases |
| Loss | Sum of squared x/y errors per slice, averaged over batch |
| Optimizer | **Assumption from handoff:** Adam, learning rate 0.001, default betas, no weight decay |
| Batch/epochs | **Choice:** batch 32; 100 epochs; full 189-grid exposure per epoch |
| Preprocessing | In-plane 1.5625 mm; centre crop/pad 192²; linear image interpolation, nearest-neighbour labels |
| Intensity scaling | **Choice:** min/max per slice to [0,1]; constant slices become zero |
| Targets | **ACDC adaptation:** centroid of label 3 after preprocessing |
| Missing centres | Nearest valid slice, tie toward stack middle; all-empty stacks raise an error |
| Tiny blood pools | Keep nonempty masks by default; `--min-bp-pixels` explicitly enables a threshold |
| Augmentation | Cartesian grid: x/y ∈ {−19.2,0,19.2}px; rotation ∈ {−30,−20,−10,0,10,20,30}°; scale ∈ {0.9,1,1.1} |
| Transform convention | **Choice:** scale/rotate about (96,96), then translate; same forward affine for targets; inverse map for resampling |
| Alignment centre | (96,96), following the handoff convention; no centre clipping |
| Validation | **Choice:** original unaugmented validation slices; lowest validation SSD checkpoint |

Padding and activations are explicit interpretations of Fig. 1. Transform
composition, normalization granularity, batch size and epoch sampling are
implementation choices where the paper/handoff does not fully specify them.
Propagated target slices follow the same augmentation as their images. No
unlabelled cine frames are treated as labelled examples.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

Inside an allocated GPU node, add `DANGI_TEST_DEVICE=cuda` to exercise the model
forward/backward and checkpoint test on CUDA. Tests cover all 189 image/target
affines, translation sign and axes, centre propagation, physical resampling,
checkpoint reload, 4D NIfTI alignment, and real ACDC loading/split isolation.

`--max-train-batches 1 --epochs 1 --output runs/smoke` is available for a training
CLI smoke test. Such a checkpoint is **not** a trained comparison baseline.
