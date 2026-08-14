# Input/output resolution experiments: native-256 rendering, high-z splatting, and the 224² DINO pilot

> **TL;DR & takeaway** — On `b2ck5kfd`, replacing the current 518² render with a
> downsampled 256² world-point field plus the original 256² slice intensities changed mean clean-val
> bbox PSNR by only **−0.118 dB** over 10 subjects. Rendering the same predictions directly into 128
> z planes did not produce z super-resolution: coverage fell from **77.4% to 12.1%**, leaving bowed
> acquisition sheets separated by holes. A 224² DINO/DPT pipeline is mechanically valid and all
> pretrained weights load; it changes the per-slice patch grid from 37×37 to 16×16 and is worth a
> controlled training test. That pilot is running from a temporary tree with **only four 518→224
> constants changed**; no tracked training code was changed. Its final quality result is pending.

**Date:** 2026-08-12/13  
**Reference run:** W&B `b2ck5kfd`, on-disk experiment
`scratch/logs/213520194_mri_volume_heartl1_w050_dynamic_axial_cmrx24only/`  
**Reference checkpoint:** `ckpts/checkpoint_best.pt`  
**Cohort for all probes here:** CMRxRecon2024-only validation split  
**Training-code status:** unchanged in the tracked repository *at the time of writing —
superseded 2026-08-13: the resolution knob + an always-native splat were ported into the
tracked pipeline, see docs/73 (note the ported render differs from this pilot's first 15
epochs, which splatted model-resolution content)*

This document records the full resolution discussion, the experiments actually run, their scope and
limitations, the temporary 224² training pilot, and the storage failure caused during that pilot.

## 1. Baseline resolution flow, verified from code

For each subject:

1. Preprocessing produces `phases` on `(T=12, D, 256, 256)`. In-plane spacing is 1.4 mm and z
   remains at the subject's native pitch/count.
2. One input frame is selected per physical slice, so `S == D` under the default sampler. Slot 0 is
   the target-phase reference slice.
3. Each selected 256² slice is bilinearly resized to **518²** and replicated to three channels.
4. DINOv2 ViT-L/14 produces a **37×37 = 1,369-patch** grid per slice. Including the camera token
   and four register tokens gives 1,374 tokens per slice.
5. VGGT applies 24 alternating frame/global attention stages.
6. DPT decodes the patch features to a per-pixel **518×518×3 residual DVF** plus one confidence
   channel. Confidence is not used by the MRI loss.
7. `world_points = scanner_coords + residual_dvf` at 518².
8. The 518² input intensities and world points are trilinearly scattered into the subject-specific
   output grid `(D, 256, 256)`, then divided by accumulated coverage.

There is no explicit 518→256 image downsample before the current splat. The reduction happens because
approximately `(518/256)^2 = 4.09` predicted points per input slice are scattered into each 256²
destination plane.

## 2. Coordinates and positional information

### 2.1 Scanner/world coordinates

At either 518² or 224², the scanner-coordinate grid is generated at the model's input resolution:

```text
x = 2 * px / (W - 1) - 1
y = 2 * py / (H - 1) - 1
z = physical_z_mm / Z_HALF_MM,  Z_HALF_MM = 90
```

Therefore x/y cover the same normalized `[-1,1]` physical FOV regardless of whether there are 518
or 224 samples. The z coordinate is physical, not a `[-1,1]` rescaling of each subject's own D.
The residual head is linear/unbounded, so predicted world points may leave the valid grid; the splat
drops out-of-bounds contributions.

### 2.2 Where position enters the model

There are three separate spatial-position mechanisms:

1. **Inside DINOv2:** its learned absolute patch-position tensor is bicubically interpolated by
   `vision_transformer.py::interpolate_pos_encoding`. At 518 it supplies a 37² grid; at 224 it is
   interpolated to 16². This is why changing input resolution does not create a parameter-shape
   mismatch.
2. **Inside VGGT attention:** `Aggregator` builds integer 2-D patch coordinates and applies 2-D
   rotary position embedding (RoPE) in both frame and global attention. RoPE is computed, not a
   learned lookup table. Camera/register special-token positions are set to zero.
3. **Inside DPT:** `DPTHead` adds a fixed UV-grid sinusoidal embedding to feature maps before and
   after feature fusion.

There is no learned *slot-index* position embedding. A non-reference input slot is identified by its
image content, its physical-z sinusoidal embedding on the camera token, and its patch coordinates.
Slot 0 is distinguished by the two-valued pretrained camera token (reference versus all other slots).
Consequently, the non-reference slots are intended to be permutation-equivariant; their list index is
not supplied as semantic information.

## 3. Experiment A: render the actual model output at native 256²

### 3.1 Question

Keep the trained 518² model forward pass, but downsample its predicted world-point field to 256² and
splat the original 256² canonical input slices instead of the bilinearly enlarged 518² slices. Does
the current four-points-per-voxel render materially help?

### 3.2 What was actually evaluated

This was **real inference from the trained `b2ck5kfd` checkpoint**, not a GT-DVF simulation:

- first 10 validation subjects;
- clean validation sampling (respiratory simulation disabled for this focused render comparison);
- one shared 518² model forward per subject;
- variant A: predicted 518² world points + actual 518² model-input intensities → `(D,256,256)`;
- variant B: bilinear 518→256 world-point resize + original cached 256² selected-slice intensities
  → `(D,256,256)`;
- the GT target-phase volume was used only to score the two reconstructions;
- no GT DVF exists or was used;
- native intensities were verified against the model inputs after re-upsampling: maximum difference
  `1.19e-7`.

Downsampling `world_points` rather than `residual_dvf` is equivalent here up to interpolation
roundoff because the x/y scanner grid is planar/linear and the z coordinate is spatially constant
within each slice.

### 3.3 Results

| Metric, mean over 10 | 518² splat | native-256² splat | 256 − 518 |
|---|---:|---:|---:|
| bbox PSNR | 30.871 dB | 30.754 dB | **−0.118 dB** |
| full PSNR | 31.482 dB | 31.321 dB | **−0.161 dB** |
| mean absolute A/B disagreement | — | — | 0.00323 normalized intensity |

Per-subject bbox deltas ranged from **+0.001 to −0.302 dB**. The largest loss was the D=6 subject;
thin edges accounted for most visible disagreement. The inference is narrow but useful: at the
current learned geometry, 518² render oversampling provides only a small anti-aliasing benefit.

This does **not** prove that a DPT head trained natively at 224² or 256² will match the 518² model.
It isolates only the renderer/sample-density effect.

### 3.4 Visualization and error-map scale

![Actual model output: 518² versus native-256² splat](../figs/splatvar_A_vs_B_axial.png)

The last column is `abs(V_518 - V_256)` in the model's normalized intensity units. Each row uses
`vmin=0` and its **own 99.5th-percentile absolute difference as `vmax`**, clipping the largest 0.5%.
There is no shared numerical colorbar in this figure, so it is a spatial-localization view, not a
cross-row magnitude comparison.

## 4. Experiment B: direct splat into 128 z planes

### 4.1 Setup

The same actual 518² predicted world points and intensities were splatted directly into
`(128,256,256)`. `z_scale` was multiplied by `(128-1)/(D-1)` so that the high-resolution grid spans
the same physical z extent as the native-D grid. This changes the destination sampling only; it does
not add source slices or retrain the model.

### 4.2 Result

| Coverage statistic, mean over 10 | Fraction with coverage > `1e-6` |
|---|---:|
| Current native-D output | 0.7742 |
| Direct 128-plane output | **0.1207** |

The direct 128-plane output consists of thin, curved sheets near the acquired planes, separated by
large empty gaps. The bending is meaningful—the model's predicted through-plane correction deforms
the source planes around anatomy—but the gaps are unavoidable because only D physical planes were
observed. Increasing destination D is therefore not z super-resolution.

![Direct 128-plane splat and coverage](../figs/splatvar_highz_coronal.png)

The first two columns interpolate native-D volumes to 128 only for display. The third column is the
actual direct 128-plane splat. The fourth is `log1p(coverage)`.

A useful 128-plane result would require a model/renderer that explicitly fills between planes—for
example a z-aware reconstruction decoder or a deliberately widened z kernel. Such output would be a
learned prior/interpolation, not recovered measurements.

## 5. Reduced input resolution: 518² → 224²

### 5.1 Mechanical change

At 224² with patch size 14:

- DINO patch grid: `16×16 = 256` instead of `37×37 = 1,369`;
- including five special tokens: **261 instead of 1,374 tokens per slice** (5.26× fewer);
- DPT output: `224×224×3` residual DVF per slice;
- scanner coordinates: `224×224×3` per slice over the same normalized physical FOV;
- world points: `224×224×3` per slice;
- splat destination remains the real target grid `(D,256,256)`;
- source density is `224²/256² = 0.7656` points per in-plane destination voxel before motion.

No architecture parameter has to be initialized from scratch. DINO's learned absolute position
embedding interpolates to 16²; VGGT RoPE is generated dynamically; DPT is convolutional and derives
the patch grid from H/W. All DINO, attention, and DPT weights can load unchanged. Fine-tuning is still
required because the feature scale/distribution and available spatial detail changed.

The physical receptive-cell tradeoff is real. Both image sizes cover the same 358.4 mm canonical
width, so one 14-pixel patch spans approximately:

- 518 input: `14/518 * 358.4 = 9.69 mm`;
- 224 input: `14/224 * 358.4 = 22.4 mm`.

DPT can output a dense 224² field, but it cannot recover image evidence that never survived the
16² tokenization. There is no high-resolution image skip into DPT. Detecting small wall motion is
therefore the main quality risk, not tensor compatibility or splat mechanics.

### 5.2 Two-sided design review

The independent advocate/skeptic review agreed on compatibility and weight transfer, but emphasized
different risks:

- **Case for 224:** attention sees 5.26× fewer tokens per slice; the desired DVF is spatially smooth;
  DINOv2/attention/DPT accept the dynamic grid; Experiment A shows that approximately one source
  point per output voxel is already sufficient for rendering.
- **Case against 224:** the perception grid becomes 16² and DPT has no image skip; a 22.4 mm patch
  may blur the local evidence needed for LV-wall motion. Lower token count does not imply a 5.26×
  end-to-end training speedup because large GEMMs, optimizer work over ~941M parameters, data,
  validation, and checkpointing remain.

Decision: a paired short pilot is justified. If 224 loses important motion information, 252²
(18×18 patches) is the next sensible intermediate point. DINOv3 is a separate migration: its
patch-16 backbone and feature statistics do not preserve compatibility with the 605M VGGT attention
weights in the same simple way.

## 6. Zero-training 224² probe of `b2ck5kfd`

The existing checkpoint was evaluated with both its normal 518² input and a bilinearly resized 224²
input/scanner grid, with no weight updates. These are actual model reconstructions on 10 clean val
subjects. The displayed bbox deltas are mixed (approximately **−0.85 to +0.99 dB**) and average
approximately **−0.05 dB from the rounded labels in the saved figure**.

![Zero-shot 518² versus 224² model input](../figs/b2ck5kfd_224_zero_shot.png)

Columns are GT, 518 output, 224 output, and `abs(224-518)`. All difference panels share one scale:
`vmin=0`, `vmax=p99.5=0.0273` normalized-intensity units, so colors are comparable across rows.

Important provenance correction: the original probe printed its raw JSON summary to the terminal but
did not persist it. An earlier conversational summary called **−0.82 dB** the mean bbox change; that
is inconsistent with the ten per-row deltas actually printed into the saved figure—**−0.82 dB is one
subject's delta, not their mean**. The durable script now writes its full records to JSON; rerun it
before citing an exact aggregate in a paper. This probe is only a warm-start compatibility check in
any case: the model was trained at 518².

The same-session forward-only timing suggested roughly 5.6× lower forward latency at 224 on the A40,
but the raw timing JSON was not retained. Treat this as an estimate until the durable probe is rerun
without another GPU workload.

## 7. 224² training pilot

### 7.1 Exact scope

The user required an exact `b2ck5kfd`-recipe pilot, CMRxRecon2024 only, changing only the resolution.
No tracked training source was edited. A copy of the `vggt-arm-heart` training tree was made at:

```text
/tmp/vggt_b2ck5kfd_224_exact/training/
```

Its diff against `/home/minsukc/vggt-arm-heart/training/` is exactly four replacements:

| Temporary file | Change |
|---|---|
| `config/default.yaml` | `img_size: 518 → 224` |
| `data/datasets/mri_dataset.py` | `INPUT_IMG_SIZE = 518 → 224` |
| `data/gpu_aug.py` | `INPUT_IMG_SIZE = 518 → 224` |
| `data/respiratory.py` | `INPUT_IMG_SIZE = 518 → 224` |

Everything else remains the `b2ck5kfd` recipe: fresh VGGT-1B weights, aggft with patch embed frozen,
LR `5e-5`, heart L1 weight `0.5`, moderate affine/photometric augmentation, respiratory simulation,
one frame per native slice, CMRx24-only `235` train / `58` ED+ES val entries, batch size 1.

Runtime validation before launch checked both ordinary GPU slice extraction and the respiratory
zero-displacement path at 224; both produced `(B,S,224,224,3)`. The model then ran real forward,
backward, validation, and checkpoint operations at 224.

### 7.2 Runtime locations and launch

There is no tracked sbatch or training launcher for this pilot. It was launched directly inside the
existing interactive A40 allocation. The current full-resume command is equivalent to:

```bash
WANDB_MODE=online \
PYTHONPATH=/tmp/vggt_b2ck5kfd_224_exact/training:/home/minsukc/vggt-arm-heart \
micromamba run -n svr python \
  /tmp/vggt_b2ck5kfd_224_exact/training/launch.py --config default \
  exp_name=b2ck5kfd_224_local_r1 \
  split_file=training/splits/cmrx24only.txt dataset_name=cmrx24only \
  limit_train_batches=235 limit_val_batches=58 max_epochs=300 \
  loss.volume.heart_weight=0.5 \
  logging.log_dir=/tmp/b2ck5kfd_224_local_r1 \
  checkpoint.save_dir=/tmp/b2ck5kfd_224_local_r1/ckpts \
  checkpoint.resume_checkpoint_path=/tmp/b2ck5kfd_224_local_r1/ckpts/checkpoint_last.pt \
  +logging.wandb_writer.resume_id=efra0f3j
```

Runtime artifacts:

```text
/tmp/b2ck5kfd_224_local_r1/log.txt
/tmp/b2ck5kfd_224_local_r1/metrics.jsonl
/tmp/b2ck5kfd_224_local_r1/val_per_subject.csv
/tmp/b2ck5kfd_224_local_r1/ckpts/checkpoint_last.pt
/tmp/b2ck5kfd_224_local_r1/ckpts/checkpoint_best.pt
/tmp/b2ck5kfd_224_local_r1/wandb/       # local mirror of W&B run efra0f3j
```

Because all of these are under node-local `/tmp`, they are temporary and disappear when that node's
temporary storage is cleared. The pilot initially ran with `WANDB_MODE=offline`, which was wrong for
this user-facing DVF inspection run. After epoch 13 was checkpointed (3,290 persisted training
steps), it was stopped at the validation boundary and resumed online from
`checkpoint_last.pt`. The resumed process loaded the model with zero missing/unexpected keys, loaded
the optimizer state, and began epoch 14. Metrics and subsequent DVF panels are available at
<https://wandb.ai/minsuk-choi/vggt-mri/runs/efra0f3j>.

### 7.3 Timing evidence and what it does not prove

After compile warmup, observed 224² training batches were approximately **0.49 s/batch** on the A40.
The historical `b2ck5kfd` 518² log showed approximately **1.52 s/batch on an L40S**, giving a raw
ratio near 3.1×. That is **not a controlled speed comparison**: GPU type and concurrent occupancy
differ, and L40S is faster than A40 for this workload. It is evidence that 224 is faster, not a
measurement of the same-GPU training speedup.

A same-A40 forward-only zero-shot probe suggested ~5.6×, but full training includes backward,
optimizer, data, validation, and checkpointing. A controlled same-node paired benchmark is still
needed for a defensible training-speed number.

### 7.4 Status and preliminary quality

The pilot is ongoing; no trained-quality verdict belongs in this document yet. Epoch-0 validation is
far too early to compare against mature `b2ck5kfd`. The final decision must use the standard paired
motion metrics (`recov_frac_heart`, motion PSNR, and `hole_frac_heart` veto), plus EF amplitude and
the DVF panels, over matched step counts—not only bbox/full PSNR.

## 8. Storage incident and recovery

The first launch incorrectly used:

```text
/home/minsukc/vggt/temp/b2ck5kfd_224_local_r1/ckpts/
```

It wrote an **8.3 GB** full `checkpoint_last.pt` and **3.6 GB** weights-only
`checkpoint_best.pt`, exhausting the user's home quota. Consequences, verified from logs:

1. The local pilot exited after executing 470 train steps. Only the epoch-0 checkpoint (235 steps)
   was durable, so the second epoch's 235 in-memory steps were lost.
2. Four unrelated VGGT SLURM jobs briefly raised `OSError: [Errno 28] No space left on device` while
   flushing their home-based SLURM console logs:
   - `57252530`: `corsegdice_w002`;
   - `57252531`: `corsegdice_w100`;
   - `57253759`: pooled moderate augmentation;
   - `57256923`: pooled aggressive augmentation.
3. Those four jobs did **not** stop or restart (`Restarts=0`). Their GPFS `metrics.jsonl` streams were
   continuous every five steps through the incident, and all later wrote complete 8.87 GB
   `checkpoint_last.pt` files on GPFS. The confirmed damage was limited to missing/interleaved home
   console-log lines.
4. Ten active MRI2CT jobs showed no ENOSPC/checkpoint error and `Restarts=0`. Their apparent
   `/home/minsukc/MRI2CT/slurm_logs` directory resolves to GPFS, and their checkpoints/W&B state also
   live on GPFS.

The two pilot checkpoints were moved to `/tmp/b2ck5kfd_224_local_r1/ckpts/`. The pilot then resumed
with model keys `0 missing / 0 unexpected`, optimizer state restored, and epoch 1/global step 235.

Operational rule added globally after the incident:

> Never save heavy files under `$HOME`. Put them on GPFS, or in `/tmp` if they are temporary.

## 9. Scripts and artifacts

### Durable repo-local experiment scripts

- `tools/exp_b2ck5kfd_splat_variants.py` — exact A/B native-256 render and direct-128-z probe;
  writes JSON to `result/resolution_experiments/b2ck5kfd_splat_variants.json` and the two splat
  figures to `figs/`.
- `tools/probe_b2ck5kfd_224_zero_shot.py` — exact 518-vs-224 zero-shot probe; now persists its raw
  per-subject metrics/timing to `result/resolution_experiments/b2ck5kfd_224_zero_shot.json` and
  writes `figs/b2ck5kfd_224_zero_shot.png`.
- `tools/compare_splat_resolution.py` — older generic render-resolution experiment. It uses a
  different historical checkpoint/protocol and is **not** the provenance for the `b2ck5kfd`
  numbers above.

### Original one-off scripts/results

The exact original files remain in the Claude scratch directory for this node/session:

```text
/tmp/claude-114459240/-home-minsukc-vggt/85fec8b1-797c-4261-a83f-2a9ba9c0311f/
  scratchpad/splat_variants_b2ck5kfd.py
  scratchpad/splat_variants_results.json
  scratchpad/splatvar.log
/tmp/probe_224_b2ck5kfd.py
```

They are temporary; use the durable `tools/` copies going forward.

### Figures

- `figs/splatvar_A_vs_B_axial.png`
- `figs/splatvar_highz_coronal.png`
- `figs/b2ck5kfd_224_zero_shot.png`

## 10. Decisions and next checks

1. **Native-256 rendering is safe enough to pursue** if inference simplicity matters; measured cost
   at fixed geometry is only ~0.1 dB. It does not reduce DINO/attention training cost by itself.
2. **Do not call a 128-plane destination z super-resolution.** It is mostly uncovered without a new
   interpolating/learned renderer or more acquired planes.
3. **Continue the 224² paired pilot.** It is mechanically sound and attacks the dominant token cost.
4. Before claiming a speedup, run same-A40 518/224 training-step benchmarks under equal occupancy.
5. Before adopting 224, compare mature checkpoints at matched steps using the standard motion/EF
   criteria and inspect online DVF panels for loss of LV-local motion.
6. If 224 loses quality, test 252² once before abandoning reduced resolution.
