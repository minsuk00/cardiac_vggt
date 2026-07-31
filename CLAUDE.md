# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) adapted for **cardiac 4D MRI slice-to-volume reconstruction** on CMRxRecon2024 (`Cine_combined`, 301 subjects split 240/30/31 train/val/test via `training/splits/random_8_1_1.txt`).

**Research goal:** enable fast real-time free-breathing cine by reconstructing the full 3D heart volume at any target cardiac phase from a *few scattered single-frame-per-slice* acquisitions (ideally one frame/slice), instead of the slow many-frames-per-slice + retrospective-sort/SVR route. No real-time training data exists, so we **simulate** the sparse scattered acquisition from gated breath-hold CMRxRecon2024 cine (each input slice = one frame at an arbitrary (phase t, z-depth)) + motion aug, and aim to generalize to true real-time cine. Currently *only* the scattered sampling + in-plane aug are simulated; realistic acquisition physics (bSSFP transient, single-shot artifacts, respiratory motion) is aspirational — see Future enhancements. **Target inference information contract:** at the one-frame-per-slice extreme the model is assumed to know only `z` per input slice — input cardiac `t` and respiratory `r` are *unavailable* (no ECG / no respiratory device / no self-gating); target-phase *queries* stay free. Design stance, not yet implemented — see `docs/04_inference_information_contract.md`.

**Active pipeline: unsupervised intensity-based, multi-phase** (`mri_volume*` configs). No GT DVF. Each sample picks a target cardiac phase `t_target ∈ {0..T-1}`; loss compares splatted predicted volume `V_canon` against the on-disk NIfTI at that target phase (`V_gt`).

**Target-phase conditioning = REFERENCE SLICE (current default, `mri_volume.yaml`).** Slot 0 is a real target-phase reference slice at the mid-ventricular plane (`reference_slot=true`, `use_reference_token=true`), marked via VGGT's native two-token `camera_token` (index 0 = anchor, 1 = the rest). The model reads the target phase from slot-0's *image content* (`V_gt = phases[t_target]` = that slice's phase) — **not** a content-free `target_t` index, which regressed every patient's EF to the cohort mean (flat-EF; `use_t_pose_embedding`/`use_target_t_pose_embedding` OFF, `target_t_indices` inert). This requires the **aggregator finetune (aggft)** so the camera_token/z_embedder specialize (freeze `*patch_embed*` only). Consequence: you reconstruct **observed** phases (≈ the recoverable limit). **EF recovery is confirmed on real final ckpts** (slope 0.77–0.79, honest Spearman ~0.55; the earlier "flat-EF" reading was an undertrained ckpt) — see `docs/24`, `docs/25`, `docs/33`. Legacy `target_t`-index path survives behind the default-off flags (`mri_finetune.yaml`).

The "**4-day baseline**" (referenced below) is the prior ED-only run at `./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/` — 31+ dB PSNR at ED. Post-refactor it's available **weights-only as a warm-start seed** (via `CKPT_ONLY` / `resume_checkpoint_path`, `strict=false`; the current reference script defaults to fresh-from-base instead), **not a true resume**: input normalization + V_gt frame changed, so its memorized codes are stale and the old PSNR won't reproduce — treat the canonical-grid pipeline as a fresh-retrain series.

**Canonical-grid refactor (2026-05-24), superseded by native-z (2026-07-31, docs/58).** Every subject's **in-plane** axes are resampled to a shared grid; input slices and V_gt both live in it. **Z is no longer resampled** — each subject keeps its own native slice pitch/count (`dz`, `D`), see "CMR data notes" below. (Originally every subject was forced onto one shared `(256,256,12)` cube including Z — that was correct only because the sole active dataset, CMRxRecon2024, is uniformly 12mm; native-z is what makes the pooled multi-pitch cohort in docs/58 possible.) The old supervised-DVF path is **fully removed** — `MRIDataset` no longer reads DVF/mask NIfTIs and `gt_dvfs`/`scale_factors` are gone from the batch. DVF NIfTIs remain on disk for repro; legacy DVF tooling (`compute_cine_dvf_elastix.py`, carmen/elastix verification) plus OCMR/CMRx4DFlow2026 recon explorations and legacy configs/sbatch all live under `_archive/`.

- MRI data: `/scratch/data/CMRxRecon2024/` (symlinked, GPFS)
- Env: `micromamba activate svr`
- SLURM: `spgpu` partition for training (A40 GPUs), `standard` for CPU jobs. **Account: use `jjparkcv0` by default** (`jjparkcv98` frequently hits `AssocGrpSubmitJobsLimit`). The existing sbatch headers say `jjparkcv98` — override with `sbatch --account=jjparkcv0 …` or edit the `#SBATCH --account` line.

## Setup

```bash
micromamba activate svr
pip install -r requirements.txt           # includes monai>=1.6,<1.7
pip install --no-deps -e /home/minsukc/MRI2CT/batchaug/  # GPU aug — see note below
```

**This repo is NOT installed as a package** — there is no `pip install -e .` step (verified: `pip show vggt` finds nothing, and `vggt` is absent from `envs/svr_torch231.freeze.txt`, so it was never installed in the old env either). Instead, **always run from the repo root with `PYTHONPATH=training:.`**. Both entries are load-bearing, for different reasons: Python puts the ***script's*** directory on `sys.path[0]`, never your cwd — so running `training/launch.py` gets `training/` for free but NOT the repo root, and `.` is what makes `from vggt.utils… import` resolve (this is exactly the job an editable install would do). `training` is what makes the Hydra `_target_` short names resolve (`loss.MultitaskLoss`, `data.dynamic_dataloader.DynamicTorchDataset`, `train_utils.gradient_clip.GradientClipper`) — redundant for `launch.py` itself, but required by the 66 scripts outside `training/` that import `data.*`/`loss`/`trainer` directly (57 in `tools/`, 5 in `baselines/`, 4 in `evaluation/`; none in `inference/`). `tests/` needs neither: `tests/conftest.py` inserts both paths itself.

**Stack: torch 2.13.0+cu130 / torchvision 0.28.0 / triton 3.7.1 / monai 1.6.0 / numpy 2.2.6** (upgraded 2026-07 from the frozen torch 2.3.1 + numpy 1.26.4 build — see the torch-2.13 upgrade doc). **monai** is `>=1.6,<1.7` (needs torch>=2.8); it was pinned `<1.5` only to hold torch at 2.3.1, a constraint now lifted. **numpy** is `>=2,<2.5` (upper bound = scipy/numba); the numpy-2 migration needed one source fix in `inference/adapters/base.py` (`np.percentile` on float32 now returns float32, which silently disabled a divide-by-zero guard → all-NaN OOD inputs). Re-verify any future dependency bump with `bash tools/verify_env_migration.sh`. **batchaug** is not on PyPI; install editable from the MRI2CT clone with `--no-deps` (keeps pip from re-resolving the pinned torch stack). torch 2.13 now supplies the matched triton 3.7.1 that batchaug's triton backend wants, but `gpu_aug.py` still forces `batchaug.set_backend("pytorch")` at runtime for reproducibility, so triton stays dormant. **fused_ssim** (the SSIM-3D metric) is a CUDA extension that must be rebuilt against the active torch (`module load gcc/11.2.0 cuda/13.1.0`, then `pip install --no-build-isolation` from its pinned git commit). Full rationale in the comment block at the bottom of `requirements.txt`.

## Training

Entry point: `training/launch.py` (Hydra).

```bash
# Active config
PYTHONPATH=training:. torchrun --nproc_per_node=1 --master_port=29507 \
    training/launch.py --config mri_volume

# NOTE: single-GPU only. DDP was removed in 284992c (no process group, device hardcoded to
# cuda:0, sampler pinned to num_replicas=1) — `--nproc_per_node>1` would run N duplicate
# trainings on GPU 0, not a data-parallel job.

# Warm-start (weights only, strict=false) from the 4-day baseline — fresh series, not a true resume
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config mri_volume \
    checkpoint.resume_checkpoint_path=./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt

# ED-only fallback (matches original pre-multi-phase behavior)
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config mri_volume t_target_fixed=0

# Override
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config mri_volume optim.base_lr=1e-4
```

**Cluster submission**: `bash sbatch/train_mri_volume_reference.sh` — self-submits via embedded `sbatch`, sets `WANDB_MODE=online`, `max_epochs=200`, and has SLURM auto-requeue (SIGUSR1 → checkpoint-and-resume across the walltime). Head variants: `sbatch/train_mri_volume_{diffusion,bspline}.sh` (+ `_diffusion_s20{,_contz}.sh`). Resume modes (edit the vars at the top of the script):
- both `RESUME_FROM`/`CKPT_ONLY` empty (**default**) → **fresh-from-base VGGT-1B** (config's base-weights resume path, `strict=false`), fresh exp dir + new wandb run.
- `RESUME_FROM=<exp_dir>` → continue same exp_name + reuse same wandb run id (crash/requeue recovery).
- `CKPT_ONLY=<ckpt_path>` → **fresh** exp dir + new wandb run, loading from `<ckpt_path>` via `checkpoint.resume_checkpoint_path` (`strict=false`). **GOTCHA (docs/37):** `resume_checkpoint_path` does a **FULL resume** (weights + optimizer + `prev_epoch`), NOT weights-only — so pointing it at a full `checkpoint_last.pt` resumes at that epoch (e.g. 191) and, if `> max_epochs`, does **zero training**, at end-of-schedule LR. The base-weights path only *acts* weights-only because `vggt1b_base.pt` has no optimizer/epoch keys. For a real weights-only warm-start, first strip to `{"model": ...}`: `torch.save({'model': torch.load(ckpt)['model']}, out)` and point `CKPT_ONLY` at `out` (→ epoch 0, fresh optimizer + fresh warmup→5e-5→cosine schedule). Example: `scratch/checkpoints/4wok_weights_only.pt`.

**Configs** (`training/config/`):
- `mri_volume.yaml` — **active** unsupervised intensity pipeline. Inherits `mri_finetune.yaml` via `defaults:` and disables the deprecated DVF loss. Sets `config_name: "mri_volume"` (used as one of the wandb tags).
- `mri_finetune.yaml` — base/parent config (shared optimizer / data / freeze pattern); `mri_volume.yaml` inherits it and is what you actually run. Running `mri_finetune` directly still carries the deprecated supervised point-loss weights (`point.weight=1.0`), not the active intensity pipeline.
- `default.yaml` / `default_dataset.yaml` — templates inherited via `defaults:`.
- Legacy variants (`mri_finetune_*`, `mri_p001_overfit`, `mri_volume_overfit`) and their sbatch scripts now live under `_archive/legacy_configs/` and `_archive/legacy_sbatch/`.

**Key knobs:**
- `max_img_per_gpu: 20` → **fixed S=20 slot budget** (multi-frame; docs/28). Reduce on OOM (~37 GB/A40 at aggft). Was 12 (one-frame-per-plane) pre-multi-frame.
- `continuous_z: false` (default) | `true` → sample non-reference slots at **continuous physical z** (±`z_jitter`=0.5 off the integer plane + 2-plane interp); off ⇒ numerically identical to the discrete grid. docs/28.
- `t_target_fixed: null` (default → multi-phase, uniform per train call) | `0` (reproduces ED-only behavior) | any int K (force `t_target=K`).
- `t_target_phases: null` (default → all T phases) | list e.g. `[0,7]` → restrict the multi-phase target pool to that subset (train samples uniformly, val cycles it deterministically). **Mutually exclusive with `t_target_fixed`** (single-phase wins if both set).
- `optim.frozen_module_names` — **two regimes**, guarded by `tests/test_freeze_pattern.py`. (1) **Head-only** (`mri_finetune.yaml`, legacy target_t path): `["*patch_embed*", "*camera_token*", "*aggregator*"]` freezes the **entire** aggregator; only `point_head` trains (~32.65M/941M). (2) **aggft** (`mri_volume.yaml` reference default + `mri_volume_bspline/diffusion`): `["*patch_embed*"]` — attention blocks, `z_embedder`, `camera_token` (the reference anchor), and `point_head` all train (~2.8× slower, ~27 GB/A40). (The old `distributed.find_unused_parameters=true` requirement is **obsolete**: DDP was removed in `284992c`, the config key is gone, and nothing reads it — no-gradient params like the register token are simply ignored now.)
- `model.train_on_residual_dvf: true` → point head outputs Δ; `world_points = scanner_coords + Δ`.
- `logging.filmstrip_every_n_val_epochs: 5` → cadence for the multi-phase cardiac-cycle visualization.
- `data.augmentation.enable: true` (default since 2026-07-31) | `false` → opt out of GPU augmentation. `data.augmentation.tier: conservative|moderate|aggressive` (default `conservative`). See "Augmentation" below.

## Volume pipeline (one forward pass)

0. **Preprocess (cached, one-time per subject; native-z, docs/58).** monai `PersistentDataset` resamples all 12 phase NIfTIs' **in-plane** axes to `1.4` mm and crops/zero-pads to `256×256` (geometric center) — **Z is never resampled**: `Spacingd(pixdim=(1.4,1.4,0.0))` and `ResizeWithPadOrCropd(spatial_size=(256,256,-1))` keep each subject's own native slice pitch (`dz`, recorded by a `RecordSpacingD` transform) and native slice count (`D`), which both vary per subject (`dz` 5–12mm, `D` 5–21 across the pooled cohort). Prior to 2026-07-31 every subject was forced onto one shared `(256,256,12)` @ `(1.4,1.4,12.0)`mm grid — that only worked because CMRxRecon2024 alone is uniformly 12mm; see `docs/58` for why forcing non-12mm data onto that grid is a measured trap (25dB ceiling, not the ~120dB a correct native-pitch splat gets). Normalizes intensity against phase_00's 0.5/99.9 percentiles (computed over non-zero FOV voxels, excluding zero-padding), and stacks into one `(T=12, 256, 256, D)` float16 tensor + a `(256,256,D)` content mask (1=native FOV, 0=zero-pad in X/Y only — Z is never padded) + `dz_mm` (that subject's spacing). Cached on `/tmp/vggt-mri_${USER}_monai_cache/`. Pipeline + custom transforms live in `training/data/preprocess.py`.
1. **Sample (multi-frame, fixed S=20; docs/28).** **z is sampled only from within the geometric anatomy bbox** (in-FOV planes). Each sample = **full z-coverage** (every in-bbox plane ≥once) **+ uniform-random extra frames** filling the S=20 budget. Planes repeat; this keeps full V_canon coverage so the full-volume L1 loss stays valid. Per-slot `t` is a random phase **used only to extract slice content** — never a model input (`t_indices`/`target_t` inert).
   - **Reference default (`mri_volume`, `reference_slot=true`):** slot 0 = `(t_target, z_mid)` (the camera-token anchor = target-phase reference); the rest = coverage + LV extras.
   - **Train vs val:** identical sampler; train draws from global `random` (fresh each epoch), val from a private `random.Random(seq_index)` (reproducible, no global-RNG leak).
   - **Fixed-phase fallback:** `t_target_fixed=K` → every sample at phase K. **Continuous z:** `continuous_z=true` jitters non-reference slots off-grid (docs/28).
2. **Aggregator.** DINOv2 patch_embed + 24× alternating frame/global attention. Per-slot special token = sinusoidal embeddings: `z_embedder(z_norm)` (linear, `z_norm = z_mm / Z_HALF_MM` — **physical**, `Z_HALF_MM=90` is a fixed constant shared by every subject regardless of that subject's own `D`/`dz`, docs/58) always on; reference default adds the two-token `camera_token` (slot 0 = target-phase anchor), legacy path adds `t_embedder`+`target_t_embedder` instead (see Project + Key knobs). Frozen vs aggft per the freeze-pattern note.
3. **Point head (trainable, DPT).** Outputs per-pixel residual Δ (3 channels) + confidence (1, unused). `world_points = scanner_coords + Δ`, all in normalized [-1, 1] (x/y index-normalized over the fixed 256×256 grid; z is physical, see above).
4. **Splat.** `splat_to_volume(world_points, intensity, (D,256,256), z_scale)` → `V_canon`, where `D` is THIS subject's own native slice count and `z_scale = Z_HALF_MM/dz` is required (no default — a missed call site must crash, not silently compress the volume). Differentiable trilinear scatter; divides by accumulated coverage (`vggt/utils/splat.py`). **`splat_weight = intensity > 1e-3` is kept** — padded X/Y slots are all-zero, and the gate prevents their zero-intensity pixels from diluting V_canon if the model's Δ ever moves them into content planes.
5. **Loss.** `loss_volume = (V_canon - V_gt).abs().mean()` + `0.1 * TV(pos_pred)` — **full-volume L1**, no anatomy mask.

**Input slices:** each canonical `(256, 256)` slice is bilinear-resized to `518×518` for DINOv2 — **no letterbox, no padding** (the canonical slice is already square). `scanner_coords[py, px] = (px/517·2−1, py/517·2−1, z_norm)` where `z_norm = (z_i − (D−1)/2) · dz / Z_HALF_MM` — x/y are a pure geometric mapping identical for every subject; z is physical (mm-based), so it's also comparable across subjects despite `D`/`dz` varying. There is no `-2.0` invalid sentinel anymore; every pixel has a valid canonical coord.

**`V_gt`** = `phases[t_target]` from the cache (canonical frame, batch key `gt_target_volume`). **`anatomy_bbox`** = `(z0,z1,y0,y1,x0,x1)` geometric bbox of the content mask (used to restrict z sampling AND for the bbox metric). Both produced by `MRIDataset.get_data`.

## CMR data notes

Native cine shapes vary: W=256 fixed, H∈{162,204,246}, Z∈{6..14}, T=12. Spacing is **not** uniform: X median 1.3438 (range 1.3438–1.5781), Y median 1.3984 (range 1.3174–1.6423), Z always 8.0 — **but that 8.0 is slice THICKNESS, not center-to-center pitch**: the true pitch is 8 mm + 4 mm gap = **12 mm** (CMRxRecon2024 protocol; `info.csv` has no gap/position field). NIfTI affines were relabeled 8→12 on disk (`docs/27`); the canonical cube uses Z=12 mm. 28 unique spacing tuples across 301 subjects. Native FOV spans X 344–404, Y 215–404, Z 48–112 mm.

`MRIDataset` (`training/data/datasets/mri_dataset.py`) maps every subject onto a canonical grid whose **in-plane** extent is fixed (`1.4mm` spacing, `256×256` voxels, `358.4×358.4`mm) but whose **z-extent is native per subject** (docs/58, native-z, 2026-07-31): `D` = that subject's own slice count (5–21 across the pooled cohort), spacing = that subject's own `dz` (5–12mm) — **z is never resampled or padded**. So in-plane voxels have the same physical size for every subject; z does not — there is no shared cube depth. Subjects with FOV < 256×256 get zero-padded in X/Y only; subjects with FOV > that get center-cropped (the heart is always near the acquisition center, so cropping loses only periphery). The model only ever sees normalized `[-1,+1]` x/y and a physical z (`z_norm = z_mm/90`, same ruler for every subject — see "Volume pipeline" above).

**Axis-order gotcha:** monai/nibabel store volumes `(X, Y, Z)`; the splat consumes `(D, H, W) = (Z, Y, X)`. The single conversion site is the `permute(0, 3, 2, 1)` in `MRIDataset.get_data` right after the cache lookup — everything downstream is splat-order. Easy to break silently; tests in `test_canonical_invariants.py` guard it.

**Orientation = LPS everywhere.** Training forces `Orientationd(axcodes="LPS")` (`training/data/preprocess.py`), so the model only ever sees LPS-oriented hearts. **ALL data must be LPS — training, in-distribution val, AND every OOD inference/visualization adapter.** MIITT & OCMR gated NIfTIs are LPS-native; **ACDC is mixed (114 LPS / 36 LAS)**, so `ACDCGatedAdapter` (`inference/adapters/acdc.py`) reorients to LPS on load (nibabel `ornt_transform`/`apply_orientation`). When adding a new dataset/adapter, **check axcodes and reorient to LPS** — a mis-oriented (e.g. AP-flipped) heart still looks like a heart and silently degrades the model's LPS-trained anatomical priors (RV location, through-plane motion, EF) with NO crash. This burned us on ACDC (see the prove-it review); pairs with the "verify geometry from data, not headers" lesson.

`mri_mode: "axial"` means **native SAX z-slicing** — not anatomical axial. The slices are short-axis views.

**Slice order = APEX AT z0, for every subject** (standardised on disk 2026-07-31, docs/58 §10a/§10b). Sources originally disagreed — CMRx2023/24 and ACDC were base-first, M&Ms apex-first, and **CMRx2025-UIH was ~50/50 within one scanner** — so `tools/fix_slice_order.py` flipped **893 of 1343** subjects (`np.flip(axis=2)` on all 6 file types: the 12 `3d_recon` frames, `4d_recon`, `heart_seg`, `heart_roi`, and both `*_canonical`). Why it matters: `respiratory.py` applies a one-sided breathing shift along array axis D with **no anatomical anchor**, so storage order alone decides whether the simulated heart moves inferiorly (physiological) or superiorly (backwards) — it was backwards for 66% of the pool. Per-subject decisions: `result/slice_order_check/slice_order_decisions.csv`; revert via `scratch/data/_provenance/slice_order_fix.json` (**undo this BEFORE docs/56's slice-roll fix**).
- **Affines are deliberately NOT touched.** Every source already declares `+z = Superior` (axis-aligned, LPS axcodes), so flipping the *array* is what makes the header honest. ⚠️ Never "fix" this by editing the affine instead — `Orientationd(axcodes="LPS")` reorders axes by what the affine says and would silently flip every subject back, with no error.
- **"LPS" here is a stamped convention, not measured geometry.** A real SAX stack is double-oblique (slice normal = LV long axis); measured on M&Ms, |S| median is **0.402**, not 1.0. CMRx ships no orientation at all (SimpleITK default) and ACDC has `sform=qform=0`. Only M&Ms' true `slice_dir_ras` survives, in `convert_meta.json`.
- Adding data? Run `tools/render_slice_order_check.py` and flip to apex-at-z0 **per subject** — never by a per-source rule (CMRx2025 proves that wrong). The detector is validated **320/320** against M&Ms' shipped GT masks (`MNMs/MNMs1/*/<ID>/<ID>_sa_gt.nii.gz`) — use those, not the converter's `+S` rule, which is unsound at small `|S|`.
- Both fix tools now **refuse to run** if the state is already applied (`--force` overrides). `fix_slice_roll.py --revert` also refuses while the flip is applied: flip and roll do NOT commute (`F∘R_k = R_{-k}∘F`), so undoing the roll first leaves the stack **2 slices wrong, silently**, and that tool still doesn't touch `heart_seg`/`heart_roi`/`*_canonical` (they postdate it).
- ⚠️ **Anything derived from the cohort before 2026-07-31 12:19 is pre-flip stale** — `scratch/eval/`'s frozen bundle, the SVR baseline outputs, and the 851 subjects under `*_recon_v1_espirit_imagedomain/` (a different recon, not in `pooled.txt`). Rebuilding the eval bundle needs three `evaluation/engine/build_inputs/cmrxrecon.py` fixes FIRST or the bugs get baked into the new GT — see docs/58 §10b.

## Augmentation

GPU augmentation via `batchaug` (`training/data/gpu_aug.py`), **ON by default since 2026-07-31** (`data.augmentation.enable: true`, tier `conservative`), train-only (val never augments). One affine per subject, applied across all 12 T-phases + content mask so cardiac motion stays phase-consistent; the trainer then re-derives `gt_target_volume`, re-extracts input slices at the original (t,z) pairs, and recomputes `anatomy_bbox` (`scanner_coords` need no update — pure geometry). Tiers (in-plane only — no through-plane rotation, no elastic, since z is coarse/anisotropic vs the 1.4 mm in-plane grid; under native-z each subject keeps its own 5–12 mm pitch): `conservative` / `moderate` / `aggressive`, escalating affine + photometric (**flip**, rotate, translate/scale, gamma, bias field — Gaussian noise is commented out in all tiers). The W-axis **flip was re-enabled 2026-07-31** (docs/58 §10c): the training objective is exactly mirror-equivariant (measured) and 29% of the pooled CMRx cohort is mirrored on disk, so chirality-robustness is wanted. Verified D-agnostic under native-z (D=5/7/12/21), though `tests/test_gpu_aug.py` still only covers D=12. Visual proof: `tools/render_augmentation_examples.py` → `result/augmentation_examples/`.

**Respiratory-motion sim** (`training/data/respiratory.py`) is a SEPARATE toggle (`data.augmentation.respiratory.enable`), **ON by default in `mri_volume`** (the proven resp/z-only recipe, docs/05; inherited by `mri_volume_diffusion`/`mri_volume_bspline`; `mri_finetune` base defaults it off). Per-input-slice deform-then-reslice SI+AP shift (Lujan `sin^{2n}`), applied **after** affine and overwriting **only the input slices** — target/`scanner_coords`/`gt_target_volume`/`anatomy_bbox`/`phases` stay at the unshifted end-expiration reference, so the model learns to **correct** breathing (blind to `r`). Applies in **both train AND val** (unlike affine): train iid per epoch from a private generator (no global-RNG leak), val **deterministic per `seq_index`** (the new batch key carrying the val seed — reproducible corrupted→clean task). **Per-subject acquisition geometry (2026-07):** the tilt direction `θ ~ U(tilt_min_deg, tilt_max_deg)` (default 0–45°, replacing the old `direction_jitter_deg=30` undershoot) and azimuth `φ` are drawn **once per subject** (not per z-plane) — the SAX obliquity is fixed per scan; and the amplitude **scale** is per-subject (one lung capacity, `amplitude_breath_jitter` adds optional per-breath tidal wobble). Only breath **phase** `r` varies per z-plane (`group_by_burst`). `tilt_max_deg=None` falls back to legacy `direction_jitter_deg` (kept for the `tools/` scripts). `=0` tilt → pure SI+AP. Needs a retrain to benefit; A/B on `inference/run_rtfb.py`. Disabling ⇒ bit-identical to pre-respiratory. Visual proof: `tools/render_respiratory_examples.py` → `_html/06_*.html`. Design: `docs/01_respiratory_motion_simulation.md`.

## Architecture

```
VGGT (vggt/models/vggt.py) — ~941M total, base weights at ./scratch/base_weights/vggt1b_base.pt (download: huggingface.co/facebook/VGGT-1B/resolve/main/model.pt — NOT a regenerable cache, keep it)
├── Aggregator
│   ├── DINOv2 patch_embed (518² inputs, patch=14 → 37² tokens)    [FROZEN, ~304M]
│   ├── 24× frame_blocks + 24× global_blocks (alternating attn)    [FROZEN, ~605M]
│   ├── rope / camera_token / register_token                        [FROZEN, ~10K]
│   └── ZIndexEmbedder, TIndexEmbedder (sinusoidal Fourier)         [FROZEN, ~28K — see Key knobs]
└── point_head — DPT upsampler → 4-channel (Δ, conf)                [TRAINABLE, ~32.65M]

Camera / depth / track heads disabled in mri_volume config.
Trainable total = ~32.65M / 941M (point_head only) in the legacy head-only freeze; the reference default (`mri_volume`, aggft) also trains the 24×24 attention blocks + z_embedder + camera_token.
```

Checkpoints save the **full 941M state dict** (~3.8 GB each), not just the trainable head. Optimizer + scaler state included.

## Inference / inspection

```python
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
preds = model(images, batch=batch)  # batch needs: z_indices, t_indices, scanner_coords
# To use compute_volume_intensity_loss: batch must also include gt_target_volume (already the t_target phase; t_target itself is only used for per-phase logging, not the loss).
```

Tools:
- `tools/preview_canonical_preprocess.py` — sanity-check the canonical resample on shape-extreme subjects (min/max Z, min/max H, typical); native vs canonical mid-z slice + content-mask + bbox overlay → `result/canonical_preview/`.
- `tools/render_augmentation_examples.py` — per-op + combined aug variant PNGs and a cardiac-cycle GIF → `result/augmentation_examples/`.
- `tools/render_volume_example.py` — random val sample, per-z V_gt/V_canon/diff panel.
- `tools/test_sequential_sampling.py` — diagonal `(t=k+offset, z=k)` for one subject; PNGs to `result/`.
- `baselines/eval_all_baselines.py`, `baselines/eval_within_body_mask.py` — PSNR sweeps over the val set (identity-Δ floor, etc.).

**Where new scripts go** (sort by *reuse potential*, not temp-vs-permanent):
- **Throwaway** one-off probe / sanity-check you won't rerun → scratchpad dir, NOT the repo.
- **Might reuse, or an experiment script backing a `docs/` finding** → `tools/` (git-tracked; several `tools/exp_*`/`toy_*` are cited by docs as repro provenance).
- **`evaluation/` is OFF-LIMITS for auto-adding** — it holds only standing eval code we always run. **NEVER add anything to `evaluation/` on your own initiative; the user decides what goes there.** Write to `tools/` and ask.

**Evaluation & SVR baselines** (external datasets / occasional runs — not the training loop): the inference/eval harness lives in `inference/` (`run_cmrxrecon.py` in-distribution EF/Dice, `run_rtfb.py` real-time free-breathing inference, `adapters/`, `seg_metrics_cmrxrecon.py`); classical SVR baselines (NiftyMIC / NeSVoR / fetal_cmr_4d) live in `baselines/`. Rationale, protocol, results: `docs/24` + `docs/29–35` (index in `docs/README.md`). The frozen breathing-simulated baseline harness is now git-tracked in **`evaluation/`** (`evaluation/README.md`; the heavy data stays on gitignored GPFS via `evaluation/volumes` → `scratch/eval`). Standing analysis/figure scripts live in `evaluation/analysis/` — but per the off-limits rule above, **never add to `evaluation/` on your own initiative; write to `tools/` and ask.**

## Logging (wandb, project `vggt-mri`)

Metrics carry a `_full` / `_bbox` suffix: `_full` = whole `D×256×256` cube (D = that subject's native slice count, native-z), `_bbox` = subject's geometric content region. Equal for full-FOV subjects; for small-FOV subjects `_full` is inflated by padded zeros in X/Y (Z is never padded under native-z, so `_full`/`_bbox` differ less in Z than they used to) — **prefer `metric_psnr_3d_bbox` as the honest number** (SSIM is `_full` only). **Don't compare PSNR across the canonical-grid refactor OR the native-z refactor** — V_gt frame, normalization, and metric defs all changed at each; treat post-2026-07-31 runs as a fresh series from pre-native-z ones.

- **Startup:** identity-Δ baseline (Δ=0 splat over val) for full+bbox → `${log_dir}/baseline_identity.json`; baked into the `val_psnr_{full,bbox}/` metric names.
- **GT-referenced ship-decision metrics (val-only, `docs/38`):** `Val_Loss/metric_recov_frac_heart` (=(MSE_id−MSE_model)/(MSE_id−MSE_oracle) on the cardiac-motion ROI; oracle-normalized so it rescales the appearance wall out — `=1` ceiling, `<0` below floor), `metric_psnr_3d_static` (flat control) vs `metric_psnr_3d_motion` (heart), `metric_hole_frac_heart` (coverage<0.5 tripwire), and breathing `metric_resp_{slope,corr}_dz`/`_epe_dz_mm`/`_frac_deep_ignored` (predicted Δz vs exact `resp_disp_mm`). **Decision rule: a change wins iff recov_frac↑ & psnr_motion↑ WITHOUT hole_frac↑.** Heart ROI = `compute_motion_mask` (no segmentation). Gated val-only via `not pos_pred.requires_grad` ⇒ training bit-identical.
- **Per train visual step (every 100):** `Train_Visuals_Volume` (input/V_gt/V_canon/diff per z) + `Train_Visuals_DVF` (input + per-slot Δx/Δy/Δz).
- **Per val epoch:** `Val_Loss/*`; `Val_Visuals_subj{0,7}_{Volume,DVF}` (subj0 t=0=ED, subj7 t=7≈ES); per-phase `val_psnr_{full,bbox}/t{k}_n{n}_base{b}` + `/mean_n{n_total}` (multi-phase only; `n_total = limit_val_batches`=200 default → ~16-17/phase, since val revisits the 30 subjects at different target phases). `save_val_volumes` (default true) dumps per-subject pred+GT NIfTIs to `${log_dir}/val_volumes/` (~360 MB, overwritten each epoch; rank-0 only under DDP).
- **Every N val epochs (`filmstrip_every_n_val_epochs`, default 5):** `Val_Visuals_cardiac_cycle` (2×12 V_gt/V_canon grid, subj0) + `_gif` (12-frame beating-heart GIF).
- **Tags:** config name + phase mode (`multiphase` / `t{K}`). **Gating:** fixed-phase (`t_target_fixed=K`) skips the per-phase `val_psnr` panels and auto-caps `limit_val_batches` to one deterministic pass over val (more iters = redundant); multi-phase unaffected. All diagnostic logging is `try/except`-wrapped — never raises into training.

## SLURM

- Stagger mamba activations in array jobs: `sleep $((SLURM_ARRAY_TASK_ID * 15))`.
- Logs: `/home/minsukc/vggt/slurm_logs/`.
- **Monai cache is node-local `/tmp`, rebuilt per job** (`/tmp/vggt-mri_${USER}_monai_cache/`, ~55 MB/subject). Lazy first-epoch rebuild ~3–10 min for ~270 subjects, overlaps GPU compute. Intentionally not on GPFS — cached GPFS reads are ~18–20× slower than /tmp, so persisting would slow every epoch to save one rebuild.

## Git / branches (multi-agent hygiene)

Multiple agents/sessions share this repo's single working tree, so a bare `git switch`/`checkout`
with **uncommitted** changes drags them onto whatever branch HEAD lands on — that is how a
session's work can silently end up on another agent's branch. **Do branch-based work in a dedicated
`git worktree`, not by switching HEAD in place:** `git worktree add ../vggt-<task> -b cleanup/<task>`
gives that branch its own isolated working directory, so switches elsewhere can't contaminate it (and
vice-versa). When the work is merged, remove it: `git worktree remove ../vggt-<task>`. (Committing
promptly also avoids the carry-over, but the worktree is the real fix when several agents are active.)

## Local gotchas

- Don't pipe `torchrun` through `| tail -N` in background — buffering. Redirect to file: `... > /tmp/run.log 2>&1 &`, then `tail -F /tmp/run.log`.
- **Checkpoint loads auto-stage to node-local `/tmp`** (`vggt/utils/checkpoint_stage.py`, docs/50) — GPFS `torch.load` is ~266s vs ~5s from `/tmp`. Training stages only the immutable base/seed weights (`resume_checkpoint_path`, NOT the mutable requeue `checkpoint_last.pt`); inference (`inference/inference.py`, all `run_*.py`) stages every load (cache validated by size+mtime, so a re-saved ckpt is never served stale). Pure copy → byte-identical; any failure falls back to the original path.
- Initial VGGT-1B load takes ~9 min cold, ~1 min cached.
- Local pilots: `WANDB_MODE=offline`. The cluster scripts (`sbatch/train_mri_volume_*.sh`) set `WANDB_MODE=online`.
- Hydra custom resolvers (`rev_ts:`, `basename:`, `phase_mode:`) are registered in `training/launch.py`. For standalone `compose()`: `OmegaConf.register_new_resolver('rev_ts', lambda: '0')`; `OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))`; `OmegaConf.register_new_resolver('phase_mode', lambda t: 'multiphase' if t is None else f't{int(t)}')`.

## Testing

```bash
micromamba run -n svr python -m pytest tests/
```
Synthetic in-memory CMR dataset (`tests/conftest.py`, native W=64, H=60, Z=8, **T=12**) — no real data needed. T=12 matches the canonical pipeline's phase count; each test session gets an isolated monai cache dir (`monai_cache_dir` fixture) so the shared `/tmp` cache isn't polluted. Test files: `test_mri_dataset.py` (dataset contract), `test_preprocess.py` (canonical transforms + geometric bbox), `test_canonical_invariants.py` (cross-subject coord consistency, V_gt zero outside bbox, axis-order), `test_gpu_aug.py` (aug identity passthrough + shape preservation), `test_loss_bbox.py` (bbox vs full metrics), `test_splat.py`, `test_freeze_pattern.py`, `test_trainer_diagnostics.py`, `test_loss.py`, `test_resume.py` (requeue/resume).

## Docs

Research findings, design decisions, and experiment write-ups live in **`docs/`** (numbered,
e.g. `docs/01_respiratory_motion_simulation.md`) — separate from per-version implementation logs
in `version_history/`. **When you make a non-trivial design choice, run a research/literature
sweep, or design/run an experiment, record the choice AND the reasoning (why, sources, rejected
alternatives) as a numbered `docs/NN_*.md`** so future agents can understand *why*, not just
*what*. Keep CLAUDE.md pointers short and link out to the doc for detail.

**Every `docs/NN_*.md` MUST open with a `> **TL;DR & takeaway**` blockquote** before any other
content — a plain-language summary of the conclusion, key decision, and status. This top block is
**human-facing** (the reader skims it and stops); everything below it is the **agent-facing**
detailed record (process, numbers, sources, open questions). Write the TL;DR for someone who will
read *only* it.

**The doc index lives in `docs/README.md`** (one line per doc). Read it to see which doc to open;
add a pointer there when you create a new `docs/NN_*.md`. Don't list individual docs here.

## Future enhancements (not implemented)

Roadmap / parking lot lives in **`docs/36_roadmap_future_enhancements.md`** — none are in the
current pipeline. Headline direction = realistic real-time acquisition simulation (bSSFP transient,
single-shot artifacts, through-plane motion). Other entries: Option-B continuous-phase query,
blind-input-phase contract (`docs/04`), free-breathing respiratory embedder, phase-recovery
fallback, fully-unsupervised loss, UNet refiner/ablation on the splat, tagging k-space as in-plane
motion GT. Each has its own blockers — see the doc.
