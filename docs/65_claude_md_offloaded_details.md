# 65 — CLAUDE.md offloaded details (verbatim archive)

> **TL;DR & takeaway** — CLAUDE.md was trimmed on 2026-08-07 (~250 → ~140 lines). Nothing was deleted: every removed passage is preserved **verbatim** below, grouped by the CLAUDE.md section it came from. CLAUDE.md keeps the current rule + a pointer here (or to the owning numbered doc); this file holds the history, evidence, and parameter-level detail. If a trimmed CLAUDE.md line seems to be missing context, look it up here first.

Each subsection quotes the removed text exactly as it stood in CLAUDE.md on 2026-08-07. Owning docs (where the material is *also* covered, often in more depth) are noted per item.

---

## From "Project"

### flat-EF / target_t history (owning docs: 24, 25, 33)

Removed detail from the reference-slice conditioning paragraph:

> The model reads the target phase from slot-0's *image content* (`V_gt = phases[t_target]` = that slice's phase) — **not** a content-free `target_t` index, which regressed every patient's EF to the cohort mean (flat-EF; `use_t_pose_embedding`/`use_target_t_pose_embedding` OFF, `target_t_indices` inert). […] **EF recovery is confirmed on real final ckpts** (slope 0.77–0.79, honest Spearman ~0.55; the earlier "flat-EF" reading was an undertrained ckpt) — see `docs/24`, `docs/25`, `docs/33`. The legacy `target_t`-index path is gone (docs/25); `target_t_indices` is still emitted by the dataset but the model ignores it.

### The "4-day baseline" (owning doc: 37 for the resume gotcha)

> The "**4-day baseline**" (referenced below) is the prior ED-only run at `./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/` — 31+ dB PSNR at ED. Post-refactor it's available **weights-only as a warm-start seed** (via `CKPT_ONLY` / `resume_checkpoint_path`, `strict=false`; the current reference script defaults to fresh-from-base instead), **not a true resume**: input normalization + V_gt frame changed, so its memorized codes are stale and the old PSNR won't reproduce — treat the canonical-grid pipeline as a fresh-retrain series.

### Canonical-grid → native-z refactor history (owning doc: 58)

> **Canonical-grid refactor (2026-05-24), superseded by native-z (2026-07-31, docs/58).** Every subject's **in-plane** axes are resampled to a shared grid; input slices and V_gt both live in it. **Z is no longer resampled** — each subject keeps its own native slice pitch/count (`dz`, `D`). (Originally every subject was forced onto one shared `(256,256,12)` cube including Z — that was correct only because the sole active dataset, CMRxRecon2024, is uniformly 12mm; native-z is what makes the pooled multi-pitch cohort in docs/58 possible.) The old supervised-DVF path is **fully removed** — `MRIDataset` no longer reads DVF/mask NIfTIs and `gt_dvfs`/`scale_factors` are gone from the batch. DVF NIfTIs remain on disk for repro; legacy DVF tooling (`compute_cine_dvf_elastix.py`, carmen/elastix verification) plus OCMR/CMRx4DFlow2026 recon explorations and legacy configs/sbatch all live under `_archive/`.

Also removed from "Volume pipeline" step 0:

> Prior to 2026-07-31 every subject was forced onto one shared `(256,256,12)` @ `(1.4,1.4,12.0)`mm grid — that only worked because CMRxRecon2024 alone is uniformly 12mm; see `docs/58` for why forcing non-12mm data onto that grid is a measured trap (25dB ceiling, not the ~120dB a correct native-pitch splat gets).

---

## From "Setup"

### PYTHONPATH — verification evidence and per-directory census

Kept in CLAUDE.md: the command and the two-line reason. Removed evidence:

> (verified: `pip show vggt` finds nothing, and `vggt` is absent from `envs/svr_torch231.freeze.txt`, so it was never installed in the old env either) […] `training` is what makes the Hydra `_target_` short names resolve (`loss.MultitaskLoss`, `data.dynamic_dataloader.DynamicTorchDataset`, `train_utils.gradient_clip.GradientClipper`) — redundant for `launch.py` itself, but required by the 66 scripts outside `training/` that import `data.*`/`loss`/`trainer` directly (57 in `tools/`, 5 in `baselines/`, 4 in `evaluation/`; none in `inference/`). `tests/` needs neither: `tests/conftest.py` inserts both paths itself.

### torch-2.13 migration narrative (owning doc: 49; also the comment block at the bottom of `requirements.txt`)

> (upgraded 2026-07 from the frozen torch 2.3.1 + numpy 1.26.4 build — see the torch-2.13 upgrade doc). **monai** is `>=1.6,<1.7` (needs torch>=2.8); it was pinned `<1.5` only to hold torch at 2.3.1, a constraint now lifted. **numpy** is `>=2,<2.5` (upper bound = scipy/numba); the numpy-2 migration needed one source fix in `inference/adapters/base.py` (`np.percentile` on float32 now returns float32, which silently disabled a divide-by-zero guard → all-NaN OOD inputs). […] torch 2.13 now supplies the matched triton 3.7.1 that batchaug's triton backend wants, but `gpu_aug.py` still forces `batchaug.set_backend("pytorch")` at runtime for reproducibility, so triton stays dormant.

---

## From "Training"

### CKPT_ONLY full-resume gotcha — full text (owning doc: 37)

> **GOTCHA (docs/37):** `resume_checkpoint_path` does a **FULL resume** (weights + optimizer + `prev_epoch`), NOT weights-only — so pointing it at a full `checkpoint_last.pt` resumes at that epoch (e.g. 191) and, if `> max_epochs`, does **zero training**, at end-of-schedule LR. The base-weights path only *acts* weights-only because `vggt1b_base.pt` has no optimizer/epoch keys. For a real weights-only warm-start, first strip to `{"model": ...}`: `torch.save({'model': torch.load(ckpt)['model']}, out)` and point `CKPT_ONLY` at `out` (→ epoch 0, fresh optimizer + fresh warmup→5e-5→cosine schedule). Example: `scratch/checkpoints/4wok_weights_only.pt`.

### Config-flattening history (owning doc: 61)

> The old three-layer chain `default → mri_finetune → mri_volume` is **gone**. It was a trap: `mri_finetune` was a HALF config (no `loss.volume` ⇒ `MultitaskLoss(volume=None)` ⇒ `objective=0` ⇒ a silent zero-loss run), and the upstream `default` layer advertised values that were immediately overridden (`/YOUR/PATH/TO/CKPT`, `frozen_module_names: ["*aggregator*"]  # example`, `enable_point: False`, `compile_attention_blocks: False`) — so reading it gave the WRONG answer about the freeze pattern and the head. The flattening was verified by diffing Hydra's fully-resolved config before/after: **byte-identical** for all three configs. `exp_name`/`config_name` deliberately keep the `mri_volume` family name (log-dir + wandb continuity).

### Obsolete DDP `find_unused_parameters` note

> (The old `distributed.find_unused_parameters=true` requirement is **obsolete**: DDP was removed in `284992c`, the config key is gone, and nothing reads it — no-gradient params like the register token are simply ignored now.)

### Warm-start launch example (removed from the command block)

```bash
# Warm-start (weights only, strict=false) from the 4-day baseline — fresh series, not a true resume
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config default \
    checkpoint.resume_checkpoint_path=./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt
```

---

## From "CMR data notes"

### Slice-order standardisation — full block (owning docs: 56, 58 §10a/§10b)

> **Slice order = APEX AT z0, for every subject** (standardised on disk 2026-07-31, docs/58 §10a/§10b). Sources originally disagreed — CMRx2023/24 and ACDC were base-first, M&Ms apex-first, and **CMRx2025-UIH was ~50/50 within one scanner** — so `tools/fix_slice_order.py` flipped **893 of 1343** subjects (`np.flip(axis=2)` on all 6 file types: the 12 `3d_recon` frames, `4d_recon`, `heart_seg`, `heart_roi`, and both `*_canonical`). Why it matters: `respiratory.py` applies a one-sided breathing shift along array axis D with **no anatomical anchor**, so storage order alone decides whether the simulated heart moves inferiorly (physiological) or superiorly (backwards) — it was backwards for 66% of the pool. Per-subject decisions: `result/slice_order_check/slice_order_decisions.csv`; revert via `scratch/data/_provenance/slice_order_fix.json` (**undo this BEFORE docs/56's slice-roll fix**).
> - **Affines are deliberately NOT touched.** Every source already declares `+z = Superior` (axis-aligned, LPS axcodes), so flipping the *array* is what makes the header honest. ⚠️ Never "fix" this by editing the affine instead — `Orientationd(axcodes="LPS")` reorders axes by what the affine says and would silently flip every subject back, with no error.
> - **"LPS" here is a stamped convention, not measured geometry.** A real SAX stack is double-oblique (slice normal = LV long axis); measured on M&Ms, |S| median is **0.402**, not 1.0. CMRx ships no orientation at all (SimpleITK default) and ACDC has `sform=qform=0`. Only M&Ms' true `slice_dir_ras` survives, in `convert_meta.json`.
> - Adding data? Run `tools/render_slice_order_check.py` and flip to apex-at-z0 **per subject** — never by a per-source rule (CMRx2025 proves that wrong). The detector is validated **320/320** against M&Ms' shipped GT masks (`MNMs/MNMs1/*/<ID>/<ID>_sa_gt.nii.gz`) — use those, not the converter's `+S` rule, which is unsound at small `|S|`.
> - Both fix tools now **refuse to run** if the state is already applied (`--force` overrides). `fix_slice_roll.py --revert` also refuses while the flip is applied: flip and roll do NOT commute (`F∘R_k = R_{-k}∘F`), so undoing the roll first leaves the stack **2 slices wrong, silently**, and that tool still doesn't touch `heart_seg`/`heart_roi`/`*_canonical` (they postdate it).
> - ⚠️ **Anything derived from the cohort before 2026-07-31 12:19 is pre-flip stale** — `scratch/eval/`'s frozen bundle, the SVR baseline outputs, and the 851 subjects under `*_recon_v1_espirit_imagedomain/` (a different recon, not in `pooled.txt`). Rebuilding the eval bundle needs three `evaluation/engine/build_inputs/cmrxrecon.py` fixes FIRST or the bugs get baked into the new GT — see docs/58 §10b.

---

## From "Augmentation"

### GPU affine augmentation — full detail (owning docs: 46 §3, 58 §10c)

> GPU augmentation via `batchaug` (`training/data/gpu_aug.py`), **ON by default since 2026-07-31** (`data.augmentation.enable: true`, tier `moderate`), train-only (val never augments). One affine per subject, applied across all 12 T-phases + content mask so cardiac motion stays phase-consistent; the trainer then re-derives `gt_target_volume`, re-extracts input slices at the original (t,z) pairs, and recomputes `anatomy_bbox` (`scanner_coords` need no update — pure geometry). Tiers (in-plane only — no through-plane rotation, no elastic, since z is coarse/anisotropic vs the 1.4 mm in-plane grid; under native-z each subject keeps its own 5–12 mm pitch): `conservative` / `moderate` / `aggressive`, escalating affine + photometric (rotate, translate/scale, gamma, bias field — Gaussian noise is commented out in all tiers). The W-axis **flip is aggressive-tier-only** (2026-08-01): it was briefly on in every tier (2026-07-31, docs/58 §10c — the objective is exactly mirror-equivariant and 29% of the pooled CMRx cohort is mirrored on disk), but `moderate` is the arm docs/46 §3 C2 measured and shipped and that arm had no flip, so flip stays out of the default. Verified D-agnostic under native-z (D=5/7/12/21), though `tests/test_gpu_aug.py` still only covers D=12. Visual proof: `tools/render_augmentation_examples.py` → `result/augmentation_examples/`.

### Respiratory-motion sim — full detail (owning docs: 01, 05)

> **Respiratory-motion sim** (`training/data/respiratory.py`) is a SEPARATE toggle (`data.augmentation.respiratory.enable`), **ON by default in `default.yaml`** (the proven resp/z-only recipe, docs/05; inherited by `exp_bspline`). Per-input-slice deform-then-reslice SI+AP shift (Lujan `sin^{2n}`), applied **after** affine and overwriting **only the input slices** — target/`scanner_coords`/`gt_target_volume`/`anatomy_bbox`/`phases` stay at the unshifted end-expiration reference, so the model learns to **correct** breathing (blind to `r`). Applies in **both train AND val** (unlike affine): train iid per epoch from a private generator (no global-RNG leak), val **deterministic per `seq_index`** (the new batch key carrying the val seed — reproducible corrupted→clean task). **Per-subject acquisition geometry (2026-07):** the tilt direction `θ ~ U(tilt_min_deg, tilt_max_deg)` (default 0–45°, replacing the old `direction_jitter_deg=30` undershoot) and azimuth `φ` are drawn **once per subject** (not per z-plane) — the SAX obliquity is fixed per scan; and the amplitude **scale** is per-subject (one lung capacity, `amplitude_breath_jitter` adds optional per-breath tidal wobble). Only breath **phase** `r` varies per z-plane (`group_by_burst`). `tilt_max_deg=None` falls back to legacy `direction_jitter_deg` (kept for the `tools/` scripts). `=0` tilt → pure SI+AP. Needs a retrain to benefit; A/B on `inference/run_rtfb.py`. Disabling ⇒ bit-identical to pre-respiratory. Visual proof: `tools/render_respiratory_examples.py` → `_html/06_*.html`. Design: `docs/01_respiratory_motion_simulation.md`.

---

## From "Inference / inspection"

### `eval_all_baselines.py` archival note

> (`eval_all_baselines.py` was archived 2026-08-01: it read the `world_points` batch key that vanished with the supervised-DVF removal, so it could not run, and its elastix-vs-carmen arm compared two identical batches because `dvf_dirname` was already being ignored.)

---

## From "Logging"

### Metric names, panels, and gating — full detail (owning docs: 38, 60)

> - **Startup:** identity-Δ baseline (Δ=0 splat over val) for full+bbox → `${log_dir}/baseline_identity.json`; baked into the `val_psnr_{full,bbox}/` metric names.
> - **GT-referenced ship-decision metrics (val-only, `docs/38`):** `Val_Loss/metric_recov_frac_heart` (=(MSE_id−MSE_model)/(MSE_id−MSE_oracle) on the cardiac-motion ROI; oracle-normalized so it rescales the appearance wall out — `=1` ceiling, `<0` below floor), `metric_psnr_3d_static` (flat control) vs `metric_psnr_3d_motion` (heart), `metric_hole_frac_heart` (coverage<0.5 tripwire), and breathing `metric_resp_{slope,corr}_dz`/`_epe_dz_mm`/`_frac_deep_ignored` (predicted Δz vs exact `resp_disp_mm`). Heart ROI = `compute_motion_mask` (no segmentation). Gated val-only via `not pos_pred.requires_grad` ⇒ training bit-identical.
> - **Per train visual step (every 100):** `Train_Visuals_Volume` (input/V_gt/V_canon/diff per z) + `Train_Visuals_DVF` (input + per-slot Δx/Δy/Δz).
> - **Per val epoch:** `Val_Loss/*`; `Val_Visuals_subj{0,7}_{Volume,DVF}` (subj0 t=0=ED, subj7 t=7≈ES); per-phase `val_psnr_{full,bbox}/t{k}_n{n}_base{b}` + `/mean_n{n_total}` (multi-phase only; `n_total = limit_val_batches`=200 default → ~16-17/phase, since val revisits the 30 subjects at different target phases). `save_val_volumes` (default true) dumps per-subject pred+GT NIfTIs to `${log_dir}/val_volumes/` (~360 MB, overwritten each epoch; rank-0 only under DDP).
> - **Every N val epochs (`filmstrip_every_n_val_epochs`, default 3):** `Val_Visuals_cardiac_cycle` (2×12 V_gt/V_canon grid, subj0) + `_gif` (12-frame beating-heart GIF).
> - **Tags:** config name + phase mode (`multiphase` / `t{K}`). **Gating:** fixed-phase (`t_target_fixed=K`) skips the per-phase `val_psnr` panels and auto-caps `limit_val_batches` to one deterministic pass over val (more iters = redundant); multi-phase unaffected. All diagnostic logging is `try/except`-wrapped — never raises into training.

---

# Second trim pass (2026-08-07, same day) — full-text archive of compressed paragraphs

The passages below were **compressed, not removed** — CLAUDE.md keeps a shorter statement of each rule. The full original wording is preserved here.

## From "Project" — research goal (full)

> **Research goal:** enable fast real-time free-breathing cine by reconstructing the full 3D heart volume at any target cardiac phase from a *few scattered single-frame-per-slice* acquisitions (ideally one frame/slice), instead of the slow many-frames-per-slice + retrospective-sort/SVR route. No real-time training data exists, so we **simulate** the sparse scattered acquisition from gated breath-hold CMRxRecon2024 cine (each input slice = one frame at an arbitrary (phase t, z-depth)) + motion aug, and aim to generalize to true real-time cine. Currently *only* the scattered sampling + in-plane aug are simulated; realistic acquisition physics (bSSFP transient, single-shot artifacts, respiratory motion) is aspirational — see Future enhancements. **Target inference information contract:** at the one-frame-per-slice extreme the model is assumed to know only `z` per input slice — input cardiac `t` and respiratory `r` are *unavailable* (no ECG / no respiratory device / no self-gating); target-phase *queries* stay free. Design stance, not yet implemented — see `docs/04_inference_information_contract.md`.

## From "Training" — cluster submission (full)

> **Cluster submission**: `bash sbatch/train_mri_volume_reference.sh` — self-submits via embedded `sbatch`, sets `WANDB_MODE=online`, `max_epochs=200`, and has SLURM auto-requeue (SIGUSR1 → checkpoint-and-resume across the walltime). Head variants: `sbatch/train_mri_volume_{diffusion,bspline}.sh` (+ `_diffusion_s20{,_contz}.sh`).

## From "Key knobs" — `img_nums` and freeze bullets (full)

> - `img_nums: [20, 20]` → **slot BUDGET/cap, not a slot count.** With `one_frame_per_slice: true` (the default) the dataset sets **S = this subject's own in-FOV plane count**, and under native-z z is never padded, so **S == D exactly** (5–21 across the pooled cohort). `img_nums` only bounds it — `get_data` raises if a subject needs more (docs/59 F19). `max_img_per_gpu` no longer exists (deleted, docs/59 F9); **to cut memory, cut D or the model, not this knob** — batch size is pinned to 1 in `dynamic_dataloader.py` because same-D-different-pitch subjects collate silently.
> - `optim.frozen_module_names` — **two regimes**, guarded by `tests/test_freeze_pattern.py`. (1) **Head-only** (legacy, no longer in any shipped config): `["*patch_embed*", "*camera_token*", "*aggregator*"]` freezes the **entire** aggregator; only `point_head` trains (~32.65M/941M). (2) **aggft** (`default.yaml` + `exp_bspline`): `["*patch_embed*"]` — attention blocks, `z_embedder`, `camera_token` (the reference anchor), and `point_head` all train (~2.8× slower, ~27 GB/A40).

## From "Volume pipeline" — steps 0, 1, and Input slices (full)

> 0. **Preprocess (cached, one-time per subject; native-z, docs/58).** monai `PersistentDataset` resamples all 12 phase NIfTIs' **in-plane** axes to `1.4` mm and crops/zero-pads to `256×256` (geometric center) — **Z is never resampled**: `Spacingd(pixdim=(1.4,1.4,0.0))` and `ResizeWithPadOrCropd(spatial_size=(256,256,-1))` keep each subject's own native slice pitch (`dz`, recorded by a `RecordSpacingD` transform) and native slice count (`D`), which both vary per subject (`dz` 5–12mm, `D` 5–21 across the pooled cohort; why z must stay native: docs/58). Normalizes intensity against phase_00's 0.5/99.9 percentiles (computed over non-zero FOV voxels, excluding zero-padding), and stacks into one `(T=12, 256, 256, D)` float16 tensor + a `(256,256,D)` content mask (1=native FOV, 0=zero-pad in X/Y only — Z is never padded) + `dz_mm` (that subject's spacing). Cached on `/tmp/vggt-mri_${USER}_monai_cache/`. Pipeline + custom transforms live in `training/data/preprocess.py`.
> 1. **Sample (ONE frame per slice; `one_frame_per_slice: true` is the default).** **z is sampled only from within the geometric anatomy bbox** (in-FOV planes) — and under native-z that bbox spans the whole stack, so **S == D**, this subject's own plane count (5–21), *not* a fixed 20. Every plane appears **exactly once**, no repeats, no extra frames: the sparse one-frame-per-slice extreme the research goal targets. Full z-coverage keeps V_canon complete so the full-volume L1 loss stays valid. Per-slot `t` is a random phase **used only to extract slice content** — never a model input (`t_indices`/`target_t` inert). (Set `one_frame_per_slice: false` for the legacy fixed-S multi-frame sampler of docs/28: full coverage + uniform-random extras filling the `img_nums` budget.)
>    - **Reference default (`mri_volume`, `reference_slot=true`):** slot 0 = `(t_target, z_mid)` (the camera-token anchor = target-phase reference); the rest = coverage + LV extras.
>    - **Train vs val:** identical sampler; train draws from global `random` (fresh each epoch), val from a private `random.Random(seq_index)` (reproducible, no global-RNG leak).
>    - **Fixed-phase fallback:** `t_target_fixed=K` → every sample at phase K. **Continuous z:** `continuous_z=true` jitters non-reference slots off-grid (docs/28).
>
> **Input slices:** each canonical `(256, 256)` slice is bilinear-resized to `518×518` for DINOv2 — **no letterbox, no padding** (the canonical slice is already square). `scanner_coords[py, px] = (px/517·2−1, py/517·2−1, z_norm)` where `z_norm = (z_i − (D−1)/2) · dz / Z_HALF_MM` — x/y are a pure geometric mapping identical for every subject; z is physical (mm-based), so it's also comparable across subjects despite `D`/`dz` varying. There is no `-2.0` invalid sentinel anymore; every pixel has a valid canonical coord.

## From "CMR data notes" — native shape/spacing stats and LPS orientation (full)

> Native cine shapes vary: W=256 fixed, H∈{162,204,246}, Z∈{6..14}, T=12. Spacing is **not** uniform: X median 1.3438 (range 1.3438–1.5781), Y median 1.3984 (range 1.3174–1.6423), Z always 8.0 — **but that 8.0 is slice THICKNESS, not center-to-center pitch**: the true pitch is 8 mm + 4 mm gap = **12 mm** (CMRxRecon2024 protocol; `info.csv` has no gap/position field). NIfTI affines were relabeled 8→12 on disk (`docs/27`); the canonical cube uses Z=12 mm. 28 unique spacing tuples across 301 subjects. Native FOV spans X 344–404, Y 215–404, Z 48–112 mm.
>
> **Orientation = LPS everywhere.** Training forces `Orientationd(axcodes="LPS")` (`training/data/preprocess.py`), so the model only ever sees LPS-oriented hearts. **ALL data must be LPS — training, in-distribution val, AND every OOD inference/visualization adapter.** MIITT & OCMR gated NIfTIs are LPS-native; **ACDC is mixed (114 LPS / 36 LAS)**, so `ACDCGatedAdapter` (`inference/adapters/acdc.py`) reorients to LPS on load (nibabel `ornt_transform`/`apply_orientation`). When adding a new dataset/adapter, **check axcodes and reorient to LPS** — a mis-oriented (e.g. AP-flipped) heart still looks like a heart and silently degrades the model's LPS-trained anatomical priors (RV location, through-plane motion, EF) with NO crash. This burned us on ACDC (see the prove-it review); pairs with the "verify geometry from data, not headers" lesson.

## From "Inference / inspection" — tools list and eval/baselines paragraph (full)

> Tools:
> - `tools/preview_canonical_preprocess.py` — sanity-check the canonical resample on shape-extreme subjects (min/max Z, min/max H, typical); native vs canonical mid-z slice + content-mask + bbox overlay → `result/canonical_preview/`.
> - `tools/render_augmentation_examples.py` — per-op + combined aug variant PNGs and a cardiac-cycle GIF → `result/augmentation_examples/`.
> - `tools/render_volume_example.py` — random val sample, per-z V_gt/V_canon/diff panel.
> - `tools/test_sequential_sampling.py` — diagonal `(t=k+offset, z=k)` for one subject; PNGs to `result/`.
> - `baselines/eval_within_body_mask.py` — PSNR sweep over the val set (identity-Δ floor, etc.). (`eval_all_baselines.py` archived 2026-08-01, could no longer run — docs/65.)
>
> **Evaluation & SVR baselines** (external datasets / occasional runs — not the training loop): the inference/eval harness lives in `inference/` (`run_cmrxrecon.py` in-distribution EF/Dice, `run_rtfb.py` real-time free-breathing inference, `adapters/`, `seg_metrics_cmrxrecon.py`); classical SVR baselines (NiftyMIC / NeSVoR / fetal_cmr_4d) live in `baselines/`. Rationale, protocol, results: `docs/24` + `docs/29–35` (index in `docs/README.md`). The frozen breathing-simulated baseline harness is now git-tracked in **`evaluation/`** (`evaluation/README.md`; the heavy data stays on gitignored GPFS via `evaluation/volumes` → `scratch/eval`). Standing analysis/figure scripts live in `evaluation/analysis/` — but per the off-limits rule above, **never add to `evaluation/` on your own initiative; write to `tools/` and ask.**

## From "Logging" — on-disk file descriptions (full)

> - `run_meta.jsonl` — one line **per process launch** (git sha+dirty, config, split/manifest md5, cohort sizes, wandb id, SLURM job/node, `resumed_from_epoch`). A requeued run has several lines.
> - `metrics.jsonl` — every scalar (mirrored from `Trainer._log_scalar`, the single scalar chokepoint).
> - `val_per_subject.csv` — one row per val sample: subject, source, `D`, `dz`, `t_target` + every `metric_*`. **This exists nowhere else** — B=1 means these are per-subject values that the AverageMeter otherwise discards. Join to `training/splits/manifest.csv` for vendor/pathology/centre.
> - `baseline_identity.json` — per-phase **and per-subject** identity floors. Per-subject PSNR is *not* comparable across this cohort (the ceiling moves with `D`/`dz`/FOV) — normalise by these first.
>
> Requeue replays steps, so duplicate `(name, step)` rows are expected; `load_run` dedupes keep-last. **Only the 8 image panels (filmstrip GIF, ED/ES, DVF, motion mask, aug, lookup) are wandb-only.**
>
> Metrics carry a `_full` / `_bbox` suffix: `_full` = whole `D×256×256` cube (D = that subject's native slice count, native-z), `_bbox` = subject's geometric content region. Equal for full-FOV subjects; for small-FOV subjects `_full` is inflated by padded zeros in X/Y (Z is never padded under native-z, so `_full`/`_bbox` differ less in Z than they used to) — **prefer `metric_psnr_3d_bbox` as the honest number** (SSIM is `_full` only). **Don't compare PSNR across the canonical-grid refactor OR the native-z refactor** — V_gt frame, normalization, and metric defs all changed at each; treat post-2026-07-31 runs as a fresh series from pre-native-z ones.

## From "Git / branches" (full)

> Multiple agents/sessions share this repo's single working tree, so a bare `git switch`/`checkout` with **uncommitted** changes drags them onto whatever branch HEAD lands on — that is how a session's work can silently end up on another agent's branch. **Do branch-based work in a dedicated `git worktree`, not by switching HEAD in place:** `git worktree add ../vggt-<task> -b cleanup/<task>` gives that branch its own isolated working directory, so switches elsewhere can't contaminate it (and vice-versa). When the work is merged, remove it: `git worktree remove ../vggt-<task>`. (Committing promptly also avoids the carry-over, but the worktree is the real fix when several agents are active.)

## From "Local gotchas" — checkpoint staging (full)

> - **Checkpoint loads auto-stage to node-local `/tmp`** (`vggt/utils/checkpoint_stage.py`, docs/50) — GPFS `torch.load` is ~266s vs ~5s from `/tmp`. Training stages only the immutable base/seed weights (`resume_checkpoint_path`, NOT the mutable requeue `checkpoint_last.pt`); inference (`inference/inference.py`, all `run_*.py`) stages every load (cache validated by size+mtime, so a re-saved ckpt is never served stale). Pure copy → byte-identical; any failure falls back to the original path.

## From "Testing" — test-file inventory (full)

> Synthetic in-memory CMR dataset (`tests/conftest.py`, native W=64, H=60, Z=8, **T=12**) — no real data needed. T=12 matches the canonical pipeline's phase count; each test session gets an isolated monai cache dir (`monai_cache_dir` fixture) so the shared `/tmp` cache isn't polluted. Test files: `test_mri_dataset.py` (dataset contract), `test_preprocess.py` (canonical transforms + geometric bbox), `test_canonical_invariants.py` (cross-subject coord consistency, V_gt zero outside bbox, axis-order), `test_gpu_aug.py` (aug identity passthrough + shape preservation), `test_loss_bbox.py` (bbox vs full metrics), `test_splat.py`, `test_freeze_pattern.py`, `test_trainer_diagnostics.py`, `test_loss.py`, `test_resume.py` (requeue/resume).

## From "Docs" + "Future enhancements" (full)

> Research findings, design decisions, and experiment write-ups live in **`docs/`** (numbered, e.g. `docs/01_respiratory_motion_simulation.md`) — separate from per-version implementation logs in `version_history/`. **When you make a non-trivial design choice, run a research/literature sweep, or design/run an experiment, record the choice AND the reasoning (why, sources, rejected alternatives) as a numbered `docs/NN_*.md`** so future agents can understand *why*, not just *what*. Keep CLAUDE.md pointers short and link out to the doc for detail.
>
> **Every `docs/NN_*.md` MUST open with a `> **TL;DR & takeaway**` blockquote** before any other content — a plain-language summary of the conclusion, key decision, and status. This top block is **human-facing** (the reader skims it and stops); everything below it is the **agent-facing** detailed record (process, numbers, sources, open questions). Write the TL;DR for someone who will read *only* it.
>
> Roadmap / parking lot lives in **`docs/36_roadmap_future_enhancements.md`** — none are in the current pipeline. Headline direction = realistic real-time acquisition simulation (bSSFP transient, single-shot artifacts, through-plane motion). Other entries: Option-B continuous-phase query, blind-input-phase contract (`docs/04`), free-breathing respiratory embedder, phase-recovery fallback, fully-unsupervised loss, UNet refiner/ablation on the splat, tagging k-space as in-plane motion GT. Each has its own blockers — see the doc.
