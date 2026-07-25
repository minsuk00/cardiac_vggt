# 48 — Removing the original-VGGT SfM stack + retiring the legacy target_t path

> **TL;DR & takeaway** (2026-07-24)
>
> A large dead-code cleanup: deleted the inherited original-VGGT multi-view-SfM machinery
> (camera / depth / track heads + their losses + the supervised-DVF point loss) **and** retired
> the deprecated **target_t-index** conditioning (`TIndexEmbedder` / `target_t_embedder`, replaced
> long ago by reference-slice conditioning). **Net −2,742 lines** across the model, loss, configs,
> and tests. The hard constraint was that the active `mri_volume` pipeline stay **byte-identical**
> (same model `state_dict`, same forward, same loss for a fixed batch) — which was **verified**, not
> assumed. All of it lives on branch **`cleanup/remove-sfm-stack`** (base commit `4b55619`), not yet
> committed at time of writing. Full test suite green (the only tests removed were ones that
> exercised the deleted code). **If you're looking for camera/track/depth heads, the target_t
> embedder, `compute_point_loss`, or `default_dataset.yaml` — they're gone on purpose; `git show 4b55619:<path>` to recover.**

## Why

`vggt/` is Meta's VGGT (a multi-view Structure-from-Motion transformer) adapted for cardiac-MRI
slice-to-volume reconstruction. The active pipeline (`mri_volume.yaml`) uses only the **point head**
+ **reference-slice conditioning** (the native `camera_token` anchor + `z_embedder`). Everything
from the original SfM use case was still present but dead:

- **camera / depth / track heads** — disabled via `enable_*=false`, so `None` at construction.
- **camera / depth losses + the supervised-DVF `compute_point_loss`** — never reached (all non-volume
  weights are 0 and the model emits no `pose_enc_list`/`depth`/`track` keys).
- **the target_t-index conditioning** (`TIndexEmbedder`, `t_embedder`, `target_t_embedder`,
  `use_t_pose_embedding`, `use_target_t_pose_embedding`) — the content-free phase index that
  regressed every patient's EF to the cohort mean (docs/24, docs/25). Replaced by the reference
  slice; flags were off, so the embedders were never even instantiated.

This was ~52 % of `vggt/` and ~52 % of `training/loss.py` carried as inherited weight.

## The byte-identical guarantee (how it was proven, not assumed)

Every removed piece is provably absent from the `mri_volume` model or unreachable in its forward:
- disabled heads are `None` → `nn.Module` never registers them → **not in `state_dict()`**;
- the target_t embedders are gated behind `if use_t_pose_embedding:` (off) → **never constructed** →
  not in `state_dict`, and their init consumes **no RNG** (so removing the blocks doesn't shift
  parameter initialization of anything else);
- the 12 dead loss functions form a **closed call-subgraph** with zero reachability from
  `compute_volume_intensity_loss` and **zero external callers** repo-wide.

**Verification harness** (kept in the session scratchpad, not committed):
- `verify_byte_identical.py` — fixed-seed CPU driver. Runs `compute_volume_intensity_loss` +
  `MultitaskLoss` over a comprehensive batch (main L1, inline TV, full/bbox metrics, motion-mask +
  seg + resp val-metrics), and builds a DINOv2-free tiny `Aggregator` with the `mri_volume` flags —
  captures **138 tensors** (loss dicts + aggregator `state_dict` + forward). Compared BEFORE (a
  `git worktree` at HEAD) vs AFTER with **`torch.equal`** (bit-exact): **BYTE-IDENTICAL** at every
  step.
- `config_resolve.py` — composes+resolves `mri_volume`/`mri_volume_bspline`/`mri_volume_diffusion`
  in both trees and diffs the flattened resolved config. Result: the **only** keys removed are the
  intended dead set (`loss.camera/depth/point/track`, `model.enable_camera/depth/track`,
  `use_t/target_t_pose_embedding`, a stray root-level `optimizer.lr`, and 4 inert `color_jitter`
  leaves that `augmentation.py` backfills to identical values with `p=0.0`). **Zero keys added,
  zero used-value changes.** In particular `optim.optimizer` + the LR schedule are untouched.

The key design decision that preserved byte-identity across ~85 call sites: **`VGGT.__init__` and
`Aggregator.__init__` got `**kwargs`.** 42 files pass `enable_camera=`/`enable_track=` and 43 pass
`use_t/target_t_pose_embedding=` (incl. active `inference/inference.py`, baselines, ~45 tools); hard
-removing the params would `TypeError` all of them. `**kwargs` absorbs the retired flags everywhere.

## What was edited (branch `cleanup/remove-sfm-stack`, base `4b55619`, 2026-07-24)

**Modified**
- `training/loss.py` (1122 → 515 lines): deleted `compute_camera_loss`, `camera_loss_single`,
  `compute_point_loss`, `tv_loss` (the standalone fn — **not** the inline TV), `compute_depth_loss`,
  `regression_loss`, `gradient_loss_multi_scale_wrapper`, `normal_loss`, `gradient_loss`,
  `point_map_to_normal`, `filter_by_quantile`, `torch_quantile`; slimmed `MultitaskLoss` to
  `__init__(self, volume=None, **kwargs)` + volume-only `forward`; dropped now-unused imports
  (`math.ceil/floor`, `torch.nn.functional`, `check_and_fix_inf_nan`, `extri_intri_to_pose_encoding`).
  **Kept**: `compute_motion_mask`, `diffusion_loss_l2`, `compute_volume_intensity_loss` (incl. its
  inline TV at ~`:375`, still driven by `tv_weight`).
- `vggt/models/vggt.py`: dropped `CameraHead`/`TrackHead` imports; added `**kwargs`; removed
  `enable_camera/depth/track` + `use_t/target_t_pose_embedding` params, the `camera_head`/`depth_head`/
  `track_head` attrs, and their forward branches. **Kept** `point_head`, refiner, reference/z flags.
- `vggt/models/aggregator.py`: deleted the `TIndexEmbedder` class + `t_embedder`/`target_t_embedder`
  construction + the `use_t`/`use_target_t` forward branches; added `**kwargs`; removed the two flag
  params. **Kept** `ZIndexEmbedder`, `camera_token`, `register_token` (reference-slice path).
- `training/trainer.py`: removed `_apply_batch_repetition` (SfM flip-augment, `repeat_batch` always
  false) and the `normalize_points` branch of `_process_batch` (→ passthrough); dropped the dead
  `normalize_camera_extrinsics_and_points_batch` import. (Also earlier this session: removed dead
  imports `torchvision`/`DictConfig`/`ListConfig`/`OmegaConf`, a duplicate loss import, and an
  unread `data_times` accumulator.) ~~**Left** `batch_size = data["extrinsics"].shape[0]`~~ — rebased
  to `data["images"]` once the fillers were stripped (docs/50).
- `training/config/default.yaml`: removed the `defaults: [default_dataset]` link, the CO3D `data`
  block, camera/depth `scalar_keys_to_log`, `loss.camera/depth/point/track`, camera/depth
  `gradient_clip` groups, and `model.enable_camera/depth/track`.
- `training/config/mri_finetune.yaml`: removed top-level + `model.` `use_t/target_t_pose_embedding`,
  the whole legacy `loss:` (camera/depth/point) block, and the stray dead root-level `optimizer:`.
- `training/config/mri_volume.yaml`: removed `use_t/target_t_pose_embedding` and
  `loss.camera/depth/point`. **Kept** `loss.volume` (incl. `tv_weight: 0.1`).
- `tests/test_freeze_pattern.py`, `tests/test_reference_conditioning.py`,
  `tests/test_trainer_diagnostics.py`: trimmed assertions/builders that referenced removed
  params/methods (the reference-slice tests all stay and still pass).

**Deleted** — `training/config/default_dataset.yaml`; `tests/test_target_t_embedder.py`;
`vggt/heads/camera_head.py`, `vggt/heads/track_head.py`, `vggt/heads/track_modules/` (5 files),
`vggt/utils/{load_fn,helper,visual_track}.py`.

**Archived** — the 4 target_t-only diagnostic tools moved to `tools/_legacy/`
(`measure_acdc_ssim_ef.py`, `render_miitt_filmstrip_refiner.py`, `measure_dvf_z_correction_t59.py`,
`target_t_3row.py`) + a `tools/_legacy/README.md`. They depend on the removed target_t machinery
and are non-functional; kept for provenance.

**Silent-degrade footgun (`**kwargs` side effect).** Any caller that still passes
`use_target_t_pose_embedding=True` no longer builds a `target_t_embedder` — the flag lands in
`**kwargs` and is dropped, so loading a legacy target_t checkpoint runs *without* target-phase
conditioning (degraded output, **no crash**). This is intended (the path is retired), but it is
silent. Affected non-active callers: `inference/inference.py`'s **legacy** `load_rtfb_model` (the
active `load_rtfb_model_reference` uses `=False` and is unaffected), `baselines/eval_all_baselines.py`,
`baselines/eval_within_body_mask.py`, and ~18 `tools/*` scripts. None is on the active `mri_volume`
training path or the active reference-inference path. If you ever need to reproduce the old target_t
baselines, `git checkout 4b55619 -- <path>`.

## Deferred (not done here — a clean follow-on)

- ~~**`vggt/dependency/` tree** (SfM/COLMAP/tracker code)~~ **DONE (`d88bd03`, 2026-07-24).** Pruned the
  SfM helpers from live `geometry.py` (removed the `dependency.distortion` import + `img_from_cam` etc.),
  then deleted the tree. Single-GPU **DDP removal** (`284992c`) also landed in the same effort.
- **Dataset filler keys**: ~~`MRIDataset.get_data` still emits dummy `extrinsics`/…~~ **DONE (docs/50,
  2026-07-24).** Stripped all 8 filler keys, deleted the `co3d`/`vkitti`/`track_util` orphans, and
  rebased `batch_size` to `data["images"]`. Byte-identical.
- ~~**`training/trainer.py` size** (~2.3k lines): ~1k lines are wandb/matplotlib viz methods…~~
  **DONE (docs/51, 2026-07-24).** 13 viz methods (1039 lines) extracted to `training/trainer_viz.py`
  as `TrainerVizMixin`; trainer.py 2227 → 1163. TensorBoard removed entirely (it received scalars
  only — the visual wrapper had zero callers). Byte-identical.
- ~~**DDP removal**~~ **DONE (`284992c`)** — already landed; this bullet was stale.

**This cleanup effort is now complete.** The follow-on pass (docs/51) also collapsed the SfM dataset
base class (`base_dataset.py` + `dataset_util.py` + `vggt/utils/geometry.py`, ~1200 lines) and deleted
the orphan SfM pose modules.

## Verification status

- Byte-identical harness: PASS (138/138 `torch.equal`, at every intermediate step).
- Config-diff: PASS (only dead keys removed; used values unchanged).
- `pytest tests/`: **206 passed** (the only tests removed were `test_target_t_embedder.py` and the
  two `_apply_batch_repetition` tests in `test_trainer_diagnostics.py`, which covered deleted code).

## Where to look
- Branch `cleanup/remove-sfm-stack`; base commit `4b55619`.
- Recover any deleted file: `git show 4b55619:<path>`.
- Plan of record: `~/.claude/plans/ok-so-everything-we-whimsical-dongarra.md` (session plan file).
- Rationale for the target_t retirement: docs/24, docs/25 (why reference-slice replaced target_t).
