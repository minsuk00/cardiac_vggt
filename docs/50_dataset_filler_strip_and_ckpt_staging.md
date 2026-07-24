# 50 — Dataset SfM filler-key strip + node-local checkpoint staging

> **TL;DR & takeaway** — Two cleanups (2026-07-24), both verified `prove-it`-clean.
> **(A) SfM filler-key strip:** `MRIDataset.get_data` used to fabricate 8 dummy
> batch keys (`extrinsics`/`intrinsics`/`depths`/`cam_points`/`world_points`/`point_masks`/
> `geom_masks`/`tracks`) purely to satisfy `ComposedDataset`'s collation, which was built
> for the inherited `co3d`/`vkitti` SfM datasets. Nothing on the MRI path read them (the
> live `world_points`/`cam_points` are the model *output*, not these inputs). Deleted the
> fillers, the `co3d.py`/`vkitti.py`/`track_util.py` orphans (no importers, no config refs),
> and the dead `load_track` path; `batch_size` now reads `data["images"]`. **Byte-identical**
> (removed data used no RNG; `images.shape[0] == extrinsics.shape[0]`); smoke reproduced
> first-step loss `0.0506`. This completes the "dataset filler keys" item deferred in docs/48.
> **(B) Checkpoint staging:** `torch.load` off GPFS is pathologically slow (seeky small
> reads, ~266 s vs ~5 s from /tmp). New shared helper `vggt/utils/checkpoint_stage.py`
> copies a checkpoint to node-local `/tmp` once and reuses it. **Training** stages only the
> immutable base/seed weights (`resume_checkpoint_path`), never the mutable
> `checkpoint_last.pt`. **Inference** (`inference/inference.py`, all 3 `run_*.py`) stages every
> load. Cache keyed by `sha1(abspath)`, validated by `(size, mtime)` so a re-saved source is
> re-copied, never served stale. Pure copy → byte-identical load; any failure falls back to
> the original path.

Companion to **docs/48** (SfM stack + target_t removal), which parked both the dataset
filler keys and this kind of infra cleanup as "Deferred". This doc records finishing the
filler-key item and adding checkpoint staging. Prior sibling cleanups from the same effort:
`vggt/dependency/` deletion (`d88bd03`) and single-GPU DDP removal (`284992c`).

## Part A — SfM filler-key strip + co3d/vkitti/track_util deletion

### What & why
VGGT was originally a Structure-from-Motion model; each sample carried camera geometry
(`extrinsics`/`intrinsics`/`depths`/`cam_points`/`world_points`) and validity masks
(`point_masks`/`geom_masks`). The MRI adaptation has none of that — its real geometry input
is `scanner_coords`. But `MRIDataset` feeds through the shared `ComposedDataset` collation
(`training/data/composed_dataset.py`), which was written for the original SfM datasets
(`co3d.py`, `vkitti.py`) and read those 6 keys **unconditionally**. So `MRIDataset` fabricated
dummy values (identity extrinsics, zeros depth, all-True masks, `scanner_coords.copy()` for
`world_points`/`cam_points`) just to avoid a `KeyError`.

**Nothing on the live MRI path read them as inputs.** The only consumers were: the
`ComposedDataset` collation itself; `train_utils/normalization.py` (dead — never imported,
`normalize_points: False`); and one line `batch_size = data["extrinsics"].shape[0]`. The
`predictions["world_points"]`/`preds["world_points"]` used in `loss.py`/`splat.py`/`vggt.py`
are the model **output**, a different dict — not the deleted batch input.

### Changes
- **Deleted** `training/data/datasets/co3d.py`, `.../vkitti.py`, `training/data/track_util.py`
  — orphans: `data/__init__.py` imports only `MRIDataset`, no Hydra `_target_` references them,
  and `track_util` was imported only by `composed_dataset.py`.
- **`MRIDataset.get_data`** — dropped the 8 filler keys and their list construction.
- **`ComposedDataset.__getitem__`** — removed the 6 unconditional collation lines + their
  `sample`-dict entries, the conditional `geom_masks`, the dead `if self.load_track:` block
  (config `load_track: False` everywhere; only ever functioned for co3d/vkitti), the
  `self.load_track`/`self.track_num` init, and `from .track_util import *`.
- **`trainer.py`** — `batch_size = data["images"].shape[0]` (was `data["extrinsics"]`).
- Tests updated: dropped two tests asserting removed behavior, added
  `test_sfm_filler_fields_absent`.

### Byte-identity
The removed list-building used only `np.zeros`/`np.eye`/`np.ones`/`.copy()` (zero RNG), and
all stochastic sampling happens *before* it — so the RNG stream is untouched.
`images.shape[0]` equals the old `extrinsics.shape[0]` (both stacked with `S` per-slot
entries through the same collation), so the metric weight is unchanged. **Verified:** 3
`prove-it` reviewers (0 bugs), full pytest green, GPU smoke reproduced first-step loss
`0.0506`. Commit `badb9e9`.

## Part B — Node-local checkpoint staging

### Problem
`torch.load` straight off GPFS reads the file storage-by-storage (many small, seeky reads),
which GPFS handles terribly — measured ~266 s for an ~8 GB ckpt vs ~5 s from `/tmp` (a
*sequential* copy read of the same file is fast). A fresh-from-base training smoke spent
~7 min just loading `vggt1b_base.pt`; inference eats it on every run.

### Helper — `vggt/utils/checkpoint_stage.py`
`stage_checkpoint_to_local(ckpt_path) -> str` copies the checkpoint to
`/tmp/vggt-ckpt-stage_<user>/<sha1(abspath)>.pt` and returns the path to load from.
- **Path-keyed by absolute path** → two runs that both name their file `checkpoint_last.pt`
  never collide.
- **Validated by `(size, mtime)`** — reuse the staged copy only if it exists, matches the
  source size, and is at least as new as the source; else re-copy (write `.tmp` +
  `os.replace`, atomic). So an immutable ckpt is copied once then cache-hits forever; a
  mutable `checkpoint_last.pt` (each save gets a strictly newer mtime) is re-staged, **never
  served stale**.
- **Never blocks a load** — not a local file / already under `/tmp` / any `Exception` →
  returns the original path (a pure perf optimization). Catches `Exception` (not
  `BaseException`) so a Ctrl-C mid-copy still propagates.

### Why `vggt/utils/` (not `train_utils/`)
`inference/inference.py` imports only from `vggt.*`/`inference.*`, and the three entry points
are inconsistent: `run_gated_ood`/`run_cmrxrecon` add `training/` to `sys.path`, but
**`run_rtfb` does not**. Putting the helper in `train_utils` would break `run_rtfb` at import.
`vggt/utils/` is the one package both training and inference already import (`vggt.utils.splat`)
— and `vggt` resolves as a PEP-420 namespace package, torch-free, no cycles.

### Wiring — different policy per caller
- **Training** (`trainer.py._load_resuming_checkpoint`) stages **only** the immutable
  `resume_checkpoint_path` (base/seed weights): `if resume_cfg and abspath(ckpt)==abspath(resume_cfg)`.
  A run's own `save_dir/checkpoint_last.pt` (overwritten each requeue) is loaded **directly** —
  deliberately not staged, so requeue never re-eats a copy for a marginal benefit (decision:
  we don't requeue often). The `(size, mtime)` validation is inert here (base weights are
  immutable) so training behavior is unchanged.
- **Inference** (`inference/inference.py`, both `load_rtfb_model`/`load_rtfb_model_reference`,
  which all 3 `run_*.py` funnel through) stages **every** load. Inference may point at a
  finished checkpoint (immutable → stage once, reuse) or, occasionally, a live
  `checkpoint_last.pt` (mutable → the `(size, mtime)` check re-stages safely). One `/tmp` copy
  per distinct checkpoint; `/tmp` bloat only matters if you sweep *hundreds* of models
  (accepted).

### Verification
3 `prove-it` reviewers, **clean bill** (staleness re-copy, `os.replace` atomicity + precise
`.tmp` cleanup, fallback-never-raises, import reachability incl. `run_rtfb`, training
equivalence, simplicity). Proven on the real 5 GB base weights: 59 s copy, matching sha256,
0 s cache-hit. 6 unit tests in `tests/test_checkpoint_staging.py`. Commits `7d454e0` (initial
base-weights-only version in `train_utils`) → this doc's refactor (moved to `vggt/utils/`,
added `(size, mtime)` validation, wired inference).

### Not done (deliberately, per "simple")
No cross-filesystem clock-skew handling and no sidecar mtime store — a stale serve would need
concurrent writers to the *same* checkpoint file or gross clock skew, neither of which occurs
in single-writer training/resume or immutable-seed/eval use.
