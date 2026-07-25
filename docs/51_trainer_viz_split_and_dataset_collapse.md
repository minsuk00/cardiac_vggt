# 51 — Trainer viz split, TensorBoard removal, dataset base-class collapse

> **TL;DR & takeaway**
> Finished the cleanup effort tracked in docs/48. Removed TensorBoard entirely (it only ever
> received scalars — the visual wrapper had zero callers), split 1039 lines of wandb/matplotlib
> code out of `trainer.py` into a `TrainerVizMixin` (2227 → 1163 lines), deleted three orphan SfM
> pose modules, and collapsed the inherited SfM dataset base class — `base_dataset.py` +
> `dataset_util.py` + `vggt/utils/geometry.py`, ~1200 lines that **could not execute**.
> **Net: −2901 lines, +27.** Verified byte-identical by control-vs-treatment (204/204 tensor
> digests and every RNG fingerprint identical), 211 tests, and GPU smokes reproducing first-step
> loss `0.0506` and identity baseline `25.45/23.97/17.27 dB`. **No experiment needs re-running.**
> Also fixed pre-existing defects found during review (6 matplotlib figure leaks, an ED/ES panel
> indexing hazard, stale DDP docs) — and **reverted one attempted fix** (`run_train`'s
> `self.epoch -= 1`, see §4 below: guarding it is worse than leaving it). Status: complete;
> docs/48's deferred list is now empty.

## Why

`trainer.py` was 2227 lines, 47% of it visualisation. The dataset layer still inherited original
VGGT's multi-view-SfM machinery (photos + camera intrinsics/extrinsics + depth maps) that the MRI
pipeline never touches. Both were the last items on docs/48's deferred list.

## What changed

### 1. TensorBoard removed
Only wandb is used. Investigation showed TB was **narrower than "unused logging"**: it received
**scalars only**. The dual-log visual wrapper `_log_visuals` had **zero callers repo-wide** — all
12 visual logs call wandb directly with `wandb.Image` objects TB cannot consume. So TB never
received a single image in this pipeline.

Deleted `train_utils/tb_writer.py` (147 lines), the `self.tb_writer` instantiation, the dead
`_log_visuals` wrapper, and the `tensorboard_writer` config blocks. Stops ~2.8 GB of event files
under `scratch/logs/*/tensorboard/` from growing (existing ones left alone).

### 2. Viz extracted to a mixin
13 methods + the `_ED_ES_SUBJECTS` constant → `training/trainer_viz.py` as `TrainerVizMixin`.
Extraction was mechanical (exact line ranges, not retyped): diffing the moved lines against the
original yields **zero non-blank differences**.

Also renamed `_log_tb_visuals` → `_log_visuals_to_wandb`. Despite the name it is the **wandb**
visual dispatcher, not TensorBoard — a genuine trap for the next reader.

**The risk that mattered:** `trainer.py` has `from train_utils.general import *`. A moved method
referencing a name that resolved only through that wildcard would raise `NameError` — and since
every viz method is `try/except`-wrapped, it would be **silently swallowed**, surfacing only on
cadences (every 5 val epochs) a short smoke never reaches. Settled by `symtable`/`ast`/`dis`
analysis over every method and nested code object: **zero unresolved globals**. All 30
heavy-dependency usages (`np`/`wandb`/`plt`/`nib`/`ef_eval`) already carried method-local imports;
`safe_makedirs` is the only ex-wildcard name needed and is imported explicitly.

### 3. Orphan SfM modules deleted
`train_utils/normalization.py`, `vggt/utils/pose_enc.py`, `vggt/utils/rotation.py`, plus
`activate_pose`/`base_pose_act` from `head_act.py` and `log_env_variables` from `general.py`.
**`inverse_log_transform` was KEPT** — `activate_head` calls it, and `activate_head` is live via
`dpt_head.py`/`bspline_head.py`. (An earlier draft listed it as dead; that was wrong, caught
before any deletion. The "unused" test used was *"referenced outside its own file"*, which
structurally cannot see same-file callers — it produced 4 false positives out of 5.)

### 4. Dataset base class collapsed
`MRIDataset` now inherits `torch.utils.data.Dataset` directly, with `BaseDataset.__getitem__`
copied in verbatim (AST-identical). Deleted `base_dataset.py`, `dataset_util.py`,
`vggt/utils/geometry.py`.

The key finding: `BaseDataset.process_one_image` is not merely unused but **unreachable** — it
reads `self.training`, an attribute neither `BaseDataset` nor `MRIDataset` ever sets (only
`ComposedDataset` does), so any call would raise `AttributeError`. It was the sole consumer of
`dataset_util.py`, which was the sole consumer of `geometry.py`.

Removed 14 now-unconsumed `common_config` keys from both train and val blocks + top-level
`patch_size`. Six were read only by the deleted `BaseDataset.__init__`, so config and code had to
move together. Top-level `img_size` stays — it still feeds `target_size`.

### 5. What was deliberately NOT done
**`ComposedDataset` was left almost intact.** An early estimate called it "~200 removable lines";
that was wrong. It does three load-bearing jobs: the numpy→tensor conversion of 17 batch keys;
`TupleConcatDataset`, whose `inside_random: True` **is the actual train subject sampler**
(`random.randint`), not scaffolding; and the `p=0.0` colour jitter, which is a visual no-op but
**empirically consumes one `torch.rand(1)` per train sample** (measured: next-rand `0.768222` vs
`0.496257`). Removing it would shift the worker RNG stream — a fresh training series, not a
cleanup. Only inert parts were removed (a vestigial `from .dataset_util import *`, an
always-failing `_data_to_batch_tensors` import, 8 dead config keys).

## Verification

Byte-identity was established by **control vs treatment**, not by headline numbers: a clean
worktree at HEAD with the other agent's in-flight files copied in so they cancel, vs this tree,
both driven through the real `DynamicTorchDataset → ComposedDataset → MRIDataset` path.

- **204/204 tensor digests identical** across train and val.
- **Every RNG fingerprint identical** — python, torch, numpy, *and the next draw from each* — so
  the colour-jitter draw and `inside_random` sampling are unperturbed.
- `activate_head` / `inverse_log_transform` AST-compared **byte-identical** to HEAD.
- **211 tests pass**; all 5 configs compose and resolve with `target_size=518`.
- GPU smoke reproduces first-step loss **0.0506**, psnr_full **18.0616**, ssim **0.5594**,
  `Grad/point` **0.7356**, identity baseline **25.45 / 23.97 / 17.27 dB**.
- A smoke with cadences forced on (`ef_eval_every_n_val_epochs=1`, `augmentation.enable=true`)
  exercised **11/13 moved methods** — all produced their wandb artifacts with **zero** swallowed
  warnings, covering exactly the every-5-epoch paths a normal smoke misses.

**Gotcha for future runs:** `Grad/aggregator` wobbles in the 4th decimal (0.1055 vs 0.1056) across
*identical* code. Confirmed pre-existing by running the same commit twice — it is the splat's
atomic scatter-add, not a regression. Don't chase it.

**Scope caveat:** the working tree concurrently carried a numpy 1.26→2.2.6 migration (docs/49,
a different effort). Training is byte-identical under it, but the eval/OOD path is not
(~1.9e-4 drift, below the 0.252 dB same-command noise floor). That is orthogonal to this cleanup.

## Pre-existing defects fixed (found by review, not introduced here)

1. **matplotlib figure leaks** at 5 `plt.figure` sites whose callers swallow exceptions — a raise
   between creation and `plt.close(fig)` leaked the figure forever. Wrapped in `try/finally`,
   matching the pattern `_log_ed_es_panels`/`_log_lookup_to_wandb` already used. Verified the
   re-indent changed only nesting: unwrapping the `try/finally` from both versions yields
   identical ASTs.
2. **`_stash_ed_es` indexed `val_targets` with `_val_iter`**, the per-*batch* counter, not the
   per-*sample* `seq_index`. Correct only while the val batch size is 1; raising
   `max_img_per_gpu` would have **silently mislabelled** which subject/phase each ED/ES panel
   belongs to. Now reads `batch["seq_index"]` as `_save_ef_volume` does. The same hardening was
   applied to `_log_visuals_to_wandb` (val-subject gate + wandb section name), defensively: that
   method's call site is **not** try/except-wrapped, so the lookup is guarded and falls back to
   `_val_iter`. Both are no-ops today — val batch size is provably 1
   (`floor(max_img_per_gpu / img_nums) = floor(20/20)`), and val is unshuffled so
   `seq_index == _val_iter` for every iteration.
3. **`WandbLogger.log_visuals`** — a 6th figure-leak site, and dead: its only caller was the
   deleted `_log_visuals`. Removed rather than patched.
4. **`run_train`'s `self.epoch -= 1`** — **TRIED AND REVERTED. Do not "fix" this again.** The
   unguarded decrement looks wrong (when the loop never runs it walks the counter backwards past
   the resumed value — the docs/37 `CKPT_ONLY` trap). Guarding it is worse. The loop is also
   skipped when a **completed** run is restarted (requeue during the final `run_val`, or
   `RESUME_FROM` a finished dir), and there the guard leaves `self.epoch == max_epochs` — an epoch
   that never ran — which flips the trailing `run_val`'s cadence gates: at `max_epochs=200`,
   `200 % 5 == 0` fires the heavy nnU-Net EF eval that previously did not run; at `max_epochs=100`,
   `100 % 3 != 0` stops the filmstrip that previously did. The decremented value is also the more
   accurate label (the last epoch actually completed). `run_train` is therefore AST-identical to
   its pre-cleanup form; only an explanatory comment was added.
5. **Stale docs**: `trainer.py`'s class docstring claimed DDP/multi-node, and CLAUDE.md +
   `sbatch/train_mri_volume_reference.sh` claimed aggft requires
   `distributed.find_unused_parameters=true`. DDP was removed in `284992c`; the config key no
   longer exists and nothing reads it. (`docs/26`/`docs/47` also mention it but were left alone —
   they are historical records of experiments run when it was true.)

Still open (pre-existing, not fixed): `vggt/models/vggt.py`'s `forward` docstring documents
`pose_enc`/`depth`/`track` returns that no longer exist — left alone because that file had
another agent's uncommitted changes.

## Review

Six independent reviewers (`/prove-it`) across name resolution, extraction fidelity, deletion
reachability, the dataset fold, config removal, and numerical/RNG equivalence. Every finding was
adversarially verified before being accepted; the headline NameError hypothesis was refuted three
independent ways. Notable technique: a **pickle-index scan of all 124 checkpoints** confirmed no
first-party class is pickled anywhere, so no `torch.load` of an existing checkpoint can require a
deleted symbol.
