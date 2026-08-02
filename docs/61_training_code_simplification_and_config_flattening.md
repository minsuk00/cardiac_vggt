# 61 — Training-code simplification, config flattening, and the bugs it surfaced

> **TL;DR & takeaway**
>
> A deletion-and-simplification pass over the whole `training/` package, done immediately before
> the pooled-1337 multi-day run. **~310 lines of inherited VGGT scaffolding removed** (dead and
> *broken* gradient accumulation, DDP leftovers, dead batch keys, dead config keys, a
> 68-line general-purpose GPU copier used on a plain dict), **the three-layer config chain
> `default → mri_finetune → mri_volume` flattened to one runnable `default.yaml` + one
> `exp_bspline.yaml`**, and the five duplicated masked-metric blocks in `loss.py` merged into two
> helpers — split by *train-path (branchless) vs val-only (loop)*, because that distinction is
> load-bearing (4 measured graph breaks).
>
> **Four real bugs were found on the way, none of which any test caught:**
> 1. `wandb_id`/`wandb_url` in `run_meta.jsonl` had **always been `null`** — `WandbLogger` stored
>    `wandb.init()`'s return in a local, never `self.run`, while `trainer.py` read `self.run`.
> 2. `respiratory.reslice_volume_vec(spacing=SPACING_MM)` defaulted to a **pre-native-z 12 mm**
>    pitch; `baselines/export_resp_stack.py` relied on that default and silently breathed every
>    non-12 mm subject at the wrong scale. `spacing` is now **required**.
> 3. **Training was sampling subjects WITH REPLACEMENT.** `inside_random: True` discarded the
>    sampler's permutation and drew a random index per call, so ~**37% of subjects went unseen
>    every epoch** — while the config comment claimed "935 = one exact pass per epoch". Removed;
>    training now does the ML default (verified 935 draws / 935 unique, reshuffled per epoch).
> 4. `baselines/eval_all_baselines.py` was **already dead** (reads a `world_points` batch key
>    deleted with the supervised-DVF path) *and* its elastix-vs-carmen arm compared two identical
>    batches. Archived.
>
> **Two behaviour changes were made deliberately** (both start a fresh numeric series):
> `default.yaml` now uses **L2 diffusion** (`tv_weight=0, diffusion_weight=1000`) instead of L1 TV,
> because that is the arm the entire docs/46 100-epoch series actually ran; and the
> `recov_frac_heart` `[-0.5, 1.5]` clamp is **gone** (it censored 98.9% of rows to −0.5).
>
> **Everything else is provably behaviour-preserving:** the config flattening was verified by
> diffing Hydra's fully-resolved config before/after (**byte-identical**, all configs), and the
> `loss.py` refactor by comparing **75 metric values at full float64 with `==`** (**75/75
> bit-identical**). `pytest` 307 → **335**. Six end-to-end smoke runs, all clean.
>
> **Status: DONE and verified, EXCEPT** multi-epoch + SLURM requeue, which is still never tested
> end-to-end despite this pass rewriting the train step, the sampler, the collate path, the
> dataset output contract, and the config layout. **Do that before the long run.**

---

## 1. Why this pass happened, and the standard it was held to

A multi-day run on 935 subjects was about to launch. The question asked was simply whether the
training code is "optimal/minimal/simple", with the explicit standard: *simple and minimal is
best, **as long as it's correct**; do not complicate things.*

The finding that shaped everything: **the code written for this project is in good shape.**
`loss.py`, `mri_dataset.py`, `preprocess.py`, `respiratory.py`, `run_log.py`, `val_logging.py`
are dense but load-bearing, and their heavy comments are earned — nearly every one cites a
specific incident (docs/56–60). `val_logging.py` and `run_log.py` came back from review with
**zero findings**.

The cruft is almost entirely **inherited upstream VGGT scaffolding**: generic multi-dataset,
multi-GPU, variable-aspect-ratio, natural-image machinery driving a pipeline that is pinned to
**one dataset, one GPU, square 256×256 slices, and B=1**. It is not bad code. It is *correct
general-purpose code solving a problem this project does not have.*

**Method.** Two independent read-across agents (data pipeline; `train_utils`/`loss`/configs) plus
a manual pass over `trainer.py`/`trainer_viz.py`. **Every claim of "dead" was re-verified by an
exhaustive repo-wide grep before acting** — this mattered: several agent claims were wrong (see
§8). Nothing was deleted on an agent's say-so alone.

---

## 2. What was deleted (dead code)

All verified by exhaustive grep across `training/ tools/ tests/ evaluation/ inference/ baselines/
sbatch/ docs/` — after deletion, the only remaining textual hits are historical prose in `docs/`.

| Removed | Where | Evidence it was dead |
|---|---|---|
| **Gradient accumulation** — `_run_steps_on_batch_chunks`, `chunk_batch_for_accum_steps`, `get_chunk_from_data`, `is_sequence_of_primitives`, `accum_steps` config+attr | `trainer.py` | `accum_steps: 1` in every config **and broken at B=1** — see §2.1 |
| `check_and_fix_inf_nan` (29 lines) | `general.py` | grep: definition only, zero callers |
| DDP helpers `get_rank` / `get_world_size` / `is_dist_avail_and_initialized` (two separate copies) | `general.py`, `worker_fn.py` | DDP removed in `284992c`; see §2.2 for the trap |
| `distributed` kwarg + `_rank` guards | `trainer.py`, `wandb_writer.py` | no config sets `distributed`; `_rank` was a constant `0` |
| `log_dict`, `log_3d_point_cloud` + `numpy`/`torch` imports | `wandb_writer.py` | zero callers |
| `AverageMeter.value/.average`, `_allow_updates`, `DurationMeter.reset/.add`, `ProgressMeter.real_meters` | `general.py` | both `ProgressMeter` call sites passed `real_meters={}` |
| `GLOB_FLAGS` + `wcmatch` import (the **dead third copy**) | `checkpoint.py` | `optimizer.py`/`freeze.py` keep their own live copies |
| `param_group_modifiers` param + body + config key | `optimizer.py`, `default.yaml` | `construct_optimizers` never passes it |
| Dead batch keys `rotations` (literally `np.zeros(3)`/slot), `ids`, `frame_ids`, `frame_num`, `original_sizes` | `mri_dataset.py`, `composed_dataset.py` | grep: producer lines only |
| `_process_batch` (documented no-op passthrough), `_point_cloud_logged_epoch` | `trainer.py` | set-never-read / identity function |
| `visuals_keys_to_log` payload (13 lines) | configs | only ever tested `is not None` and `phase in …`; contents never read |
| `loss_conf_point`, `loss_reg_point`, `loss_grad_point` | `mri_finetune.yaml` | yaml-only; supervised point loss deleted long ago |
| `visuals_per_batch_to_log`, `video_logging_fps` | `mri_finetune.yaml` | yaml-only |
| `dvf_dirname` param + its accepted-and-ignored warning | `mri_dataset.py` + 2 `baselines/` call sites | supervised-DVF path removed |
| `aspect_ratio` plumbing (sampler → tuple → `__getitem__` → `**kwargs`) | `dynamic_dataloader.py`, `composed_dataset.py`, `mri_dataset.py` | landed unread in `get_data(**kwargs)`; always 1.0 |
| `TupleConcatDataset`'s `ConcatDataset`/`bisect`/`cumulative_sizes` | `composed_dataset.py` | exactly one dataset configured ⇒ always `datasets[0]` |
| `copy_data_to_device`'s namedtuple/defaultdict/dataclass/Protocol branches (68 → 16 lines) | `general.py` | batch is a plain dict of tensors |
| `training/scratch/` (984 KB) | repo | stray relative-path artifact of a July DDP smoke run |

### 2.1 Gradient accumulation was dead **and broken**

Not merely unused. `batch_size` is hardcoded to **1** in `dynamic_dataloader.py` (documented as
the only configuration that is safe by construction under native-z, docs/59 F7/F9/F19). Chunking a
1-sample batch into N>1 pieces computes `start = (1 // N) * i = 0`, `end = 0` — **empty tensors**.
So `accum_steps > 1` could never have worked. Removing it also removed a source of confusion
(`default.yaml` shipped `accum_steps: 2` with a comment inviting you to raise it on OOM).

### 2.2 The `get_world_size` trap — why naive deletion would have changed training

`worker_fn.default_worker_init_fn` computed:

```python
worker_seed = (rank*num_workers*1 + worker_id*1 + seed
               + world_size*1 + epoch*12345 + RANK*1042)
```

`rank` → 0 and `RANK` → 0 drop out — but **`world_size` is 1, so it contributes `+1`**. Deleting
the DDP calls naively yields `worker_id + seed + epoch*12345`, off by one, which **reseeds every
dataloader worker and changes the training data stream**. The `+1` was kept, and equivalence
proven over 8 workers × 3 worker-counts × 200 epochs × 3 seeds → **0 mismatches**.

*Generalizable lesson: when deleting "always-constant" code, check whether the constant is 0
(vanishes) or 1 (does not).*

---

## 3. Bugs found and fixed

### 3.1 `wandb_id` had always been null in `run_meta.jsonl`

`WandbLogger.__init__` did `run = wandb.init(...)` — a **local**. `trainer.py:299` read
`getattr(self.wandb_writer, "run", None)`, which was therefore always `None`. Every
`run_meta.jsonl` ever written (docs/60's entire point is disk-analysable runs) had
`"wandb_id": null, "wandb_url": null` — the link back to the dashboard was silently missing.
Confirmed on the real `_docs60_v2` run before fixing.

**Fix:** `self.run = None` up front, `run = self.run = wandb.init(...)`. Verified: new runs record
`"wandb_id": "kbphwgbs"`.

### 3.2 `respiratory.py`'s stale 12 mm spacing default

`reslice_volume_vec(V, disp, spacing=SPACING_MM)` where `SPACING_MM = (12.0, 1.4, 1.4)` — correct
only for the pre-native-z shared cube. Under native-z each subject keeps its own 5–12 mm pitch, so
a missed `spacing=` **silently breathes that subject at 12 mm**: plausible output, wrong physics,
no error. All `evaluation/` adapters passed it explicitly; `baselines/export_resp_stack.py:101`
did not, and had been relying on the wrong default.

**Fix:** `spacing` is now **required** on `reslice_volume_vec` and keyword-only-required on
`reslice_volume`. `export_resp_stack.py` now passes the subject's own `dz_mm`. The two `tools/`
callers and 5 test call sites pass `SPACING_MM` explicitly (their data is uniformly-12 mm
CMRxRecon2024, so values are unchanged).

### 3.3 Training sampled subjects **with replacement**

`TupleConcatDataset.__getitem__` contained:

```python
if self.inside_random:                      # True for train, False for val
    idx = random.randint(0, len(self) - 1)  # discard the sampler's index
```

`DynamicDistributedSampler` produced a proper seeded permutation, which was then **thrown away**.
Consequences per 935-step epoch: expected **~344 subjects (37%) never drawn** (coupon-collector,
`935/e`), ~344 once, ~172 twice, ~57 three times — a *different* 37% each epoch. Meanwhile the
config asserted *"935 pooled train subjects = one exact pass per epoch"* and warned about
"re-drawing a fixed, seed-invariant prefix" — reasoning that only makes sense for a permutation.

Over 200 epochs every subject is still seen ~200 times, so this did **not** bias the model or
exclude anyone. What it cost: **noisier epoch-to-epoch curves** (each epoch trains on a different
random subset) and a meaningless notion of "epoch". It exists upstream because materialising a
shuffled index list for very large multi-dataset training was a memory concern — irrelevant at 935
paths.

**Fix:** `inside_random` removed entirely (config keys, code branch, `random` import). Training now
does the ML default: `DistributedSampler`'s seeded permutation, reshuffled each epoch via
`set_epoch`. **Verified** by iterating the real sampler: `epoch 0/1/2 → n=935, unique=935`, with
different orders. The `limit_train_batches` comment was corrected to describe reality.

### 3.4 `baselines/eval_all_baselines.py` was already broken

Two independent failures, both predating this pass: it calls `stack("world_points")`, a batch key
`MRIDataset` stopped producing when supervised-DVF was removed (⇒ `KeyError` on subject 0); and it
built two batches with `dvf_dirname="dvf_elastix"` vs `"dvf_carmen"` expecting them to differ, when
the dataset had been **ignoring that argument** — so the headline elastix-vs-carmen comparison was
comparing a batch against itself. **Archived** to `_archive/` rather than silently "fixed", since
its premise (GT DVFs in the batch) no longer exists.

---

## 4. The 64 MB/step that was built and thrown away

`MRIDataset.get_data` ends by building the model input: pick S slices, upsample 256→518, replicate
to RGB → `(20, 518, 518, 3)` float32 = **64.4 MB per sample**, in a dataloader worker. That tensor
is collated, copied to the GPU — and then `gpu_augment_batch` **overwrites `batch["images"]`
wholesale**, because respiratory has to re-extract the slices at their breathing-displaced
positions. `do_resp` is just `respiratory_cfg.enable`, `true` in both train and val, so on the
shipped config the CPU copy was discarded **100% of the time**.

It was never a wall-clock cost (4 prefetch workers keep ahead: Data Time 0.017 s vs Batch Time
0.99 s) — it is wasted worker CPU, worker RSS, and host→device traffic.

**Fix — and the contract that makes it safe.** `MRIDataset(defer_input_images=True)` skips the
upsample/RGB step and **omits the `images` key entirely**. `gpu_augment_batch` treats a *missing*
key as "extract unconditionally", on **every** path including the no-augmentation early return and
the path where the affine build fails. An absent key is a signal that cannot be mistaken for stale
data; a placeholder tensor could.

**This is where the pass caused its own regression, and it is the most instructive part of this
doc.** `_compute_identity_baseline` and `_log_cardiac_cycle_filmstrip` call `mri_ds.get_data()`
**directly**, bypassing `gpu_augment_batch` entirely. With deferral on the val dataset they raised
`KeyError: 'images'` for every subject — swallowed by the per-subject `try/except` guard — and the
run produced **no `baseline_identity.json` at all**, i.e. silently lost the reference every val
metric name is built from. `pytest` was green throughout; only the end-to-end smoke run exposed it.

Two fixes: (a) the shared `_subject_device_batch` helper now **builds `images` itself** from
`phases` when absent (float32, matching the dataset exactly); (b) **deferral is train-only**.
Investigating the blast radius found **12 scripts** that instantiate the `data.val` config node and
then read `get_data()["images"]` directly — including `inference/run_cmrxrecon.py`, a standing eval
entry point. Deferring on val would have broken all of them, discovered days later. Train is 935 of
1201 steps/epoch, so train-only keeps the bulk of the saving at zero blast radius.

*Generalizable lesson: a "safe because the caller always does X" contract is only as good as your
enumeration of callers. Grep for the callers before trusting the invariant.*

---

## 5. Config flattening

**Before:** `default.yaml` → `mri_finetune.yaml` → `mri_volume.yaml` → `{bspline, diffusion}`.

Two problems:
- **`mri_finetune.yaml` was a half-config.** `loss.volume.*` does not exist until `mri_volume`, so
  composing it alone gives `MultitaskLoss(volume=None)` → `objective = 0` → a **silent zero-loss
  run**. Its own header said so.
- **The base layer lied.** `mri_volume` overrode **38 resolved keys**, including the ones you would
  actually look up: `frozen_module_names` (`["*aggregator*"]  # example` → `["*patch_embed*"]`),
  `use_reference_token` (False → True), `enable_point` (False → True), `compile_attention_blocks`
  (False → True), plus a placeholder `resume_checkpoint_path: /YOUR/PATH/TO/CKPT`. Reading
  `default.yaml` to learn what training does gave the **wrong answer on the central architectural
  choices**.

**After:** `default.yaml` (complete, runnable, `--config default`) + `exp_bspline.yaml` (~25 lines).

**How it was made safe.** Hydra can print the fully-resolved config, so the merge is *checkable*,
not a matter of care:

```bash
python training/launch.py --config <name> --cfg job --resolve   # before vs after
```

Resolved configs for `mri_volume`, `mri_volume_bspline`, `mri_volume_diffusion` were captured
before, and the new `default` / `exp_bspline` / `exp_diffusion` after: **byte-identical, all
three.** 36 references across 29 files were re-pointed (`compose(config_name=…)`, `--config`,
`CONFIG=`), **including 4 files in `evaluation/`** — normally off-limits, unavoidable for a rename,
and one string per file.

`exp_name` and `config_name` **deliberately keep the `mri_volume` family name**. They name log dirs
and wandb runs; changing them would fork the run history and break `RESUME_FROM` paths for no
benefit. The *file* is `default.yaml`; the *experiment family* is still `mri_volume`.

### 5.1 L2 diffusion is now the default (deliberate behaviour change)

Verified from the runs' own wandb configs — not inferred from directory names — that **every recent
100-epoch experiment** (the docs/46 series: control0/nogather, gather05, contz, s20, s20contz,
dino_ft, aug_moderate, lowdiff100, ftgather/ftctrl) ran under the old `mri_volume_diffusion.yaml`
with `tv_weight=0, diffusion_weight=1000, gather_weight=0.5, one_frame_per_slice=true`.

The shipped default was L1 TV (`tv_weight=0.1`) — an arm **nothing was ever measured under**.
`default.yaml` now carries L2 diffusion, so the config you launch *is* the arm the evidence was
collected on. That made `exp_diffusion.yaml` a no-op ⇒ deleted, its 12 sbatch scripts point at
`default`.

**Side effect caught:** `exp_bspline` previously inherited `tv_weight=0.1` and is documented as
"keeps the L1 TV reg" — it would have silently changed **both** the head and the regularizer. It
now pins `tv_weight: 0.1 / diffusion_weight: 0.0` explicitly; its resolved config is byte-identical
to pre-refactor.

---

## 6. `loss.py` masked-metric helpers

Five blocks compute "square the error, average over a mask, convert to dB". They *look*
copy-pasted. They are not freely interchangeable, and the difference tracks **where the code runs**:

- **Train-path** (bbox, motion): vectorized `torch.where` + `sum/count`, deliberately
  **branchless** — the existing comment records that a Python-level decision here costs **4 graph
  breaks, measured**. Runs every training step.
- **Val-only** (heartseg, docs/38 heart, docs/38 seg): a `for b in range(B)` loop with boolean
  indexing. Gated on `not pos_pred.requires_grad`, so host syncs are free and the loop is clearer.

Both are NaN-safe, and that is *why* it is `torch.where` and not `err * mask.float()`:
**`NaN * 0.0 == NaN`**, so a multiply lets one bad voxel anywhere in the cube poison the metric.

**Merged accordingly — one helper per class, not one for all five:** `_masked_stats_vec(err, mask)`
(branchless) and `_masked_mse(a, b, mask)` (boolean indexing), plus `_psnr_from_mse`. `loss.py` went
502 → 517 lines: *longer*, because the helpers carry the documentation of why two styles exist,
while the five duplicated bodies are gone.

### 6.1 The golden harness, and its fault injection

`tests/test_loss_masked_metrics_golden.py` (10 tests) pins all five blocks on a fixed synthetic
batch plus the degenerate cases that distinguish the two styles. **A green test proves nothing
until it is shown to fail**, so three bugs were injected into `loss.py` and the harness re-run:

| Injected fault | Caught by | Result |
|---|---|---|
| `torch.where(mask, err**2, 0)` → `err**2 * mask.float()` | `test_non_finite_voxel_outside_mask_does_not_leak` | ✅ FAILS |
| PSNR clamp `1e-10` → `1e-8` | `test_psnr_clamp_floor_is_pinned` | ✅ FAILS *(see below)* |
| empty-ROI `if not m.any(): continue` removed | `test_empty_mask_does_not_poison_the_batch` | ✅ FAILS |

**The clamp fault initially slipped through** — the synthetic batch's MSE is ~0.32, far above any
plausible clamp floor, so the clamp never engaged. Fixed by adding a case that feeds `V_canon`
back as `V_gt`, making the residual exactly zero, where the clamp is the *only* thing determining
the answer (100 dB). That is a real harness gap that fault injection found and assertion-writing
would not have.

### 6.2 Bit-identity: measured, with one stated exception

The golden test uses `abs=1e-4`, which is **not** proof of bit-identity. So the pre- and
post-refactor `loss.py` were run on identical inputs and compared **at full float64 with `==`**:

```
metrics compared : 75   (25 distinct metrics × 3 seeds)
BIT-IDENTICAL    : 75
CHANGED          : 0
```

**Exception — `recov_frac_heart` / `recov_frac_seg` are not among those 25.** They never fire on
synthetic data: the `span = mse_id − mse_or > 1e-6` guard rejects every sample (the oracle splat is
not better than identity on random volumes), and perturbing the predictions did not change that.
**The synthetic harness structurally cannot reach the recov family** — recorded here as a known
limitation, not glossed over.

> ⚠️ **CORRECTED by docs/62 §2.2 (2026-08-01): "structurally cannot" is WRONG.** The recov family
> IS reachable on synthetic data — at `D=12` with perturbed predictions the `span > 1e-6` guard
> passes. The limitation was in *this harness's* choice of inputs, not in the code, so the gap is
> closable in-repo. (docs/62 independently re-verified the refactor anyway, by running the
> pre-refactor tree: `recov_frac_heart` is the single deliberate old-vs-new delta, old `1.5` = the
> clamp ceiling vs new `1.975`.)

That gap was closed *empirically* by the real smoke run, where the guard does pass:
`recov_frac_heart = −4.83` (per-subject −3.12 … −7.83), `recov_frac_seg = −4.58` (−1.43 … −7.71).
**All 8 subjects sit far below the old −0.5 floor**, i.e. every one would previously have been
censored — direct confirmation of the 98.9%-censoring pathology the clamp removal targets.

### 6.3 `recov_frac_heart` clamp removed (deliberate behaviour change)

`.clamp(-0.5, 1.5)` censored **98.9% of rows to exactly −0.5** early in training. `recov_frac_seg`
was already unclamped for this reason. The only argument for keeping it was continuity with wandb
curves that native-z had already invalidated — a dead constraint, since this whole cohort/pipeline
is a fresh series. The `span > 1e-6` guard (which keeps the denominator positive) is the part that
actually matters and is retained.

---

## 7. Other changes made in this pass

- **Augmentation flip is now aggressive-tier-only.** It was briefly enabled in all tiers
  (2026-07-31, docs/58 §10c), but `moderate` is the arm docs/46 §3 C2 measured and shipped, and
  that arm had **no flip**. Kept as a *commented-out* line in conservative/moderate (user request)
  so the option stays visible. `tests/test_gpu_aug.py::test_flip_is_aggressive_only` guards it.
- **EF/CorSeg now runs every val epoch** (`ef_eval_every_n_val_epochs: 5 → 1`). Measured cost:
  **~1.75 min** for the whole EF block (write 266 pred NIfTIs + segment + metrics) on top of a
  **~4.2 min** val loop (L40S, 133 subjects × {ED,ES}). The `5` was sized for the slower nnU-Net
  subprocess hop, which CorSeg replaced.
- **`self.target_size` is now honoured.** It was stored and ignored — every site hardcoded
  `INPUT_IMG_SIZE = 518`. `get_data` now reads it for both the slice upsample and the
  `scanner_coords` normalization, so it is a real experiment knob. ⚠️ Must remain a multiple of
  **14** (DINOv2 patch size; 518 = 37×14), and any R ≠ 518 interpolates the pretrained position
  embeddings ⇒ fresh numeric series.
- **Duplicated batch-building in `trainer_viz.py` merged** into `_subject_device_batch` (`:89` and
  `:320` each hand-rolled the same `get_data → GPU batch` conversion, including two verbatim
  copies of a nested `st()` helper). This is the shape of the docs/59 F14 bug, where a fix landed
  in one copy and not its twin.
- **`compute_geometric_bbox` / `recompute_bbox_gpu` deduped.** Character-identical bodies apart
  from a `padding` arg no production caller sets. Both call sites are live and at different
  pipeline stages: pre-aug (dataset, `mri_dataset.py:341`) and post-aug (`gpu_aug.py:339`, after an
  affine moves the content mask). `gpu_aug` now imports the `preprocess` one.
- **`MultitaskLoss` docstring corrected.** It advertised "Camera loss / Depth loss / Point loss /
  Tracking loss (dirty code is at the bottom of this file)" — none of which exist, and there is no
  code at the bottom of the file. The `@dataclass(eq=False)` on it was a no-op (custom `__init__`,
  zero fields).
- **`CorSeg/` gitignored** (pristine upstream clone at `b006956`, nested `.git`, we edit nothing).
- **All `sbatch/*.sh` accounts → `jjparkcv0`** (`jjparkcv98` frequently hits
  `AssocGrpSubmitJobsLimit`).
- **CLAUDE.md corrected** where it had gone stale: it claimed `max_img_per_gpu: 20 → fixed S=20
  slot budget (multi-frame)`, but `max_img_per_gpu` was deleted (docs/59 F9) and under
  `one_frame_per_slice: true` (the default) **S == D**, this subject's own plane count (5–21).

---

## 8. Deliberately KEPT — and why

This section exists so a future agent does not "clean up" something load-bearing.

| Kept | Why |
|---|---|
| **Two masking styles in `loss.py`** | Not redundancy. Branchless on the train path is worth **4 measured graph breaks**; loops in val-only code are clearer and syncs are free there. Merging *across* the classes would either slow the train step or obfuscate val. |
| **`self.mri_mode`** | Genuinely used (`seq_name = f"mri_{mri_mode}_{rel}"`, parsed in 3 places). Removing the param means **58 call-site edits** (33 tests, 18 tools, 6 baselines) and the top-level config key survives anyway (it is interpolated into `exp_name`). Cost ≫ benefit. |
| **`per_slot` as a `RespiratoryConfig` field** | Inert *in our config* (`group_by_burst: true` takes the other branch) but real API — 4 `evaluation/` adapters set `rcfg.per_slot = False`, ~15 `tools/` pass it, and a test covers it. Only the redundant **config line** was removed. |
| **`ef_seg_backend: "nnunet"` fallback** | Deliberate escape hatch, and nnU-Net Task114 is still the method-matched operator that produced the GT labels. |
| **`freeze.py`'s `recursive` param; `ef_eval._spearman`'s scipy fallback** | Single-valued / unreachable-in-practice, but small and legitimate API surface. Not worth the churn. |
| **Val-side `defer_input_images: false`** | 12 scripts (incl. `inference/run_cmrxrecon.py`) read `data.val`'s `images` directly. See §4. |
| **`exp_name`/`config_name` = `mri_volume`** | Log-dir and wandb continuity; also what makes the resolved-config diff clean. |
| **All the native-z guard clauses** | `one_frame_per_slice` budget assert, `z_norm` range raise, dz/z_scale uniformity checks, `cache_signature`, `StripMetaD`. Each traces to a docs/59 finding; they are the cheap insurance that makes B=1 + native-z safe. |
| **`img_nums: [20, 20]`** | **Not dead.** Under `one_frame_per_slice` it is the **cap** enforced by the docs/59 F19 guard (raises if a subject needs more slots than the memory budget was sized for), not a slot count. |

### Rejected after analysis

- **Merging `loss.py`'s five blocks into ONE helper** — see above; rejected on the graph-break
  measurement, and instead split into two.
- **Removing `self.mri_mode`** — proposed by review, rejected on the 58-call-site count.
- Two review claims were checked and found **wrong**: that `mri_mode` is "stored and never used"
  (it is read at `mri_dataset.py:592`), and the initial framing of `img_nums` as dead (it is the
  live F19 cap).

---

## 9. Verification summary

| Check | Result |
|---|---|
| `pytest tests/` | **335 passed** (307 at session start; +10 golden, +5 deferral, others) |
| Config flattening | Hydra resolved config **byte-identical**, all 3 configs |
| `loss.py` refactor | **75/75 metrics bit-identical** at float64 `==`, 3 seeds |
| Golden harness | **3/3 injected faults caught** (after closing the clamp gap) |
| Worker-seed preservation | **0 mismatches** over 8 workers × 3 counts × 200 epochs × 3 seeds |
| Train sampling is a permutation | epochs 0/1/2 → **935 draws, 935 unique**, different orders |
| End-to-end smoke runs | **6 runs, 0 tracebacks**, clean exit records, all artifacts present |
| `wandb_id` fix | populated (`kbphwgbs`) where it was previously always `null` |
| `recov_frac` clamp removal | confirmed on real data (all 8 subjects below the old floor) |

Repro:

```bash
# tests
micromamba run -n svr python -m pytest tests/ -q

# a full smoke on the flattened config
WANDB_MODE=offline PYTHONPATH=training:. python training/launch.py --config default \
    max_epochs=1 limit_train_batches=3 limit_val_batches=8 ef_val_sweep=false \
    logging.log_dir=scratch/logs/_smoke

# resolved-config diff (the gate used for the flattening)
python training/launch.py --config default --cfg job --resolve
```

---

## 10. Known gaps and what is NOT verified

1. 🔴 **Multi-epoch + SLURM requeue has never been exercised end-to-end.** Every verification run
   here was `max_epochs=1`. This pass rewrote the train step, the sampler, the collate path, the
   dataset output contract, `loss.py` internals and the config layout — so this is now the largest
   untested surface in the repo. Suggested: `max_epochs=3` with a mid-run `kill -USR1`, then check
   that `metrics.jsonl` has duplicate `(name, step)` rows (`load_run` dedupes keep-last),
   `val_per_subject.csv` has 3 epochs of rows, and `run_meta.jsonl` has 2 launch lines with the
   second carrying `resumed_from_epoch`.
2. 🟠 **The golden harness cannot reach `recov_frac_*`** (§6.2). Its behaviour is confirmed only by
   the real smoke run and by diff inspection.
3. 🟠 **`baselines/eval_within_body_mask.py` was not audited** — it is the sibling of the script
   archived in §3.4 and may share its rot.
4. 🟠 The **64 MB deferral is train-only**, so val still pays it (266 of 1201 steps/epoch). Closing
   that requires migrating the 12 direct `data.val` readers.
5. ⚪ `_archive/` was excluded from all greps by design; a resurrected archived script may reference
   removed symbols (`batch["ids"]`, `check_and_fix_inf_nan`, …).
6. ⚪ **Nothing here changes any `state_dict`.** Old checkpoints load exactly as before.

---

## 11. What a future agent should read first

- **§3.3** — training was sampling with replacement. If you see old runs whose per-epoch curves
  look noisier than they should, that is why.
- **§4** — the deferral contract, and the identity-baseline regression it caused. The lesson
  (enumerate your callers before trusting an invariant) generalizes.
- **§8** — before deleting anything in `loss.py`, `respiratory.py`, or the native-z guards.
- **§10.1** — do the requeue test before the next long run.
