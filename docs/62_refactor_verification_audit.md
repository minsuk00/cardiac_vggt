# 62 — Independent verification audit of the docs/60+61 refactor

> **TL;DR & takeaway**
>
> An adversarial re-audit of the whole `e1c38ac..527093c` range (docs/60 on-disk run logs +
> docs/61 simplification/config-flattening), run because that range rewrote the train step,
> the sampler, the collate path, the dataset output contract and the config layout immediately
> before a multi-day run. **13 agents** (9 whole-target reviewers + 4 refutation verifiers) plus
> runtime checks on an L40S.
>
> **The headline question — "is it bit-identical to before?" — is answered YES, by execution,
> not by reading.** A pre-refactor worktree was built at `e1c38ac` and run against the same data:
> **72/72 batch tensors bit-identical** (4 real subjects, native-z `D`=10/11/12, aug + respiratory
> on, deferral active), and **every loss/metric value equal to within 3.6e-7 relative — below the
> 1.8e-6 same-code CUDA rerun noise floor** — with exactly one exception, `recov_frac_heart`, which
> is the deliberate clamp removal. `vggt/` is untouched, so identical batch + identical model +
> identical loss ⇒ **the whole train step is bit-identical by composition.**
>
> **Four real defects were found, all in diagnostics/tooling, none affecting training numerics:**
> 1. `Train_Visuals_Augmentation` is **permanently dead** — `defer_input_images` made an old vacuous
>    guard bite. Runtime-confirmed: every other panel logged, this one absent.
> 2. **ED val visual panels are silently overwritten by ES** — all 15 keys written twice at the same
>    frozen step; wandb keeps the last.
> 3. **`frac_slots_dimmed` measures nothing** — its band is off by one plane, so it flags exactly the
>    slots that retain **100 %** of signal (0 overlap with truly-attenuated slots, at a 7–35 % rate).
>    Its unit test encodes the same error.
> 4. **`_widen` can silently truncate `val_per_subject.csv`** — it commits the widened column list
>    before the rewrite succeeds; fault-injected, a reader then saw **2 of 4 rows**.
>
> **Also: there are FIVE deliberate behaviour changes in this range, not the two usually cited.**
> Beyond `inside_random` removal and the `recov_frac` clamp: flip aug is now aggressive-only (so the
> shipped `moderate` arm's data stream changed), EF moved to CorSeg every val epoch, and —
> least obvious — **`sbatch/train_mri_volume_reference.sh` silently switched regularizer arms**
> from L1 TV (`tv=0.1`) to L2 diffusion (`diffusion=1000`) when its `CONFIG=` was repointed.
>
> **docs/61 §10.1's headline gap is now closed:** multi-epoch + resume was exercised end-to-end
> (4 epochs across 2 launches, clean). Two claims were **REFUTED** and are recorded here so they are
> not re-litigated: mid-epoch requeue starvation, and `export_resp_stack.py`'s `range(12)`.
>
> **Status: the refactor is cleared for the long run.** The 4 defects are diagnostic/robustness and
> can be fixed independently of it.
>
> **UPDATE 2026-08-01 — all findings resolved; see §11.** Every defect in §5 and every stale-text
> item in §7 is fixed and verified, each against a check proven to FIRE on the pre-fix code. Two
> exceptions, both deliberate: the ED/ES panel collision (§5.2) is **accepted, not fixed** —
> `media_val_ED_ES` already carries the ED+ES pair — and the five behaviour changes in §4 are
> **confirmed as intended**, including the reference sbatch's L1-TV → L2-diffusion arm switch.
> §11 also records **one error in this document** (§7's claim that all four `evaluation/` adapters
> pass `n_planes` is false) and **one error in the resolution itself**: the §5.1 fix shipped
> broken on its first pass — it omitted the `(B,S,518,518,3)/[0,255] → (B,S,3,518,518)/[0,1]`
> conversion, so the restored panel rendered a saturated 518×3 sliver, and the "panel appears in
> the wandb log" check **passed anyway**. Corrected and re-verified on the rendered PNG's pixels
> (§11.7). *For a diagnostic fix, "it now emits something" is not a proof.*
> A post-fix re-sweep then found **one more launch-blocker the original audit missed** (§11.8):
> 8 `sbatch/*.sh` still passed `max_img_per_gpu=12`, a key deleted in docs/59 — and Hydra refuses to
> override a key that does not exist, so those jobs died at `compose()`. The long-run script was
> never affected. Fixed; **all 55 override sets across every `sbatch/*.sh` now compose.**

---

## 1. Why this audit, and what standard it was held to

docs/61 shipped with an explicit self-declared gap (§10.1: "multi-epoch + SLURM requeue has never
been exercised end-to-end … now the largest untested surface in the repo") and a bit-identity claim
resting on two artifacts: a resolved-config diff and a 75-value float64 comparison of `loss.py`.
Both are good evidence for the specific things they cover; neither covers the *data pipeline*, which
is where the refactor did the most invasive work (dataset output contract, deferral, sampler,
collate, worker seeding).

The standard applied here: **a claim of "unchanged" is only credible if old and new code were run
side by side on the same input and the outputs compared bitwise.** Reading is used to find
mechanisms; execution is used to settle them.

### 1.1 Method

- A pre-refactor **git worktree** at `e1c38ac` (`/home/minsukc/vggt-proveit-old`, with `scratch`/`data`
  symlinked so it resolves the same cohort and the same node-local monai cache).
- **9 reviewers**, each reading its whole target plus the full `training/` diff, with a distinct
  lens: trainer loop/resume · sampler+dataloader+worker seeding · `mri_dataset`+`preprocess` ·
  `gpu_aug`+`respiratory` · `loss.py` · `trainer_viz`+`val_logging` · `run_log`+`wandb`+`ef_eval` ·
  `train_utils` deletions · config-delta + repo-wide call-site sweep.
- **4 refutation verifiers**, each briefed to *disprove* a specific finding and to default to
  "not a bug". Two findings died here or were reclassified.
- **Orchestrator-only runtime work** on the allocated L40S (subagents were barred from heavy runs):
  the old-vs-new dumps, the loss comparison, the 4-epoch resume run, and the fault injections.

---

## 2. The bit-identity result

### 2.1 Data pipeline — 72/72 tensors bit-identical

A harness (`dump_batches.py`) reproduces the trainer's exact data path in either tree —
`set_seeds(seed_value, …)` → `instantiate(cfg.data.train)` → `get_loader(epoch=0)` →
`copy_data_to_device` → `gpu_augment_batch(train=True)` — and dumps every batch tensor.

Confounds neutralised so the comparison is apples-to-apples:

| Confound | Handling |
|---|---|
| `inside_random` removal | old run overrides `data.train.common_config.inside_random=false` (⇒ both use the sampler permutation) |
| L2-diffusion default | old run uses `mri_volume_diffusion`, the config `default.yaml` was derived from |
| Flip tier change | both run `data.augmentation.tier=aggressive`, whose flip is present in **both** trees (a comment-stripped diff of `gpu_aug.py` shows the only behavioural delta is the two `RandFlipd` lines removed from conservative/moderate) |

Result over 4 batches (4 distinct real subjects, native-z `D` = 10, 12, 11, 10):

```
bit-identical tensors: 72   differing: 0
VERDICT: ALL BIT-IDENTICAL
keys only in OLD: ['ids', 'rotations']       # the deleted dead keys, as documented
```

This single comparison covers, jointly: the sampler permutation, worker seeding, the dataset's
sampling RNG consumption order, `S == D` native-z slot construction, `scanner_coords`, `z_scale`/
`dz_mm`, the affine + photometric aug pipeline, the respiratory deform-and-reslice, the
`defer_input_images` rebuild, the collate, and `copy_data_to_device`.

### 2.2 Loss — every delta below the CUDA noise floor except the deliberate one

`dump_loss.py` runs the configured `MultitaskLoss` on the *same* dumped batches with a
deterministic pseudo-prediction (`scanner_coords + fixed residual field`), on **both** the train
path (`requires_grad=True` ⇒ branchless vectorized metrics) and the val path (`requires_grad=False`
⇒ the loop-based val-only metrics), at float64.

The splat's scatter-add is nondeterministic on CUDA, so a **noise floor** was measured first by
re-running the *new* code against itself:

```
NEW vs NEW (same code, rerun): compared=248  identical=216  differing=32
OLD vs NEW                   : compared=228  identical=199  differing=29

max relative OLD-vs-NEW  : 2.406e-01     <- entirely one metric
max relative NEW-vs-NEW  : 1.827e-06     <- the noise floor
OLD-vs-NEW deltas above 1e-5 relative: {'b1/val/metric_recov_frac_heart': 0.2406}
```

Read that carefully: **old-vs-new differs in FEWER values than the same code differs from itself**,
and every non-`recov_frac` delta is ≤ 3.6e-7 relative — an order of magnitude *below* the rerun
noise. Several metrics even swap which side is larger between the train and val runs of the same
batch, the signature of run-to-run nondeterminism rather than a code difference. The single real
outlier is `recov_frac_heart` (old `1.5` = the clamp ceiling, new `1.975`), i.e. docs/61 §6.3's
deliberate clamp removal.

An independent reviewer reproduced this in a separate process pair across 11 scenarios (including
float64 inputs, NaN-outside-mask, empty/degenerate ROI, `D`=5 and 12, zero residual) and found
**0 bitwise diffs on everything including the gradient w.r.t. `world_points`**, with the same lone
`recov_frac_heart` exception.

> Side note, correcting docs/61 §6.2: the claim that the synthetic harness *structurally cannot*
> reach the `recov_frac` family is **not correct** — it is reachable on synthetic data with `D=12`
> and perturbed predictions. The stated verification gap is closable in-repo.

### 2.3 The rest of the bit-identity case

| Check | Result |
|---|---|
| `vggt/` (the 941M model) | `git diff` over the range is **empty** — model code untouched |
| `preprocess.py`, `freeze.py`, `gradient_clip.py`, `logging.py` | byte-identical |
| Train step at `accum_steps==1` | bit-identical incl. gradients (CPU probe: old `_run_steps_on_batch_chunks` vs new `_run_step_and_backward`; the only deleted arithmetic is `loss /= 1`) |
| `save_checkpoint` / `_load_resuming_checkpoint` | byte-identical to pre-refactor |
| Freeze partition under `["*patch_embed*"]` | replayed against a **real pre-refactor run's** recorded `frozen.txt`/`trainable.txt`: **344 frozen / 930 trainable, 0 missing, 0 extra** |
| Optimizer param groups | exactly one group containing all params, in both trees (no `param_names`/`module_cls_names` ⇒ `itertools.product` yields one) |
| Worker seed formula | runtime: `43,44,45,46` at epoch 0 and `12388` at epoch 1 = `worker_id + 42 + 1 + epoch*12345`. Identical in both trees — docs/61 §2.2's "+1 from `world_size`" claim **holds** |
| `pytest tests/` | old **256 passed** → new **335 passed** |

Composing 2.1 + 2.2 + `vggt/` untouched: identical input batch, identical model code and weights,
identical loss ⇒ **the full training step is bit-identical.**

---

## 3. The one measured numeric deviation (not on the shipped config)

With **both** `data.augmentation.enable=false` **and** `data.augmentation.respiratory.enable=false`,
`images` differs by **≤ 1 ULP** (`maxabs = 5.960464e-08`, i.e. `2^-24`, on values in `[0,1]`):

```
✗ batch 0 images: NOT bit-identical | maxabs=5.960464e-08 mean=1.104e-08 mismatched=4469163/8049720
```

**Mechanism, measured (an initial hypothesis was refuted first).** The dataset built `images` on
**CPU** in a dataloader worker; the new deferral path builds them on **CUDA** in
`gpu_augment_batch`'s no-augmentation early return. Reconstructing both paths from the identical
dumped `phases`/`timesteps`/`slice_indices`:

```
dev=cpu   cast_before_index=True/False:  ==old True   ==new False
dev=cuda  cast_before_index=True/False:  ==old False  ==new True
```

So it is purely the device of the `256→518` bilinear `F.interpolate`; dtype, `align_corners`, the
`*255 → clamp → /255` order and the RGB replication all match exactly. (A first probe on *random*
input showed CPU and CUDA agreeing bitwise, which wrongly refuted the hypothesis — real data
disagrees. Random-input probes are weak evidence for last-ULP questions.)

**Why it does not matter:** on the shipped config `respiratory.enable: true`, so **both** trees take
the respiratory branch and extract on GPU from the fp16 `phases` — bit-identical, as §2.1 shows.
The deviation is confined to a configuration nothing ships, and 6e-8 is ~5 orders of magnitude below
the bf16 precision the model computes in.

**Action:** the comment at `gpu_aug.py:290` — *"float32, matching the dataset's own extraction
exactly … deferring must be a numeric no-op"* — is one ULP too strong. It matches in dtype and
algorithm, not bit-for-bit, because the device differs.

---

## 4. FIVE deliberate behaviour changes, not two

docs/61 highlights two (L2-diffusion default, `recov_frac` clamp) and the commit message adds
`inside_random`. The complete list of things that change *what a run does* relative to `e1c38ac`:

| # | Change | Effect | Where documented |
|---|---|---|---|
| 1 | `inside_random` removed | train now sees a true permutation (935 unique/epoch) instead of sampling **with replacement** | docs/61 §3.3 |
| 2 | Flip aug → **aggressive-tier only** | the shipped `moderate` tier no longer flips ⇒ its data stream differs from pre-refactor | docs/61 §7 |
| 3 | `default.yaml` = **L2 diffusion** (`tv=0`, `diffusion=1000`) | ⚠️ **`sbatch/train_mri_volume_reference.sh` was on `mri_volume` (`tv=0.1`, `diffusion=0`) and its `CONFIG=` was repointed to `default` — so the repo's documented cluster entry point silently changed regularizer arm.** Its header still reads "mri_volume.yaml is now the reference pipeline" | docs/61 §5.1 documents the default; **the arm switch is not called out anywhere** |
| 4 | `recov_frac_heart` clamp removed | metric only, no training effect | docs/61 §6.3 |
| 5 | `ef_seg_backend` **nnU-Net → CorSeg**, every val epoch (5→1) | val diagnostics only. The old tree had no `ef_seg_backend` key at all — `trainer_viz` called `run_nnunet` unconditionally | docs/61 §7 (cadence only) |

**#3 is the one to confirm before the long run.** `exp_bspline` is now the only surviving `tv=0.1`
arm; if the reference series is meant to stay on L1 TV it needs
`loss.volume.tv_weight=0.1 loss.volume.diffusion_weight=0.0` in its overrides.

Cross-check of every remaining resolved-config delta (old `mri_volume_diffusion` → new `default`)
found them all inert, in particular:

- **`per_slot: true` removed from the config** — safe **twice over**: the `RespiratoryConfig`
  dataclass default is `per_slot: bool = True` (equal to the old explicit value), *and* the field is
  unreachable because `group_by_burst: true` with a non-`None` `group_ids` always takes the other
  branch.
- `accum_steps`, `param_group_modifiers`, `fix_img_num`, `fix_aspect_ratio`, `augs.aspects`,
  `video_logging_fps`, `visuals_per_batch_to_log`, the `visuals_keys_to_log` payload — **zero live
  readers**; `{train: {}, val: {}}` still satisfies the two surviving `is not None` / `phase in …`
  tests.
- No config key is read-but-missing, and the only never-read leaves (`dataset_name`,
  `mri_data_mode`) are consumed by the `exp_name` YAML interpolation.

---

## 5. Confirmed defects

All four are **diagnostics or robustness**. None changes a weight, a loss, or a sampled batch.

### 5.1 🔴 `Train_Visuals_Augmentation` is permanently dead — `trainer.py:976`

```python
_orig_images = batch["images"].detach().clone() if (_aug_log and "images" in batch) else None
```

`defer_input_images: true` (train) means the dataset **omits** `images`; it is created a few lines
later inside `gpu_augment_batch`. So `_orig_images` is always `None` and
`_log_augmentation_to_wandb` (`trainer_viz.py:645`) returns on its first line.

- The `"images" in batch` guard predates the refactor (`271856b`) where it was **vacuous** — the
  dataset always emitted the key. Deferral (`72c215a`) made it bite. **New regression.**
- Every conjunct of `_aug_log` is True in a real run: transforms not None, respiratory on,
  `log_visuals: True`, `filmstrip_every_n_val_epochs: 3` ⇒ true at epoch 0 / iter 0.
- **Runtime-confirmed** on a real GPU run: the offline wandb transaction log contains
  `Train_Visuals_Volume`, `Train_Visuals_DVF`, `Val_Visuals_{Volume,DVF,Lookup}`,
  `Val_Visuals_cardiac_cycle_gif`, `val_motion_mask_example` — and **no** aug panel at all.
- Silent: no warning, no crash, no test.

**Fix:** build the pre-aug snapshot from `phases` via `extract_slices_from_phases` when the key is
absent (the same two lines `trainer_viz._subject_device_batch` already uses). Per-run workaround:
`data.train.dataset.dataset_configs.0.defer_input_images=false`.

### 5.2 🔴 ED val visual panels are silently overwritten by ES — `trainer_viz.py:1118,1169`

The gate moved from the raw `val_idx` to `subj_idx`:

```python
subj_idx  = _vt[val_idx % len(_vt)][0] if _vt else val_idx
should_log = subj_idx in VAL_VISUAL_SUBJECT_INDICES
name, group = "Val_Visuals", f"media_val_subj{subj_idx}_{val_sid or 'unknown'}"
```

Under `ef_val_sweep: true`, `val_targets = ed_list + es_list`, so subject *i* appears at
`seq_index = i` (ED) **and** `N + i` (ES); both map to the same `subj_idx`, both pass the gate, and
both log the **same key** at the **same step** — `log_step` for val is `self.steps["train"]`, which
`_step` never increments during a val epoch.

- Traced on the real 133-subject pooled sweep: **15 distinct (key, step) pairs, all 15 written
  twice** (5 subjects × Volume/DVF/Lookup).
- wandb's last-write-wins was **measured**, not assumed: replaying the exact
  `wandb.log({name: data}, step=step)` calls into an offline run and reading back the transaction
  log gives **1 history record, ES only**.
- `limit_val_batches` is no refuge: `trainer.py:746` *overrides* it with `len(val_targets)`, so the
  ES half is always visited.
- Pre-refactor the gate was `val_idx in (0, 7, 14, 21)` — all `< N` — so only the ED half matched:
  **4 keys, 0 collisions.** **New regression.**

Consequence: you see 5 subjects at ES every val epoch while the surrounding comment promises an
ED/ES pair. Partial mitigation: `_log_ed_es_panels` writes a separate non-colliding key, but only
every 3rd epoch and without DVF/Lookup rows.

**Fix:** put the role in the key (`_ED`/`_ES`, derivable as `"ED" if val_idx < len(vt)//2 else "ES"`,
the same expression `_stash_ed_es` already uses), or restrict `should_log` to the ED half.

### 5.3 🔴 `frac_slots_dimmed` is 100 % false positives — `val_logging.py:147`

```python
off    = (landing < 0) | (landing > D - 1)
dimmed = ((landing > 0) & (landing < 1)) | ((landing > D - 2) & (landing < D - 1))
```

The comment justifies `dimmed` as "inside the slab but not on an exact end plane, so trilinear
interpolation mixes in zero-padding". That is false: `extract_slices_with_respiratory_vec` uses
`padding_mode="zeros", align_corners=True`, so a landing in `(0,1)` interpolates between planes 0
and 1 — **both real**. Zero-padding only enters on `(-1,0) ∪ (D-1,D)`.

Measured against the real reslicer (uniform-ones volume, retained intensity):

| landing | retained | code `off` | code `dimmed` |
|---|---|---|---|
| −1.0 | **0.0000** | True | False |
| −0.5 | **0.5000** | True | False |
| 0.25 / 0.5 / 0.999 | **1.0000** | False | **True** |
| D−2+ε … D−1−ε | **1.0000** | False | **True** |
| D−1+0.25 | **0.7500** | True | False |
| D−0.5 | **0.5000** | True | False |
| ≥ D | **0.0000** | True | False |

Under the shipped respiratory config across six real `(D, dz)` pairs × 4000 draws:

```
  D    dz | code dimmed  TRUE partial  code off  TRUE blank | overlap
  5  12.0 |      34.98%        19.18%     22.85%       3.67% |    0
 12  12.0 |      14.77%         7.76%      9.38%       1.62% |    0
 21   5.0 |       7.41%         4.58%      8.17%       3.60% |    0
```

`overlap(dimmed ∧ truly-attenuated) = 0` in every configuration, at a 7–35 % reported rate.

**`frac_slots_offslab` is numerically correct** (`off ⇔ retained < 1`, and
`n_off == n_true_partial + n_true_blank` exactly) — but its name and docstring read as *total* loss,
so e.g. the `D=5` subject's 22.85 % is really 3.67 % blanked + 19.18 % partially attenuated. That is
a semantics defect, not a value defect.

⚠️ **The unit test encodes the same wrong band.**
`tests/test_val_logging_helpers.py:274` drives plane 0 to landing 0.5 and asserts `dimmed == 0.25`;
that landing measurably retains 1.0000. The test would **block** a fix rather than catch the bug.

New in `72c215a` (docs/60 item 9) — `offslab`/`dimmed` do not exist at `e1c38ac`. Consumers are
`{train,val}/resp/*` scalars and one `val_per_subject.csv` column; nothing reads it into loss,
model, sampling, or checkpointing. Correct bands would be `off = (landing <= -1) | (landing >= D)`
and `dimmed = (-1 < landing < 0) | (D-1 < landing < D)`.

### 5.4 🟠 `_widen` can silently truncate `val_per_subject.csv` — `run_log.py:188`

```python
self._subject_fields = self._subject_fields + sorted(new_keys)   # committed FIRST
...
with open(tmp, "w", newline="") as f:                            # can fail
...
os.replace(tmp, path)          # "atomic: a crash mid-rewrite leaves the old file"
```

The `os.replace` is atomic for the **file**, but `_subject_fields` is already widened in memory. If
the rewrite raises, the exception unwinds to `subject_row`'s handler, which only logs a warning —
and the next call sees `new_keys == []`, so the widen is **never retried**. Every subsequent row is
written with N+k fields under an N-field header, and `pandas.read_csv(on_bad_lines="skip")` drops
over-long rows **without warning**.

Fault-injected (ENOSPC on the `.tmp` write, then 2 more rows):

```
epoch,seq_name,t_target,metric_a          <- narrow header
0,s1,0,1.0
0,s2,1,2.0
1,s4,1,4.0,8.8                            <- 5 fields
2,s5,0,5.0,7.7                            <- 5 fields
rows a reader sees: 2 | seq_names: ['s1', 's2']       (4 were written, s3 lost outright)
```

Needs an I/O error to trigger — realistic on GPFS over a multi-day run, which is exactly the horizon
this file exists for. **Fix:** assign `self._subject_fields` only after `os.replace` succeeds.

Related, lower reachability: `_resolve_fields` catches an unreadable header and falls through to a
fresh field list, but `subject_row` then sees `new_file == False` and writes no header.

### 5.5 Lower-severity confirmed items

| Item | Location | Evidence | Note |
|---|---|---|---|
| **`seed_value` does not reach the sampler** | `dynamic_dataloader.py` (`seed=42` default); `trainer._setup_dataloaders` sets `.seed` *after* construction | **Measured**: `seed_value=42` and `=7` produce the **identical** subject permutation; epochs do differ | Pre-existing wiring, but **inert until now** — `inside_random` used to discard the permutation. A second seed would silently reuse run 1's order |
| **4 tools entry points crash at Hydra compose** | `tools/verify_env_migration.sh:66-75` (all 10 `run_case` lines), `tools/profile_trainstep_compile.py:27`, `tools/render_cardiac_filmstrip_multislice.py:49-53`, `tools/render_ed_input_vs_pred_artifact.py:45-49` | Direct inspection; they name `mri_volume` / `mri_volume_diffusion` / `mri_volume_bspline`, none of which exist | Missed by the 36-reference repoint sweep. `verify_env_migration.sh` is the script prescribed for dependency-bump verification, so it is now useless in its main block |
| **`_ef_stats` guard misses its own documented case** | `ef_eval.py:174` | **Measured**: `_ef_stats(gts=[54,55,56], preds=[40,50,60])` → `slope: 10.0`. σ(gts)=0.82 ≫ 1e-6 | The docstring cites exactly "54/55/56" as what it rejects; only *exactly constant* GT is rejected. Not biting today (val healthy σ=6.2, n=60); reachable on a re-seeded split or finer grouping |
| **`target_size` half-threaded** | `gpu_aug.py:61`, `respiratory.py:37` still hardcode `INPUT_IMG_SIZE = 518` | Reproduced: `img_size=490` → `scanner_coords (…,490,490,3)` vs `images (…,518,518,3)` → the `assert scanner_coords.shape == dvf.shape` fires (a broadcast `RuntimeError` under `python -O`) | **Reclassified by the verifier**: at `e1c38ac` `self.target_size` was assigned and *never read*, so `img_size != 518` was a silently-ignored dead knob. It went from *no-op* to *hard crash*, not from working to broken. Nothing ships a non-518 value. The comment at `mri_dataset.py:125` ("really honoured now") is false under train deferral |

---

## 6. REFUTED — recorded so they are not re-litigated

### 6.1 "Mid-epoch requeue replays the same permutation prefix forever" — REFUTED

The *mechanism* is real: there is no mid-epoch checkpoint, `save_checkpoint` runs only after a
completed `train_epoch`, and `DistributedSampler.set_epoch(N)` replays a byte-identical permutation.
But:

- **The differentiating premise is false.** `inside_random`'s `random.randint` ran in a dataloader
  **worker**, whose `random` module is seeded as a pure function of `(worker_id, seed, epoch)` — no
  clock, no urandom — in **both** trees. Two simulated launches at the same epoch produce identical
  draws in the old code too. Nothing regressed.
- **The consequence needs walltime < one epoch, off by ~600×.**
  `sbatch/train_mri_volume_reference.sh` requests `--time=14-00:00:00`; measured epoch cost from a
  real run log is **~34 min** (935 train + 266 val steps at ~1.6 s). 200 epochs ≈ 4.7 days, so the
  whole run fits inside one walltime and the SIGUSR1 path likely never fires.
- Worst real cost is ≤ 1 epoch of recomputed work per requeue; optimizer/scaler/`steps` are all
  restored from the epoch boundary, so no LR-schedule drift.
- `launch.py` is **byte-identical** to `e1c38ac`.

### 6.2 "`export_resp_stack.py` loops `range(12)` over native-`D` volumes" — REAL but PRE-EXISTING and unreachable

`N_CANON_Z = 12` with `shifted_vol[z]` genuinely breaks for `D != 12` (the script's own subjects are
`D=9` and `D=11`). But those lines are **byte-identical at `e1c38ac`** — the breakage dates to the
native-z migration (docs/58), not this refactor. And it is already unreachable: the script's
`SPLIT_FILE` is the deprecated `random_8_1_1.txt`, and `MRIDataset._find_subjects` now **raises
`FileNotFoundError`** on its missing entries, so construction fails before the loop. The refactor's
only change here (passing the subject's real `dz_mm`) is a genuine **correctness improvement** that
happens to sit behind two pre-existing failures.

*(Two further native-z stalenesses in the same file if it is ever revived: `sample_resp_disp(1,
N_CANON_Z, …)` allocates 12 slots instead of `D`, and `SPACING_XYZ` stamps a hardcoded 12 mm
z-spacing on the output affine.)*

---

## 7. Stale text found (no code impact)

- **docs/61 §6.2** — "the synthetic harness structurally cannot reach the recov family" is **wrong**;
  it is reachable at `D=12` with perturbed predictions.
- **`trainer.py:1112`** — cites `_run_steps_on_batch_chunks`, deleted in this very pass. The
  NaN-backward-skip it refers to **does survive**, at `trainer.py:1071`.
- **`evaluation/engine/build_inputs/cmrxrecon.py:129-133`** — says `reslice_volume_vec` "still
  defaults to `SPACING_MM`"; that default was removed, so the comment now asserts the opposite of
  the code.
- **`respiratory.py:133-138`** — says `N_CANON_PLANES` is kept for "`evaluation/` callers not yet
  migrated"; all four adapters now pass `n_planes` explicitly.
- **`sbatch/train_mri_volume_reference.sh:18-19`** — "mri_volume.yaml is now the reference pipeline";
  that file no longer exists, and see §4 #3.
- **`sbatch/train_pooled1337_nativez.sh:22`** and the repo guide still name `exp_diffusion.yaml`,
  deleted in this pass.
- The repo guide also states the default aug tier is `conservative` (actual: `moderate`, in both
  trees) and `filmstrip_every_n_val_epochs: 5` (actual: `3`).
- **`evaluation/engine/run_vggt.py:412`** — `--config` default is the nonexistent
  `mri_volume_diffusion`; it is provenance-only (no Hydra compose), but 18 rows of
  `evaluation/models.json` now name a config that cannot be composed.

---

## 8. What docs/61 §10.1 asked for — now done

**Multi-epoch + resume was exercised end-to-end** on the L40S: a 2-epoch run, then a relaunch with
`max_epochs=4` against the same `log_dir`.

```
L0 launch: resumed_from_epoch=0  max_epochs=2  steps_at_launch={'train': 0, 'val': 0}  wandb_id=ski428pi
L1 exit  : status=completed  final_epoch=1  steps={'train': 8, 'val': 12}
L2 launch: resumed_from_epoch=2  max_epochs=4  steps_at_launch={'train': 8, 'val': 6}  wandb_id=m44t4q3f
L3 exit  : status=completed  final_epoch=3  steps={'train': 16, 'val': 18}
```

`tools/load_run.py` reads it back cleanly: 2 launches, 514 scalars over 130 names, 24 per-subject
rows (**6 per epoch × 4 epochs**, 44 columns), and a per-source groupby. Zero tracebacks in either
segment. `wandb_id` is populated, confirming the docs/61 §3.1 fix, and `baseline_identity.json`
exists with per-phase **and** per-subject entries, confirming the §4 regression is repaired.

**Residual gap:** the resume happened at an **epoch boundary**, so no steps were replayed —
`metrics.jsonl` had **514 rows / 514 unique `(name, step)`, 0 duplicates**. The duplicate-row dedupe
path therefore remains verified only by simulation (a reviewer truncated both files mid-line and
restarted: `run_meta.jsonl` got 2 lines, exactly one `metrics.jsonl` row was lost, the CSV kept its
header, `_needs_newline` prevented a concatenation cascade). A real `scontrol requeue` was **not**
run — inside an interactive allocation it would requeue the session's own job.

---

## 9. Reproduction

```bash
# pre-refactor worktree
git worktree add --detach ../vggt-proveit-old e1c38ac
ln -s /path/to/gpfs/vggt ../vggt-proveit-old/scratch
ln -s scratch/data ../vggt-proveit-old/data

# batch bit-identity (run in each tree, then compare)
CFG_DIR=$PWD/training/config PYTHONPATH=training:. python dump_batches.py \
    default  batches_new.pt 4 data.augmentation.tier=aggressive
CFG_DIR=$PWD/training/config PYTHONPATH=training:. python dump_batches.py \
    mri_volume_diffusion batches_old.pt 4 \
    data.augmentation.tier=aggressive data.train.common_config.inside_random=false
python cmp_batches.py batches_old.pt batches_new.pt

# loss bit-identity — ALWAYS measure the same-code rerun noise floor first
CFG_DIR=.../config PYTHONPATH=training:. python dump_loss.py default batches_new.pt loss_new.pt
CFG_DIR=.../config PYTHONPATH=training:. python dump_loss.py default batches_new.pt loss_new2.pt   # noise floor
CFG_DIR=.../config PYTHONPATH=training:. python dump_loss.py mri_volume_diffusion batches_new.pt loss_old.pt

# multi-epoch + resume
WANDB_MODE=offline PYTHONPATH=training:. python training/launch.py --config default \
    max_epochs=2 limit_train_batches=4 limit_val_batches=6 ef_val_sweep=false \
    logging.log_dir=scratch/logs/_proveit_requeue
# ... then relaunch the same command with max_epochs=4 and the SAME log_dir
PYTHONPATH=training:. python tools/load_run.py scratch/logs/_proveit_requeue
```

Harness scripts were written to a scratch dir, not the repo, per the "throwaway probes don't go in
`tools/`" rule; the three above are worth re-creating if this comparison is ever needed again (e.g.
before the next large refactor).

---

## 10. Generalizable lessons

1. **"Bit-identical" is a claim about execution, not about a diff.** The resolved-config diff and
   the 75-value loss comparison in docs/61 were both sound *and* both blind to the data pipeline —
   which is where this refactor did its most invasive work. A pre-refactor worktree costs minutes
   and settles the question directly.
2. **On a nondeterministic device, measure the noise floor before interpreting any delta.** The
   splat's scatter-add makes the *same code* differ from itself at 1.8e-6 relative. Without that
   baseline, the 29 old-vs-new differences look alarming; with it, they are provably smaller than
   the noise and the single real change is unmistakable.
3. **Random-input probes are weak evidence for last-ULP questions.** A CPU-vs-CUDA interpolate probe
   on random data showed exact agreement and wrongly killed a correct hypothesis; the same probe on
   real cached data reproduced the 1-ULP difference immediately.
4. **A guard that is vacuous when written becomes a silent bug when its precondition changes.**
   `"images" in batch` was a no-op for months, and `defer_input_images` turned it into a permanently
   dead diagnostic — the same shape as the identity-baseline regression docs/61 §4 already
   documents. Grepping for *guards keyed on a contract you just changed* is as important as grepping
   for callers.
5. **A test can encode the bug.** `test_resp_offslab_counts_a_partial_plane_as_dimmed` asserts the
   wrong band, so it passes today and would fail a correct fix. Green tests written from the same
   mental model as the code are not independent evidence.
6. **When repointing N references, enumerate mechanically.** 36 of 40 config references were
   repointed; the 4 misses are all in `tools/`, including the script the project designates for
   verifying dependency bumps.

---

## 11. Resolution — what was fixed, and how each fix was proven (2026-08-01)

Everything below was applied *after* the audit above, in the same working tree, before the long
run. §1–§10 are left exactly as written; this section only records what happened to each finding.

**Verification standard used here is the same one §1.1 applied:** no fix is recorded as done
unless its check was first shown to **FIRE on the pre-fix code**. Every "proven" claim below means
old-code-fails / new-code-passes was executed, not reasoned.

### 11.1 Summary table

| Finding | Disposition | Proof |
|---|---|---|
| §5.1 dead `Train_Visuals_Augmentation` | **FIXED** (took two passes — see 11.7) | real run: logged PNG's original row has std 0.436, real anatomy |
| §5.2 ED panels overwritten by ES | **ACCEPTED, not fixed** | `media_val_ED_ES` already carries the pair — see 11.3 |
| §5.3 `frac_slots_dimmed` false positives | **FIXED** | 17/17 landings match the real reslicer; test rewritten |
| §5.4 `_widen` truncation | **FIXED** | fault injection: old 2 rows → new 4 rows |
| §5.5 `seed_value` not reaching sampler | **FIXED** | old: seeds 42/7 identical → new: differ, 42 bit-identical |
| §5.5 4 `tools/` entry points | **FIXED** | all 7 repointed config+override combinations compose |
| §5.5 `_ef_stats` guard | **FIXED** (1 line) | 54/55/56 → `None`; suite green |
| §5.5 `target_size` half-threaded | **FIXED** (fail early) | now raises at construction with an actionable message |
| §3 `gpu_aug.py:290` overclaiming comment | **FIXED** | comment now states the 1-ULP CPU-vs-CUDA difference |
| §5.4 "related": unreadable header ⇒ no header written | **FIXED** | fault injection: old → 1 junk column, new → 2 clean rows |
| §4 five behaviour changes | **CONFIRMED INTENDED** | incl. the L2-diffusion arm — see 11.2 |
| §7 stale text (7 items) | **FIXED**, one of them **CORRECTED** | see 11.4 |
| docs/61 §6.2's "structurally cannot reach" | **CORRECTED in docs/61** | inline warning added at that section |
| **NEW** — `max_img_per_gpu=12` in 8 `sbatch/*.sh` | **FIXED** (found post-fix, see 11.8) | all **55** override sets across every `sbatch/*.sh` now compose |

`pytest tests/`: **335 passed** before and after (the §5.3 fix rewrote one test rather than adding
one). Total diff: 18 files, +197/−76.

### 11.2 The §4 #3 decision: the reference series stays on L2 diffusion

§4 flagged that `sbatch/train_mri_volume_reference.sh` silently switched regularizer arm (L1 TV
`tv=0.1` → L2 diffusion `diffusion=1000`) when its `CONFIG=` was repointed to `default`.
**Confirmed as intended — the series continues on L2 diffusion.** The switch is now written into
the script header as a deliberate choice, with the exact override to revert it
(`loss.volume.tv_weight=0.1 loss.volume.diffusion_weight=0.0`) and a note that `exp_bspline` is
the only shipped config still on `tv=0.1`. The other four changes are likewise confirmed.

### 11.3 Why §5.2 was accepted rather than fixed

`_log_ed_es_panels` is gated by `self._viz_ed_es` = `epoch % filmstrip_every_n_val_epochs == 0`
(=3), over the **same** subject set (`VAL_VISUAL_SUBJECT_INDICES` is an alias of
`_ED_ES_SUBJECTS`), and writes a 6-row per-z figure — input/V_gt/V_canon for **both** ED and ES —
under the non-colliding key `media_val_ED_ES/…`. That is a *richer* panel than the one being
clobbered, so the ED information is not actually lost.

Residual, accepted knowingly:

- on the 2-in-3 epochs without `media_val_ED_ES`, `Val_Visuals` shows **ES only**;
- **DVF and Lookup rows for ED are never rendered anywhere** — `_log_ed_es_panels` has Volume rows
  only;
- the key and its comment still promise an ED/ES pair and deliver ES.

The proposed one-line key fix (`"ED" if val_idx < len(vt)//2 else "ES"`) was **rejected**: it
re-derives the role from index arithmetic that can silently disagree with the actual
`val_targets` construction, i.e. it can label a panel wrongly, which is worse than a known
overwrite. If this is ever revisited, take the role from the same source `_stash_ed_es` reads
rather than recomputing it.

### 11.4 Corrections to this document

**§7's `respiratory.py` item is wrong.** It states "all four adapters now pass `n_planes`
explicitly". Checked: **only `cmrxrecon.py` does**; `acdc.py`, `miitt.py` and `ocmr.py` pass
neither `n_planes` nor `group_ids`.

It does not matter, for a reason §7 did not state: `n_planes` is consulted **only** inside
`if cfg.group_by_burst and group_ids is not None`. With no `group_ids`, all three take the
per-slot `else` branch and never reach the `N_CANON_PLANES` fallback at all. So no live caller
can currently be mis-graded by it — but the *reason* is unreachability, not migration. The
comment now says that, and names the three adapters that pass nothing.

(§7's other six items were confirmed as written and fixed as described.)

### 11.5 Fix details worth carrying forward

**§5.3 — `off` was left alone on purpose.** §5.3 proposed narrowing `off` to
`(landing <= -1) | (landing >= D)` so `off` and `dimmed` become disjoint. Rejected: `off` as
written is already exactly `retained < 1`, and narrowing it would silently change the meaning of
an already-logged scalar. Instead only `dimmed` moved, onto the true partial band
`(-1,0) ∪ (D-1,D)`, giving `dimmed ⊂ off` — so blank-fraction is now simply `offslab − dimmed`
and no existing metric changes meaning. Verified against `extract_slices_with_respiratory_vec`
on a uniform-ones volume at 17 landings, all 17 consistent, including the five that were
previously 100% false positives (0.25, 0.5, 0.999, D−2+ε, D−1−ε → now `dimmed = 0`).

**§5.3's test was rewritten, not extended.** `test_resp_offslab_counts_a_partial_plane_as_dimmed`
asserted the wrong band (landing 0.5 → `dimmed == 0.25`), so it would have blocked the fix. It now
pins all three regimes at once — 0.5 (two real planes, not dimmed), −0.5 (partial), −1.5 (fully
blank, `off` but not `dimmed`) — and asserts `dimmed ⊆ off`.

**§5.5 seed — the fix is a property setter, not a re-ordering.** `DynamicTorchDataset.seed`
became a property whose setter also assigns `self.sampler.seed`, because the sampler is built in
`__init__` (ctor default 42) while `trainer._setup_dataloaders` assigns the real `seed_value`
afterwards. Measured: pre-fix, `seed_value=42` and `=7` produce the **identical** permutation;
post-fix they differ, epochs still differ within a run, and **`seed=42` is byte-identical to
before** — so the shipped configuration is unaffected.

**§5.5 `_ef_stats` — deliberately kept to one line.** The threshold moved `1e-6 → 1.0` EF
percentage points and nothing else. An earlier version of this fix also returned and logged a
`gt_std_pct` field; that was **reverted as scope creep** — nothing consumes it, and it would have
added new metric names to a run that was about to start. Note the threshold is bounded on both
sides by documented cases: it must reject σ=0.82 (the slope-10.0 example in the function's own
docstring) and must keep σ=1.41 (`test_ef_group_metrics.py`'s narrow group, which recovers its
0.5 slope correctly). The finding is **latent, not live** — real groups are σ 6.2 / 16.2.

**§5.5 `target_size` — rejected rather than threaded.** Threading `R` through
`gpu_aug.INPUT_IMG_SIZE` and `respiratory.INPUT_IMG_SIZE` would change signatures that
`evaluation/`'s adapters call, for a knob nothing ships. `MRIDataset.__init__` now raises on
`target_size != 518` with a message naming both hardcoded sites, converting a late shape assert
(or a `python -O` broadcast error mid-step) into an immediate, explained failure. The false
comment claiming R was "really honoured now" is replaced with the honest half-threaded statement.

**§5.5 `verify_env_migration.sh` also had a second break.** Beyond the dead config names, three
`run_case` lines passed `max_img_per_gpu=12`, a key deleted in docs/59 F9. Both are fixed; the L1
TV arm is reproduced via a `TV_ARM` override so the matrix still covers both regularizers. All 7
distinct config+override combinations were confirmed to compose under Hydra.

**§5.4's "related" item was a second, independent data-loss path.** If `val_per_subject.csv`
exists but its first line is unreadable, `_resolve_fields` invented a fresh field list while
`subject_row` still computed `new_file == False` — so **no header was ever written** and every
subsequent row was appended under whatever junk the first line held. Fault-injected (a file
starting with NUL bytes): pre-fix, `pandas` read the garbage line as the header and the two real
rows collapsed into a single meaningless `Unnamed: 0` column. The damaged file is now **renamed
aside** (`val_per_subject.csv.corrupt.<ts>`) rather than appended to, so the next write creates a
proper new file with a header and nothing is destroyed. Both fixes together mean the CSV now
survives the two realistic multi-day failure modes: a failed widen (ENOSPC) and a corrupt header
from a killed resume.

**§3's action item is done.** The comment at `gpu_aug.py` claiming deferral "must be a numeric
no-op" now states the measured truth: dtype, `align_corners`, the `*255→clamp→/255` order and RGB
replication all match, but the CPU→CUDA move of the 256→518 bilinear `interpolate` differs by up
to 1 ULP (5.96e-08), reachable only with both affine and respiratory off.

### 11.6 Additional lesson

7. **A "fix" that adds a metric is not free.** Two of the fixes above initially grew extra
   returned fields and extra logged scalars beyond the defect. Both were reverted. On the eve of a
   multi-day run, a new metric name is a change to the run's output contract; the defect was that a
   number was *wrong*, not that a number was *missing*.

### 11.7 The §5.1 fix was wrong on its first pass — and the proof that cleared it was too weak

Worth recording in full, because the failure is exactly the class of mistake this document exists
to catch.

**The bad fix.** The first version rebuilt the pre-aug snapshot as

```python
_orig_images = extract_slices_from_phases(batch["phases"].float(), ...).detach()
```

`extract_slices_from_phases` returns `(B,S,518,518,3)` in **[0,255]**. The `batch["images"]`
contract — and what the consumer `_log_augmentation_to_wandb._gray` is written against, and what
`aug_images` still is — is `(B,S,3,518,518)` in **[0,1]**. `gpu_aug.py` performs the conversion at
**all three** of its own assignment sites (`:302`, `:382`, `:395`); the fix omitted it.

Nothing crashes. `_gray` does `.clamp(0,1).mean(dim=1)`, so on a `(S,H,W,C)` tensor it clamps
[0,255] to ~all-1.0 and then averages over **H** instead of the channel axis. Fed both formats
verbatim:

```
aug_images / pre-refactor _orig_images -> (4, 518, 518)  range 0.004..0.996   correct
first-pass  _orig_images               -> (4, 518,   3)  range 1.000..1.000   100% saturated
```

The "original" row of the panel becomes a 518×3 near-white sliver.

**Why the §11 proof missed it.** The check was "does `Train_Visuals_Augmentation` appear in the
offline wandb transaction log?" It does — the bad code logs a panel happily. **The check fired and
verified *presence*, not *correctness*.** It was a valid regression test for the original defect
(the panel was absent) and completely blind to the defect the fix introduced.

**The corrected fix** matches the gpu_aug contract exactly:

```python
_orig_images = extract_slices_from_phases(
    batch["phases"].float(), batch["timesteps"], batch["slice_indices"]
).permute(0, 1, 4, 2, 3).contiguous().div(255.0).detach()
```

**Re-verified on the rendered artifact, not on the log index.** A real offline run's logged PNG was
opened and split into its two rows: original row **mean 0.560, std 0.436**, augmented row
**mean 0.548, std 0.453** — both real anatomy, neither saturated. (A saturated row would read
std ≈ 0, frac>0.99 ≈ 100%.) Panel saved at `result/aug_panel_fixed.png`.

**Lesson 8, and the sharpest one in this document:** *for a fix to a diagnostic, "it now emits
something" is not a proof — verify the emitted artifact's CONTENT.* A visualization defect can be
fully repaired at the plumbing level and still render garbage, and the plumbing-level check will
pass. This is the same shape as §10's lesson 5 (a test can encode the bug): the check was written
from the same mental model as the fix, so it could only confirm what the fix already assumed.

### 11.8 A post-fix sweep found one more launch-blocker: `max_img_per_gpu=12` in 8 sbatch scripts

Re-running §5.5's repo-wide config sweep *after* the fixes turned up a break of the same class that
the original sweep missed, because it searched for stale **config names** and this is a stale
**override key**.

`max_img_per_gpu` was deleted in docs/59 F9. Hydra's `key=value` syntax means *override an existing
key* and refuses to create one — deliberately, so a typo (`max_epoch=100`) cannot silently become a
new no-op key instead of setting `max_epochs`. Adding a key requires the `+` prefix. Confirmed:

```
ConfigCompositionException: Could not override 'max_img_per_gpu'.
To append to your config use +max_img_per_gpu=12
```

Eight scripts still passed it as a live override — the six `sbatch/oneframe_*.sh` (the docs/46
ablation series) and `train_mri_volume_diffusion_ft_{control,gather}.sh`. All eight died at
`compose()` in `launch.py`.

Two mitigating facts, both checked rather than assumed:

- **`sbatch/train_pooled1337_nativez.sh` — the long-run script — was never affected.** Its only
  occurrence is a comment stating the key is deliberately absent.
- The failure is at compose time, *before* the model loads, so an affected job dies in seconds with
  an actionable message rather than burning a GPU allocation or failing mid-run.

**Pre-existing, not refactor fallout** — the key died in docs/59, one commit range earlier. It cost
only the reproducibility of the already-run docs/46 ablations.

**Fixed** by stripping the override from all eight (and from the two `echo` lines that printed its
value, which would otherwise have described the run falsely). Verified by composing **every**
`ABLATION_OVERRIDES` / `EXTRA_OVERRIDES` / `RECIPE_OVERRIDES` string in every `sbatch/*.sh` against
that script's own `CONFIG=`: **55/55 override sets compose**, and no live `max_img_per_gpu=` remains
anywhere in `sbatch/` or `tools/`.

**Lesson 9:** *a config-rename sweep must grep for dead override KEYS, not just dead config NAMES.*
Both produce the same symptom — an instant crash at launch — but only the second is found by
searching for the thing you renamed. The generalization of §10's lesson 6: enumerate what the
change made *invalid*, not just what it made *moved*.
