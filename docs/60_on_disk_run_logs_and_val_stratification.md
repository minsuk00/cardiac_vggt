# 60 — On-disk run logs, and val logging for the heterogeneous pooled cohort

> **TL;DR & takeaway**
>
> **Every run now writes its numbers to disk, so analysing a finished run no longer needs wandb.**
> Three append-only files land in `log_dir` — `run_meta.jsonl` (one line per process launch: git sha,
> config, split/manifest md5, cohort sizes, wandb id, SLURM job), `metrics.jsonl` (every scalar), and
> `val_per_subject.csv` (one row per val sample: subject, source, `D`, `dz`, and every metric). Read
> them with `tools/load_run.py`. wandb is unchanged and still gets everything — this is a mirror, not
> a replacement. **The one thing NOT on disk is the 8 image panels** (filmstrip GIF, ED/ES, DVF, …);
> those stay wandb-only.
>
> Before this, the only structured numeric artifact a finished run left behind was an 822-byte
> `baseline_identity.json`. Everything else was either in wandb or in a 22 MB `log.txt` as prose.
>
> **Alongside it, the val logging was fixed and extended for the pooled 1337-subject cohort.** Two
> visualization bugs mattered: every visual panel was pinned to val indices `(0, 7, 14, 21)`, which
> under the sorted pooled split are **3 ACDC + 1 CMRx2023 — zero M&Ms** (the largest val source, 33
> subjects), zero CMRx2024, zero CMRx2025; and the filmstrip rendered `D//2 ± 2`, which is **provably
> the reference slot** and covers 83%→28% of the stack depending on `D`. Both are fixed. New: a
> per-subject CSV, live per-source and per-pitch val curves, respiratory off-slab tripwires, EF split
> by pathology, and a non-finite guard that stops one NaN batch poisoning a whole epoch's meters.
>
> **Status:** implemented and verified end-to-end (`pytest` 269 green; a real `--config mri_volume`
> run writes all three files). No metric definition changed, so this is additive — curves stay
> comparable to runs from before it.

---

## 1. Why on-disk logs

A completed run directory (checked on a real one, `215949615_…lowdiff100`) contained:

| artifact | size | machine-readable |
|---|---|---|
| `baseline_identity.json` | 822 B | ✅ the only one |
| `log.txt` | 22 MB | ⚠️ prose (`Loss/train_metric_mae_3d_full: 0.0118 (0.0118)`), regex-only, mixes instantaneous and running-average |
| `wandb/` | 875 MB | ⚠️ binary, needs wandb to parse |
| `ckpts/`, `val_volumes/`, `ef_tmp/` | GB | volumes, overwritten each epoch |

So a future agent had three bad options: regex a 22 MB log, hit the wandb API (network + auth +
run-id discovery, and `run.history()` downsamples to 500 points unless you use the slow
`scan_history`), or rerun. This is why the docs/44-era PSNR-drift investigation was painful.

Four properties push toward local files even though wandb is already there:

1. **wandb metric names bake in `n` and the baseline** (`val/psnr/bbox/t0_n3_base23.9`), so **any
   split change mints a new series and orphans the old one** — precisely breaking cross-run
   comparison. A CSV with `n` as a *column* joins fine. (The per-phase panels bake `n` in
   deliberately, as a determinism smoke alarm — see `trainer.py`. The new stratified metrics do
   **not**, and log `n` as its own scalar.)
2. **Exactness.** The JSONL is every logged point; wandb's history fetch is sampled by default.
3. **Survives everything** — offline runs, a deleted project, a rotated account, and a requeue that
   splits one job across several wandb runs.
4. **Per-subject rows exist nowhere else, at any price** — see §3.

## 2. The three files

Written by `training/train_utils/run_log.py` (`RunLog`), wired into `Trainer`.

### `run_meta.jsonl` — one line per **process launch**, plus one on exit

Records carry `"event": "launch"` or `"event": "exit"`. Provenance lives only on launch lines, so
readers must take the last **launch** record, not `meta[-1]`. The exit record carries
`status` (`completed` / `error`), the exception text, `final_epoch` and `steps` — without it a
crashed run and a completed one are indistinguishable from the files, and you have to guess from
whether a `.tmp` checkpoint was left behind. A SIGUSR1 requeue calls `os._exit(0)` and bypasses it;
that segment is identified instead by the *next* launch line carrying `resumed_from_epoch`.

Not per run. A requeued run appends a line per segment, so a mid-run code edit shows up as a changed
`git.sha` between segments instead of being overwritten. Carries git sha + dirty flag, the resolved
config, `split_md5`, `manifest_md5`, `cardiac_phase_md5`, `data_cache_signature`, cohort sizes,
wandb id/url, SLURM job id, node/hostname, GPU model, torch + CUDA versions, `resumed_from_epoch`.

The git sha and the hashes are the load-bearing fields: **a PSNR is meaningless without knowing
which code, which cohort, and which val protocol produced it**, and none of it is recoverable from
wandb (its config snapshot has no git state and no hashes). Three distinct things are hashed
because they fail independently:

- `split_md5` — *which subjects*. Changes when the split file is edited.
- `cardiac_phase_md5` — *the val protocol*. This file decides the 266 `(subject, t_target)` pairs
  every val number is averaged over, so editing it moves every metric **and** the identity floor.
- `data_cache_signature` — *the pixels*. `split_md5` hashes subject NAMES and does not move when
  the voxels change. This repo flipped 893 subjects' arrays and rolled 464 by one slice within a
  week (docs/56, docs/58 §10a); without this, two runs on different data look identical. Reusing
  the monai cache signature is free and is exactly the value that already gates cache invalidation.

### `metrics.jsonl` — every scalar

`{"t": unix_seconds, "step": N, "epoch": E, "name": ..., "value": ...}` per line.

**`epoch` is the field to join on, not `step`.** The trainer logs in two step spaces: val
scalars use `steps["val"]` (via `_update_and_log_scalars`) while the val-epoch panels and the
respiratory scalars use the *train* step. So a naive `pivot_table(index="step")` silently
aligns val-step-N with train-step-N. `t` gives a wall-clock axis, which is how you see where a
run slowed down or died. Mirrored from `Trainer._log_scalar`, which is the
**single chokepoint** for scalars — the other eight `wandb_writer.log` call sites all pass
`wandb.Image`/`wandb.Video`. So a 3-line mirror there guarantees no scalar reaches wandb without
reaching disk, and every metric added in §4 lands on disk for free. Deliberately **not** gated on
`self.wandb_writer`: an offline run must still leave a complete record.

### `val_per_subject.csv` — one row per val sample

`epoch, step, seq_name, source, t_target, dz_mm, D, S, seq_index` + every `metric_*` the loss
computed. Batch size is pinned to 1 (docs/59 F9), so **the value already computed per batch IS a
per-subject measurement** — the `AverageMeter` at `trainer.py` simply averages it away.

### Resume semantics (the part that is easy to get wrong)

`steps` is saved into the checkpoint and restored, and checkpoints are written at **epoch
boundaries** — so on requeue every step between the last checkpoint and the SIGUSR1 kill is
**replayed**, appending a second row with the same `(phase, step, name)` and a *different* value
(different data order and aug draw).

**The writer keeps both**; `tools/load_run.py` dedupes on `(name, step)` keeping the last. Truncating
the file back to the resume point instead would destroy the evidence that a requeue happened.

A kill can also land mid-`write()`. That is why these are JSONL and not one JSON array: a truncated
array is wholly unparseable, a truncated JSONL loses one row. Two consequences handled in code:

- every line is flushed as written, so at most one row is ever lost;
- **the next append repairs a missing trailing newline first** — otherwise it concatenates onto the
  partial line and corrupts *two* rows where the format is meant to cost one. This was caught by
  `tests/test_run_log.py`, not by inspection, and applies to the CSV as well as the JSONL.

**The CSV's columns GROW to fit.** Several metrics are per-*sample* conditional — the heart-ROI
family needs a valid `heart_roi_canonical`, `recov_frac_heart` needs a non-degenerate oracle span —
so the first row is not a reliable schema. An earlier version froze the header on row 0, which meant
that if val subject 0 happened to lack one, **that column was silently dropped for the entire run**,
including the headline heartseg metrics. Now a new key rewrites the file under the union header
(atomically, via `os.replace`); it is rare, since the set stabilises within one val epoch. A resumed
process computing *fewer* metrics leaves the column blank rather than shifting values left.

Non-finite scalars are stored as **JSON `null`**. `json.dumps(float("nan"))` emits a bare `NaN`
token, which Python reads back but `jq`, JS, Go and `pandas.read_json` all reject — one diverged
metric would otherwise make the whole file unparseable by everything except this repo.

A single transient I/O error does not disable logging: only `MAX_CONSECUTIVE_FAILURES` in a row
does, and any success re-arms it. A GPFS hiccup at hour 3 of a 4-day run must not cost the log.

### Reading it

```python
from tools.load_run import load_run, load_identity_baseline
meta, scalars, subjects = load_run("scratch/logs/<exp_dir>")
subjects.groupby("source")["metric_psnr_3d_bbox"].mean()
```

`python tools/load_run.py <log_dir>` prints a summary.

## 3. Log raw, slice later

The alternative — pre-aggregated subgroup means as live wandb panels — was considered and mostly
rejected:

1. **Cost is identical.** With B=1 the per-subject value is already computed; aggregation is a lossy
   projection applied at write time that saves nothing.
2. **The cut is not knowable in advance.** `manifest.csv` carries 23 covariates; source × pitch alone
   is 5 × 12 = 60 cells. Whatever the failure turns out to be, the pre-aggregation chosen in advance
   will be the wrong one.
3. **Per-cell `n` is tiny.** Val has 4 subjects at 5 mm, 1 at 7 mm, 0 at 8 mm. A live
   `val/psnr/pitch_7mm` curve is one subject's trajectory dressed up as a statistic.
4. **Joins are free offline and impossible online.**

Pre-aggregation earns its place only where a curve must be watched *during* the run to justify
killing it. That is exactly the two live strata in §4.

## 4. What changed in the val logging

Cohort facts this is sized against (measured from `manifest.csv` + `pooled.txt`): val n=133 —
**source** CMRx2025 37 / MNMs 33 / CMRx2024 29 / CMRx2023 19 / ACDC 15; **pitch** 12 mm 72, 10 mm 52,
sub-10 mm 9; **pathology** diseased 73 / healthy 60; **vendor** Siemens 83 / Philips 16 / UIH 15 /
Canon 10 / GE 9. `D` spans 5–18 in train, 6–18 in val, 6–21 in test.

### Fixes

| # | Was | Now |
|---|---|---|
| 1 | visual panels pinned to val idx `(0,7,14,21)` = 3 ACDC + 1 CMRx23 | one index per source (`pick_one_index_per_source`) |
| 2 | filmstrip `D//2 ± 2` = the reference plane, 83%→28% of stack | `pick_planes(D, 5)`, evenly spaced, always includes apex 0 and base `D-1` |
| 2b | motion-mask panel: one mid-bbox plane | 3 planes spanning the stack |
| 3 | `baseline_identity.json`: per-phase means only | + `per_subject` floors |
| 4 | NaN check ran *after* logging; val had none | `_log_if_finite` before logging, both phases |
| 5 | one `try` around the whole identity-baseline loop | per-subject `try/continue` |
| 12a | `_save_val_volumes` wrote `np.eye(4)` | `diag(dz, 1.4, 1.4, 1)` |
| 13 | val visual panels matched a **seq_index** against **subject** indices, so under the sweep only the ED half ever rendered | compare the mapped `subj_idx`; ES renders too |
| 14 | panel names were a bare index (`subj0`) | patient id **appended**: `media_val_subj15_ACDC_patient050` |

Note the `media_val_ED_ES/…_ED_ES` panel was **always** correct for both phases (`_stash_ed_es`
maps via `val_targets[i][0]` and takes the role from the blocked layout). Item 13 is a *different*
panel family — `Val_Visuals_subj{…}_Volume` / `_DVF` / `_Lookup`.

**Why #1 and #2 are the same bug in two places.** `MRIDataset.get_data` sets the reference slot to
`z_mid = (bbox_z0 + bbox_z1) // 2`, and under native-z z is never padded so the bbox is always
`[0, D)` — therefore `z_mid == D // 2` **exactly**. The filmstrip window and the motion-mask panel
were both centred on the one plane the model is handed for free. At `D=6` the old window was planes
1–5, so **apex plane 0 — the plane docs/59 F1 was about — was never rendered at all**.

**Why #3 matters most for later analysis.** Achievable PSNR varies enormously with `D`, `dz` and FOV
across this cohort, so a raw per-subject PSNR is **not comparable between subjects**. Dividing by
that subject's own identity floor is what makes any slicing in §3 honest. It also surfaces an
F1-class geometry bug at epoch 0, before a single gradient step.

**#4 is an ordering bug.** `_step` logged, and only then did `_run_steps_on_batch_chunks` check
`math.isfinite` — so a NaN batch's metrics were already in the `AverageMeter` (which then reads NaN
for the *whole epoch*, since NaN poisons the running sum) and already in wandb. Val had no check at
all, and NaN is a value rather than an exception, so every `try/except` passed it through. The new
guard **only suppresses logging** and names the offending `seq_name`; train control flow (backward /
skip) is untouched, so this is a pure observability change.

### Additions

| # | Metric | Rationale |
|---|---|---|
| 6 | `val_per_subject.csv` | §2/§3 |
| 7 | `val/strata/source/<src>/…` | pooled mean hides a one-source collapse: 15 of 133 subjects moving 5 dB shifts it 0.6 dB. All 5 groups clear n≥15 |
| 8 | `val/strata/pitch/{coarse_ge10mm, fine_lt10mm}/…` | the native-z tripwire — if fine sits at its identity floor while coarse improves, kill the run |
| 9 | `resp_offslab_stats` — `frac_slots_{offslab,dimmed}`, `disp_frac_of_extent`, **per subject in the CSV** plus a `{train,val}/resp/*` scalar | see below |
| 10 | `val/ef/<pathology>/{slope, spearman, mae_pct, n}` | see below |

**Built then REMOVED — `cohort/z_norm_max/<src>/{mean,min,max,n}`.** It was justified as
"unreconstructible if the split is re-seeded", but that justification is false given item #3 shipped
in the same change: the per-subject floors carry `D`, `dz_mm` and `source`, written seconds earlier
in the same startup pass, and `max|z_norm| = (D-1)·dz/2/90` reproduces exactly from them (verified to
the last digit). ~35 lines buying zero information. The band it reported is still worth recording
here, since it motivated `Z_HALF_MM=90`: **0.178–0.944 across the cohort, strongly source-correlated**
(ACDC 0.250–0.556, CMRx23 0.467–0.800, CMRx24 0.333–0.733, CMRx25 0.467–0.944, MNMs 0.500–0.722).

**#9.** docs/59 F16 is *accepted, not fixed*: the breathing shift is one-sided (`d ≥ 0`), so basal
slots run off the slab and `padding_mode="zeros"` blanks or dims them. The damaged **fraction** of a
stack is `d / ((D−1)·dz)` — over half of a short 32 mm M&Ms stack versus ~14% of a 132 mm CMRx one.
The simulated corruption is therefore systematically harsher on exactly the short-extent, fine-pitch
subjects native-z was introduced to support. Without this number, a later "fine pitch underperforms"
finding is unattributable — it cannot be separated from "native-z geometry is wrong" — and it is
unrecoverable post-hoc, since the corruption is applied on GPU and never persisted.

Two details worth knowing. It is recorded **per subject in the CSV**, not only as a wandb scalar:
`_log_resp_disp_scalar` runs at `data_iter == 0`, and val is deterministic, so the scalar is always
the *same single subject* — structurally unable to show the fine-pitch subjects it exists for. And
the "dimmed" bounds are **strict**: a slot landing exactly on plane 0 or `D−1` is on an exact plane
and mixes in no padding, so counting it gave every subject a spurious ~2/S floor at zero
displacement.

**#10.** Slope is a regression over the cohort's GT-EF spread, and that spread is dominated by the
diseased half: measured on val, GT-EF σ = **16.2** for diseased (n=73) vs **6.2** for healthy (n=60).
A slope estimated over a 6-point spread is attenuated by range restriction no matter how good the
model is, so a pooled slope drifts with the val health mix and can be misread as a model regression.

### Deliberately not added

**Vendor per-epoch panels.** Worth knowing that **Canon has 0 train subjects** (train: Siemens 689,
UIH 149, Philips 74, GE 23, Canon 0) and 10 in val — but n=10 gives roughly ±0.5–1 dB, so it is a
tripwire, not a curve. Vendor generalization should be claimed on **test**, where Canon is n=40. Same
for `centre` (15 val levels, median n=5, and the naming schemes are not comparable across sources —
`Fudan`/`Dijon`/bare `1`–`5`/`Center001`), `scanner_model` (largest n=19, most n=1–7),
`field_strength` (19 distinct measured floats), `pathology_detail` (32 levels, incompatible
vocabularies), and age/sex (missingness is source-structured, so any split is a source split in
disguise). All of these are one `groupby` away in the CSV.

### Confounding, measured not assumed

Normalized mutual information on the val split: source×vendor **0.412**, source×pitch **0.476**,
source×pathology **0.470**, vendor×pitch **0.328**, n_z×source 0.128.

- **Vendor is not redundant with source** — M&Ms alone spans Canon 10 / GE 9 / Philips 10 / Siemens 4.
  But ACDC, CMRx2023 and CMRx2024 are 100% Siemens, so the "Siemens" bucket is really a source
  mixture; do not read Siemens-vs-rest as a vendor effect.
- **Pitch is heavily source-tied**: all 72 val 12 mm subjects are CMRx. The cohort-wide 72-vs-52 split
  is a *source* comparison wearing a pitch label. The honest within-source contrast is **CMRx2025
  alone, 24 @ 12 mm vs 13 @ 10 mm** — a CSV `groupby`, not a live curve.
- **Pathology is near-determined by source** (CMRx2023 0/19 diseased, CMRx2024 0/29, CMRx2025 37/37).
  Hence its slice is kept narrow — EF only — rather than duplicating the PSNR panels.

### Which metrics the strata slice (`STRATA_METRICS` in `trainer.py`)

Set by the user's stated priority: **heart-seg ROI first, full/bbox PSNR demoted.**

```
metric_psnr_3d_heartseg   headline
metric_mae_3d_heartseg    matches the L1 objective; PSNR is log-MSE and is dominated by the
                          darkest ROI voxels
metric_recov_frac_heart   docs/38 elector
metric_hole_frac_heart    docs/38 VETO — stratifying the electors without it lets a
                          per-source coverage regression pass the ship gate
metric_psnr_3d_bbox       demoted, kept only as continuity with the pre-docs/60 series
```

`n` is emitted **per metric** (`…/{metric}_n`), not once per group. The heart-ROI metrics are
conditional — `loss.py` requires a shape-valid `heart_roi_canonical`, and `recov_frac_heart` is
additionally dropped when `span = mse_identity − mse_oracle ≤ 1e-6` — so a group's metrics can have
different counts. A single shared `n` name was last-write-wins, i.e. the count displayed belonged to
whichever metric sorted last. Watch `n`: a stratum mean over a *varying* subset moves for reasons
unrelated to the model.

## 4b. Comparing runs

`tools/load_run.py::compare_runs([dirs])` returns paired per-subject values plus a
**commensurability table**. Paired beats comparing cohort means — the same 133 subjects appear in
every run and per-subject achievable quality varies enormously, so a paired delta cancels the
subject effect that dominates an unpaired difference. The table diffs `split_md5`, `manifest_md5`,
`cardiac_phase_md5`, `data_cache_signature` and `n_val_subjects` across runs and warns when they
disagree — two runs on different pixels or a different val protocol are not comparable no matter how
clean the plot looks.

Caveat worth knowing: the identity floor is recomputed per run and depends on the respiratory-sim
config, so two runs with different `data.augmentation.respiratory.*` have **different floors** and
their raw PSNRs are not directly commensurable (their `…_base{b}` wandb metric names also differ,
which orphans the curve). Compare floor-normalised values, or compare runs with matching aug config.

## 5. Dismissed failure modes (checked in code, do not re-instrument)

- **Sampler over-representing a source** — `len_train = len(self.subjects)` since docs/59 F6; exactly
  one pass per subject per epoch, so source shares equal subject shares.
- **Loss/gradient scaling with `D`** — `loss_volume` is a mean over `D·256·256` and `S == D` exactly
  under `one_frame_per_slice`, so total gradient magnitude is ~`D`-independent. The per-subject
  weighting that *does* vary is inherent to native-z, not a bug (docs/58 §10 "fresh series").
- **Augmentation at extreme pitch** — all tiers freeze the D axis (`translate_range=(0.0, X, X)`),
  no parameter is in through-plane voxel units, and the respiratory `dz`/`n_planes` threading is
  fault-injection-tested (`tests/test_respiratory_native_z.py`).
- **`torch.compile` churn from variable `D`** — `D` never reaches compiled code (docs/58 §6.4).
- **Splat coverage varying with `D`** — already covered by `metric_coverage_frac`,
  `metric_coverage_mean`, `metric_hole_frac_heart`; now per-subject in the CSV.

## 6. Open issues

### 6a. RESOLVED — per-subject floor for the heart-seg ROI

**Fixed.** The original text below described the gap; the resolution is recorded first.

`metric_psnr_3d_heartseg` is a perfectly good per-subject number — it does measure how well the
model reconstructed that patient's heart. What it cannot do is rank *different* patients, because
scans differ in intrinsic difficulty. Measured on 266 real val rows:

```
corr(ROI size fraction, heartseg PSNR) = -0.126      ← size is NOT the confound
corr(whole-volume PSNR, heartseg PSNR) = +0.791      ← scan difficulty IS
heartseg PSNR spans 5.5 → 23.7 dB
```

(An earlier draft of this doc claimed the confound was ROI *size*. That is **wrong** — MSE is a mean
over voxels, so voxel count cancels. The real term is per-scan difficulty, which the 0.791 shows.)

The fix reuses the identity and oracle splats **already computed** in the docs/38 block and applies
the segmentation ROI to them — no extra forward pass, still val-only:

| metric | meaning |
|---|---|
| `metric_psnr_seg_gain_db` | dB the model gains over doing nothing, on this subject's own ROI. The difficulty term cancels, so this **is** safe to average over a mixed cohort. Headline. |
| `metric_recov_frac_seg` | recovered fraction on the **same** mask as `psnr_heartseg` — `recov_frac_heart` uses the intensity motion mask, a different "heart". **Unclamped**: the `span > 0` guard already keeps the denominator positive, so the −0.5 clamp that censored 98.9% of `recov_frac_heart` rows is unnecessary. |
| `metric_mse_seg_{identity,model,oracle}` | raw terms, so anything else is derivable at read time |

`STRATA_METRICS` now leads with the first two.

### 6a-historic. The gap as originally written

`baseline_identity.json`'s `per_subject` rows carry `psnr_full`, `psnr_bbox`, `psnr_motion` — **not**
`psnr_heartseg`. Under the user's priority that is the wrong omission, and it is structurally
blocked twice over:

1. `loss.py` gates the heartseg block on `pos_pred is not batch["scanner_coords"]`, and the identity
   pass calls exactly that — so the branch is skipped by construction.
2. The identity batch is hand-built in `_compute_identity_baseline` and never contains
   `heart_roi_canonical`, so the `"heart_roi_canonical" in batch` guard fails too.

Why it matters: §4 #3's argument applies *more* strongly to the ROI than to bbox. The ROI is a
small, subject-specific fraction of the volume (`metric_heartseg_frac`), so identity MSE differs
wildly between a subject with a 3%-of-volume ROI and one with 8%, and raw `psnr_heartseg`
differences between subjects are dominated by ROI geometry rather than model quality. **Today
per-subject heartseg numbers are interpretable only as a within-subject trend over epochs.**

Partial mitigation already present: `metric_mse_heart_{identity,oracle}` *are* per-row floor and
ceiling and do land in the CSV — but they use `compute_motion_mask`, **not** the nnU-Net seg. The two
"heart" families are on different masks. Fix is ~15 lines (relax the gate + pass the ROI into the
identity batch + store three more keys); deliberately not done here because it touches `loss.py`.

### 6b. RESOLVED — EF segmentation moved to CorSeg

The nnU-Net path failed `rc=1` on all 5 retries in a verification run (258 of 266 segs produced),
so the EF metric never computed. That hop is `micromamba run -n nnunet` into a separate env for
nnU-Net *v1*, 5-fold + TTA.

Switched to **CorSeg-CineSAX** (`logging.ef_seg_backend: "corseg"`, docs/57), which runs in `svr`
with no installs. docs/57 already recommended exactly this scoping — *"if CorSeg is used for EF,
restrict it to the canonical-cube arms and keep nnU-Net for the SVR baselines"* — and
`save_pred_volume` writes precisely the full canonical cube. CorSeg's known collapse is on
heart-ROI-**cropped** input (Dice 0.889 → 0.413, its fixed 224² canvas ends up 17% full); at full
FOV the canvas is 99.7% full, its in-distribution regime.

Measured here on real canonical volumes:

| | |
|---|---|
| EF MAE (docs/57, ACDC human GT) | **2.51 pp** vs nnU-Net 4.67 |
| LV volume MAE | **4.4 mL** vs 8.9 |
| Peak VRAM | **0.88 GiB** |
| Marginal cost | **0.28 s/volume** → ~76 s for a 266-volume EF epoch |
| Checkpoint load | 44 s from GPFS → **1.1 s** after `stage_checkpoint_to_local` (docs/50) |
| Geometry | preserved — `segment_nifti` propagates the input header |

⚠️ **The two segmenters use different LV-cavity labels**: nnU-Net Task114 `1`, CorSeg `2`
(CorSeg's `1` is myocardium). `_lv_ml` previously hardcoded `== 1`, so a naive swap would have
computed EF from the myocardium — a plausible-looking wrong number. The index now comes from
`ef_eval.segment()` and is threaded into `compute_ef_metrics(lv_label=…)`; never hardcode it.
`"nnunet"` remains available and is still the method-matched operator for the GT labels.

### 6b-historic. nnU-Net EF flakiness

The verification run wrote all 266 pred volumes correctly, then `run_nnunet` failed `rc=1` on all
5 retries (`SimpleITK ... file does not exist` while reading back its own export; 258 of 266 segs
were produced). `retries=5` predates this change and the failure is entirely inside the subprocess,
so `compute_ef_metrics` was never reached. Consequence: **the EF-by-pathology path is covered by
unit tests (`tests/test_ef_group_metrics.py`) rather than end-to-end.**

### 6c. Latent

`_compute_identity_baseline` calls `get_data(img_per_seq=num_slices)` with `num_slices = 20`, and
docs/59 F19 makes `get_data` **raise** when `S = D` exceeds that budget. Max val `D` is 18 today, so
it cannot fire — but the pool holds `D = 19/20/21` (currently all in test), so a re-seeded split
would start skipping those subjects. Item #5 above downgrades that from "the whole baseline is lost"
to "those subjects are skipped, with a warning", which is the right failure mode but still not a fix.
Raising `img_nums` is the actual fix if such a split is ever used.

## 7. Files

- `training/train_utils/run_log.py` — `RunLog` (new)
- `tools/load_run.py` — reader + `compare_runs` (new)
- `tests/test_run_log.py`, `tests/test_val_logging_helpers.py`, `tests/test_ef_group_metrics.py` (new)
- `training/trainer.py` — `_log_scalar` mirror, `_write_run_meta`, `_record_val_subject`,
  `_log_val_strata`, `_log_if_finite`, `_pitch_bucket`, respiratory off-slab
- `training/trainer_viz.py` — `subject_source`, `pick_one_index_per_source`, `pick_planes`,
  `seq_index_to_subject`, `load_subject_groups`, `_log_z_coverage`, + the fixes in §4
- `training/ef_eval.py` — `compute_ef_metrics(..., groups=)` → `by_group`
