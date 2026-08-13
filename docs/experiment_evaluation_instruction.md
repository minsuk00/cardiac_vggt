# Experiment evaluation — how to read and compare VGGT-MRI runs

> **TL;DR & takeaway**
>
> **This is the standing instruction for evaluating and comparing runs. Read it before reporting
> any number.**
>
> 1. **Everything you need is on disk** in each run's `log_dir` — `val_per_subject.csv`,
>    `metrics.jsonl`, `run_meta.jsonl`, `baseline_identity.json`. **Do not go to wandb** (only the
>    8 image panels are wandb-only). Load with `tools/load_run.py`.
> 2. **Lead with `metric_psnr_3d_motion` and `metric_recov_frac_heart`.** Never lead with
>    `psnr_3d_full` (inflated by zero-padding) or `psnr_3d_bbox` (dominated by static tissue the
>    model gets for free — it hides whether any motion correction happened at all).
> 3. **`metric_hole_frac_heart` is a VETO, not a score.** A model can fake an MSE win by tearing
>    coverage holes. It must not rise. It going *down* is not automatically good.
> 4. **The ship rule (docs/38):** a change wins iff **recov_frac↑ AND psnr_motion↑ WITHOUT
>    hole_frac↑**.
> 5. **Raw PSNR is NOT comparable across this cohort** — the achievable ceiling moves with `D`,
>    `dz`, FOV and contrast. Normalise by each subject's own identity floor (`baseline_identity.json`)
>    or use the already-normalised `recov_frac_*` / `psnr_seg_gain_db`.
> 6. **Compare runs PAIRED per-subject** (`load_run.compare_runs`), never cohort mean vs cohort
>    mean, and **check the commensurability table first** — different data or val protocol means the
>    numbers are not comparable no matter how good the plot looks.
> 7. **Pair over a plateau, never a single epoch.** The floor and ceiling move epoch to epoch
>    because target phases and breathing are redrawn.
> 8. **Do not compare across the canonical-grid (2026-05-24) or native-z (2026-07-31) refactors.**
>    Different V_gt frame, normalisation and metric definitions. Treat post-native-z runs as a
>    fresh series.

---

## 1. Where the numbers live

Every run mirrors all its scalars to disk (docs/60). No network, no auth, no run-id lookup.

```python
from tools.load_run import load_run, load_identity_baseline
meta, scalars, subjects = load_run("scratch/logs/<exp_dir>")
subjects.groupby("source")["metric_recov_frac_heart"].mean()
# or from the shell:  python tools/load_run.py <log_dir>
```

| File | Contents | Notes |
|---|---|---|
| **`val_per_subject.csv`** | **One row per val sample**, 44 columns: `epoch, step, seq_name, source, t_target, dz_mm, D, S` + every `metric_*` | **The most important file. It exists nowhere else** — B=1, so these are true per-subject values the AverageMeter otherwise discards. Everything in this doc can be recomputed from it. |
| `metrics.jsonl` | Every scalar, with `epoch` **and** wall-clock `t`. Names are `val/metric/*`, `val/psnr/*`, `val/ef/*`, `val/resp/*`, `val/strata/*`, `train/*` — **NOT** the `Val_Loss/*` wandb panel names, see §6.1 | Requeue replays steps ⇒ duplicate `(name, step)` rows are EXPECTED; `load_run` dedupes keep-last. **Join on `epoch`, not `step`** — val scalars and val panels use different step spaces. |
| `run_meta.jsonl` | One line **per process launch**: resolved config, git sha+dirty, split/manifest md5, `data_cache_signature`, cohort sizes, wandb id, SLURM job/node, `resumed_from_epoch` | A requeued run has several lines. Use the last non-`exit` line for provenance. |
| `baseline_identity.json` | Identity-Δ floors: `full` / `bbox` / `motion`, each with `mean_psnr` + `per_phase_mean`, plus **`per_subject`** | Computed at startup, before training. The per-subject entries are what make cross-subject comparison honest. |

Join `seq_name` → `training/splits/manifest.csv` for vendor, pathology, centre, field strength, scanner.

**Only these are wandb-only:** the 8 image panels (filmstrip GIF, ED/ES, DVF, motion mask,
augmentation, lookup). Every *number* is on disk.

---

## 2. The three reference points

Almost every metric here is defined against one of two references. Understand these first or you
will misread everything else.

- **Identity floor (`V_id`)** — splat with `Δ = 0`: every input pixel stays where it already is.
  This is "the model did nothing". A model below this floor is actively harmful.
- **Oracle ceiling (`V_or`)** — the same splat, but sampling the **true target-phase content** at
  each pixel's home position. This is the **recoverable limit given the one-frame-per-slice input
  contract**, not perfection. The model→oracle gap is the *appearance wall* (docs/19–21).
- **Model (`V_canon`)** — what the run actually produced.

All three are logged per subject as `metric_mse_heart_{identity,model,oracle}` (motion ROI) and
`metric_mse_seg_{identity,model,oracle}` (segmentation ROI), so you can recompute any derived
quantity yourself.

### The two "heart" masks — they are NOT the same

| Mask | How it's built | Used by |
|---|---|---|
| **motion mask** | `compute_motion_mask`: `max_t(phases) − min_t(phases) > τ`. No segmentation needed — static tissue barely changes across the cycle, the beating heart does. | `psnr_3d_motion`, `recov_frac_heart`, `hole_frac_heart`, `psnr_3d_static` (its complement within content) |
| **segmentation ROI** | `heart_roi_canonical`, a real anatomical mask shipped with the data | `psnr_3d_heartseg`, `psnr_seg_gain_db`, `recov_frac_seg` |

If `recov_frac_heart` and `recov_frac_seg` **diverge**, that is a signal worth investigating — the
motion mask may be picking up something non-cardiac (respiratory edges, aliasing).

---

## 3. Metric reference

### 3.1 Primary — lead with these

**`metric_psnr_3d_motion`** — PSNR over **dynamic voxels only**.
This is the headline. Cardiac motion correction is the entire task; `bbox`/`full` PSNR is dominated
by static tissue the model reconstructs for free and will look fine even if the model does *zero*
motion correction. Compare against `baseline_identity.json["motion"]["mean_psnr"]`.

**`metric_recov_frac_heart`** — fraction of the recoverable gap closed, on the motion ROI:

```
recov = (MSE_identity − MSE_model) / (MSE_identity − MSE_oracle)
```

- `0` = at the identity floor · `1` = at the oracle ceiling · `< 0` = worse than doing nothing.
- **Unclamped since 2026-08-01** (docs/62 §4). The old `.clamp(-0.5, 1.5)` pinned 98.9% of rows to
  −0.5 early in training. ⇒ **values are not comparable to pre-2026-08-01 curves**, and early-epoch
  values can now be large and negative without anything being wrong.
- Guarded by `span = MSE_id − MSE_or > 1e-6`; samples failing it are omitted, not zeroed.
- The logged value is the **mean of per-subject ratios**, which is *not* the ratio of mean MSEs
  (Jensen). If you recompute from the logged mean MSEs you will get a slightly different number —
  the per-subject mean is the correct one.

Why the third term matters: "+0.85 dB over identity" is meaningless alone. If the ceiling is 1 dB
away that is near-perfect; if it is 7 dB away it is an early-training start. `recov` tells you
which, and **tells you when to stop** — a plateau at 0.4 means you have taken 40% of what this input
contract can give, and the rest is the appearance wall, not a bug to chase.

### 3.2 The veto

**`metric_hole_frac_heart`** — fraction of motion-ROI voxels with splat `coverage < 0.5`.

**This is a tripwire, never an objective.** A model can improve MSE by *tearing coverage holes* —
moving pixels out of hard regions so the coverage-division leaves them near-empty rather than
wrong. That reads as a recov_frac gain while the reconstruction degrades. This exact failure mode
sank the stop-grad-denominator "fix" (~4 dB below floor at 33% holes).

- Rising while recov rises ⇒ **distrust the recov gain**.
- Falling is **not** automatically good and must not be reported as an improvement on its own.

### 3.3 Secondary — cross-subject-safe

**`metric_psnr_seg_gain_db`** = `10·log10(MSE_identity / MSE_model)` on the segmentation ROI.
dB gained over doing nothing, referenced to each subject's own floor ⇒ **averageable across the
heterogeneous cohort**, unlike raw `psnr_3d_heartseg`.

**`metric_recov_frac_seg`** — the recovered fraction on the segmentation ROI. Cross-check for
`recov_frac_heart` (§2).

**`metric_psnr_3d_heartseg`** — raw PSNR on the segmentation ROI. **Do NOT average this across
subjects**: 79% of its between-subject spread is intrinsic scan difficulty, not model quality. Use
it per-subject or use `psnr_seg_gain_db` instead.

### 3.4 Controls — should NOT improve

**`metric_psnr_3d_static`** — content that does *not* beat (the motion mask's complement).
A control. If it drops while motion PSNR rises, the model is robbing static tissue to pay the
heart. Should stay flat.

**`metric_motion_frac`**, **`metric_heartseg_frac`** — ROI sizes. Sanity only. `motion_frac == 0`
marks a degenerate sample that consumers must filter (it cannot occur for real cardiac data).

### 3.5 Research-goal metrics — the actual point of the project

**`val/ef/slope`**, **`val/ef/spearman`**, **`val/ef/mae_pct`**, and the same under
`val/ef/<group>/*`.

`slope = d(pred_EF)/d(true_EF)`. **≈1 = per-patient contraction amplitude recovered; ≈0 = every
patient regressed to the cohort mean EF** (the "flat-EF" failure the reference-slice conditioning
exists to fix, docs/24/25/33).

- **Meaningless early in training.** Prior runs only reached slope 0.77–0.79 at the *end*. Do not
  report a low slope before ~epoch 100 as a finding.
- **Slope is attenuated by range restriction.** Measured val GT-EF σ is **16.2 diseased** vs
  **6.2 healthy**, so a pooled slope drifts with the val health mix — this is why `by_group` exists.
  Always read the per-group slopes, not just the pooled one.
- `_ef_stats` returns `None` when `n < 3` or GT-EF σ < 1.0 pct — a missing group is a *guard*, not
  a failure.

### 3.6 Breathing diagnostics

**`metric_resp_slope_dz`** / **`_corr_dz`** / **`_epe_dz_mm`** / **`_frac_deep_ignored`** —
predicted Δz vs the exact `resp_disp_mm` that was applied. `slope→1`, `epe→0` is good.

Context before you act on these: **breathing is not the bottleneck.** Estimation is already good
(slope 0.844, EPE 1.10 mm, 0/100 real breaths missed) and ~88% of reconstruction error is shared
appearance-synthesis error. A real 2× placement improvement buys ≈ +0.04 dB.

**`frac_slots_offslab`** — fraction of input slices breathing pushed off the slab. ⚠️ **The name
overstates it**: it counts *any* attenuation, not blanked slices. At `D=5, dz=12` it reads 22.85%
when only 3.67% are truly blank.
**`frac_slots_dimmed`** — the *partially* attenuated subset (`dimmed ⊂ offslab`), so
**blanked = offslab − dimmed**. ⚠️ This metric was **100% false positives before 2026-08-01**
(docs/62 §5.3); values from earlier runs are meaningless.

### 3.7 Ignore these

- **`metric_psnr_3d_full`** — includes X/Y zero-padding, which inflates it for small-FOV subjects.
- **`metric_psnr_3d_bbox`** — keep only for continuity with older runs. Static tissue dominates it.
- `metric_ssim_2d_full` — `_full` only, same padding caveat.

---

## 4. How to compare two runs

### 4.1 Check commensurability FIRST

```python
from tools.load_run import compare_runs
wide, commens = compare_runs([dir_a, dir_b], metric="metric_psnr_3d_motion")
print(commens)      # <-- read this BEFORE the numbers
```

`compare_runs` returns the comparison **and** a commensurability table. If the runs used different
data or a different val protocol, **the numbers are not comparable no matter how good the plot
looks.**

Verified output shape (2026-08-01, the two live pooled1337 arms):

```
wide: (266, 2)   indexed on (seq_name, t_target)  <- already paired, one row per val sample
                                          aug     noaug
ACDC_patient006  t0                    20.442    20.405
ACDC_patient015  t0                    17.011    17.360

commensurability columns: run | git_sha | split_md5 | manifest_md5 |
                          cardiac_phase_md5 | data_cache_signature | n_val_subjects
```

**All seven commensurability columns must match** before you compare. Above they do, so those two
arms are comparable.

- `split_md5` alone is **not sufficient** — it hashes subject *names*, so it does not move when the
  voxels change. This repo flipped 893 subjects' arrays in a single day.
- **`data_cache_signature` is what catches that.** Check it.
- Also verify `metric_mse_heart_oracle` is comparable between the runs. The oracle depends on the
  data and the val protocol, not the model — **if two runs disagree on the oracle, they are not
  measuring the same task** and any recov_frac comparison is invalid.

### 4.2 Compare PAIRED, per subject

The same val subjects appear in every run, and per-subject achievable quality varies enormously.
A paired delta cancels the subject effect that otherwise dominates an unpaired difference of means.
Never compare cohort mean to cohort mean.

### 4.3 Pair over a plateau, not one epoch

Each val epoch redraws target phases and breathing, so **the floor and the ceiling move between
epochs** (observed: identity 17.68 → 17.34 dB, oracle 25.28 → 24.38 dB across two consecutive
epochs). A single-epoch delta is noise. Average over a stable window.

### 4.4 Mind the LR schedule when reading early epochs

Warmup is **5% of `max_epochs`** (`lengths: [0.05, 0.95]`). On a 300-epoch run that is **15 epochs**
before peak LR is reached. Gradient norms and losses in epochs 0–15 say nothing about the
configured peak LR.

### 4.5 Hard incomparability boundaries

Do **not** compare across:

- the **canonical-grid refactor (2026-05-24)** — V_gt frame + normalisation changed;
- the **native-z refactor (2026-07-31)** — z is no longer resampled; per-subject `D`/`dz`;
- the **slice-order standardisation (2026-07-31 12:19)** — 893 subjects' arrays were flipped;
  anything derived before this is stale;
- the **`recov_frac_heart` clamp removal (2026-08-01)**;
- `compile_attention_blocks` / fused AdamW on-vs-off — these are ~4e-6 numeric series changes, not
  bit-identical.

---

## 5. Standing rules for reporting

1. **Never assert without evidence you actually gathered.** Run it, read the specific lines, or
   measure it. Label unverified claims out loud.
2. **Measure; do not reason from mechanism.** Plausible-mechanism stories have a bad track record
   here — every one of docs/44's four refuted conclusions was one.
3. **Verify a "fix" by its output, not its plumbing.** A restored diagnostic can emit a panel of
   garbage and pass an "is it logged?" check (docs/62 §11.7). Check the artifact's *content*.
4. **Fault-inject a verifier before trusting it.** A "0 problems" report is worthless until each
   check is proven to FIRE on a broken copy.
5. **Never visualize only the mid slice** — that plane *is* the reference slot, the easiest and
   least informative. Span all z-planes.
6. **A per-source or per-subgroup difference is usually cohort composition, not model quality.**
   Measured NMI on val: source×pitch 0.476, source×pathology 0.470, source×vendor 0.412 — the
   cohort-wide 12-vs-10 mm "pitch" split is a *source* comparison wearing a pitch label.
7. **Expect a plateau below 1.0.** The appearance wall is an information limit of the
   one-frame-per-slice contract, not a bug. An oracle-transport probe bounded the available gain at
   ~1.4 dB over floor, collapsing at ES.

### A worked example (real, epoch 2, aug arm)

```
source   recov_heart   psnr_motion   hole_frac    D     dz
ACDC        0.390         17.16        0.134    10.2   8.5
CMRx23      0.104         19.60        0.094     9.9  12.0
CMRx24      0.146         20.27        0.086    10.2  12.0
CMRx25      0.093         19.01        0.061    12.2  11.3
MNMs        0.180         18.82        0.061    11.9   9.9
```

ACDC has **4× the recovered fraction of CMRx25 but the worst absolute motion PSNR**. Neither number
alone is interpretable — ACDC's floor is low, so there is more headroom to take, not because the
model prefers ACDC. Reading either column in isolation misleads you in *opposite* directions. This
is exactly why §3.1's normalised metric and §4.2's pairing exist. (Also note ACDC carries the
highest `hole_frac` — worth watching whether 0.134 climbs.)

---

## 6. Gotchas in the logging (as of 2026-08-01)

### 6.1 ⚠️ On-disk scalar names are NOT the wandb panel names

**This will cost you time if you grep for the wrong thing.** `Val_Loss/metric_psnr_3d_motion` is a
**wandb panel** name (that is the naming CLAUDE.md and docs/38 quote). In `metrics.jsonl` the same
quantity is `val/metric/...` or `val/psnr/...`. Grepping `metrics.jsonl` for `Val_Loss` returns
**zero rows**, and so does `Loss/train` — which looks like the file is missing val metrics. It is
not. The on-disk families are:

| Prefix | Contains |
|---|---|
| `val/metric/*` | `recov_frac_heart`, `recov_frac_seg`, `hole_frac_heart`, `mse_{heart,seg}_{identity,model,oracle}`, coverage, mae/mse, ssim |
| `val/psnr/*` | `motion/`, `bbox/`, `full/`, `heartseg`, `static`, `metric_psnr_seg_gain_db` — **each with the identity baseline baked into the key name**, e.g. `val/psnr/motion/mean_n266_base18.5`, plus a per-target-phase breakdown `t0_n115_base18.3` … `t11_n13_base18.1` |
| `val/loss/*` | `objective`, `volume`, `diffusion`, `gather`, `pos_tv` |
| `val/ef/*` | `slope`, `spearman`, `mae_pct`, `n` — pooled **and** `val/ef/healthy/*`, `val/ef/diseased/*` |
| `val/resp/*` | `slope_dz`, `corr_dz`, `epe_dz_mm`, `frac_deep_ignored`, `frac_slots_{offslab,dimmed}`, `disp_*` |
| `val/strata/*` | per-source and per-pitch slices, each with its own `_n` |
| `train/*` | the train-side mirror |

The `base<X>` suffix in the `val/psnr/*` keys is the identity floor for that slice — **you do not
have to look it up separately.**

### 6.2 `run_meta.jsonl` field names

Provenance **is** recorded, but not under the names you might guess:

- `git` → `{"sha": ..., "dirty": ...}` — **not** `git_sha`/`git_dirty`
- `gpu` → e.g. `"NVIDIA L40S"` — **not** `gpu_name`
- `n_train_subjects` / `n_val_subjects` — **not** `cohort_sizes`
- also present: `split_md5`, `manifest_md5`, `data_cache_signature`, `cardiac_phase_md5`,
  `slurm_job_id`, `slurm_node`, `torch_version`, `cuda_version`, `wandb_id`, `wandb_url`,
  `resumed_from_epoch`, `steps_at_launch`, `seed`, and the full resolved `config`.

Querying a wrong key returns `None` and looks exactly like a logging bug. Print
`sorted(meta[0].keys())` before concluding anything is missing.

### 6.3 Real remaining gap

No per-subject identity floor exists for the **heart-seg** ROI (structurally blocked twice — the
loss gates heartseg off the identity pass, and the identity batch does not carry
`heart_roi_canonical`), so per-subject heartseg is a within-subject trend only. Use
`psnr_seg_gain_db`, which is already per-subject referenced.

## 7. Provenance of the claims in this document

Be honest about which class a statement falls into before you repeat it.

**Machine-verified against live runs on 2026-08-01** (47/47 checks passed, plus 2 function smoke
tests). Re-run these cheaply if you suspect drift:

- every file in §1 exists; every scalar-name family and every specific key quoted in §6.1;
- every `run_meta.jsonl` key in §6.2; the `baseline_identity.json` structure;
- every `metric_*` name and CSV column referenced anywhere in this doc;
- `recov_frac_heart` recomputed from the raw MSE columns matches the logged column to **2.2e-07**;
- `load_identity_baseline` returns a per-subject DataFrame; `compare_runs` returns a paired
  `(266, n_runs)` frame + the 7-column commensurability table;
- the §5 worked example (per-source table) is real output from the live aug arm at epoch 2;
- warmup = 5% of `max_epochs` (from the resolved config).

**Cited from prior work, NOT re-measured here.** Each is sourced; verify before leaning on one:

| Claim | Source |
|---|---|
| 79% of `psnr_3d_heartseg` spread is scan difficulty | docs/60 |
| stop-grad denominator: ~4 dB below floor at 33% holes | toy-experiment, docs/44-era |
| breathing slope 0.844 / EPE 1.10 mm / 0-of-100 deep breaths missed; ~88% of error is appearance | docs/38, docs/46 |
| oracle-transport upper bound ~1.4 dB, collapses at ES (recov 0.09) | `tools/oracle_transport_probe.py` |
| EF slope 0.77–0.79 reached only at end of training | docs/33 |
| GT-EF σ 16.2 diseased vs 6.2 healthy; NMI confounding 0.476/0.470/0.412 | docs/60 |
| the removed clamp pinned 98.9% of rows to −0.5 | docs/61 |
| `frac_slots_offslab` reads 22.85% where only 3.67% is truly blank | docs/62 §5.3 |
| 893 subjects' arrays flipped 2026-07-31 | docs/58 §10a/b |

## 8. Related reading

`docs/README.md` is the index. Most relevant here: **docs/38** (the ship-decision metrics and the
veto rule), **docs/60** (the on-disk logging + val stratification), **docs/62** (the verification
audit; §5.3 the `frac_slots_dimmed` defect, §11 the resolutions), **docs/24/25/33** (EF and
reference-slice conditioning), **docs/19–21** (the appearance wall), **docs/58** (native-z).
