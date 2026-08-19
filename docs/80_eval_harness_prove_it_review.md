# 80 — `evaluation/` + `inference/` adversarial review: what held, what broke, what was refuted

> **TL;DR & takeaway** — A 7-reviewer / 6-verifier adversarial pass over the whole live
> `evaluation/` + `inference/` tree (~3,357 lines), plus GPU runtime checks. **The scientific core
> is correct and was proven by running it**: a full re-reconstruction from the checkpoint reproduces
> every stored metric to **0.000000** on 5 subjects spanning `D ∈ {6,11,11,15,18}` and
> `dz ∈ {5,8.5,10,12}` across 4 sources, and `aggregate.py` reproduces 3 cohort summaries
> byte-identically. **Five real defects were found and fixed**, of which one was a hard crash
> (`--arms clean` → `UnboundLocalError` after writing the recons) and two were already wrong in
> git-tracked results (64 of 136 subjects mislabelled volunteer/patient; 16 bare `NaN` tokens making
> two summaries invalid JSON). **Eight plausible-mechanism claims were refuted by measurement** —
> most importantly the claim that the harness ROI differs from the trainer's, which is false:
> `heart & FOV == heart` for **all 144** subjects, 0 voxels dropped, so docs/79 §7b's reconciliation
> is exact rather than approximate. A follow-up pass then closed the **two reachable latent hazards**
> — the unenforced split (the only silent-wrong-number path) and the stale `recon_clean/` that could
> difference two checkpoints into `cost_psnr` — plus the archived-path defaults in the two classical
> baseline shells and the `README.md` self-contradictions; each fix is fault-injected to prove the
> check fires, and the 7 committed summaries are unchanged but for an added `"split"` key. The
> remaining hazards need an unusual operator action and stay unfixed by choice (§6).

Follows docs/79 (the rebuild). This is the audit of that rebuild.

---

## 1. Method

Seven independent reviewers, each reading the **whole** target with a distinct lens: the `run_vggt`
spine / training-batch contract · geometry & axis order · state, control flow & partial failure ·
determinism & the frozen-bundle promise · metrics & numerics · path layer & analysis scripts ·
minimality & dead code. Every suspected bug then went to a fresh verifier briefed to **refute** it,
and reachability was checked against the real on-disk corpus rather than argued. Runtime checks were
run by the orchestrator only, on an L40S inside an existing allocation (nothing submitted).

The refutation step earned its keep: **8 of the 21 distinct claims died there**, all of them
plausible mechanism stories of exactly the kind docs/79 §7 warns about.

## 2. What was proven correct, by running it

| check | result |
|---|---|
| end-to-end re-reconstruction vs stored `metrics.json`, 5 subjects / 4 sources | **all metrics reproduce to 0.000000**; `resp_diag` slope to full float precision |
| … spanning the D and dz extremes | `D ∈ {6, 11, 11, 15, 18}`, `dz ∈ {5.0, 8.5, 10.0, 12.0}` |
| `aggregate.py` re-run (output redirected to scratch), 3 cohorts | byte-identical to the committed summaries |
| `render_gif` + `slice_panels.py` — **never exercised before** (every prior run used `SKIP_GIF=1`) | run clean on real data |
| `check_paths.py` **fault-injected 8 ways** | all 8 fire; baseline exits 0 |
| `pytest tests/` | 363 passed |

The batch-construction contract also holds under scrutiny: `build_batch` covers every key `get_data`
emits with the right dtype, `gpu_augment_batch` genuinely mutates in place (so `_extract`'s
discarded return is safe), nothing goes stale across the 12-phase sweep, and **none of
`make_dataset`'s `kw.pop`s change sampling** — every val `t_target` path consumes zero RNG draws and
`rng` is constructed after them, so forcing `t_target_fixed=0` leaves the z-shuffle and per-slot `t`
stream bit-identical to training.

## 3. Defects found and fixed

**3a. `--arms clean` crashed after writing the recons** — `run_vggt.py:421`/`:430`. `metadata_draw`
was bound only inside `if breathing:` but read unconditionally. Runtime-reproduced: 12
`recon_clean/vol_t*.nii.gz` + `timing.json` + `resp_diag.json` written, then `UnboundLocalError`
before `metadata.json`. The residue defeats the guard it lands beside — `check_overwrite` treats a
metadata-less arm as "fresh", so a later run with a different checkpoint would not conflict.
Reachable from the shipped driver (`ARMS=clean`). Found independently by 5 of 7 reviewers.
**Fix:** initialise `metadata_draw = {}` with `timing`/`rdiag`.

**3b. The volunteer/patient split was wrong in 3 committed summaries — now DELETED.** `group_of`
keyed on `"patient" in subject.lower()`, a property of the *name*. Measured against
`training/splits/manifest.csv`'s `pathology_label`: **64 of 136 labelled subjects mislabelled** —
all 37 cmrx2025 (diseased, reported as volunteers), 24 of 33 mnms, and 3 healthy ACDC patients
reported as pathology. Deleted rather than repaired: every pooled cohort is single-group under that
rule, so the block only ever duplicated `ALL` — it cost a wrong label and bought nothing. **If a
pathology split is ever wanted, join `manifest.csv:pathology_label`; the ground truth is already in
the repo.** Recorded here because "add the group back" is the obvious wrong instinct.

**3c. Bare `NaN` tokens in git-tracked results** — `aggregate.py`. `results/{ocmr,miitt}` each
carried 16, from `stat()` returning `nan` for a cohort with no `clean` arm. Python reads them back;
**every strict JSON parser rejects the file**, and these are the citable cohort numbers. Fixed with
a `json_safe()` NaN/Inf → `null` pass plus `allow_nan=False`, so a future NaN fails loudly instead
of silently emitting an invalid token.

**3d. `compare_methods.py` crashed where the scorer degrades** — it loaded `mask_heart.nii.gz`
unconditionally (and even without `--mask`), while the heart ROI is optional by design
(`pooled.py` warns-and-skips it; `assemble_and_gif.py:233-234` falls back to the FOV).
Runtime-confirmed `FileNotFoundError`. Latent — 0 of 144 current subjects lack one. Fixed to mirror
the scorer's fallback.

**3e. SSIM's dynamic range was derived from the data** — `assemble_and_gif.py:155`,
`L = max(a.max(), b.max()) - min(a.min(), b.min())`. `L` sets `c1=(0.01L)²`, `c2=(0.03L)²`, which
exist only to stabilise the denominator and are meant to be a fixed property of the data format. A
data-derived `L` makes them method- and subject-dependent: at this harness's real ROI size
(28k–67k voxels) a single bright outlier widens `L`, inflates the stabilisers and **raises** SSIM
while PSNR falls. Measured +0.019 SSIM for a change costing 0.6 dB PSNR; after pinning `L = 1.0`
the same outlier lowers SSIM (−0.0013), matching PSNR's direction. Correct constant because
everything is normalised to [0,1] by construction (`preprocess.py` for GT, `prep_recon` for the
self-normalising baselines).

All **144** subjects were re-scored under the fixed `L`. Verified against a pre-re-score snapshot:
`psnr`, `psnr_unit_peak` and `ncc` are **bit-identical (max |Δ| = 0.0000000000)** across every
subject, both arms, and every one of the 12 phases — **only SSIM moved**, by up to **0.0537**
per phase / **0.0508** per subject-mean, and +0.0008…+0.0077 at cohort level. (An earlier
18-subject sample put the ceiling at 0.022; over the full cohort it is 0.051 — the smaller figure
was undersampled, not wrong.) The measured cross-method spread remains 0.0004, so no ranking
changes. The re-score also cleared the §5 stale `metrics.json`.

**3f. Two regressions the `NaN → null` change caused, found by running the consumers.**
`compare_table.py` raised on `None.__format__` and `compare_bars.py` raised inside matplotlib
(`int + NoneType`) — both had been silently limping on `nan`, and `compare_table`'s `—` absent-value
sentinel had been *unreachable* because `stat()` always returned a 2-list. Both fixed to treat
`[null, null]` as absent. **Worth noting as method:** these were invisible to review and only
appeared because the changed code was actually executed.

## 4. Claims that MEASUREMENT REFUTED

Same convention as docs/79 §7 — recorded because the plausible version is what a reader will
otherwise re-derive.

**4a. "The harness ROI is `heart ∩ FOV` while the trainer's is `heart`, so docs/79 §7b's
reconciliation is only approximate."** **False.** Measured across **all 144** subjects:
`(heart & FOV).sum() == heart.sum()`, **0 voxels dropped, 0 subjects affected**. The heart ROI is
already inside the FOV, so the intersection is a no-op and the two suites score the identical voxel
set. docs/79 §7b is exact.

**4b. "`img_per_seq=ds.num_slices` reads a different knob than the trainer's `img_nums`."**
Refuted. Under `one_frame_per_slice` `img_per_seq` only *arms* the D>budget guard — the slot count
is forced to D either way. All 43 runs with a `run_meta.jsonl` have `num_slices == img_nums[0] == 20`
and max eval-bundle D is 18, so neither claimed consequence can fire. The same pattern is used at
all three `trainer_viz.py` call sites — it is the repo's convention, not a deviation.

**4c. "The bundle-vs-live `dz` mismatch silently mis-scales the volume."** Refuted, and the proposed
fix would have *created* the bug. `dz` cancels exactly in the splat (`pz = z_val·z_scale + (D-1)/2`
with `z_val ∝ dz` and `z_scale ∝ 1/dz`) — measured bit-identical identity splat at dz=8 vs dz=12.
The recon and GT bundle affines are both stamped from the frozen `man["dz_mm"]`, so `load_canon`
never resamples; stamping the *live* `dz` instead would make the affines disagree and trigger a
~20% z-compression. Residual exposure is confined to `resp_diag`'s mm conversion.

**4d. "No `reference_slot` guard, so a legacy checkpoint yields a meaningless sweep."** Refuted.
All 43 loadable runs are `true` on both the dataset and model halves; the 104 older runs hard-fail
on the missing `run_meta.jsonl`, whose introducing commit is the *same* one that deleted
`reference_slot: false` from the shipped config. A guard would be dead code that blocks a legitimate
ablation — and the sweep would honestly penalise a phase-blind model, not flatter it.

**4e–g. The three breathing-diagnostic claims** — pooled slope conflating within/between-subject
variance (Simpson's), the missing `[-3,3]` clamp and 3-slot minimum, and `corr` being NaN-dropped
where the trainer returns 0.0. All three mechanisms are real in synthetic form; **none can fire on
this data.** Swept over 1,266 `resp_diag.json` (144 current + 1,122 archived): pooled slope 0.9109
vs subject-centered 0.9117 on the current arm, max divergence 0.055 across all 58 groups ever
produced — because **within-subject applied variance is 3.7–17× the between-subject variance**, so
the pooled fit *is* a within-subject fit. Min `n_slots` = 5, max |slope| = 1.25, min applied-std
1.37 mm against a 1e-6 gate, and 0 of 1,266 subjects have a NaN `corr`.

**4h. "An all-zero GT ROI gives contradictory finite metrics."** The numbers reproduce exactly
(psnr −20.0, unit-peak +100.0, ssim 1.0, ncc NaN; −107 dB if the recon is non-zero) but the state is
unreachable: **0 hits across 50,928 phase-metric entries**, global min PSNR 10.49 dB.

## 5. One on-disk anomaly, verified harmless

`cmrx2024/CMRx24_Test_P012` carried a `metrics.json` **15.6 min older than its own recons** — scored,
re-reconstructed by a later run, never re-scored — and `aggregate` published it to git with no
warning, because `recon_mtime` is written for exactly this purpose and read by nobody. Isolated
(1 of 144). The number was right: an independent GPU re-reconstruction reproduced the stored values
to 0.000000. Cleared by the §3e re-score.

## 6. Confirmed hazards — the two reachable ones fixed, the rest deliberate

The two below were the only ones an operator could reach without an unusual action. Both were fixed
in a follow-up pass (2026-08-16), each with a fault-injection test proving the check fires:

- **~~The split was enforced in one of four places.~~ FIXED.** `run_vggt.check_bundle_split` honoured
  `manifest["split"]`; `aggregate.py` did not, and neither the bundle dir nor
  `results/<ds>/<arm>.json` is split-keyed, so a `test`-split bundle would have been averaged into
  the val cohort silently — **the only silent-wrong-number path in the harness**. The rule now lives
  once, in `paths.filter_by_split`, and both callers use it; `aggregate.py` filters on `$SPLIT`
  (default `val`), reports what it excluded, split-filters `expected` so an off-split bundle no
  longer reads as a *missing* val subject, and stamps `"split"` into the summary. Verified: the 7
  committed summaries are unchanged except for the added `"split": "val"` key (n=19/29/37/15/33/3/8);
  `SPLIT=test` excludes all 29 cmrx2024 subjects and refuses to write rather than averaging; a
  synthetic tree confirms a `test` bundle and an unreadable manifest are both dropped while a
  legacy no-`split` manifest is kept.
- **~~A stale `recon_clean/` was re-scored as current.~~ FIXED.** `run_vggt` writes only the variants
  in `--arms` and never invalidates the others, while `assemble_and_gif` discovers by `.is_dir()`, so
  a re-run with the driver's default `--arms breath` left an older `recon_clean/` to be differenced
  into `cost_psnr` — undetectable, because `metadata.json` is per ARM, not per variant. Now each
  `recon_<variant>/` gets its own `stamp.json` (ckpt + fingerprint + commit), written after its
  phases so a crashed variant stays unstamped; `assemble_and_gif.check_variant_stamps` raises when
  present variants disagree and records `stamps_agree` in `metrics.json`. Fault-injected 6 ways:
  matching→True, differing→raise, stamped-vs-unstamped→raise, fully-legacy→warn+False (so the 144
  pre-stamp subjects still re-score), single-variant→True, `ALLOW_MIXED_ARMS=1`→warn instead of
  raise.

The rest are real and traced, but each needs an unusual operator action nobody has taken across 144
subjects. Listed so the next reader does not re-derive them.

- **`pooled.py --overwrite` + a caught per-subject exception** leaves an old manifest paired with
  half-new pixels, `"skipped"` forever. Scanned all 144: none affected.
- **`run_vggt` has no per-subject `try/except`** (unlike `pooled.py`), so one bad subject aborts the
  sweep under `set -e`. `pooled.py`'s zero-subject `SystemExit` fires *today* if `miitt`/`ocmr` are
  added to `SOURCES` without changing `SPLIT_FILE`.
- **`aggregate`'s `use_fp` degrades open** — one fingerprint-less row makes the whole cohort key by
  realpath, so two different checkpoints at the same `checkpoint_last.pt` path collapse to one key
  and the mixed-checkpoint warning never fires.
- **`ef_dice.py`**: `ef_of` returns 100% EF if any phase segments to zero LV, and Dice returns 0.0
  (not NaN) when the GT mask is empty but the prediction is not. Run exactly once, into the stale
  pre-native-z `_ef_ood/gather05`; 0 of 183 curves triggered either.
- **`slice_panels.py:144`** — the native-z port made `z_cont` an algebraic identity, so continuous-z
  slot depth is rounded to the integer plane (proven by direct execution). Dormant:
  `continuous_z: false`.

## 7. Minimality backlog — measured, not fixed

The "one implementation of the native-z contract" claim **holds for the geometry**. It does not hold
for everything else: `name_seed` is duplicated verbatim in `run_vggt.py` and `pooled.py`,
`INPLANE_MM = 1.4` is re-declared in 3 files while the real source is `preprocess.TARGET_SPACING`
(already imported for `Z_HALF_MM` in both), the arm-slug regex exists in 3 copies and the `_contz`
fallback resolver in 3 more. Roughly 120 lines are verified dead: the 49-line
`plane_note`/`plane_coverage`/`splat_z_weights` chain in `slice_panels.py:158-206` (no callers —
and this branch *maintained* it through the native-z port instead of deleting it), `paths.panel_dvf`
(0 callers), the `common` OmegaConf dict in both builders (`MRIDataset` never reads `common_conf`),
`metrics["regime"]` (confirmed `None` in every committed `metrics.json`), and an unreachable
`if ed_pack is None`.

~~Live stale defaults in `run_svrtk3d.sh:31` / `run_nesvor.sh:35`~~ **FIXED**: `${EVAL_DATASET:-cmrxrecon}`
named a retired source whose directory is archived, so the first classical-baseline run of the new
bundles would have written into a dead path. `EVAL_DATASET` is now required (`:?` with the valid
source list); verified both scripts abort with that message when it is unset.

Still open: `inference/seg_metrics_cmrxrecon.py` is orphaned (its producer is archived; it hardcodes
the retired `VOX_ML = 1.4·1.4·12.0` cube) while `README.md:66` calls `inference/` "model loading only".

~~`evaluation/README.md` contradicts itself~~ **FIXED** (2026-08-16): the `--regime` /
`--continuous-z` / `--refiner` flags do not exist (confirmed against `run_vggt`'s argparse), so the
"Extending" section no longer tells you to pass them; `MODELS.md` / `models.json` are documented as
on-demand `build_models_table.py` output (the tracked pair was archived by `1b8b454`), with the
per-arm `metadata.json` named as the actual source of truth. The README also now documents the split
rule and the per-variant stamp above.

## 8. Open

- `sbatch/eval_pooled_val.sh` has still **never run end-to-end** (`bash -n` only). Its
  source-coverage gap is now closed, though: it defaulted to **5 of the 7 sources** and
  structurally could not reach `miitt`/`ocmr` — one global `SPLIT_FILE`, and `pooled.txt` has 0
  MIITT and 0 OCMR lines, so even adding them to `SOURCES` would have hit `pooled.py`'s
  "no <source> subjects" `SystemExit`. Both results were nevertheless committed, i.e. produced by
  hand. Fixed 2026-08-16: `SOURCES` now lists all seven and `split_file_for()` routes each source
  to its own split file (`pooled.txt` / `pooled_miitt.txt` / `evaluation/splits/ocmr_eval.txt`),
  with `SPLIT_FILE=<path>` still forcing one file for everything. Verified by calling
  `pooled.read_split` per source: **19/29/37/15/33/3/8 = 144 val subjects, exactly the cohort on
  disk**, and the old single-file behaviour still `SystemExit`s for miitt/ocmr. Two adjacent
  hazards fixed in the same pass: `REPO` is now derived from the script's own path (it hardcoded
  `/home/minsukc/vggt`, so a worktree copy silently ran main-tree code — verified it resolves to
  the worktree), and the ~144-invocation assemble loop no longer goes through `micromamba run`,
  whose lockfile deadlocks under exactly that pattern. `$SPLIT` is exported so `aggregate.py`
  enforces the same split the build used.
- The classical baselines (SVRTK / NeSVoR / NiftyMIC) have still not been re-run on the new bundles.
- §7's backlog and §6's hazards are unaddressed by choice, not by oversight.
