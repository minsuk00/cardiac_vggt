# evaluation/

Git-tracked home for the **gated + breathing-simulated** reconstruction benchmark: the
frozen-bundle head-to-head harness (VGGT vs classical SVR baselines), its provenance, and
small citable results. Heavy data (recon volumes, checkpoints) lives on GPFS and is
symlinked in.

Modeled on the MRI2CT `evaluation/` pattern — a real dir in git, only the large binaries
symlinked out — with **one deliberate divergence**: the data under `volumes/` is
**subject-major**, not arm-major (see "Why subject-major").

## Layout

```
evaluation/
├── paths.py            # source of truth for paths + arm-name conventions on the READ side
├── check_paths.py      # read-only self-check: paths.py resolves the real tree
├── build_models_table.py  # harvest metadata.json -> models.json (git-tracked registry)
├── src/
│   ├── engine/         # makes VOLUMES (run_vggt, run_svrtk3d, run_nesvor, run_seg,
│   │                   #   build_inputs/pooled.py; run_vggt_rt = real MIITT real-time
│   │                   #   free-breathing recon, qualitative only — docs/87)
│   ├── score/          # makes NUMBERS: run.py (THE entry point), image_metrics.py (PSNR/SSIM/NCC),
│   │                   #   ef_dice.py (EF/Dice via nnU-Net), aggregate.py (folds image + breathing
│   │                   #   resp_diag + timing [+ EF/Dice] into ONE metric_results/<ds>/<arm>.json)
│   └── analysis/       # makes FIGURES (viz = GT-vs-pred GIF, breathing, slice panels,
│                       #   compare_methods = multi-arm GIF, compare_table / compare_bars = cross-arm ranking)
├── splits/          # split files for sources never trained on (e.g. ocmr_eval.txt)
├── metric_results/<ds>/<arm>.json   # small cohort summaries (git-tracked, citable)
│   ├── _ef/<arm>.json               # EF/Dice chain output (cross-cohort, merged into the above)
│   └── _archive/                    # pre-restructure summaries (read-only record)
├── _archive/        # superseded code + pre-native-z results, incl. the old MODELS.md / models.json
│
├── volumes/     -> GPFS (subject-major PRECIOUS data; gitignored)
│   └── <dataset>/out/<subject>/
│       ├── manifest.json  gt/  clean/  breath/  mask*  heart_seg*   # shared frozen bundle
│       ├── cine_gt.nii.gz                                           # shared 4D GT (written once)
│       └── <arm>/ recon_clean/ recon_breath/ metrics.json timing.json ed_dvf.npz
│                  cine_{clean,breath}.nii.gz                        # scored 4D cines (image_metrics.py)
│                  gif_{clean,breath}.gif                            # per-arm renders (viz.py) live WITH the recons
│                  panel_input.gif  panel_dvf.png  panel_lookup.png  #   (VGGT arms; slice_panels, on demand)
│                  _old_scorer/                                      # pre-2026-08-20 assemble_and_gif record
├── comparison_figures/ -> GPFS (DISPOSABLE cross-arm/cohort figures; gitignored, `rm -rf`-safe)
│   └── <dataset>/  [<subject>/_compare/compare_*.gif]  [<arm>_breathing.{json,png}]  [ef_*.png]
└── checkpoints/ -> GPFS (COPIED ckpts per arm; gitignored)
```

Sources: one dir per **pooled source** — `cmrx2023` `cmrx2024` `cmrx2025` `acdc` `mnms`
`miitt` `ocmr`. There is no in-distribution/OOD split any more: ACDC and M&Ms are in the
training pool, and every source here is gated + breathing-simulated, so they differ by
provenance rather than regime.

**Geometry is native-z (docs/58).** Each subject keeps its own slice count `D` and pitch
`dz`; there is no 12-plane cube and nothing is snapped to 12 mm. Read `D`, `dz_mm` and `T`
from `manifest.json` — never hardcode any of them. Every source is converted to 12 phases
on disk, so `T == 12` today, but the rule stands.

`_ef_ood/` is a separate derived product with its own layout, not part of this harness, and
is **stale** (it derives from the pre-native-z bundles archived in
`scratch/eval/_archive_prenativez_20260712/`).

## Scope

This dir is **only** the gated + breathing-simulated pipeline (the frozen-bundle harness).
Real-time free-breathing (RTFB) inference is **out of scope** and is archived in
`inference/_archive/`.

## What lives here vs elsewhere (the curation rule)

- **`evaluation/` holds only scripts run on *every* eval** — the core harness (build_inputs,
  run_*, score/) and the standing analysis. Everything here must be
  **simple and 100% correct**; it is not a scratchpad.
- **One-off / report-specific / exploratory scripts stay in `tools/`.** Do not migrate a
  script into `evaluation/` unless it is re-run on every eval.
- **`src/analysis/` is human-curated.** Do not add a script here on your own initiative —
  write it to `tools/` and ask.
- **Relationship to sibling dirs:** `inference/` = model loading only
  (`load_run.load_model_from_run`, which reads the protocol from the checkpoint's own
  `run_meta.jsonl`). Its dataset adapters are **gone** — every source now goes through
  `MRIDataset.get_data`, so the geometry contract has ONE implementation, the one training
  uses. `baselines/` = classical-method implementations (SVRTK / NeSVoR / NiftyMIC
  internals). `evaluation/` = the frozen-bundle harness that drives both against one shared
  input and scores them.

## Data rules

- **Volumes are symlinked, never moved or copied.** `evaluation/volumes -> ../scratch/eval`
  points at the existing GPFS tree in place. `volumes/` and `checkpoints/` are gitignored so
  no GPFS binary or absolute path ever enters git; only code + small `metric_results/` +
  provenance are tracked.
- **The breathing bundle is frozen and shared.** `gt/ clean/ breath/ manifest.json` +
  masks are byte-identical inputs for every arm — that is the fairness guarantee. Never
  regenerate the bundle under a subject without re-running every arm on it.
- **A cohort is defined by its split, and the tree does not enforce it.** Neither the bundle
  dir nor `metric_results/<ds>/<arm>.json` is split-keyed, so a `test`-split bundle built into the
  same `out/` would otherwise be reconstructed, scored and averaged into the val numbers with
  no warning. Every consumer that defines a cohort honours `manifest["split"]` via
  `paths.filter_by_split`: `run_vggt` skips off-split bundles, `aggregate.py` excludes them
  from the summary (`SPLIT=val` by default; the chosen split is stamped into the summary).
- **Each `recon_<variant>/` carries its own `stamp.json`** (ckpt + fingerprint + commit).
  `run_vggt` only rewrites the variants named in `--arms`, so re-running an arm with the
  default `--arms breath` leaves the previous run's `recon_clean/` in place — and
  `cost_psnr = clean − breath` would then subtract two different checkpoints. The scorer
  refuses to score variants whose stamps disagree (`ALLOW_MIXED_ARMS=1` to override); dirs
  written before stamping warn instead, and record `stamps_agree: false` in `metrics.json`.
- **Checkpoints are COPIES** (`checkpoints/<arm>/checkpoint.pt`), not symlinks — the
  original may be deleted; a copy guarantees the arm can be reproduced. The per-arm ckpt /
  config / scheme / wandb id is recorded in each arm's own `metadata.json`.
- **Never delete regenerable outputs to force a rebuild.** Write to a new path and swap
  after verifying.

## Naming rule (slug in name, scheme in registry)

- An arm dir name is a short **identity slug** (`vggt_gather05`, `svrtk3d`, `nesvor`).
  Input scheme, epoch, date, wandb id, ckpt path, and config all live in the arm's own
  `metadata.json` (written by `run_vggt.py`), not in the folder name. Adding a new baseline
  or a differently-configured VGGT run = one new arm dir — no naming rules to overload.
- **Provenance registry.** `metadata.json` is the source of truth. `build_models_table.py`
  harvests every arm into `evaluation/models.json`, which **is git-tracked** — re-run the
  script and commit the diff whenever an arm is added or re-scored. (`MODELS.md` is no
  longer produced; the pre-native-z pair was archived to `_archive/` by `1b8b454`.)
- `paths.canonical_arm(model_name, continuous_z=...)` is the single place a VGGT arm name is
  built (guards the historical `_contz` doubling). Build names only through it.
- **Do not rename the existing dated `vggt_<date>_...` dirs** (they are referenced by frozen
  summaries); new runs adopt the slug scheme and the old ones age out.

## Using paths.py

```python
import sys; sys.path.insert(0, "<repo>/evaluation"); import paths
for arm in paths.arms("cmrx2024"):               # arm-style iteration over subject-major disk
    for subj in paths.subjects("cmrx2024"):
        vol = paths.recon("cmrx2024", subj, arm, "clean", phase=0)
```

Run `python evaluation/check_paths.py` after any layout change — it asserts every resolver
matches a raw glob of the real tree, across every source.

## Running the harness

Pipeline per dataset: build the frozen bundle once → reconstruct each method → score →
aggregate → analysis. The **read/scoring** side (`run_vggt`, `src/score/*`, analysis)
resolves every path through `paths.py`; the bundle **builders**
and the classical-baseline shells write to the same location via their own `OUT_ROOT`/`SD`
(verbatim snapshots — see "What lives here").

```bash
# 0. everything at once (build -> recon -> score -> gifs -> aggregate) — ALL SEVEN sources by default,
#    each routed to its own split file (pooled.txt / pooled_miitt.txt / evaluation/splits/ocmr_eval.txt).
#    Narrow with SOURCES="cmrx2024 ocmr"; force one split file for every source with SPLIT_FILE=<path>.
sbatch sbatch/eval_pooled_val.sh
# 1. build the frozen breathing bundle — ONE builder for every source; idempotent and
#    incremental (a subject with a manifest.json is skipped unless --overwrite)
python evaluation/src/engine/build_inputs/pooled.py --source <src> \
       --split-file training/splits/pooled.txt --split val [--subjects A,B]
# 2. reconstruct — VGGT [GPU], or a classical baseline. The model protocol (img_size,
#    backbone, sampling knobs) comes from the ckpt's OWN run_meta.jsonl, so there are no
#    --regime / --continuous-z / --refiner flags to get wrong.
python evaluation/src/engine/run_vggt.py --dataset <src> --ckpt <pt> --model-name <slug>
#    baseline shells take (subject, variant); the arm/method is the METHOD env var, ONE call per variant:
EVAL_DATASET=<ds> METHOD=svrtk3d bash evaluation/src/engine/run_svrtk3d.sh <subj> clean
EVAL_DATASET=<ds> METHOD=svrtk3d bash evaluation/src/engine/run_svrtk3d.sh <subj> breath
# 3+4. score every subject + cohort summary in one go (THE entry point):
python evaluation/src/score/run.py --method <arm> [--datasets <ds...>] [--split val]
#    or per subject / per dataset:
EVAL_DATASET=<ds> python evaluation/src/score/image_metrics.py <subj> <arm>   # -> <arm>/metrics.json + cine_*
SPLIT=val python evaluation/src/score/aggregate.py <ds> <arm>                 # -> metric_results/<ds>/<arm>.json
#    GIFs are decoupled from scoring:
EVAL_DATASET=<ds> python evaluation/src/analysis/viz.py <subj> <arm>          # reads the cine_* files
```

Standing analysis. **Per-arm** panels (`slice_panels` → `panel_input.gif` / `panel_dvf.png` /
`panel_lookup.png`) write INTO the arm dir (`volumes/<ds>/out/<subj>/<arm>/`, beside the gifs) and
render-on-demand (scoring no longer auto-renders them). **Cross-arm / cohort** figures write to the
gitignored `comparison_figures/` tree on GPFS (`compare_methods` → `comparison_figures/<ds>/<subj>/_compare/`;
breathing + EF → `comparison_figures/<ds>/`). A metric-only sweep (score/run.py without viz) persists
just `metrics.json` + `metric_results/*.json`.

```bash
python evaluation/src/analysis/breathing_pred_vs_applied.py --dataset <ds> --arm <arm>
python evaluation/src/analysis/slice_panels.py --cohort <ds> --method <arm> --arm breath
# EF/EDV/ESV/LVM (+RV) + Dice/HD95 — reads the SCORED cine_* files (run score/ first):
python evaluation/src/score/ef_dice.py dump <dir> --method <arm> --cohorts <ds...>
bash   evaluation/src/engine/run_seg.sh   <dir> <seg_dir>      # nnU-Net Task114 2d (nnunet env, wrapped)
python evaluation/src/score/ef_dice.py score <seg_dir> --input <dir>   # -> metric_results/_ef/<arm>.json
python evaluation/src/score/ef_dice.py plot  metric_results/_ef/<arm>.json --out <ef.png>
# then re-run score/aggregate.py (or run.py): it folds the _ef file into metric_results/<ds>/<arm>.json
```

Cross-method comparison (any mix of arms — classical baselines + vggt — one subject / cohort):

```bash
# multi-arm cardiac-cycle GIF: GT row + one recon row per arm, same subject (auto-picked if omitted)
python evaluation/src/analysis/compare_methods.py --cohort <ds> --subject <s> --arms svrtk3d nesvor vggt_<slug> --variant breath
# rank every arm of a dataset by a metric, straight from metric_results/<ds>/*.json
python evaluation/src/analysis/compare_table.py <ds> --metric breath_psnr [--arms svrtk3d nesvor vggt_<slug> ...]
```

Cohort numbers live in git at `metric_results/<dataset>/<arm>.json`; per-arm provenance in each arm's
`metadata.json` (tabulated on demand by `build_models_table.py`).

## Extending

- **New VGGT arm (any input scheme)** — one command: `run_vggt.py --dataset <src> --ckpt <pt>
  --model-name <slug>`. There is nothing else to configure: the input scheme comes from the
  ckpt's own `run_meta.jsonl`, so there are no `--regime` / `--continuous-z` /
  `--frames-per-slice` flags (a differently-configured run = a different checkpoint).
  `canonical_arm` builds the dir name; scheme/epoch/date land in the arm's `metadata.json`.
  *Caveat:* `slice_panels.py` only reproduces `regime='onef'` slot ordering and raises on a
  multiframe dir — a multiframe arm needs new diagnostic code, not just a run.
- **New baseline method** — write `src/engine/run_<method>.sh` (mirror the `(subject, variant)` +
  `METHOD=` shell contract), score/aggregate are arm-name-agnostic. If its output isn't on the
  GT `[0,1]` scale, add it to `SELF_NORM_METHODS` / `PURE_SCALE_METHODS` in `score/image_metrics.py`.
- **New dataset/cohort** — no longer the sore spot. It used to need a prep function, a builder
  and an adapter per dataset; the work is now **upstream of this dir**: convert the source to the
  standard 12-phase layout (`tools/convert_*_to_12phase.py` → `<SRC>_sax/`, mirroring
  `ACDC_sax`/`MNMs_sax`) so `MRIDataset` can read it. Inside `evaluation/` only **two** entries
  change: `paths.DATASETS` and `build_inputs/pooled.py:SOURCE_PREFIX`. Everything downstream
  (`run_vggt`, `score/*`, `slice_panels`, `viz`) is source-agnostic and
  reads geometry per subject from `manifest.json`.
  A source that is never trained on also needs a split file — put it in **`evaluation/splits/`**
  (e.g. `ocmr_eval.txt`), not `training/splits/`, so it cannot be pulled into a training pool by
  accident.

**contz naming (historical):** existing OOD contz arms are stored *doubled*
(`vggt_..._contz_contz`) because an old `run_vggt` appended `_contz` twice. `canonical_arm`
fixes this for **new** runs (single `_contz`); `slice_panels.method_dir` / `rep_subject` still
probe both suffixes for the legacy dirs. `ef_dice.method_dir` deliberately does NOT — it
requires the exact literal arm name, because a probed suffix would make the
`metric_results/_ef/<arm>.json` key diverge from the arm name `aggregate.py` joins on and the
EF block would silently never fold. Dump a contz arm under its full dir name.

## Why subject-major (the one divergence from MRI2CT)

MRI2CT is arm-major because every entity there is one immutable file per subject. VGGT is
not: each `(method, subject)` cell is a *directory* (`recon_clean`/`recon_breath` × T
phases + `metrics.json`/`timing.json`, which **span both clean and breath in one file**),
and every method consumes a shared *generated* breathing bundle. Subject-major keeps each
subject's bundle welded to the recons derived from it and matches the data's shape at zero
migration cost; arm-style iteration is recovered in `paths.py`. Model identity is a short
slug in the dir name; scheme/epoch/date/ckpt live in the arm's `metadata.json`.

## Adding val subjects later (the incremental guarantee)

Append them to the split file and re-run the build for that source. **Nothing already on
disk is invalidated**, because both random draws are keyed on the subject NAME, never on
its position in the split:

| | seeded by |
|---|---|
| breathing realization | `sha256("<source>/<subject>")` → `build_inputs/pooled.py` |
| input slot draw (which z, which t per slot) | the same hash → `run_vggt.py`, via a one-subject `MRIDataset` |

The slot draw needs that one-subject dataset because `MRIDataset.get_data` uses `seq_index`
for BOTH the subject index (`seq_index % len(subjects)`) and the val RNG seed
(`random.Random(seq_index)`). With a single subject the index term is always 0, which frees
`seq_index` to be the name hash.

Verified, not assumed: inserting a subject at the head of `[val]` — the worst case, which
shifts every `seq_index` — leaves an existing subject's seed and every NIfTI byte identical.
Existing bundles are skipped (`manifest.json` present), so a re-run only does the new work.

## Reconciling harness numbers with the trainer's

The trainer already writes the full metric suite per val subject to
`<log_dir>/val_per_subject.csv`. The harness is **not** there to re-derive those; it exists
so the classical baselines and the model see a byte-identical corruption, and to emit recon
volumes for EF / Dice / GIFs.

When you do want to compare the two, use **`psnr_unit_peak`**, not `psnr`. They differ by
exactly `20*log10(gt[roi].max())`:

- harness `psnr` normalizes by the GT's max **inside the ROI** — the cross-method convention
  here, so SVRTK / NeSVoR / NiftyMIC / VGGT stay mutually comparable.
- trainer `metric_psnr_3d_*` uses **peak = 1.0**.

Measured on `CMRx24_Test_P012` (heart ROI, both the same `heart_roi_canonical`):
`gt[roi].max() = 0.353` → a **−9.04 dB** offset. Harness `psnr` 20.59, harness
`psnr_unit_peak` 29.62, trainer `metric_psnr_3d_heartseg` **29.49**. The residual is the
different breathing realization (name hash vs `seq_index`), not the metric.

Same check on the geometry, over the anatomy bbox: harness 30.78 dB vs the trainer's 30.39.
Fault-injected to prove the comparison is sensitive rather than accidental — forcing a wrong
`z_scale` collapses it to 21.93 / 20.18 dB.
