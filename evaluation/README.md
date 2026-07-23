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
├── paths.py            # the ONE source of truth for every path + arm-name convention
├── check_paths.py      # read-only self-check: paths.py resolves the real tree
├── MODELS.md           # provenance: one row per arm -> ckpt / config / scheme / wandb
├── engine/             # the frozen-bundle harness (run_vggt, run_svrtk3d, run_nesvor,
│                       #   assemble_and_gif, aggregate, per-dataset build_inputs)
├── diagnostics/        # the standing every-eval diagnostics (breathing, slice panels, EF/Dice)
├── results/<ds>/<arm>.json   # small cohort summaries (git-tracked, citable)
│
├── volumes/     -> GPFS (subject-major data; gitignored)
│   └── <dataset>/out/<subject>/
│       ├── manifest.json  gt/  clean/  breath/  mask*  heart_seg*   # shared frozen bundle
│       └── <arm>/ recon_clean/ recon_breath/ metrics.json timing.json …   # one dir per method
└── checkpoints/ -> GPFS (COPIED ckpts per arm; gitignored)
```

Datasets: `cmrxrecon` (paired, has GT) + OOD transfer `acdc` / `miitt` / `ocmr`. Phase
count `T` is **per-dataset** (cmrx 12, ocmr 18, miitt/acdc 30) — always read `T` from
`manifest.json`, never hardcode 12. `_ef_ood/` is a separate derived product with its own
layout, not part of this harness.

## Scope

This dir is **only** the gated + breathing-simulated pipeline (the frozen-bundle harness).
Real-time free-breathing (RTFB) inference is **out of scope for now** and stays in
`inference/run_rtfb.py`.

## What lives here vs elsewhere (the curation rule)

- **`evaluation/` holds only scripts run on *every* eval** — the core harness (build_inputs,
  run_*, assemble_and_gif, aggregate) and the standing diagnostics. Everything here must be
  **simple and 100% correct**; it is not a scratchpad.
- **One-off / report-specific / exploratory scripts stay in `tools/`.** Do not migrate a
  script into `evaluation/` unless it is re-run on every eval.
- **`diagnostics/` is human-curated.** Do not add a script here on your own initiative —
  write it to `tools/` and ask.
- **Relationship to sibling dirs:** `inference/` = the shared library (model loading +
  dataset adapters; the harness imports it). `baselines/` = classical-method
  implementations (SVRTK / NeSVoR / NiftyMIC internals). `evaluation/` = the frozen-bundle
  harness that drives both against one shared input and scores them.

## Data rules

- **Volumes are symlinked, never moved or copied.** `evaluation/volumes -> ../scratch/eval`
  points at the existing GPFS tree in place. `volumes/` and `checkpoints/` are gitignored so
  no GPFS binary or absolute path ever enters git; only code + small `results/` +
  provenance are tracked.
- **The breathing bundle is frozen and shared.** `gt/ clean/ breath/ manifest.json` +
  masks are byte-identical inputs for every arm — that is the fairness guarantee. Never
  regenerate the bundle under a subject without re-running every arm on it.
- **Checkpoints are COPIES** (`checkpoints/<arm>/checkpoint.pt`), not symlinks — the
  original may be deleted; a copy guarantees the arm can be reproduced. Recorded in
  `MODELS.md`.
- **Never delete regenerable outputs to force a rebuild.** Write to a new path and swap
  after verifying.

## Naming rule (slug in name, scheme in registry)

- An arm dir name is a short **identity slug** (`vggt_gather05`, `svrtk3d`, `nesvor`).
  Input scheme, epoch, date, wandb id, ckpt path, and config all live as **columns in
  `MODELS.md`**, not in the folder name. Adding a new baseline or a differently-configured
  VGGT run (fixed-z snapped, continuous physical-z, more frames, …) = one new arm dir + one
  registry row — no naming rules to overload.
- `paths.canonical_arm(model_name, continuous_z=...)` is the single place a VGGT arm name is
  built (guards the historical `_contz` doubling). Build names only through it.
- **Do not rename the existing dated `vggt_<date>_...` dirs** (they are referenced by frozen
  summaries); new runs adopt the slug scheme and the old ones age out.

## Using paths.py

```python
import sys; sys.path.insert(0, "<repo>/evaluation"); import paths
for arm in paths.arms("cmrxrecon"):              # arm-style iteration over subject-major disk
    for subj in paths.subjects("cmrxrecon"):
        vol = paths.recon("cmrxrecon", subj, arm, "clean", phase=0)
```

Run `python evaluation/check_paths.py` after any layout change — it asserts every resolver
matches a raw glob of the real tree, across all four datasets.

## Running the harness

Pipeline per dataset: build the frozen bundle once → reconstruct each method → score →
aggregate → diagnostics. Everything reads/writes through `paths.py`.

```bash
# 1. build the frozen breathing bundle (once per dataset)
python evaluation/engine/build_inputs/<dataset>.py ...
# 2. reconstruct — VGGT [GPU], or a classical baseline
python evaluation/engine/run_vggt.py --dataset <ds> --ckpt <pt> --model-name <slug>
EVAL_DATASET=<ds> bash evaluation/engine/run_svrtk3d.sh <subj> <arm>
# 3. score per subject -> <subj>/<arm>/metrics.json (+ gifs)
EVAL_DATASET=<ds> python evaluation/engine/assemble_and_gif.py <subj> <arm>
# 4. cohort summary -> results/<ds>/<arm>.json  (git-tracked, citable)
python evaluation/engine/aggregate.py <ds> <arm>
```

Standing diagnostics (write to the gitignored `diagnostics/out/`):

```bash
python evaluation/diagnostics/breathing_pred_vs_applied.py --dataset <ds> --arm <arm>
python evaluation/diagnostics/slice_panels.py --cohort <ds> --method <arm> --arm breath
python evaluation/diagnostics/ef_dice.py dump <dir> --method <arm> --cohorts <ds...>
#   then nnUNet_predict (nnunet env) -> ef_dice.py score <seg_dir> --input <dir> --out <json>
```

Cohort numbers live in git at `results/<dataset>/<arm>.json`; per-arm provenance in `MODELS.md`.

## Why subject-major (the one divergence from MRI2CT)

MRI2CT is arm-major because every entity there is one immutable file per subject. VGGT is
not: each `(method, subject)` cell is a *directory* (`recon_clean`/`recon_breath` × T
phases + `metrics.json`/`timing.json`, which **span both clean and breath in one file**),
and every method consumes a shared *generated* breathing bundle. Subject-major keeps each
subject's bundle welded to the recons derived from it and matches the data's shape at zero
migration cost; arm-style iteration is recovered in `paths.py`. Model identity is a short
slug in the dir name; scheme/epoch/date/ckpt live in `MODELS.md`.
