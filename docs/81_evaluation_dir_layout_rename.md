# 81 — `evaluation/` layout rename: `src/{engine,analysis}`, `comparison_figures/`, `metric_results/`

> **TL;DR & takeaway**
> Pure layout refactor of `evaluation/` (2026-08-18, on `eval/nativez-transition`, no behavior
> change): `engine/` and `analysis/` moved under `src/`, `figures/` → `comparison_figures/`,
> `results/` → `metric_results/`. Because every read-side path is built in `paths.py`, the data
> renames were a two-constant edit; the code moves needed only the scripts' *self-location*
> lines (one extra parent level) plus the two callers (`sbatch/eval_pooled_val.sh`,
> `tools/render_all_gifs.sh`). Verified by running `check_paths.py` (ALL PASS), import-smoking
> all 10 moved python scripts, and a real `compare_table.py cmrx2024` read off
> `metric_results/`. If you hold a stale path: prepend `src/` for code, and map
> `figures→comparison_figures`, `results→metric_results` for data.

## What changed

| old | new |
|---|---|
| `evaluation/engine/` | `evaluation/src/engine/` |
| `evaluation/analysis/` | `evaluation/src/analysis/` |
| `evaluation/figures` (symlink → `../scratch/eval_figures`) | `evaluation/comparison_figures` |
| `evaluation/results/` (git-tracked cohort summaries) | `evaluation/metric_results/` |

`paths.py`, `check_paths.py`, `build_models_table.py`, `splits/`, `volumes/`, `checkpoints/`,
`_archive/` stay at `evaluation/` root, unchanged. All moves via `git mv` (the `figures` symlink
is gitignored, so plain `mv`), so the branch's uncommitted edits ride along as renames and
history follows.

## The edits that made it work

**Data renames — one edit, by design.** `paths.py` is the single source of truth
(`RESULTS = EVAL_ROOT / "metric_results"`, `FIGURES = EVAL_ROOT / "comparison_figures"`);
every reader/writer of those trees (`aggregate.summary`, `compare_dir`, `cohort_fig_dir`,
`compare_table`, `compare_bars`, `breathing_pred_vs_applied`, `ef_dice`) resolves through it,
so nothing else needed a functional change. This is exactly the layout-change-is-one-edit
guarantee `paths.py`'s docstring promises, and it held.

**Code moves — only self-location lines.** Each moved script finds `evaluation/` (for
`import paths`) or the repo root relative to `__file__`; going one level deeper meant:

- `parents[1]` → `parents[2]`: `src/engine/{aggregate,assemble_and_gif}.py`,
  `src/analysis/{ef_dice,breathing_pred_vs_applied,slice_panels}.py`
- one extra `os.path.dirname(...)`: `src/engine/run_vggt.py` (`ROOT`),
  `src/engine/build_inputs/pooled.py` (`ROOT`)
- `EVAL / "engine"` → `EVAL / "src" / "engine"`: `src/analysis/compare_methods.py`;
  `paths.EVAL_ROOT / "engine"` likewise in `src/analysis/slice_panels.py`
- **needed NO change:** `assemble_and_gif.py`'s panel hook `parents[1] / "analysis"` —
  `parents[1]` is now `src/`, and `analysis/` moved with it, so the sibling relation is
  preserved; also the root-walking `next(p for p in HERE.parents if (p / "evaluation").is_dir())`
  in `compare_table.py` / `compare_bars.py` / `compare_methods.py` (depth-independent by
  construction — prefer that pattern for new scripts).

**Callers:** `sbatch/eval_pooled_val.sh` (4 invocations + the final `echo`),
`tools/render_all_gifs.sh`. The baseline shells (`run_svrtk3d.sh`, `run_nesvor.sh`,
`run_seg.sh`) reference nothing inside `evaluation/` at runtime — they write straight to
`$VGGT/scratch/eval/...` — so only their usage comments changed.

**Housekeeping:** `evaluation/.gitignore` (`/comparison_figures`, `/src/analysis/out/`),
`evaluation/README.md` (layout tree + every command), usage docstrings in all moved scripts,
and the live `CLAUDE.md`/`AGENTS.md` pointers. Historical `docs/` (79, 80, README index
entries, …) deliberately left with the old paths — they record what was true when written.

## Verification (run, not inspected)

- `python evaluation/check_paths.py` → ALL PASS (resolves the real GPFS tree across all 7
  sources, against raw globs).
- Import-smoke of all 10 moved python scripts from the worktree: each executes its module
  top-level, `paths.RESULTS.name == 'metric_results'`, `paths.FIGURES.name ==
  'comparison_figures'`, and `run_vggt.py`/`pooled.py` report `ROOT=/home/minsukc/vggt-evalfix`
  (this worktree, not the main tree).
- Real end-to-end read: `compare_table.py cmrx2024` prints the committed
  `vggt_augaggr224hw2_ep300` cohort row (29/29, breath 20.67±1.46) from `metric_results/`.

## Gotchas / open

- **Main tree untouched.** The refactor was briefly started in `/home/minsukc/vggt` by
  mistake, fully reverted (git status clean), and redone here. The main tree still has the old
  layout; this rename lands on `main` when `eval/nativez-transition` merges.
- Pre-existing, not touched: `run_svrtk3d.sh` / `run_nesvor.sh` / `run_seg.sh` hardcode
  `VGGT=/home/minsukc/vggt` (the MAIN tree) for their GPFS/env paths — the same class of bug
  `eval_pooled_val.sh` and `check_paths.py` already fixed for themselves. Fine while the data
  contract is stable, but a worktree that changes those scripts' *inputs* won't be exercised.
- `metric_results/` file *contents* are unchanged (`RM` = rename + the branch's own pre-existing
  modifications); the summaries remain the citable numbers.
