#!/usr/bin/env python
"""compare_table.py — rank all arms of a dataset by a metric, from the git-tracked cohort summaries.

Reads metric_results/<dataset>/<arm>.json (written by src/score/aggregate.py) and prints an arm x
metric table (mean +/- std), sorted by --metric. No recompute — pure metric_results/ read. Flags any
arm whose cohort is incomplete (missing subjects, from aggregate's n_expected/missing fields).

Run:
  python evaluation/src/analysis/compare_table.py cmrx2024 --metric breath_psnr
  python evaluation/src/analysis/compare_table.py cmrx2024 --arms svrtk3d nesvor vggt_augaggr224hw2_ep300 --out comparison_figures/cmrx2024/tbl.md
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "evaluation").is_dir())   # repo root (works from tools/ or evaluation/src/analysis/)
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(EVAL))
import paths  # noqa: E402

COLS = ["clean_psnr", "breath_psnr", "cost_psnr", "clean_ssim", "breath_ssim", "clean_ncc", "breath_ncc"]


def cell(v):
    """summary['all'][col] is [mean, std], absent, or [null, null].

    The null form is a breath-only cohort: aggregate.py has no `clean` arm to average, and now
    encodes that as JSON null rather than the bare `NaN` token it used to emit. Both mean "no
    value", so both take the `—` sentinel — without the None check this renders `nan±nan` (before)
    or raises on `None.__format__` (after)."""
    if not (isinstance(v, (list, tuple)) and len(v) == 2) or v[0] is None or v[1] is None:
        return f"{'—':>11}"
    return f"{v[0]:6.2f}±{v[1]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(paths.DATASETS))
    ap.add_argument("--metric", default="breath_psnr", choices=COLS)
    ap.add_argument("--arms", nargs="*", default=None, help="default: every arm with a results json")
    ap.add_argument("--out", default=None, help="also write the table to this path (markdown/plain)")
    a = ap.parse_args()

    rdir = paths.RESULTS / a.dataset
    files = ([rdir / f"{arm}.json" for arm in a.arms] if a.arms else sorted(rdir.glob("*.json")))
    rows = []
    for f in files:
        if not f.is_file():
            print(f"  !! missing results json: {f.name}", file=sys.stderr)
            continue
        d = json.load(open(f))
        allm = d.get("all") or {}
        rows.append({"arm": f.stem, "n": d.get("n"), "n_expected": d.get("n_expected"),
                     "missing": d.get("missing", []), **{c: allm.get(c) for c in COLS}})
    if not rows:
        sys.exit(f"no results found under {rdir}")

    # A breath-only arm stores clean/cost metrics as [null, null] — a truthy list, so a plain
    # `or` fallback would keep the None and crash the sort.
    def sort_key(r):
        v = r.get(a.metric)
        return v[0] if v and v[0] is not None else float("-inf")
    rows.sort(key=sort_key, reverse=True)

    w = max(len(r["arm"]) for r in rows)
    hdr = f"{'arm':<{w}}  {'n':>7}  " + "  ".join(f"{c:>11}" for c in COLS)
    lines = [f"# {a.dataset}  (sorted by {a.metric}, {len(rows)} arms)", "", hdr, "-" * len(hdr)]
    for r in rows:
        n = f"{r['n']}/{r['n_expected']}" if r.get("n_expected") else str(r["n"])
        flag = f"  !! missing {len(r['missing'])}" if r["missing"] else ""
        lines.append(f"{r['arm']:<{w}}  {n:>7}  " + "  ".join(cell(r[c]) for c in COLS) + flag)
    out = "\n".join(lines)
    print(out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(out + "\n")
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
