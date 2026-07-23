#!/usr/bin/env python
"""compare_table.py — rank all arms of a dataset by a metric, from the git-tracked cohort summaries.

Reads results/<dataset>/<arm>.json (written by engine/aggregate.py) and prints an arm x metric table
(mean +/- std), sorted by --metric. No recompute — pure results/ read. Flags any arm whose cohort is
incomplete (missing subjects, from aggregate's n_expected/missing fields).

Run:
  python tools/compare_table.py cmrxrecon --metric breath_psnr
  python tools/compare_table.py cmrxrecon --arms svrtk3d nesvor vggt_20260713_gather05 --out analysis/out/tbl.md
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "evaluation").is_dir())   # repo root (works from tools/ or evaluation/analysis/)
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(EVAL))
import paths  # noqa: E402

COLS = ["clean_psnr", "breath_psnr", "cost_psnr", "clean_ssim", "breath_ssim"]


def cell(v):
    """summary['all'][col] is [mean, std] (or absent)."""
    return f"{v[0]:6.2f}±{v[1]:.2f}" if isinstance(v, (list, tuple)) and len(v) == 2 else f"{'—':>11}"


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

    rows.sort(key=lambda r: (r.get(a.metric) or [float("-inf")])[0], reverse=True)

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
