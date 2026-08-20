"""THE scoring entry point: score one method across datasets, then aggregate.

Thin driver over image_metrics.py + aggregate.py (in-process, not subprocess).

Usage:
  micromamba run -n svr python evaluation/src/score/run.py --method vggt_augaggr224hw2_ep300 \
      [--datasets cmrx2024 acdc ...] [--split val]
"""
import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths            # noqa: E402
import image_metrics    # noqa: E402
import aggregate        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(paths.DATASETS))
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    failures, n_scored = [], 0
    for ds in args.datasets:
        subjects, dropped = paths.filter_by_split(ds, paths.subjects(ds), args.split)
        # Only subjects this method actually reconstructed (breath is the required arm).
        todo = [s for s in subjects if paths.recon_dir(ds, s, args.method, "breath").is_dir()]
        print(f"\n### {ds}: {len(todo)}/{len(subjects)} split-'{args.split}' subjects have a "
              f"{args.method} breath recon ({len(dropped)} dropped by split)")
        ds_failed = []
        for s in todo:
            try:
                image_metrics.score_subject(ds, s, args.method)
                n_scored += 1
            except Exception as e:
                ds_failed.append(s)
                failures.append((ds, s, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        if todo:
            # exclude this run's failed subjects: a surviving metrics.json from an EARLIER run
            # must not be silently averaged in as if it were fresh (stale-row leak).
            try:
                aggregate.aggregate(ds, args.method, args.split, exclude=ds_failed)
            except SystemExit as e:   # aggregate sys.exit()s when nothing scored
                failures.append((ds, "<aggregate>", str(e)))
            except Exception as e:    # one corrupt metrics.json must not abort the other datasets
                failures.append((ds, "<aggregate>", f"{type(e).__name__}: {e}"))
                traceback.print_exc()

    if failures:
        print(f"\n!! {len(failures)} FAILURE(S):")
        for ds, s, why in failures:
            print(f"   {ds}/{s}: {why}")
        sys.exit(1)
    if n_scored == 0:
        sys.exit(f"no subjects scored for method '{args.method}' in {args.datasets} — "
                 f"wrong --method name, or no recon_breath dirs?")
    print(f"\nall done, {n_scored} subjects scored, no failures")


if __name__ == "__main__":
    main()
