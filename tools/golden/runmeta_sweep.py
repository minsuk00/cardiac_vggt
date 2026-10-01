"""G7: every recorded run config still builds. For each scratch/logs/*/run_meta.jsonl, check that
load_run's model kwargs bind to VGGT.__init__ and its val dataset kwargs bind to
MRIDataset.__init__. Records "ok" or the error per run, so old vs new outcomes can be diffed.

  python tools/golden/runmeta_sweep.py --out temp/golden/<tag>/g7.json
"""
import argparse
import glob
import inspect
import json
import os

from common import REPO, dump

import inference.load_run as lr
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT

LOGS = os.path.join(REPO, "scratch", "logs")


def outcome(fn):
    try:
        fn()
        return "ok"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()

    vggt_sig = inspect.signature(VGGT.__init__)
    ds_sig = inspect.signature(MRIDataset.__init__)
    res = {}
    for path in sorted(glob.glob(os.path.join(LOGS, "*", "run_meta.jsonl"))):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        cfg = next(r["config"] for r in rows if "config" in r)
        res[os.path.basename(os.path.dirname(path))] = {
            "model": outcome(lambda: vggt_sig.bind(None, **lr.model_kwargs_from_config(cfg))),
            "dataset": outcome(lambda: ds_sig.bind(None, None, **lr.mri_dataset_kwargs(cfg))),
        }
    bad = {k: v for k, v in res.items() if "ok" not in v.values() or set(v.values()) != {"ok"}}
    print(f"{len(res)} runs, {len(bad)} with a non-ok outcome")
    dump(res, a.out)


if __name__ == "__main__":
    main()
