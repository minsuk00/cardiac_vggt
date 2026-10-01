"""Compare two golden JSONs (G1 sample_trace, G2 run_train, G3 ckpt_forward). Exit 1 on mismatch.

  python tools/golden/compare.py OLD NEW [--removed KEY ...] [--tol 0]

--removed: metric names / batch keys allowed to vanish in NEW. A removed metric must have been
exactly 0 everywhere in OLD (e.g. train/loss/pos_tv); a removed batch key is just dropped.
Wall-clock fields (`t`, and metric names containing time/sec/_ms/mem/throughput) are ignored.
"""
import argparse
import json
import re
import sys

NOISE = re.compile(r"(time|_sec|sec_|_ms|mem|throughput|it_s)", re.I)


def metric_map(rows):
    m = {(r.get("epoch"), r["step"], r["name"]): r["value"] for r in rows if not NOISE.search(r["name"])}
    if len(m) != sum(not NOISE.search(r["name"]) for r in rows):
        sys.exit("duplicate (epoch, step, name) rows: the run dir was reused")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--removed", nargs="*", default=[])
    p.add_argument("--tol", type=float, default=0.0)
    a = p.parse_args()
    old, new = json.load(open(a.old)), json.load(open(a.new))
    bad = []

    def strip_removed(d):
        return {k: v for k, v in d.items() if k not in a.removed}

    if "metrics" in old:                                    # G2
        mo, mn = metric_map(old["metrics"]), metric_map(new["metrics"])
        for k in set(mo) - set(mn):
            if not any(k[2].endswith(r) for r in a.removed):
                bad.append(f"metric vanished: {k}")
            elif mo[k] != 0:
                bad.append(f"removed metric was nonzero: {k}={mo[k]}")
        for k in set(mn) - set(mo):
            bad.append(f"new metric: {k}")
        diffs = [(k, mo[k], mn[k]) for k in set(mo) & set(mn)
                 if not (mo[k] == mn[k] or (a.tol and abs(mo[k] - mn[k]) <= a.tol))]
        bad += [f"metric {k}: {x!r} -> {y!r}" for k, x, y in sorted(diffs, key=str)[:20]]
        if len(diffs) > 20:
            bad.append(f"... {len(diffs)} metric diffs total")
        if old["model"] != new["model"]:
            n = sum(old["model"].get(k) != v for k, v in new["model"].items())
            bad.append(f"model weights differ in {n} tensors (keys equal: {old['model'].keys() == new['model'].keys()})")
        if old["optimizer"] != new["optimizer"]:
            bad.append("optimizer param_groups differ")
        print(f"G2: {len(set(mo) & set(mn))} metrics compared, warnings {old['n_warnings']} -> {new['n_warnings']}")
    elif "train" in old:                                    # G1
        for split in ("train", "val"):
            for i, (x, y) in enumerate(zip(old[split], new[split])):
                if strip_removed(x) != strip_removed(y):
                    diff = [k for k in set(x) | set(y) if k not in a.removed and x.get(k) != y.get(k)]
                    bad.append(f"{split}[{i}] differs in {diff}")
            if len(old[split]) != len(new[split]):
                bad.append(f"{split}: {len(old[split])} vs {len(new[split])} batches")
        print(f"G1: {len(old['train'])} train + {len(old['val'])} val batches compared")
    else:                                                   # G3
        if old != new:
            for arm in old:
                if old[arm] != new.get(arm):
                    bad.append(f"{arm}: {old[arm]} -> {new.get(arm)}")
        print(f"G3: {len(old)} checkpoints compared")

    for b in bad:
        print("  MISMATCH", b)
    print("PASS" if not bad else f"FAIL ({len(bad)})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
