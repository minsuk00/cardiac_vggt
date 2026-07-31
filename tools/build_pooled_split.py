"""Assign the pooled train/val/test split (docs/58 sec 2.1, TODO item 6) on top of the
manifest built by tools/build_manifest.py, then emit the .txt split file MRIDataset reads.

Per-source rule (decided with the user 2026-07-31):
  ACDC   — official `testing/` kept intact (published benchmark, docs/46 reports on it);
           val = 3 subjects per pathology group carved from `training/`; rest -> train.
  MNMs   — official split as-shipped: Training/{Labeled,Unlabeled} -> train,
           Validation -> val, Testing -> test. (Canon appears ONLY in Validation/Testing —
           respecting the official split buys a free unseen-vendor test.)
  CMRx2023/2024 — random 8:1:1, independently per year (single-vendor, single/near-single-
           centre, healthy cohorts — nothing to stratify by). 2024 additionally soft-
           preserves membership from the deprecated training/splits/random_8_1_1.txt where
           those subjects still exist on disk, so continuity isn't gratuitously broken.
  CMRx2025 — the 12 Philips subjects are PINNED out of train entirely (split across
           val/test) because Philips is otherwise absent from train/val entirely (see
           docs/58 discussion) — mirrors the M&Ms Canon treatment. Everyone else
           (Siemens/UIH) is stratified by (vendor, pathology_label) into whatever count is
           needed to make the WHOLE POOL's grand total land exactly on 7:1:2 (70/10/20) —
           ACDC/M&Ms/2023/2024 are all fixed by their own rules above, so 2025 is the one
           source that absorbs the residual (round_targets(1343,[.7,.1,.2]) minus what those
           four already contribute, minus Philips's forced val/test-only split).

Writes the `split` column back into training/splits/manifest.csv and generates
training/splits/pooled.txt in the [train]/[val]/[test] format MRIDataset._find_subjects
parses (data_root=scratch/data, one rel_path per line).
"""

import argparse
import csv
import os
import random
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "training", "splits", "manifest.csv")
LEGACY_2024_SPLIT = os.path.join(REPO, "training", "splits", "random_8_1_1.txt")
SEED = 42


def read_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def read_legacy_2024_membership():
    """{'val': {id, ...}, 'test': {id, ...}} from the deprecated 2024-only split file."""
    out = {"val": set(), "test": set()}
    section = None
    with open(LEGACY_2024_SPLIT) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
            elif line and not line.startswith("#") and section in ("val", "test"):
                out[section].add(line)
    return out


def round_targets(n, fracs):
    """Split n into len(fracs) integer counts summing exactly to n (largest-remainder).
    Ties (e.g. equal val/test fracs) break toward the LATER index — i.e. test over val —
    so a tied split lands as val<=test, not val>=test (user preference: test should be at
    least as large as val, not the other way around)."""
    raw = [n * f for f in fracs]
    base = [int(x) for x in raw]
    remainder = n - sum(base)
    order = sorted(range(len(fracs)), key=lambda i: (raw[i] - base[i], i), reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base  # [train_n, val_n, test_n]


def stratified_assign(ids_by_stratum, train_n, val_n, test_n, rng):
    """Round-robin across strata (each pre-shuffled) -> a well-mixed flat order, then a
    straight positional cut into [0:train_n), [train_n:train_n+val_n), [rest). Gives each
    split an approximately proportional share of every stratum without exact-remainder
    bookkeeping per stratum."""
    for ids in ids_by_stratum.values():
        rng.shuffle(ids)
    queues = list(ids_by_stratum.values())
    flat = []
    while any(queues):
        for q in queues:
            if q:
                flat.append(q.pop())
    assert len(flat) == train_n + val_n + test_n, (len(flat), train_n, val_n, test_n)
    return {"train": flat[:train_n], "val": flat[train_n:train_n + val_n], "test": flat[train_n + val_n:]}


def assign_acdc(rows):
    split = {}
    by_group = defaultdict(list)
    for r in rows:
        if r["official_split"] == "testing":
            split[r["id"]] = "test"
        else:
            by_group[r["pathology_detail"]].append(r["id"])
    rng = random.Random(SEED)
    for group, ids in by_group.items():
        rng.shuffle(ids)
        for i in ids[:3]:
            split[i] = "val"
        for i in ids[3:]:
            split[i] = "train"
    return split


def assign_mnms(rows):
    split = {}
    for r in rows:
        os_ = r["official_split"]
        split[r["id"]] = {"Labeled": "train", "Unlabeled": "train",
                           "Validation": "val", "Testing": "test"}[os_]
    return split


def assign_cmrx_random(rows, source_label):
    ids = [r["id"] for r in rows]
    train_n, val_n, test_n = round_targets(len(ids), [0.8, 0.1, 0.1])
    rng = random.Random(SEED)

    preassigned = {}
    if source_label == "CMRxRecon2024":
        legacy = read_legacy_2024_membership()
        id_set = set(ids)
        legacy_val = sorted(legacy["val"] & id_set)
        legacy_test = sorted(legacy["test"] & id_set)
        rng.shuffle(legacy_val)
        rng.shuffle(legacy_test)
        for i in legacy_val[:val_n]:
            preassigned[i] = "val"
        for i in legacy_test[:test_n]:
            preassigned[i] = "test"

    remaining = [i for i in ids if i not in preassigned]
    rng.shuffle(remaining)
    rem_val_n = val_n - sum(1 for v in preassigned.values() if v == "val")
    rem_test_n = test_n - sum(1 for v in preassigned.values() if v == "test")
    rem_train_n = len(remaining) - rem_val_n - rem_test_n

    split = dict(preassigned)
    split.update({i: "train" for i in remaining[:rem_train_n]})
    split.update({i: "val" for i in remaining[rem_train_n:rem_train_n + rem_val_n]})
    split.update({i: "test" for i in remaining[rem_train_n + rem_val_n:]})
    return split


def assign_cmrx2025(rows, train_n, val_n, test_n):
    """Philips is pinned out of train entirely (split across val/test); everyone else
    (Siemens/UIH) is stratified by (vendor, pathology_label) into the exact train_n/val_n/
    test_n handed in by the caller — sized (in main()) as the RESIDUAL needed to make the
    whole pool's grand total exactly 7:1:2 (70:10:20), given ACDC/M&Ms/2023/2024 are already
    fixed by their own independent rules."""
    philips = [r["id"] for r in rows if r["vendor"] == "Philips"]
    rest = [r for r in rows if r["vendor"] != "Philips"]
    assert train_n + val_n + test_n == len(rest), (train_n, val_n, test_n, len(rest))

    rng = random.Random(SEED)
    rng.shuffle(philips)
    half = len(philips) // 2
    split = {i: "val" for i in philips[:half]}
    split.update({i: "test" for i in philips[half:]})

    by_stratum = defaultdict(list)
    for r in rest:
        by_stratum[(r["vendor"], r["pathology_label"])].append(r["id"])
    assigned = stratified_assign(by_stratum, train_n, val_n, test_n, rng)
    for s, ids in assigned.items():
        for i in ids:
            split[i] = s
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "training", "splits", "pooled.txt"))
    args = ap.parse_args()

    rows = read_manifest()
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    split = {}
    split.update(assign_acdc(by_source["ACDC"]))
    split.update(assign_mnms(by_source["MNMs"]))
    split.update(assign_cmrx_random(by_source["CMRxRecon2023"], "CMRxRecon2023"))
    split.update(assign_cmrx_random(by_source["CMRxRecon2024"], "CMRxRecon2024"))

    # 2025 absorbs the RESIDUAL needed to make the whole pool's grand total exactly 7:1:2
    # (70/10/20), given ACDC/M&Ms/2023/2024 are already fixed by their own independent
    # rules above. Compute the grand-total target first, subtract what's already fixed, and
    # that remainder is 2025's target (Philips's forced val/test-only split is subtracted
    # out of that first, since it's decided independently of the ratio).
    grand_train, grand_val, grand_test = round_targets(len(rows), [0.7, 0.1, 0.2])
    fixed_train = sum(1 for s in split.values() if s == "train")
    fixed_val = sum(1 for s in split.values() if s == "val")
    fixed_test = sum(1 for s in split.values() if s == "test")
    cmrx2025_train = grand_train - fixed_train
    cmrx2025_val = grand_val - fixed_val
    cmrx2025_test = grand_test - fixed_test
    assert cmrx2025_train + cmrx2025_val + cmrx2025_test == len(by_source["CMRxRecon2025"])

    n_philips = sum(1 for r in by_source["CMRxRecon2025"] if r["vendor"] == "Philips")
    philips_val = n_philips // 2
    philips_test = n_philips - philips_val
    split.update(assign_cmrx2025(by_source["CMRxRecon2025"], cmrx2025_train,
                                  cmrx2025_val - philips_val, cmrx2025_test - philips_test))

    assert set(split.keys()) == {r["id"] for r in rows}, "every manifest row must get a split"

    for r in rows:
        r["split"] = split[r["id"]]
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r["rel_path"])
    with open(args.out, "w") as f:
        f.write("# Pooled multi-dataset split (docs/58). Generated by tools/build_pooled_split.py\n")
        f.write(f"# from {os.path.relpath(MANIFEST, REPO)}. data_root = scratch/data.\n\n")
        for section in ("train", "val", "test"):
            f.write(f"[{section}]\n")
            for p in sorted(by_split[section]):
                f.write(p + "\n")
            f.write("\n")

    print(f"wrote split column -> {MANIFEST}")
    print(f"wrote pooled split -> {args.out}")
    print(f"totals: train={len(by_split['train'])} val={len(by_split['val'])} test={len(by_split['test'])}")
    print()
    print("per-source breakdown:")
    for src, srows in by_source.items():
        c = defaultdict(int)
        for r in srows:
            c[split[r["id"]]] += 1
        print(f"  {src}: train={c['train']} val={c['val']} test={c['test']}")
    print()
    print("CMRxRecon2025 vendor coverage per split (Philips must be 0 in train):")
    vs = defaultdict(lambda: defaultdict(int))
    for r in by_source["CMRxRecon2025"]:
        vs[split[r["id"]]][r["vendor"]] += 1
    for s in ("train", "val", "test"):
        print(f"  {s}: {dict(vs[s])}")


if __name__ == "__main__":
    main()
