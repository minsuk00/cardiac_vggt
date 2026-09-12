"""Re-split the CURATED cohort (training/splits/pooled_curated_v1.txt, 898 subjects) into
train/val/test so that val and test have the SAME composition as train (docs/97).

Why: pooled.txt (and v1, which only filtered it in place) inherited the ACDC / M&Ms official
splits, so test was 60% pathology and held all Canon + most GE while val looked like train —
val did not predict test. Nobody publishes slice-to-volume numbers on those official test
sets, so respecting them buys nothing; only the unseen-vendor holdout is worth keeping.

Rule:
  HOLDOUT  (vendor == Canon, or CMRxRecon2025 Center012 == the only Philips there): NEVER in
           train; split val:test = 1:2 so val also carries an unseen-vendor signal.
  OTHERS   stratified by (source, vendor, centre, pathology_label). Per-stratum quotas are
           proportional to the stratum size, rounded by largest remainder so the section
           totals land EXACTLY on 70/10/20 of the whole 898 (holdouts count toward val/test).
           Strata smaller than MIN_STRATUM_FOR_EVAL go entirely to train.
  CHURN    inside a stratum, a section quota is filled first with subjects already in that
           section in v1 (so most eval bundles / val subjects carry over), then the surplus is
           drawn at random (seed 42). Proportions are unaffected by this tie-break.

Writes training/splits/pooled_curated_v2.txt and a `split_curated_v2` column into
training/splits/manifest.csv (rows not in the curated cohort get "").
"""

import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(REPO, "training", "splits")
MANIFEST = os.path.join(SPLITS, "manifest.csv")
V1 = os.path.join(SPLITS, "pooled_curated_v1.txt")
OUT = os.path.join(SPLITS, "pooled_curated_v2.txt")
SEED = 42
FRACS = (0.70, 0.10, 0.20)
HOLDOUT_VAL_FRAC = 1 / 3          # of each holdout group -> val, rest -> test
MIN_STRATUM_FOR_EVAL = 3          # smaller strata are train-only
SECTIONS = ("train", "val", "test")


def read_split(path):
    sec, out = None, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"\[(\w+)\]$", line)
            if m:
                sec = m.group(1)
                continue
            assert sec in SECTIONS and line not in out, line
            out[line] = sec
    return out


def largest_remainder(raw):
    """raw: list of non-negative floats whose sum is (close to) an integer -> ints, same sum."""
    total = int(round(sum(raw)))
    base = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i], -i), reverse=True)
    for i in order[: total - sum(base)]:
        base[i] += 1
    return base


def is_holdout(r):
    return r["vendor"] == "Canon" or (r["source"] == "CMRxRecon2025" and r["centre"] == "Center012")


def fill(ids, quota, prev, section, rng, taken):
    """Pick `quota` ids from `ids` (not in `taken`), preferring those whose v1 section == section."""
    pool = [i for i in ids if i not in taken]
    keep = [i for i in pool if prev[i] == section]
    rest = [i for i in pool if prev[i] != section]
    rng.shuffle(keep)
    rng.shuffle(rest)
    chosen = (keep + rest)[:quota]
    assert len(chosen) == quota, (quota, len(pool))
    taken.update(chosen)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print tables, write nothing")
    a = ap.parse_args()

    prev = read_split(V1)
    with open(MANIFEST) as f:
        man_rows = list(csv.DictReader(f))
        fieldnames = list(man_rows[0].keys())
    by_rel = {r["rel_path"]: r for r in man_rows}
    rows = [by_rel[rel] for rel in prev]           # v1 order, all must exist in the manifest
    n = len(rows)
    rng = random.Random(SEED)

    hold = [r for r in rows if is_holdout(r)]
    main_rows = [r for r in rows if not is_holdout(r)]
    tot_train, tot_val, tot_test = largest_remainder([n * f for f in FRACS])

    new = {}
    taken = set()

    # --- holdouts: val:test = 1:2 per group, never train
    hold_groups = defaultdict(list)
    for r in hold:
        hold_groups[(r["source"], r["vendor"], r["centre"])].append(r["rel_path"])
    hold_val = hold_test = 0
    for key in sorted(hold_groups):
        ids = sorted(hold_groups[key])
        v, t = largest_remainder([len(ids) * HOLDOUT_VAL_FRAC, len(ids) * (1 - HOLDOUT_VAL_FRAC)])
        for i in fill(ids, v, prev, "val", rng, taken):
            new[i] = "val"
        for i in fill(ids, t, prev, "test", rng, taken):
            new[i] = "test"
        hold_val += v
        hold_test += t

    # --- everyone else: stratified, exact global quotas
    strata = defaultdict(list)
    for r in main_rows:
        strata[(r["source"], r["vendor"], r["centre"], r["pathology_label"])].append(r["rel_path"])
    keys = sorted(strata)
    sizes = [len(strata[k]) for k in keys]
    eligible = [s >= MIN_STRATUM_FOR_EVAL for s in sizes]
    n_elig = sum(s for s, e in zip(sizes, eligible) if e)
    need_val, need_test = tot_val - hold_val, tot_test - hold_test
    q_val = largest_remainder([s / n_elig * need_val if e else 0.0 for s, e in zip(sizes, eligible)])
    q_test = largest_remainder([s / n_elig * need_test if e else 0.0 for s, e in zip(sizes, eligible)])
    for k, qv, qt in zip(keys, q_val, q_test):
        ids = sorted(strata[k])
        assert qv + qt <= len(ids), (k, qv, qt, len(ids))
        for i in fill(ids, qv, prev, "val", rng, taken):
            new[i] = "val"
        for i in fill(ids, qt, prev, "test", rng, taken):
            new[i] = "test"
        for i in ids:
            if i not in taken:
                new[i] = "train"
                taken.add(i)

    # --- checks
    assert set(new) == set(prev) and len(new) == n
    counts = Counter(new.values())
    assert (counts["train"], counts["val"], counts["test"]) == (tot_train, tot_val, tot_test), counts
    assert not any(new[r["rel_path"]] == "train" for r in hold)
    moved = sum(1 for k in prev if prev[k] != new[k])

    # --- report
    print(f"total {n}: train/val/test = {tot_train}/{tot_val}/{tot_test}  "
          f"({100*tot_train/n:.1f}/{100*tot_val/n:.1f}/{100*tot_test/n:.1f})")
    print(f"holdout {len(hold)} subjects -> val {hold_val} / test {hold_test}")
    print(f"changed section vs v1: {moved} ({100*moved/n:.1f}%)   "
          + "  ".join(f"{a}->{b}:{c}" for (a, b), c in sorted(Counter((prev[k], new[k]) for k in prev if prev[k] != new[k]).items())))
    print()
    hdr = f"{'source/vendor/centre':40s} {'train':>6} {'val':>5} {'test':>5} | {'tr%':>5} {'va%':>5} {'te%':>5}"
    print(hdr)
    grp = defaultdict(Counter)
    for r in rows:
        grp[f"{r['source']}/{r['vendor']}/{r['centre']}"][new[r["rel_path"]]] += 1
    for g in sorted(grp):
        c = grp[g]
        print(f"{g:40s} {c['train']:6d} {c['val']:5d} {c['test']:5d} | "
              f"{100*c['train']/tot_train:5.1f} {100*c['val']/tot_val:5.1f} {100*c['test']/tot_test:5.1f}")
    for col in ("vendor", "pathology_label", "source"):
        print()
        gc = defaultdict(Counter)
        for r in rows:
            gc[r[col]][new[r["rel_path"]]] += 1
        for g in sorted(gc):
            c = gc[g]
            print(f"{col}={g:28s} {c['train']:6d} {c['val']:5d} {c['test']:5d} | "
                  f"{100*c['train']/tot_train:5.1f} {100*c['val']/tot_val:5.1f} {100*c['test']/tot_test:5.1f}")

    if a.dry_run:
        return

    header = (
        "# pooled_curated_v2 — THE live train/val/test split (docs/97). Generated by tools/build_curated_split_v2.py\n"
        "# from pooled_curated_v1.txt (the misalignment-curated 898-subject cohort, docs/96), RE-SPLIT so that\n"
        "# train, val and test share the same (source, vendor, centre, pathology) composition:\n"
        "#   * Canon (MNMs centre 5) and CMRxRecon2025 Center012 (Philips) are UNSEEN-VENDOR HOLDOUTS:\n"
        "#     never in train, split val:test = 1:2.\n"
        "#   * everyone else: stratified by (source, vendor, centre, pathology_label), per-stratum quotas\n"
        "#     by largest remainder so the totals are exactly 70/10/20; strata < 3 subjects are train-only.\n"
        "#   * v1 membership is kept wherever the quota allows (minimal churn). seed=42.\n"
        f"# {tot_train}/{tot_val}/{tot_test} train/val/test; {moved} subjects changed section vs v1.\n"
    )
    with open(OUT, "w") as f:
        f.write(header)
        for sec in SECTIONS:
            f.write(f"\n[{sec}]\n")
            for rel in prev:                     # keep v1 (= pooled.txt) line order within a section
                if new[rel] == sec:
                    f.write(rel + "\n")
    print(f"\nwrote {OUT}")

    col = "split_curated_v2"
    if col not in fieldnames:
        fieldnames.append(col)
    for r in man_rows:
        r[col] = new.get(r["rel_path"], "")
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(man_rows)
    print(f"updated {MANIFEST} column {col}")


if __name__ == "__main__":
    main()
