#!/usr/bin/env python
"""One-off migration: make evaluation/ provenance mtime-free BEFORE refreshing GPFS timestamps.

Why: the eval harness used file mtimes as proxies — `ckpt_fingerprint = size:int(mtime)` (arm
identity) and "derived file older than gt_t00" (cine freshness). A GPFS purge-avoidance `touch`
rewrites every mtime, which would turn every existing arm into a "DIFFERENT checkpoint" and flip
freshness checks at random. evaluation/paths.py now uses content ids; this script stamps the
EXISTING files with them, using the old mtime rules ONE LAST TIME (so run it before any touch).

Per subject under evaluation/volumes/<ds>/out/<subj>/:
  cine_gt.nii.gz            -> write cine_gt.src.json {"gt_sha256"} iff cine mtime >= gt_t00 mtime
  <arm>/metadata.json       -> ckpt_fingerprint  "size:mtime" -> "v2:size:sha"  (only if the legacy
                               id still matches the ckpt on disk — proves it is the same file)
  <arm>/metrics.json        -> same fingerprint rewrite; add "gt_sha256" iff every existing
                               cine_{clean,breath} has mtime >= gt_t00 mtime
  <arm>/recon_*/stamp.json  -> ckpt_fingerprint / container_id  legacy -> v2

Only those fields change: fingerprints are replaced as substrings (nothing else in the file is
touched); the gt_sha256 key is added through json only when a re-dump reproduces the file
byte-for-byte otherwise. Everything is dry-run unless --apply. Anything that cannot be verified is
reported and left alone.

    PYTHONPATH=training:. python tools/migrate_eval_fingerprints.py            # dry run
    PYTHONPATH=training:. python tools/migrate_eval_fingerprints.py --apply
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import paths  # noqa: E402

LEGACY = re.compile(r"^\d+:\d+$")
# Containers whose legacy size:mtime ids appear in svrtk3d/nesvor stamp.json (no path is recorded
# in the stamp, so map by the known SIF locations).
SIFS = [os.environ.get("FCMR_SIF") or str(ROOT / "scratch/fetal_cmr_4d/sif/svrtk.sif"),
        str(ROOT / "scratch/nesvor/sif/nesvor.sif")]

stats = Counter()
_v2_cache = {}


def legacy_id(path):
    try:
        st = os.stat(path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return None


def v2_for(path):
    if path not in _v2_cache:
        _v2_cache[path] = paths.ckpt_fingerprint(path)
    return _v2_cache[path]


def note(kind, msg):
    stats[kind] += 1
    print(f"  [{kind}] {msg}")


def write(path, text, apply):
    if apply:
        tmp = f"{path}.migrate{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)


def map_fingerprint(old, ckpt, what):
    """legacy -> v2 for `ckpt`, or None (with a note) if it cannot be proven the same file."""
    if not old or not LEGACY.match(str(old)):
        return None                                   # already v2 / absent / not ours
    if not ckpt or not os.path.exists(ckpt):
        note("skip-ckpt-missing", f"{what}: ckpt {ckpt!r} not on disk; fingerprint left as {old}")
        return None
    cur = legacy_id(ckpt)
    if cur != old:
        note("skip-ckpt-changed", f"{what}: ckpt {ckpt} is now {cur}, stamp says {old} — NOT the same "
             f"file (already touched, or replaced); left alone")
        return None
    return v2_for(ckpt)


def replace_field(path, key, new, apply):
    """Substring-replace the value of `key` (a legacy id) with `new`; nothing else changes."""
    text = open(path).read()
    old = json.loads(text).get(key)
    pattern = f'"{key}": "{old}"'
    if text.count(pattern) != 1:
        note("skip-unparseable", f"{path}: could not locate {pattern!r} exactly once")
        return False
    write(path, text.replace(pattern, f'"{key}": "{new}"'), apply)
    note("rewrite-" + key, f"{os.path.relpath(path, paths.VOLUMES)}: {old} -> {new}")
    return True


def add_key(path, key, value, apply):
    text = open(path).read()
    d = json.loads(text)
    if key in d:
        return
    if json.dumps(d, indent=2) != text.rstrip("\n"):
        note("skip-unparseable", f"{path}: not a plain indent=2 json dump; {key} not added")
        return
    d[key] = value
    write(path, json.dumps(d, indent=2), apply)
    note("add-" + key, f"{os.path.relpath(path, paths.VOLUMES)}")


def migrate_subject(subj_dir, apply):
    gt0 = subj_dir / "gt" / "gt_t00.nii.gz"
    if not gt0.exists():
        note("skip-no-gt", str(subj_dir))
        return
    gt0_mtime = os.path.getmtime(gt0)
    gt_sha = paths.file_sha256(gt0)

    cgt, src = subj_dir / "cine_gt.nii.gz", subj_dir / "cine_gt.src.json"
    if cgt.exists() and not src.exists():
        if os.path.getmtime(cgt) >= gt0_mtime:
            write(src, json.dumps({"gt_sha256": gt_sha}, indent=2), apply)
            note("add-cine_gt.src.json", os.path.relpath(src, paths.VOLUMES))
        else:
            note("stale-cine_gt", f"{os.path.relpath(cgt, paths.VOLUMES)} older than gt_t00 — left "
                 f"unstamped; image_metrics.py will regenerate it")

    for arm_dir in sorted(p for p in subj_dir.iterdir() if p.is_dir() and p.name not in paths.BUNDLE_DIRS):
        meta_p, metr_p = arm_dir / "metadata.json", arm_dir / "metrics.json"
        ckpt, v2 = None, None
        if meta_p.exists():
            meta = json.load(open(meta_p))
            ckpt = meta.get("ckpt")
            v2 = map_fingerprint(meta.get("ckpt_fingerprint"), ckpt, str(meta_p))
            if v2:
                replace_field(meta_p, "ckpt_fingerprint", v2, apply)
        if metr_p.exists():
            metr = json.load(open(metr_p))
            fp = metr.get("ckpt_fingerprint")
            if fp and LEGACY.match(str(fp)):
                v2m = v2 if (v2 and ckpt == metr.get("ckpt")) else map_fingerprint(fp, metr.get("ckpt"), str(metr_p))
                if v2m:
                    replace_field(metr_p, "ckpt_fingerprint", v2m, apply)
            cines = [arm_dir / f"cine_{v}.nii.gz" for v in ("clean", "breath")]
            cines = [c for c in cines if c.exists()]
            if cines and "gt_sha256" not in metr:
                if all(os.path.getmtime(c) >= gt0_mtime for c in cines):
                    add_key(metr_p, "gt_sha256", gt_sha, apply)
                else:
                    note("stale-cine", f"{os.path.relpath(arm_dir, paths.VOLUMES)}: a cine_* predates "
                         f"gt_t00 — gt_sha256 NOT added; re-run image_metrics.py for this arm")
        for stamp_p in sorted(arm_dir.glob("recon_*/stamp.json")):
            st = json.load(open(stamp_p))
            fp = st.get("ckpt_fingerprint")
            if fp and LEGACY.match(str(fp)):
                v2s = v2 if (v2 and ckpt == st.get("ckpt")) else map_fingerprint(fp, st.get("ckpt"), str(stamp_p))
                if v2s:
                    replace_field(stamp_p, "ckpt_fingerprint", v2s, apply)
            cid = st.get("container_id")
            if cid and LEGACY.match(str(cid)):
                match = [s for s in SIFS if legacy_id(s) == cid]
                if len(match) == 1:
                    replace_field(stamp_p, "container_id", v2_for(match[0]), apply)
                else:
                    note("skip-sif-unknown", f"{os.path.relpath(stamp_p, paths.VOLUMES)}: container_id {cid} "
                         f"matches {len(match)} known SIFs; left alone")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--datasets", nargs="*", help="restrict to these volumes/<ds> (default: all)")
    ap.add_argument("--subject", help="restrict to one subject name (debugging)")
    args = ap.parse_args()

    datasets = args.datasets or sorted(p.name for p in paths.VOLUMES.iterdir()
                                       if (p / "out").is_dir() and not p.name.startswith("_"))
    print(f"{'APPLY' if args.apply else 'DRY RUN'} over {datasets}")
    for ds in datasets:
        for subj_dir in sorted(Path(p) for p in glob.glob(str(paths.VOLUMES / ds / "out" / "*")) if os.path.isdir(p)):
            if args.subject and subj_dir.name != args.subject:
                continue
            migrate_subject(subj_dir, args.apply)
    print("\nsummary:")
    for k, n in sorted(stats.items()):
        print(f"  {n:6d}  {k}")
    if not args.apply:
        print("\n(dry run — re-run with --apply to write)")


if __name__ == "__main__":
    main()
