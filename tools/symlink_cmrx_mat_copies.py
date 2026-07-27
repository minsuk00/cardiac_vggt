"""Replace the 1 GB-per-subject `sax/cine_sax.mat` COPIES in CMRxRecon2024/Cine_combined with symlinks.

`_archive/batch_reconstruct_cmrxrecon2024.py:152` does `shutil.copy2(mat_file, ...)`, so every
reconstructed subject carries a full duplicate of its raw k-space — ~324 GB across 301 subjects,
which is essentially all of Cine_combined's 327 GB.

This is safe and already the convention here: the sibling `lax/cine_lax.mat` (298/298) and
`lvot/cine_lvot.mat` (300/300) in the same tree are ALREADY absolute symlinks, created by
`tools/reconstruct_cmrxrecon_lax.py`. Only the SAX copy was left real. We match that convention
(absolute symlinks) rather than introducing a second style.

SAFETY: every subject is verified BEFORE its copy is touched — source exists, size matches, and
md5 of three 8 MB windows (head/mid/tail) matches (--full-hash for a complete md5). Any subject
that fails is SKIPPED and reported, never removed. Dry-run is the default.

    python tools/symlink_cmrx_mat_copies.py                 # dry run (default) — changes nothing
    python tools/symlink_cmrx_mat_copies.py --apply         # verify, then replace
    python tools/symlink_cmrx_mat_copies.py --revert --apply # copy the real files back from source
"""

import argparse
import hashlib
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "scratch", "data", "CMRxRecon2024")
CINE = os.path.join(DATA, "Cine_combined")

SRC_ROOT = {
    "Train": "ChallengeData/Cine/TrainingSet/FullSample",
    "Val": "ChallengeData_AfterCompetition/Cine/ValidationSet/FullSample",
    "Test": "ChallengeData_AfterCompetition/Cine/TestSet/FullSample",
}
WIN = 8 << 20  # 8 MB verification windows


def window_md5(path, size, full=False):
    h = hashlib.md5()
    with open(path, "rb") as f:
        if full:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
            return h.hexdigest()
        for off in (0, max(0, size // 2 - WIN // 2), max(0, size - WIN)):
            f.seek(off)
            h.update(f.read(WIN))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually modify the filesystem")
    ap.add_argument("--revert", action="store_true", help="restore real copies from source")
    ap.add_argument("--full-hash", action="store_true", help="full md5 instead of 3 windows (slow)")
    args = ap.parse_args()

    subs = sorted(d for d in os.listdir(CINE) if os.path.isdir(os.path.join(CINE, d)))
    todo, already, failed, missing = [], [], [], []
    total = 0

    for sub in subs:
        copy = os.path.join(CINE, sub, "sax", "cine_sax.mat")
        if not os.path.exists(copy) and not os.path.islink(copy):
            missing.append((sub, "no cine_sax.mat"))
            continue
        try:
            split, pid = sub.split("_", 1)
            src = os.path.join(DATA, SRC_ROOT[split], pid, "cine_sax.mat")
        except (ValueError, KeyError):
            failed.append((sub, "cannot parse split/PID"))
            continue

        if os.path.islink(copy):
            if args.revert:
                if not os.path.exists(src):
                    failed.append((sub, "REVERT: source missing"))
                    continue
                todo.append((sub, copy, src, os.path.getsize(src)))
            else:
                already.append(sub)
            continue

        if args.revert:
            already.append(sub)  # already a real file
            continue

        if not os.path.exists(src):
            failed.append((sub, "SOURCE MISSING — keeping the copy"))
            continue
        cs, ss = os.path.getsize(copy), os.path.getsize(src)
        if cs != ss:
            failed.append((sub, f"SIZE MISMATCH copy={cs} src={ss}"))
            continue
        if window_md5(copy, cs, args.full_hash) != window_md5(src, ss, args.full_hash):
            failed.append((sub, "HASH MISMATCH — keeping the copy"))
            continue
        todo.append((sub, copy, src, cs))
        total += cs

    mode = ("REVERT" if args.revert else "SYMLINK") + ("" if args.apply else "  [DRY RUN]")
    print(f"[{mode}]  subjects={len(subs)}")
    print(f"  verified & ready : {len(todo)}")
    print(f"  already done     : {len(already)}")
    print(f"  FAILED (skipped) : {len(failed)}")
    print(f"  no .mat          : {len(missing)}")
    if not args.revert:
        print(f"  space reclaimed  : {total/1e9:.1f} GB")
    for s, why in failed[:20]:
        print(f"    !! {s}: {why}")
    for s, why in missing[:10]:
        print(f"    -- {s}: {why}")

    if not args.apply:
        print("\n  dry run — nothing changed. Re-run with --apply.")
        for s, c, sr, sz in todo[:3]:
            print(f"    would link {c}\n            -> {sr}")
        return

    done = 0
    for sub, copy, src, sz in todo:
        if args.revert:
            tmp = copy + ".restoring"
            shutil.copy2(src, tmp)
            os.remove(copy)
            os.replace(tmp, copy)
        else:
            os.remove(copy)
            os.symlink(src, copy)  # absolute, matching the existing lax/lvot convention
        done += 1
        if done % 50 == 0:
            print(f"    {done}/{len(todo)}")
    print(f"\n  {'restored' if args.revert else 'symlinked'}: {done}")


if __name__ == "__main__":
    main()
