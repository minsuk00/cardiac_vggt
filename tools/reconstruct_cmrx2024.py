"""Re-reconstruct CMRxRecon2024 SAX, sharded, with the fixed ESPIRiT input domain (docs/54).

Why this exists rather than calling `_archive/batch_reconstruct_cmrxrecon2024.py --subjects`:
that script's own `main()` does NOT patch `shutil.copy2`, so `reconstruct_subject`'s
packaging step does `copy2(<GPFS original>, <out>/sax/cine_sax.mat)` -- and in the live
2024 tree that destination is ALREADY A SYMLINK pointing at the same original (created by
`tools/symlink_cmrx_mat_copies.py`). copy2 onto a symlink-to-source raises
`shutil.SameFileError` and kills the whole batch. Observed: jobs 55163115_[0-1] died after
3 subjects. `tools/reconstruct_cmrx{2023,2025}.py` are immune because both patch copy2.

Same shape as the 2023/2025 drivers: recon math is NOT reimplemented -- `reconstruct_subject`
is imported unmodified from the archive script (which is itself a symlink to the canonical
copy parked next to the data at scratch/data/CMRxRecon2024/recon_code/).

  python tools/reconstruct_cmrx2024.py --subjects Train/P001 Test/P020
  python tools/reconstruct_cmrx2024.py --subject-file scratch/recon_v2_shards/2024_shard0.txt
"""

import argparse
import glob
import importlib.util
import os
import shutil
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D24 = os.path.join(REPO, "scratch", "data", "CMRxRecon2024")
ARCHIVE = os.path.join(REPO, "_archive", "batch_reconstruct_cmrxrecon2024.py")

DS_MAP = {
    "Train": "ChallengeData/Cine/TrainingSet/FullSample",
    "Val": "ChallengeData_AfterCompetition/Cine/ValidationSet/FullSample",
    "Test": "ChallengeData_AfterCompetition/Cine/TestSet/FullSample",
}
CSV_MAP = {
    "Train": "ChallengeData/Cine/TrainingSet/ImgSnapshot",
    "Val": "ChallengeData_AfterCompetition/Cine/ValidationSet/ImgSnapshot",
    "Test": "ChallengeData_AfterCompetition/Cine/TestSet/ImgSnapshot",
}

_orig_mat = {"path": None}   # the GPFS original, so the symlink never points at /tmp staging


def _copy2_symlinking_mat(src, dst):
    """Stand-in for shutil.copy2 used ONLY inside reconstruct_subject.

    The recon packages two files this way: the info CSV (tiny -> real copy) and the ~1 GB
    .mat (-> symlink to the GPFS original). Replacing rather than copying also side-steps
    SameFileError when `dst` is already a symlink to `src`.
    """
    if str(src).endswith(".mat"):
        target = _orig_mat["path"] or src
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(target), dst)
        return dst
    if os.path.lexists(dst) and os.path.islink(dst):
        os.remove(dst)
    return shutil._real_copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", help="'<Dataset>/<PID>', e.g. Train/P001")
    ap.add_argument("--subject-file", help="file with one '<Dataset>/<PID>' per line")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage-dir", default=f"/tmp/cmrx2024_recon_{os.environ.get('USER','u')}")
    ap.add_argument("--out-root", default=os.path.join(D24, "Cine_combined"))
    ap.add_argument("--force", action="store_true", help="redo subjects that already have output")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("recon2024", ARCHIVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(shutil, "_real_copy2"):          # idempotent: never wrap our own wrapper
        shutil._real_copy2 = shutil.copy2
    mod.shutil.copy2 = _copy2_symlinking_mat        # packaging only; the recon math is untouched

    subs = list(args.subjects or [])
    if args.subject_file:
        with open(args.subject_file) as f:
            subs += [ln.strip() for ln in f if ln.strip()]
    if not subs:                                    # default: everything on disk
        for ds, rel in DS_MAP.items():
            root = os.path.join(D24, rel)
            if os.path.isdir(root):
                subs += [f"{ds}/{p}" for p in sorted(os.listdir(root))
                         if os.path.exists(os.path.join(root, p, "cine_sax.mat"))]
    if args.limit:
        subs = subs[: args.limit]

    os.makedirs(args.stage_dir, exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)
    print(f"reconstructing {len(subs)} subjects -> {args.out_root}", flush=True)

    t_all = time.time()
    done = skipped = failed = 0
    for i, sub in enumerate(subs, 1):
        if "/" not in sub or sub.split("/", 1)[0] not in DS_MAP:
            print(f"[{sub}] SKIP: expected '<Dataset>/<PID>'", flush=True)
            failed += 1
            continue
        ds, pid = sub.split("/", 1)
        # CMRx24_ prefix (2026-07-27): 2024 used to be the only year without a year prefix, which
        # is ambiguous in a pooled cohort -- 2023 and 2024 both have a Train_P001. Omitting it here
        # silently writes to a NEW unprefixed dir next to the real one instead of updating it.
        cid = f"CMRx24_{ds}_{pid}"
        out_dir = os.path.join(args.out_root, cid)
        if not args.force and os.path.exists(os.path.join(out_dir, "sax", "4d_recon.nii.gz")):
            skipped += 1
            continue
        mat = os.path.join(D24, DS_MAP[ds], pid, "cine_sax.mat")
        csvp = os.path.join(D24, CSV_MAP[ds], pid, "cine_sax_info.csv")
        if not (os.path.exists(mat) and os.path.exists(csvp)):
            print(f"[{cid}] MISSING mat={os.path.exists(mat)} csv={os.path.exists(csvp)}", flush=True)
            failed += 1
            continue

        local = os.path.join(args.stage_dir, f"{cid}.mat")
        try:
            # GPFS strided reads are pathologically slow; one sequential copy to node-local
            # /tmp then read from there (same rationale as the 2023 driver).
            t0 = time.time()
            shutil._real_copy2(mat, local)
            t_stage = time.time() - t0
            t0 = t_start = time.time()
            _orig_mat["path"] = mat
            mod.reconstruct_subject(cid, local, csvp, out_dir, device_id=0)
            # reconstruct_subject signals failure by printing and returning, so verify the OUTPUT.
            # `os.path.exists(4d_recon)` alone is vacuous under --force (a stale file from an
            # earlier run satisfies it), and it misses zero-byte frames from an interrupted write
            # -- both actually happened on CMRx24_Train_P105. Check every file has real bytes.
            sax = os.path.join(out_dir, "sax")
            f4 = os.path.join(sax, "4d_recon.nii.gz")
            frames = sorted(glob.glob(os.path.join(sax, "3d_recon", "sax_frame_*.nii.gz")))
            empty = [f for f in frames + [f4] if not os.path.exists(f) or os.path.getsize(f) == 0]
            if not frames or empty:
                raise RuntimeError(f"incomplete output: {len(frames)} frames, "
                                   f"missing/empty={[os.path.basename(f) for f in empty][:5]}")
            if os.path.getmtime(f4) < t_start:
                raise RuntimeError("4d_recon.nii.gz was not rewritten by this run (stale file)")
            print(f"[{i}/{len(subs)}] {cid} stage {t_stage:.0f}s recon {time.time()-t0:.0f}s", flush=True)
            done += 1
        except Exception as e:
            print(f"[{i}/{len(subs)}] {cid} FAILED: {type(e).__name__}: {e}", flush=True)
            failed += 1
        finally:
            _orig_mat["path"] = None
            if os.path.exists(local):
                os.remove(local)        # keep /tmp bounded: one staged .mat at a time

    print(f"\ndone={done} skipped={skipped} failed={failed}  elapsed {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
