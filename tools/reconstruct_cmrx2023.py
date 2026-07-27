"""Reconstruct CMRxRecon2023 SAX using the EXACT 2024 recon function.

The recon itself is not reimplemented: `reconstruct_subject` is imported unmodified from
`_archive/batch_reconstruct_cmrxrecon2024.py` (the path verified against the shipped 2024 NIfTIs at
135-137 dB PSNR, see ../CMRxRecon2024/recon_code/README.md). This file is only the driver:

  * 2023 k-space roots instead of 2024's, and geometry CSVs BORROWED from CMRxRecon-300
    (2023 ships none) via tools/scan_cmrx2023_donor_geometry.py --write;
  * the subject list comes from SUBJECT_MANIFEST.csv (reconstruct==1), which already excludes
    the 68 test-side duplicates and P118 (no donor exists);
  * output ids are PREFIXED `CMRx23_` so they cannot collide with 2024's `Train_P001` when the
    years are pooled (subj_id is the directory basename);
  * the .mat is SYMLINKED rather than `shutil.copy2`-ed (2024 spent 324 GB on those copies);
  * each .mat is staged to node-local /tmp first, because GPFS small/strided reads are slow.

Usage:
    python tools/reconstruct_cmrx2023.py --limit 2      # smoke test
    python tools/reconstruct_cmrx2023.py                # all 196
"""

import argparse
import csv
import importlib.util
import os
import shutil
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D23 = os.path.join(REPO, "scratch", "data", "CMRxRecon2023")
ARCHIVE = os.path.join(REPO, "_archive", "batch_reconstruct_cmrxrecon2024.py")

KSP_ROOT = {
    "TrainingSet": "ChallengeData/MultiCoil/Cine/TrainingSet/FullSample",
    "ValidationSet": "ChallengeData_validation/MultiCoil/Cine/ValidationSet/FullSample",
    "TestSet": "ChallengeData_test/MultiCoil/Cine/TestSet/FullSample",
}

_orig_mat = {"path": None}  # the GPFS original, so the symlink never points at the /tmp staging copy


def _copy2_symlinking_mat(src, dst):
    """Stand-in for shutil.copy2 used ONLY inside reconstruct_subject.

    The recon writes two files this way: the info CSV (tiny -> keep a real copy) and the ~1 GB
    .mat (-> symlink to the GPFS original, matching what lax/lvot already do in 2024).
    """
    if src.endswith(".mat"):
        target = _orig_mat["path"] or src
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(target, dst)
        return dst
    return shutil._real_copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N subjects (0 = all)")
    ap.add_argument("--stage-dir", default=f"/tmp/cmrx2023_recon_{os.environ.get('USER','u')}")
    ap.add_argument("--out-root", default=os.path.join(D23, "Cine_combined"))
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("recon2024", ARCHIVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil._real_copy2 = shutil.copy2
    mod.shutil.copy2 = _copy2_symlinking_mat  # packaging only; the recon math is untouched

    with open(os.path.join(D23, "SUBJECT_MANIFEST.csv")) as f:
        subs = [r for r in csv.DictReader(f) if r["reconstruct"] == "1"]
    if args.limit:
        subs = subs[: args.limit]
    os.makedirs(args.stage_dir, exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)
    print(f"reconstructing {len(subs)} subjects -> {args.out_root}", flush=True)

    t_all = time.time()
    done = skipped = failed = 0
    for i, r in enumerate(subs, 1):
        cid, section, pid = r["combined_id"], r["section"], r["pid"]
        out_dir = os.path.join(args.out_root, cid)
        if os.path.exists(os.path.join(out_dir, "sax", "4d_recon.nii.gz")):
            skipped += 1
            continue
        mat = os.path.join(D23, KSP_ROOT[section], pid, "cine_sax.mat")
        csvp = os.path.join(D23, "_geometry_csv", section, pid, "cine_sax_info.csv")
        if not (os.path.exists(mat) and os.path.exists(csvp)):
            print(f"[{cid}] MISSING mat={os.path.exists(mat)} csv={os.path.exists(csvp)}", flush=True)
            failed += 1
            continue

        local = os.path.join(args.stage_dir, f"{cid}.mat")
        try:
            t0 = time.time()
            if not os.path.exists(local):
                shutil._real_copy2(mat, local)
            t_stage = time.time() - t0
            _orig_mat["path"] = mat  # symlink target = GPFS original, NOT `local`
            t0 = time.time()
            mod.reconstruct_subject(cid, local, csvp, out_dir, device_id=0)
            print(f"[{i}/{len(subs)}] {cid} six_mm={r['six_mm']} "
                  f"stage {t_stage:.0f}s recon {time.time()-t0:.0f}s", flush=True)
            done += 1
        except Exception as e:
            print(f"[{cid}] FAILED: {type(e).__name__}: {e}", flush=True)
            failed += 1
        finally:
            _orig_mat["path"] = None
            if os.path.exists(local):
                os.remove(local)  # keep /tmp bounded: one staged .mat at a time

    print(f"\ndone={done} skipped={skipped} failed={failed}  "
          f"elapsed {(time.time()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
