"""Undo the CMRxRecon odd-Z slice roll: move the last stored slice back to the front.

See `docs/56`. Every CMRxRecon *challenge* release ships SAX stacks with an odd slice count
cyclically rolled by -1 (the most basal slice is stored LAST, after the apex). The fix is
`np.roll(vol, +1, axis=z)` on the affected subjects.

Design decisions (docs/56 §7):

* **Detect per subject, do not apply the parity rule blind.** The rule is 464/466, but there
  is one genuine odd-Z subject that is NOT rolled and one whose stack has no coherent order at
  all. Both estimators from `probe_slice_roll` must agree on k==1 or the subject is SKIPPED and
  reported.
* **The affine is left untouched.** Every subject's affine has origin (0,0,0) — CMRxRecon ships
  no slice position or orientation at all, and the recon writes the SimpleITK default. Rolling
  the voxels therefore translates the stack by one pitch in a coordinate frame that carries no
  absolute meaning, and the canonical pipeline (`Orientationd`/`Spacingd`/`ResizeWithPadOrCropd`)
  never reads the origin. Nothing downstream can see the translation.
* **3D frames and the 4D volume are rolled together**, so they stay mutually consistent
  (`verify_recon_v2.py` checks every 3d frame is bit-identical to its 4d slice).
* **Atomic** (tmp file + `os.replace`), fully reversible via the sidecar written to
  `<dataset>/_provenance/slice_roll_fix.json`.

Usage:
    python tools/fix_slice_roll.py                 # dry run: report what would change
    python tools/fix_slice_roll.py --apply
    python tools/fix_slice_roll.py --verify        # re-detect; expect 0 rolled everywhere
    python tools/fix_slice_roll.py --revert --apply

⚠️ After applying, DELETE the monai cache — `PersistentDataset` keys on the input paths, not on
file contents, so an existing cache would silently serve pre-fix volumes:
    rm -rf /tmp/vggt-mri_${USER}_monai_cache/
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_slice_roll import CMRX_YEARS, analyze  # noqa: E402

PROV = {y: os.path.join(os.path.dirname(d.rstrip("/")), "_provenance", "slice_roll_fix.json")
        for y, d in CMRX_YEARS.items()}


def subject_files(root, subj):
    sax = os.path.join(root, subj, "sax")
    fs = [os.path.join(sax, "3d_recon", f"sax_frame_{i:02d}.nii.gz") for i in range(12)]
    fs.append(os.path.join(sax, "4d_recon.nii.gz"))
    return [f for f in fs if os.path.exists(f)]


def roll_file(args):
    path, shift = args
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    rolled = np.roll(data, shift, axis=2)          # axis 2 == z for both (X,Y,Z) and (X,Y,Z,T)
    out = nib.Nifti1Image(rolled, img.affine, img.header)
    # the tmp name must keep the .nii.gz suffix -- nibabel infers the format from the extension
    tmp = path[: -len(".nii.gz")] + ".tmp_roll.nii.gz"
    nib.save(out, tmp)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--revert", action="store_true", help="undo using the sidecar record")
    ap.add_argument("--verify", action="store_true", help="re-detect only, no writes")
    ap.add_argument("--years", nargs="*", default=list(CMRX_YEARS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--audit", default=None,
                    help="write the full per-subject audit (.json + .csv) to this path prefix; "
                         "works in dry-run so the review lists can be inspected before writing")
    a = ap.parse_args()

    if a.revert:
        # ── ordering guard (docs/58 §10b) ────────────────────────────────────────────────────
        # The 2026-07-31 slice-order fix flipped z on 893 subjects, 410 of which are also in this
        # roll list. flip and roll do NOT commute: F∘R_k = R_{-k}∘F. Disk holds F(R_{+1}(x)); undoing
        # the roll first applies R_{-1}, giving R_{-1}FR_{+1}x = F(R_{+2}x) — the stack ends up TWO
        # slices wrong, shape-preserving and silent. The flip must be reverted FIRST.
        # ⚠️ ALSO: subject_files() below covers only the 12 frames + 4d_recon. heart_seg/heart_roi
        # (and both *_canonical) did not exist when this fix was applied but DO now — reverting
        # would move the images and leave every label one plane off. Fix that before using --revert.
        order_prov = os.path.join(os.path.dirname(os.path.dirname(next(iter(CMRX_YEARS.values())).rstrip("/"))),
                                  "_provenance", "slice_order_fix.json")
        if os.path.exists(order_prov):
            print(f"REFUSING: {order_prov} exists — the slice-order flip is still applied.")
            print("  Reverting the roll underneath it leaves the stack rolled by TWO slices, silently.")
            print("  Run `python tools/fix_slice_order.py --revert --apply` FIRST.")
            print("  Also note: this tool does NOT touch heart_seg/heart_roi/*_canonical, which now")
            print("  exist — reverting without extending subject_files() desyncs labels by one plane.")
            return 1
        for year in a.years:
            rec = PROV[year]
            if not os.path.exists(rec):
                print(f"{year}: no sidecar at {rec}, nothing to revert")
                continue
            subs = json.load(open(rec))["rolled_subjects"]
            files = [(f, -1) for s in subs for f in subject_files(CMRX_YEARS[year], s)]
            print(f"{year}: reverting {len(subs)} subjects / {len(files)} files"
                  + ("" if a.apply else "   [DRY RUN]"))
            if a.apply:
                with ProcessPoolExecutor(max_workers=a.workers) as ex:
                    list(ex.map(roll_file, files, chunksize=4))
                os.replace(rec, rec + ".reverted")
        return

    # ---- detect
    tasks = [(y, s, os.path.join(CMRX_YEARS[y], s, "sax", "3d_recon", "sax_frame_00.nii.gz"))
             for y in a.years for s in sorted(os.listdir(CMRX_YEARS[y]))]
    tasks = [t for t in tasks if os.path.exists(t[2])]
    print(f"detecting on {len(tasks)} subjects ...", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(analyze, tasks, chunksize=4))

    todo, skipped, audit = {y: [] for y in a.years}, [], []
    for r in res:
        if "error" in r:
            skipped.append((r["subj"], r["error"]))
            audit.append({"year": r["tag"], "subject": r["subj"], "action": "skipped",
                          "reason": r["error"]})
            continue
        Z = r["Z"]
        row = {"year": r["tag"], "subject": r["subj"], "Z": Z,
               "parity": "odd" if Z % 2 else "even",
               "k_adjacent": r["k_adjacent"], "k_global": r["k_global"], "agree": r["agree"],
               "corr_last_first": r["r_cyclic"][Z - 1],
               "corr_lastm1_last": r["r_cyclic"][Z - 2],
               "r_min": r["r_min"], "r_second_min": r["r_second_min"],
               "margin": round(r["r_second_min"] - r["r_min"], 4)}
        if not r["agree"] or r["k_adjacent"] not in (0, 1):
            skipped.append((r["subj"], f"Z={Z} k_adj={r['k_adjacent']} k_glob={r['k_global']}"))
            row["action"] = "skipped"
            row["reason"] = "estimators disagree or k not in {0,1}"
        elif r["k_adjacent"] == 1:
            todo[r["tag"]].append(r["subj"])
            row["action"] = "rolled"
        else:
            row["action"] = "left_alone"
        audit.append(row)

    # the two review lists: subjects that defy the parity expectation either way
    odd_not_rolled = [r for r in audit if r.get("parity") == "odd" and r.get("action") == "left_alone"]
    even_rolled = [r for r in audit if r.get("parity") == "even" and r.get("action") == "rolled"]

    def audit_blob():
        return {"doc": "docs/56", "shift": +1, "axis": "z (array axis 2)",
                "affine": "unchanged (origin is (0,0,0); no absolute position is shipped)",
                "counts": {"total": len(audit),
                           "rolled": sum(r.get("action") == "rolled" for r in audit),
                           "left_alone": sum(r.get("action") == "left_alone" for r in audit),
                           "skipped": sum(r.get("action") == "skipped" for r in audit)},
                "odd_z_NOT_rolled": odd_not_rolled,
                "even_z_rolled_UNEXPECTED": even_rolled,
                "skipped_for_review": [s for s, _ in skipped],
                "audit": audit}

    def write_audit(prefix):
        import csv
        json.dump(audit_blob(), open(prefix + ".json", "w"), indent=1)
        cols = ["year", "subject", "Z", "parity", "action", "k_adjacent", "k_global", "agree",
                "corr_last_first", "corr_lastm1_last", "r_min", "r_second_min", "margin", "reason"]
        with open(prefix + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in sorted(audit, key=lambda x: (x["year"], x["subject"])):
                w.writerow(r)
        print(f"wrote {prefix}.json / .csv  ({len(audit)} subjects)")

    print("\n--- REVIEW: odd-Z subjects that were NOT rolled "
          f"({len(odd_not_rolled)}) — expected to be rolled, verify these:")
    for r in odd_not_rolled:
        print(f"    {r['subject']:<50} Z={r['Z']:>2}  corr(last,first)={r['corr_last_first']:+.2f}"
              f"  corr(last-1,last)={r['corr_lastm1_last']:+.2f}  margin={r['margin']:.2f}")
    print(f"--- REVIEW: even-Z subjects that WERE rolled ({len(even_rolled)}) — should be zero:")
    for r in even_rolled:
        print(f"    {r['subject']:<50} Z={r['Z']}")

    if a.audit:
        write_audit(a.audit)

    if a.verify:
        for y in a.years:
            rows = [r for r in res if r.get("tag") == y and "error" not in r]
            print(f"{y}: still rolled {sum(r['k_adjacent'] == 1 for r in rows)}/{len(rows)}")
        for s, why in skipped:
            print(f"  SKIP/REVIEW {s}: {why}")
        return

    total = 0
    for y in a.years:
        n = len(todo[y])
        files = sum(len(subject_files(CMRX_YEARS[y], s)) for s in todo[y])
        total += files
        print(f"{y}: roll +1 on {n} subjects / {files} files")
    print(f"TOTAL {total} files" + ("" if a.apply else "   [DRY RUN — pass --apply to write]"))
    for s, why in skipped:
        print(f"  SKIP (needs review, NOT rolled) {s}: {why}")

    if not a.apply:
        return

    for y in a.years:
        files = [(f, 1) for s in todo[y] for f in subject_files(CMRX_YEARS[y], s)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, _ in enumerate(ex.map(roll_file, files, chunksize=4)):
                if (i + 1) % 500 == 0:
                    print(f"  {y}: {i+1}/{len(files)}", flush=True)
        os.makedirs(os.path.dirname(PROV[y]), exist_ok=True)
        blob = audit_blob()
        blob["rolled_subjects"] = todo[y]                       # --revert reads this
        blob["odd_z_NOT_rolled"] = [r for r in odd_not_rolled if r["year"] == y]
        blob["audit"] = [r for r in audit if r["year"] == y]
        json.dump(blob, open(PROV[y], "w"), indent=1)
        print(f"{y}: wrote {PROV[y]}  ({len(todo[y])} rolled, "
              f"{len(blob['odd_z_NOT_rolled'])} odd-Z left alone)")
    print("\nDONE. Now: rm -rf /tmp/vggt-mri_$USER_monai_cache/   and re-run with --verify")


if __name__ == "__main__":
    # propagate main()'s return code -- the --revert ordering guard above returns 1, and without
    # this the process would still exit 0 and sail through a shell `&&` chain. All other paths
    # return None => SystemExit(None) => exit 0, unchanged.
    raise SystemExit(main())
