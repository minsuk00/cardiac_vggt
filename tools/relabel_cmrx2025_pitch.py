"""Relabel CMRxRecon2025 slice pitch for centres whose true pitch was MEASURED, not assumed.

Background. 114 subjects at three Siemens centres ship an EMPTY `SliceThickness`, so their pitch
was defaulted to 12 mm. Measuring LV length on the long-axis view (known in-plane spacing) against
the count of SAX slices containing LV -- calibrated on control subjects with a documented
thickness, which reproduced their known 12.0 mm to +2.1% -- gave:

    Center006 / Siemens_30T_Prisma   10.25 +/- 0.55 mm (n=6, pooled slope 10.05)  -> 10.0  CHANGE
    Center001 / Siemens_30T_Vida     12.22 +/- 0.58 mm (n=2, slope 12.27)         -> 12.0  no change
    Center004 / Siemens_15T_Aera     13.53 +/- 1.09 mm (n=4, slope 13.11)         -> 12.0  UNRESOLVED

Aera is genuinely inconclusive between 12 and 14 and cannot be improved this way -- only 4 of its
44 subjects ship any long-axis series and all 4 were used. It is left at 12 mm and documented.

Writes are atomic (tmp + os.replace) so a kill cannot truncate a NIfTI, and idempotent (a subject
already at the target is skipped). `recon_report.json` is updated in step with the headers --
otherwise `tools/verify_recon_v2.py`, which takes its expected 2025 pitch from that report, would
flag every relabeled subject. The original value is preserved as `pitch_mm_assumed`.

    micromamba run -n svr python tools/relabel_cmrx2025_pitch.py --dry-run
    micromamba run -n svr python tools/relabel_cmrx2025_pitch.py
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import nibabel as nib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D25 = os.path.join(REPO, "scratch", "data", "CMRxRecon2025")
REPORT = os.path.join(D25, "recon_report.json")
INPLANE_MAX_MM = 4.0

# (center, scanner) -> measured pitch in mm. Only entries that DIFFER from the recorded pitch
# actually cause a rewrite.
MEASURED = {("Center006", "Siemens_30T_Prisma"): 10.0}


def slice_axis(A):
    norms = [float(np.linalg.norm(A[:3, i])) for i in range(3)]
    big = [i for i, n in enumerate(norms) if n > INPLANE_MAX_MM]
    if len(big) != 1:
        raise ValueError(f"ambiguous slice axis, column norms {norms}")
    return big[0], norms[big[0]]


def relabel_file(path, target):
    img = nib.load(path)
    A = img.affine.copy()
    ax, cur = slice_axis(A)
    if abs(cur - target) < 1e-4:
        return "skip", cur
    A[:3, ax] = A[:3, ax] * (target / cur)
    img.set_sform(A)
    img.set_qform(A)
    tmp = path + ".relabeltmp.nii.gz"
    nib.save(img, tmp)
    os.replace(tmp, path)                       # atomic on the same filesystem
    return "change", cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rep = json.load(open(REPORT))
    by_cid = {x["cid"]: x for x in rep}
    targets = []
    for x in rep:
        t = MEASURED.get((x["center"], x["scanner"]))
        if t is None or abs(float(x["pitch_mm"]) - t) < 1e-4:
            continue
        sax = os.path.join(D25, "Cine_combined", x["cid"], "sax")
        if not os.path.isdir(sax):
            continue                            # archived (the duplicate) or not reconstructed
        targets.append((x["cid"], sax, t, float(x["pitch_mm"])))

    print(f"{len(targets)} subjects to relabel")
    for cid, _, t, old in targets[:3]:
        print(f"  e.g. {cid}: {old} -> {t} mm")
    if not targets:
        print("nothing to do")
        return
    if args.dry_run:
        print("(dry run -- nothing written)")
        return

    counts = {"change": 0, "skip": 0}
    for i, (cid, sax, t, _) in enumerate(targets, 1):
        files = sorted(glob.glob(os.path.join(sax, "3d_recon", "sax_frame_*.nii.gz")))
        files.append(os.path.join(sax, "4d_recon.nii.gz"))
        for f in files:
            if os.path.exists(f):
                r, _ = relabel_file(f, t)
                counts[r] += 1
        if i % 10 == 0:
            print(f"  {i}/{len(targets)}  {counts}", flush=True)

    # Keep the report in step with the headers, preserving the superseded value.
    for cid, _, t, old in targets:
        x = by_cid[cid]
        x["pitch_mm_assumed"] = old
        x["pitch_mm"] = t
        x["pitch_provisional"] = False
        x["pitch_source"] = "measured_lax_lv_length_2026-07-27"
    shutil.copy2(REPORT, REPORT + ".bak_preprisma")
    with open(REPORT, "w") as fh:
        json.dump(rep, fh, indent=1)

    print(f"\nDONE  files changed={counts['change']} skipped={counts['skip']}")
    print(f"recon_report.json updated for {len(targets)} subjects "
          f"(backup at {os.path.basename(REPORT)}.bak_preprisma)")


if __name__ == "__main__":
    main()
