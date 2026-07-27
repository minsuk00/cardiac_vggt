"""Can CMRxRecon2023's missing `cine_sax_info.csv` be borrowed from CMRxRecon-300?

The 2023 CHALLENGE release ships raw k-space but **zero** geometry CSVs, so
`_archive/batch_reconstruct_cmrxrecon2024.py` hard-returns at its csv check.
CMRxRecon-300 is a different packaging of the SAME 2023 cohort and DOES ship
`cine_sax_info.csv`. This scans every 2023 subject, joins it to the donor by
(section, patient id), and validates the join against the raw k-space:

    SliceNum      == nz          (donor slice count matches k-space)
    ReconMatrix_Y == ny          (donor phase-encode matches k-space)
    nx            == 2 * ReconMatrix_X   (the 2x readout oversampling 2024 relies on)

Never blind-join on ID: TestSet/P118 is a known cross-release mismatch.

Usage:
    python tools/scan_cmrx2023_donor_geometry.py
    python tools/scan_cmrx2023_donor_geometry.py --write scratch/data/CMRxRecon2023/_geometry_csv
"""

import argparse
import csv
import json
import os
from collections import Counter

import h5py

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "scratch", "data")

# 2023 challenge section -> (k-space root, CMRxRecon-300 donor section)
SECTIONS = {
    "TrainingSet": ("CMRxRecon2023/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample", "TrainingSet"),
    "ValidationSet": ("CMRxRecon2023/ChallengeData_validation/MultiCoil/Cine/ValidationSet/FullSample", "ValidationSet"),
    "TestSet": ("CMRxRecon2023/ChallengeData_test/MultiCoil/Cine/TestSet/FullSample", "TestSet"),
}
REQUIRED = ["ReconMatrix_X", "ReconMatrix_Y", "FOVx", "FOVy", "SliceThickness"]


def read_csv(p):
    m = {}
    with open(p) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) == 2:
                m[row[0].strip()] = row[1].strip()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="if set, emit validated CSVs to <dir>/<Section>/<P###>/cine_sax_info.csv")
    ap.add_argument("--json-out", default="/tmp/cmrx2023_donor_scan.json")
    ap.add_argument("--manifest", default=None, help="if set, write a per-subject manifest CSV here")
    args = ap.parse_args()

    # duplicate map (skip the '<kept> <==> <redundant test copy>' format-comment line)
    dup_of = {}
    dpath = os.path.join(D, "CMRxRecon2023", "DUPLICATES.txt")
    if os.path.exists(dpath):
        for line in open(dpath):
            if "<==>" not in line or "<kept>" in line:
                continue
            a, b = (x.strip() for x in line.split("<==>"))
            dup_of[b] = a
    SHORT = {"TrainingSet": "train", "ValidationSet": "val", "TestSet": "test"}
    PREFIX = {"TrainingSet": "Train", "ValidationSet": "Val", "TestSet": "Test"}

    rows = []
    for section, (ksp_rel, donor_section) in SECTIONS.items():
        ksp_root = os.path.join(D, ksp_rel)
        if not os.path.isdir(ksp_root):
            print(f"!! missing {ksp_root}")
            continue
        for pid in sorted(os.listdir(ksp_root)):
            mat = os.path.join(ksp_root, pid, "cine_sax.mat")
            if not os.path.exists(mat):
                continue
            rec = {"section": section, "pid": pid}
            donor = os.path.join(D, "CMRxRecon-300", donor_section, pid, "cine_sax_info.csv")
            rec["donor_exists"] = os.path.exists(donor)
            try:
                with h5py.File(mat, "r") as f:
                    key = "kspace_full" if "kspace_full" in f else list(f.keys())[0]
                    nf, nz, nc, ny, nx = f[key].shape
                rec.update(nf=nf, nz=nz, nc=nc, ny=ny, nx=nx, h5key=key)
            except Exception as e:
                rec["error"] = str(e)[:80]
                rows.append(rec)
                continue
            if not rec["donor_exists"]:
                rec["verdict"] = "NO_DONOR"
                rows.append(rec)
                continue
            m = read_csv(donor)
            rec["missing_fields"] = [k for k in REQUIRED if not m.get(k, "")]
            try:
                sn, ry, rx = int(m["SliceNum"]), int(m["ReconMatrix_Y"]), int(m["ReconMatrix_X"])
            except (KeyError, ValueError):
                rec["verdict"] = "DONOR_UNPARSEABLE"
                rows.append(rec)
                continue
            rec.update(donor_SliceNum=sn, donor_Ry=ry, donor_Rx=rx,
                       donor_thick=m.get("SliceThickness", ""), donor_TemporalPhase=m.get("TemporalPhase", ""))
            checks = {"slice": sn == nz, "phase_enc": ry == ny, "readout_os2": nx == 2 * rx}
            rec["checks"] = checks
            rec["verdict"] = "OK" if all(checks.values()) and not rec["missing_fields"] else "MISMATCH"
            rec["failed"] = [k for k, v in checks.items() if not v]
            rows.append(rec)

            if args.write and rec["verdict"] == "OK":
                dst = os.path.join(args.write, section, pid)
                os.makedirs(dst, exist_ok=True)
                # lineterminator="\n": csv.writer defaults to CRLF, which would make the emitted
                # file byte-differ from the donor for no reason and break byte-level verification.
                with open(os.path.join(dst, "cine_sax_info.csv"), "w", newline="") as f:
                    w = csv.writer(f, lineterminator="\n")
                    w.writerow(["Parameter", "Value"])
                    for k, v in m.items():
                        w.writerow([k, v])

    if args.manifest:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
        with open(args.manifest, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["combined_id", "section", "pid", "verdict", "duplicate_of", "six_mm",
                        "reconstruct", "nz", "ny", "nx", "ReconMatrix_X", "ReconMatrix_Y",
                        "FOVx", "FOVy", "SliceThickness", "TemporalPhase", "reason_excluded"])
            for r in rows:
                key = f"{SHORT[r['section']]}/{r['pid']}"
                thick = str(r.get("donor_thick", ""))
                six = thick == "6"
                dup = dup_of.get(key, "")
                reasons = []
                if r.get("verdict") != "OK":
                    reasons.append(f"verdict={r.get('verdict')}")
                if dup:
                    reasons.append(f"duplicate_of={dup}")
                m = read_csv(os.path.join(D, "CMRxRecon-300", r["section"], r["pid"], "cine_sax_info.csv")) \
                    if r.get("donor_exists") else {}
                w.writerow([f"CMRx23_{PREFIX[r['section']]}_{r['pid']}", r["section"], r["pid"],
                            r.get("verdict"), dup, int(six), int(not reasons),
                            r.get("nz"), r.get("ny"), r.get("nx"),
                            r.get("donor_Rx"), r.get("donor_Ry"),
                            m.get("FOVx", ""), m.get("FOVy", ""), thick,
                            r.get("donor_TemporalPhase", ""), ";".join(reasons)])
        recon = [r for r in rows if r.get("verdict") == "OK"
                 and f"{SHORT[r['section']]}/{r['pid']}" not in dup_of]
        six_recon = [r for r in recon if str(r.get("donor_thick")) == "6"]
        print(f"\nmanifest -> {args.manifest}")
        print(f"  reconstruct = {len(recon)}  "
              f"{dict(Counter(PREFIX[r['section']] for r in recon))}")
        six_ids = [SHORT[r["section"]] + "/" + r["pid"] for r in six_recon]
        print(f"  of which 6 mm (KEPT, flagged six_mm=1): {six_ids}")

    json.dump(rows, open(args.json_out, "w"), indent=1)
    n = len(rows)
    print(f"2023 SAX subjects scanned: {n}")
    print(f"  verdicts: {dict(Counter(r.get('verdict', 'ERROR') for r in rows))}")
    print(f"  donor missing        : {sum(1 for r in rows if not r.get('donor_exists'))}")
    print(f"  h5 unreadable        : {sum(1 for r in rows if 'error' in r)}")
    bad = [r for r in rows if r.get("verdict") not in ("OK", None)]
    if bad:
        print(f"\n  --- {len(bad)} problem subjects ---")
        for r in bad[:40]:
            print(f"   {r['section']}/{r['pid']}: {r.get('verdict')} failed={r.get('failed')} "
                  f"ksp(nz,ny,nx)=({r.get('nz')},{r.get('ny')},{r.get('nx')}) "
                  f"donor(SliceNum,Ry,Rx)=({r.get('donor_SliceNum')},{r.get('donor_Ry')},{r.get('donor_Rx')}) "
                  f"missing={r.get('missing_fields')}")
    ok = [r for r in rows if r.get("verdict") == "OK"]
    print(f"\n  ✅ usable with borrowed geometry: {len(ok)} / {n}")
    print(f"  thickness values among OK: {dict(Counter(r['donor_thick'] for r in ok))}")
    print(f"  nframe among OK          : {dict(Counter(r['nf'] for r in ok))}")
    print(f"  ncoil among OK           : {dict(Counter(r['nc'] for r in ok))}")
    print(f"\n  json -> {args.json_out}")
    if args.write:
        print(f"  CSVs written under {args.write}")


if __name__ == "__main__":
    main()
