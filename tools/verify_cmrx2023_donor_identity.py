"""Prove that CMRxRecon2023 `P###` and CMRxRecon-300 `P###` are the SAME PERSON.

Borrowing 2023's missing `cine_sax_info.csv` from CMRxRecon-300 joins the two releases by
(section, patient id). `tools/scan_cmrx2023_donor_geometry.py` validates that join structurally
(SliceNum/ReconMatrix vs k-space) — but that check is NOT sufficient: within a k-space shape
group the donor FOVy varies by up to ~20% (e.g. shape (10,204,512) has 6 distinct FOVy from
268.75 to 322.5 mm), so a wrong-but-same-shape donor passes the structural check and silently
yields in-plane spacing wrong by up to ~20%. TestSet/P118 proves cross-release ID drift is real.

This closes the gap by comparing image CONTENT: a cheap RSS image straight from the 2023
challenge k-space vs CMRxRecon-300's shipped `reconstruction/sax_4d.nii.gz` for the same ID,
with a wrong-ID negative control.

Usage: python tools/verify_cmrx2023_donor_identity.py [--n 15]
"""

import argparse
import csv
import json
import os
import random

import h5py
import numpy as np
import SimpleITK as sitk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "scratch", "data")
SECTIONS = {
    "TrainingSet": "CMRxRecon2023/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample",
    "ValidationSet": "CMRxRecon2023/ChallengeData_validation/MultiCoil/Cine/ValidationSet/FullSample",
    "TestSet": "CMRxRecon2023/ChallengeData_test/MultiCoil/Cine/TestSet/FullSample",
}


def centered_ifft2(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def challenge_rss(mat, z, ry, rx):
    """One slice, frame 0, from the challenge k-space -> magnitude image cropped to (ry, rx)."""
    with h5py.File(mat, "r") as f:
        d = f["kspace_full"]
        k = d["real"][0, z] + 1j * d["imag"][0, z]  # (ncoil, ny, nx)
    img = np.sqrt((np.abs(centered_ifft2(k)) ** 2).sum(0))
    ny, nx = img.shape
    y0, x0 = max(0, (ny - ry) // 2), max(0, (nx - rx) // 2)
    return img[y0 : y0 + ry, x0 : x0 + rx]


def donor_slice(nii, z):
    a = sitk.GetArrayFromImage(sitk.ReadImage(nii))  # (t, z, y, x)
    return a[0, min(z, a.shape[1] - 1)]


def ncc(a, b):
    """Normalised cross-correlation, CENTRE-cropping to the common shape.

    Centre, not top-left: a challenge RSS image is `nx = 2 * ReconMatrix_X` wide because of the
    readout oversampling, and the anatomy sits in the MIDDLE 256 of those 512 columns. A top-left
    crop compares the wrong half of the image and makes a true match look like noise (this silently
    broke an exhaustive P118 search until a positive control caught it).
    """
    if a.shape != b.shape:
        n0, n1 = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        ay, ax = (a.shape[0] - n0) // 2, (a.shape[1] - n1) // 2
        by, bx = (b.shape[0] - n0) // 2, (b.shape[1] - n1) // 2
        a, b = a[ay : ay + n0, ax : ax + n1], b[by : by + n0, bx : bx + n1]
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float((a * b).mean())


def read_meta(p):
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
    ap.add_argument("--n", type=int, default=15, help="subjects to sample per run (spread over sections)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank1", type=int, default=0,
                    help="if >0, rank the true donor against this many random same-section donors. "
                         "This is the decisive test: a single random control only shows 'better than one "
                         "other person', while rank-1 shows 'the best of N'.")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to 'Section/P###' entries")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cands = []
    for section, rel in SECTIONS.items():
        root = os.path.join(D, rel)
        if not os.path.isdir(root):
            continue
        pids = sorted(p for p in os.listdir(root) if os.path.exists(os.path.join(root, p, "cine_sax.mat")))
        for pid in pids:
            nii = os.path.join(D, "CMRxRecon-300", section, pid, "reconstruction", "sax_4d.nii.gz")
            csvp = os.path.join(D, "CMRxRecon-300", section, pid, "cine_sax_info.csv")
            if os.path.exists(nii) and os.path.exists(csvp):
                cands.append((section, pid, os.path.join(root, pid, "cine_sax.mat"), nii, csvp, pids))

    if args.only:
        want = set(args.only)
        cands = [c for c in cands if f"{c[0]}/{c[1]}" in want]
    rng.shuffle(cands)
    sample = cands[: args.n] if not args.only else cands
    print(f"{len(cands)} 2023 subjects have a CMRxRecon-300 recon + csv; testing {len(sample)}\n")
    print(f"{'subject':<22}{'NCC same-ID':>13}{'NCC wrong-ID':>14}   verdict")

    out, same, ctrl = [], [], []
    for section, pid, mat, nii, csvp, pids in sample:
        m = read_meta(csvp)
        ry, rx, nz = int(m["ReconMatrix_Y"]), int(m["ReconMatrix_X"]), int(m["SliceNum"])
        z = nz // 2
        try:
            a = challenge_rss(mat, z, ry, rx)
            b = donor_slice(nii, z)
            c_same = ncc(a, b)
            # negative control: a DIFFERENT subject's donor recon, same section
            others = [p for p in pids if p != pid and os.path.exists(
                os.path.join(D, "CMRxRecon-300", section, p, "reconstruction", "sax_4d.nii.gz"))]
            wrong = rng.choice(others)
            c_ctrl = ncc(a, donor_slice(
                os.path.join(D, "CMRxRecon-300", section, wrong, "reconstruction", "sax_4d.nii.gz"), z))
        except Exception as e:
            print(f"{section}/{pid:<12} ERROR {str(e)[:60]}")
            continue
        rec = dict(section=section, pid=pid, ncc_same=c_same, ncc_ctrl=c_ctrl, ctrl_pid=wrong)
        if args.rank1:
            # Decisive test: is the same-ID donor the ARGMAX over many candidates?
            pool = rng.sample(others, min(args.rank1, len(others)))
            scores = []
            for op in pool:
                try:
                    scores.append(ncc(a, donor_slice(
                        os.path.join(D, "CMRxRecon-300", section, op, "reconstruction", "sax_4d.nii.gz"), z)))
                except Exception:
                    pass
            beaten = sum(1 for s in scores if s >= c_same)
            rec.update(rank=beaten + 1, pool=len(scores), runner_up=max(scores) if scores else None)
            v = "RANK-1" if beaten == 0 else f"*** RANK {beaten+1}"
            print(f"{section}/{pid:<12}{c_same:>13.3f}{c_ctrl:>14.3f}   {v} of {len(scores)+1}"
                  f"  (runner-up {max(scores):.3f})" if scores else "")
        else:
            v = "MATCH" if c_same > 0.7 and c_same > c_ctrl + 0.2 else "*** LOW-ABS"
            print(f"{section}/{pid:<12}{c_same:>13.3f}{c_ctrl:>14.3f}   {v}  (ctrl={wrong})")
        rec["verdict"] = v
        same.append(c_same)
        ctrl.append(c_ctrl)
        out.append(rec)

    if same:
        print(f"\nsame-ID  NCC: mean {np.mean(same):.3f}  min {np.min(same):.3f}")
        print(f"wrong-ID NCC: mean {np.mean(ctrl):.3f}  max {np.max(ctrl):.3f}")
        print(f"separation  : {np.min(same) - np.max(ctrl):+.3f}  "
              f"({'CLEAN' if np.min(same) > np.max(ctrl) else 'OVERLAP -- join not proven'})")
        print(f"suspects    : {sum(1 for r in out if 'SUSPECT' in r['verdict'])} / {len(out)}")
    json.dump(out, open("/tmp/cmrx2023_donor_identity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
