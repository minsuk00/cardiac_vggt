"""Rescale the x (readout) voxel size of the CMRxRecon2025 Philips volumes. HEADER ONLY.

Why (docs/55). `reconstruct_subject` derives in-plane spacing as `FOVx / ReconMatrix_X`, which is
the pixel size only if `ReconMatrix_X` counts ACQUIRED readout samples. It does not -- it is an
OUTPUT grid size. For every Siemens subject the two coincide (`nx = 2*rx`, `ReadOutOversample = 2`,
so `base = nx/2 = rx`), which is why the formula worked for 490 of 502 Siemens volumes. Philips
acquires `nx = 304` and reconstructs onto `rx = 256`, so `base = 152 != 256` and the stamped
1.168 mm under-scales the readout axis by `rx/base = 256/152 = 1.684`.

Cropping (x: 304 -> 256) preserves pixel size; zero-filling (y: ~114 -> 256) preserves FOV. So
only x is wrong; `pixel_y = FOVy/ReconMatrix_Y` is already correct and is NOT touched. The result
is an anisotropic 1.967 x 1.168 mm grid whose ACQUIRED voxel is isotropic at 1.967 mm -- exactly
the pattern the UIH volumes already carry.

The voxel data is not read or rewritten; only the affine's first column (and the translation, so
the volume centre stays put) changes. Writes are atomic (tmp + os.replace). Every original zoom is
recorded in a sidecar so `--revert` restores byte-equivalent geometry.

STATUS: applied 2026-07-27 at the user's instruction. Quantitative confirmation via nnU-Net LV
segmentation is still OUTSTANDING -- see docs/55 sec. "What is still owed".

Usage:
    python tools/fix_philips_pixel_x.py --dry-run
    python tools/fix_philips_pixel_x.py --apply
    python tools/fix_philips_pixel_x.py --revert --apply
"""

import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = f"{REPO}/scratch/data/CMRxRecon2025"
SIDECAR = f"{ROOT}/_provenance/philips_pixel_x_fix.json"


def read_roos(cid):
    """ReadOutOversample from the staged CSV (copied verbatim from the source)."""
    p = f"{ROOT}/Cine_combined/{cid}/sax/cine_sax_info.csv"
    for line in open(p):
        k, _, v = line.partition(",")
        if k.strip().split("(")[0] == "ReadOutOversample":
            return float(v.strip())
    raise KeyError(f"ReadOutOversample missing in {p}")


def subject_files(cid):
    d = f"{ROOT}/Cine_combined/{cid}/sax"
    return sorted(glob.glob(f"{d}/4d_recon.nii.gz") + glob.glob(f"{d}/3d_recon/*.nii.gz"))


def set_staged_fovx(cid, new_fovx):
    """The STAGED csv is not a copy of the source -- `reconstruct_cmrx2025.normalize()` pre-bakes the
    intended pixel size into it (`FOVx = pixel_x * rx`), which is why UIH carries 540 where its source
    says 720. `tools/verify_recon_v2.py` re-derives expected in-plane spacing as FOV/ReconMatrix from
    THIS file, so it must move with the header or every Philips subject fails the `inplane` check."""
    p = f"{ROOT}/Cine_combined/{cid}/sax/cine_sax_info.csv"
    lines = open(p).read().splitlines(keepends=True)
    out, old = [], None
    for ln in lines:
        k, sep, _ = ln.partition(",")
        if sep and k.strip().split("(")[0] == "FOVx":
            old = float(ln.split(",")[1])
            ln = f"{k},{new_fovx:.6f}\n"
        out.append(ln)
    if old is None:
        raise KeyError(f"FOVx not found in {p}")
    tmp = p + ".fixtmp"
    open(tmp, "w").writelines(out)
    os.replace(tmp, p)
    return old


def update_report(updates):
    """Keep recon_report.json's pixel_mm in step -- precedent set by docs/54 sec.10c, where the Prisma
    pitch relabel updated the report because the verifier reads its expected pitch from there."""
    p = f"{ROOT}/recon_report.json"
    rows = json.load(open(p))
    n = 0
    for r in rows:
        if r["cid"] in updates:
            new_px, tag = updates[r["cid"]]
            r.setdefault("pixel_mm_assumed", list(r["pixel_mm"]))
            r["pixel_mm"] = [r["pixel_mm"][0], round(new_px, 4)]
            r["in_plane_aniso"] = round(max(new_px, r["pixel_mm"][0]) / min(new_px, r["pixel_mm"][0]), 3)
            r["pixel_x_source"] = tag
            n += 1
    tmp = p + ".fixtmp"
    json.dump(rows, open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    return n


def rescale(path, factor):
    """Scale the affine's axis-0 spacing by `factor`, keeping the volume centre fixed."""
    img = nib.load(path)
    aff = img.affine.copy()
    shape = np.array(img.shape[:3], dtype=float)
    centre = aff @ np.append(shape / 2.0, 1.0)
    aff[:3, 0] *= factor
    aff[:3, 3] += (centre[:3] - (aff @ np.append(shape / 2.0, 1.0))[:3])
    new = nib.Nifti1Image(np.asanyarray(img.dataobj), aff, img.header)
    new.header.set_zooms(tuple(np.array(img.header.get_zooms()) * ([factor] + [1.0] * (img.ndim - 1))))
    new.set_sform(aff, code=int(img.header["sform_code"]) or 1)
    new.set_qform(aff, code=int(img.header["qform_code"]) or 1)
    # tmp must keep the .nii.gz suffix -- nibabel infers the format from the extension.
    tmp = path[:-len(".nii.gz")] + ".fixtmp.nii.gz"
    nib.save(new, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    ap.add_argument("--revert", action="store_true", help="undo using the sidecar")
    args = ap.parse_args()

    rows = json.load(open(f"{ROOT}/recon_report.json"))
    ph = [r for r in rows if "Philips" in r["scanner"]]

    if args.revert:
        rec = json.load(open(SIDECAR))
        print(f"reverting {len(rec['subjects'])} subjects")
        upd = {}
        for cid, info in rec["subjects"].items():
            for f in subject_files(cid):
                if args.apply:
                    rescale(f, 1.0 / info["factor"])
            if args.apply and "old_fovx" in info:
                set_staged_fovx(cid, info["old_fovx"])
                upd[cid] = (info["old_px"], "reverted")
            print(f"  {cid}  /{info['factor']:.6f}")
        if args.apply:
            if upd:
                print("  report rows restored:", update_report(upd))
            os.rename(SIDECAR, SIDECAR + ".reverted")
        return

    if os.path.exists(SIDECAR):
        raise SystemExit(f"{SIDECAR} exists -- already applied. Use --revert first.")

    rec = {"note": "docs/55: pixel_x = FOVx/base, base = nx/ReadOutOversample (not ReconMatrix_X)",
           "subjects": {}}
    print(f"{'subject':50s} {'rx':>4} {'nx':>4} {'ROos':>5} {'base':>5} {'old_px':>7} {'new_px':>7} {'x':>7}")
    for r in ph:
        cid = r["cid"]
        rx, nx = r["recon_matrix"][0], r["shape_in"][4]
        roos = read_roos(cid)
        base = nx / roos
        factor = rx / base
        files = subject_files(cid)
        old = float(nib.load(files[0]).header.get_zooms()[0])
        rec["subjects"][cid] = {"factor": factor, "old_px": old, "new_px": old * factor,
                                "rx": rx, "nx": nx, "roos": roos, "n_files": len(files)}
        print(f"{cid:50s} {rx:4d} {nx:4d} {roos:5.1f} {base:5.0f} {old:7.4f} {old*factor:7.4f} "
              f"{factor:7.4f}  ({len(files)} files)")
        if args.apply:
            for f in files:
                rescale(f, factor)
            # keep the staged csv and the run report consistent with the header (see the helpers)
            rec["subjects"][cid]["old_fovx"] = set_staged_fovx(cid, old * factor * rx)
    if args.apply:
        n = update_report({c: (v["new_px"], "docs/55: FOVx/(nx/ReadOutOversample), 2026-07-27")
                           for c, v in rec["subjects"].items()})
        print(f"  recon_report.json rows updated: {n}")
        os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)
        json.dump(rec, open(SIDECAR, "w"), indent=2)
        print(f"\nAPPLIED. sidecar -> {SIDECAR}")
    else:
        print("\nDRY RUN -- nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
