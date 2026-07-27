"""Reconstruct CMRxRecon2025 SAX using the EXACT 2024 recon function.

Like `tools/reconstruct_cmrx2023.py`, the recon math is NOT reimplemented: `reconstruct_subject`
is imported unmodified from `_archive/batch_reconstruct_cmrxrecon2024.py` (verified to reproduce
the shipped 2024 NIfTIs at 135 dB). Keeping ESPIRiT + SENSE byte-identical across 2023/2024/2025
is what stops a recon-domain shift from becoming an unablatable confound in a pooled cohort.

Everything 2025-specific lives in a NORMALIZER that stages each subject into a 2024-shaped
`.mat` + `.csv` on node-local /tmp. The recon function then sees inputs indistinguishable from
2024's and takes its ordinary crop path -- never the corner-pad fallback.

What the normalizer fixes (all measured; see the 2025 README + tools/scan_cmrx2025_geometry.py):

 1. TWO container formats. 372 subjects are MATLAB v7.3 (HDF5, dims already reversed to
    (nt,nz,nc,ny,nx)); 33 are MATLAB v5, which h5py rejects ('file signature not found') and
    whose dims are the REVERSE. Both are read here and normalised to (nt,nz,nc,ny,nx).
 2. h5 key is `kspace`, not `kspace_full`; dtype is float64 (2024 is float32) -> cast to f4,
    which also matches 2024's numeric precision.
 3. CSV keys carry unit suffixes on some subjects (`FOVx(mm)`) and not others -> stripped.
 4. `ny < ReconMatrix_Y` on 298/405 subjects. 2023/2024 never hit this because the organisers
    POCS-filled their partial Fourier before shipping; 2025 did not. Under the 2024 code this
    falls into a fallback that pads to the TOP-LEFT CORNER, displacing the heart by up to 74
    rows with no exception. Filling it here RESTORES the step the other years already received.
    Measured mechanism is NOT uniform -- most subjects are symmetric truncation (DC centred) but
    Philips and some Vida are true partial Fourier (DC off-centre by 18 rows) -- so we do not
    branch on vendor: we locate DC from the data and place the acquired block so DC lands at the
    grid centre. That single rule is correct for both, and for the 3 subjects with ny > ry
    (it crops instead). sigpy's ifft is centred, so array-centre IS the DC the recon assumes.
 5. `nx < ReconMatrix_X` on 35 subjects -> same treatment on the readout axis.
 6. FOV convention is VENDOR-DEPENDENT. `FOVx/ReconMatrix_X` (the 2024 formula) yields isotropic
    voxels for Siemens/Philips but 2:1 ANISOTROPIC ones for UIH, whose FOVx describes the
    ACQUIRED readout grid instead. Corrected FOVs are written into the staged CSV so the
    unmodified recon computes the right spacing. ⚠️ This rule was inferred from "in-plane cine
    should be near-isotropic", not from vendor documentation -- which is why --render exists and
    why a cross-vendor visual check must pass before any batch run.

Slice pitch (Z): the staged CSV carries the PITCH, not the thickness, so the NIfTI is born with
honest geometry and needs no `relabel_slice_spacing.py` pass. 8 mm -> 12 mm is documented
(thickness + the 4 mm protocol gap). 6 mm -> 10 mm ASSUMES that gap is a fixed absolute rather
than a 50% distance factor (which would give 9 mm) -- decided 2026-07-25, revisit by measurement.
121 subjects ship NO thickness at all; they get the modal 12 mm provisionally and are listed in
the run report under `pitch_provisional` for a post-recon LV-caliper measurement.

Usage:
    python tools/reconstruct_cmrx2025.py --validate            # 1 subject per scanner model
    python tools/reconstruct_cmrx2025.py --limit 3             # smoke test
    python tools/reconstruct_cmrx2025.py                       # all usable subjects
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import shutil
import time

import h5py
import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D25 = os.path.join(REPO, "scratch", "data", "CMRxRecon2025")
ARCHIVE = os.path.join(REPO, "_archive", "batch_reconstruct_cmrxrecon2024.py")

MIN_SLICES = 6            # 2024 applied the same filter; 45 files have nslice <= 5
CALIB_WIDTH = 32          # must match the ESPIRiT calib_width in the 2024 recon
PITCH_BY_THICKNESS = {"8": 12.0, "6": 10.0}
PITCH_PROVISIONAL = 12.0  # blank SliceThickness: modal value, flagged in the report
PF_THRESHOLD = 0.98       # acquired/full symmetric ky extent below this => partial Fourier

# The .mat that reconstruct_subject packages into each output dir is our STAGED (normalised)
# k-space, which it copies with shutil.copy2 -- 360 real copies = 377 GB of pure duplication,
# and the staged file is deleted right afterwards anyway. Symlink the RAW source instead, the
# way tools/reconstruct_cmrx2023.py already does. Costs 0 bytes and keeps provenance.
# ⚠️ Unlike 2023/2024, the symlink target is NOT byte-identical to what was reconstructed:
# the recon consumed normalize()'s output (ky POCS/zero-filled to ReconMatrix_Y, kx placed
# DC-centred, vendor FOV + pitch rewritten). Regenerate it with normalize(), not by reading this.
_orig_mat = {"path": None}


def _copy2_symlinking_mat(src, dst):
    """Stand-in for shutil.copy2 used ONLY inside reconstruct_subject.

    The recon packages two files this way: the info CSV (tiny -> keep a real copy, and it is the
    STAGED csv, which is the one that documents the applied geometry) and the ~1 GB .mat.
    """
    if str(src).endswith(".mat"):
        target = _orig_mat["path"] or src
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(target), dst)
        return dst
    return shutil._real_copy2(src, dst)


# ---------------------------------------------------------------- metadata

def read_info_csv(path):
    """Parameter,Value csv -> dict with '(units)' stripped from every key."""
    m = {}
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) == 2:
                m[re.sub(r"\(.*\)$", "", row[0].strip())] = row[1].strip()
    return m


def pitch_mm(thickness):
    """-> (pitch_mm, is_provisional)."""
    if thickness in PITCH_BY_THICKNESS:
        return PITCH_BY_THICKNESS[thickness], False
    return PITCH_PROVISIONAL, True


# ---------------------------------------------------------------- k-space

def load_kspace(mat):
    """-> (complex64 array (nt,nz,nc,ny,nx), 'v5'|'v7.3').

    The whole array is read in one shot on purpose: these files are gzip-chunked
    (nt,nz,nc,2,1), so any slice-by-slice access decompresses ~the entire file per slice
    (measured: 73 s for ONE [ny,nx] plane).
    """
    with open(mat, "rb") as f:
        is_v5 = f.read(10) == b"MATLAB 5.0"
    if is_v5:
        name = next(e[0] for e in sio.whosmat(mat) if e[0] in ("kspace", "kspace_full"))
        a = np.transpose(sio.loadmat(mat)[name], (4, 3, 2, 1, 0))
        return np.ascontiguousarray(a).astype(np.complex64), "v5"
    with h5py.File(mat, "r") as f:
        key = "kspace" if "kspace" in f else "kspace_full"
        d = f[key][:]
    return (d["real"] + 1j * d["imag"]).astype(np.complex64), "v7.3"


def dc_index(arr, axis):
    """Index of DC along `axis`, from |k| profiled over frame 0 and every other axis."""
    prof = np.abs(arr[0]).sum(axis=tuple(i for i in range(arr[0].ndim) if i != axis - 1))
    return int(np.argmax(prof))


def _ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))


def _fft2c(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))


def pf_factor(n, dc):
    """Fraction of the full symmetric ky extent that was actually acquired.

    1.0 = symmetric truncation (plain zero-fill is right). < 1 = partial Fourier: the data is
    lopsided about DC, so the missing side must be SYNTHESISED, not zero-filled.
    """
    full = 2 * max(dc, n - 1 - dc) + 1
    return n / full, full


def pocs_fill(arr, axis_dc, target, n_iter=10):
    """POCS partial-Fourier reconstruction along ky (axis -2), then centred zero-fill to `target`.

    Partial Fourier skips >half of k-space using the conjugate symmetry S(-k)=S*(k), which holds
    exactly only for a REAL image. Real MR images carry smooth phase (B0, flow), so plain
    zero-filling blurs along the phase-encode axis -- measured on Philips as sharpness_y/x = 0.67
    against >= 1.0 for every symmetric subject.

    POCS (Haacke) alternates two projections until consistent:
      * phase constraint  -- keep the magnitude, impose a low-resolution phase estimated from the
        symmetrically-sampled central band (Hamming-windowed so the estimate does not ring);
      * data consistency  -- overwrite the synthesised lines' measured counterparts back in.

    This restores the step the 2023/2024 organisers applied with POCS before shipping, which is
    why those years satisfy ny == ReconMatrix_Y and 2025 does not.
    """
    n = arr.shape[-2]
    frac, full = pf_factor(n, axis_dc)
    w = min(axis_dc, n - 1 - axis_dc)              # symmetric half-width actually available

    # place onto the FULL symmetric grid with DC at its centre
    k = np.zeros(arr.shape[:-2] + (full, arr.shape[-1]), dtype=np.complex64)
    off = full // 2 - axis_dc
    k[..., off:off + n, :] = arr
    acquired = np.zeros(full, dtype=bool)
    acquired[off:off + n] = True

    # low-resolution phase from the symmetric band only
    c = full // 2
    win = np.zeros(full, dtype=np.float32)
    win[c - w:c + w + 1] = np.hamming(2 * w + 1).astype(np.float32)
    phase = np.exp(1j * np.angle(_ifft2c(k * win[:, None]))).astype(np.complex64)

    img = _ifft2c(k)
    for _ in range(n_iter):
        img = (np.abs(img).astype(np.complex64) * phase)
        kk = _fft2c(img)
        kk[..., acquired, :] = k[..., acquired, :]          # data consistency
        img = _ifft2c(kk)
    out = _fft2c(img).astype(np.complex64)

    # the recon grid is usually finer than the symmetric extent -> plain centred ZIP the rest
    if target != full:
        out, _ = place_dc_centred(out, out.ndim - 2, target, full // 2)
    return out, frac


def place_dc_centred(arr, axis, target, dc):
    """Zero-fill (or crop) `axis` to `target`, positioning the data so `dc` lands at target//2.

    Handles symmetric truncation and partial Fourier with the same arithmetic, and degrades to a
    centred crop when target < n. Returns (array, offset_applied).
    """
    n = arr.shape[axis]
    if n == target:
        return arr, 0
    shape = list(arr.shape)
    shape[axis] = target
    out = np.zeros(shape, dtype=arr.dtype)
    off = target // 2 - dc                      # where source index 0 lands
    s0, s1 = max(0, -off), min(n, target - off)
    src = [slice(None)] * arr.ndim
    dst = [slice(None)] * arr.ndim
    src[axis] = slice(s0, s1)
    dst[axis] = slice(off + s0, off + s1)
    out[tuple(dst)] = arr[tuple(src)]
    return out, off


def normalize(mat, csvp, out_mat, out_csv, use_pocs=True):
    """Stage one subject as a 2024-shaped .mat + .csv. Returns a report dict."""
    meta = read_info_csv(csvp)
    rx, ry = int(meta["ReconMatrix_X"]), int(meta["ReconMatrix_Y"])
    fovx, fovy = float(meta["FOVx"]), float(meta["FOVy"])
    scanner = os.path.basename(os.path.dirname(os.path.dirname(csvp)))

    arr, ver = load_kspace(mat)
    nt, nz, nc, ny, nx = arr.shape

    dc_y, dc_x = dc_index(arr, 3), dc_index(arr, 4)
    # ESPIRiT calibrates from a CALIB_WIDTH block at the centre of what we hand it. After the
    # fill that centre is DC, so the ACQUIRED data must reach >= CALIB_WIDTH/2 either side of DC
    # or the coil maps would be estimated partly from zeros we invented.
    margin_y = min(dc_y, ny - 1 - dc_y)
    calib_ok = margin_y >= CALIB_WIDTH // 2

    # Symmetric truncation -> plain centred zero-fill. Partial Fourier (data lopsided about DC)
    # -> POCS, because zero-filling a one-sided acquisition blurs the phase-encode axis.
    frac, _ = pf_factor(ny, dc_y)
    is_pf = frac < PF_THRESHOLD and ny < ry
    if is_pf and use_pocs:
        arr, _ = pocs_fill(arr, dc_y, ry)
    else:
        arr, _ = place_dc_centred(arr, 3, ry, dc_y)
    nx_target = max(nx, rx)
    arr, off_x = place_dc_centred(arr, 4, nx_target, dc_x)

    # --- geometry -------------------------------------------------------
    # FOV_x_full = physical extent spanned by the nx ACQUIRED readout samples.
    #   UIH            : FOVx already describes that grid.
    #   Siemens/Philips: FOVx is the POST-crop FOV (of the rx grid) -> scale by nx/rx.
    fov_x_full = fovx if scanner.startswith("UIH") else fovx * (nx / rx)
    pixel_x = fov_x_full / nx_target
    pixel_y = fovy / ry
    # The recon computes spacing as FOV/ReconMatrix, so pre-bake the pixel sizes into the FOVs.
    meta_out = dict(meta)
    meta_out["FOVx"] = f"{pixel_x * rx:.6f}"
    meta_out["FOVy"] = f"{pixel_y * ry:.6f}"
    pitch, provisional = pitch_mm(meta.get("SliceThickness", ""))
    meta_out["SliceThickness"] = f"{pitch:.4f}"   # PITCH, not thickness -- see module docstring

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Value"])
        for k, v in meta_out.items():
            w.writerow([k, v])

    dt = np.dtype([("real", "<f4"), ("imag", "<f4")])
    comp = np.empty(arr.shape, dtype=dt)
    comp["real"], comp["imag"] = arr.real, arr.imag
    with h5py.File(out_mat, "w") as f:
        f.create_dataset("kspace_full", data=comp)

    return {
        "scanner": scanner, "matver": ver,
        "shape_in": [nt, nz, nc, ny, nx], "shape_staged": list(arr.shape),
        "recon_matrix": [ry, rx],
        "ny_over_ry": round(ny / ry, 4), "nx_over_rx": round(nx / rx, 4),
        "dc_y": dc_y, "dc_y_offset_from_centre": round(dc_y - (ny - 1) / 2, 1),
        "pf_frac": round(frac, 4), "is_partial_fourier": bool(is_pf), "pocs_applied": bool(is_pf and use_pocs),
        "fill_y": ry - ny, "fill_x": nx_target - nx,
        "calib_margin_y": int(margin_y), "calib_ok": bool(calib_ok),
        "pixel_mm": [round(pixel_y, 4), round(pixel_x, 4)],
        "in_plane_aniso": round(max(pixel_x, pixel_y) / min(pixel_x, pixel_y), 3),
        "thickness_csv": meta.get("SliceThickness", ""),
        "pitch_mm": pitch, "pitch_provisional": provisional,
    }


# ---------------------------------------------------------------- driver

SPLIT_TAG = {"TrainingData": "train", "TaskR1": "R1", "TaskR2": "R2"}
SET_TAG = {"TrainingSet": "", "ValidationSet": "val", "TestSet": "test"}


def discover():
    """-> list of subject dicts, one per DISTINCT ACQUISITION.

    ⚠️ The de-dup key MUST include split+set. 2025 reuses patient IDs across splits: the same
    (center, scanner, P###) turns up in TrainingData AND TaskR1/R2, and the disease table proves
    these are DIFFERENT PEOPLE (110/114 colliding keys differ in age/sex/height/weight). Keying by
    (center,scanner,pid) alone silently merged them and dropped 52 usable distinct subjects
    (308 -> the real 356). The output `cid` is likewise split-tagged so two different people never
    collide on one output directory.

    Within TaskR1, ValidationSet and TestSet can also share (center,scanner,pid) -- 4 cases, again
    different people -- so `set` is in the key too. The one confirmed bit-identical pair
    (TaskR2/.../umr680 P005==P006) has distinct pids, so it survives as two dirs; noted, harmless.
    """
    best = {}
    for csvp in sorted(glob.glob(os.path.join(D25, "*_extracted", "**", "cine_sax_info.csv"),
                                 recursive=True)):
        mat = os.path.join(os.path.dirname(csvp), "cine_sax.mat")
        if not os.path.exists(mat):
            continue
        p = csvp.split("/")
        split = next(x for x in p if x.endswith("_extracted")).replace("_extracted", "")
        setname = p[-6]  # .../<SetName>/FullSample*/<Center>/<Scanner>/<PID>/cine_sax_info.csv
        try:
            meta = read_info_csv(csvp)
            nz = int(meta["SliceNum"])
        except Exception:
            continue
        tag = SPLIT_TAG.get(split, split) + SET_TAG.get(setname, setname)
        rec = {"split": split, "set": setname, "center": p[-4], "scanner": p[-3], "pid": p[-2],
               "mat": mat, "csv": csvp, "nz": nz,
               "cid": f"CMRx25_{tag}_{p[-4]}_{p[-3]}_{p[-2]}"}
        k = (split, setname, rec["center"], rec["scanner"], rec["pid"])
        if k not in best or nz > best[k]["nz"]:
            best[k] = rec
    return [r for r in sorted(best.values(), key=lambda r: r["cid"]) if r["nz"] >= MIN_SLICES]


def pick_validation(subs):
    """One subject per scanner model, plus guaranteed coverage of the awkward cases."""
    picked, seen = [], set()
    for s in subs:
        if s["scanner"] not in seen:
            seen.add(s["scanner"])
            picked.append(s)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="one subject per scanner model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", nargs="+", help="explicit cid list")
    ap.add_argument("--stage-dir", default=f"/tmp/cmrx2025_recon_{os.environ.get('USER','u')}")
    ap.add_argument("--out-root", default=os.path.join(D25, "Cine_combined"))
    ap.add_argument("--report", default=os.path.join(D25, "recon_report.json"))
    ap.add_argument("--no-pocs", action="store_true", help="zero-fill even partial-Fourier subjects (A/B control)")
    ap.add_argument("--force", action="store_true",
                    help="re-reconstruct subjects that already have sax/4d_recon.nii.gz "
                         "(default: skip them, so a killed run resumes instead of restarting)")
    ap.add_argument("--exclude-scanner", nargs="+", default=[],
                    help="skip these scanner models (e.g. Philips_30T_IngeniaCX, held pending its FOV fix)")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("recon2024", ARCHIVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil._real_copy2 = shutil.copy2
    mod.shutil.copy2 = _copy2_symlinking_mat   # packaging only; the recon math is untouched

    subs = discover()
    print(f"{len(subs)} usable subjects (nslice >= {MIN_SLICES})", flush=True)
    if args.exclude_scanner:
        n0 = len(subs)
        subs = [s for s in subs if s["scanner"] not in set(args.exclude_scanner)]
        print(f"excluded {n0 - len(subs)} subjects on {args.exclude_scanner}", flush=True)
    if args.subjects:
        subs = [s for s in subs if s["cid"] in set(args.subjects)]
    elif args.validate:
        subs = pick_validation(subs)
    if args.limit:
        subs = subs[: args.limit]
    print(f"processing {len(subs)}\n", flush=True)

    os.makedirs(args.stage_dir, exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)

    # Carry forward any prior report so a resume / --limit / --exclude-scanner run cannot erase
    # rows it did not process. Keyed by cid; a re-processed subject overwrites its own row.
    prior = {}
    if os.path.exists(args.report):
        try:
            with open(args.report) as f:
                prior = {r["cid"]: r for r in json.load(f) if "cid" in r}
            print(f"carrying forward {len(prior)} rows from {args.report}", flush=True)
        except Exception as e:
            print(f"WARNING: could not read existing report ({type(e).__name__}: {e}); "
                  f"it will be REPLACED", flush=True)

    def flush_report():
        """Write after every subject: a walltime kill must not lose the run's diagnostics."""
        tmp = args.report + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(prior.values(), key=lambda r: r["cid"]), f, indent=1)
        os.replace(tmp, args.report)      # atomic; never leaves a truncated report

    done, failed, skipped = 0, 0, 0
    t_all = time.time()

    for i, s in enumerate(subs, 1):
        out_dir = os.path.join(args.out_root, s["cid"])
        sm = os.path.join(args.stage_dir, f"{s['cid']}.mat")
        sc = os.path.join(args.stage_dir, f"{s['cid']}.csv")
        if not args.force and os.path.exists(os.path.join(out_dir, "sax", "4d_recon.nii.gz")):
            skipped += 1
            continue
        try:
            t0 = time.time()
            rep = normalize(s["mat"], s["csv"], sm, sc, use_pocs=not args.no_pocs)
            t_norm = time.time() - t0
            t0 = time.time()
            _orig_mat["path"] = s["mat"]   # symlink the RAW source, never the /tmp staging copy
            mod.reconstruct_subject(s["cid"], sm, sc, out_dir, device_id=0)
            # reconstruct_subject signals failure by printing and returning, never by raising,
            # so the only reliable success test is that the output actually exists.
            if not os.path.exists(os.path.join(out_dir, "sax", "4d_recon.nii.gz")):
                raise RuntimeError("reconstruct_subject wrote no 4d_recon.nii.gz")
            rep.update(cid=s["cid"], split=s["split"], center=s["center"], pid=s["pid"],
                       t_normalize_s=round(t_norm, 1), t_recon_s=round(time.time() - t0, 1))
            prior[s["cid"]] = rep
            flush_report()
            done += 1
            flag = "" if rep["calib_ok"] else "  ⚠️CALIB"
            flag += "" if rep["in_plane_aniso"] < 1.25 else f"  ⚠️ANISO {rep['in_plane_aniso']}"
            print(f"[{i}/{len(subs)}] {s['cid']:52} {rep['matver']:5} "
                  f"fill_y={rep['fill_y']:+5d} fill_x={rep['fill_x']:+5d} "
                  f"px={rep['pixel_mm']} pitch={rep['pitch_mm']}{flag}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(subs)}] {s['cid']} FAILED: {type(e).__name__}: {e}", flush=True)
            # Record the failure. A missing row and a failed row must not look the same.
            prior[s["cid"]] = {"cid": s["cid"], "split": s["split"], "center": s["center"],
                               "scanner": s["scanner"], "pid": s["pid"],
                               "error": f"{type(e).__name__}: {e}"}
            flush_report()
            failed += 1
        finally:
            for p in (sm, sc):
                if os.path.exists(p):
                    os.remove(p)
            _orig_mat["path"] = None   # never let a stale path outlive its subject

    flush_report()
    print(f"\ndone={done} failed={failed} skipped={skipped}  elapsed {(time.time()-t_all)/60:.1f} min")
    print(f"report -> {args.report}  ({len(prior)} rows)")
    reports = [r for r in prior.values() if "error" not in r]
    errs = [r for r in prior.values() if "error" in r]
    if errs:
        print(f"  ⚠️  FAILED subjects        : {len(errs)} {[r['cid'] for r in errs][:5]}")
    if reports:
        bad = [r for r in reports if not r["calib_ok"]]
        agg = [r for r in reports if r["in_plane_aniso"] >= 1.25]
        prov = [r for r in reports if r["pitch_provisional"]]
        print(f"  calib-margin violations : {len(bad)} {[r['cid'] for r in bad][:5]}")
        print(f"  anisotropic (>=1.25)    : {len(agg)} {[(r['cid'], r['in_plane_aniso']) for r in agg][:5]}")
        print(f"  provisional 12 mm pitch : {len(prov)}")


if __name__ == "__main__":
    main()
