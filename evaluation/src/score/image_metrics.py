"""Per-subject IMAGE metrics — resample a method's recons to the subject grid, score vs GT.

Successor to the scoring half of _archive/assemble_and_gif.py (the metric/gauge functions below
are copied VERBATIM from it so this dir stands alone); rendering lives in analysis/viz.py.
Other metric families live beside this file: ef_dice.py (function/seg), aggregate.py (folds
image + EF/Dice + breathing resp_diag + timing into ONE metric_results/<ds>/<arm>.json).

READ-ONLY CONTRACT: every input (bundles, recons, stamps) is opened read-only. This script
writes ONLY its own outputs:
    <subject>/<method>/metrics.json
    <subject>/<method>/cine_{clean,breath}.nii.gz
    <subject>/cine_gt.nii.gz                      (only if absent — never rewritten)
`_guarded_write_path` enforces this structurally: a path bug crashes instead of clobbering.
(The pre-restructure assemble_and_gif records were archived to <arm>/_old_scorer/ first, so
these clean names collide with nothing.)

Pose correction + PSF downsampling for the classical baselines (docs/83) hook into the load
step (`load_canon`) in a later phase; the VGGT path is anchored by construction (predicts off
absolute scanner_coords) so it loads untouched and records pose="none", psf="none".

Run: EVAL_DATASET=<ds> micromamba run -n svr python evaluation/src/score/image_metrics.py <subject> <method>
Cohort sweeps: evaluation/src/score/run.py. Paths/naming go through evaluation/paths.py.
"""
import json
import os
import sys

import numpy as np
import nibabel as nib
import nibabel.processing as nibproc

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

# Basenames this scorer is allowed to (over)write; everything else on disk is history.
_WRITABLE = {"metrics.json", "cine_clean.nii.gz", "cine_breath.nii.gz"}


def _guarded_write_path(path):
    """Refuse to write onto any pre-existing path that is not one of OUR new outputs.

    Our own outputs are re-runnable (a code fix must not need a new filename); anything else
    that already exists is the immutable record — a wrong path must crash, not clobber
    (same spirit as splat's no-default z_scale)."""
    path = Path(path)
    if path.exists() and path.name not in _WRITABLE:
        raise RuntimeError(f"refusing to overwrite pre-existing file: {path} "
                           f"(image_metrics.py may only write {sorted(_WRITABLE)} + a new cine_gt)")
    return str(path)


def subject_grid(ds, subj):
    """This subject's own scoring grid `(shape_xyz, affine)`, read from its GT bundle.

    Native-z (docs/58): D and dz belong to the subject, so there is no single canonical grid to
    resample onto."""
    img = nib.load(str(paths.bundle_stack(ds, subj, "gt", 0)))
    return tuple(img.shape[:3]), img.affine


def load_canon(path, shape_xyz, affine):
    """Load a NIfTI onto the subject's grid, resampling only if it is not already there.

    The resample is still load-bearing for the CLASSICAL baselines: SVRTK / NeSVoR / NiftyMIC
    reconstruct on their own (typically 1.4 mm isotropic) grid and have to be brought onto the GT
    grid to be scored at all. Bundle stacks and VGGT recons are already on it, so they load
    untouched — which is the point: no method should be silently resampled.
    """
    img = nib.load(path)
    if tuple(img.shape[:3]) == tuple(shape_xyz) and np.allclose(img.affine, affine):
        return np.asarray(img.dataobj, dtype=np.float32)
    out = nibproc.resample_from_to(img, (tuple(shape_xyz), affine), order=1, cval=0.0)
    return np.asarray(out.dataobj, dtype=np.float32)


def clip_sentinel(rec):
    """SVRTK writes -1 for voxels outside the recon mask (real intensities are >=0, hard gap
    at 0 — it's a flag, not intensity). Clip to 0 = treat as no-data/background. NO intensity
    calibration: SVRTK reconstructs in the normalized [0,1] input space and preserves that
    scale inside the mask, so a linear a*gt+b fit would absorb real reconstruction error and
    inflate PSNR (doc 29 ⚠️ correction). Score as-is."""
    return np.clip(rec, 0.0, None)


# Methods whose output is NOT on the input [0,1] scale (measured, CMRx24_Train_P053): NeSVoR pins its
# output to --output-intensity-mean=700 (arbitrary global gauge, k≈12000 vs GT); NiftyMIC adds a
# +1.0 pedestal (c≈+1.0). Scale-preserving methods (SVRTK: k≈1.05, c≈-0.01) are NOT listed —
# self-normalizing them LOSES ~1.9 dB of real reconstruction signal (29.85→27.93). This is a
# per-method rule keyed on measured scale-preservation, NOT a blanket uniform rescale.
SELF_NORM_METHODS = {"nesvor", "niftymic"}
# Of the self-norm methods, those whose gauge is a PURE global SCALE (offset≈0) — measured on
# CMRx24_Train_P053: NeSVoR pred≈2065·gt (offset/scale≈0.012). For these, subtracting the in-ROI p0.5 would
# inject a small ARTIFICIAL offset (the heart floor is not the true zero) and *under*-score the recon
# by ~0.3 dB (conservative but wrong-direction: it flatters our own method). Divide-only is the
# GT-consistent map (a perfect pure-scale recon → GT exactly). Methods with a real additive pedestal
# (NiftyMIC: c≈+1.0) are NOT here — they genuinely need the subtraction.
PURE_SCALE_METHODS = {"nesvor"}


def prep_recon(rec, method, roi):
    """Bring a method's recon onto the GT [0,1] scale for scoring, per-method:
      - scale-preserving (SVRTK): clip the -1 sentinel, score AS-IS.
      - pure-scale gauge (NeSVoR): divide by the recon's OWN in-ROI p99.9, clamp[0,1] (NO subtract).
      - offset+scale gauge (NiftyMIC): subtract in-ROI p0.5, divide by (p99.9-p0.5), clamp[0,1].
    All self-referenced (recon's OWN percentiles, NO GT → no leak). One global scale over all phases
    (keeps real phase-to-phase contrast, removes the arbitrary gauge). This is a GT-FREE APPROXIMATION
    of GT's normalization (preprocess.py: 0.5/99.9 over the non-zero FOV + clamp[0,1]) — we use the
    heart&FOV scoring ROI, not the full FOV, so it doesn't EXACTLY reproduce GT's affine, but the
    residual is ~0.3 dB on our data (GT heart p0.5≈0.013≈0). For a scale-INVARIANT read that sidesteps
    this entirely, use the ncc() metric (which needs no normalization)."""
    rec = np.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)  # harden: a NaN/Inf in the recon would
    # else make np.percentile return NaN and silently poison this method's whole PSNR/SSIM/NCC mean.
    if method not in SELF_NORM_METHODS:
        return clip_sentinel(rec)
    if not np.asarray(roi).any():           # empty ROI: no voxels to self-normalize against, and
        return np.clip(rec, 0.0, 1.0)       # np.percentile would IndexError on the zero-size slice.
    #                                         Scoring then NaNs on the empty mask (psnr/ssim/ncc guard it).
    vals = rec[:, roi] if rec.ndim == 4 else rec[roi]
    hi = np.percentile(vals, 99.9)
    if method in PURE_SCALE_METHODS:
        return np.clip(rec / max(hi, 1e-6), 0.0, 1.0)                # pure scale → divide only
    lo = np.percentile(vals, 0.5)
    return np.clip((rec - lo) / max(hi - lo, 1e-6), 0.0, 1.0)        # offset+scale → subtract then divide


def psnr(a, b, m):
    """Harness PSNR: peak = the GT's max INSIDE the ROI.

    ⚠️ NOT comparable to the trainer's `metric_psnr_3d_*`, which uses peak = 1.0. The ROI is the
    same object (`heart_roi_canonical`); only the normalization differs, by exactly
    `20*log10(gt[roi].max())`. Kept as the headline because it is the harness's cross-method
    convention; `psnr_unit_peak` below is emitted alongside for trainer reconciliation.
    """
    if not m.any():                       # empty ROI: b[m].max() would raise on a zero-size array
        return float("nan")
    mse = (((a - b) ** 2)[m]).mean()
    peak = max(b[m].max(), 1e-6)
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-10)))


def psnr_unit_peak(a, b, m):
    """PSNR with peak = 1.0 — the TRAINER's convention (`loss._psnr_from_mse`), for [0,1] data."""
    if not m.any():
        return float("nan")
    mse = (((a - b) ** 2)[m]).mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def ssim(a, b, m):
    """Global (whole-ROI) SSIM on the fixed [0,1] data range — NOT windowed, so not comparable to
    skimage's default. `L` is pinned to 1.0 rather than derived from the data: a data-derived range
    makes c1/c2 method-dependent (one bright outlier widens L and *raises* SSIM while PSNR falls);
    the inputs are normalized to [0,1] by construction, so the fixed range is the honest one.
    """
    a, b = a[m], b[m]
    if a.size < 2:
        return float("nan")
    mu_a, mu_b, va, vb = a.mean(), b.mean(), a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    L = 1.0
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def ncc(a, b, m):
    """Normalized cross-correlation over the ROI — INVARIANT to affine intensity (a·x+b), so it needs
    NO intensity normalization and is immune to prep_recon's self-norm scale/region choice. Standard SVR
    metric (the NeSVoR paper reports NCC). Returns NaN for <2 voxels or a constant image."""
    if not m.any() or m.sum() < 2:
        return float("nan")
    x, y = a[m], b[m]
    sx, sy = x.std(), y.std()
    if sx < 1e-8 or sy < 1e-8:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def check_variant_stamps(ds, subj, method, present):
    """Refuse to score `clean` and `breath` recons that came from DIFFERENT runs.

    Variants are discovered by `.is_dir()` and `run_vggt` only rewrites the ones in `--arms`, so a
    re-run under the same `--model-name` with the default `--arms breath` leaves a stale
    `recon_clean/` that is silently scored and differenced into `cost_psnr`. `paths.recon_stamp`
    is per VARIANT and can see this. Returns True only when every present variant carries an
    identical stamp; legacy unstamped dirs warn and return False; a MIX of stamped and unstamped
    raises — set ALLOW_MIXED_ARMS=1 if you know otherwise.
    """
    if len(present) < 2:
        return True
    stamps = {}
    for v in present:
        p = paths.recon_stamp(ds, subj, method, v)
        try:
            stamps[v] = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            stamps[v] = None
    if all(s is None for s in stamps.values()):
        print(f"  !! {subj} [{method}]: no per-variant stamps (pre-stamp run) — cannot verify "
              f"clean/breath came from the same run; cost_psnr is unverified", flush=True)
        return False
    if all(s is not None for s in stamps.values()) and len({json.dumps(s, sort_keys=True)
                                                            for s in stamps.values()}) == 1:
        return True
    detail = "\n".join(f"    recon_{v}: {s if s else 'UNSTAMPED (older run)'}"
                       for v, s in stamps.items())
    msg = (f"{subj} [{method}]: clean and breath recons are from DIFFERENT runs — cost_psnr would "
           f"subtract two checkpoints:\n{detail}\n"
           f"  -> re-run run_vggt with --arms clean breath, or set ALLOW_MIXED_ARMS=1 to score anyway.")
    if os.environ.get("ALLOW_MIXED_ARMS") == "1":
        print(f"  !! ALLOW_MIXED_ARMS=1: {msg}", flush=True)
        return False
    raise RuntimeError(msg)


def _save_nifti(arr_txyz, affine, path):
    """Atomic NIfTI write ((T,X,Y,Z) array -> (X,Y,Z,T) on disk) through the overwrite guard."""
    path = _guarded_write_path(path)
    tmp = f"{path}.tmp{os.getpid()}.nii.gz"
    nib.save(nib.Nifti1Image(np.moveaxis(arr_txyz, 0, -1), affine), tmp)
    os.replace(tmp, path)


def score_subject(ds, subj, method):
    """Score one (subject, method); returns the metrics dict (also written to metrics.json)."""
    manifest = json.load(open(paths.manifest(ds, subj)))
    T = manifest["T"]
    # Scoring ROI = GT whole-heart seg (dilated +-1 plane) INTERSECT native-FOV mask: the dilation
    # spills onto zero-padded edge planes with no acquired data; intersecting with the native FOV
    # drops those no-data planes -> honest metric.
    shape_xyz, aff = subject_grid(ds, subj)
    D = shape_xyz[2]
    content = load_canon(str(paths.fov_mask(ds, subj)), shape_xyz, aff) > 0.5
    has_heart = os.path.exists(paths.heart_mask(ds, subj))
    heart = load_canon(str(paths.heart_mask(ds, subj)), shape_xyz, aff) > 0.5 if has_heart else content
    mask = heart & content
    print(f"{subj} [{method}]: T={T} D={D}  scoring ROI={'heart&FOV' if has_heart else 'FOV only'}  "
          f"mask_voxels={int(mask.sum())}")

    # Breathing magnitude actually applied (frozen in the manifest; identical across all methods).
    disp = np.asarray(manifest["breath"]["disp_dhw_mm"], dtype=np.float64)   # (D,3): d_Z, d_Y, d_X
    disp_mag = np.linalg.norm(disp, axis=1)      # tilt-invariant vector magnitude per plane (mm)
    dz = np.abs(disp[:, 0])

    gt = np.stack([load_canon(str(paths.bundle_stack(ds, subj, "gt", t)), shape_xyz, aff)
                   for t in range(T)])

    metrics = {"subject": subj, "planes": list(range(D)), "D": int(D),
               "dz_mm": float(abs(aff[2, 2])), "scoring_roi": "heart&FOV" if has_heart else "FOV",
               "breath_mean_disp_mm": float(disp_mag.mean()), "breath_max_disp_mm": float(disp_mag.max()),
               "breath_mean_dz_mm": float(dz.mean()), "breath_max_dz_mm": float(dz.max()),
               "breath_disp_per_plane_mm": disp_mag.tolist(), "per_phase": {},
               "scorer": "image_metrics.py", "pose": "none", "psf": "none"}
    # `clean` is opt-in (run_vggt --arms; default is breath only, the deliverable). Score whichever
    # arms exist; `breath` is required.
    present = [v for v in ("clean", "breath") if paths.recon_dir(ds, subj, method, v).is_dir()]
    if "breath" not in present:
        raise FileNotFoundError(f"{subj} [{method}]: no recon_breath — that arm is the deliverable")
    metrics["arms"] = present
    metrics["stamps_agree"] = check_variant_stamps(ds, subj, method, present)

    for var in present:
        rec = np.stack([load_canon(str(paths.recon(ds, subj, method, var, t)), shape_xyz, aff)
                        for t in range(T)])
        rec = prep_recon(rec, method, mask)
        rec = rec * content[None]        # zero the recon in no-data planes (score mask ⊆ content,
        #                                  and GT is already 0 there — display consistency only)
        pv, sv, nv, uv = [], [], [], []
        for t in range(T):
            pv.append(psnr(rec[t], gt[t], mask)); sv.append(ssim(rec[t], gt[t], mask))
            nv.append(ncc(rec[t], gt[t], mask)); uv.append(psnr_unit_peak(rec[t], gt[t], mask))
        metrics["per_phase"][var] = {"psnr": pv, "ssim": sv, "ncc": nv, "psnr_unit_peak": uv}
        metrics[f"{var}_psnr_mean"] = float(np.nanmean(pv))   # nanmean so a degenerate phase drops
        metrics[f"{var}_ssim_mean"] = float(np.nanmean(sv))   # consistently across all metrics
        metrics[f"{var}_ncc_mean"] = float(np.nanmean(nv))
        metrics[f"{var}_psnr_unit_peak_mean"] = float(np.nanmean(uv))
        print(f"  {var}: PSNR {np.nanmean(pv):.2f} dB (unit-peak {np.nanmean(uv):.2f})  "
              f"SSIM {np.nanmean(sv):.3f}  NCC {np.nanmean(nv):.3f}")
        _save_nifti(rec, aff, paths.cine(ds, subj, method, var))

    # Shared 4D GT cine: deterministic from the read-only gt_t* files. Refresh when missing OR
    # older than the bundle's gt_t00 — a plain skip-if-exists went permanently stale whenever a
    # bundle was rebuilt (has happened: the native-z rebuild), showing viz/seg consumers a GT
    # different from the one scored. The write bypasses _guarded_write_path deliberately: the
    # content is derived, byte-reproducible, and staler-than-source — the one legitimate refresh
    # of a pre-existing file. tmp+os.replace keeps it atomic, which also makes two arms of the
    # same subject scoring in parallel (render_all_gifs -P4) a benign identical overwrite
    # instead of a race.
    cgt = paths.cine_gt(ds, subj)
    gt0_mtime = os.path.getmtime(paths.bundle_stack(ds, subj, "gt", 0))
    if not cgt.exists() or os.path.getmtime(cgt) < gt0_mtime:
        tmp = f"{cgt}.tmp{os.getpid()}.nii.gz"
        nib.save(nib.Nifti1Image(np.moveaxis(gt, 0, -1), aff), tmp)
        os.replace(tmp, cgt)

    # Provenance: tie this metrics.json to the recon it scored (aggregate's mix checks).
    meta_path = str(paths.metadata(ds, subj, method))
    arm_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    recon_files = [str(paths.recon(ds, subj, method, v, t)) for v in ("clean", "breath") for t in range(T)]
    metrics["method"] = method
    metrics["ckpt"] = arm_meta.get("ckpt")
    metrics["ckpt_fingerprint"] = arm_meta.get("ckpt_fingerprint")
    metrics["regime"] = arm_meta.get("regime")
    metrics["git_commit"] = arm_meta.get("git_commit")
    metrics["recon_mtime"] = max((os.path.getmtime(f) for f in recon_files if os.path.exists(f)), default=0.0)

    out = _guarded_write_path(paths.metrics(ds, subj, method))
    json.dump(metrics, open(out, "w"), indent=2)
    print(f"done -> {out}")
    return metrics


def main():
    ds = os.environ.get("EVAL_DATASET", "cmrx2024")
    if len(sys.argv) < 3:
        sys.exit("usage: EVAL_DATASET=<ds> python image_metrics.py <subject> <method>")
    score_subject(ds, sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
