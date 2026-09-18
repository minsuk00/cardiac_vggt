"""Per-subject IMAGE metrics — resample a method's recons to the subject grid, score vs GT.

Successor to the scoring half of _archive/assemble_and_gif.py (the metric/gauge functions below
are copied VERBATIM from it so this dir stands alone); rendering lives in analysis/viz.py.
Other metric families live beside this file: ef_dice.py (function/seg), aggregate.py (folds
image + EF/Dice + breathing resp_diag + timing into ONE metric_results/<split>/<ds>/<arm>.json).

READ-ONLY CONTRACT: every input (bundles, recons, stamps) is opened read-only. This script
writes ONLY its own outputs:
    <subject>/<method>/metrics.json
    <subject>/<method>/cine_{clean,breath}.nii.gz
    <subject>/cine_gt.nii.gz                      (only if absent — never rewritten)
`_guarded_write_path` enforces this structurally: a path bug crashes instead of clobbering.
(The pre-restructure assemble_and_gif records were archived to <arm>/_old_scorer/ first, so
these clean names collide with nothing.)

Every arm is loaded through pose_psf.py (docs/83): PSF-blurred on its own grid when the method
is a deconvolver (`pose_psf.PSF_METHODS`), then 6-DOF rigid-registered to GT per phase by NCC
inside the scoring ROI, then resampled onto the subject grid. That registered read is the
headline (`{var}_psnr_mean` ...) and is what `cine_<var>` holds. The plain unregistered,
unblurred read is scored alongside as `{var}_*_raw_mean` so the gauge penalty each method
would otherwise pay stays visible. metrics.json records the per-phase poses.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import pose_psf  # noqa: E402

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


def norm_map(rec, method, content, stacks):
    """The self-norm two-point map (rlo, rhi, slo, shi) for a self-norm method, from the RAW
    (unregistered, unblurred) recon's own coverage — see prep_recon. None for other methods.
    Computed once per variant and applied to both the raw and the registered read, so they share
    one gauge: deriving it from the PSF-blurred volume instead was measured to inflate NeSVoR
    1.6x (the blur spreads coverage into bright non-heart stack voxels: stack p99.9 0.49->0.69,
    PSNR 19.9 -> 15.2 dB at P028) while pose-only sat at 21.7 dB."""
    if pose_psf.base_method(method) not in SELF_NORM_METHODS:
        return None
    if stacks is None:   # a missed call site must crash, not silently score on the wrong scale
        raise ValueError(f"norm_map: {method} is a self-norm method and needs its input stacks")
    rec = np.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)
    cov = ((rec.max(axis=0) if rec.ndim == 4 else rec) > 1e-6) & content
    if not cov.any():                       # recon has no data in the FOV: nothing to anchor
        return None                         # against; prep_recon then just clamps.
    rv = rec[:, cov] if rec.ndim == 4 else rec[cov]
    sv = stacks[:, cov] if stacks.ndim == 4 else stacks[cov]
    rlo, rhi = np.percentile(rv, [0.5, 99.9])
    slo, shi = np.percentile(sv, [0.5, 99.9])
    return float(rlo), float(rhi), float(slo), float(shi)


def prep_recon(rec, method, content, stacks=None, nmap=None):
    """Bring a method's recon onto the GT [0,1] scale for scoring, per-method (docs/88):
      - scale-preserving (SVRTK, VGGT): clip the -1 sentinel, score AS-IS.
      - self-norm methods (NeSVoR, NiftyMIC): ONE uniform two-point map anchoring the recon to
        its own INPUT stacks — (recon_p0.5, recon_p99.9) → (stack_p0.5, stack_p99.9), percentiles
        over the recon's coverage region (∩ content FOV), then clamp[0,1].
    The anchor is GT-free (the stacks are data the method already received) and restores exactly
    the input scale the method discarded. It replaces the old recon-only self-norm, which was
    REFUTED (docs/88 §2): the baselines reconstruct a heart-centered crop (~6% of the FOV), so a
    recon-only percentile maps heart-max→1.0 while GT's heart peaks at ~0.35 of the FOV-normalized
    scale → NeSVoR ~3× too bright, 6 dB PSNR at healthy NCC. For a no-offset method the two-point
    map degenerates to pure scale (both floors ≈0, measured 20.60≈20.51 dB), so no divide-only
    special case. One global map over all phases (keeps real phase-to-phase contrast). For a
    scale-INVARIANT read that sidesteps gauges entirely, use the ncc() metric.
    `nmap` = a precomputed norm_map() (from the raw read); when omitted it is derived from `rec`."""
    rec = np.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)  # harden: a NaN/Inf in the recon would
    # else make np.percentile return NaN and silently poison this method's whole PSNR/SSIM/NCC mean.
    if pose_psf.base_method(method) not in SELF_NORM_METHODS:
        return clip_sentinel(rec)
    if nmap is None:
        nmap = norm_map(rec, method, content, stacks)
    if nmap is None:                        # recon has no data in the FOV: nothing to anchor
        return np.clip(rec, 0.0, 1.0)      # against; scoring then reflects the empty volume.
    rlo, rhi, slo, shi = nmap
    return np.clip((rec - rlo) * (shi - slo) / max(rhi - rlo, 1e-6) + slo, 0.0, 1.0)


def psnr(a, b, m):
    """Harness PSNR: peak = the GT's max INSIDE the ROI.

    ⚠️ NOT comparable to the trainer's `metric_psnr_3d_*`, which uses peak = 1.0. The ROI is the
    same object (`heart_roi_canonical`); only the normalization differs, by exactly
    `20*log10(gt[roi].max())`. Kept under its historical key for continuity with archived JSONs;
    since docs/106 the HEADLINE is `psnr_unit_peak` below (peak 1.0 = the field convention —
    fastMRI / CMRxRecon use the full-image GT max, never an ROI-restricted max).
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
    """Standard windowed SSIM (Wang et al. 2004: 11x11 Gaussian window, sigma 1.5), computed 2D
    per SAX slice and averaged over the ROI voxels of all slices. 2D, not 3D: the grid is 1.4 mm
    in-plane but 12 mm through-plane, so a 3D window would span ~80 mm in z. `data_range` is
    pinned to 1.0 rather than derived from the data: a data-derived range makes c1/c2
    method-dependent (one bright outlier widens it and *raises* SSIM while PSNR falls); the
    inputs are normalized to [0,1] by construction, so the fixed range is the honest one.
    (Replaced the harness's earlier single global-statistics SSIM over the ROI pool, which was
    not comparable to the standard definition and blind to local blur/misalignment.)"""
    from skimage.metrics import structural_similarity
    vals = []
    for z in range(a.shape[2]):
        mz = m[:, :, z]
        if not mz.any():
            continue
        smap = structural_similarity(a[:, :, z].astype(np.float64), b[:, :, z].astype(np.float64),
                                     data_range=1.0, gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False, full=True)[1]
        vals.append(smap[mz])
    if not vals:
        return float("nan")
    return float(np.concatenate(vals).mean())


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


def ensure_cine_gt(ds, subj, gt=None, aff=None):
    """Write `<subject>/cine_gt.nii.gz` (+ its gt_sha256 sidecar) if missing or stale; returns gt_sha.

    Shared 4D GT cine: deterministic from the read-only gt_t* files. Refresh when missing OR
    derived from a different gt bundle — a plain skip-if-exists went permanently stale whenever a
    bundle was rebuilt (has happened: the native-z rebuild), showing viz/seg consumers a GT
    different from the one scored. Freshness is keyed on the gt_t00 CONTENT hash recorded in the
    cine_gt.src.json sidecar, not on mtimes (GPFS purge-avoidance `touch`es rewrite every mtime).
    The write bypasses _guarded_write_path deliberately: the content is derived,
    byte-reproducible, and staler-than-source — the one legitimate refresh of a pre-existing
    file. tmp+os.replace keeps it atomic, which also makes two arms of the same subject scoring
    in parallel (render_all_gifs -P4) a benign identical overwrite instead of a race. Cine first,
    sidecar second: a sidecar that matches implies the cine beside it is already current.

    `gt`/`aff` may be passed by score_subject (already loaded); otherwise (ef_dice --gt-only)
    they are read from the bundle here."""
    cgt, cgt_src = paths.cine_gt(ds, subj), paths.cine_gt_src(ds, subj)
    gt_sha = paths.gt_sha256(ds, subj)
    try:
        cgt_fresh = cgt.exists() and json.load(open(cgt_src)).get("gt_sha256") == gt_sha
    except (OSError, json.JSONDecodeError):
        cgt_fresh = False
    if not cgt_fresh:
        if gt is None:
            shape_xyz, aff = subject_grid(ds, subj)
            T = json.load(open(paths.manifest(ds, subj)))["T"]
            gt = np.stack([load_canon(str(paths.bundle_stack(ds, subj, "gt", t)), shape_xyz, aff)
                           for t in range(T)])
        tmp = f"{cgt}.tmp{os.getpid()}.nii.gz"
        nib.save(nib.Nifti1Image(np.moveaxis(gt, 0, -1), aff), tmp)
        os.replace(tmp, cgt)
        tmp = f"{cgt_src}.tmp{os.getpid()}"
        json.dump({"gt_sha256": gt_sha}, open(tmp, "w"), indent=2)
        os.replace(tmp, cgt_src)
    return gt_sha


def score_subject(ds, subj, method):
    """Score one (subject, method); returns the metrics dict (also written to metrics.json)."""
    manifest = json.load(open(paths.manifest(ds, subj)))
    T = manifest["T"]
    # Scoring ROI = GT whole-heart seg (dilated +-1 plane), padded +10mm in-plane (docs/107),
    # INTERSECT native-FOV mask: the dilation spills onto zero-padded edge planes with no
    # acquired data; intersecting with the native FOV drops those no-data planes -> honest metric.
    # Same mask_heart_pad10 as the segmentation crop / baseline recon mask -- one ROI everywhere.
    shape_xyz, aff = subject_grid(ds, subj)
    D = shape_xyz[2]
    content = load_canon(str(paths.fov_mask(ds, subj)), shape_xyz, aff) > 0.5
    has_heart = os.path.exists(paths.heart_mask_pad(ds, subj))
    heart = load_canon(str(paths.heart_mask_pad(ds, subj)), shape_xyz, aff) > 0.5 if has_heart else content
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
               "breath_disp_per_plane_mm": disp_mag.tolist(), "per_phase": {}, "poses": {},
               "scorer": "image_metrics.py", "pose": "rigid6_ncc_per_phase",
               "psf": "gauss_thickness" if pose_psf.needs_psf(method) else "none"}
    # `clean` is opt-in (run_vggt --arms; default is breath only, the deliverable). Score whichever
    # arms exist; `breath` is required.
    present = [v for v in ("clean", "breath") if paths.recon_dir(ds, subj, method, v).is_dir()]
    if "breath" not in present:
        raise FileNotFoundError(f"{subj} [{method}]: no recon_breath — that arm is the deliverable")
    metrics["arms"] = present
    metrics["stamps_agree"] = check_variant_stamps(ds, subj, method, present)

    for var in present:
        thick = pose_psf.stamp_thickness(ds, subj, method, var) if pose_psf.needs_psf(method) else None
        # Registered read (headline): PSF-blur on the recon's own grid (deconvolvers only), fit a
        # 6-DOF pose per phase by NCC inside the scoring ROI, resample onto the subject grid.
        # Raw read: the plain anchored trilinear resample, no blur, no pose.
        rec, raw, poses = [], [], []
        for t in range(T):
            rp = paths.recon(ds, subj, method, var, t)
            vol_t, vaff = pose_psf.load_blurred(rp, thick, float(abs(aff[0, 0])))
            pose = pose_psf.fit_rigid(vol_t, vaff, gt[t], mask, aff)
            rec.append(pose_psf.resample(vol_t, vaff, shape_xyz, aff, pose))
            raw.append(load_canon(str(rp), shape_xyz, aff))
            poses.append(pose.record())
        metrics["poses"][var] = poses
        # Self-norm methods anchor to the input stacks the method ACTUALLY received: this
        # variant's gated stack, or scatter/ for a *_scatter arm (measured: breath vs scatter
        # heart p99.9 differ 5-7% -> a ±0.5 dB scale gauge if the wrong one is used). Loaded
        # lazily so scale-preserving arms (VGGT, SVRTK) never pay the I/O.
        stack_kind = "scatter" if method.endswith("_scatter") else var
        stacks = np.stack([load_canon(str(paths.bundle_stack(ds, subj, stack_kind, t)), shape_xyz, aff)
                           for t in range(T)]) if pose_psf.base_method(method) in SELF_NORM_METHODS else None
        raw = np.stack(raw)
        nmap = norm_map(raw, method, content, stacks)   # one gauge for both reads (see norm_map)
        per = {}
        for tag, vol in (("", np.stack(rec)), ("_raw", raw)):
            vol = prep_recon(vol, method, content, stacks=stacks, nmap=nmap)
            vol = vol * content[None]    # zero the recon in no-data planes (score mask ⊆ content,
            #                              and GT is already 0 there — display consistency only)
            pv, sv, nv, uv = [], [], [], []
            for t in range(T):
                pv.append(psnr(vol[t], gt[t], mask)); sv.append(ssim(vol[t], gt[t], mask))
                nv.append(ncc(vol[t], gt[t], mask)); uv.append(psnr_unit_peak(vol[t], gt[t], mask))
            per.update({f"psnr{tag}": pv, f"ssim{tag}": sv, f"ncc{tag}": nv, f"psnr_unit_peak{tag}": uv})
            metrics[f"{var}_psnr{tag}_mean"] = float(np.nanmean(pv))   # nanmean so a degenerate phase
            metrics[f"{var}_ssim{tag}_mean"] = float(np.nanmean(sv))   # drops consistently
            metrics[f"{var}_ncc{tag}_mean"] = float(np.nanmean(nv))
            metrics[f"{var}_psnr_unit_peak{tag}_mean"] = float(np.nanmean(uv))
            if tag == "":
                _save_nifti(vol, aff, paths.cine(ds, subj, method, var))
        metrics["per_phase"][var] = per
        print(f"  {var}: PSNR {metrics[f'{var}_psnr_mean']:.2f} dB (raw {metrics[f'{var}_psnr_raw_mean']:.2f})  "
              f"SSIM {metrics[f'{var}_ssim_mean']:.3f}  NCC {metrics[f'{var}_ncc_mean']:.3f} "
              f"(raw {metrics[f'{var}_ncc_raw_mean']:.3f})  "
              f"z-shift mm {[p['shift_mm_xyz'][2] for p in poses]}")

    gt_sha = ensure_cine_gt(ds, subj, gt, aff)

    # Provenance: tie this metrics.json to the recon it scored (aggregate's mix checks).
    meta_path = str(paths.metadata(ds, subj, method))
    arm_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    recon_files = [str(paths.recon(ds, subj, method, v, t)) for v in ("clean", "breath") for t in range(T)]
    metrics["method"] = method
    metrics["ckpt"] = arm_meta.get("ckpt")
    metrics["ckpt_fingerprint"] = arm_meta.get("ckpt_fingerprint")
    metrics["regime"] = arm_meta.get("regime")
    metrics["git_commit"] = arm_meta.get("git_commit")
    # Which gt bundle the cine_{clean,breath} beside this file were scored against; ef_dice.py
    # compares it to the live bundle instead of cine-vs-gt_t00 mtimes.
    metrics["gt_sha256"] = gt_sha
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
