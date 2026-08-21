"""Pose-gauge fit + PSF acquisition operator for scoring the classical SVR baselines (docs/83).

Two protocol defects, both measured in docs/83, are fixed here:
  (A) the free pose gauge — SVRTK/NeSVoR reconstruct in their own reference frame and float
      relative to GT (measured +4.0 / -4.0 mm in z on the probe subject, opposite signs!),
      while VGGT is anchored by construction (predicts off absolute scanner_coords);
  (B) the resampling operator — a GT voxel is the MRI signal integrated over the slice slab
      (8 mm thickness at 12 mm pitch for CMRx2024), but trilinear `load_canon` point-samples
      a thin plane. The PSF simulates the acquisition: Gaussian blur with FWHM = slice
      thickness through-plane / 1.2x voxel in-plane (Kuklisova-Murgasova 2012), THEN sample
      at the GT slice positions. The 4 mm gap needs no special handling — gap tissue enters
      only via the Gaussian tails, exactly as in the scanner.

Protocol (agreed, handoff 2026-08-20 §8 + docs/83 §3.3/§6):
  * ONE global rigid shift per (subject, arm, variant): 3-DOF translation. The corruption is
    pure translation (respiratory.py: "no rotation"), so translation-only is a defensible
    simplification; 6-DOF stays a future flag, escalate only if residuals demand it.
  * Fit OFF-METRIC: masked NCC on ONE phase (default t00), coarse grid init + Adam refine,
    then FREEZE and apply the same shift to all T phases. Never fit by maximizing the
    reported PSNR (docs/83 §3.3 — that yields an upper bound, not a fair number).
  * The shift is applied on the recon's own 1.4 mm grid (conceptually: blur + sample at
    shifted world points), NEVER by translating the 12 mm-pitch volume — a half-voxel blend
    there is a ~6 mm smoothing kernel (docs/83 §3.4).
  * Every arm goes through the SAME fitter (VGGT included — expected shift ~0; symmetry of
    treatment, not of result). PSF applies ONLY to the classical arms: VGGT's output is
    already thick-slice-native and blurrier than GT in z (docs/83 §4.4) — PSF-ing it would
    double-blur.
  * Per-subject thickness comes from the recon's stamp.json (docs/84 records it); pitch and
    grid come from the subject's own GT bundle (native-z: never hardcode 8/12).

This module is READ-ONLY over evaluation data: it loads bundles/recons/stamps and returns
arrays / fit results. The probe CLI writes its JSON to temp/ (gitignored), nothing else.
Hooking `pose_correct_load` into image_metrics.score_subject (three metric columns:
anchored / _psf / _posed) is a separate, later edit.

Probe (validation vs docs/83: SVRTK ~ +4 mm z, NeSVoR ~ -4 mm, VGGT ~ 0):
    micromamba run -n svr python evaluation/src/score/pose_psf.py \
        --dataset cmrx2024 --subject CMRx24_Test_P012 \
        --methods svrtk3d nesvor vggt_augaggr224hw2_ep300
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import image_metrics as im  # noqa: E402  (reuses masks/metrics; imports only, never edits)

# Arms whose output is an anatomy ESTIMATE on an isotropic grid -> gets the PSF operator.
# VGGT (and FC-SVR) output on the thick GT grid already and are excluded (docs/83 §4.4).
CLASSICAL_METHODS = ("svrtk3d", "nesvor", "niftymic")

INPLANE_FWHM_FACTOR = 1.2          # in-plane FWHM = 1.2 x in-plane voxel (Kuklisova-Murgasova)
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))   # sigma = FWHM / 2.3548


def is_classical(method):
    return method in CLASSICAL_METHODS


# --- geometry helpers (diagonal, axis-aligned affines only) -----------------
def _diag_zooms(affine):
    """Voxel sizes of a diagonal affine; crash on rotation/shear (none exist on disk — every
    recon/GT affine is diagonal in one shared world frame; a rotated one means new data that
    this simplification does not cover)."""
    R = np.asarray(affine, dtype=np.float64)[:3, :3]
    zooms = np.abs(np.diag(R))
    if not np.allclose(R, np.diag(np.diag(R)), atol=1e-3 * max(zooms.max(), 1.0)):
        raise ValueError(f"non-diagonal affine (rotation/shear) unsupported by pose_psf:\n{R}")
    return zooms


def _load_vol(path):
    """NIfTI -> (float32 (X,Y,Z) array, affine), NaN/Inf zeroed, SVRTK's -1 outside-mask
    sentinel clipped to 0 (it is a flag, not intensity — same treatment as clip_sentinel;
    left unclipped it would bleed negative rings through the Gaussian blur)."""
    img = nib.load(str(path))
    arr = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, 0.0, None), np.asarray(img.affine, dtype=np.float64)


def _to_tensor(arr_xyz):
    """(X,Y,Z) -> (1,1,Z,Y,X) so grid_sample's (x,y,z) grid convention indexes (W,H,D)=(X,Y,Z)."""
    return torch.from_numpy(np.ascontiguousarray(arr_xyz.transpose(2, 1, 0)))[None, None]


def psf_sigma_mm(thickness_mm, inplane_mm):
    """World-frame Gaussian sigmas (x, y, z): FWHM = 1.2 x in-plane voxel in-plane, slice
    THICKNESS (not pitch!) through-plane. 8 is thickness, 12 is pitch — never conflate
    (docs/27)."""
    return (INPLANE_FWHM_FACTOR * inplane_mm * _FWHM_TO_SIGMA,
            INPLANE_FWHM_FACTOR * inplane_mm * _FWHM_TO_SIGMA,
            float(thickness_mm) * _FWHM_TO_SIGMA)


def _gauss_kernel(sigma_vox):
    if sigma_vox < 0.25:               # sub-quarter-voxel blur is numerically a no-op
        return None
    r = max(1, int(math.ceil(3.0 * sigma_vox)))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    return k / k.sum()


def blur_psf(vol_t, vol_affine, thickness_mm):
    """Separable Gaussian PSF on the volume's own grid. Each thick voxel later sampled from
    this is a Gaussian-weighted average of ALL fine voxels through the slab (deterministic
    convolution, not sampling). Kernel is shift-independent, so it is applied ONCE outside
    the registration loop (blur commutes with translation) and gradients flow through the
    subsequent grid_sample untouched."""
    zx, zy, zz = _diag_zooms(vol_affine)
    sx, sy, sz = psf_sigma_mm(thickness_mm, inplane_mm=float(min(zx, zy)))
    out = vol_t
    # tensor layout (1,1,Z,Y,X): dim 2 = Z (world z), 3 = Y, 4 = X. Zero padding = background.
    for dim, sig_vox in ((2, sz / zz), (3, sy / zy), (4, sx / zx)):
        k = _gauss_kernel(sig_vox)
        if k is None:
            continue
        shape = [1, 1, 1, 1, 1]
        shape[dim] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = pad[2 * (4 - dim) + 1] = k.numel() // 2
        out = F.conv3d(F.pad(out, pad), k.reshape(shape))
    return out


def _world_points(shape_xyz, affine):
    """World coords (mm) of every voxel center of a diagonal-affine grid -> (X,Y,Z,3) tensor."""
    zooms = np.diag(np.asarray(affine)[:3, :3])
    orig = np.asarray(affine)[:3, 3]
    axes = [orig[i] + zooms[i] * np.arange(shape_xyz[i], dtype=np.float64) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return torch.from_numpy(np.stack([gx, gy, gz], axis=-1).astype(np.float32))


def _sample_world(vol_t, vol_affine, pts_world, shift_mm):
    """Sample `vol_t` (its grid described by `vol_affine`) at `pts_world - shift_mm`.

    Shifting the volume by +s means vol'(x) = vol(x - s), so the correction "recon sits +4 mm
    off GT" is undone by shift_mm = -(that float) — the fitter's sign convention is
    "shift applied TO the recon". Differentiable in shift_mm (a (3,) tensor). Points outside
    the recon's own FOV sample 0 (background). pts_world: (..., 3); returns (...)."""
    zooms = torch.tensor(np.diag(np.asarray(vol_affine)[:3, :3]).copy(), dtype=torch.float32)
    orig = torch.tensor(np.asarray(vol_affine)[:3, 3].copy(), dtype=torch.float32)
    n = torch.tensor([vol_t.shape[4], vol_t.shape[3], vol_t.shape[2]], dtype=torch.float32)  # (X,Y,Z)
    idx = (pts_world - shift_mm - orig) / zooms                     # continuous voxel index (x,y,z)
    grid = 2.0 * idx / torch.clamp(n - 1.0, min=1.0) - 1.0          # align_corners=True normalization
    flat = grid.reshape(1, -1, 1, 1, 3)
    out = F.grid_sample(vol_t, flat, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.reshape(pts_world.shape[:-1])


def _masked_ncc_t(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std().clamp_min(1e-8) * b.std().clamp_min(1e-8))


# --- the two public operations ---------------------------------------------
def fit_shift(vol_t, vol_affine, gt_xyz, mask_xyz, gt_affine,
              coarse_mm=8.0, coarse_step_mm=2.0, refine_steps=150, refine_lr=0.3):
    """Fit the 3-DOF world-frame shift (mm) of one recon against ONE GT phase.

    Off-metric by construction: the objective is masked NCC (affine-intensity-invariant, so
    no gauge normalization needed and it is NOT the reported PSNR). Coarse +-`coarse_mm` grid
    search (NCC over +-8 mm is not convex; Adam from zero can stall) then Adam sub-voxel
    refinement. `vol_t` must already be PSF-blurred for classical arms (blur commutes with
    the shift). Returns (shift_mm (3,) np.float64, ncc_at_zero, ncc_at_fit)."""
    pts = _world_points(gt_xyz.shape, gt_affine)[torch.from_numpy(mask_xyz)]   # (Nmask, 3)
    gt_vals = torch.from_numpy(gt_xyz[mask_xyz].astype(np.float32))

    def ncc_of(shift):
        return _masked_ncc_t(_sample_world(vol_t, vol_affine, pts, shift), gt_vals)

    with torch.no_grad():
        ncc0 = float(ncc_of(torch.zeros(3)))
        steps = torch.arange(-coarse_mm, coarse_mm + 1e-6, coarse_step_mm)
        best, best_ncc = torch.zeros(3), ncc0
        for sx in steps:
            for sy in steps:
                for sz in steps:
                    cand = torch.tensor([sx, sy, sz])
                    v = float(ncc_of(cand))
                    if v > best_ncc:
                        best, best_ncc = cand, v
    shift = best.clone().requires_grad_(True)
    opt = torch.optim.Adam([shift], lr=refine_lr)
    for _ in range(refine_steps):
        opt.zero_grad()
        loss = -ncc_of(shift)
        loss.backward()
        opt.step()
    with torch.no_grad():
        fit = shift.detach()
        # keep the refinement honest: if Adam somehow wandered below the coarse optimum, keep coarse
        if float(ncc_of(fit)) < best_ncc:
            fit = best
        return fit.numpy().astype(np.float64), ncc0, float(ncc_of(fit))


def apply_and_downsample(path, shape_xyz, gt_affine, shift_mm, thickness_mm, apply_psf):
    """`load_canon` drop-in for one phase volume: load the recon on ITS OWN grid, optionally
    PSF-blur there (classical arms), then sample at the (shifted) GT voxel centers.
    shift_mm=(0,0,0) + apply_psf=True is the `_psf` column; fitted shift is `_posed`.
    Returns float32 (X,Y,Z) on the subject grid."""
    arr, vol_aff = _load_vol(path)
    vol_t = _to_tensor(arr)
    if apply_psf:
        vol_t = blur_psf(vol_t, vol_aff, thickness_mm)
    pts = _world_points(shape_xyz, gt_affine)
    with torch.no_grad():
        out = _sample_world(vol_t, vol_aff, pts, torch.tensor(shift_mm, dtype=torch.float32))
    return out.numpy().astype(np.float32)


def stamp_thickness(ds, subj, method, variant):
    """Slice thickness (mm) from the recon's stamp.json (docs/84 records it per subject).
    No default: a classical arm without a recorded thickness must crash, not silently get 8."""
    p = paths.recon_stamp(ds, subj, method, variant)
    stamp = json.load(open(p))
    return float(stamp["thickness_mm"])


def fit_subject_arm(ds, subj, method, variant="breath", fit_phase=0):
    """Full per-(subject, arm) fit: masks + GT from the bundle, thickness from the stamp,
    PSF for classical arms only, NCC fit on `fit_phase` alone (frozen for the other T-1).
    Returns a plain dict (no writes)."""
    shape_xyz, aff = im.subject_grid(ds, subj)
    content = im.load_canon(str(paths.fov_mask(ds, subj)), shape_xyz, aff) > 0.5
    has_heart = os.path.exists(paths.heart_mask(ds, subj))
    heart = im.load_canon(str(paths.heart_mask(ds, subj)), shape_xyz, aff) > 0.5 if has_heart else content
    mask = heart & content
    gt_fit = im.load_canon(str(paths.bundle_stack(ds, subj, "gt", fit_phase)), shape_xyz, aff)

    classical = is_classical(method)
    arr, vol_aff = _load_vol(paths.recon(ds, subj, method, variant, fit_phase))
    vol_t = _to_tensor(arr)
    thickness = stamp_thickness(ds, subj, method, variant) if classical else None
    if classical:
        vol_t = blur_psf(vol_t, vol_aff, thickness)
    shift, ncc0, ncc1 = fit_shift(vol_t, vol_aff, gt_fit, mask, aff)
    return {"subject": subj, "method": method, "variant": variant, "fit_phase": fit_phase,
            "dof": 3, "objective": "masked_ncc", "classical": classical,
            "psf": f"gauss_thickness{thickness:g}mm" if classical else "none",
            "shift_mm_xyz": [round(float(v), 3) for v in shift],
            "ncc_fitphase_anchored": round(ncc0, 5), "ncc_fitphase_posed": round(ncc1, 5)}


# --- probe CLI (validation vs docs/83; writes ONLY to temp/) ----------------
def _probe_columns(ds, subj, method, variant, shift_mm, thickness, mask, content, gt, shape_xyz, aff):
    """The three agreed metric columns on all T phases: anchored (trilinear, shift 0 — the
    current scorer), _psf (PSF operator only, still anchored), _posed (PSF + fitted shift).
    Non-classical arms get no PSF in any column (their _psf == anchored by construction)."""
    classical = is_classical(method)
    T = gt.shape[0]
    stacks = np.stack([im.load_canon(str(paths.bundle_stack(ds, subj, variant, t)), shape_xyz, aff)
                       for t in range(T)]) if method in im.SELF_NORM_METHODS else None
    cols = {}
    for name, s, psf in (("anchored", (0., 0., 0.), False),
                         ("psf", (0., 0., 0.), classical),
                         ("posed", tuple(shift_mm), classical)):
        rec = np.stack([
            im.load_canon(str(paths.recon(ds, subj, method, variant, t)), shape_xyz, aff)
            if (not psf and s == (0., 0., 0.)) else
            apply_and_downsample(paths.recon(ds, subj, method, variant, t), shape_xyz, aff,
                                 s, thickness, psf)
            for t in range(T)])
        rec = im.prep_recon(rec, method, content, stacks=stacks)
        cols[name] = {
            "psnr": round(float(np.nanmean([im.psnr(rec[t], gt[t], mask) for t in range(T)])), 3),
            "ncc": round(float(np.nanmean([im.ncc(rec[t], gt[t], mask) for t in range(T)])), 4),
        }
    return cols


def main():
    ap = argparse.ArgumentParser(description="pose/PSF probe — read-only over evaluation data")
    ap.add_argument("--dataset", default="cmrx2024")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--variant", default="breath", choices=list(paths.VARIANTS))
    ap.add_argument("--fit-phase", type=int, default=0)
    ap.add_argument("--out", default=None, help="output JSON (default temp/pose_psf_probe/...)")
    args = ap.parse_args()

    ds, subj = args.dataset, args.subject
    shape_xyz, aff = im.subject_grid(ds, subj)
    content = im.load_canon(str(paths.fov_mask(ds, subj)), shape_xyz, aff) > 0.5
    has_heart = os.path.exists(paths.heart_mask(ds, subj))
    heart = im.load_canon(str(paths.heart_mask(ds, subj)), shape_xyz, aff) > 0.5 if has_heart else content
    mask = heart & content
    T = json.load(open(paths.manifest(ds, subj)))["T"]
    gt = np.stack([im.load_canon(str(paths.bundle_stack(ds, subj, "gt", t)), shape_xyz, aff)
                   for t in range(T)])

    results = []
    for method in args.methods:
        r = fit_subject_arm(ds, subj, method, args.variant, args.fit_phase)
        thickness = stamp_thickness(ds, subj, method, args.variant) if r["classical"] else None
        r["columns"] = _probe_columns(ds, subj, method, args.variant, r["shift_mm_xyz"],
                                      thickness, mask, content, gt, shape_xyz, aff)
        c = r["columns"]
        print(f"{method:34s} shift(x,y,z)mm={tuple(r['shift_mm_xyz'])} psf={r['psf']}\n"
              f"  PSNR anchored {c['anchored']['psnr']:6.2f} -> psf {c['psf']['psnr']:6.2f} "
              f"-> posed {c['posed']['psnr']:6.2f}   "
              f"NCC {c['anchored']['ncc']:.4f} -> {c['psf']['ncc']:.4f} -> {c['posed']['ncc']:.4f}",
              flush=True)
        results.append(r)

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[3] / "temp" / "pose_psf_probe" / f"{ds}_{subj}_{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": ds, "fit_phase": args.fit_phase, "results": results},
              open(out, "w"), indent=2)
    print(f"probe -> {out}")


if __name__ == "__main__":
    main()
