"""Pose registration + PSF acquisition operator for scoring recon volumes against GT (docs/83).

Two protocol defects, both measured in docs/83, are fixed here:
  (A) the free pose gauge — SVRTK/NeSVoR reconstruct in their own reference frame and float
      relative to GT (measured +4.0 / -4.0 mm in z on the probe subject, opposite signs!),
      while VGGT is anchored by construction (predicts off absolute scanner_coords). A global
      rotation is as invisible to a slice-consistency objective as a global translation, so
      the gauge is 6-DOF rigid.
  (B) the resampling operator — a GT voxel is the MRI signal integrated over the slice slab
      (8 mm thickness at 12 mm pitch for CMRx2024), but a trilinear read point-samples a thin
      plane. The PSF simulates the acquisition: Gaussian blur with FWHM = slice thickness
      through-plane / 1.2x voxel in-plane (Kuklisova-Murgasova 2012), THEN sample at the GT
      slice positions. The 4 mm gap needs no special handling — gap tissue enters only via
      the Gaussian tails, exactly as in the scanner.

Protocol (handoff 2026-09-11):
  * ONE rigid pose (6-DOF: translation mm + rotation deg about the mask centroid) per
    (subject, arm, variant, PHASE) — every phase is fit on its own because SVRTK/NeSVoR
    reconstruct each phase independently and can drift differently per phase.
  * Fit by masked NCC: coarse translation grid (+-8 mm, NCC is not convex over that range)
    then Adam on all 6 parameters. NCC is also the reported cross-method metric; 6 rigid
    numbers against ~1e5 masked voxels cannot fake alignment or hide content error.
  * The pose is applied by READING the recon at transformed GT voxel centers (after the PSF
    blur on the recon's own fine grid), never by translating the 12 mm-pitch volume — a
    half-voxel blend there is a ~6 mm smoothing kernel (docs/83 §3.4).
  * Every arm goes through the SAME fitter (VGGT included — expected pose ~0; symmetry of
    treatment, not of result). PSF applies ONLY to arms in PSF_METHODS: deconvolvers that
    output a sharp estimate on a fine grid. VGGT/FC-SVR output thick-slice-native content
    already blurrier than GT in z (docs/83 §4.4) — PSF-ing them would double-blur.
  * Per-subject thickness comes from the recon's stamp.json (docs/84 records it); pitch and
    grid come from the subject's own GT bundle (native-z: never hardcode 8/12).

Blur commutes with a rigid transform, so the PSF is applied once per volume and the fit just
reads the blurred volume at candidate poses.

Per-phase use (image_metrics.score_subject):
    vol_t, vol_aff = load_blurred(path, thickness_mm if needs_psf(method) else None, gt_inplane_mm)
    pose = fit_rigid(vol_t, vol_aff, gt_xyz, mask_xyz, gt_affine)      # Pose (params, center, nccs)
    rec_xyz = resample(vol_t, vol_aff, shape_xyz, gt_affine, pose)

Probe CLI (read-only; writes JSON to temp/):
    micromamba run -n svr python evaluation/src/score/pose_psf.py \
        --dataset cmrx2024 --subject CMRx24_Test_P012 --methods svrtk3d nesvor vggt_<slug>
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

# Arms whose output is a deconvolved anatomy ESTIMATE on a fine grid -> gets the PSF operator.
# Keyed on what the method outputs, not on classical-vs-learned (CiNeVol is learned and needs
# it; FC-SVR is learned and does not). An unlisted arm gets no PSF.
PSF_METHODS = ("svrtk3d", "nesvor", "niftymic", "fetal_cmr_4d", "cinevol")

INPLANE_FWHM_FACTOR = 1.2          # in-plane FWHM = 1.2 x in-plane voxel (Kuklisova-Murgasova)
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))   # sigma = FWHM / 2.3548


# All fitting/resampling runs here; results come back as numpy. GPU when present (a 6-DOF fit
# is ~3 s on 4 CPU cores, well under 1 s on an A40), CPU otherwise — identical math.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def base_method(method):
    """Arm name -> engine name: 'svrtk3d_scatter' / 'nesvor_scatter' (same-input arms, see
    run_svrtk3d.sh INPUT=scatter) and '*_debug' share their base method's PSF/gauge treatment."""
    for suf in ("_scatter", "_debug"):
        if method.endswith(suf):
            return method[: -len(suf)]
    return method


def needs_psf(method):
    return base_method(method) in PSF_METHODS


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
    """NIfTI -> (float32 (X,Y,Z) array, affine, valid (X,Y,Z) bool). NaN/Inf zeroed. SVRTK
    writes -1 outside its recon mask: a no-data flag, not intensity — those voxels are
    clipped to 0 AND excluded from `valid`, so the PSF averages only over acquired data
    (see blur_psf) instead of bleeding the zero exterior into the heart."""
    img = nib.load(str(path))
    arr = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = arr > -0.5
    return np.clip(arr, 0.0, None), np.asarray(img.affine, dtype=np.float64), valid


def _to_tensor(arr_xyz):
    """(X,Y,Z) -> (1,1,Z,Y,X) so grid_sample's (x,y,z) grid convention indexes (W,H,D)=(X,Y,Z)."""
    return torch.from_numpy(np.ascontiguousarray(arr_xyz.transpose(2, 1, 0)))[None, None].to(DEVICE)


def _world_points(shape_xyz, affine):
    """World coords (mm) of every voxel center of a diagonal-affine grid -> (X,Y,Z,3) tensor."""
    zooms = np.diag(np.asarray(affine)[:3, :3])
    orig = np.asarray(affine)[:3, 3]
    axes = [orig[i] + zooms[i] * np.arange(shape_xyz[i], dtype=np.float64) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return torch.from_numpy(np.stack([gx, gy, gz], axis=-1).astype(np.float32)).to(DEVICE)


# --- PSF -------------------------------------------------------------------
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
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=DEVICE)
    k = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    return k / k.sum()


def _gauss_blur(vol_t, sigma_vox_zyx):
    """Separable Gaussian over tensor dims (2,3,4)=(Z,Y,X), zero-padded."""
    out = vol_t
    for dim, sig_vox in zip((2, 3, 4), sigma_vox_zyx):
        k = _gauss_kernel(sig_vox)
        if k is None:
            continue
        shape = [1, 1, 1, 1, 1]
        shape[dim] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = pad[2 * (4 - dim) + 1] = k.numel() // 2   # F.pad: last dim first
        out = F.conv3d(F.pad(out, pad), k.reshape(shape))
    return out


def blur_psf(vol_t, vol_affine, thickness_mm, acq_inplane_mm, valid_t=None, min_coverage=0.05):
    """PSF on the volume's own grid: each thick voxel later sampled from this is a
    Gaussian-weighted average of the fine voxels through the slab. Sigmas: slice THICKNESS
    through-plane, 1.2 x the ACQUIRED in-plane voxel (`acq_inplane_mm`, the GT grid's — not
    the recon's own, which may be finer) in-plane.

    Normalized convolution: with `valid_t` (1 where the recon has data), the result is
    blur(vol*valid) / blur(valid), so voxels near the recon's mask edge / FOV border average
    only over acquired data instead of being pulled toward the zero exterior (measured
    without it: 10.6% of SVRTK's heart-ROI voxels darkened at P012). Where blurred coverage
    < `min_coverage` there is no data and the output is 0."""
    zx, zy, zz = _diag_zooms(vol_affine)
    sx, sy, sz = psf_sigma_mm(thickness_mm, inplane_mm=float(acq_inplane_mm))
    sig = (sz / zz, sy / zy, sx / zx)
    if valid_t is None:
        return _gauss_blur(vol_t, sig)
    num = _gauss_blur(vol_t * valid_t, sig)
    den = _gauss_blur(valid_t, sig)
    return torch.where(den > min_coverage, num / den.clamp_min(min_coverage), torch.zeros_like(num))


def load_blurred(path, thickness_mm=None, acq_inplane_mm=None):
    """Load one recon volume on ITS OWN grid; PSF-blur it there when `thickness_mm` is given
    (`acq_inplane_mm` = the GT grid's in-plane voxel, required with it).
    Returns (vol_t (1,1,Z,Y,X), affine) — the input to both fit_rigid and resample."""
    arr, aff, valid = _load_vol(path)
    vol_t = _to_tensor(arr)
    if thickness_mm is not None:
        if acq_inplane_mm is None:
            raise ValueError("load_blurred: PSF needs the acquired in-plane voxel (GT grid)")
        vol_t = blur_psf(vol_t, aff, thickness_mm, acq_inplane_mm, _to_tensor(valid.astype(np.float32)))
    return vol_t, aff


# --- rigid pose ------------------------------------------------------------
def _rotation(rot_deg):
    """Rz·Ry·Rx from Euler angles in degrees (differentiable), (3,) -> (3,3)."""
    rx, ry, rz = torch.deg2rad(rot_deg)
    cx, sx, cy, sy, cz, sz = rx.cos(), rx.sin(), ry.cos(), ry.sin(), rz.cos(), rz.sin()
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx]).reshape(3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy]).reshape(3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one]).reshape(3, 3)
    return Rz @ Ry @ Rx


def _transform(pts_world, pose, center):
    """Where in the RECON's world frame to read for each GT point: R^T(p - c) + c - t.
    pose = (tx,ty,tz [mm], rx,ry,rz [deg]); rotation about `center` (3,) mm. With zero rotation
    this is p - t, i.e. the posed recon is vol'(p) = vol(p - t): "shift applied TO the recon",
    so a recon sitting +4 mm off GT is corrected by t = -4."""
    R = _rotation(pose[3:])
    # pts @ R == R^T pts: reading the recon at R^T(p - c) + c - t means the POSED recon is the
    # original rotated by +R about c and translated by +t — one convention for both halves of
    # the pose, so the recorded (shift, rot) both describe the correction applied to the recon.
    return (pts_world - center) @ R + center - pose[:3]


def _sample(vol_t, vol_affine, pts_world):
    """Trilinear read of `vol_t` (grid described by `vol_affine`) at world points (..., 3).
    Points outside the recon's FOV sample 0 (background). Differentiable in pts_world."""
    zooms = torch.tensor(np.diag(np.asarray(vol_affine)[:3, :3]).copy(), dtype=torch.float32, device=DEVICE)
    orig = torch.tensor(np.asarray(vol_affine)[:3, 3].copy(), dtype=torch.float32, device=DEVICE)
    n = torch.tensor([vol_t.shape[4], vol_t.shape[3], vol_t.shape[2]], dtype=torch.float32, device=DEVICE)  # (X,Y,Z)
    idx = (pts_world - orig) / zooms                                # continuous voxel index (x,y,z)
    grid = 2.0 * idx / torch.clamp(n - 1.0, min=1.0) - 1.0          # align_corners=True normalization
    out = F.grid_sample(vol_t, grid.reshape(1, -1, 1, 1, 3), mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return out.reshape(pts_world.shape[:-1])


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).mean() / (a.std().clamp_min(1e-8) * b.std().clamp_min(1e-8))


class Pose:
    """One rigid pose: params (6,) = [tx,ty,tz mm, rx,ry,rz deg], rotation about `center` (mm).
    The center travels with the pose so fit and apply rotate about the same point."""

    def __init__(self, params, center, ncc_anchored=None, ncc_posed=None):
        self.params = np.asarray(params, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.ncc_anchored, self.ncc_posed = ncc_anchored, ncc_posed

    @classmethod
    def identity(cls, center):
        return cls(np.zeros(6), center)

    def record(self):
        """JSON-ready summary."""
        return {"shift_mm_xyz": [round(float(v), 3) for v in self.params[:3]],
                "rot_deg_xyz": [round(float(v), 3) for v in self.params[3:]],
                "center_mm_xyz": [round(float(v), 2) for v in self.center],
                "ncc_anchored": None if self.ncc_anchored is None else round(self.ncc_anchored, 5),
                "ncc_posed": None if self.ncc_posed is None else round(self.ncc_posed, 5)}


def _grid_search(ncc_of, start, axes, values):
    """Best of `start` with `axes` (param indices) replaced by every combination of `values`."""
    best, best_v = start.clone(), float(ncc_of(start))
    for combo in itertools.product(values, repeat=len(axes)):
        cand = start.clone()
        cand[list(axes)] = torch.tensor(combo, dtype=torch.float32, device=DEVICE)
        v = float(ncc_of(cand))
        if v > best_v:
            best, best_v = cand, v
    return best, best_v


def fit_rigid(vol_t, vol_affine, gt_xyz, mask_xyz, gt_affine,
              coarse_mm=8.0, coarse_step_mm=2.0, coarse_deg=8.0, coarse_step_deg=4.0,
              refine_steps=200, refine_lr=0.3):
    """Fit the 6-DOF rigid pose of one (already PSF-blurred, if applicable) recon volume
    against ONE GT phase by masked NCC. Rotation is about the mask centroid. Returns a Pose.

    Search (NCC over +-8 mm is not convex, so gradient descent alone can stall):
      1. translation grid, +-coarse_mm at coarse_step_mm, rotation 0        (9^3 = 729 evals)
      2. rotation grid at that translation, +-coarse_deg at coarse_step_deg (5^3 = 125 evals)
      3. Adam on all 6 (lr in mm / deg per step, cosine-decayed to ~0), keeping the best
         iterate seen — never worse than the grid optimum.

    The pose is BOUNDED to the search box (|t| <= coarse_mm, |r| <= coarse_deg): the gauge this
    corrects is a method's own reference-frame offset (measured <= ~6 mm, ~2 deg), while the
    simulated breathing displaces planes by up to ~20 mm. Unbounded, the fit on a subject whose
    breath moved half the planes chases whichever half dominates NCC at each phase (measured on
    CMRx24_Test_P020: pose flips between ~0 and ~(6,11,11) mm phase to phase) — that is
    uncorrected motion being credited as pose. The box keeps the correction a gauge fix."""
    if mask_xyz.sum() < 2:   # NCC and the centroid are undefined; a NaN centre would silently zero the resample
        raise ValueError(f"fit_rigid: scoring mask has {int(mask_xyz.sum())} voxel(s)")
    pts = _world_points(gt_xyz.shape, gt_affine)[torch.from_numpy(mask_xyz).to(DEVICE)]   # (Nmask, 3)
    center = pts.mean(0)
    gt_vals = torch.from_numpy(gt_xyz[mask_xyz].astype(np.float32)).to(DEVICE)

    def ncc_of(pose):
        return _ncc(_sample(vol_t, vol_affine, _transform(pts, pose, center)), gt_vals)

    zero = torch.zeros(6, device=DEVICE)
    with torch.no_grad():
        ncc0 = float(ncc_of(zero))
        mm = torch.arange(-coarse_mm, coarse_mm + 1e-6, coarse_step_mm).tolist()
        deg = torch.arange(-coarse_deg, coarse_deg + 1e-6, coarse_step_deg).tolist()
        best, best_ncc = _grid_search(ncc_of, zero, (0, 1, 2), mm)
        best, best_ncc = _grid_search(ncc_of, best, (3, 4, 5), deg)
    pose = best.clone().requires_grad_(True)
    opt = torch.optim.Adam([pose], lr=refine_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, refine_steps, eta_min=refine_lr * 0.02)
    for _ in range(refine_steps):
        opt.zero_grad()
        loss = -ncc_of(pose)
        loss.backward()
        val = -loss.item()                     # value at the CURRENT iterate (pre-step)
        if val > best_ncc:
            best, best_ncc = pose.detach().clone(), val
        opt.step()
        sched.step()
        with torch.no_grad():                  # stay inside the search box (see docstring)
            pose[:3].clamp_(-coarse_mm, coarse_mm)
            pose[3:].clamp_(-coarse_deg, coarse_deg)
    return Pose(best.cpu().numpy(), center.cpu().numpy(), ncc0, best_ncc)


def resample(vol_t, vol_affine, shape_xyz, gt_affine, pose):
    """Read the (blurred) recon at the posed GT voxel centers -> float32 (X,Y,Z) on the GT grid.
    `pose` is a Pose; Pose.identity(c) is the anchored (no-registration) read."""
    pts = _world_points(shape_xyz, gt_affine)
    params = torch.as_tensor(pose.params, dtype=torch.float32, device=DEVICE)
    center = torch.as_tensor(pose.center, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out = _sample(vol_t, vol_affine, _transform(pts, params, center))
    return out.cpu().numpy().astype(np.float32)


def stamp_thickness(ds, subj, method, variant):
    """Slice thickness (mm) from the recon's stamp.json (docs/84 records it per subject).
    No default: a PSF arm without a recorded thickness must crash, not silently get 8."""
    stamp = json.load(open(paths.recon_stamp(ds, subj, method, variant)))
    return float(stamp["thickness_mm"])


# --- probe CLI (validation vs docs/83; writes ONLY to temp/) ----------------
def main():
    import image_metrics as im   # local: image_metrics imports this module

    ap = argparse.ArgumentParser(description="pose/PSF probe — read-only over evaluation data")
    ap.add_argument("--dataset", default="cmrx2024")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--variant", default="breath", choices=list(paths.VARIANTS))
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--out", default=None, help="output JSON (default temp/pose_psf_probe/...)")
    args = ap.parse_args()

    ds, subj, t = args.dataset, args.subject, args.phase
    shape_xyz, aff = im.subject_grid(ds, subj)
    content = im.load_canon(str(paths.fov_mask(ds, subj)), shape_xyz, aff) > 0.5
    heart_p = paths.heart_mask(ds, subj)
    heart = im.load_canon(str(heart_p), shape_xyz, aff) > 0.5 if heart_p.exists() else content
    mask = heart & content
    gt = im.load_canon(str(paths.bundle_stack(ds, subj, "gt", t)), shape_xyz, aff)

    results = []
    for method in args.methods:
        thick = stamp_thickness(ds, subj, method, args.variant) if needs_psf(method) else None
        vol_t, vaff = load_blurred(paths.recon(ds, subj, method, args.variant, t), thick, float(abs(aff[0, 0])))
        pose = fit_rigid(vol_t, vaff, gt, mask, aff)
        r = {"method": method, "psf": f"gauss_thickness{thick:g}mm" if thick else "none",
             **pose.record()}
        print(f"{method:34s} shift(x,y,z)mm={r['shift_mm_xyz']} rot(deg)={r['rot_deg_xyz']} "
              f"psf={r['psf']}  NCC {pose.ncc_anchored:.4f} -> {pose.ncc_posed:.4f}", flush=True)
        results.append(r)

    out = Path(args.out) if args.out else \
        Path(__file__).resolve().parents[3] / "temp" / "pose_psf_probe" / f"{ds}_{subj}_{args.variant}_t{t:02d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": ds, "subject": subj, "variant": args.variant, "phase": t, "results": results},
              open(out, "w"), indent=2)
    print(f"probe -> {out}")


if __name__ == "__main__":
    main()
