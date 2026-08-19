"""Step 3/3 — resample a method's recons to the canonical grid, score vs GT, and render GT-vs-pred
montage GIFs (all z-planes, animated over the cardiac cycle).

Layout (see README "Directory layout"): <subject>/ holds the SHARED frozen bundle; each method writes
under <subject>/<method>/. This reads recons from <subject>/<method>/recon_{clean,breath}/ and writes:
  <subject>/cine_gt.nii.gz                        4D canonical GT (method-independent, shared, once)
  <subject>/<method>/cine_{clean,breath}.nii.gz   this method's recons on the canonical grid
  <subject>/<method>/metrics.json                 per-phase PSNR/SSIM (heart&FOV ROI), clean & breath
  <subject>/<method>/gif_{clean,breath,combined}.gif

Run: EVAL_DATASET=<ds> micromamba run -n svr python evaluation/engine/assemble_and_gif.py <subject> [method=svrtk3d]

Paths/naming go through evaluation/paths.py (the single source of truth).
"""
import json
import os
import re
import sys

import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

DATASET = os.environ.get("EVAL_DATASET", "cmrx2024")
INPLANE_MM = 1.4          # in-plane only; z is NEVER a constant here (docs/58 native-z)


def subject_grid(ds, subj):
    """This subject's own scoring grid `(shape_xyz, affine)`, read from its GT bundle.

    Native-z (docs/58): D and dz belong to the subject, so there is no single canonical grid to
    resample onto. This file used to hardcode `SHAPE_XYZ = (256,256,12)` / `(1.4,1.4,12.0)` and
    force EVERY volume onto it — which under native-z silently re-snapped each subject's real
    stack (D ranges 9-18 at dz 8/10/12 mm across the pooled cohort) back onto a 12-plane 12 mm
    cube at load time, in the SCORER, after the model had correctly reconstructed it.
    """
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
    `20*log10(gt[roi].max())`. That is a large offset in practice because the heart ROI rarely
    contains the volume's global maximum — measured on CMRx24_Test_P012: gt[roi].max() = 0.353,
    i.e. **-9.04 dB**, which is the whole of the 20.6-vs-29.6 discrepancy between this scorer and
    the trainer's heartseg number.

    Kept as the headline because it is the harness's cross-method convention (SVRTK / NeSVoR /
    NiftyMIC / VGGT are all scored this way, so the head-to-head is internally consistent), but
    `psnr_unit_peak` below is emitted alongside it so harness and trainer numbers can be
    reconciled without re-deriving this every time.
    """
    if not m.any():                       # empty ROI: b[m].max() would raise on a zero-size array
        return float("nan")
    mse = (((a - b) ** 2)[m]).mean()
    peak = max(b[m].max(), 1e-6)
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-10)))


def psnr_unit_peak(a, b, m):
    """PSNR with peak = 1.0 — the TRAINER's convention (`loss._psnr_from_mse`), for [0,1] data.

    Verified equivalent: on CMRx24_Test_P012 at ED over `heart_roi_canonical` this gives 29.62 dB
    against the trainer's own recorded 29.49 dB for the same subject/epoch (the residual is the
    different breathing realization, not the metric).
    """
    if not m.any():
        return float("nan")
    mse = (((a - b) ** 2)[m]).mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def ssim(a, b, m):
    a, b = a[m], b[m]
    if a.size < 2:
        return float("nan")
    mu_a, mu_b, va, vb = a.mean(), b.mean(), a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    L = max(a.max(), b.max()) - min(a.min(), b.min()) + 1e-9
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


def render_gif(out_path, rows, planes, T, vmax, titles, fps=3, plane_disp=None):
    """rows: list of (label, cine[T,X,Y,Z]); one animation frame per cardiac phase t,
    each frame = len(rows) x len(planes) montage. plane_disp: optional per-z applied breathing
    |disp| (mm), shown under each z-label so you can read the corruption per plane."""
    nrow, ncol = len(rows), len(planes)
    H = nrow * 1.15 + 0.8            # reserve a fixed strip at top for the title + z/disp labels
    top = 1.0 - 0.68 / H            # so the title never overlaps the montage
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.15, H))
        axes = np.atleast_2d(axes)
        for ri, (label, cine) in enumerate(rows):
            for ci, z in enumerate(planes):
                ax = axes[ri, ci]
                ax.imshow(cine[t, :, :, z].T, cmap="gray", vmin=0, vmax=vmax,
                          origin="lower", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    v = None if plane_disp is None else plane_disp[z]
                    lbl = f"z{z}" if v is None else f"z{z}\n{v:.1f}mm"   # blank on padding planes
                    ax.set_title(lbl, fontsize=6.5)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=8)
        fig.suptitle(titles.format(t=t), fontsize=9, y=0.985, va="top")
        fig.subplots_adjust(left=0.06, right=0.99, top=top, bottom=0.01,
                            wspace=0.03, hspace=0.06)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())   # mpl>=3.8: tostring_rgb removed
        frames.append(buf[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)
    print(f"  -> {out_path}")


def main():
    subj = sys.argv[1] if len(sys.argv) > 1 else "CMRx24_Train_P053"
    method = sys.argv[2] if len(sys.argv) > 2 else "svrtk3d"
    # Short display tag for the GIF row labels: the raw arm slug (e.g.
    # "vggt_20260719_1f_gather05_ep99") overflows the rotated y-label slot and clips.
    # Strip the "vggt_<date>_1f_" prefix + "_ep##..." suffix -> "gather05"; baseline
    # names (svrtk3d/nesvor) don't match either pattern and pass through unchanged.
    arm_label = re.sub(r"_ep\d+.*$", "", re.sub(r"^vggt_\d+_1f_", "", method))
    ds = DATASET
    sd = str(paths.subject_dir(ds, subj))                # subject dir = SHARED frozen bundle
    md = str(paths.arm_dir(ds, subj, method)); os.makedirs(md, exist_ok=True)   # per-method outputs
    manifest = json.load(open(paths.manifest(ds, subj)))
    T = manifest["T"]
    # Scoring ROI = GT whole-heart seg (dilated +-1 plane) INTERSECT native-FOV mask. The dilation
    # spills the ROI onto zero-padded edge planes with no acquired data; SVRTK (told to reconstruct
    # inside -mask) hallucinates spurious content there, and scoring it vs zero-padding GT drags PSNR
    # down artifactually (e.g. CMRx24_Train_P053 clean 20.2->27.5 dB once z0 is dropped). Intersecting with
    # the native FOV drops those no-data planes -> honest metric + no edge-plane flicker in the GIF.
    shape_xyz, aff = subject_grid(ds, subj)               # THIS subject's native grid (docs/58)
    D = shape_xyz[2]
    heart_p = str(paths.heart_mask(ds, subj))
    # The heart ROI is optional now: build_inputs/pooled.py warns-and-skips it for sources that
    # ship no canonical seg. Fall back to the FOV mask so those subjects still score (on a wider
    # ROI) instead of crashing — metrics.json records which was used.
    content = load_canon(str(paths.fov_mask(ds, subj)), shape_xyz, aff) > 0.5   # (X,Y,Z) native FOV
    has_heart = os.path.exists(heart_p)
    heart = load_canon(heart_p, shape_xyz, aff) > 0.5 if has_heart else content
    mask = heart & content                                # SCORING ROI: drop no-data padding planes
    planes = list(range(D))                               # DISPLAY: every NATIVE plane of this subject
    print(f"{subj} [{method}]: T={T} D={D} display planes z0-z{D-1}  "
          f"scoring ROI={'heart&FOV' if has_heart else 'FOV only (no heart seg)'}  "
          f"mask_voxels={int(mask.sum())} (heart-only {int(heart.sum())}, dropped {int(heart.sum()-mask.sum())})")

    # Breathing magnitude actually applied to this subject (frozen in the manifest; identical across
    # ALL baselines + the model, but different per subject). Per-plane displacement norm in mm.
    disp = np.asarray(manifest["breath"]["disp_dhw_mm"], dtype=np.float64)   # (D,3): d_Z (through-plane), d_Y, d_X
    disp_mag = np.linalg.norm(disp, axis=1)                                   # 3D VECTOR MAGNITUDE per plane (mm)
    dz = np.abs(disp[:, 0])                                                    # through-plane component (reference)
    # Report the vector MAGNITUDE: it's the quantity actually sampled (|SI+AP| = SI*sqrt(1+ap_ratio^2))
    # and is TILT-INVARIANT -- the per-subject tilt is a rigid rotation, so it preserves |disp| and only
    # redistributes it between through-plane dZ and in-plane. dZ alone varies with tilt, so magnitude is
    # the consistent thing to log (dZ kept as a secondary field).
    breath_mean_mm = float(disp_mag.mean()); breath_max_mm = float(disp_mag.max())
    print(f"  breathing |disp|: mean {breath_mean_mm:.2f} mm  max {breath_max_mm:.2f} mm  "
          f"(tilt-invariant; through-plane dZ mean {dz.mean():.2f} mm; frozen, same for all methods)")

    gt = np.stack([load_canon(str(paths.bundle_stack(ds, subj, "gt", t)), shape_xyz, aff)
                   for t in range(T)])
    cines = {"gt": gt}
    metrics = {"subject": subj, "planes": planes, "D": int(D),
               "dz_mm": float(abs(aff[2, 2])), "scoring_roi": "heart&FOV" if has_heart else "FOV",
               "breath_mean_disp_mm": breath_mean_mm, "breath_max_disp_mm": breath_max_mm,
               "breath_mean_dz_mm": float(dz.mean()), "breath_max_dz_mm": float(dz.max()),
               "breath_disp_per_plane_mm": disp_mag.tolist(), "per_phase": {}}
    # `clean` is opt-in (run_vggt --arms; default is breath only, the deliverable). Score whichever
    # arms exist rather than crashing on a deliberately absent one. `breath` is required.
    present = [v for v in ("clean", "breath") if paths.recon_dir(ds, subj, method, v).is_dir()]
    if "breath" not in present:
        raise FileNotFoundError(f"{subj} [{method}]: no recon_breath — that arm is the deliverable")
    metrics["arms"] = present
    for var in present:
        rec = np.stack([load_canon(str(paths.recon(ds, subj, method, var, t)), shape_xyz, aff)
                        for t in range(T)])
        rec = prep_recon(rec, method, mask)  # per-method: SVRTK as-is; nesvor/niftymic self-percentile [0,1]
        rec = rec * content[None]            # DISPLAY: zero the recon in NO-DATA planes (content FOV empty:
                                             # z0/z10/z11) — NeSVoR's 1.4mm-iso→12mm resample bleeds a blob
                                             # into them (SVRTK's hard -mask doesn't). Scoring is UNAFFECTED
                                             # (score mask ⊆ content, and GT is already 0 there).
        cines[var] = rec
        pv, sv, nv, uv = [], [], [], []
        for t in range(T):
            pv.append(psnr(rec[t], gt[t], mask)); sv.append(ssim(rec[t], gt[t], mask)); nv.append(ncc(rec[t], gt[t], mask))
            uv.append(psnr_unit_peak(rec[t], gt[t], mask))    # trainer-comparable (peak=1.0)
        metrics["per_phase"][var] = {"psnr": pv, "ssim": sv, "ncc": nv, "psnr_unit_peak": uv}
        metrics[f"{var}_psnr_mean"] = float(np.nanmean(pv))   # nanmean for all three so a degenerate phase
        metrics[f"{var}_ssim_mean"] = float(np.nanmean(sv))   # (constant/empty-ROI -> NaN) drops CONSISTENTLY
        metrics[f"{var}_ncc_mean"] = float(np.nanmean(nv))    # across metrics (NCC could already be NaN)
        metrics[f"{var}_psnr_unit_peak_mean"] = float(np.nanmean(uv))
        print(f"  {var}: PSNR {np.nanmean(pv):.2f} dB (unit-peak {np.nanmean(uv):.2f})  "
              f"SSIM {np.nanmean(sv):.3f}  NCC {np.nanmean(nv):.3f}")

    # save 4D cines (X,Y,Z,T): cine_gt is method-independent -> subject level (shared, written once);
    # the recon cines -> the method dir.
    nib.save(nib.Nifti1Image(np.moveaxis(cines["gt"], 0, -1), aff), os.path.join(sd, "cine_gt.nii.gz"))
    for k in present:
        nib.save(nib.Nifti1Image(np.moveaxis(cines[k], 0, -1), aff), os.path.join(md, f"cine_{k}.nii.gz"))
    metrics["method"] = method
    # Provenance stamp: tie this metrics.json to the recon it scored so aggregate can catch stale or
    # mixed-provenance arms. VGGT arms carry metadata.json (ckpt/regime/commit); classical baselines
    # don't -> those fields stay None. recon_mtime = newest scored recon (staleness reference).
    meta_path = str(paths.metadata(ds, subj, method))
    arm_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    recon_files = [str(paths.recon(ds, subj, method, v, t)) for v in ("clean", "breath") for t in range(T)]
    metrics["ckpt"] = arm_meta.get("ckpt")
    metrics["ckpt_fingerprint"] = arm_meta.get("ckpt_fingerprint")   # content id (size:mtime) for aggregate's mix check
    metrics["regime"] = arm_meta.get("regime")
    metrics["git_commit"] = arm_meta.get("git_commit")
    metrics["recon_mtime"] = max((os.path.getmtime(f) for f in recon_files if os.path.exists(f)), default=0.0)
    json.dump(metrics, open(os.path.join(md, "metrics.json"), "w"), indent=2)

    # Shared display window across all rows (GT + both recons), computed ONCE over all phases so it
    # never changes frame-to-frame. Use the 99.9th percentile over the UNION of GT+clean+breath in-ROI
    # voxels: p99 let ~0.7-3.4% of breath voxels saturate to white and that fraction swung per phase
    # (white-saturation flicker); p99.9 drops it to <=0.6% and near-constant. (A residual ~17% per-phase
    # brightness variation remains -- that's SVR re-matching intensity independently per phase, genuine,
    # not a windowing artifact.) Metric untouched.
    # SKIP_GIF=1: metrics.json only, no GIF rendering. For cohort sweeps where only the numbers are
    # wanted (rendering 3 GIFs x 43 subjects x 6 models dominates wall-clock on a 4-core node).
    # Default is unset => every existing output reproduces byte-identically.
    if os.environ.get("SKIP_GIF") == "1":
        print(f"done (SKIP_GIF=1, metrics only) -> {md}")
        return

    _in = mask[None].repeat(T, axis=0)
    _roi_vals = np.concatenate([gt[_in]] + [cines[k][_in] for k in present])
    vmax = float(np.percentile(_roi_vals, 99.9)) if _roi_vals.size else 1.0  # guard empty ROI
    # (all other percentile/score sites guard it; this one would IndexError on a heart&FOV-empty subject)
    breath_tag = f"breathing |disp| mean {breath_mean_mm:.1f} / max {breath_max_mm:.1f} mm"
    # per-z applied |disp| (mm) under the z-labels. Under native-z every source is indexed the same
    # way — `disp_dhw_mm` has one row per NATIVE plane and the display shows those same planes — so
    # the old "canonical vs native indexing" caveat is gone. The length check stays as a guard
    # against a stale bundle built before the native-z rebuild.
    aligned_disp = len(disp_mag) == D
    if not aligned_disp:
        print(f"  WARNING: manifest disp has {len(disp_mag)} planes but D={D}; "
              f"per-plane labels suppressed (stale bundle?)")
    pd = [float(disp_mag[z]) if (aligned_disp and content[:, :, z].any()) else None
          for z in range(D)]
    if "clean" in present:
        render_gif(os.path.join(md, "gif_clean.gif"),
                   [("GT", gt), (f"{arm_label}\n(no breath)", cines["clean"])], planes, T, vmax,
                   f"{subj} [{method}]  —  clean input (no breathing)   phase t={{t}}", plane_disp=pd)
    render_gif(os.path.join(md, "gif_breath.gif"),
               [("GT", gt), (f"{arm_label}\n(breathing)", cines["breath"])], planes, T, vmax,
               f"{subj} [{method}]  —  breathing input (mm under z = applied |disp|; {breath_tag})   phase t={{t}}",
               plane_disp=pd)
    # combined = the clean-vs-breath contrast; only meaningful when both arms were run.
    if "clean" in present:
        render_gif(os.path.join(md, "gif_combined.gif"),
                   [("GT", gt), (f"{arm_label}\nno-breath", cines["clean"]),
                    (f"{arm_label}\nbreathing", cines["breath"])],
                   planes, T, vmax,
                   f"{subj} [{method}]  —  GT vs {method} (mm under z = applied |disp|; {breath_tag})   phase t={{t}}",
                   plane_disp=pd)

    # Auto-render the VGGT per-arm diagnostic panels (panel_input/dvf/lookup) alongside the gifs in
    # the arm dir. VGGT arms only — baselines have no ed_dvf.npz, so this is skipped for them; and
    # fully best-effort (a missing dep / multiframe dir must NEVER break scoring), hence try/except.
    # Gated by the SKIP_GIF early-return above, so metric-only sweeps skip these too.
    if os.path.exists(os.path.join(md, "ed_dvf.npz")):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
            import slice_panels as _sp
            _sp.build(DATASET, subj, method, "breath", panels=("input", "dvf", "lookup"))
        except Exception as e:
            print(f"  [panels skipped — scoring unaffected] {type(e).__name__}: {e}")
    print(f"done -> {md}")


if __name__ == "__main__":
    main()
