#!/usr/bin/env python
"""compare_methods.py — one figure comparing MULTIPLE reconstruction methods on the SAME subject.

Stacks the shared GT and one recon row per arm into a single cardiac-cycle GIF, per z-plane.
Method-AGNOSTIC on purpose: no Δz/slot/ed_dvf machinery (that is VGGT-only and lives in slice_panels),
so classical baselines (svrtk3d, nesvor) and any vggt_* arm compare side by side on the same GT.

  row 0      GT (clean gated ground truth), animating over the cardiac cycle
  row 1..N   one recon row per --arm, same subject / same native z-planes, animating
  header     z{k} + applied breathing |disp| (mm) per plane (frozen bundle -> identical for every arm)

Pure disk read (CPU, no GPU/model). Reuses assemble_and_gif's loaders + per-method prep_recon, so each
recon is displayed on exactly the intensity scale it is SCORED on (SVRTK as-is, NeSVoR self-normalized).

Run:
  PY=/home/minsukc/micromamba/envs/svr/bin/python
  $PY evaluation/src/analysis/compare_methods.py --cohort cmrx2024 --subject CMRx24_Test_P012 \
      --arms svrtk3d nesvor vggt_augaggr224hw2_ep300 --variant breath
  # --subject omitted -> first built subject that has ALL requested arms.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import nibabel as nib


def short(arm):
    """Compact row label: strip the vggt_<date>_1f_ prefix + _ep## suffix; baselines pass through."""
    return re.sub(r"_ep\d+.*$", "", re.sub(r"^vggt_\d+_1f_", "", arm))

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "evaluation").is_dir())   # repo root (works from tools/ or evaluation/src/analysis/)
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "src" / "engine"))
import paths                       # noqa: E402
import assemble_and_gif as A       # noqa: E402  (subject_grid, load_canon, prep_recon, render_gif)


def resolve_arm_dir(ds, subj, arm):
    """Real on-disk arm dir name; legacy OOD contz dirs carry a _contz suffix (try both). None if absent."""
    for suf in ("", "_contz"):
        if paths.arm_dir(ds, subj, arm + suf).is_dir():
            return arm + suf
    return None


def load_recon(ds, subj, arm_real, variant, disp_mask):
    """Load the CANONICAL, already-placed + per-method-normalized recon that assemble scored
    (<arm>/cine_<variant>.nii.gz, shape X,Y,Z,T). Using the placed cine — NOT recon_<variant>/vol_t* —
    is what makes the classical baselines show correctly: SVRTK/NeSVoR reconstruct on their OWN grid
    (e.g. 78x96x79 at 1.4 mm iso), which assemble_and_gif already resampled onto this subject's GT
    grid when it scored them. Reading vol_t* here would re-do that placement in a second place."""
    cine = paths.arm_dir(ds, subj, arm_real) / f"cine_{variant}.nii.gz"
    if not cine.is_file():
        sys.exit(f"missing {cine} — run assemble_and_gif on this arm first (it writes cine_<variant>)")
    v = np.asarray(nib.load(str(cine)).dataobj, dtype=np.float32)   # (X,Y,Z,T) canonical, prep'd, fov-zeroed
    return np.moveaxis(v, -1, 0) * disp_mask[None]                  # -> (T,X,Y,Z), then display mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--arms", nargs="+", required=True, help="arm dir names to stack (e.g. svrtk3d nesvor vggt_...)")
    ap.add_argument("--subject", default=None, help="default: first built subject that has ALL --arms")
    ap.add_argument("--variant", default="breath", choices=list(paths.VARIANTS))
    ap.add_argument("--mask", action="store_true",
                    help="restrict the display to the heart ROI (heart & FOV) — matches the scoring "
                         "region + how the SVR baselines reconstruct; default shows the full FOV")
    ap.add_argument("--out", default=None,
                    help="default: comparison_figures/<ds>/<subject>/_compare/compare_<variant>[_masked].gif (GPFS)")
    a = ap.parse_args()
    ds = a.cohort

    def has_all(s):
        return all(resolve_arm_dir(ds, s, arm) for arm in a.arms)
    subj = a.subject or next((s for s in paths.subjects(ds) if has_all(s)), None)
    if subj is None:
        sys.exit(f"no built subject in {ds} has all arms {a.arms}")
    absent = [arm for arm in a.arms if not resolve_arm_dir(ds, subj, arm)]
    if absent:
        sys.exit(f"subject {subj} is missing arms: {absent}")

    manifest = json.load(open(paths.manifest(ds, subj)))
    T = manifest["T"]
    # Native-z (docs/58): the scoring grid belongs to the SUBJECT (its own D and dz), so every load
    # goes onto that grid — same as assemble_and_gif, which is where these volumes were scored.
    shape_xyz, aff = A.subject_grid(ds, subj)
    D = shape_xyz[2]
    content = A.load_canon(str(paths.fov_mask(ds, subj)), shape_xyz, aff) > 0.5
    # The heart ROI is optional — build_inputs/pooled.py warns and skips it for sources that ship
    # no canonical seg. Fall back to the FOV, same as assemble_and_gif, instead of FileNotFound.
    heart_p = str(paths.heart_mask(ds, subj))
    heart = (A.load_canon(heart_p, shape_xyz, aff) > 0.5) if os.path.exists(heart_p) else content
    mask = heart & content
    disp_mask = mask if a.mask else content        # --mask: restrict the display to the heart ROI

    gt = np.stack([A.load_canon(str(paths.bundle_stack(ds, subj, "gt", t)), shape_xyz, aff)
                   for t in range(T)])
    rows = [("GT", gt)]                             # GT is ALWAYS unmasked — the full reference
    for arm in a.arms:
        rows.append((short(arm),
                     load_recon(ds, subj, resolve_arm_dir(ds, subj, arm), a.variant, disp_mask)))

    vmax = float(np.percentile(gt[gt > 0], 99.5)) if (gt > 0).any() else 1.0
    disp = np.asarray(manifest["breath"]["disp_dhw_mm"], dtype=np.float64)     # (D, 3), one row per plane
    plane_disp = None
    if a.variant == "breath":                                                  # per-plane |disp| label
        dmag = np.linalg.norm(disp, axis=1)
        # Under native-z every source is indexed the same way — disp has one row per native plane and
        # the display shows those same planes. The length check is a guard against a stale bundle
        # built before the native-z rebuild (matches assemble_and_gif).
        aligned_disp = len(dmag) == D
        if not aligned_disp:
            print(f"  WARNING: manifest disp has {len(dmag)} planes but D={D}; labels suppressed")
        plane_disp = [float(dmag[z]) if (aligned_disp and content[:, :, z].any()) else None
                      for z in range(D)]

    planes = list(range(D))
    # Cross-arm figure -> the FIGURES tree (GPFS), under the subject's _compare/ (owns no single arm).
    suffix = "_masked" if a.mask else ""
    out = a.out or str(paths.compare_dir(ds, subj) / f"compare_{a.variant}{suffix}.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    roi = "heart ROI" if a.mask else "full FOV"
    title = (f"{ds} / {subj} / {a.variant} ({roi})  —  GT vs {', '.join(r[0] for r in rows[1:])}"
             f"\ncardiac phase t={{t}}/{T - 1}")
    A.render_gif(out, rows, planes, T, vmax, title, fps=3, plane_disp=plane_disp)
    print(f"\n-> {out}  ({len(rows)} rows x {len(planes)} planes, T={T})")


if __name__ == "__main__":
    main()
