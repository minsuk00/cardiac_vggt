#!/usr/bin/env python
"""compare_methods.py — one figure comparing MULTIPLE reconstruction methods on the SAME subject.

Stacks the shared GT and one recon row per arm into a single cardiac-cycle GIF, per canonical plane.
Method-AGNOSTIC on purpose: no Δz/slot/ed_dvf machinery (that is VGGT-only and lives in slice_panels),
so classical baselines (svrtk3d, nesvor) and any vggt_* arm compare side by side on the same GT.

  row 0      GT (clean gated ground truth), animating over the cardiac cycle
  row 1..N   one recon row per --arm, same subject / same 12 canonical planes, animating
  header     z{k} + applied breathing |disp| (mm) per plane (frozen bundle -> identical for every arm)

Pure disk read (CPU, no GPU/model). Reuses assemble_and_gif's loaders + per-method prep_recon, so each
recon is displayed on exactly the intensity scale it is SCORED on (SVRTK as-is, NeSVoR self-normalized).

Run:
  PY=/home/minsukc/micromamba/envs/svr/bin/python
  $PY evaluation/analysis/compare_methods.py --cohort cmrxrecon --subject Train_P001 \
      --arms svrtk3d nesvor vggt_20260713_gather05 --variant breath
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
ROOT = next(p for p in HERE.parents if (p / "evaluation").is_dir())   # repo root (works from tools/ or evaluation/analysis/)
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL / "engine"))
import paths                       # noqa: E402
import assemble_and_gif as A       # noqa: E402  (load_canon, prep_recon, render_gif, SHAPE_XYZ)


def resolve_arm_dir(ds, subj, arm):
    """Real on-disk arm dir name; legacy OOD contz dirs carry a _contz suffix (try both). None if absent."""
    for suf in ("", "_contz"):
        if paths.arm_dir(ds, subj, arm + suf).is_dir():
            return arm + suf
    return None


def load_recon(ds, subj, arm_real, variant, disp_mask):
    """Load the CANONICAL, already-placed + per-method-normalized recon that assemble scored
    (<arm>/cine_<variant>.nii.gz, shape X,Y,Z,T). Using the placed cine — NOT recon_<variant>/vol_t* —
    is what makes OOD baselines show correctly: SVRTK/NeSVoR recons on miitt/OOD are in NATIVE space
    (e.g. 78x96x79), and load_canon can't place a native affine onto the canonical grid, so it renders
    them all-zero (black). The cine was placed via the dataset's adapter at assemble time."""
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
                    help="default: figures/<ds>/<subject>/_compare/compare_<variant>[_masked].gif (GPFS)")
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
    heart = A.load_canon(str(paths.heart_mask(ds, subj))) > 0.5
    content = A.load_canon(str(paths.fov_mask(ds, subj))) > 0.5
    mask = heart & content
    disp_mask = mask if a.mask else content        # --mask: restrict the display to the heart ROI

    gt = np.stack([A.load_canon(str(paths.bundle_stack(ds, subj, "gt", t))) for t in range(T)])
    rows = [("GT", gt)]                             # GT is ALWAYS unmasked — the full reference
    for arm in a.arms:
        rows.append((short(arm),
                     load_recon(ds, subj, resolve_arm_dir(ds, subj, arm), a.variant, disp_mask)))

    vmax = float(np.percentile(gt[gt > 0], 99.5)) if (gt > 0).any() else 1.0
    disp = np.asarray(manifest["breath"]["disp_dhw_mm"], dtype=np.float64)     # (native_Z, 3) — NOT 12 on OOD
    plane_disp = None
    if a.variant == "breath":                                                  # per-canonical-plane |disp| label
        dmag = np.linalg.norm(disp, axis=1)                                    # length = native Z (10/11/12/13)
        # only canonical-indexed disp (cmrx, len==12) maps 1:1 to display plane z; OOD is native-indexed
        # so disp[z] would MISLABEL the plane shown -> blank it there (matches assemble_and_gif).
        canonical_disp = len(dmag) == A.SHAPE_XYZ[2]
        plane_disp = [float(dmag[z]) if (canonical_disp and content[:, :, z].any()) else None
                      for z in range(A.SHAPE_XYZ[2])]

    planes = list(range(A.SHAPE_XYZ[2]))
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
