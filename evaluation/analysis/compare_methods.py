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
  $PY tools/compare_methods.py --cohort cmrxrecon --subject Train_P001 \
      --arms svrtk3d nesvor vggt_20260713_gather05 --variant breath
  # --subject omitted -> first built subject that has ALL requested arms.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

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


def load_recon(ds, subj, arm_real, variant, T, mask, content):
    rec = np.stack([A.load_canon(str(paths.recon(ds, subj, arm_real, variant, t))) for t in range(T)])
    rec = A.prep_recon(rec, arm_real.split("_contz")[0], mask)   # per-method norm keyed on the base arm name
    return rec * content[None]                                   # zero no-data planes (matches assemble display)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=list(paths.DATASETS))
    ap.add_argument("--arms", nargs="+", required=True, help="arm dir names to stack (e.g. svrtk3d nesvor vggt_...)")
    ap.add_argument("--subject", default=None, help="default: first built subject that has ALL --arms")
    ap.add_argument("--variant", default="breath", choices=list(paths.VARIANTS))
    ap.add_argument("--out", default=None, help="default: analysis/out/<ds>/compare_<subject>_<variant>.gif")
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

    gt = np.stack([A.load_canon(str(paths.bundle_stack(ds, subj, "gt", t))) for t in range(T)])
    rows = [("GT", gt)]
    for arm in a.arms:
        rows.append((arm.replace("vggt_", ""),
                     load_recon(ds, subj, resolve_arm_dir(ds, subj, arm), a.variant, T, mask, content)))

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
    out = a.out or str(EVAL / "analysis" / "out" / ds / f"compare_{subj}_{a.variant}.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    title = (f"{ds} / {subj} / {a.variant}  —  GT vs {', '.join(r[0] for r in rows[1:])}"
             f"\ncardiac phase t={{t}}/{T - 1}")
    A.render_gif(out, rows, planes, T, vmax, title, fps=3, plane_disp=plane_disp)
    print(f"\n-> {out}  ({len(rows)} rows x {len(planes)} planes, T={T})")


if __name__ == "__main__":
    main()
