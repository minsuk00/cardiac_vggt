"""Shared scoring contract for the motion_epe baseline scripts (fetal_cmr_4d, niftymic, svrtk3d, dangi).

Every method is scored on the SAME D input slices per subject: the scatter/stack_t00 input that
VGGT and the scatter baselines receive -- plane z at breath/ frame manifest["scatter"]
["phase_per_plane"][z], the reference plane at frame 0 (pooled.add_scatter: stack_t{k}'s reference
plane is breath/stack_t{k}). Fetal CMR 4D and CiNeVol see every frame of every plane, which includes
these, so their per-frame pose is read at exactly these slots. vggt.py (ed_dvf slot_t) and cinevol.py
(scatter frames, reference at 0) already score this set.

Per method: each script hands over its prediction ALREADY on the true axes (dz = the stack's
through-plane axis, dy/dx = its in-plane Y/X). Every bundle stack is axis-aligned (diagonal affine)
and every engine's pose lives in world mm, so the axis is fixed geometry, not fitted. Only one pooled
SIGN per axis is fitted (engines differ in whether they report the slice's or the volume's motion,
and ITK's LPS flips X/Y). No axis search: docs/105 §11's best-|corr| search could pick an in-plane
axis for dz because the simulated breathing couples dz and dy, crediting in-plane tracking as
through-plane. The error is demeaned per subject.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = str(next(p for p in HERE.parents if (p / "evaluation").is_dir()))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "evaluation")]
import paths                                                               # noqa: E402
from build_af_bundle import breathing_model                                # noqa: E402
from data.respiratory import lujan_displacement                            # noqa: E402

SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")
AXES = ("dz", "dy", "dx")


def slots(man):
    """[(z, f)] for z in 0..D-1: the breath/ frame plane z shows in scatter/stack_t00."""
    sc = man["scatter"]
    f = [int(x) for x in sc["phase_per_plane"]]
    f[int(sc["ref_plane"])] = 0
    return list(enumerate(f))


def applied(man, sl):
    """(len(sl), 3) TRUE (dz, dy, dx) mm at each (z, f) slot. Rhythm manifest: frame-wise breathing;
    plain manifest: one frozen shift per plane (f irrelevant)."""
    if "rhythm" not in man:
        d = np.asarray(man["breath"]["disp_dhw_mm"], float)
        return np.array([d[z] for z, _ in sl])
    u, amp, r0, n, _ = breathing_model(man)
    P, frozen = man["rhythm"]["params"], man["rhythm"]["frozen_breath"]
    r = [r0[z] if frozen else (r0[z] + f * P["DT"] / P["T_BREATH"]) % 1.0 for z, f in sl]
    return np.array([u * amp * float(lujan_displacement(float(x), 1.0, n=n)) for x in r])


def subjects(arm, rel):
    """[(subject_id, subject_dir, manifest)] with <subject_dir>/<rel> present. `arm` = rhythm suffix
    (af12 -> every <source>_af12) or any plain source name (-> all 5 plain sources, docs/105 §11)."""
    out = []
    for src in SOURCES:
        ds = f"{src}_{arm}" if arm not in SOURCES else src
        root = paths.dataset_root(ds)
        if not root.is_dir():
            continue
        for sd in sorted(glob.glob(str(root / "*"))):
            if os.path.exists(os.path.join(sd, rel)):
                out.append((f"{ds}/{os.path.basename(sd)}", sd, json.load(open(os.path.join(sd, "manifest.json")))))
    return out


def check_scatter_input(recon_dir):
    """The scatter baselines' t00 pose is only VGGT's slot if the arm was run on scatter/ input."""
    st = json.load(open(os.path.join(recon_dir, "stamp.json")))
    if st.get("input_stack") != "scatter":
        raise SystemExit(f"{recon_dir}: input_stack={st.get('input_stack')!r}, need 'scatter'")


def mirtk_centroid_disp(params, p):
    """World-mm displacement of point p under a MIRTK rigid pose (tx,ty,tz, rx,ry,rz deg): R p + t - p.
    MIRTK rotates about the world ORIGIN, so raw Translation* is not the slice's shift. Convention
    (intrinsic XYZ, negated angles) recovered against reconstructCardiac's own MeanDisplacementX/Y/Z
    (SVRTK ReconstructionCardiac4D::CalculateDisplacement = mean over slice pixels of T(p) - p): on
    2544 af12 rows with p = the slice's heart-mask centroid, |diff| = 0.03/0.03/0.12 mm (X/Y/Z)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("XYZ", -np.asarray(params[3:6], float), degrees=True)
    return R.apply(p) + np.asarray(params[:3], float) - p


def mask_centroid_world(mask, affine, z):
    """World mm of plane z's heart-mask centroid (image centre if the plane has no mask)."""
    ij = np.argwhere(mask[..., z])
    c = ij.mean(0) if len(ij) else (np.array(mask.shape[:2]) - 1) / 2
    return (affine @ np.r_[c, z, 1.0])[:3]


def score(per_subj, key, pred_ax, axes=AXES):
    """per_subj: {subject_id: (applied (n, len(axes)), pred (n, len(axes)), planes z (n,))}, pred
    column i already on true axis i (pred_ax[i] names the engine quantity), NaN slots already
    dropped. Returns (rows for the --json file, dz per-subject dump, per-slice dump for
    write_slices -- sign applied)."""
    if not per_subj:
        raise SystemExit(f"[{key}] no scorable subject")
    # sign from per-subject-DEMEANED values (the metric is demeaned; a raw pooled corr can be
    # dominated by per-subject constant offsets) -- same rule as cinevol.py
    kept = [v for v in per_subj.values() if len(v[0]) >= 2]
    A = np.concatenate([v[0] - v[0].mean(0) for v in kept])
    Pp = np.concatenate([v[1] - v[1].mean(0) for v in kept])
    print(f"[{key}] {len(per_subj)} subjects, {sum(len(v[0]) for v in per_subj.values())} slots")
    out, dump = {}, {}
    signs = []
    for i, nm in enumerate(axes):
        c = float(np.corrcoef(A[:, i], Pp[:, i])[0, 1]) if Pp[:, i].std() > 1e-12 else 0.0
        sgn = 1.0 if c >= 0 else -1.0
        signs.append(sgn)
        print(f"  {nm:4} <- {pred_ax[i]:18} pooled demeaned corr {c:+.3f} -> sign {sgn:+.0f}")
        rs, zs = [], []
        for sid, (a_, p_, _z) in per_subj.items():
            if len(a_) < 2:
                continue
            a_i, p_j = a_[:, i], sgn * p_[:, i]
            err = p_j - a_i
            dm = float(np.abs(err - err.mean()).mean())
            # "epe" = demeaned (what tools/rhythm24_ef_table.py reads), raw in "epe_raw" -- cinevol.py's convention
            rs.append({"epe": dm, "epe_demeaned": dm, "epe_raw": float(np.abs(err).mean()),
                       "applied_abs": float(np.abs(a_i).mean())})
            zs.append({"epe_demeaned": float(np.abs(a_i - a_i.mean()).mean()), "applied_abs": float(np.abs(a_i).mean())})
            if nm == "dz":
                dump[sid] = {"applied": a_i.tolist(), "pred": p_j.tolist()}
        mean = lambda rows, k: float(np.nanmean([r[k] for r in rows]))       # noqa: E731
        out[f"{key}_{nm}"] = {"epe": mean(rs, "epe"), "epe_demeaned": mean(rs, "epe_demeaned"), "epe_raw": mean(rs, "epe_raw"),
                              "applied_abs": mean(rs, "applied_abs"), "n": len(rs), "input": "scatter_t00",
                              "pred_axis": pred_ax[i], "pred_sign": sgn, "pooled_corr": c}
        out[f"predict_nothing_{key}_{nm}"] = {"epe_demeaned": mean(zs, "epe_demeaned"),
                                              "applied_abs": mean(zs, "applied_abs"), "n": len(zs)}
    if "dz" in axes:   # through-plane alias under the plain method key (tools/rhythm24_ef_table.py)
        out[key] = out[f"{key}_dz"]
        out[f"predict_nothing_{key}"] = out[f"predict_nothing_{key}_dz"]
    print(f"\n{'row':34}{'n':>5}{'EPE demeaned':>14}{'|applied|':>11}")
    for m, o in out.items():
        print(f"{m:34}{o['n']:5d}{o['epe_demeaned']:14.2f}{o['applied_abs']:11.2f}")
    slices = {sid: {int(z): {"applied": a_[k].tolist(), "pred": (np.array(signs) * p_[k]).tolist()}
                    for k, z in enumerate(z_)} for sid, (a_, p_, z_) in per_subj.items()}
    return out, dump, slices


def write_slices(path, method, axes, slices):
    """Per-slice dump for common_set.py: {"method", "axes", "slices": {subject_id: {z: {applied, pred}}}}.
    `pred` carries the method's fitted sign, so common_set.py only intersects and scores."""
    json.dump({"method": method, "axes": list(axes), "slices": slices}, open(path, "w"))
    print(f"wrote {path} ({sum(len(v) for v in slices.values())} slices)")


def write_json(path, out):
    prev = json.load(open(path)) if os.path.isfile(path) else {}
    prev.update(out)
    json.dump(prev, open(path, "w"), indent=1)
    print(f"wrote {path} ({len(prev)} rows)")
