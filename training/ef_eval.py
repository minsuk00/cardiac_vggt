"""Predicted-EF validation metric (val-only, opt-in).

Segments the reconstructed ED/ES volumes (written by the trainer to a temp dir in nnU-Net input
format) with nnU-Net Task114 in the isolated `nnunet` env, computes each subject's predicted
EF = (LV_ED - LV_ES) / LV_ED, and correlates it against the ground-truth EF in
scratch/data/whs/cardiac_phase.csv (itself produced by the same Task114 operator). Reports
slope / Spearman / MAE over the clean (seg_flag == "ok") subjects.

The nnU-Net call is ONE batched subprocess over the whole pred dir (model loads once), using the
default 5-fold + TTA config — identical to how the GT was made, so pred and GT share the segmenter
and the EF comparison is method-matched. All heavy work is triggered by the trainer on rank 0 every
N val epochs, wrapped in try/except so it never raises into training.
"""
import csv
import logging
import os
import subprocess
import time

import numpy as np
import nibabel as nib

IN_PLANE_MM = 1.4        # canonical in-plane spacing — FIXED for every subject (native-z, docs/58)
# NOTE (docs/59 F14): there is deliberately NO canonical z spacing constant. Under native-z each
# subject keeps its own acquired pitch (5-12 mm), so a module-level `CANON_SPACING=(1.4,1.4,12.0)`
# stamped 12 mm onto every written volume regardless of the subject — up to 2.4x wrong at 5 mm.
# That was harmless for the REPORTED metric (EF is a ratio, so the voxel volume cancels; and
# nnU-Net `-m 2d` preserves input geometry verbatim, measured), but it made the dumped NIfTIs
# geometrically false and every absolute mL wrong by dz/12 — and it would have started changing
# the segmentation itself the moment anyone switched to `3d_fullres`, which DOES resample z.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_SH = os.path.join(_REPO, "tools", "nnunet_mnms_eval", "env.sh")


def save_pred_volume(V_dhw, out_dir, subject, t, dz_mm):
    """Write one reconstructed volume in nnU-Net input format.

    V_dhw: canonical (D,H,W) splat-order (Z,Y,X) -> nnU-Net (X,Y,Z), suffixed _0000.
    dz_mm: REQUIRED, no default — THIS subject's own native slice pitch (`batch["dz_mm"]`).
        Required rather than defaulted for the same reason as `splat.py`'s `z_scale`: a silent
        fallback writes a plausible-looking volume with false geometry and nothing errors.
    """
    arr = np.transpose(np.asarray(V_dhw, dtype=np.float32), (2, 1, 0))   # (X,Y,Z)
    affine = np.diag([IN_PLANE_MM, IN_PLANE_MM, float(dz_mm), 1.0])
    nib.save(nib.Nifti1Image(arr, affine),
             os.path.join(out_dir, f"{subject}_t{t:02d}_0000.nii.gz"))


def run_nnunet(in_dir, out_dir, retries=5):
    """One batched nnU-Net Task114 2D prediction (default 5-fold + TTA = the GT's operator).
    The model loads once for the whole `in_dir`. Retries guard the mamba cache lock."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["micromamba", "run", "-n", "nnunet", "bash", "-c",
           f"source '{_ENV_SH}' && nnUNet_predict -i '{in_dir}' -o '{out_dir}' "
           f"-t 114 -m 2d -tr nnUNetTrainerV2_MMS"]
    for i in range(1, retries + 1):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return
        logging.warning(f"[ef] nnU-Net predict rc={r.returncode} (try {i}/{retries}): "
                        f"{(r.stderr or '')[-500:]}")
        time.sleep(i * 8)
    raise RuntimeError(f"nnU-Net predict failed after {retries} tries")


def _lv_ml(seg_path):
    # Voxel volume comes from the seg's OWN header, not a module constant (docs/59 F14), so this
    # is automatically right for every subject's native pitch and cannot drift from what
    # `save_pred_volume` wrote. Safe because nnU-Net `-m 2d` reproduces the input geometry
    # verbatim — MEASURED over 133 real ED/ES pairs: zooms and shape identical in/out, including
    # D=8..13 passing through unresampled.
    im = nib.load(seg_path)
    vox_ml = float(np.prod(im.header.get_zooms()[:3])) / 1000.0
    return float((np.asarray(im.dataobj) == 1).sum()) * vox_ml     # LV cavity = label 1


def _spearman(x, y):
    # Average-rank Spearman (handles ties correctly, and returns nan for a constant input —
    # a flat/degenerate EF prediction, which a naive double-argsort would score ±1.0). scipy is
    # in the svr env; fall back to a tie-agnostic estimate only if it were ever missing.
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")   # correlation undefined when either side is constant
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y).correlation
        return float(r) if r == r else float("nan")
    except Exception:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])


def load_gt_ef(csv_path):
    """{subject: (ed, es, ef_pct, seg_flag)} from cardiac_phase.csv."""
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gt[row["subject"]] = (int(row["ED"]), int(row["ES"]),
                                  float(row["EF_pct"]), row["seg_flag"])
    return gt


def compute_ef_metrics(pred_seg_dir, subjects_ed_es, csv_path):
    """Predicted EF from the fresh segs vs GT EF from the CSV, over clean subjects.

    subjects_ed_es: list of (subject_id, ed, es). Excludes subjects with seg_flag != "ok"
    (e.g. CMRx24_Test_P044), a missing seg, or an empty ED cavity. Returns a metrics dict or None.
    slope = d(pred_EF)/d(GT_EF) (~1 ideal, ~0 = flat/cohort-mean regression)."""
    gt = load_gt_ef(csv_path)
    gts, preds, skipped = [], [], []
    for subject, ed, es in subjects_ed_es:
        if subject not in gt or gt[subject][3] != "ok":
            skipped.append(subject); continue                      # bad/unreliable GT
        p_ed = os.path.join(pred_seg_dir, f"{subject}_t{ed:02d}.nii.gz")
        p_es = os.path.join(pred_seg_dir, f"{subject}_t{es:02d}.nii.gz")
        if not (os.path.exists(p_ed) and os.path.exists(p_es)):
            skipped.append(subject); continue                      # seg missing
        v_ed, v_es = _lv_ml(p_ed), _lv_ml(p_es)
        if v_ed <= 0:
            skipped.append(subject); continue                      # empty pred cavity
        preds.append((v_ed - v_es) / v_ed * 100.0)                 # predicted EF %
        gts.append(gt[subject][2])                                 # GT EF %
    if len(preds) < 3:
        return None
    gts, preds = np.asarray(gts), np.asarray(preds)
    slope = float(np.polyfit(gts, preds, 1)[0])                    # d(pred)/d(GT)
    return dict(slope=slope, spearman=_spearman(gts, preds),
                mae_pct=float(np.abs(preds - gts).mean()),
                n=len(preds), n_skipped=len(skipped))
