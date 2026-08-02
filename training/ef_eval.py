"""Predicted-EF validation metric (val-only, opt-in).

Segments the reconstructed ED/ES volumes (written by the trainer to a temp dir), computes each
subject's predicted EF = (LV_ED - LV_ES) / LV_ED, and correlates it against the ground-truth EF in
scratch/data/whs/cardiac_phase.csv. Reports slope / Spearman / MAE over the clean
(seg_flag == "ok") subjects, optionally split by a subject group (e.g. pathology).

TWO SEGMENTER BACKENDS, selected by `logging.ef_seg_backend`:
  "corseg" (default) — CorSeg-CineSAX, in-env, ~0.28 s/volume. Better at exactly what this file
      reports (docs/57: EF MAE 2.51 vs 4.67 pp, LV volume MAE 4.4 vs 8.9 mL).
  "nnunet"           — Task114 5-fold + TTA via a subprocess into the isolated `nnunet` env. This
      is the operator that produced the GT labels, so it is method-matched to the GT; it is also
      ~8.5x slower and the cross-env hop is flaky (hence `retries`).

⚠️ The two use DIFFERENT label indices for the LV cavity — see `LV_LABEL`. Always take the index
from `segment()` rather than hardcoding it.

All heavy work is triggered by the trainer on rank 0 every N val epochs, wrapped in try/except so
it never raises into training.
"""
import csv
import logging
import os
import subprocess
import time

import numpy as np
import nibabel as nib
import torch

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


# LV-CAVITY label index, per segmenter. These DIFFER, and getting it wrong silently
# computes EF from the myocardium instead of the cavity — a plausible-looking wrong number.
#   nnU-Net Task114 (M&Ms): 1 = LV cavity, 2 = LV myo, 3 = RV
#   CorSeg-CineSAX:         1 = LV myo,    2 = LV cavity, 3 = RV
LV_LABEL = {"nnunet": 1, "corseg": 2}


def run_corseg(in_dir, out_dir, device="cuda"):
    """Segment every `*_0000.nii.gz` in `in_dir` with CorSeg-CineSAX (docs/57).

    Chosen over nnU-Net for the training-time EF metric because, on the only cohort here
    with human GT, it is clearly better at exactly what this file computes — EF MAE
    2.51 vs 4.67 pp, LV volume MAE 4.4 vs 8.9 mL — while being ~8.5x cheaper and running
    IN `svr`, so it replaces a flaky cross-env `micromamba run -n nnunet` subprocess hop.

    Safe here specifically because we feed the FULL canonical cube: CorSeg collapses on
    heart-ROI-cropped input (Dice 0.889 -> 0.413, its fixed 224^2 canvas ends up 17% full),
    which is why docs/57 restricts it to canonical-cube arms and keeps nnU-Net for the SVR
    baselines. Output filenames match the nnU-Net convention (`{stem}.nii.gz`).
    """
    import glob
    import sys
    sys.path.insert(0, os.path.join(_REPO, "tools", "corseg"))
    from corseg_infer import load_corseg, segment_nifti

    os.makedirs(out_dir, exist_ok=True)
    # Stage the 741 MB checkpoint to node-local /tmp: measured 44 s to load from GPFS vs
    # 0.28 s to segment a volume, so the load dominates an EF epoch. Same rationale as
    # docs/50 for the model weights. Falls back to the original path on any failure.
    from corseg_infer import CKPT_DEFAULT
    from vggt.utils.checkpoint_stage import stage_checkpoint_to_local
    model, _ = load_corseg(stage_checkpoint_to_local(CKPT_DEFAULT), device=device)
    try:
        for src in sorted(glob.glob(os.path.join(in_dir, "*_0000.nii.gz"))):
            lab, affine, header, _ = segment_nifti(model, src, device=device)
            stem = os.path.basename(src)[: -len("_0000.nii.gz")]
            nib.save(nib.Nifti1Image(lab, affine, header),
                     os.path.join(out_dir, f"{stem}.nii.gz"))
    finally:
        del model
        torch.cuda.empty_cache()


def segment(in_dir, out_dir, backend="corseg"):
    """Run the configured segmenter. Returns its LV-cavity label index."""
    if backend == "corseg":
        run_corseg(in_dir, out_dir)
    else:
        run_nnunet(in_dir, out_dir)
    return LV_LABEL[backend]


def _lv_ml(seg_path, lv_label=1):
    # Voxel volume comes from the seg's OWN header, not a module constant (docs/59 F14), so this
    # is automatically right for every subject's native pitch and cannot drift from what
    # `save_pred_volume` wrote. Safe because nnU-Net `-m 2d` reproduces the input geometry
    # verbatim — MEASURED over 133 real ED/ES pairs: zooms and shape identical in/out, including
    # D=8..13 passing through unresampled.
    # CorSeg preserves geometry too: segment_nifti propagates the input header verbatim.
    im = nib.load(seg_path)
    vox_ml = float(np.prod(im.header.get_zooms()[:3])) / 1000.0
    return float((np.asarray(im.dataobj) == lv_label).sum()) * vox_ml


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


def _ef_stats(gts, preds):
    """slope / spearman / mae over one set of (GT EF, predicted EF) pairs, or None if the
    set cannot support a regression.

    Rejects a degenerate GT spread as well as a small n: `np.polyfit` on near-constant x
    does NOT raise — it returns lstsq's minimum-norm slope (plus a RankWarning on exactly
    constant input), so a healthy group of 3 subjects at EF 54/55/56 yields a large,
    finite, meaningless slope. Groups this narrow are exactly what `by_group` targets.

    The spread threshold is 1.0 EF percentage points. It was `1e-6` until 2026-08-01
    (docs/62 §5.5), which rejected only an EXACTLY constant GT, so the 54/55/56 example above
    (sigma = 0.82) sailed through and returned slope = 10.0. Latent, not live: today's groups
    are sigma 6.2 (healthy, n=60) and 16.2 (diseased, n=73), so nothing currently reported
    changes — this only guards the narrow groups a re-seeded split or finer `by_group` creates.
    """
    if len(preds) < 3 or np.std(np.asarray(gts, dtype=float)) < 1.0:
        return None
    gts, preds = np.asarray(gts), np.asarray(preds)
    return dict(slope=float(np.polyfit(gts, preds, 1)[0]),         # d(pred)/d(GT)
                spearman=_spearman(gts, preds),
                mae_pct=float(np.abs(preds - gts).mean()),
                n=len(preds))


def compute_ef_metrics(pred_seg_dir, subjects_ed_es, csv_path, groups=None, lv_label=1):
    """Predicted EF from the fresh segs vs GT EF from the CSV, over clean subjects.

    subjects_ed_es: list of (subject_id, ed, es). Excludes subjects with seg_flag != "ok"
    (e.g. CMRx24_Test_P044), a missing seg, or an empty ED cavity. Returns a metrics dict or None.
    slope = d(pred_EF)/d(GT_EF) (~1 ideal, ~0 = flat/cohort-mean regression).

    groups: optional {subject_id: group_name}. When given, the same statistics are ALSO
    computed per group and returned under "by_group". This exists because slope is a
    regression over the cohort's GT-EF spread, and the pooled cohort's spread is dominated
    by its diseased half — measured on val, GT-EF sigma is 16.2 for diseased (n=73) versus
    6.2 for healthy (n=60). A slope estimated over a 6-point spread is attenuated by range
    restriction no matter how good the model is, so a pooled slope silently drifts with the
    val health mix. Splitting is what stops that being misread as a model regression.
    """
    gt = load_gt_ef(csv_path)
    rows, skipped = [], []
    for subject, ed, es in subjects_ed_es:
        if subject not in gt or gt[subject][3] != "ok":
            skipped.append(subject); continue                      # bad/unreliable GT
        p_ed = os.path.join(pred_seg_dir, f"{subject}_t{ed:02d}.nii.gz")
        p_es = os.path.join(pred_seg_dir, f"{subject}_t{es:02d}.nii.gz")
        if not (os.path.exists(p_ed) and os.path.exists(p_es)):
            skipped.append(subject); continue                      # seg missing
        v_ed, v_es = _lv_ml(p_ed, lv_label), _lv_ml(p_es, lv_label)
        if v_ed <= 0:
            skipped.append(subject); continue                      # empty pred cavity
        rows.append((subject, gt[subject][2], (v_ed - v_es) / v_ed * 100.0))

    overall = _ef_stats([r[1] for r in rows], [r[2] for r in rows])
    if overall is None:
        return None
    overall["n_skipped"] = len(skipped)
    if groups:
        by_group = {}
        for name in sorted({groups.get(s) for s, _, _ in rows} - {None}):
            sel = [(g, p) for s, g, p in rows if groups.get(s) == name]
            stats = _ef_stats([g for g, _ in sel], [p for _, p in sel])
            if stats is not None:
                by_group[name] = stats
        overall["by_group"] = by_group
    return overall
