"""Aggregate ALL per-subject metrics for one arm into ONE cohort summary.

The single collector (successor to _archive/aggregate.py): for each subject it reads
  - <subj>/<arm>/metrics.json          image metrics (image_metrics.py)
  - <subj>/<arm>/resp_diag.json        breathing motion pred-vs-applied (run_vggt; baselines
                                       once the transform-saving svrtk3d_debug run exists)
  - <subj>/<arm>/timing.json | <arm>/recon_breath/total_wall.sec     recon wall-clock
and writes ONE  metric_results/<dataset>/<method>.json  (the git-tracked citable numbers).
The EF/Dice chain (ef_dice.py, separate because it crosses to the nnunet env) writes
metric_results/_ef/<method>.json; when that exists its per-subject biventricular metrics are
joined into the rows and its cohort aggregate lands under summary["ef"] — absent file = null
block, nothing fails. Re-run this aggregator after the seg chain to fold them in.

Only subjects whose `manifest["split"]` matches `$SPLIT` (default "val") are summarized — see
`paths.filter_by_split` for why that must be enforced here and not only in run_vggt.

Run: [SPLIT=val] micromamba run -n svr python evaluation/src/score/aggregate.py <dataset> [method]
"""
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402


def json_safe(o):
    """NaN/Inf -> None, recursively — these summaries are the git-tracked citable numbers and
    must stay strict-JSON parseable."""
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def stat(xs):
    """Mean/std/n over the VALID (non-NaN) values, so one unscorable subject doesn't poison
    the cohort number."""
    xs = np.asarray(xs, dtype=np.float64)
    valid = xs[~np.isnan(xs)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return float(valid.mean()), float(valid.std()), int(valid.size)


def _resp_row(ds, subj, method):
    """Breathing-motion summary from resp_diag.json's breath arm, if this arm has one.
    Adds demeaned EPE (per-subject mean error removed): a constant offset is the pose gauge,
    not breath-tracking — the number that stays comparable across anchored/floating methods."""
    p = paths.resp_diag(ds, subj, method)
    try:
        b = json.load(open(p)).get("breath") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    pred, appl = b.get("pred_dz_mm"), b.get("applied_dz_mm")
    # A malformed file (missing key, mismatched lengths) must degrade to "no resp data", not
    # crash the whole cohort — and a length-1 applied would silently BROADCAST into a wrong
    # demeaned EPE, so require exact length match.
    if not pred or not appl or len(pred) != len(appl):
        if pred or appl:
            print(f"  !! {subj} [{method}]: malformed resp_diag.json "
                  f"(pred {len(pred or [])} vs applied {len(appl or [])} slots) — resp metrics skipped")
        return {}
    err = np.asarray(pred, dtype=np.float64) - np.asarray(appl, dtype=np.float64)
    return {"resp_epe_dz_mm": b.get("epe_dz_mm"),
            "resp_epe_dz_demeaned_mm": float(np.mean(np.abs(err - err.mean()))),
            "resp_slope": b.get("slope"), "resp_corr": b.get("corr")}


def _wall_sec(ds, subj, method):
    """Breath-arm recon wall-clock per subject: VGGT arms carry timing.json (feed-forward),
    classical arms a total_wall.sec next to the recons."""
    t = paths.arm_dir(ds, subj, method) / "timing.json"
    try:
        return float(json.load(open(t))["breath"]["total_sec"])
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    w = paths.recon_dir(ds, subj, method, "breath") / "total_wall.sec"
    try:
        return float(open(w).read().strip())
    except (OSError, ValueError):
        return None


def _ef_data(dataset, method):
    """(per-subject ef rows for this dataset keyed by subject, cohort aggregate block) from the
    EF/Dice chain's output — ({}, None) when the chain hasn't run for this arm."""
    p = paths.ef_summary(method)
    try:
        d = json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return {}, None
    per = {r["subject"]: {k: v for k, v in r.items() if k not in ("cohort", "subject")}
           for r in d.get("per_subject", []) if r.get("cohort") == dataset}
    return per, d.get("aggregate", {}).get(dataset)


def aggregate(dataset, method, split, exclude=()):
    """`exclude`: subjects whose scoring FAILED in the calling run — their surviving
    metrics.json (from an earlier run) must not be averaged in as if fresh (stale-row leak);
    they are dropped here and listed in the summary as `excluded_stale`."""
    root = paths.dataset_root(dataset)
    files = sorted(glob.glob(str(root / "*" / method / "metrics.json")))
    if not files:
        sys.exit(f"no metrics found at {root}/*/{method}/metrics.json")

    # A cohort is defined by its SPLIT; nothing in the layout enforces that, so a test-split
    # bundle scored into the same tree would silently average into the val numbers.
    def _subj_of(f):
        return os.path.basename(os.path.dirname(os.path.dirname(f)))
    keep, dropped = paths.filter_by_split(dataset, [_subj_of(f) for f in files], split)
    if dropped:
        print(f"  !! EXCLUDED {len(dropped)} scored subject(s) not in split '{split}':")
        for s, why in dropped:
            print(f"       {s}: {why}")
    keep = set(keep)
    excluded_stale = sorted(set(exclude) & {_subj_of(f) for f in files})
    if excluded_stale:
        print(f"  !! EXCLUDED {len(excluded_stale)} subject(s) whose scoring failed this run "
              f"(their on-disk metrics.json is a stale earlier record): {', '.join(excluded_stale[:8])}"
              f"{' ...' if len(excluded_stale) > 8 else ''}")
    keep -= set(exclude)
    files = [f for f in files if _subj_of(f) in keep]
    if not files:
        sys.exit(f"no split-'{split}' subjects scored for arm '{method}' in {dataset}")

    ef_rows, ef_agg = _ef_data(dataset, method)
    if ef_rows:
        print(f"  ef/dice chain found for '{method}' — folding biventricular metrics "
              f"({len(ef_rows)} subject(s) in {dataset})")
    rows = []
    for f in files:
        d = json.load(open(f))
        rows.append({
            "subject": d["subject"],
            **_resp_row(dataset, d["subject"], method),
            **ef_rows.get(d["subject"], {}),
            "recon_wall_sec": _wall_sec(dataset, d["subject"], method),
            # `clean` is opt-in, so every clean field may legitimately be absent.
            "clean_psnr": d.get("clean_psnr_mean"), "clean_ssim": d.get("clean_ssim_mean"),
            "clean_ncc": d.get("clean_ncc_mean"), "breath_ncc": d.get("breath_ncc_mean"),
            "breath_psnr": d["breath_psnr_mean"], "breath_ssim": d["breath_ssim_mean"],
            "breath_psnr_unit_peak": d.get("breath_psnr_unit_peak_mean"),
            "clean_psnr_unit_peak": d.get("clean_psnr_unit_peak_mean"),
            "arms": d.get("arms"),
            "cost_psnr": (d["clean_psnr_mean"] - d["breath_psnr_mean"])
                         if "clean_psnr_mean" in d else None,
            "breath_disp_mm": d["breath_mean_disp_mm"],
            "pose": d.get("pose"), "psf": d.get("psf"), "scorer": d.get("scorer"),
            "ckpt": d.get("ckpt"), "ckpt_fingerprint": d.get("ckpt_fingerprint"),
            "stamps_agree": d.get("stamps_agree"),
        })

    # Completeness + provenance checks (a partial or mixed-ckpt cohort must NOT summarize as if
    # whole). One keying mode for the whole cohort: fingerprints only if EVERY ckpt-bearing row
    # has one, else realpath (mirrors run_vggt._same_ckpt).
    ckpt_rows = [r for r in rows if r.get("ckpt")]
    use_fp = bool(ckpt_rows) and all(r.get("ckpt_fingerprint") for r in ckpt_rows)
    def _ckpt_key(r):
        if not r.get("ckpt"):
            return None
        return r["ckpt_fingerprint"] if use_fp else os.path.realpath(r["ckpt"])
    expected, _ = paths.filter_by_split(dataset, paths.subjects(dataset), split)
    missing = sorted(set(expected) - {r["subject"] for r in rows})
    ckpts = sorted({k for r in rows if (k := _ckpt_key(r))})
    if missing:
        print(f"  !! WARNING: {len(rows)}/{len(expected)} subjects scored; MISSING {len(missing)}: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    if len(ckpts) > 1:
        print(f"  !! WARNING: arm '{method}' mixes {len(ckpts)} distinct checkpoints across subjects "
              f"(re-run under a reused name?): {ckpts}")
    # cost_psnr differences two SEPARATELY reconstructed volumes; stamps_agree != True means the
    # clean/breath pair was never verified same-run.
    unverified = sorted(r["subject"] for r in rows
                        if r["cost_psnr"] is not None and r["stamps_agree"] is not True)
    if unverified:
        print(f"  !! WARNING: cost_psnr UNVERIFIED for {len(unverified)}/{len(rows)} subject(s): "
              f"{', '.join(unverified[:8])}{' ...' if len(unverified) > 8 else ''}")

    has_clean = any(r["clean_psnr"] is not None for r in rows)
    print(f"\n=== {dataset} / {method} / split={split}  (n={len(rows)}"
          f"{'' if has_clean else ', breath arm only'}) ===")
    def _f(v, w, p=2):
        return f"{v:>{w}.{p}f}" if v is not None else f"{'n/a':>{w}}"
    hdr = f"{'subject':<40}{'clean':>8}{'breath':>8}{'cost':>7}{'|disp|mm':>9}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["subject"]):
        print(f"{r['subject']:<40}{_f(r['clean_psnr'],8)}"
              f"{_f(r['breath_psnr'],8)}{_f(r['cost_psnr'],7)}{_f(r['breath_disp_mm'],9)}")

    def summarize(subset, label):
        if not subset:
            return None
        cp = stat([r["clean_psnr"] for r in subset]); cs = stat([r["clean_ssim"] for r in subset])
        cn = stat([r["clean_ncc"] for r in subset]); bn = stat([r["breath_ncc"] for r in subset])
        bp = stat([r["breath_psnr"] for r in subset]); bs = stat([r["breath_ssim"] for r in subset])
        ct = stat([r["cost_psnr"] for r in subset]); dz = stat([r["breath_disp_mm"] for r in subset])
        bu = stat([r["breath_psnr_unit_peak"] for r in subset])
        ep = stat([r.get("resp_epe_dz_mm", float("nan")) for r in subset])
        ed = stat([r.get("resp_epe_dz_demeaned_mm", float("nan")) for r in subset])
        sl = stat([r.get("resp_slope", float("nan")) for r in subset])
        ws = stat([r["recon_wall_sec"] if r["recon_wall_sec"] is not None else float("nan")
                   for r in subset])
        print(f"\n[{label}]  n={bp[2]}")
        if cp[2]:
            print(f"  clean : PSNR {cp[0]:6.2f} +- {cp[1]:.2f} dB   SSIM {cs[0]:.3f} +- {cs[1]:.3f}   NCC {cn[0]:.3f} +- {cn[1]:.3f}")
        print(f"  breath: PSNR {bp[0]:6.2f} +- {bp[1]:.2f} dB   SSIM {bs[0]:.3f} +- {bs[1]:.3f}   NCC {bn[0]:.3f} +- {bn[1]:.3f}")
        if bu[2]:
            print(f"  breath: PSNR {bu[0]:6.2f} +- {bu[1]:.2f} dB  [unit-peak, trainer-comparable]")
        if ct[2]:
            print(f"  breathing cost (clean-breath): {ct[0]:.2f} +- {ct[1]:.2f} dB   |disp| {dz[0]:.2f} +- {dz[1]:.2f} mm")
        else:
            print(f"  |disp| {dz[0]:.2f} +- {dz[1]:.2f} mm   (no clean arm -> no breathing-cost delta)")
        if ep[2]:
            print(f"  breathing motion: EPE {ep[0]:.2f} +- {ep[1]:.2f} mm  "
                  f"(demeaned {ed[0]:.2f} +- {ed[1]:.2f} mm)   slope {sl[0]:.2f} +- {sl[1]:.2f}  [n={ep[2]}]")
        if ws[2]:
            print(f"  recon wall-clock: {ws[0]:.1f} +- {ws[1]:.1f} s per 12-phase cine  [n={ws[2]}]")
        # n keys off the BREATH count: the deliverable arm and the only one always present.
        return {"n": bp[2], "n_clean": cp[2],
                "clean_psnr": cp[:2], "clean_ssim": cs[:2], "clean_ncc": cn[:2],
                "breath_psnr": bp[:2], "breath_ssim": bs[:2], "breath_ncc": bn[:2],
                "breath_psnr_unit_peak": bu[:2],
                "cost_psnr": ct[:2], "breath_disp_mm": dz[:2],
                "resp_epe_dz_mm": ep[:2], "resp_epe_dz_demeaned_mm": ed[:2],
                "resp_slope": sl[:2], "n_resp": ep[2],
                "recon_wall_sec": ws[:2], "n_timing": ws[2]}

    summary = {"dataset": dataset, "method": method, "split": split, "n": len(rows),
               "n_expected": len(expected), "missing": missing, "ckpts": ckpts,
               "cost_psnr_unverified": unverified,
               "excluded_stale": excluded_stale,
               # honest provenance: the set of scorers that actually produced the folded rows
               # (migrated pre-rename rows say "score.py", fresh ones "image_metrics.py")
               "scorer": sorted({r.get("scorer") or "unknown" for r in rows}),
               "ef": ef_agg,          # EF/Dice cohort block; null until the seg chain has run
               "all": summarize(rows, "ALL")}
    summary["per_subject"] = rows

    out = paths.summary(dataset, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(json_safe(summary), open(out, "w"), indent=2, allow_nan=False)
    print(f"\n-> {out}")
    return summary


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: aggregate.py <dataset> [method]   datasets: {', '.join(paths.DATASETS)}")
    aggregate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "svrtk3d",
              os.environ.get("SPLIT", "val"))


if __name__ == "__main__":
    main()
