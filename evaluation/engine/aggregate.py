"""Aggregate per-subject metrics.json across a dataset into a cohort summary.

Reads  volumes/<dataset>/out/<subject>/<method>/metrics.json  (produced by
assemble_and_gif.py) and reports cohort clean/breath PSNR/SSIM + breathing-cost
(clean - breath) mean+-std, plus a volunteer-vs-patient split (subjects whose
name contains 'Patient' are pathology; everything else is treated as volunteer/
healthy). CMRxRecon is all-volunteer so it collapses to one group; MIITT is
mixed (10 volunteers + 3 patients).

Run: micromamba run -n svr python evaluation/engine/aggregate.py <dataset> [method=svrtk3d]

Paths/naming go through evaluation/paths.py (the single source of truth).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402


def group_of(name):
    return "patient" if "patient" in name.lower() else "volunteer"


def stat(xs):
    """Mean/std/n over the VALID (non-NaN) values, so one unscorable subject (empty-ROI NaN) doesn't
    poison the whole cohort number. n = count of valid values."""
    xs = np.asarray(xs, dtype=np.float64)
    valid = xs[~np.isnan(xs)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    return float(valid.mean()), float(valid.std()), int(valid.size)


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: aggregate.py <dataset> [method]   datasets: {', '.join(paths.DATASETS)}")
    dataset = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "svrtk3d"
    root = paths.dataset_root(dataset)
    files = sorted(glob.glob(str(root / "*" / method / "metrics.json")))
    if not files:
        sys.exit(f"no metrics found at {root}/*/{method}/metrics.json")

    rows = []
    for f in files:
        d = json.load(open(f))
        s = d["subject"]
        rows.append({
            "subject": s, "group": group_of(s),
            # `clean` is opt-in (run_vggt --arms; default is breath only, the deliverable), so
            # every clean field may legitimately be absent. `.get` rather than `[...]`: a
            # breath-only arm must summarize, not KeyError. `cost_psnr` — the no-breathing ceiling
            # minus the breathing score — simply does not exist without that arm.
            "clean_psnr": d.get("clean_psnr_mean"), "clean_ssim": d.get("clean_ssim_mean"),
            "clean_ncc": d.get("clean_ncc_mean"), "breath_ncc": d.get("breath_ncc_mean"),
            "breath_psnr": d["breath_psnr_mean"], "breath_ssim": d["breath_ssim_mean"],
            # trainer-comparable PSNR (peak=1.0); see assemble_and_gif.psnr_unit_peak
            "breath_psnr_unit_peak": d.get("breath_psnr_unit_peak_mean"),
            "clean_psnr_unit_peak": d.get("clean_psnr_unit_peak_mean"),
            "arms": d.get("arms"),
            "cost_psnr": (d["clean_psnr_mean"] - d["breath_psnr_mean"])
                         if "clean_psnr_mean" in d else None,
            "breath_disp_mm": d["breath_mean_disp_mm"],
            # stamped by assemble_and_gif (None for pre-stamp metrics / baselines)
            "ckpt": d.get("ckpt"), "ckpt_fingerprint": d.get("ckpt_fingerprint"),
        })

    # Completeness + provenance checks (a partial or mixed-ckpt cohort must NOT summarize as if whole).
    # Pick ONE keying mode for the whole cohort (not per-row): use fingerprints only if EVERY
    # ckpt-bearing row has one, else key everything by realpath(path). Per-row keying would give a
    # fingerprinted subject and a legacy path-only subject of the SAME ckpt two different keys -> a
    # false mix warning. Fingerprint mode catches a same-path retrain; realpath mode ignores abs-vs-rel
    # spelling. None (legacy / baseline) drops out either way. (Mirrors run_vggt._same_ckpt's rule.)
    ckpt_rows = [r for r in rows if r.get("ckpt")]
    use_fp = bool(ckpt_rows) and all(r.get("ckpt_fingerprint") for r in ckpt_rows)
    def _ckpt_key(r):
        if not r.get("ckpt"):
            return None
        return r["ckpt_fingerprint"] if use_fp else os.path.realpath(r["ckpt"])
    expected = paths.subjects(dataset)
    missing = sorted(set(expected) - {r["subject"] for r in rows})
    ckpts = sorted({k for r in rows if (k := _ckpt_key(r))})
    if missing:
        print(f"  !! WARNING: {len(rows)}/{len(expected)} subjects scored; MISSING {len(missing)}: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    if len(ckpts) > 1:
        print(f"  !! WARNING: arm '{method}' mixes {len(ckpts)} distinct checkpoints across subjects "
              f"(re-run under a reused name?): {ckpts}")

    # per-subject table (sorted by group then subject)
    # `clean` is opt-in, so a breath-only cohort must print rather than crash on a None format.
    has_clean = any(r["clean_psnr"] is not None for r in rows)
    print(f"\n=== {dataset} / {method}  (n={len(rows)}"
          f"{'' if has_clean else ', breath arm only'}) ===")
    def _f(v, w, p=2):
        return f"{v:>{w}.{p}f}" if v is not None else f"{'n/a':>{w}}"
    hdr = f"{'subject':<40}{'grp':<10}{'clean':>8}{'breath':>8}{'cost':>7}{'|disp|mm':>9}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["group"], r["subject"])):
        print(f"{r['subject']:<40}{r['group']:<10}{_f(r['clean_psnr'],8)}"
              f"{_f(r['breath_psnr'],8)}{_f(r['cost_psnr'],7)}{_f(r['breath_disp_mm'],9)}")

    # cohort + per-group summaries
    def summarize(subset, label):
        if not subset:
            return None
        cp = stat([r["clean_psnr"] for r in subset]); cs = stat([r["clean_ssim"] for r in subset])
        cn = stat([r["clean_ncc"] for r in subset]); bn = stat([r["breath_ncc"] for r in subset])
        bp = stat([r["breath_psnr"] for r in subset]); bs = stat([r["breath_ssim"] for r in subset])
        ct = stat([r["cost_psnr"] for r in subset]); dz = stat([r["breath_disp_mm"] for r in subset])
        bu = stat([r["breath_psnr_unit_peak"] for r in subset])
        print(f"\n[{label}]  n={bp[2]}")
        if cp[2]:      # clean arm present
            print(f"  clean : PSNR {cp[0]:6.2f} +- {cp[1]:.2f} dB   SSIM {cs[0]:.3f} +- {cs[1]:.3f}   NCC {cn[0]:.3f} +- {cn[1]:.3f}")
        print(f"  breath: PSNR {bp[0]:6.2f} +- {bp[1]:.2f} dB   SSIM {bs[0]:.3f} +- {bs[1]:.3f}   NCC {bn[0]:.3f} +- {bn[1]:.3f}")
        if bu[2]:      # trainer-comparable normalization (peak=1.0), for cross-checking val_per_subject.csv
            print(f"  breath: PSNR {bu[0]:6.2f} +- {bu[1]:.2f} dB  [unit-peak, trainer-comparable]")
        if ct[2]:
            print(f"  breathing cost (clean-breath): {ct[0]:.2f} +- {ct[1]:.2f} dB   |disp| {dz[0]:.2f} +- {dz[1]:.2f} mm")
        else:
            print(f"  |disp| {dz[0]:.2f} +- {dz[1]:.2f} mm   (no clean arm -> no breathing-cost delta)")
        # n keys off the BREATH count: it is the deliverable arm and the only one always present.
        # (It used to key off clean, which reports n=0 for a breath-only cohort.)
        return {"n": bp[2], "n_clean": cp[2],
                "clean_psnr": cp[:2], "clean_ssim": cs[:2], "clean_ncc": cn[:2],
                "breath_psnr": bp[:2], "breath_ssim": bs[:2], "breath_ncc": bn[:2],
                "breath_psnr_unit_peak": bu[:2],
                "cost_psnr": ct[:2], "breath_disp_mm": dz[:2]}

    summary = {"dataset": dataset, "method": method, "n": len(rows),
               "n_expected": len(expected), "missing": missing, "ckpts": ckpts,
               "all": summarize(rows, "ALL")}
    for g in ("volunteer", "patient"):
        sub = [r for r in rows if r["group"] == g]
        if sub:
            summary[g] = summarize(sub, g.upper())
    summary["per_subject"] = rows

    out = paths.summary(dataset, method)              # git-tracked results/<ds>/<arm>.json
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
