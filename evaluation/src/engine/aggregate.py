"""Aggregate per-subject metrics.json across a dataset into a cohort summary.

Reads  volumes/<dataset>/out/<subject>/<method>/metrics.json  (produced by
assemble_and_gif.py) and reports cohort clean/breath PSNR/SSIM + breathing-cost
(clean - breath) mean+-std.

There is deliberately NO volunteer-vs-patient split. It used to key off `"patient"
in subject.lower()`, which is a property of the NAME, not the subject: measured
against `training/splits/manifest.csv`'s `pathology_label`, that rule mislabelled
64 of 136 eval subjects — all 37 cmrx2025 (diseased, reported as volunteers), 24 of
33 mnms, and 3 healthy ACDC patients reported as pathology. Every pooled cohort is
also single-group under that rule, so the block only ever duplicated ALL. If a
pathology split is wanted, join `manifest.csv:pathology_label` — the ground truth
is already in the repo — rather than parsing the name.

Only subjects whose `manifest["split"]` matches `$SPLIT` (default "val") are summarized — see
`paths.filter_by_split` for why that must be enforced here and not only in run_vggt.

Run: [SPLIT=val] micromamba run -n svr python evaluation/src/engine/aggregate.py <dataset> [method=svrtk3d]

Paths/naming go through evaluation/paths.py (the single source of truth).
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
    """NaN/Inf -> None, recursively. `json.dump` writes the bare token `NaN`, which Python reads
    back but every strict JSON parser (jq, JS, most tooling) rejects — and these summaries are the
    git-tracked citable numbers. A cohort with no `clean` arm legitimately has no clean mean, so
    `null` is also the honest encoding of it."""
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


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
    split = os.environ.get("SPLIT", "val")
    root = paths.dataset_root(dataset)
    files = sorted(glob.glob(str(root / "*" / method / "metrics.json")))
    if not files:
        sys.exit(f"no metrics found at {root}/*/{method}/metrics.json")

    # A cohort is defined by its SPLIT, and nothing in the layout enforces that: neither the bundle
    # dir nor metric_results/<ds>/<arm>.json is split-keyed, so a test-split bundle scored into the same
    # tree would be averaged into the val numbers with no warning — the one silently-wrong-number
    # path in the harness. run_vggt already honours manifest["split"]; the summary must too, via the
    # same helper so the two cannot drift.
    def _subj_of(f):
        return os.path.basename(os.path.dirname(os.path.dirname(f)))
    keep, dropped = paths.filter_by_split(dataset, [_subj_of(f) for f in files], split)
    if dropped:
        print(f"  !! EXCLUDED {len(dropped)} scored subject(s) not in split '{split}' "
              f"(set SPLIT=<other> to summarize a different split):")
        for s, why in dropped:
            print(f"       {s}: {why}")
    keep = set(keep)
    files = [f for f in files if _subj_of(f) in keep]
    if not files:
        sys.exit(f"no split-'{split}' subjects scored for arm '{method}' in {dataset}")

    rows = []
    for f in files:
        d = json.load(open(f))
        s = d["subject"]
        rows.append({
            "subject": s,
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
            # check_variant_stamps' verdict: True only if every present arm carries an identical
            # per-variant stamp. False = warned-but-scored (pre-stamp run, or ALLOW_MIXED_ARMS=1);
            # absent = scored before the key existed. Either way `cost_psnr` is UNVERIFIED.
            "stamps_agree": d.get("stamps_agree"),
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
    # split-filtered too, else a built-but-other-split bundle reads as a MISSING val subject
    expected, _ = paths.filter_by_split(dataset, paths.subjects(dataset), split)
    missing = sorted(set(expected) - {r["subject"] for r in rows})
    ckpts = sorted({k for r in rows if (k := _ckpt_key(r))})
    if missing:
        print(f"  !! WARNING: {len(rows)}/{len(expected)} subjects scored; MISSING {len(missing)}: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    if len(ckpts) > 1:
        print(f"  !! WARNING: arm '{method}' mixes {len(ckpts)} distinct checkpoints across subjects "
              f"(re-run under a reused name?): {ckpts}")
    # `cost_psnr` differences two SEPARATELY reconstructed volumes, so a stale recon_clean/ left by a
    # `--arms breath` re-run makes it measure checkpoint drift, not breathing. check_variant_stamps
    # already RAISES on the dangerous mixed-stamp case; the soft cases (pre-stamp run,
    # ALLOW_MIXED_ARMS=1) only warn and land as stamps_agree=False, which nothing read until now — an
    # unverified cohort's summary was indistinguishable from a verified one. Flag, don't fail:
    # ALLOW_MIXED_ARMS exists precisely so a known-good mix can still be scored.
    unverified = sorted(r["subject"] for r in rows
                        if r["cost_psnr"] is not None and r["stamps_agree"] is not True)
    if unverified:
        print(f"  !! WARNING: cost_psnr UNVERIFIED for {len(unverified)}/{len(rows)} subject(s) — "
              f"clean/breath recons not confirmed same-run (pre-stamp, or ALLOW_MIXED_ARMS=1): "
              f"{', '.join(unverified[:8])}{' ...' if len(unverified) > 8 else ''}")

    # per-subject table (sorted by group then subject)
    # `clean` is opt-in, so a breath-only cohort must print rather than crash on a None format.
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

    # cohort summary
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

    summary = {"dataset": dataset, "method": method, "split": split, "n": len(rows),
               "n_expected": len(expected), "missing": missing, "ckpts": ckpts,
               "cost_psnr_unverified": unverified,
               "all": summarize(rows, "ALL")}
    summary["per_subject"] = rows

    out = paths.summary(dataset, method)              # git-tracked metric_results/<ds>/<arm>.json
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(json_safe(summary), open(out, "w"), indent=2, allow_nan=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
