#!/usr/bin/env python
"""docs/110 arrhythmia pilot: gate quality, GT degeneracy, image metrics, EF — one report.

Five tables, in the order the causal chain runs:

  1. GATING ERROR, DECOMPOSED. How far the self-gated theta is from the truth, measured directly
     against the ORACLE gate — not against `gate.json`'s `diag_vs_manifest_roll`, which compares
     the ED anchor to `rolled.roll_per_plane`. That stored diagnostic is the WINDOW START, not the
     cardiac position of any frame: meaningful for `regular*`, misleading for `hrv`/`af`.

     err[z,f] = angdiff(theta_self - theta_true), global circular mean removed (a constant shared
     by every image only relabels which output bin is "phase 0", and the ref_phase readout samples
     the same shifted theta, so it cancels exactly). What remains splits in two:

       per-slice constant  — the ED-anchor error. NOT harmless: it mis-registers slices against
                             each other, the thing the paper's inter-slice sync fixes and our
                             single-orientation SAX adaptation cannot (docs/34/35).
       within-slice residual — the STRUCTURAL cost of the constant-heart-rate assumption. It is
                             exactly 0 for any periodic rhythm (both thetas are then uniform ramps,
                             so their difference is a per-slice constant by construction — the
                             `regular*` rows are therefore a self-check on this decomposition) and
                             no choice of ED anchor can reduce it.

  2. MIS-BINNING. The engine pools images into 12 phase bins by theta, so a bin error is the
     mechanism by which bad gating damages the reconstruction. `collisions` counts pairs of images
     that are truly at DIFFERENT phases but get pooled into the SAME bin.

  3. GT DEGENERACY. Under `af` a window can hold at full ED, making targets byte-identical. Such a
     subject inflates PSNR (a near-constant volume is easy) AND destroys EF, so it is flagged and
     excluded from the cohort means in tables 4-5 (reported separately, never silently averaged).

  4. IMAGE METRICS — `<arm>/metrics.json` from score/run.py.
  5. EF — `metric_results/<split>/_ef/<arm>.json`.

Usage:
    PYTHONPATH=training:. python tools/af_pilot_report.py [--split test] [--json OUT.json]
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

T = 12
COHORTS = ["cmrx2024_regular_frozen", "cmrx2024_regular", "cmrx2024_hrv", "cmrx2024_af"]
# Full cmrx2024 test split (38 subjects), not the 4-subject pilot list. A hardcoded pilot list
# here previously meant the degenerate-window exclusion (table 3/4/5) only ever fired on P004 --
# 4 more real degenerate af subjects (P013, P023, P038, P026, found by full-cohort review) would
# have been silently averaged into the cohort means instead of excluded. Derived dynamically so
# it can never drift from the actual test split again.
SUBJECTS = None               # resolved in main() via paths.filter_by_split; keep the name for
                               # anyone importing this module directly (falls back to the pilot's
                               # 4 subjects only if paths can't be imported at all).
_PILOT_SUBJECTS = ["CMRx24_Test_P016", "CMRx24_Test_P004", "CMRx24_Test_P017", "CMRx24_Train_P046"]
FETAL = ["fetal_cmr_4d", "fetal_cmr_4d_oracle", "fetal_cmr_4d_oracle_balanced"]
VGGT = "vggt_final518_diff1000_ep300"
DEGEN_DUP_PAIRS = 5          # > this many byte-identical GT pairs (of 66) => degenerate window


def ang(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def circ_mean(x):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(x)))))


def read_theta(cohort, subj, arm):
    p = paths.subject_dir(cohort, subj) / arm / "gate" / "cardphase.txt"
    if not p.is_file():
        return None
    return np.array(open(p).read().split(), float).reshape(-1, T)


def gate_stats(cohort, subj):
    a = read_theta(cohort, subj, "fetal_cmr_4d")
    b = read_theta(cohort, subj, "fetal_cmr_4d_oracle")
    if a is None or b is None or a.shape != b.shape:
        return None
    k = T / (2 * np.pi)
    e = ang(a - b)
    e = ang(e - circ_mean(e.ravel()))
    off = np.array([circ_mean(e[z]) for z in range(e.shape[0])])
    res = ang(e - off[:, None])
    db = np.round(e * k).astype(int)                       # bin displacement per image
    tb = np.round(b * k).astype(int) % T
    sb = np.round(a * k).astype(int) % T
    tb, sb = tb.ravel(), sb.ravel()
    n = tb.size
    iu = np.triu_indices(n, 1)
    coll = float(((tb[iu[0]] != tb[iu[1]]) & (sb[iu[0]] == sb[iu[1]])).mean())
    return {"D": int(a.shape[0]),
            "mean_abs_frames": float(np.abs(e * k).mean()),
            "rms_frames": float(np.sqrt(((e * k) ** 2).mean())),
            "rms_slice_const": float(np.sqrt(((off * k) ** 2).mean())),
            "rms_within_slice": float(np.sqrt(((res * k) ** 2).mean())),
            "misbinned_frac": float((db != 0).mean()),
            "mean_abs_bin_err": float(np.abs(db).mean()),
            "collision_frac": coll}


def gt_stats(cohort, subj):
    d = paths.subject_dir(cohort, subj)
    mp = d / "mask_heart_pad10.nii.gz"
    if not mp.is_file():
        return None
    m = np.asarray(nib.load(str(mp)).dataobj) > 0.5
    try:
        V = np.stack([np.asarray(nib.load(str(d / "gt" / f"gt_t{f:02d}.nii.gz")).dataobj,
                                 dtype=np.float32) for f in range(T)])
    except FileNotFoundError:
        return None
    v = V[:, m]
    iu = np.triu_indices(T, 1)
    r = np.sqrt(((v[iu[0]] - v[iu[1]]) ** 2).mean(1))
    man = json.load(open(paths.manifest(cohort, subj)))
    pos = np.asarray(man["rhythm"]["ref_pos"], float)
    span = float(np.ptp(np.unwrap(pos * 2 * np.pi / T) * T / (2 * np.pi)))
    dup = int((r < 1e-9).sum())
    return {"pairwise_rms": float(r.mean()), "dup_pairs": dup, "pos_span_frames": span,
            "degenerate": dup > DEGEN_DUP_PAIRS}


def read_metrics(cohort, subj, arm):
    p = paths.subject_dir(cohort, subj) / arm / "metrics.json"
    return json.load(open(p)) if p.is_file() else None


def read_ef(split, arm):
    """-> {(cohort, subject): (pred_ef, gt_ef)} from the _ef json's per_subject rows.

    Per-subject, not the `aggregate` block: the aggregate is computed over every subject in the
    cohort including degenerate ones, and a held AF window drives EF to nonsense (P004: GT EF
    collapses 70.6 -> 14.4), so it must be excluded on the same rule table 3 applies.
    NOTE the _ef json is SHARED with the live campaign and merges per cohort, so it also holds
    the published rows — filter to the cohorts asked for and never rewrite this file by hand.
    """
    p = os.path.join(ROOT, "evaluation", "metric_results", split, "_ef", f"{arm}.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    out = {}
    for r in d.get("per_subject", []):
        if r.get("ef_breath") is not None and r.get("ef_gt") is not None:
            out[(r.get("cohort"), r.get("subject"))] = (r["ef_breath"], r["ef_gt"])
    return out


def table(rows, headers):
    if not rows:
        return "  (no data yet)"
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    out = ["  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)),
           "  ".join("-" * w[i] for i in range(len(headers)))]
    for r in rows:
        out.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--cohorts", nargs="+", default=COHORTS)
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="default: the full cmrx2024 <split> subject list (38 for test), not the "
                         "4-subject pilot set. Pass --subjects <pilot 4> to reproduce the pilot report.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.subjects is None:
        try:
            args.subjects, _ = paths.filter_by_split("cmrx2024", paths.subjects("cmrx2024"), args.split)
        except Exception as e:
            print(f"  !! could not resolve the full {args.split} split ({e}); "
                  f"falling back to the 4-subject pilot list", file=sys.stderr)
            args.subjects = _PILOT_SUBJECTS
    short = lambda c: c.replace("cmrx2024_", "")            # noqa: E731
    blob = {"gate": {}, "gt": {}, "image": {}, "ef": {}}

    # ---- 3 first (it defines which subjects the later means may use) ----
    degen = set()
    gtrows = []
    for c in args.cohorts:
        for s in args.subjects:
            g = gt_stats(c, s)
            if not g:
                continue
            blob["gt"].setdefault(c, {})[s] = g
            if g["degenerate"]:
                degen.add((c, s))
            gtrows.append([short(c), s.replace("CMRx24_", ""), f"{g['pairwise_rms']:.4f}",
                           g["dup_pairs"], f"{g['pos_span_frames']:.2f}",
                           "DEGENERATE" if g["degenerate"] else ""])

    print("\n=== 1. GATING ERROR, DECOMPOSED  (self-gated theta vs truth; units = cine frames) ===")
    print("    within-slice residual is the STRUCTURAL cost — exactly 0 for a periodic rhythm,")
    print("    and no ED anchor can reduce it.\n")
    rows = []
    for c in args.cohorts:
        per = [g for g in (gate_stats(c, s) for s in args.subjects) if g]
        for s, g in zip([s for s in args.subjects if gate_stats(c, s)], per):
            blob["gate"].setdefault(c, {})[s] = g
        if per:
            rows.append([short(c), len(per),
                         f"{np.mean([p['mean_abs_frames'] for p in per]):.3f}",
                         f"{np.mean([p['rms_frames'] for p in per]):.3f}",
                         f"{np.mean([p['rms_slice_const'] for p in per]):.3f}",
                         f"{np.mean([p['rms_within_slice'] for p in per]):.3f}"])
    print(table(rows, ["cohort", "n", "mean|err|", "rms", "slice-const", "WITHIN-SLICE"]))

    print("\n\n=== 2. MIS-BINNING  (how bad gating actually damages the reconstruction) ===\n")
    rows = []
    for c in args.cohorts:
        per = [g for g in (gate_stats(c, s) for s in args.subjects) if g]
        if per:
            rows.append([short(c), len(per),
                         f"{np.mean([p['misbinned_frac'] for p in per]) * 100:.1f}%",
                         f"{np.mean([p['mean_abs_bin_err'] for p in per]):.2f}",
                         f"{np.mean([p['collision_frac'] for p in per]) * 100:.1f}%"])
    print(table(rows, ["cohort", "n", "misbinned", "mean|bin err|", "collisions"]))

    print("\n\n=== 3. GT DEGENERACY  (a held window makes targets identical) ===\n")
    print(table(gtrows, ["cohort", "subject", "pairwise RMS", "dup/66", "span", "flag"]))
    if degen:
        print(f"\n  EXCLUDED from cohort means below: {sorted((short(c), s) for c, s in degen)}")

    # PAIRED subject set: a subject degenerate in ANY cohort is dropped from EVERY cohort.
    # Excluding per (cohort, subject) instead would average different subject sets across cohorts
    # and make the columns non-comparable. That is not hypothetical: P004 is degenerate only under
    # `af` and scores 28-30 dB against everyone else's 21-26, so the per-cell rule reports
    # af - regular_frozen as -2.09 dB for fetal self and -1.11 dB for VGGT, versus the paired
    # -0.60 and +0.03 -- a SIGN FLIP on VGGT, and it flips the fetal EF direction too.
    paired = [s for s in args.subjects if not any((c, s) in degen for c in args.cohorts)]
    dropped = [s for s in args.subjects if s not in paired]
    print(f"\n\n=== 4. IMAGE METRICS  (breath arm, headline psnr_unit_peak) ===")
    print(f"    PAIRED subject set, n={len(paired)}: {[s.replace('CMRx24_', '') for s in paired]}")
    if dropped:
        print(f"    dropped from EVERY cohort (degenerate somewhere): "
              f"{[s.replace('CMRx24_', '') for s in dropped]}\n")
    rows = []
    for c in args.cohorts:
        for arm in FETAL + [VGGT]:
            keys = ("breath_psnr_unit_peak_mean", "breath_psnr_mean",
                    "breath_ssim_mean", "breath_ncc_mean")
            vals = {k: [] for k in keys}
            for s in args.subjects:
                m = read_metrics(c, s, arm)
                if not m:
                    continue
                blob["image"].setdefault(c, {}).setdefault(arm, {})[s] = {k: m.get(k) for k in keys}
                if s not in paired:
                    continue
                for k in keys:
                    if m.get(k) is not None:
                        vals[k].append(m[k])
            if vals[keys[0]]:
                rows.append([short(c), arm, len(vals[keys[0]]),
                             f"{np.mean(vals[keys[0]]):.2f}", f"{np.mean(vals[keys[1]]):.2f}",
                             f"{np.mean(vals[keys[2]]):.3f}", f"{np.mean(vals[keys[3]]):.3f}"])
    print(table(rows, ["cohort", "arm", "n", "PSNR_unit", "PSNR_roi", "SSIM", "NCC"]))

    print("\n\n=== 5. EF  (same PAIRED subject set as table 4) ===")
    print("    regular_frozen vs regular is a PERTURBATION-SENSITIVITY BOUND, not a pure noise")
    print("    floor: frame-wise breathing is itself a real ~-0.27 dB effect (docs/111 s4), so the")
    print("    gap mixes a systematic term with noise. Read no rhythm effect smaller than it.\n")
    rows = []
    for arm in FETAL + [VGGT]:
        ef = read_ef(args.split, arm)
        if not ef:
            continue
        for c in args.cohorts:
            use = [(s, ef[(c, s)]) for s in args.subjects
                   if (c, s) in ef and s in paired]
            if not use:
                continue
            err = [abs(p - g) for _, (p, g) in use]
            blob["ef"].setdefault(c, {})[arm] = {s: v for s, v in use}
            rows.append([short(c), arm, len(use),
                         f"{np.mean(err):.2f}", f"{np.mean([p - g for _, (p, g) in use]):+.2f}",
                         f"{np.mean([p for _, (p, _) in use]):.1f}",
                         f"{np.mean([g for _, (_, g) in use]):.1f}"])
    print(table(rows, ["cohort", "arm", "n", "EF MAE", "EF bias", "pred EF", "GT EF"]))
    for arm in FETAL + [VGGT]:
        ef = read_ef(args.split, arm)
        if not ef:
            continue
        # same paired set as the table above, or the bound and the table would use different
        # subjects -- exactly the mismatch this rewrite exists to remove.
        pair = [(s, ef[("cmrx2024_regular_frozen", s)], ef[("cmrx2024_regular", s)])
                for s in paired
                if ("cmrx2024_regular_frozen", s) in ef and ("cmrx2024_regular", s) in ef]
        if pair:
            d = [abs(a[0] - b[0]) for _, a, b in pair]
            print(f"  SENSITIVITY BOUND {arm:30s} (n={len(pair)}) |dEF| frozen-vs-framewise: "
                  f"mean {np.mean(d):.1f} pp, max {np.max(d):.1f} pp")

    if args.json:
        json.dump(blob, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
