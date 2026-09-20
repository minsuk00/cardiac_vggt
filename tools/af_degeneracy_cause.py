#!/usr/bin/env python
"""docs/111 open item: WHY are 5/38 `af` subjects degenerate? Only P004's cause was confirmed.

A subject is degenerate when its 12 GT targets contain byte-identical duplicates: the GT for frame
f is the clean volume at the REFERENCE plane's true cardiac position for frame f, so duplicates
appear exactly when that plane's position trace repeats -- i.e. when the heart is parked at the
full-ED hold (`pos == T`) for several consecutive frames. docs/111 s10.2 confirmed the RATE is real
(Monte Carlo 15.4% vs measured 13.2%) but never identified which subjects, or why those.

The suspect is the disclosed window-open-phase limitation (docs/110 s11.5): `simulate`'s `t0` is a
TIME offset into beat BURN, and under `pos_physio` (af only) that maps to a stored phase of
`min(p_start + frac*rr*T, T)` -- so an `af` window can open already parked at the hold, which no
other arm can do. P004's cause was traced to exactly this.

Three questions, three measurements:
  1. Does opening inside the hold predict degeneracy across all 38 subjects? (contingency)
  2. Is it the WINDOW, or the subject's beat draw? Counterfactual: keep each subject's beats
     (same RNG stream) and sweep the open time across beat BURN; how often is the result
     degenerate? A subject that is degenerate at almost any open time is intrinsically degenerate;
     one that is degenerate only near its actual t0 was made degenerate by the window.
  3. How much of the cohort-wide degeneracy is attributable to the window opening in a hold?

Self-check: the recomputed reference-plane trace must equal the manifest's stored `rhythm.ref_pos`
(the positions the bundle was actually built with) -- printed, and a hard assert.

Usage:  PYTHONPATH=training:. python tools/af_degeneracy_cause.py [--sweep 400]
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import paths  # noqa: E402
from build_af_bundle import BURN, DT, N_BEATS, T, pos_physio, rr_sequence  # noqa: E402

COHORT = "cmrx2024_af"
SRC = "cmrx2024"             # the SOURCE bundle the af cohort was built from
DEGEN_DUP_PAIRS = 5          # same threshold as af_pilot_report.py


def dup_pairs(pos):
    """Pairs of the 12 targets that land on the SAME cardiac position => identical GT volumes.

    Matches `af_pilot_report.gt_stats`' voxel-level count: two targets rendered from the same
    float position are byte-identical, and (verified there) nothing else produces a 0.0 RMS pair.
    """
    p = np.asarray(pos, float)
    iu = np.triu_indices(len(p), 1)
    return int((np.abs(p[iu[0]] - p[iu[1]]) < 1e-9).sum())


def lv_curve(subj):
    """(T,) whole-LV voxel count per GT phase -- `pos_physio`'s volume-matched resume needs it.

    Read from the SOURCE cohort, exactly as `build_af_bundle.load_subject` did: the af bundle's
    own seg_gt/ is a gt_map-PERMUTED copy, so its curve is in a different phase order and would
    silently feed pos_physio a different volume trajectory.
    """
    d = paths.subject_dir(SRC, subj)
    segs = sorted(glob.glob(str(d / "seg_gt" / "seg_t*.nii.gz")))
    if len(segs) != T:
        return None
    v = np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in segs], float)
    return v if v.max() > 0 else None


def trace(vol, rr, t0, n=T):
    """The reference plane's position trace for a window opening at absolute time `t0`."""
    pos, _ = pos_physio(t0 + np.arange(n) * DT, rr, vol)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=400,
                    help="open times sampled uniformly across beat BURN for the counterfactual")
    ap.add_argument("--seeds", type=int, default=0,
                    help="if >0, also re-draw each subject's beats with this many alternative "
                         "seeds (at the subject's own open rule) to separate 'this subject's LV "
                         "curve can never hold' from 'this seed's beat draw was unlucky'")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    subs, _ = paths.filter_by_split("cmrx2024", paths.subjects("cmrx2024"), "test")
    rows, blob = [], {}
    for s in subs:
        man = json.load(open(paths.manifest(COHORT, s)))
        ref = int(man["rhythm"]["ref_plane"])
        seed = int(man["seed"])
        roll = man["rolled"]["roll_per_plane"][ref]
        stored = np.asarray(man["rhythm"]["ref_pos"], float)
        vol = lv_curve(s)
        if vol is None:
            print(f"  !! {s}: no usable LV curve, skipped", file=sys.stderr)
            continue

        # Reproduce the bundle's own draw exactly: one private stream per (subject, plane).
        rng = np.random.default_rng([seed, ref])
        rr = rr_sequence("af", N_BEATS, rng)
        t0 = rr[:BURN].sum() + ((-roll) % T) / T * rr[BURN]
        got = trace(vol, rr, t0)
        assert np.allclose(got, stored, atol=1e-9), \
            f"{s}: recomputed ref trace != manifest rhythm.ref_pos (max {np.abs(got-stored).max():.3g})"

        dp = dup_pairs(stored)
        degen = dp > DEGEN_DUP_PAIRS
        opens_in_hold = bool(stored[0] >= T - 1e-9)
        hold_frames = int((stored >= T - 1e-9).sum())

        # Counterfactual: same beats, sweep the open time across the whole of beat BURN.
        t_lo = rr[:BURN].sum()
        sweep = t_lo + np.linspace(0.0, 1.0, args.sweep, endpoint=False) * rr[BURN]
        d_sweep = np.array([dup_pairs(trace(vol, rr, t)) for t in sweep])
        p_degen = float((d_sweep > DEGEN_DUP_PAIRS).mean())
        # Restrict to open times that do NOT start in the hold -- the "window-fixed" counterfactual.
        starts_hold = np.array([trace(vol, rr, t, n=1)[0] >= T - 1e-9 for t in sweep])
        p_degen_nohold = (float((d_sweep[~starts_hold] > DEGEN_DUP_PAIRS).mean())
                          if (~starts_hold).any() else float("nan"))

        # Re-draw the beats themselves. Same subject (same LV curve, same roll => same open rule),
        # different RNG stream. Separates a subject that CANNOT be degenerate (its LV curve never
        # lets the volume-matched resume park) from one that merely drew a benign rhythm.
        p_degen_seed = float("nan")
        if args.seeds > 0:
            hits = 0
            for k in range(args.seeds):
                r2 = rr_sequence("af", N_BEATS, np.random.default_rng([seed, ref, 10_000 + k]))
                t2 = r2[:BURN].sum() + ((-roll) % T) / T * r2[BURN]
                hits += dup_pairs(trace(vol, r2, t2)) > DEGEN_DUP_PAIRS
            p_degen_seed = hits / args.seeds

        blob[s] = {"dup_pairs": dp, "degenerate": degen, "open_phase": float(stored[0]),
                   "p_degen_reseed": p_degen_seed,
                   "opens_in_hold": opens_in_hold, "hold_frames": hold_frames,
                   "p_degen_sweep": p_degen, "p_degen_sweep_nohold": p_degen_nohold,
                   "frac_open_in_hold": float(starts_hold.mean())}
        rows.append((s, dp, degen, stored[0], opens_in_hold, hold_frames,
                     p_degen, p_degen_nohold, float(starts_hold.mean()), p_degen_seed))

    # ---------------- report ----------------
    print(f"\n=== per-subject ({len(rows)} af subjects; recomputed traces match manifest) ===\n")
    hdr = ("subject", "dup/66", "DEGEN", "open", "in-hold", "holds", "P(deg|sweep)",
           "P(deg|no-hold)", "P(open in hold)", "P(deg|reseed)")
    print("  ".join(h.ljust(w) for h, w in zip(hdr, [10, 6, 5, 6, 7, 5, 12, 14, 15, 13])))
    for s, dp, dg, op, ih, hf, pd, pdn, foh, pds in sorted(rows, key=lambda r: -r[1]):
        print("  ".join([s.replace("CMRx24_", "").ljust(10), str(dp).ljust(6),
                         ("YES" if dg else "").ljust(5), f"{op:.2f}".ljust(6),
                         ("yes" if ih else "no").ljust(7), str(hf).ljust(5),
                         f"{pd:.3f}".ljust(12), f"{pdn:.3f}".ljust(14), f"{foh:.3f}".ljust(15),
                         f"{pds:.3f}".ljust(13)]))

    dg = np.array([r[2] for r in rows])
    ih = np.array([r[4] for r in rows])
    print(f"\n=== Q1. contingency: opens inside the hold  x  degenerate  (n={len(rows)}) ===\n")
    print(f"                 degenerate   not")
    print(f"  opens in hold    {int((dg & ih).sum()):5d}      {int((~dg & ih).sum()):5d}")
    print(f"  opens normally   {int((dg & ~ih).sum()):5d}      {int((~dg & ~ih).sum()):5d}")
    if ih.any():
        print(f"\n  P(degenerate | opens in hold)  = {dg[ih].mean():.3f}  ({int(dg[ih].sum())}/{int(ih.sum())})")
    if (~ih).any():
        print(f"  P(degenerate | opens normally) = {dg[~ih].mean():.3f}  ({int(dg[~ih].sum())}/{int((~ih).sum())})")
    try:
        from scipy.stats import fisher_exact
        odds, p = fisher_exact([[int((dg & ih).sum()), int((~dg & ih).sum())],
                                [int((dg & ~ih).sum()), int((~dg & ~ih).sum())]])
        print(f"  Fisher exact: odds ratio {odds:.3g}, p = {p:.3g}")
    except Exception as e:                                        # scipy optional
        print(f"  (no Fisher test: {e})")

    pd_all = np.array([r[6] for r in rows])
    pdn_all = np.array([r[7] for r in rows])
    print(f"\n=== Q2. counterfactual: same beats, window opened elsewhere in beat {BURN} ===\n")
    print(f"  mean P(degenerate) over all open times      : {np.nanmean(pd_all):.3f}")
    print(f"  mean P(degenerate) EXCLUDING hold openings  : {np.nanmean(pdn_all):.3f}")
    print(f"  mean P(window opens inside a hold)          : {np.mean([r[8] for r in rows]):.3f}")
    print(f"\n  degenerate subjects only:")
    for s, dp, d_, op, ih_, hf, pdd, pdn, foh, pds in sorted(rows, key=lambda r: -r[1]):
        if d_:
            print(f"    {s.replace('CMRx24_',''):10s} P(deg|sweep)={pdd:.3f}  "
                  f"P(deg|no-hold open)={pdn:.3f}  P(deg|reseed)={pds:.3f}")

    pdseed = np.array([r[9] for r in rows], float)
    if np.isfinite(pdseed).any():
        print(f"\n=== Q2b. re-drawn beats, subject's own open rule ({args.seeds} seeds each) ===\n")
        print(f"  cohort mean P(degenerate) over re-draws : {np.nanmean(pdseed):.3f}"
              f"   (measured this seed: {dg.mean():.3f})")
        n0 = int((pdseed < 1e-9).sum())
        print(f"  subjects that are degenerate under NO re-draw : {n0}/{len(rows)}")
        print(f"  subjects degenerate under >50% of re-draws    : {int((pdseed > 0.5).sum())}/{len(rows)}")
        print(f"  P(deg|reseed) among the 5 measured-degenerate : "
              f"{np.nanmean(pdseed[dg]):.3f}  vs  {np.nanmean(pdseed[~dg]):.3f} among the rest")

    print(f"\n=== Q3. attributable fraction ===\n")
    print(f"  measured degenerate rate                    : {dg.mean():.3f} ({int(dg.sum())}/{len(rows)})")
    print(f"  expected if windows never opened in a hold  : {np.nanmean(pdn_all):.3f}")
    if args.json:
        json.dump(blob, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
