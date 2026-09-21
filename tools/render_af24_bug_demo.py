#!/usr/bin/env python
"""Picture of the two pos_physio defects found in the 2026-09-20 review, and the fix.

Three panels, one real subject each:
  A  the subject's LV curve with `es = argmin(vol)`. ES is LATE here (es=10) and it is a REAL
     minimum, not an argmin-on-a-plateau artifact (frame 10 is ~9 % below frame 8 -- an earlier
     version of this panel claimed a tie and was wrong). Late ES leaves only T - es = 2 phases of
     relaxation, which is what both defects below feed on.
  B  the teleport. A beat whose R-R is shorter than that long contraction is cut mid-systole.
     The shipped code clamps its end position UP to `es`, so the next beat resumes FULLY
     contracted -- the position jumps forward, skipping late systole. The fix resumes at the phase
     the heart actually reached.
  C  the freeze. With only `T - es` = 2 phases of relaxation left to stretch across a whole beat,
     the position advances ~0.17 phase/frame -- the heart is effectively static for 12+ frames.
     This is the same pathology the "no full-ED hold" rewrite removed, arriving a different way.

Usage: PYTHONPATH=training:. python tools/render_af24_bug_demo.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import nibabel as nib                                              # noqa: E402
import numpy as np                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation"),
                os.path.join(ROOT, "tools")]
import paths                                                       # noqa: E402
from build_af_bundle import BURN, DT, N_BEATS, T, rr_sequence      # noqa: E402

NF = 24
OUT = os.path.join(ROOT, "temp", "af24_bug_demo")
TELE = ("cmrx2025", "CMRx25_R1test_Center004_UIH_15T_umr680_P030")   # worst forward jump, +5.08
FREEZE = ("cmrx2025", "CMRx25_R2val_Center004_UIH_15T_umr680_P017")  # worst static run, 16 frames


def lv(ds, subj):
    f = sorted(glob.glob(str(paths.subject_dir(ds, subj) / "seg_gt" / "seg_t*.nii.gz")))
    return np.array([(np.asarray(nib.load(p).dataobj) == 1).sum() for p in f], float)


def simulate(vol, es, rr, times, fixed):
    """Returns (pos, beat). `fixed` resumes a systole-cut beat at the phase actually reached."""
    vol_p = np.r_[vol, vol[0]]
    sys_v = np.minimum.accumulate(vol[:es + 1])
    starts = np.concatenate([[0.0], np.cumsum(rr)])
    p_start = np.zeros(len(rr))
    for i in range(len(rr) - 1):
        p0 = p_start[i]
        t_c = max(es - p0, 0.0) * DT
        t_nom = t_c + (T - es) * DT
        cut_systole = rr[i] <= t_c
        if rr[i] >= t_nom:
            p_end = float(T)
        elif cut_systole:
            p_end = p0 + rr[i] / DT
        else:
            p_end = es + (rr[i] - t_c) / DT
        if fixed and cut_systole:
            p_start[i + 1] = p_end            # it already IS a systolic phase; nothing to invert
            continue
        p_end = max(min(p_end, float(T)), float(es))          # <- the clamp that teleports
        vv = np.interp(p_end, np.arange(T + 1), vol_p)
        p_start[i + 1] = (0.0 if vv >= sys_v[0]
                          else np.interp(vv, sys_v[::-1], np.arange(es + 1)[::-1]))
    b = np.searchsorted(starts, times, side="right") - 1
    el = times - starts[b]
    pos = np.empty(len(times))
    for k in range(len(times)):
        i = int(b[k]); p0 = p_start[i]; e = el[k]
        t_c = max(es - p0, 0.0) * DT
        t_nom = t_c + (T - es) * DT
        if e <= t_c:
            pos[k] = p0 + e / DT
        elif rr[i] >= t_nom:
            pos[k] = es + min((e - t_c) / max(rr[i] - t_c, 1e-9), 1.0) * (T - es)
        else:
            pos[k] = min(es + (e - t_c) / DT, float(T))
    return pos, b


def worst_plane(ds, subj, vol, es, key):
    """The plane showing the defect most clearly. key in {'jump','static'}."""
    man = json.load(open(paths.manifest(f"{ds}_af24", subj)))
    roll = man["rolled"]["roll_per_plane"]; seed = int(man["seed"])
    best, best_z, best_rr, best_t0 = -1, 0, None, 0.0
    for z in range(int(man["D"])):
        rng = np.random.default_rng([seed, z])
        rr = rr_sequence("af", N_BEATS, rng)
        t0 = rr[:BURN].sum() + ((-roll[z]) % T) / T * rr[BURN]
        pos, b = simulate(vol, es, rr, t0 + np.arange(NF) * DT, fixed=False)
        d = np.diff(pos); same = np.diff(b) == 0
        if key == "jump":
            v = max([d[k] for k in range(len(d)) if not same[k]] or [-1])
        else:
            run = v = 0
            for k in range(len(d)):
                run = run + 1 if (same[k] and abs(d[k]) < 0.25) else 0
                v = max(v, run)
        if v > best:
            best, best_z, best_rr, best_t0 = v, z, rr, t0
    return best_z, best_rr, best_t0


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # --- A: the LV curve and its (late, genuine) ES --------------------------------------
    ds, subj = FREEZE
    vol = lv(ds, subj); n = vol / vol.max()
    es_arg = int(np.argmin(vol))
    ax[0].plot(range(T), n, "o-", color="#2f6f4e", lw=2)
    ax[0].axvline(es_arg, color="#d9534f", lw=2, label=f"es = argmin = {es_arg}")
    ax[0].set_title(f"A  late ES: only T - es = {T - es_arg} phases of relaxation\n"
                    f"(a real minimum, not a tie)")
    ax[0].set_xlabel("stored cine phase"); ax[0].set_ylabel("LV volume (normalised)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)

    # --- B: the teleport, shipped vs fixed ----------------------------------------------
    ds, subj = TELE
    vol = lv(ds, subj); es = int(np.argmin(vol))
    z, rr, t0 = worst_plane(ds, subj, vol, es, "jump")
    times = t0 + np.arange(NF) * DT
    p_bad, b = simulate(vol, es, rr, times, fixed=False)
    p_fix, _ = simulate(vol, es, rr, times, fixed=True)
    ax[1].plot(range(NF), p_bad, "o-", color="#d9534f", lw=2, label="shipped")
    ax[1].plot(range(NF), p_fix, "s--", color="#2f6f4e", lw=2, ms=4, label="fixed")
    for k in range(NF - 1):
        if b[k] != b[k + 1]:
            ax[1].axvline(k + .5, color="#999", lw=1, ls=":")
            if p_bad[k + 1] - p_bad[k] > 1.5:
                ax[1].annotate("", xy=(k + 1, p_bad[k + 1]), xytext=(k, p_bad[k]),
                               arrowprops=dict(arrowstyle="->", color="#d9534f", lw=2))
                ax[1].text(k + 1.15, (p_bad[k] + p_bad[k + 1]) / 2,
                           f"+{p_bad[k+1]-p_bad[k]:.1f}\nphases", color="#d9534f", fontsize=9)
    ax[1].axhline(es, color="#888", lw=1)
    ax[1].text(0.2, es + .15, f"es={es} (fully contracted)", fontsize=8, color="#555")
    ax[1].set_title(f"B  TELEPORT  {subj[:26]} z{z}\nbeat cut mid-squeeze -> clamped forward to es")
    ax[1].set_xlabel("acquisition frame"); ax[1].set_ylabel("cardiac position (phase)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)

    # --- C: the freeze --------------------------------------------------------------------
    ds, subj = FREEZE
    vol = lv(ds, subj); es = int(np.argmin(vol))
    z, rr, t0 = worst_plane(ds, subj, vol, es, "static")
    times = t0 + np.arange(NF) * DT
    p_bad, b = simulate(vol, es, rr, times, fixed=False)
    ax[2].plot(range(NF), p_bad, "o-", color="#d9534f", lw=2)
    d = np.diff(p_bad); same = np.diff(b) == 0
    run = s = 0
    for k in range(len(d)):
        if same[k] and abs(d[k]) < 0.25:
            if run == 0:
                s = k
            run += 1
        else:
            if run >= 6:
                ax[2].axvspan(s, k, color="#d9534f", alpha=.15)
                ax[2].text((s + k) / 2, p_bad[s] + .6, f"{run} frames\nnearly static",
                           ha="center", color="#d9534f", fontsize=9)
            run = 0
    if run >= 6:
        ax[2].axvspan(s, NF - 1, color="#d9534f", alpha=.15)
        ax[2].text((s + NF - 1) / 2, p_bad[s] + .6, f"{run} frames\nnearly static",
                   ha="center", color="#d9534f", fontsize=9)
    ax[2].axhline(es, color="#888", lw=1)
    ax[2].set_title(f"C  FREEZE  {subj[:26]} z{z}\nonly T-es={T-es} phases left to fill a whole beat")
    ax[2].set_xlabel("acquisition frame"); ax[2].set_ylabel("cardiac position (phase)")
    ax[2].grid(alpha=.25)

    fig.tight_layout()
    p = os.path.join(OUT, "af24_pos_physio_bugs.png")
    fig.savefig(p, dpi=125)
    print("wrote", p)


if __name__ == "__main__":
    main()
