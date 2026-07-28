"""Turn human LV measurements into a slice-pitch estimate, calibrated on control subjects.

Two estimators, because the right one is not obvious a priori:
    pitch_gap  = L / (N - 1)   treats L as the distance between the OUTERMOST slice CENTRES
    pitch_slab = L / N         treats each slice as covering a slab ~pitch thick, so N slices
                               covering the LV span N*pitch  (LV extends ~half a slice beyond
                               the outermost centres at each end)
The controls have a documented thickness (8 mm) and, with the accepted +4 mm gap, a known
pitch of 12.0 mm -- so they decide which estimator to trust and quantify any observer bias.

    micromamba run -n svr python tools/analyze_pitch_measurements.py
"""
import numpy as np

CONTROL_TRUE_PITCH = 12.0

# (label, centre, n_slices_in_stack, L_min, L_max, N_min, N_max, is_control)
M = [
    ("CIMAX_P008",  "Center002/CIMAX",  9,  100, 120,  9,  9, True),
    ("CIMAX_P020",  "Center002/CIMAX", 11,  110, 130,  9, 10, True),
    ("Vida5_P020",  "Center005/Vida",  10,  130, 130, 10, 10, True),   # user said 11; stack has 10
    ("Vida5_P012",  "Center005/Vida",  12,  100, 100,  9,  9, True),
    # --- UNKNOWNS (blank SliceThickness) ---
    ("Aera_P009",   "Center004/Aera",  12,  120, 120, 11, 11, False),
    ("Aera_P012",   "Center004/Aera",  13,  150, 150, 11, 11, False),
    ("Prisma_P024", "Center006/Prisma",18,  120, 120, 12, 14, False),  # "~13"
    ("Prisma_P020", "Center006/Prisma",13,  125, 125, 10, 10, False),  # revised: 125 mm / 10 slices
    ("Prisma_P005", "Center006/Prisma",13,  110, 110, 12, 12, False),
    ("Prisma_P017", "Center006/Prisma",17,  130, 130, 11, 11, False),
    # Prisma_P010 (Z=9): observer could not identify the LV boundary -- deliberately EXCLUDED
    # rather than guessed.
    ("Vida1_P006",  "Center001/Vida",  13,  140, 140, 11, 11, False),
    ("Vida1_P012",  "Center001/Vida",  11,  110, 110,  9,  9, False),
    # --- batch 2, unambiguous only ---
    ("Prisma_P006b", "Center006/Prisma", 14, 120, 120, 12, 12, False),
    ("Prisma_P012b", "Center006/Prisma", 13, 130, 130, 13, 13, False),
    ("Aera_P002",    "Center004/Aera",    9, 130, 130,  8,  8, False),
    ("Aera_P008",    "Center004/Aera",    9, 130, 130,  9,  9, False),
]
# Protocol pitches = thickness + the accepted 4 mm gap. Thickness 2..12 mm are all realistic,
# and 1.5T scanners routinely use THICKER slices for SNR -- so the list must extend above 12,
# or a centre that genuinely sits at 14 gets silently snapped down to 12.
CANDIDATES = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]


def est(L, N, mode):
    d = (N - 1) if mode == "gap" else N
    return L / d if d > 0 else np.nan


def main():
    print(f"{'subject':13s} {'centre':18s} {'Z':>2s} {'L(mm)':>9s} {'N':>5s} "
          f"{'L/(N-1)':>14s} {'L/N':>14s}")
    rows = []
    for lab, ctr, Z, lo, hi, nlo, nhi, ctl in M:
        gaps = [est(l, n, "gap") for l in (lo, hi) for n in (nlo, nhi)]
        slabs = [est(l, n, "slab") for l in (lo, hi) for n in (nlo, nhi)]
        flag = "  <-- N EXCEEDS STACK" if nhi > Z else ""
        print(f"{lab:13s} {ctr:18s} {Z:2d} {lo:4d}-{hi:<4d} {nlo:2d}-{nhi:<2d} "
              f"{min(gaps):6.1f}-{max(gaps):<6.1f} {min(slabs):6.1f}-{max(slabs):<6.1f}{flag}")
        rows.append((lab, ctl, np.mean(gaps), np.mean(slabs)))

    for mode, idx in (("L/(N-1)", 2), ("L/N", 3)):
        v = np.array([r[idx] for r in rows if r[1]])
        bias = v.mean() / CONTROL_TRUE_PITCH
        print(f"\n{mode:9s} on controls: mean {v.mean():5.2f} mm  (range {v.min():.2f}-{v.max():.2f})"
              f"   true = {CONTROL_TRUE_PITCH}   bias x{bias:.3f}   "
              f"err {100*(bias-1):+.1f}%")

    # Calibrate on the controls, then apply to the unknown centres.
    ctl = np.array([r[3] for r in rows if r[1]])
    bias = ctl.mean() / CONTROL_TRUE_PITCH
    ctl_spread = ctl.std(ddof=1) / ctl.mean()
    print(f"\n=== per-centre estimate (L/N, bias-corrected /{bias:.3f}) ===")
    print(f"    per-subject scatter measured on controls: +/-{100*ctl_spread:.0f}%")
    by = {}
    for (lab, ctr, Z, lo, hi, nlo, nhi, is_ctl), r in zip(M, rows):
        if not is_ctl:
            by.setdefault(ctr, []).append((lab, r[3] / bias))
    for ctr, vals in by.items():
        v = np.array([x for _, x in vals])
        m = v.mean()
        # Do NOT use the control scatter here -- these subjects scatter much more. Take the
        # larger of (control-derived SE) and (observed SE), so n=2 cannot fake precision.
        se_ctl = ctl_spread * m / np.sqrt(len(v))
        se_obs = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.inf
        se = max(se_ctl, se_obs)
        detail = ", ".join(f"{l}={x:.1f}" for l, x in vals)
        best = min(CANDIDATES, key=lambda c: abs(c - m))
        within = [c for c in CANDIDATES if abs(c - m) <= 2 * se]
        verdict = "CONCLUSIVE" if len(within) == 1 else "inconclusive"
        print(f"  {ctr:18s} n={len(v)}  -> {m:5.2f} +/- {se:.2f} mm"
              f"   nearest {best:4.1f}   within 2SE: {within}  {verdict}")
        print(f"    {detail}")

    # Pooled estimator: L = pitch * N through the origin. Uses every subject at once and is
    # driven by subjects with large N, rather than averaging equally-weighted noisy ratios.
    print("\n=== pooled through-origin slope (L = pitch * N) ===")
    for ctr in by:
        pts = [(np.mean([lo, hi]), np.mean([nlo, nhi]))
               for (lab, c, Z, lo, hi, nlo, nhi, ctl) in M if c == ctr and not ctl]
        L = np.array([p[0] for p in pts]); N = np.array([p[1] for p in pts])
        slope = (L * N).sum() / (N ** 2).sum()
        print(f"  {ctr:18s} n={len(L)}  slope = {slope:5.2f} mm  "
              f"(bias-corrected {slope/bias:5.2f})")


if __name__ == "__main__":
    main()
