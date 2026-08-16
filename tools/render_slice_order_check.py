"""Visual + multi-feature check of base-vs-apex slice ordering, for all three sources.

Provenance for docs/58 §10a. `tools/probe_slice_order.py` decides the ordering from a single
feature (total labeled area vs z) and prints only a tally; this script (a) adds two more features
so agreement can be inspected rather than assumed, and (b) renders the LONG-AXIS SIDE VIEW of the
LV so a human can see the cone taper and confirm the call by eye.

Why a side view: every SAX plane looks like a donut, so the taper -- the only anatomical signal
that distinguishes base from apex -- is invisible slice-by-slice. Reformatting through the LV
centroid along z turns the stack into the familiar cone: wide chamber at the base, closing to a
point at the apex.

Features (all computed on the ED frame of `sax/heart_seg.nii.gz`, labels 1=LV 2=MYO 3=RV):

  f1  slope of TOTAL labeled area vs z          (what probe_slice_order.py uses)
  f2  slope of LV-CAVITY area vs z              (label 1 alone; the cavity vanishes at the apex
                                                 while myocardium persists, so this is the
                                                 cleaner taper signal)
  f3  slope of cavity fraction LV/(LV+MYO) vs z (shape, not size: near the apex the wall is thick
                                                 relative to a small cavity, near the base thin
                                                 relative to a large one -- so this survives a
                                                 stack that is cropped before the true apex)

  slope > 0  ->  the quantity GROWS with z  ->  z increases toward the BASE  ->  z0 = apex.

DECISION RULE (selected by measurement, not by argument -- see docs/58 §10a):

    call = sign(f1) if sign(f1) == sign(f2), else UNDETERMINED.  f3 is reported, never voted.

M&Ms is the only source with a metadata-derived ground truth (its converter guarantees apex-first
for all 345), so aggregation rules were scored against it. Results on the subjects where features
disagree -- i.e. exactly where the rule matters:

    f3 alone            3/22  (14%)   <- ANTI-informative on hard cases
    max-|z| tie-break  13/22  (59%)
    f1 alone           19/22  (86%)
    f1+f2 / majority   21/22  (95%)

and on the whole cohort, f1+f2-with-agreement leaves only 10 undetermined (vs 94 for a 3-feature
majority) at the same M&Ms accuracy. Hence: drop f3 from the vote, keep it in the report. A
magnitude-weighted vote and a robust-scaled mean were both tried and are WORSE (6 M&Ms errors);
do not "improve" this by reintroducing them without re-running the M&Ms check.

NOTE these are views of one underlying taper, not independent measurements. The segmentation
carries no basal-only structure (no atria, no great vessels in Task114's label set), so a
genuinely independent anatomical anchor is not available from this input.

Usage:
    python tools/render_slice_order_check.py                  # 4 subjects/source, seeded
    python tools/render_slice_order_check.py --n 6
    python tools/render_slice_order_check.py --subjects ACDC_patient001,MNMs_A0S9V9
    python tools/render_slice_order_check.py --csv result/slice_order_check/features.csv
                                                              # score ALL subjects, no rendering

Outputs -> result/slice_order_check/
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")
OUT = os.path.join(ROOT, "result/slice_order_check")

SOURCES = [
    ("CMRx", os.path.join(DATA, "CMRxRecon202*/Cine_combined/*/sax")),
    ("ACDC", os.path.join(DATA, "ACDC_sax/*/sax")),
    ("M&Ms", os.path.join(DATA, "MNMs_sax/*/sax")),
    ("MIITT", os.path.join(DATA, "MIITT_sax/*/sax")),
]

LV, MYO, RV = 1, 2, 3
HALF_WINDOW_MM = 70.0   # in-plane crop half-width around the LV centroid, for legibility


# ── feature extraction ───────────────────────────────────────────────────────────────────────

def _slope(y):
    """Least-squares slope of y vs index, normalized by |y|.max() so it is scale-free."""
    y = np.asarray(y, dtype=float)
    if y.size < 4 or not np.isfinite(y).all():
        return float("nan")
    peak = np.abs(y).max()
    if peak <= 0:
        return float("nan")
    return float(np.polyfit(np.arange(y.size), y / peak, 1)[0])


def features(seg_path, min_planes=4):
    """-> dict of per-subject ordering features, or None if unusable."""
    seg = np.asarray(nib.load(seg_path).dataobj)
    if seg.ndim == 4:
        seg = seg[..., 0]                      # ED frame (frame 0 is ED by construction)
    if seg.ndim != 3:
        return None

    total = np.array([(seg[..., z] > 0).sum() for z in range(seg.shape[2])], dtype=float)
    idx = np.flatnonzero(total)
    if idx.size < min_planes:
        return None
    sl = slice(idx[0], idx[-1] + 1)            # labeled span only

    cav = np.array([(seg[..., z] == LV).sum() for z in range(seg.shape[2])], dtype=float)
    myo = np.array([(seg[..., z] == MYO).sum() for z in range(seg.shape[2])], dtype=float)
    denom = cav + myo
    frac = np.divide(cav, denom, out=np.full_like(cav, np.nan), where=denom > 0)

    f1, f2 = _slope(total[sl]), _slope(cav[sl])
    fr = frac[sl]
    f3 = _slope(fr) if np.isfinite(fr).all() else float("nan")

    # DECISION RULE (see the module docstring): f1 and f2 only, and they must agree.
    # f3 is computed and reported but deliberately NOT voted -- measured 3/22 correct on the
    # hard (disagreeing) M&Ms subjects, i.e. anti-informative exactly where a tie-break is needed.
    call, agree = None, False
    if f1 == f1 and f2 == f2:
        agree = (f1 > 0) == (f2 > 0)
        if agree:
            call = "apex-first" if f1 > 0 else "base-first"

    return dict(f1_total=f1, f2_cavity=f2, f3_cavfrac=f3, order=call, agree=agree,
                n_labeled=int(idx.size), total=total, cav=cav, myo=myo, frac=frac, seg=seg,
                z0=int(idx[0]), z1=int(idx[-1]))


# ── rendering ────────────────────────────────────────────────────────────────────────────────

def _reformat(vol, seg, axis, pos):
    """Long-axis reformat: (in-plane, z) image + matching seg slab."""
    if axis == 0:
        return vol[pos, :, :], seg[pos, :, :]
    return vol[:, pos, :], seg[:, pos, :]


def render_subject(sax_dir, feat, ax_row, proposal=None, all_planes=False):
    """Draw two orthogonal long-axis reformats + the taper curves onto three axes."""
    subj = os.path.basename(os.path.dirname(sax_dir))
    img_path = os.path.join(sax_dir, "3d_recon", "sax_frame_00.nii.gz")
    im = nib.load(img_path)
    vol = np.asarray(im.dataobj).astype(np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    dx, dy, dz = [float(v) for v in im.header.get_zooms()[:3]]
    seg = feat["seg"]

    # LV centroid over the labeled span -> where to cut the long-axis views
    mask = seg == LV
    if mask.sum() == 0:
        mask = seg > 0
    cx, cy = [int(round(c)) for c in np.array(np.nonzero(mask))[:2].mean(axis=1)]
    cx = int(np.clip(cx, 0, vol.shape[0] - 1))
    cy = int(np.clip(cy, 0, vol.shape[1] - 1))

    lo, hi = np.percentile(vol[vol > 0], [1, 99.5]) if (vol > 0).any() else (0, 1)
    D = vol.shape[2]

    for ax, (axis, pos, row_c, ip_mm, name) in zip(
        ax_row[:2],
        [(1, cy, cx, dx, "cut along X"), (0, cx, cy, dy, "cut along Y")],
    ):
        sl_img, sl_seg = _reformat(vol, seg, axis, pos)
        # sl_img is (in-plane, z); imshow rows -> y, cols -> x, so z lands horizontal as wanted.
        # Crop the in-plane axis to a window around the LV, otherwise the ~360mm FOV against a
        # ~100mm stack renders as an unreadable sliver under aspect="equal".
        half = int(round(HALF_WINDOW_MM / ip_mm))
        r0 = max(0, row_c - half)
        r1 = min(sl_img.shape[0], row_c + half)
        sl_img, sl_seg = sl_img[r0:r1], sl_seg[r0:r1]
        extent = [0, D * dz, 0, sl_img.shape[0] * ip_mm]     # physical mm, z horizontal
        ax.imshow(np.clip((sl_img - lo) / max(hi - lo, 1e-6), 0, 1),
                  cmap="gray", origin="lower", extent=extent, aspect="equal",
                  vmin=0, vmax=1)
        for lab, color in ((LV, "#ff4d4d"), (MYO, "#4dff88"), (RV, "#4da6ff")):
            m = (sl_seg == lab).astype(float)
            if m.any():
                ax.contour(m, levels=[0.5], colors=[color], linewidths=0.9,
                           extent=extent, origin="lower")
        ax.set_xlabel(f"z (mm)   [{name}]", fontsize=8)
        ax.tick_params(labelsize=7)

        # Annotate the ends. A PROPOSAL (human/agent judgement on an undetermined subject) is
        # drawn in orange and clearly marked, so it is never confused with the detector's call.
        # "undetermined" is not a call -- must not fall through to the base-first branch below.
        shown = proposal if proposal in ("apex-first", "base-first") else feat["order"]
        color = "#ff9f1a" if (proposal and feat["order"] is None) else "yellow"
        if shown is None:
            left = right = "?"
        else:
            left, right = ("APEX", "BASE") if shown == "apex-first" else ("BASE", "APEX")
        ax.annotate(left, xy=(0.02, 0.94), xycoords="axes fraction", color=color,
                    fontsize=9, weight="bold", ha="left")
        ax.annotate(right, xy=(0.98, 0.94), xycoords="axes fraction", color=color,
                    fontsize=9, weight="bold", ha="right")
        ax.annotate("z=0", xy=(0.02, 0.03), xycoords="axes fraction", color="w", fontsize=7)

    ax = ax_row[2]
    zs = np.arange(len(feat["total"]))
    ax.plot(zs, feat["total"] / max(feat["total"].max(), 1), "o-", ms=3, label="all labels")
    ax.plot(zs, feat["cav"] / max(feat["cav"].max(), 1), "s-", ms=3, label="LV cavity")
    ax.plot(zs, feat["frac"], "^-", ms=3, label="cav/(cav+myo)")
    ax.set_xlabel("z index", fontsize=8)
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(alpha=0.25)

    # ── end-plane panel: the two extreme LABELED short-axis planes, side by side ──────────────
    # The most directly readable evidence there is: the apical end shows a small cavity inside a
    # relatively thick wall (or myocardium with the cavity closed off); the basal end shows a
    # large thin-walled cavity. Judgeable by eye without trusting any slope fit.
    if len(ax_row) > 3:
        ax = ax_row[3]
        hw = int(round(55.0 / dx))
        panes = []
        # all_planes: for tiny / mostly-unsegmented stacks the two extreme LABELED planes carry no
        # information, so tile EVERY z plane and let the images decide.
        zlist = range(vol.shape[2]) if all_planes else (feat["z0"], feat["z1"])
        for z in zlist:
            sub_i = vol[max(0, cx - hw):cx + hw, max(0, cy - hw):cy + hw, z]
            sub_s = seg[max(0, cx - hw):cx + hw, max(0, cy - hw):cy + hw, z]
            pane = np.clip((sub_i - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb = np.repeat(pane[..., None], 3, axis=2)
            rgb[sub_s == LV] = [1.0, 0.30, 0.30]          # cavity
            rgb[sub_s == MYO] = rgb[sub_s == MYO] * 0.45 + np.array([0.0, 0.55, 0.25]) * 0.55
            panes.append(rgb)
        h = min(p.shape[0] for p in panes)
        w = min(p.shape[1] for p in panes)
        gap = np.ones((h, 3, 3))
        tiles = []
        for i, pane in enumerate(panes):
            if i:
                tiles.append(gap)
            tiles.append(pane[:h, :w])
        ax.imshow(np.concatenate(tiles, axis=1), origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        if all_planes:
            ax.annotate("z=0", xy=(0.01, 0.02), xycoords="axes fraction", color="w", fontsize=7.5)
            ax.annotate(f"z={vol.shape[2]-1}", xy=(0.99, 0.02), xycoords="axes fraction",
                        color="w", fontsize=7.5, ha="right")
            ax.set_title("ALL z planes, left->right — apical end = small cavity / thick wall",
                         fontsize=7)
        else:
            ax.annotate(f"z={feat['z0']} (first)", xy=(0.02, 0.02), xycoords="axes fraction",
                        color="w", fontsize=7.5)
            ax.annotate(f"z={feat['z1']} (last)", xy=(0.98, 0.02), xycoords="axes fraction",
                        color="w", fontsize=7.5, ha="right")
            ax.set_title("end planes — apical end = small cavity / thick wall", fontsize=7)

    flag = "" if feat["agree"] else "  ** UNDETERMINED (f1/f2 disagree) **"
    prop = (f"\nPROPOSED: {proposal}"
            if (proposal in ("apex-first", "base-first") and feat["order"] is None) else "")
    ax_row[0].set_ylabel(f"{subj}\n{feat['order'] or 'UNDETERMINED'}{flag}{prop}", fontsize=8)
    ax_row[1].set_title(
        f"f1(total) {feat['f1_total']:+.3f}   f2(cavity) {feat['f2_cavity']:+.3f}   "
        f"f3(cav frac) {feat['f3_cavfrac']:+.3f}    dz={dz:.2f}mm  D={D}",
        fontsize=7)


def find_subjects():
    out = {}
    for label, pattern in SOURCES:
        out[label] = sorted(glob.glob(pattern))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="subjects rendered per source")
    ap.add_argument("--subjects", default="", help="comma-separated subject ids (overrides --n)")
    ap.add_argument("--csv", default="", help="score ALL subjects into this CSV; skip rendering")
    ap.add_argument("--proposals", default="",
                    help="CSV with columns subject,proposed — draws the proposal in ORANGE on "
                         "undetermined subjects and adds the end-plane panel for adjudication")
    ap.add_argument("--min-planes", type=int, default=4,
                    help="minimum labeled planes for a slope fit; lower it to inspect stacks the "
                         "detector refuses to call")
    ap.add_argument("--all-planes", action="store_true",
                    help="tile EVERY z plane in the end-plane panel instead of just the extremes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    found = find_subjects()

    if args.csv:
        rows = []
        for label, dirs in found.items():
            for d in dirs:
                seg = os.path.join(d, "heart_seg.nii.gz")
                if not os.path.exists(seg):
                    continue
                f = features(seg)
                subj = os.path.basename(os.path.dirname(d))
                if f is None:
                    rows.append(dict(source=label, subject=subj, order="undetermined",
                                     f1_total="", f2_cavity="", f3_cavfrac="", agree="",
                                     n_labeled=""))
                    continue
                rows.append(dict(source=label, subject=subj,
                                 order=f["order"] or "undetermined",
                                 f1_total=round(f["f1_total"], 4),
                                 f2_cavity=round(f["f2_cavity"], 4),
                                 f3_cavfrac=round(f["f3_cavfrac"], 4) if f["f3_cavfrac"] == f["f3_cavfrac"] else "",
                                 agree=f["agree"], n_labeled=f["n_labeled"]))
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}  ({len(rows)} subjects)")
        for label in found:
            sub = [r for r in rows if r["source"] == label and r["order"] in ("apex-first", "base-first")]
            if not sub:
                continue
            a = sum(r["order"] == "apex-first" for r in sub)
            und = sum(r["source"] == label and r["order"] == "undetermined" for r in rows)
            print(f"  {label:6} determined {len(sub):4d}  apex-first {a:4d}  "
                  f"base-first {len(sub)-a:4d}   undetermined {und:3d}")
        return 0

    want = [s.strip() for s in args.subjects.split(",") if s.strip()]
    rng = random.Random(args.seed)
    picks = []
    for label, dirs in found.items():
        if want:
            sel = [d for d in dirs if os.path.basename(os.path.dirname(d)) in want]
        else:
            sel = rng.sample(dirs, min(args.n, len(dirs))) if dirs else []
        picks += [(label, d) for d in sel]

    if not picks:
        print("no subjects matched")
        return 1

    proposals = {}
    if args.proposals:
        with open(args.proposals) as fh:
            for r in csv.DictReader(fh):
                proposals[r["subject"]] = r["proposed"].strip()

    ncol = 4 if proposals else 3
    # the all-planes strip needs far more horizontal room than a single reformat
    wr = [1, 1, 1] + ([2.6 if args.all_planes else 1.0] if ncol == 4 else [])
    fig, axes = plt.subplots(len(picks), ncol,
                             figsize=(4.0 * sum(wr), 3.6 * len(picks)),
                             gridspec_kw={"width_ratios": wr}, squeeze=False)
    for row, (label, d) in enumerate(picks):
        seg = os.path.join(d, "heart_seg.nii.gz")
        f = features(seg, args.min_planes)
        if f is None:
            axes[row][0].set_ylabel("undetermined", fontsize=8)
            continue
        subj = os.path.basename(os.path.dirname(d))
        render_subject(d, f, axes[row], proposal=proposals.get(subj),
                       all_planes=args.all_planes)
        axes[row][0].text(-0.45, 0.5, label, transform=axes[row][0].transAxes,
                          rotation=90, va="center", fontsize=10, weight="bold")

    fig.suptitle(
        "Long-axis side view (LV cone) + taper features — is z=0 the apex or the base?\n"
        "YELLOW = detector's call   |   ORANGE = PROPOSED call on an UNDETERMINED subject "
        "(needs confirmation)\n"
        "The cone should close at the APEX end; the apical end plane = small cavity in a thick wall",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, "slice_order_sideviews.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
