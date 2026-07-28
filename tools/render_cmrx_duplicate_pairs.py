"""Visual side-by-side of confirmed CMRxRecon duplicate pairs (mid slice, t=0).

One row per pair: KEPT | REDUNDANT copy | |difference|. If the pair really is one person
released twice, the difference panel is black and max|diff| is 0.

Both members are rendered with the SAME intensity window (from the kept volume) -- per-panel
autoscaling would let a brightness difference hide a real mismatch, or manufacture one. Panels
use TRUE physical aspect from the header in-plane spacing.

Both sides must come from the SAME recon version, or the ESPIRiT fix shows up as a difference:
  2024 -- archived Train_P19x carry v1 recons, re-reconstructed into `_dupcheck_v2/`
  2023 -- the redundant test-side copies were NEVER reconstructed, made into `_dupcheck_v2/`

    micromamba run -n svr python tools/render_cmrx_duplicate_pairs.py --year 2024
    micromamba run -n svr python tools/render_cmrx_duplicate_pairs.py --year 2023
    micromamba run -n svr python tools/render_cmrx_duplicate_pairs.py --year 2025
"""
import argparse
import csv
import json
import os

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "scratch", "data")
T = 0

# 2024: kept lives in Cine_combined (CMRx24_ prefix), redundant was archived then re-reconstructed.
PAIRS_2024 = [("Test_P009", "Train_P196"), ("Test_P010", "Train_P194"), ("Test_P011", "Train_P192"),
              ("Test_P012", "Train_P199"), ("Test_P013", "Train_P198"), ("Val_P052", "Train_P193"),
              ("Val_P024", "Train_P200")]
SEC23 = {"train": "Train", "val": "Val", "test": "Test"}


def pairs_2024():
    live = os.path.join(DATA, "CMRxRecon2024", "Cine_combined")
    dup = os.path.join(DATA, "CMRxRecon2024", "_dupcheck_v2")
    return [(f"CMRx24_{k}", f"CMRx24_{k}", os.path.join(live, f"CMRx24_{k}", "sax", "4d_recon.nii.gz"),
             r, os.path.join(dup, r, "sax", "4d_recon.nii.gz")) for k, r in PAIRS_2024]


def pairs_2023():
    d23 = os.path.join(DATA, "CMRxRecon2023")
    dup = os.path.join(d23, "_dupcheck_v2")
    rows = list(csv.DictReader(open(os.path.join(d23, "SUBJECT_MANIFEST.csv"))))
    out = []
    for r in rows:
        red = r["combined_id"]
        pr = os.path.join(dup, red, "sax", "4d_recon.nii.gz")
        if not (r["duplicate_of"].strip() and os.path.exists(pr)):
            continue
        s, p = r["duplicate_of"].split("/")
        kept = f"CMRx23_{SEC23[s]}_{p}"
        pk = os.path.join(d23, "Cine_combined", kept, "sax", "4d_recon.nii.gz")
        if os.path.exists(pk):
            out.append((kept, kept, pk, red, pr))
    return out


def pairs_2025():
    """Confirmed groups come straight from the scan's own JSON -- no hand-typed pair list."""
    d25 = os.path.join(DATA, "CMRxRecon2025")
    rep = os.path.join(d25, "duplicates_scan_v2.json")
    if not os.path.exists(rep):
        raise SystemExit(f"{rep} not found -- run tools/scan_cmrx2025_duplicates.py first")
    blob = json.load(open(rep))
    out = []
    for members in blob["confirmed"].values():
        for a, b in zip(members[:-1], members[1:]):
            out.append((_label(a), _label(a), _recon_of(d25, a), _label(b), _recon_of(d25, b)))
    return out


def _label(key):
    p = key.split("/")
    return f"{p[0]}/{p[1]}\n{p[-3]} {p[-2]}\n{p[-1]}"


# TaskR1/TaskR2 each ship a TestSet AND a ValidationSet, and the same Center/Scanner/PID can
# appear in both as DIFFERENT people -- so the Set must be part of the lookup, not just the task.
SET_TOKEN = {("TaskR1", "TestSet"): "R1test", ("TaskR1", "ValidationSet"): "R1val",
             ("TaskR2", "TestSet"): "R2test", ("TaskR2", "ValidationSet"): "R2val",
             ("TrainingData", "TrainingSet"): "Train"}


def _recon_of(d25, key):
    """Map a scan key (split/Set/FullSample/Center/Scanner/PID) to its reconstructed volume."""
    p = key.split("/")
    token = SET_TOKEN.get((p[0], p[1]))
    if token is None:
        raise SystemExit(f"no cid token for split/set {p[0]}/{p[1]} -- extend SET_TOKEN")
    cid = f"CMRx25_{token}_{p[-3]}_{p[-2]}_{p[-1]}"
    return os.path.join(d25, "Cine_combined", cid, "sax", "4d_recon.nii.gz")


def load(p):
    img = nib.load(p)
    return np.asanyarray(img.dataobj), img.header.get_zooms()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, choices=["2023", "2024", "2025"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pairs = {"2023": pairs_2023, "2024": pairs_2024, "2025": pairs_2025}[args.year]()
    if args.limit:
        pairs = pairs[: args.limit]
    rows = []
    for klabel, _, pk, rlabel, pr in pairs:
        if not (os.path.exists(pk) and os.path.exists(pr)):
            print(f"  SKIP {klabel} <-> {rlabel}: kept={os.path.exists(pk)} redundant={os.path.exists(pr)}")
            continue
        a, za = load(pk)
        b, _ = load(pr)
        rows.append((klabel, rlabel, a, b, za))
    if not rows:
        raise SystemExit("no renderable pairs")

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, len(rows) * 2.5), dpi=140)
    axes = np.atleast_2d(axes)
    for r, (klabel, rlabel, a, b, za) in enumerate(rows):
        z = a.shape[2] // 2
        vmax = float(np.percentile(a[..., T], 99.5)) or 1.0
        kw = dict(cmap="gray", vmin=0, vmax=vmax, aspect=za[1] / za[0], interpolation="nearest")
        axes[r, 0].imshow(a[:, :, z, T].T, **kw)
        if a.shape == b.shape:
            axes[r, 1].imshow(b[:, :, z, T].T, **kw)
            d = np.abs(a[..., T].astype(np.float64) - b[..., T].astype(np.float64))
            mx = float(d.max())
            corr = float(np.corrcoef(a[..., T].ravel(), b[..., T].ravel())[0, 1])
            axes[r, 2].imshow(d[:, :, z].T, **kw)
            note = f"max|diff|={mx:.3g}   corr={corr:.6f}" + ("   [BIT-IDENTICAL]" if mx == 0 else "")
            col = "green" if mx == 0 else "darkorange"
        else:
            axes[r, 1].imshow(b[:, :, b.shape[2] // 2, T].T, cmap="gray", interpolation="nearest")
            axes[r, 2].set_visible(False)
            note, col = f"SHAPE MISMATCH {a.shape} vs {b.shape}", "red"
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_title(f"KEPT  {klabel}", fontsize=7)
        axes[r, 1].set_title(f"REDUNDANT  {rlabel}", fontsize=7)
        axes[r, 2].set_title("|difference|", fontsize=7)
        axes[r, 0].text(0.0, -0.09, note, transform=axes[r, 0].transAxes, fontsize=7, color=col)

    fig.suptitle(f"CMRxRecon{args.year} — duplicate pairs, both sides reconstructed with the fixed "
                 f"ESPIRiT\n(mid slice, t={T}, identical intensity window, true physical aspect)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = args.out or os.path.join(REPO, "result", f"cmrx{args.year}_duplicate_pairs_midslice.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
