"""Measure REAL respiratory motion in the OCMR real-time free-breathing SAX cines, to test
the docs/08 claim that OCMR is "clean because no breathing".

Method (per slice's real-time cine): rigid-register each frame to the temporal-median frame
(skimage phase cross-correlation, subpixel) -> in-plane displacement (dy,dx) in mm over time.
That trajectory mixes CARDIAC (fast, ~17 frames/cycle at ~47ms) and RESPIRATORY (slow, ~75
frames/cycle). Separate by a moving-average low-pass (window = one cardiac cycle): the smoothed
drift = respiratory; the residual = cardiac. Report peak-to-peak respiratory excursion (mm) and
compare to the val SYNTHETIC sim (16+/-8 mm SI). In-plane is a LOWER BOUND on true SI respiratory
amplitude (SAX is tilted ~20-45 deg off SI, so part of SI motion is through-plane).

Run: micromamba run -n svr python tools/measure_ocmr_breathing.py
"""
import glob
import json
import os

import numpy as np
import SimpleITK as sitk
from skimage.registration import phase_cross_correlation

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
RECON = os.path.join(ROOT, "scratch/data/ocmr/recon")
TR_MS_FALLBACK = 47.0
CARDIAC_MS = 800.0


def smooth(x, w):
    if w < 2 or w >= len(x):
        return np.full_like(x, x.mean())
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def measure_subject(subj_dir):
    cine = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subj_dir, "sax_cine.nii.gz"))).astype(np.float32)
    meta = json.load(open(os.path.join(subj_dir, "meta.json")))
    nF, nS, H, W = cine.shape
    ix, iy = meta["inplane_mm"]                       # W spacing, H spacing
    tr = meta.get("TRes_ms", TR_MS_FALLBACK) or TR_MS_FALLBACK
    card_w = max(3, int(round(CARDIAC_MS / tr)))      # frames per cardiac cycle
    # Heart-centred crop (central third): respiratory translation of the heart is strongest there,
    # and the static chest wall (which dilutes a whole-image registration) is excluded.
    y0, y1, x0, x1 = H // 3, H - H // 3, W // 3, W - W // 3
    resp_ptp, card_amp, total_ptp = [], [], []
    for s in range(nS):
        clip = cine[:, s, y0:y1, x0:x1]               # (F,h,w) heart crop
        if clip.max() <= 0:
            continue
        ref = np.median(clip, axis=0)
        dy = np.zeros(nF); dx = np.zeros(nF)
        for f in range(nF):
            (sy, sx), _, _ = phase_cross_correlation(ref, clip[f], upsample_factor=10, normalization=None)
            dy[f] = sy * iy; dx[f] = sx * ix          # mm
        dy -= dy.mean(); dx -= dx.mean()
        # respiratory = low-pass (smooth out one cardiac cycle); cardiac = residual
        dys, dxs = smooth(dy, card_w), smooth(dx, card_w)
        resp_mag = np.sqrt(dys ** 2 + dxs ** 2)
        card_res = np.sqrt((dy - dys) ** 2 + (dx - dxs) ** 2)
        total = np.sqrt(dy ** 2 + dx ** 2)
        resp_ptp.append(resp_mag.max() - resp_mag.min())
        card_amp.append(card_res.std())
        total_ptp.append(total.max() - total.min())
    return dict(
        subject=meta["subject"], nF=nF, nS=nS, TRes=tr, card_w=card_w,
        inplane_mm=[round(ix, 2), round(iy, 2)],
        resp_ptp_mm=float(np.median(resp_ptp)), resp_ptp_max=float(np.max(resp_ptp)),
        card_amp_mm=float(np.median(card_amp)), total_ptp_mm=float(np.median(total_ptp)),
        cardiac_cycles=round(nF * tr / CARDIAC_MS, 1),
    )


def main():
    subs = sorted(glob.glob(os.path.join(RECON, "us_*")))
    rows = []
    print(f"{'subject':18s} {'nF':>4} {'cyc':>4} {'inplane':>10}  {'RESP_ptp':>9} {'resp_max':>8} {'cardiac':>8} {'total':>7}")
    print(f"{'':18s} {'':>4} {'':>4} {'':>10}  {'(mm,med)':>9} {'(mm)':>8} {'(mm,std)':>8} {'(mm)':>7}")
    for sd in subs:
        if not os.path.exists(os.path.join(sd, "sax_cine.nii.gz")):
            continue
        r = measure_subject(sd)
        rows.append(r)
        print(f"{r['subject']:18s} {r['nF']:4d} {r['cardiac_cycles']:4.1f} "
              f"{str(r['inplane_mm']):>10}  {r['resp_ptp_mm']:9.2f} {r['resp_ptp_max']:8.2f} "
              f"{r['card_amp_mm']:8.2f} {r['total_ptp_mm']:7.2f}", flush=True)
    # headline: long cines (>=3 cardiac cycles) give the cleanest respiratory estimate
    longr = [r for r in rows if r["cardiac_cycles"] >= 3]
    allr = rows
    def med(rs, k): return float(np.median([r[k] for r in rs])) if rs else float("nan")
    print("\n=== SUMMARY (in-plane; LOWER BOUND on true SI respiratory amplitude) ===")
    print(f"  long cines (>=3 cardiac cycles, n={len(longr)}): "
          f"resp_ptp median {med(longr,'resp_ptp_mm'):.2f} mm, max {max((r['resp_ptp_max'] for r in longr), default=float('nan')):.2f} mm")
    print(f"  all subjects (n={len(allr)}):                    "
          f"resp_ptp median {med(allr,'resp_ptp_mm'):.2f} mm")
    print(f"  val SYNTHETIC sim for comparison: 16 +/- 8 mm SI (8-24 mm), per-slice, deform-then-reslice")
    json.dump(rows, open(os.path.join(ROOT, "result/ocmr_cleaner/breathing_amplitude.json"), "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
