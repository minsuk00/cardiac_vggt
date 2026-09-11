"""Misalignment metric v2 (read-only). Per slice z of a SAX stack:
  seg-spike : c_z - (c_{z-1}+c_{z+1})/2 of the LV-cavity (label 1) and epicardial (1+2) centroid, mm,
              per phase, then phase-median vector (real breath-hold offset is phase-invariant).
  img-spike : NCC translation registration of slice z to z-1 and to z+1 within the heart ROI,
              spike = (d(z,z-1)+d(z,z+1))/2, per phase, phase-median. Independent of segmentation.
  line-resid: robust line residual of LV centroid (the docs/94-style metric, LV-only, deg 1).
("spike" was called "kink" in the first two versions of this tool; same formula.)
Per-slice registration bookkeeping: img_pin_{prev,next} = fraction of phases whose NCC peak sat at the
search bound (offset beyond MAXSHIFT_MM: NOT "no shift", so analyze.py falls back to the segmentation
estimate there), img_flat_{prev,next} = fraction with a featureless ROI (peak NCC < 0.05; genuinely
no answer). Both kinds are NaN in the displacement columns.
Outputs per-slice rows to a CSV. Usage: python metric_v2.py <out.csv> [--limit N] [--sections train,val,test]
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
import torch, torch.nn.functional as F
import pandas as pd

dev = "cuda"
MAXSHIFT_MM = 40.0   # was 20.0 -> a 28 mm real block shift (ACDC_patient081 z7|z8) was pinned at the bound,
                     # NaN'd, and the "both must agree" rule then discarded the segmentation's correct answer.

from ncc_fast import ncc_shift_fast as ncc_shift
def ncc_shift_slow(mov, ref, mask, maxpx):
    """mov, ref: (T,H,W) float tensors; mask (H,W) bool on mov. Returns (T,2) best (dx,dy) in px
    such that mov shifted by (dx,dy) matches ref best, subpixel-refined, and peak NCC (T,)."""
    T, H, W = mov.shape
    m = mask.float()
    # candidate shifts
    s = torch.arange(-maxpx, maxpx + 1, device=dev)
    # pad ref and extract shifted windows via unfold-like indexing
    refp = F.pad(ref, (maxpx, maxpx, maxpx, maxpx))
    n = m.sum()
    mov_m = mov * m
    mu_m = mov_m.sum((1, 2)) / n
    movc = (mov - mu_m[:, None, None]) * m
    var_m = (movc ** 2).sum((1, 2))
    best = torch.full((T,), -2.0, device=dev); bi = torch.zeros(T, 2, dtype=torch.long, device=dev)
    ncc = torch.zeros(T, len(s), len(s), device=dev)
    for i, dy in enumerate(s.tolist()):
        for j, dx in enumerate(s.tolist()):
            # ref sampled at (x+dx, y+dy) compared with mov at (x,y) => mov moved by (dx,dy) lands on ref
            win = refp[:, maxpx + dy: maxpx + dy + H, maxpx + dx: maxpx + dx + W]
            mu_r = (win * m).sum((1, 2)) / n
            rc = (win - mu_r[:, None, None]) * m
            ncc[:, i, j] = (movc * rc).sum((1, 2)) / torch.sqrt(var_m * (rc ** 2).sum((1, 2)) + 1e-8)
    flat = ncc.view(T, -1); pk, idx = flat.max(1)
    iy = idx // len(s); ix = idx % len(s)
    out = torch.zeros(T, 2, device=dev)
    for t in range(T):
        y, x = iy[t].item(), ix[t].item()
        sx, sy = float(s[x]), float(s[y])
        # parabolic subpixel
        if 0 < x < len(s) - 1:
            a, b, c = ncc[t, y, x - 1].item(), ncc[t, y, x].item(), ncc[t, y, x + 1].item()
            den = a - 2 * b + c
            if den < 0: sx += 0.5 * (a - c) / den
        if 0 < y < len(s) - 1:
            a, b, c = ncc[t, y - 1, x].item(), ncc[t, y, x].item(), ncc[t, y + 1, x].item()
            den = a - 2 * b + c
            if den < 0: sy += 0.5 * (a - c) / den
        out[t, 0] = sx; out[t, 1] = sy
    return out, pk

def load_subject(rel):
    seg, (dx, dy, dz) = load_seg(rel)
    img = load_4d(rel)  # (X,Y,Z,T)
    assert img.shape == seg.shape, (img.shape, seg.shape)
    roi = np.asarray(nib.load(os.path.join(DATA, rel, "sax", "heart_roi.nii.gz")).dataobj) > 0  # (X,Y,Z)
    return seg, img, roi, (dx, dy, dz)

def analyze(rel):
    return analyze_arrays(rel, *load_subject(rel))

def analyze_arrays(rel, seg, img, roi, spacing):
    """seg/img (X,Y,Z,T), roi (X,Y,Z) bool, spacing (dx,dy,dz) mm. Split out from analyze() so a
    fault-injection test can shift slices in memory (test_fault_injection.py) without touching disk."""
    dx, dy, dz = spacing
    X, Y, Z, T = seg.shape
    # ---- seg centroids (X,Y voxel) per z,t for LV (1) and epi (1,2)
    C = {k: np.full((Z, T, 2), np.nan) for k in ("lv", "epi")}
    A = {k: np.zeros((Z, T)) for k in ("lv", "epi")}
    for t in range(T):
        for k, labs in (("lv", (1,)), ("epi", (1, 2))):
            m = np.isin(seg[..., t], labs)
            for z in range(Z):
                mz = m[:, :, z]; n = mz.sum()
                if n < 15: continue
                ix, iy = np.nonzero(mz); C[k][z, t] = (ix.mean() * dx, iy.mean() * dy); A[k][z, t] = n
    # ---- image registration to neighbours
    im = torch.from_numpy(np.ascontiguousarray(img)).to(dev)  # (X,Y,Z,T)
    im = im.permute(2, 3, 1, 0).contiguous()  # (Z,T,H=Y,W=X)
    mask = torch.from_numpy(np.ascontiguousarray(roi)).to(dev).permute(2, 1, 0)  # (Z,H,W)
    maxpx = int(np.ceil(MAXSHIFT_MM / min(dx, dy)))
    D_prev = np.full((Z, T, 2), np.nan); D_next = np.full((Z, T, 2), np.nan)
    P_prev = np.full((Z, T), np.nan); P_next = np.full((Z, T), np.nan)
    PIN_prev = np.full(Z, np.nan); PIN_next = np.full(Z, np.nan); FLAT_prev = np.full(Z, np.nan); FLAT_next = np.full(Z, np.nan)
    for z in range(Z):
        if mask[z].sum() < 200: continue
        for nb, Dst, Pst, PIN, FLAT in ((z - 1, D_prev, P_prev, PIN_prev, FLAT_prev), (z + 1, D_next, P_next, PIN_next, FLAT_next)):
            if nb < 0 or nb >= Z: continue
            sh, pk = ncc_shift(im[z], im[nb], mask[z], maxpx)
            sh = sh.cpu().numpy(); pkn = pk.cpu().numpy()
            # NCC returns the shift that moves slice z ONTO its neighbour (= c_nb - c_z). Negate so that
            # d(z,nb) = c_z - c_nb, the same sign convention as the segmentation spike (verified by fault
            # injection: +5.4 mm injected -> seg +5.06 / img -5.08 before this fix).
            # Degenerate registrations -> NaN, but the two kinds are recorded separately: a flat ROI
            # (peak NCC ~0) is "no answer"; a peak pinned at the search bound is "offset >= MAXSHIFT_MM",
            # which analyze.py treats as agreeing with any large segmentation estimate.
            flat = pkn < 0.05; pinned = (~flat) & (np.abs(sh).max(1) >= maxpx - 0.5)
            sh = np.where((flat | pinned)[:, None], np.nan, -sh)
            Dst[z, :, 0] = sh[:, 0] * dx; Dst[z, :, 1] = sh[:, 1] * dy
            Pst[z] = pkn; PIN[z] = pinned.mean(); FLAT[z] = flat.mean()
    rows = []
    for z in range(Z):
        r = dict(rel=rel, src=rel.split("/")[0], Z=Z, z=z, dx=dx, dy=dy, dz=dz)
        for k in ("lv", "epi"):
            c = C[k]  # (Z,T,2)
            # spike: c_z - mean(c_{z-1}, c_{z+1}) per phase
            if 0 < z < Z - 1:
                kk = c[z] - 0.5 * (c[z - 1] + c[z + 1])
            else:
                kk = np.full((T, 2), np.nan)
            # one-sided (extrapolation) for ends
            if z == 0 and Z > 2: kk1 = c[z] - (2 * c[z + 1] - c[z + 2])
            elif z == Z - 1 and Z > 2: kk1 = c[z] - (2 * c[z - 1] - c[z - 2])
            else: kk1 = kk
            med = np.nanmedian(kk, axis=0) if np.isfinite(kk).any() else np.array([np.nan, np.nan])
            med1 = np.nanmedian(kk1, axis=0) if np.isfinite(kk1).any() else np.array([np.nan, np.nan])
            nph = int(np.isfinite(kk[:, 0]).sum())
            # cross-phase noise: RMS of per-phase deviation from median
            noise = float(np.sqrt(np.nanmean(np.sum((kk - med) ** 2, axis=1)))) if nph >= 3 else np.nan
            cmed = np.nanmedian(c[z], axis=0) if np.isfinite(c[z, :, 0]).any() else np.array([np.nan, np.nan])
            r.update({f"{k}_cx_med": cmed[0], f"{k}_cy_med": cmed[1],
                      f"{k}_spike_x": med[0], f"{k}_spike_y": med[1], f"{k}_spike": float(np.hypot(*med)),
                      f"{k}_spike_nph": nph, f"{k}_spike_noise": noise,
                      f"{k}_spike1": float(np.hypot(*med1)),
                      f"{k}_area_ed": A[k][z, 0], f"{k}_present_ed": bool(np.isfinite(c[z, 0, 0]))})
        # image spike
        kk = 0.5 * (D_prev[z] + D_next[z])  # (T,2) nan at ends
        if np.isfinite(kk).any():
            med = np.nanmedian(kk, axis=0); nph = int(np.isfinite(kk[:, 0]).sum())
            noise = float(np.sqrt(np.nanmean(np.sum((kk - med) ** 2, axis=1))))
        else:
            med = np.array([np.nan, np.nan]); nph = 0; noise = np.nan
        r.update(img_spike_x=med[0], img_spike_y=med[1], img_spike=float(np.hypot(*med)), img_spike_nph=nph,
                 img_spike_noise=noise, img_ncc_prev=float(np.nanmedian(P_prev[z])) if np.isfinite(P_prev[z]).any() else np.nan,
                 img_ncc_next=float(np.nanmedian(P_next[z])) if np.isfinite(P_next[z]).any() else np.nan,
                 img_dprev_x=float(np.nanmedian(D_prev[z, :, 0])), img_dprev_y=float(np.nanmedian(D_prev[z, :, 1])),
                 img_dnext_x=float(np.nanmedian(D_next[z, :, 0])), img_dnext_y=float(np.nanmedian(D_next[z, :, 1])),
                 img_pin_prev=PIN_prev[z], img_pin_next=PIN_next[z], img_flat_prev=FLAT_prev[z], img_flat_next=FLAT_next[z],
                 roi_area=int(mask[z].sum().item()))
        rows.append(r)
    # line residual (LV, deg1, ED and phase-median)
    lr = np.full((Z, T), np.nan)
    for t in range(T):
        ok = np.isfinite(C["lv"][:, t, 0]); zz = np.nonzero(ok)[0]
        if len(zz) < 4: continue
        rx = robust_polyfit(zz.astype(float), C["lv"][zz, t, 0], 1); ry = robust_polyfit(zz.astype(float), C["lv"][zz, t, 1], 1)
        lr[zz, t] = np.hypot(rx, ry)
    for r in rows:
        r["lv_lineres_ed"] = lr[r["z"], 0]; r["lv_lineres_med"] = np.nanmedian(lr[r["z"]]) if np.isfinite(lr[r["z"]]).any() else np.nan
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sections", default="train,val,test"); ap.add_argument("--subjects", default="")
    a = ap.parse_args()
    subs = read_pooled(tuple(a.sections.split(",")))
    if a.subjects: subs = [(s, "?") for s in a.subjects.split(",")]
    if a.limit: subs = subs[: a.limit]
    allrows = []; t0 = time.time(); nerr = 0
    for i, (rel, sec) in enumerate(subs):
        try:
            rows = analyze(rel)
            for r in rows: r["split"] = sec
            allrows += rows
        except Exception as e:
            nerr += 1; print("ERR", rel, repr(e), flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(subs):
            print(f"{i+1}/{len(subs)} err={nerr} {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(allrows).to_csv(a.out, index=False)
    pd.DataFrame(allrows).to_csv(a.out, index=False)
