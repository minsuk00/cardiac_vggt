"""Slab-style reslice panel per subject: each SAX slice drawn as a band of its acquired THICKNESS at its
true z (pitch), inter-slice gap black, nearest-neighbour (no interpolation) so a breath-hold-shifted slice
shows as a stair. Coronal + sagittal cuts through the LV centre, per-slice spike/step annotated, and the
real 4-chamber LAX cine frame beside it when the source has one (CMRx; no geometry, visual reference only).
Usage: python render_slab.py <subjects_v3.csv> <slices_v3.csv> <steps_v3.csv> <outdir> [rel ...]
"""
import sys, os, glob, csv; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # writes to temp/misalign_v2/slabs by default via build_gallery.py's SL
from lib import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd

def thickness_mm(rel, dz):
    p = os.path.join(DATA, rel, "sax", "cine_sax_info.csv")
    if os.path.exists(p):
        for row in csv.reader(open(p)):
            if row and row[0] == "SliceThickness":
                try:
                    t = float(row[1]); return (min(t, dz), "header" if t <= dz else "header, capped at pitch")
                except: pass
    return min(8.0, dz), "assumed"   # ACDC/MNMs ship no thickness; typical 6-10 mm

def lax_frame(rel):
    p = os.path.join(DATA, rel, "lax", "4d_recon.nii.gz")
    if not os.path.exists(p): return None
    a = np.asarray(nib.load(p).dataobj)
    if a.ndim == 4: a = a[..., 0]
    return [a[:, :, k] for k in range(a.shape[2])]   # CMRx ships 2ch/3ch/4ch as 3 "slices"; order not guaranteed

BAND_PX = 22  # rows per slice band, equal height, zero gap -- NOT physically-true spacing (that version was
              # too thin/sparse to read); this maximizes legibility of the stair pattern. No interpolation: each
              # band is a flat fill of that slice's own pixels (nearest-neighbour), never blended with a neighbour.
def slab(vol2d, Z, dz, th, pix):
    """vol2d: (N, Z) in-plane-by-slice; returns image with Z equal-height bands stacked with NO gap (flat fill,
    no interpolation across slices). Display with imshow(..., aspect='auto')."""
    out = np.repeat(vol2d.T, BAND_PX, axis=0)  # (Z*BAND_PX, N), each slice's row block repeated -> flat band
    return out

def render(rel, sub, sl, st, out_png):
    seg, (dx, dy, dz) = load_seg(rel); img = load_4d(rel); X, Y, Z, T = seg.shape
    th, thsrc = thickness_mm(rel, dz)
    ed = img[..., 0]; lv = seg[..., 0] == 1
    zs = [z for z in range(Z) if lv[:, :, z].sum() > 15]
    cxm = int(np.median([np.nonzero(lv[:, :, z])[0].mean() for z in zs])); cym = int(np.median([np.nonzero(lv[:, :, z])[1].mean() for z in zs]))
    w = int(40 / dx); h = int(40 / dy); x0, x1 = max(cxm - w, 0), min(cxm + w, X); y0, y1 = max(cym - h, 0), min(cym + h, Y)
    crop = ed[x0:x1, y0:y1]; vmax = np.percentile(crop[crop > 0], 99) if (crop > 0).any() else ed.max()
    d = sl[sl.rel == rel].set_index("z"); ss = st[st.rel == rel].set_index("z") if len(st) else None
    lax = lax_frame(rel)
    ncol = 3 if lax is not None else 2
    fig = plt.figure(figsize=(3.4 * ncol + 1.7, 4.2), dpi=130)
    gs = fig.add_gridspec(1, ncol + 1, width_ratios=[1] * ncol + [0.55])
    k = 0
    if lax is not None:
        sg = gs[0, 0].subgridspec(len(lax), 1); k += 1
        for i, L in enumerate(lax):
            ax = fig.add_subplot(sg[i, 0]); ax.imshow(L.T, cmap="gray", vmin=0, vmax=np.percentile(L[L > 0], 99.5), origin="lower", aspect="auto"); ax.axis("off")
            if i == 0: ax.set_title("real LAX views (reference only)", fontsize=7)
    c3 = 3  # average a ~4-8 mm thick cut around the LV centre for SNR
    epi = np.isin(seg[..., 0], (1, 2))
    for name, arr, pix, ext in (("coronal", ed[x0:x1, max(cym - c3, 0):cym + c3 + 1, :].mean(1), dx, [epi[x0:x1, cym, z] for z in range(Z)]),
                                ("sagittal", ed[max(cxm - c3, 0):cxm + c3 + 1, y0:y1, :].mean(0), dy, [epi[cxm, y0:y1, z] for z in range(Z)])):
        ax = fig.add_subplot(gs[0, k]); k += 1
        ax.imshow(slab(arr, Z, dz, th, pix), cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
        ax.set_title(f"{name}: native slices stacked, equal height, no gap, no interpolation", fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        for z in range(Z):
            zc = (z + 0.5) * BAND_PX; nz = np.nonzero(ext[z])[0]
            if len(nz): ax.plot([nz.min(), nz.max()], [zc, zc], "|", color="yellow", ms=7, mew=1.2)
            if z in d.index and np.isfinite(d.loc[z, "spike"]) and d.loc[z, "spike"] > 5:
                ax.plot(1.5, zc, "r>", ms=5)
    ax = fig.add_subplot(gs[0, k]); ax.axis("off"); lines = []
    # † = segmentation-only value (registration pinned at its search bound, i.e. offset beyond the window)
    for z in range(Z):
        kk = d.loc[z, "spike"] if z in d.index else np.nan; sp = ss.loc[z, "step"] if (ss is not None and z in ss.index) else np.nan
        kf = "†" if (z in d.index and d.loc[z, "spike_src"] == "seg_fallback") else " "
        sf = "†" if (ss is not None and z in ss.index and ss.loc[z, "step_src"] == "seg_fallback") else " "
        lines.append(f"z{z:2d}  spike {kk:4.1f}{kf} step→{z+1} {sp:4.1f}{sf}" if np.isfinite(kk) or np.isfinite(sp) else f"z{z:2d}   —")
    ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=6.5, va="top", transform=ax.transAxes)
    r = sub.loc[rel]
    fig.suptitle(f"{rel}   severity {r.severity:.1f} mm (spike {r.max_spike:.1f} / step {r.max_step:.1f})   Z={Z} pitch {dz:.0f} mm   {r.centre_key}", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

if __name__ == "__main__":
    sub = pd.read_csv(sys.argv[1], index_col=0); sl = pd.read_csv(sys.argv[2]); st = pd.read_csv(sys.argv[3]); outdir = sys.argv[4]; os.makedirs(outdir, exist_ok=True)
    rels = sys.argv[5:] or list(sub[(sub.measurable) & (sub.severity > 8)].sort_values("severity", ascending=False).index)
    for i, rel in enumerate(rels):
        try: render(rel, sub, sl, st, os.path.join(outdir, rel.split("/")[-1] + ".png"))
        except Exception as e: print("ERR", rel, repr(e), flush=True)
        if (i + 1) % 25 == 0: print(i + 1, "/", len(rels), flush=True)
    print("done", len(rels))
