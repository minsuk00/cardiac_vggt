"""Render LAX + full SAX stack panels so a human can measure the true slice pitch.

Background: 114 CMRxRecon2025 subjects (Prisma/C006, Aera/C004, Vida/C001 -- all Siemens) ship an
EMPTY `SliceThickness`, so their pitch was defaulted to 12 mm. This renders, per subject:

  * the 4ch (and 2ch, if present) LONG-AXIS view with a 20 mm grid -- measure LV length `L`
    from the mitral valve plane to the LV apex. In-plane spacing here is KNOWN, and the render
    pipeline is validated to reproduce `reconstruct_subject`'s grid and FOV exactly.
  * every SAX slice, numbered -- record the most basal (`i`) and most apical (`j`) slice that
    still shows LV.

Then:  pitch = L / (j - i)      <- INTERVALS, not slices

The LAX is reconstructed here by RSS coil-combine straight from k-space (no SENSE) -- fine for
measuring anatomy, and it avoids depending on a SAX-shaped recon path.

    micromamba run -n svr python tools/render_pitch_measurement_panels.py
"""
import csv
import json
import os

import h5py
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D25 = os.path.join(REPO, "scratch", "data", "CMRxRecon2025")
OUTDIR = os.path.join(REPO, "result", "pitch_measurement")
TOK = {("TaskR1", "TestSet"): "R1test", ("TaskR1", "ValidationSet"): "R1val",
       ("TaskR2", "TestSet"): "R2test", ("TaskR2", "ValidationSet"): "R2val",
       ("TrainingData", "TrainingSet"): "train"}

# Batch 2: 6 more Prisma spanning Z=9..17 (leverage for the through-origin L-vs-N slope) plus the
# only 2 remaining Aera subjects that have a LAX at all. Aera therefore caps at n=4.
SUBJECTS_BATCH2 = [
    ("UNKNOWN", "CMRx25_R2test_Center006_Siemens_30T_Prisma_P010"),   # Z=9
    ("UNKNOWN", "CMRx25_R2test_Center006_Siemens_30T_Prisma_P005"),   # Z=13
    ("UNKNOWN", "CMRx25_R2test_Center006_Siemens_30T_Prisma_P012"),   # Z=13
    ("UNKNOWN", "CMRx25_R2test_Center006_Siemens_30T_Prisma_P006"),   # Z=14
    ("UNKNOWN", "CMRx25_R1test_Center006_Siemens_30T_Prisma_P021"),   # Z=15
    ("UNKNOWN", "CMRx25_R2test_Center006_Siemens_30T_Prisma_P017"),   # Z=17
    ("UNKNOWN", "CMRx25_R1test_Center004_Siemens_15T_Aera_P002"),     # Z=9
    ("UNKNOWN", "CMRx25_R1test_Center004_Siemens_15T_Aera_P008"),     # Z=9
]

SUBJECTS = [
    ("UNKNOWN", "CMRx25_R2test_Center006_Siemens_30T_Prisma_P020"),
    ("UNKNOWN", "CMRx25_R1test_Center006_Siemens_30T_Prisma_P024"),
    ("UNKNOWN", "CMRx25_R1test_Center004_Siemens_15T_Aera_P009"),
    ("UNKNOWN", "CMRx25_R1test_Center004_Siemens_15T_Aera_P012"),
    ("UNKNOWN", "CMRx25_R2test_Center001_Siemens_30T_Vida_P012"),
    ("UNKNOWN", "CMRx25_R2test_Center001_Siemens_30T_Vida_P006"),
    ("CONTROL", "CMRx25_R2test_Center005_Siemens_30T_Vida_P020"),
    ("CONTROL", "CMRx25_R2test_Center005_Siemens_30T_Vida_P012"),
    ("CONTROL", "CMRx25_R2test_Center002_Siemens_30T_CIMAX_P008"),
    ("CONTROL", "CMRx25_R2test_Center002_Siemens_30T_CIMAX_P020"),
]


def read_info(p):
    d = {}
    for r in csv.reader(open(p)):
        if len(r) >= 2:
            k = r[0].strip()
            for suf in ("(mm)", "(ms)", "(degree)"):
                k = k.replace(suf, "")
            d[k] = r[1].strip()
    return d


def load_k(mat):
    """2025 ships BOTH HDF5 v7.3 and MATLAB v5."""
    with open(mat, "rb") as f:
        v5 = f.read(10) == b"MATLAB 5.0"
    if v5:
        import scipy.io as sio
        name = next(e[0] for e in sio.whosmat(mat) if e[0] in ("kspace", "kspace_full"))
        return np.transpose(sio.loadmat(mat)[name], (4, 3, 2, 1, 0))
    with h5py.File(mat, "r") as f:
        k = f["kspace" if "kspace" in f else "kspace_full"][:]
    return k["real"] + 1j * k["imag"]


def recon_lax(mat, rx, ry):
    """RSS coil-combine of t=0, slice 0. Crop readout oversampling to ReconMatrix."""
    k = load_k(mat)[0, 0]
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1)),
                          axes=(-2, -1))
    m = np.sqrt((np.abs(img) ** 2).sum(0))
    ny, nx = m.shape
    if nx > rx:
        x0 = (nx - rx) // 2
        m = m[:, x0:x0 + rx]
    if ny > ry:
        y0 = (ny - ry) // 2
        m = m[y0:y0 + ry]
    return m


def raw_map():
    scan = json.load(open(os.path.join(D25, "duplicates_scan_v2.json")))["records"]
    out = {}
    for k, v in scan.items():
        p = k.split("/")
        t = TOK.get((p[0], p[1]))
        if t:
            out[f"CMRx25_{t}_{p[-3]}_{p[-2]}_{p[-1]}"] = os.path.dirname(v["path"])
    return out


def draw_lax(ax, base, view, label):
    csvp = f"{base}/cine_lax_{view}_info.csv"
    matp = f"{base}/cine_lax_{view}.mat"
    if not (os.path.exists(csvp) and os.path.exists(matp)):
        ax.set_visible(False)
        return False
    d = read_info(csvp)
    rx, ry = int(float(d["ReconMatrix_X"])), int(float(d["ReconMatrix_Y"]))
    px, py = float(d["FOVx"]) / rx, float(d["FOVy"]) / ry
    m = recon_lax(matp, rx, ry)
    H, W = m.shape
    ax.imshow(m, cmap="gray", vmin=0, vmax=np.percentile(m, 99),
              extent=[0, W * px, H * py, 0], aspect=1.0)
    for g in range(0, int(W * px) + 1, 20):
        ax.axvline(g, color="cyan", lw=0.4, alpha=0.5)
    for g in range(0, int(H * py) + 1, 20):
        ax.axhline(g, color="cyan", lw=0.4, alpha=0.5)
    y0 = H * py - 14
    ax.plot([10, 60], [y0, y0], color="yellow", lw=4, solid_capstyle="butt")
    ax.text(10, y0 - 5, "50 mm", color="yellow", fontsize=11, fontweight="bold")
    ax.set_title(f"{label}   (grid = 20 mm, pixel {px:.2f}x{py:.2f} mm)", fontsize=9)
    ax.set_xlabel("mm", fontsize=8)
    ax.tick_params(labelsize=7)
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1, choices=[1, 2])
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="keep this central fraction of the SAX FOV (the heart sits near the "
                         "acquisition centre), so the LV is big enough to count reliably")
    ap.add_argument("--ncol", type=int, default=6)
    args = ap.parse_args()
    subjects = SUBJECTS if args.batch == 1 else SUBJECTS_BATCH2

    os.makedirs(OUTDIR, exist_ok=True)
    raw = raw_map()
    rep = {x["cid"]: x for x in json.load(open(os.path.join(D25, "recon_report.json")))}

    for tag, cid in subjects:
        base = raw.get(cid)
        nii = os.path.join(D25, "Cine_combined", cid, "sax", "4d_recon.nii.gz")
        if not (base and os.path.exists(nii)):
            print(f"  SKIP {cid}: raw={bool(base)} nii={os.path.exists(nii)}")
            continue
        img = nib.load(nii)
        v = np.asanyarray(img.dataobj)
        zo = img.header.get_zooms()
        nz = v.shape[2]
        if args.zoom < 1.0:                       # centre-crop so the LV is large enough to read
            X, Y = v.shape[0], v.shape[1]
            cx, cy = int(X * args.zoom), int(Y * args.zoom)
            v = v[(X - cx) // 2:(X - cx) // 2 + cx, (Y - cy) // 2:(Y - cy) // 2 + cy]
        ncol = args.ncol
        nrow = int(np.ceil(nz / ncol))

        fig = plt.figure(figsize=(16, 8.5 + 3.6 * nrow), dpi=115)
        gs = GridSpec(1 + nrow, ncol, figure=fig, height_ratios=[3.4] + [1.0] * nrow,
                      hspace=0.30, wspace=0.06)

        ax4 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[0, 3:6])
        draw_lax(ax4, base, "4ch", "LONG AXIS 4ch  — measure LV length L (mitral plane → apex)")
        draw_lax(ax2, base, "2ch", "LONG AXIS 2ch  — cross-check L")

        vmax = float(np.percentile(v[..., 0], 99.0)) or 1.0
        for z in range(nz):
            ax = fig.add_subplot(gs[1 + z // ncol, z % ncol])
            ax.imshow(v[:, :, z, 0].T, cmap="gray", vmin=0, vmax=vmax,
                      aspect=zo[1] / zo[0], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"SAX  z = {z}", fontsize=10, fontweight="bold", color="darkblue")

        x = rep.get(cid, {})
        note = (f"{tag}   {cid}\n"
                f"SAX slices = {nz}   |   currently ASSUMED pitch = {x.get('pitch_mm','?')} mm"
                f"   |   SliceThickness in source = "
                f"{'(EMPTY)' if not x.get('thickness_csv') else 'documented'}\n"
                f"Measure L on the long axis, then record most-basal / most-apical SAX slice "
                f"showing LV.    pitch = L / (j - i)")
        fig.suptitle(note, fontsize=11, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        out = os.path.join(OUTDIR, f"{tag}_{cid}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}  (Z={nz})")


if __name__ == "__main__":
    main()
