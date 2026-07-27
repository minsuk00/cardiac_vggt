"""Visual side-by-side of the CMRxRecon2023 <-> CMRxRecon-300 subject-ID join.

For each subject: a quick RSS image straight from the 2023 challenge k-space (no CSV needed —
iFFT + coil RSS, so this works BEFORE any geometry is borrowed), next to CMRxRecon-300's shipped
recon for the SAME id, next to a DIFFERENT id as a contrast.

If the ID join is real, column 2 is obviously the same heart as column 1 and column 3 obviously is not.

Usage: python tools/render_cmrx2023_donor_identity_panel.py
Writes result/cmrx2023_donor_identity/panel.png
"""

import importlib.util
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("v", os.path.join(REPO, "tools", "verify_cmrx2023_donor_identity.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
D = v.D

# best case, worst case, thinnest rank-1 margin, and a KNOWN non-match (P118)
CASES = [
    ("TrainingSet", "P007", "P050", "highest NCC of the sample"),
    ("TestSet", "P076", "P099", "highest NCC of the sample"),
    ("ValidationSet", "P011", "P047", "low absolute NCC, still rank-1/41"),
    ("TestSet", "P099", "P098", "LOWEST NCC of the sample, rank-1/41"),
    ("TrainingSet", "P052", "P029", "THINNEST rank-1 margin (+0.055)"),
    ("TestSet", "P118", "P097", "KNOWN NON-MATCH — no donor exists"),
]


def match_shape(a, shape):
    """Centre-crop / zero-pad `a` to `shape` so the two can be shown and compared side by side."""
    out = np.zeros(shape, dtype=a.dtype)
    ys, xs = min(a.shape[0], shape[0]), min(a.shape[1], shape[1])
    ay, ax = (a.shape[0] - ys) // 2, (a.shape[1] - xs) // 2
    oy, ox = (shape[0] - ys) // 2, (shape[1] - xs) // 2
    out[oy : oy + ys, ox : ox + xs] = a[ay : ay + ys, ax : ax + xs]
    return out


def challenge_mid(section, pid):
    root = os.path.join(D, v.SECTIONS[section], pid, "cine_sax.mat")
    with h5py.File(root, "r") as f:
        d = f["kspace_full"]
        nf, nz = d["real"].shape[0], d["real"].shape[1]
        z = nz // 2
        k = d["real"][0, z] + 1j * d["imag"][0, z]
    return np.sqrt((np.abs(v.centered_ifft2(k)) ** 2).sum(0)), nz


def donor_mid(section, pid, z_frac=0.5):
    nii = os.path.join(D, "CMRxRecon-300", section, pid, "reconstruction", "sax_4d.nii.gz")
    a = sitk.GetArrayFromImage(sitk.ReadImage(nii))
    return a[0, int(a.shape[1] * z_frac)], a.shape


def show(ax, img, label, colour=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5) if img.max() > 0 else 1,
              aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.8))
    if colour:
        for s in ax.spines.values():
            s.set_color(colour)
            s.set_linewidth(3.5)


def main():
    fig, axes = plt.subplots(len(CASES), 3, figsize=(12, 2.9 * len(CASES)))
    for r, (section, pid, wrong, note) in enumerate(CASES):
        q, nz = challenge_mid(section, pid)
        same, _ = donor_mid(section, pid)
        diff, _ = donor_mid(section, wrong)
        n_same, n_diff = v.ncc(q, same), v.ncc(q, diff)
        # show all three at the SAME shape so they render at the same scale and can be eyeballed
        shape = same.shape
        show(axes[r, 0], match_shape(q, shape), f"2023 challenge\n{section}/{pid}\n(raw k-space, no CSV)")
        show(axes[r, 1], same, f"CMRxRecon-300  SAME id\n{section}/{pid}\nNCC {n_same:.3f}",
             "tab:red" if n_same < 0.4 else "lime")
        show(axes[r, 2], match_shape(diff, shape), f"CMRxRecon-300  DIFFERENT id\n{section}/{wrong}\nNCC {n_diff:.3f}")
        axes[r, 0].set_ylabel(note, fontsize=8.5)

    fig.suptitle("Does P### mean the same person in both releases?\n"
                 "left = 2023 raw k-space   middle = SAME id in donor release   right = a DIFFERENT id\n"
                 "(green = matches, red = does not)", fontsize=12)
    out_dir = os.path.join(REPO, "result", "cmrx2023_donor_identity")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "panel.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
