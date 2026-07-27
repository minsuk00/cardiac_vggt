"""Visualise the slice correspondence between CMRxRecon2023 challenge k-space and CMRxRecon-300 recons.

Some confirmed same-subject pairs scored NCC ~0.57 when slice `nz//2` was compared to slice `Z//2`,
but 0.998 when the best donor slice was chosen. This renders WHY: the two challenge/donor slice
stacks side by side, plus the full NCC matrix so the correspondence (offset? reversal? trimming?)
is directly visible.

Usage: python tools/render_cmrx2023_slice_correspondence.py
Writes result/cmrx2023_donor_identity/slice_correspondence_<Section>_<PID>.png
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

CASES = [("TestSet", "P099"), ("TestSet", "P076")]


def load(section, pid):
    with h5py.File(os.path.join(D, v.SECTIONS[section], pid, "cine_sax.mat"), "r") as f:
        d = f["kspace_full"]
        raw = d["real"][0][:] + 1j * d["imag"][0][:]  # one contiguous read: frame 0, all slices
    q = [np.sqrt((np.abs(v.centered_ifft2(raw[z])) ** 2).sum(0)) for z in range(raw.shape[0])]
    a = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(D, "CMRxRecon-300", section, pid, "reconstruction", "sax_4d.nii.gz")))
    return q, a


def main():
    for section, pid in CASES:
        q, a = load(section, pid)
        nz, Z = len(q), a.shape[1]
        M = np.array([[v.ncc(q[i], a[0, j]) for j in range(Z)] for i in range(nz)])
        am = M.argmax(1)

        ncol = max(nz, Z)
        fig = plt.figure(figsize=(1.5 * ncol + 4, 9))
        gs = fig.add_gridspec(3, ncol, height_ratios=[1, 1, 1.5], hspace=0.35)

        for i in range(nz):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(q[i], cmap="gray", vmin=0, vmax=np.percentile(q[i], 99.5))
            ax.set_title(f"z={i}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("2023 challenge\n(from k-space)", fontsize=8)
        for j in range(Z):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(a[0, j], cmap="gray", vmin=0, vmax=np.percentile(a[0, j], 99.5))
            ax.set_title(f"z={j}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel("CMRxRecon-300\n(shipped recon)", fontsize=8)

        axm = fig.add_subplot(gs[2, :])
        im = axm.imshow(M, cmap="viridis", vmin=-0.1, vmax=1.0, aspect="auto")
        axm.set_xlabel("CMRxRecon-300 donor slice", fontsize=9)
        axm.set_ylabel("2023 challenge slice", fontsize=9)
        axm.set_xticks(range(Z)); axm.set_yticks(range(nz))
        for i in range(nz):
            axm.plot(am[i], i, "r*", ms=13)
            axm.text(am[i], i, f" {M[i, am[i]]:.2f}", color="r", fontsize=7, va="center")
        axm.plot(range(min(nz, Z)), range(min(nz, Z)), "w--", lw=1.2, label="naive z→z (what I compared)")
        axm.legend(fontsize=8, loc="upper right")
        fig.colorbar(im, ax=axm, fraction=0.02, pad=0.01, label="NCC")
        off = [int(am[i]) - i for i in range(nz)]
        axm.set_title(f"{section}/{pid} — NCC(challenge z, donor z).  red ★ = best match per row.  "
                      f"offset(best−naive) = {off}", fontsize=10)

        fig.suptitle(f"{section}/{pid}: why the naive slice pairing scored low  "
                     f"(challenge nz={nz}, donor Z={Z})", fontsize=12)
        out_dir = os.path.join(REPO, "result", "cmrx2023_donor_identity")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"slice_correspondence_{section}_{pid}.png")
        fig.savefig(out, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
        print(f"  {section}/{pid}: nz={nz} Z={Z}  argmax={am.tolist()}  offset={off}  "
              f"diag={[round(float(M[i,i]),2) for i in range(min(nz,Z))]}", flush=True)


if __name__ == "__main__":
    main()
