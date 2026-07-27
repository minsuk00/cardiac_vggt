"""Does the 2024 recon's ESPIRiT call being fed image-domain data actually matter?

`_archive/batch_reconstruct_cmrxrecon2024.py` does:
    ref_image = sp.ifft(ref_kspace)          # image domain
    smap = mr.app.EspiritCalib(ref_image, ...)   # <-- API expects K-SPACE

This compares, on one slice/frame:
  A  as-shipped   : EspiritCalib(ifft(ksp))  + SENSE combine   (what made Cine_combined/)
  B  textbook     : EspiritCalib(ksp)        + SENSE combine
  C  RSS          : sqrt(sum |ifft(ksp)|^2)  (no sensitivity maps at all)

Usage: python tools/probe_espirit_domain.py
Writes result/recon_verify_2024/espirit_domain_check.png
"""

import os

import cupy as cp
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.mri as mr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT = f"{REPO}/scratch/data/CMRxRecon2024/ChallengeData/Cine/TrainingSet/FullSample/P010/cine_sax.mat"
RECON_Y, RECON_X = 204, 256
SLC, FRAME = 4, 0
DEV = sp.Device(0)


def sense_combine(smap, img):
    num = cp.sum(cp.conj(smap) * img, axis=0)
    den = cp.sum(cp.abs(smap) ** 2, axis=0) + 1e-8
    return cp.abs(num / den).get().astype(np.float32)


def crop(a):
    ny, nx = a.shape
    y0 = max(0, (ny - RECON_Y) // 2)
    x0 = max(0, (nx - RECON_X) // 2)
    return a[y0 : y0 + RECON_Y, x0 : x0 + RECON_X]


def nrmse(a, b):
    """Scale-invariant: b is least-squares rescaled onto a first (maps carry arbitrary gain)."""
    s = float(np.sum(a * b) / np.sum(b * b))
    return float(np.linalg.norm(a - s * b) / np.linalg.norm(a)), s


def main():
    with h5py.File(MAT, "r") as f:
        d = f["kspace_full"]
        ksp = d["real"][FRAME, SLC] + 1j * d["imag"][FRAME, SLC]  # (ncoil, ny, nx)
        ref = d["real"][0, SLC] + 1j * d["imag"][0, SLC]

    ksp_g, ref_g = cp.array(ksp), cp.array(ref)
    img_g = sp.ifft(ksp_g, axes=[-2, -1])
    ref_img_g = sp.ifft(ref_g, axes=[-2, -1])

    kw = dict(crop=0.80, thresh=0.01, calib_width=32, device=DEV, show_pbar=False)
    smap_A = mr.app.EspiritCalib(ref_img_g, **kw).run()  # as-shipped (image in)
    smap_B = mr.app.EspiritCalib(ref_g, **kw).run()  # textbook (k-space in)

    A = crop(sense_combine(smap_A, img_g))
    B = crop(sense_combine(smap_B, img_g))
    C = crop(cp.sqrt(cp.sum(cp.abs(img_g) ** 2, axis=0)).get().astype(np.float32))

    cov_A = float((cp.abs(smap_A).get() > 0).mean())
    cov_B = float((cp.abs(smap_B).get() > 0).mean())

    print(f"ESPIRiT map support (nonzero frac): A(as-shipped)={cov_A:.3f}  B(textbook)={cov_B:.3f}")
    for name, X in (("B textbook", B), ("C RSS", C)):
        e, s = nrmse(A, X)
        print(f"A vs {name:12s}: scale-invariant NRMSE = {e:.4f}  (gain {s:.3f})  corr = {np.corrcoef(A.ravel(), X.ravel())[0,1]:.5f}")
    e, s = nrmse(B, C)
    print(f"B vs C RSS      : scale-invariant NRMSE = {e:.4f}  corr = {np.corrcoef(B.ravel(), C.ravel())[0,1]:.5f}")

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    panels = [
        ("A  as-shipped\nEspirit(ifft(ksp))", A),
        ("B  textbook\nEspirit(ksp)", B),
        ("C  RSS\n(no maps)", C),
    ]
    for i, (t, X) in enumerate(panels):
        ax[0, i].imshow(X, cmap="gray", vmin=0, vmax=np.percentile(X, 99.5))
        ax[0, i].set_title(t, fontsize=9)
    eB, sB = nrmse(A, B)
    eC, sC = nrmse(A, C)
    ax[0, 3].imshow(np.abs(A - sB * B), cmap="magma")
    ax[0, 3].set_title(f"|A - B|\nNRMSE {eB:.3f}", fontsize=9)
    ax[1, 0].imshow(np.abs(cp.asnumpy(cp.abs(smap_A))[0]), cmap="viridis")
    ax[1, 0].set_title(f"|smap A| coil0 (support {cov_A:.2f})", fontsize=9)
    ax[1, 1].imshow(np.abs(cp.asnumpy(cp.abs(smap_B))[0]), cmap="viridis")
    ax[1, 1].set_title(f"|smap B| coil0 (support {cov_B:.2f})", fontsize=9)
    ax[1, 2].imshow(np.abs(A - sC * C), cmap="magma")
    ax[1, 2].set_title(f"|A - RSS|\nNRMSE {eC:.3f}", fontsize=9)
    ax[1, 3].axis("off")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Train_P010 slice {SLC} frame {FRAME} — ESPIRiT input-domain check", fontsize=11)
    out = f"{REPO}/result/recon_verify_2024/espirit_domain_check.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
