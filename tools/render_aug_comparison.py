"""Visualize augmentation effects: Rician, Low-Res, Gibbs, FDA.

Loads a real CMRx SAX slice, applies each augmentation, shows before/after,
and times each transform on GPU.

Output: result/aug_comparison.png
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "training")
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── batchaug ──────────────────────────────────────────────────────────────────
import batchaug as _B
_B.set_backend("pytorch")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NIFTI = "/home/minsukc/vggt/data/CMRxRecon2024/Cine_combined/CMRx24_Test_P001/sax/4d_recon.nii.gz"
OUT   = "/home/minsukc/vggt/result/aug_comparison.png"

N_REPEATS = 50   # timing repeats per transform

# ── Load one 2D SAX slice ─────────────────────────────────────────────────────
img4d = nib.load(NIFTI).get_fdata()   # (X, Y, Z, T)
# Pick mid-z, phase 0 — square crop
mid_z = img4d.shape[2] // 2
sl = img4d[:, :, mid_z, 0].astype(np.float32)
# Normalise to [0, 1]
sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)

# Convert to 5D batchaug tensor: (B=1, T=1, D=1, H, W)
H, W = sl.shape
t = torch.from_numpy(sl).float().to(DEVICE)
t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0)   # (1, 1, 1, H, W)

def to_np(x):
    """5D → 2D numpy, squeeze all singleton dims."""
    return x.squeeze().cpu().float().numpy()

def time_transform(fn, n=N_REPEATS):
    """Warm up, then time n rounds, return ms/call."""
    for _ in range(5):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000   # ms

# ── Define transforms ─────────────────────────────────────────────────────────

# 1. Rician noise (physics-correct MRI noise)
rician = _B.RandRicianNoise(prob=1.0, std=0.03, relative=False, sample_std=False)

def apply_rician():
    return rician(t)

# 2. Simulated low-resolution (nnU-Net style)
# batchaug expects 5D (B,C,H,W,D) — use trilinear upsample (D=1 is the singleton)
lowres = _B.RandSimulateLowResolution(prob=1.0, zoom_range=(0.4, 0.6),
                                      downsample_mode="nearest",
                                      upsample_mode="trilinear",
                                      align_corners=True)

def apply_lowres():
    return lowres(t)

# 3. Gibbs / k-space ringing
gibbs = _B.RandGibbsNoise(prob=1.0, alpha=(0.75, 0.80))

def apply_gibbs():
    return gibbs(t)

# 4. FDA — Fourier Amplitude Mixup
# We need two images. Use a different phase of the same volume as the "target style".
img4d2 = img4d[:, :, mid_z, 6].astype(np.float32)   # phase 6 as style donor
img4d2 = (img4d2 - img4d2.min()) / (img4d2.max() - img4d2.min() + 1e-8)
t_style = torch.from_numpy(img4d2).float().to(DEVICE).unsqueeze(0).unsqueeze(0).unsqueeze(0)

def fda_2d(src, trg, beta=0.08):
    """Fourier amplitude mixup on (B,C,D,H,W) via 2D FFT over H,W."""
    src_f  = torch.fft.fft2(src.float())
    trg_f  = torch.fft.fft2(trg.float())
    src_f  = torch.fft.fftshift(src_f,  dim=(-2, -1))
    trg_f  = torch.fft.fftshift(trg_f,  dim=(-2, -1))
    amp_s  = torch.abs(src_f)
    amp_t  = torch.abs(trg_f)
    phase_s = torch.angle(src_f)
    # Swap central low-freq amplitude region
    _, _, _, h, w = src.shape
    ch, cw = int(h * beta), int(w * beta)
    h2, w2 = h // 2, w // 2
    amp_mixed = amp_s.clone()
    amp_mixed[..., h2 - ch:h2 + ch, w2 - cw:w2 + cw] = \
        amp_t[...,   h2 - ch:h2 + ch, w2 - cw:w2 + cw]
    fused = amp_mixed * torch.exp(1j * phase_s)
    out   = torch.fft.ifftshift(fused, dim=(-2, -1))
    return torch.fft.ifft2(out).real.clamp(0, 1).to(src.dtype)

def apply_fda():
    return fda_2d(t, t_style, beta=0.08)

# ── Run & time ─────────────────────────────────────────────────────────────────
print(f"Running on: {DEVICE}")
print("Timing transforms...")

rician_img  = to_np(apply_rician())
lowres_img  = to_np(apply_lowres())
gibbs_img   = to_np(apply_gibbs())
fda_img     = to_np(apply_fda())
orig_img    = to_np(t)
style_img   = to_np(t_style)

ms_rician  = time_transform(apply_rician)
ms_lowres  = time_transform(apply_lowres)
ms_gibbs   = time_transform(apply_gibbs)
ms_fda     = time_transform(apply_fda)

print(f"  Rician noise     : {ms_rician:.3f} ms/call")
print(f"  Simulated low-res: {ms_lowres:.3f} ms/call")
print(f"  Gibbs ringing    : {ms_gibbs:.3f} ms/call")
print(f"  FDA amplitude    : {ms_fda:.3f} ms/call")

# ── Compute residuals (amplified for visibility) ──────────────────────────────
AMP = 5.0   # amplification factor
res_rician  = np.clip((rician_img  - orig_img)  * AMP + 0.5, 0, 1)
res_lowres  = np.clip((lowres_img  - orig_img)  * AMP + 0.5, 0, 1)
res_gibbs   = np.clip((gibbs_img   - orig_img)  * AMP + 0.5, 0, 1)
res_fda     = np.clip((fda_img     - orig_img)  * AMP + 0.5, 0, 1)

# ── Plot ────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor="#1a1a2e")
fig.suptitle("GPU Augmentation Comparison — VGGT-MRI", fontsize=15,
             color="white", fontweight="bold", y=0.99)

cols = 5
rows = 3   # original, augmented, residual
gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.40, wspace=0.08,
                       left=0.03, right=0.97, top=0.95, bottom=0.04)

cmap_gray = "gray"
cmap_res  = "RdBu_r"   # diverging: blue=negative, red=positive diff

def add_ax(fig, gs_pos, img, title, subtitle=None, highlight=False, cmap="gray", vmin=0, vmax=1):
    ax = fig.add_subplot(gs_pos)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    color = "#f39c12" if highlight else "white"
    ax.set_title(title, color=color, fontsize=8.5, fontweight="bold", pad=3)
    if subtitle:
        ax.text(0.5, -0.04, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.set_facecolor("#0d0d1a")
    return ax

# Row 0 — originals / style donors
add_ax(fig, gs[0, 0], orig_img,   "Original", "Phase 0 (no aug)")
add_ax(fig, gs[0, 1], orig_img,   "→ Rician Noise",  "input")
add_ax(fig, gs[0, 2], orig_img,   "→ Simulated Low-Res", "input")
add_ax(fig, gs[0, 3], orig_img,   "→ Gibbs Ringing", "input")
add_ax(fig, gs[0, 4], style_img,  "FDA Style Donor", "Phase 6 (different contrast)")

# Row 1 — augmented
add_ax(fig, gs[1, 0], orig_img,   "Original", "(reference)")
add_ax(fig, gs[1, 1], rician_img, "Rician Noise",
       f"std=0.03 | {ms_rician:.2f} ms", highlight=True)
add_ax(fig, gs[1, 2], lowres_img, "Simulated Low-Res",
       f"zoom 0.4–0.6× | {ms_lowres:.2f} ms", highlight=True)
add_ax(fig, gs[1, 3], gibbs_img,  "Gibbs Ringing",
       f"alpha 0.75–0.80 | {ms_gibbs:.2f} ms", highlight=True)
add_ax(fig, gs[1, 4], fda_img,    "FDA Amplitude Mix",
       f"β=0.08 | {ms_fda:.2f} ms", highlight=True)

# Row 2 — residuals (augmented - original, amplified 5×, centred at 0.5)
add_ax(fig, gs[2, 0], np.ones_like(orig_img) * 0.5, "Residual (aug − orig)",
       f"amplified {AMP:.0f}×, grey=no change", cmap=cmap_res, vmin=0, vmax=1)
add_ax(fig, gs[2, 1], res_rician, "Rician residual",
       None, highlight=True, cmap=cmap_res, vmin=0, vmax=1)
add_ax(fig, gs[2, 2], res_lowres, "Low-Res residual",
       None, highlight=True, cmap=cmap_res, vmin=0, vmax=1)
add_ax(fig, gs[2, 3], res_gibbs,  "Gibbs residual",
       None, highlight=True, cmap=cmap_res, vmin=0, vmax=1)
add_ax(fig, gs[2, 4], res_fda,    "FDA residual",
       None, highlight=True, cmap=cmap_res, vmin=0, vmax=1)

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor=fig.get_facecolor())
print(f"Saved to {OUT}")
