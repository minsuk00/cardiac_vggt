"""Re-render gather05 mid-ref all-slices from saved npz: blank input/DVF where no real input
slot exists (z0,z11), and use TRUE coverage range (no clip)."""
import numpy as np, sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.adapters.base import D_CANON

NF = 30; FPS = 6
d = dict(np.load("result/gather05_allslices_V1.npz"))
RE, CO, DV, IN = d["recon"], d["cov"], d["dvf"], d["inp"]      # (NF,12,256,256)*, dvf (NF,12,256,256,3)
slot_plane = d["slot_plane"]
has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]   # planes with a real input slice
print("planes with real input slice:", [p for p in range(D_CANON) if has_slot[p]], flush=True)

vlx, vly, vlz = [max(1.0, np.percentile(np.abs(DV[..., k]), 99)) for k in range(3)]
covmax = float(CO.max())                                       # TRUE range, no clip
ivmax = np.percentile(IN, 99.5); rvmax = np.percentile(RE, 99.5)
blank = np.zeros((256, 256), np.float32)
rows = [("input", IN, "gray", 0, ivmax, True),
        ("recon", RE, "gray", 0, rvmax, False),
        (f"Dx ±{vlx:.0f}", DV[..., 0], "bwr", -vlx, vlx, True),
        (f"Dy ±{vly:.0f}", DV[..., 1], "bwr", -vly, vly, True),
        (f"Dz ±{vlz:.0f}mm", DV[..., 2], "bwr", -vlz, vlz, True),
        (f"cov 0-{covmax:.0f}", CO, "viridis", 0, covmax, False)]
frames = []
for t in range(NF):
    fig, axs = plt.subplots(6, D_CANON, figsize=(24, 12), dpi=90)
    for ri, (lab, arr, cm, lo, hi, blankable) in enumerate(rows):
        for p in range(D_CANON):
            img = arr[t, p] if (has_slot[p] or not blankable) else blank
            axs[ri, p].imshow(img, cmap=cm, vmin=lo, vmax=hi); axs[ri, p].axis("off")
            if ri == 0: axs[ri, p].set_title(f"z{p}" + ("" if has_slot[p] else "\n(no input)"), fontsize=9)
            if p == 0: axs[ri, p].text(-0.35, 0.5, lab, transform=axs[ri, p].transAxes, rotation=90, va="center", fontsize=10)
    fig.suptitle(f"gather05 (snapped), reference=z5. rows input/recon/Dx/Dy/Dz/coverage; blank input/DVF = no acquired slice. frame {t}/{NF-1}", fontsize=12)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimwrite("result/mp4_gather05_allslices_fixed_V1.mp4", frames, fps=FPS, codec="libx264",
                 macro_block_size=16, ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p"])
print(f"saved result/mp4_gather05_allslices_fixed_V1.mp4  true covmax={covmax:.1f}\nDONE", flush=True)
