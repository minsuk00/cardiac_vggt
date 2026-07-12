"""gather05 all-slices, INPUT + RECON rows only, as a GIF (grayscale -> GIF palette fine)."""
import numpy as np, sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.adapters.base import D_CANON
NF = 30
d = dict(np.load("result/gather05_allslices_V1.npz"))
IN, RE, slot_plane = d["inp"], d["recon"], d["slot_plane"]
has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]
ivmax = np.percentile(IN, 99.5); rvmax = np.percentile(RE, 99.5); blank = np.zeros((256, 256), np.float32)
frames = []
for t in range(NF):
    fig, axs = plt.subplots(2, D_CANON, figsize=(24, 4.4), dpi=95)
    for p in range(D_CANON):
        axs[0, p].imshow(IN[t, p] if has_slot[p] else blank, cmap="gray", vmin=0, vmax=ivmax); axs[0, p].axis("off")
        axs[0, p].set_title(f"z{p}" + ("" if has_slot[p] else "\n(no input)"), fontsize=9)
        axs[1, p].imshow(RE[t, p], cmap="gray", vmin=0, vmax=rvmax); axs[1, p].axis("off")
        if p == 0:
            axs[0, p].text(-0.35, 0.5, "input", transform=axs[0, p].transAxes, rotation=90, va="center", fontsize=11)
            axs[1, p].text(-0.35, 0.5, "recon", transform=axs[1, p].transAxes, rotation=90, va="center", fontsize=11)
    fig.suptitle(f"gather05 (snapped), reference=z5 — input & recon, all z-planes. frame {t}/{NF-1}", fontsize=12)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave("result/gif_gather05_inputrecon_V1.gif", frames, duration=0.16, loop=0)
print("saved result/gif_gather05_inputrecon_V1.gif", flush=True)
