"""Re-render {model}_allslices_V1.npz as a 6-row GIF (blank input/DVF where no slice; true coverage)."""
import numpy as np, sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.adapters.base import D_CANON
NF = 30
for name in ["control0", "s20contz"]:
    d = dict(np.load(f"result/{name}_allslices_V1.npz"))
    RE, CO, DV, IN, slot_plane, refp = d["recon"], d["cov"], d["dvf"], d["inp"], d["slot_plane"], int(d["refp"])
    has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]
    vlx, vly, vlz = [max(1.0, np.percentile(np.abs(DV[..., k]), 99)) for k in range(3)]
    covmax = float(CO.max()); ivmax = np.percentile(IN, 99.5); rvmax = np.percentile(RE, 99.5); blank = np.zeros((256, 256), np.float32)
    rows = [("input", IN, "gray", 0, ivmax, True), ("recon", RE, "gray", 0, rvmax, False),
            (f"Dx ±{vlx:.0f}", DV[..., 0], "bwr", -vlx, vlx, True), (f"Dy ±{vly:.0f}", DV[..., 1], "bwr", -vly, vly, True),
            (f"Dz ±{vlz:.0f}mm", DV[..., 2], "bwr", -vlz, vlz, True), (f"cov 0-{covmax:.0f}", CO, "viridis", 0, covmax, False)]
    frames = []
    for t in range(NF):
        fig, axs = plt.subplots(6, D_CANON, figsize=(24, 12), dpi=80)
        for ri, (lab, arr, cm, lo, hi, bl) in enumerate(rows):
            for p in range(D_CANON):
                img = arr[t, p] if (has_slot[p] or not bl) else blank
                axs[ri, p].imshow(img, cmap=cm, vmin=lo, vmax=hi); axs[ri, p].axis("off")
                if ri == 0: axs[ri, p].set_title((f"z{p}*REF" if p == refp else f"z{p}") + ("" if has_slot[p] else "\n(no input)"), fontsize=9)
                if p == 0: axs[ri, p].text(-0.35, 0.5, lab, transform=axs[ri, p].transAxes, rotation=90, va="center", fontsize=10)
        fig.suptitle(f"{name} — ALL z-planes, reference=z{refp}. frame {t}/{NF-1}", fontsize=13)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(f"result/gif_{name}_allslices_V1.gif", frames, duration=0.16, loop=0)
    print(f"saved result/gif_{name}_allslices_V1.gif", flush=True)
print("DONE", flush=True)
