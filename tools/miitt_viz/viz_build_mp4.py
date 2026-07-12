"""Rebuild per-model panels as MP4 (H.264, no palette quantization) from saved npzs.
No GPU / no recompute — reads result/viz_{model}_V1.npz."""
import numpy as np, sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import to_canonical_inplane, assign_canonical_z

NF = 30; FPS = 6; subj = "Volunteer1"
names = ["control0", "gather05", "s20contz"]
D = {n: dict(np.load(f"result/viz_{n}_V1.npz")) for n in names}
ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
cine = ad.load(); pos = ad.slice_positions_mm()
_, _, _, rc = ad.build_batch_multiframe("cpu", 1, 1, continuous_z=False)
vmin, vmax = rc["scale"]; inpl = rc["inplane"]
fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))
def canon(img): return to_canonical_inplane(np.clip((img-vmin)/(vmax-vmin), 0, 1), inpl).numpy()
def inp_slices(zmap, planes): return [min(zmap, key=lambda t: abs(t[0]-p))[1] for p in planes]

INP = {}
for n in names:
    zmap = assign_canonical_z(pos, bool(D[n]["cz"]))
    INP[n] = np.stack([[canon(cine[f, si]) for f in fr] for si in inp_slices(zmap, list(D[n]["planes"]))])

alld = np.concatenate([D[n]["dvf"].reshape(-1, 3) for n in names], 0)
vlx, vly, vlz = [max(1.0, np.percentile(np.abs(alld[:, k]), 99)) for k in range(3)]
covmax = max(np.percentile(D[n]["cov"], 99.5) for n in names)
rl = ["mid-1", "mid", "mid+1"]

def write_mp4(path, frames):
    imageio.mimwrite(path, frames, fps=FPS, codec="libx264", quality=None,
                     macro_block_size=16, ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p"])

for n in names:
    recon = D[n]["recon"].transpose(1, 0, 2, 3); dvf = D[n]["dvf"]; cov = D[n]["cov"]
    rvmax = np.percentile(recon, 99.5); ivmax = np.percentile(INP[n], 99.5)
    colspec = [
        ("input", lambda r, t: (INP[n][r, t], "gray", 0, ivmax)),
        ("recon", lambda r, t: (recon[r, t], "gray", 0, rvmax)),
        (f"Dx (±{vlx:.0f}mm)", lambda r, t: (dvf[t, r, :, :, 0], "bwr", -vlx, vlx)),
        (f"Dy (±{vly:.0f}mm)", lambda r, t: (dvf[t, r, :, :, 1], "bwr", -vly, vly)),
        (f"Dz (±{vlz:.0f}mm)", lambda r, t: (dvf[t, r, :, :, 2], "bwr", -vlz, vlz)),
        (f"coverage (0-{covmax:.1f})", lambda r, t: (cov[t, r], "viridis", 0, covmax)),
    ]
    frames = []
    for t in range(NF):
        fig, axs = plt.subplots(3, 6, figsize=(18, 9.5), dpi=110)
        for c, (title, fn) in enumerate(colspec):
            for r in range(3):
                img, cm, lo, hi = fn(r, t)
                axs[r, c].imshow(img, cmap=cm, vmin=lo, vmax=hi); axs[r, c].axis("off")
                if r == 0: axs[r, c].set_title(title, fontsize=11)
                if c == 0: axs[r, c].text(-0.15, 0.5, rl[r], transform=axs[r, c].transAxes, rotation=90, va="center", fontsize=11)
        fig.suptitle(f"{n} (as-trained z) — input | recon | DVF Dx/Dy/Dz | coverage.  frame {t}/{NF-1}", fontsize=13)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    write_mp4(f"result/mp4_model_{n}_V1.mp4", frames)
    print(f"saved result/mp4_model_{n}_V1.mp4", flush=True)
print(f"vlims dx/dy/dz={vlx:.1f}/{vly:.1f}/{vlz:.1f} covmax={covmax:.2f}\nMP4 DONE", flush=True)
