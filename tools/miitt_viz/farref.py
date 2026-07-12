"""gather05 with a FAR reference plane (z1, basal) instead of mid (z5). All slices.
Custom batch builder = _build_batch_multiframe_core but reference = slot nearest target plane."""
import numpy as np, glob, sys, gc, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.inference import load_rtfb_model_reference, forward
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import (assign_canonical_z, percentile_scale, to_canonical_inplane,
                                MM_PER_NORM, D_CANON, INPUT_IMG_SIZE)

dev = "cuda"; NF = 30; FPS = 6; subj = "Volunteer1"; TARGET_REF_PLANE = 1   # far basal
ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
cine = ad.load(); scale = percentile_scale(cine); inpl = ad.inplane_mm()
zmap = assign_canonical_z(ad.slice_positions_mm(), continuous_z=False)
vmin, vmax = scale
def to256(a): idx = np.linspace(0, a.shape[0]-1, 256).astype(int); return a[np.ix_(idx, idx)]

def build_far(target_plane):
    rng = np.random.default_rng(0)
    py, px = np.meshgrid(np.arange(INPUT_IMG_SIZE), np.arange(INPUT_IMG_SIZE), indexing="ij")
    xn = (px/(INPUT_IMG_SIZE-1)*2-1).astype(np.float32); yn = (py/(INPUT_IMG_SIZE-1)*2-1).astype(np.float32)
    def extract(si, f):
        return F.interpolate(to_canonical_inplane(np.clip((cine[f, si]-vmin)/(vmax-vmin), 0, 1), inpl)[None, None],
                             size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE), mode="bilinear", align_corners=True)[0, 0].numpy()
    n = cine.shape[0]
    def burst(k): k = min(k, n); s0 = int(rng.integers(max(1, n-k+1))); return list(range(s0, s0+k))
    ref_i = int(np.argmin([abs(z-target_plane) for z, _ in zmap]))
    ref_z, ref_slice = zmap[ref_i]; rest = zmap[:ref_i] + zmap[ref_i+1:]
    rfi = burst(1); slots = [(ref_z, ref_slice, rfi[0])] + [(ref_z, ref_slice, f) for f in rfi]
    for zc, si in rest: slots += [(zc, si, f) for f in burst(1)]
    imgs, coords, zidx = [], [], []
    for zc, si, f in slots:
        imgs.append(np.repeat(extract(si, f)[None], 3, 0)); zv = zc/max(1, D_CANON-1)*2-1
        coords.append(np.stack([xn, yn, np.full_like(xn, zv)], -1)); zidx.append([zv])
    batch = {"images": torch.from_numpy(np.stack(imgs)).float()[None].to(dev),
             "scanner_coords": torch.from_numpy(np.stack(coords)).float()[None].to(dev),
             "z_indices": torch.tensor(zidx, dtype=torch.float32)[None].to(dev)}
    return batch, np.stack(coords), ref_z, ref_slice, cine[:, ref_slice], [s[0] for s in slots]

batch, coords, ref_z, ref_slice, cine_ref, slot_planes = build_far(TARGET_REF_PLANE)
slot_planes = np.array(slot_planes); refp = int(round(ref_z))
print(f"far reference: plane z{refp} (native slice {ref_slice}); slot planes {np.round(slot_planes,1)}", flush=True)
slot_of_plane = [int(np.argmin(np.abs(slot_planes - p))) for p in range(D_CANON)]
has_slot = [min(abs(slot_planes - p)) < 0.5 for p in range(D_CANON)]
inp_sl = [min(zmap, key=lambda t: abs(t[0]-p))[1] for p in range(D_CANON)]
fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))
def canon(img): return to_canonical_inplane(np.clip((img-vmin)/(vmax-vmin), 0, 1), inpl).numpy()
IN = np.stack([[canon(cine[f, si]) for f in fr] for si in inp_sl]).transpose(1, 0, 2, 3)

ck = glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0]
print("loading gather05 ...", flush=True); model = load_rtfb_model_reference(ck, refiner=False, device=dev)
RE = np.zeros((NF, D_CANON, 256, 256), np.float32); CO = np.zeros((NF, D_CANON, 256, 256), np.float32)
DV = np.zeros((NF, D_CANON, 256, 256, 3), np.float32)
for ti, f in enumerate(fr):
    up = F.interpolate(to_canonical_inplane(np.clip((cine_ref[f]-vmin)/(vmax-vmin), 0, 1), inpl)[None, None],
                       size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE), mode="bilinear", align_corners=True)[0, 0].numpy()
    batch["images"][:, 0] = torch.from_numpy(up).to(dev).repeat(3, 1, 1)
    r = forward(model, batch, want=("V", "world_points", "coverage"), device=dev)
    RE[ti] = r["V"]; CO[ti] = r["coverage"]
    for p in range(D_CANON):
        d = (r["world_points"][slot_of_plane[p]] - coords[slot_of_plane[p]]) * np.array(MM_PER_NORM)[None, None, :]
        DV[ti, p] = np.stack([to256(d[..., k]) for k in range(3)], -1)
del model; gc.collect(); torch.cuda.empty_cache()
np.savez_compressed("result/gather05_farref_z1_V1.npz", recon=RE, cov=CO, dvf=DV, inp=IN, slot_planes=slot_planes, refp=refp)
print("saved npz", flush=True)

vlx, vly, vlz = [max(1.0, np.percentile(np.abs(DV[..., k]), 99)) for k in range(3)]
covmax = float(CO.max()); ivmax = np.percentile(IN, 99.5); rvmax = np.percentile(RE, 99.5)
blank = np.zeros((256, 256), np.float32)
rows = [("input", IN, "gray", 0, ivmax, True), ("recon", RE, "gray", 0, rvmax, False),
        (f"Dx ±{vlx:.0f}", DV[..., 0], "bwr", -vlx, vlx, True), (f"Dy ±{vly:.0f}", DV[..., 1], "bwr", -vly, vly, True),
        (f"Dz ±{vlz:.0f}mm", DV[..., 2], "bwr", -vlz, vlz, True), (f"cov 0-{covmax:.0f}", CO, "viridis", 0, covmax, False)]
frames = []
for t in range(NF):
    fig, axs = plt.subplots(6, D_CANON, figsize=(24, 12), dpi=90)
    for ri, (lab, arr, cm, lo, hi, bl) in enumerate(rows):
        for p in range(D_CANON):
            img = arr[t, p] if (has_slot[p] or not bl) else blank
            axs[ri, p].imshow(img, cmap=cm, vmin=lo, vmax=hi); axs[ri, p].axis("off")
            if ri == 0: axs[ri, p].set_title((f"z{p}*REF" if p == refp else f"z{p}") + ("" if has_slot[p] else "\n(no input)"), fontsize=9)
            if p == 0: axs[ri, p].text(-0.35, 0.5, lab, transform=axs[ri, p].transAxes, rotation=90, va="center", fontsize=10)
    fig.suptitle(f"gather05 (snapped), FAR reference = z{refp} (basal). frame {t}/{NF-1}", fontsize=13)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimwrite(f"result/mp4_gather05_farref_z{refp}_V1.mp4", frames, fps=FPS, codec="libx264",
                 macro_block_size=16, ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p"])
print(f"saved result/mp4_gather05_farref_z{refp}_V1.mp4  covmax={covmax:.1f}\nDONE", flush=True)
