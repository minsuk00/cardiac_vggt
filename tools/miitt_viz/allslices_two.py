"""control0 (snapped) + s20contz (continuous), ALL 12 z-planes, mid reference, 6-row MP4.
Blanks input/DVF where no acquired slice; true coverage range."""
import numpy as np, glob, sys, gc, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.inference import load_rtfb_model_reference, forward
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import to_canonical_inplane, assign_canonical_z, MM_PER_NORM, D_CANON, INPUT_IMG_SIZE

dev = "cuda"; NF = 30; FPS = 6; subj = "Volunteer1"
ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
cine = ad.load(); pos = ad.slice_positions_mm()
def to256(a): idx = np.linspace(0, a.shape[0]-1, 256).astype(int); return a[np.ix_(idx, idx)]
fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))

specs = [("control0", "216539845_*ftctrl_gather0*1frame*", False),
         ("s20contz", "216949414_*s20contz*", True)]
for name, pat, cz in specs:
    b, S, _, rcx = ad.build_batch_multiframe(dev, 1, 1, continuous_z=cz)
    vmin, vmax = rcx["scale"]; inpl = rcx["inplane"]; cs = rcx["cine_slice"]
    sc = b["scanner_coords"][0].cpu().numpy()
    slot_plane = (sc[:, 0, 0, 2] + 1) / 2 * (D_CANON - 1); refp = int(round(float(slot_plane[0])))
    slot_of_plane = [int(np.argmin(np.abs(slot_plane - p))) for p in range(D_CANON)]
    has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]
    zmap = assign_canonical_z(pos, cz)
    inp_sl = [min(zmap, key=lambda t: abs(t[0]-p))[1] for p in range(D_CANON)]
    def canon(img): return to_canonical_inplane(np.clip((img-vmin)/(vmax-vmin), 0, 1), inpl).numpy()
    IN = np.stack([[canon(cine[f, si]) for f in fr] for si in inp_sl]).transpose(1, 0, 2, 3)
    ck = glob.glob(f"scratch/logs/{pat}/ckpts/checkpoint_last.pt")[0]
    print(f"loading {name} (ref z{refp}) ...", flush=True); model = load_rtfb_model_reference(ck, refiner=False, device=dev)
    RE = np.zeros((NF, D_CANON, 256, 256), np.float32); CO = np.zeros((NF, D_CANON, 256, 256), np.float32)
    DV = np.zeros((NF, D_CANON, 256, 256, 3), np.float32)
    for ti, f in enumerate(fr):
        up = F.interpolate(to_canonical_inplane(np.clip((cs[f]-vmin)/(vmax-vmin), 0, 1), inpl)[None, None],
                           size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE), mode="bilinear", align_corners=True)[0, 0].numpy()
        b["images"][:, 0] = torch.from_numpy(up).to(dev).repeat(3, 1, 1)
        r = forward(model, b, want=("V", "world_points", "coverage"), device=dev)
        RE[ti] = r["V"]; CO[ti] = r["coverage"]
        for p in range(D_CANON):
            d = (r["world_points"][slot_of_plane[p]] - sc[slot_of_plane[p]]) * np.array(MM_PER_NORM)[None, None, :]
            DV[ti, p] = np.stack([to256(d[..., k]) for k in range(3)], -1)
    del model; gc.collect(); torch.cuda.empty_cache()
    np.savez_compressed(f"result/{name}_allslices_V1.npz", recon=RE, cov=CO, dvf=DV, inp=IN, slot_plane=slot_plane, refp=refp)

    vlx, vly, vlz = [max(1.0, np.percentile(np.abs(DV[..., k]), 99)) for k in range(3)]
    covmax = float(CO.max()); ivmax = np.percentile(IN, 99.5); rvmax = np.percentile(RE, 99.5); blank = np.zeros((256, 256), np.float32)
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
        fig.suptitle(f"{name} — ALL z-planes, reference=z{refp}. rows input/recon/Dx/Dy/Dz/coverage. frame {t}/{NF-1}", fontsize=13)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimwrite(f"result/mp4_{name}_allslices_V1.mp4", frames, fps=FPS, codec="libx264",
                     macro_block_size=16, ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p"])
    print(f"saved result/mp4_{name}_allslices_V1.mp4  ref z{refp} covmax={covmax:.1f} dz±{vlz:.0f}", flush=True)
print("DONE", flush=True)
