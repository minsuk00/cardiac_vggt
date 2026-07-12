"""gather05 on Volunteer2: 3-row GIF — real-time beat | faithful input (ref swept, others static) | recon."""
import numpy as np, glob, sys, gc, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.inference import load_rtfb_model_reference, forward
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import to_canonical_inplane, assign_canonical_z, D_CANON, INPUT_IMG_SIZE

dev = "cuda"; NF = 30; subj = "Volunteer2"; name = "gather05"
ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
cine = ad.load()
b, S, picks, ref_ctx = ad.build_batch_multiframe(dev, 1, 1, continuous_z=False)
vmin, vmax = ref_ctx["scale"]; inpl = ref_ctx["inplane"]; ref_slice = ref_ctx["slice_idx"]; cs = ref_ctx["cine_slice"]
slot_plane = np.array([p[0] for p in picks]); refp = int(np.floor(float(ref_ctx["z_canon"]) + 0.5))
slot_of_plane = [int(np.argmin(np.abs(slot_plane - p))) for p in range(D_CANON)]
has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]; has_slot[refp] = True
zmap = assign_canonical_z(ad.slice_positions_mm(), False)
inp_slice = [min(zmap, key=lambda t: abs(t[0]-p))[1] for p in range(D_CANON)]
fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))
def canon(im): return to_canonical_inplane(np.clip((im-vmin)/(vmax-vmin), 0, 1), inpl).numpy()
print(f"V2 {name}: ref z{refp}, S={S}", flush=True)

ck = glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0]
model = load_rtfb_model_reference(ck, refiner=False, device=dev)
RE = np.zeros((NF, D_CANON, 256, 256), np.float32)
for ti, f in enumerate(fr):
    up = F.interpolate(to_canonical_inplane(np.clip((cs[f]-vmin)/(vmax-vmin), 0, 1), inpl)[None, None],
                       size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE), mode="bilinear", align_corners=True)[0, 0].numpy()
    b["images"][:, 0] = torch.from_numpy(up).to(dev).repeat(3, 1, 1)
    RE[ti] = forward(model, b, want=("V",), device=dev)["V"]
del model; gc.collect(); torch.cuda.empty_cache()

BEAT = np.zeros((NF, D_CANON, 256, 256), np.float32); FAITH = np.zeros((NF, D_CANON, 256, 256), np.float32)
for p in range(D_CANON):
    if not has_slot[p]: continue
    for ti, f in enumerate(fr): BEAT[ti, p] = canon(cine[f, inp_slice[p]])
    if p == refp:
        for ti, f in enumerate(fr): FAITH[ti, p] = canon(cine[f, ref_slice])
    else:
        _, si, ff, _ = picks[slot_of_plane[p]]; FAITH[:, p] = canon(cine[ff, si])[None]
np.savez_compressed(f"result/{name}_allslices_V2.npz", recon=RE, inp=FAITH, slot_plane=slot_plane, refp=refp)

bvmax = np.percentile(BEAT, 99.5); ivmax = np.percentile(FAITH, 99.5); rvmax = np.percentile(RE, 99.5); blank = np.zeros((256, 256), np.float32)
rows = [("real-time\n(each slice beats)", BEAT, bvmax, True), ("input\n(model sees)", FAITH, ivmax, True), ("recon", RE, rvmax, False)]
frames = []
for t in range(NF):
    fig, axs = plt.subplots(3, D_CANON, figsize=(24, 6.6), dpi=95)
    for ri, (lab, arr, vm, bl) in enumerate(rows):
        for p in range(D_CANON):
            axs[ri, p].imshow(arr[t, p] if (has_slot[p] or not bl) else blank, cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
            if ri == 0: axs[ri, p].set_title((f"z{p}*REF" if p == refp else f"z{p}") + ("" if has_slot[p] else "\n(no input)"), fontsize=8)
            if p == 0: axs[ri, p].text(-0.42, 0.5, lab, transform=axs[ri, p].transAxes, rotation=90, va="center", fontsize=9)
    fig.suptitle(f"gather05 on Volunteer2, ref=z{refp} — real-time beat | model input | recon. frame {t}/{NF-1}", fontsize=12)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave("result/gif_gather05_3row_V2.gif", frames, duration=0.16, loop=0)
print("saved result/gif_gather05_3row_V2.gif\nDONE", flush=True)
