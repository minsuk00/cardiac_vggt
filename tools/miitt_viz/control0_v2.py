"""control0 on Volunteer2: compute all-slice recon + faithful input+recon GIF (ref swept, others static)."""
import numpy as np, glob, sys, gc, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.inference import load_rtfb_model_reference, forward
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import to_canonical_inplane, MM_PER_NORM, D_CANON, INPUT_IMG_SIZE

dev = "cuda"; NF = 30; subj = "Volunteer2"; name = "control0"
ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
cine = ad.load()
b, S, picks, ref_ctx = ad.build_batch_multiframe(dev, 1, 1, continuous_z=False)
vmin, vmax = ref_ctx["scale"]; inpl = ref_ctx["inplane"]; ref_slice = ref_ctx["slice_idx"]; cs = ref_ctx["cine_slice"]
sc = b["scanner_coords"][0].cpu().numpy()
slot_plane = np.array([p[0] for p in picks]); refp = int(np.floor(float(ref_ctx["z_canon"]) + 0.5))
slot_of_plane = [int(np.argmin(np.abs(slot_plane - p))) for p in range(D_CANON)]
has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]; has_slot[refp] = True
fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))
def canon(im): return to_canonical_inplane(np.clip((im-vmin)/(vmax-vmin), 0, 1), inpl).numpy()
print(f"V2 {name}: ref z{refp}, S={S}, slot planes {np.round(slot_plane,1)}", flush=True)

ck = glob.glob("scratch/logs/216539845_*ftctrl_gather0*1frame*/ckpts/checkpoint_last.pt")[0]
model = load_rtfb_model_reference(ck, refiner=False, device=dev)
RE = np.zeros((NF, D_CANON, 256, 256), np.float32)
for ti, f in enumerate(fr):
    up = F.interpolate(to_canonical_inplane(np.clip((cs[f]-vmin)/(vmax-vmin), 0, 1), inpl)[None, None],
                       size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE), mode="bilinear", align_corners=True)[0, 0].numpy()
    b["images"][:, 0] = torch.from_numpy(up).to(dev).repeat(3, 1, 1)
    RE[ti] = forward(model, b, want=("V",), device=dev)["V"]
del model; gc.collect(); torch.cuda.empty_cache()

# faithful input: ref plane swept, others static
IN = np.zeros((NF, D_CANON, 256, 256), np.float32)
for p in range(D_CANON):
    if not has_slot[p]: continue
    if p == refp:
        for ti, f in enumerate(fr): IN[ti, p] = canon(cine[f, ref_slice])
    else:
        _, si, f, _ = picks[slot_of_plane[p]]; IN[:, p] = canon(cine[f, si])[None]
np.savez_compressed(f"result/{name}_allslices_V2.npz", recon=RE, inp=IN, slot_plane=slot_plane, refp=refp)

ivmax = np.percentile(IN, 99.5); rvmax = np.percentile(RE, 99.5); blank = np.zeros((256, 256), np.float32)
frames = []
for t in range(NF):
    fig, axs = plt.subplots(2, D_CANON, figsize=(24, 4.4), dpi=95)
    for p in range(D_CANON):
        axs[0, p].imshow(IN[t, p] if has_slot[p] else blank, cmap="gray", vmin=0, vmax=ivmax); axs[0, p].axis("off")
        axs[0, p].set_title((f"z{p}*REF(swept)" if p == refp else f"z{p}") + ("" if has_slot[p] else "\n(no input)"), fontsize=8)
        axs[1, p].imshow(RE[t, p], cmap="gray", vmin=0, vmax=rvmax); axs[1, p].axis("off")
        if p == 0:
            axs[0, p].text(-0.4, 0.5, "input\n(what model sees)", transform=axs[0, p].transAxes, rotation=90, va="center", fontsize=9)
            axs[1, p].text(-0.4, 0.5, "recon", transform=axs[1, p].transAxes, rotation=90, va="center", fontsize=10)
    fig.suptitle(f"{name} on Volunteer2, ref=z{refp} — faithful input (ref swept, others static) vs recon. frame {t}/{NF-1}", fontsize=12)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave("result/gif_control0_faithful_inputrecon_V2.gif", frames, duration=0.16, loop=0)
print("saved result/gif_control0_faithful_inputrecon_V2.gif\nDONE", flush=True)
