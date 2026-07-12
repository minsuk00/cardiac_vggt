"""control0 V2: 3-row GIF — real-time BEATING (each slice's true motion) | faithful INPUT (model
sees: ref swept, others static) | RECON. CPU only (beating row from cine; input/recon from npz)."""
import numpy as np, sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import assign_canonical_z, percentile_scale, to_canonical_inplane, D_CANON

NF = 30; subj = "Volunteer2"
d = dict(np.load("result/control0_allslices_V2.npz"))
RE, FAITH, slot_plane, refp = d["recon"], d["inp"], d["slot_plane"], int(d["refp"])
ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
cine = ad.load(); vmin, vmax = percentile_scale(cine); inpl = ad.inplane_mm()
zmap = assign_canonical_z(ad.slice_positions_mm(), False)
inp_slice = [min(zmap, key=lambda t: abs(t[0]-p))[1] for p in range(D_CANON)]
has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]; has_slot[refp] = True
fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))
def canon(im): return to_canonical_inplane(np.clip((im-vmin)/(vmax-vmin), 0, 1), inpl).numpy()
# real-time beating: every plane's slice animated over the sweep
BEAT = np.zeros((NF, D_CANON, 256, 256), np.float32)
for p in range(D_CANON):
    if has_slot[p]:
        for ti, f in enumerate(fr): BEAT[ti, p] = canon(cine[f, inp_slice[p]])

bvmax = np.percentile(BEAT, 99.5); ivmax = np.percentile(FAITH, 99.5); rvmax = np.percentile(RE, 99.5); blank = np.zeros((256, 256), np.float32)
rows = [("real-time\n(each slice beats)", BEAT, bvmax, True),
        ("input\n(model sees)", FAITH, ivmax, True),
        ("recon", RE, rvmax, False)]
frames = []
for t in range(NF):
    fig, axs = plt.subplots(3, D_CANON, figsize=(24, 6.6), dpi=95)
    for ri, (lab, arr, vm, bl) in enumerate(rows):
        for p in range(D_CANON):
            axs[ri, p].imshow(arr[t, p] if (has_slot[p] or not bl) else blank, cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
            if ri == 0: axs[ri, p].set_title((f"z{p}*REF" if p == refp else f"z{p}") + ("" if has_slot[p] else "\n(no input)"), fontsize=8)
            if p == 0: axs[ri, p].text(-0.42, 0.5, lab, transform=axs[ri, p].transAxes, rotation=90, va="center", fontsize=9)
    fig.suptitle(f"control0 on Volunteer2, ref=z{refp} — real-time beat | model input (ref swept, others static) | recon. frame {t}/{NF-1}", fontsize=12)
    fig.tight_layout(); fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave("result/gif_control0_3row_V2.gif", frames, duration=0.16, loop=0)
print("saved result/gif_control0_3row_V2.gif\nDONE", flush=True)
