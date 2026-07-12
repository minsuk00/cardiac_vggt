"""Faithful input+recon GIF, all 3 models. Input row = EXACTLY what the model gets:
reference plane SWEPT (30 frames), every other plane STATIC at its one fixed frame.
CPU only: rebuilds the deterministic batch to recover each slot's fixed frame; recon from saved npz."""
import numpy as np, sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import to_canonical_inplane, D_CANON

NF = 30; subj = "Volunteer1"
specs = [("control0", False), ("gather05", False), ("s20contz", True)]
for name, cz in specs:
    d = dict(np.load(f"result/{name}_allslices_V1.npz"))
    RE = d["recon"]
    ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
    cine = ad.load()
    batch, S, picks, ref_ctx = ad.build_batch_multiframe("cpu", 1, 1, continuous_z=cz)  # deterministic
    vmin, vmax = ref_ctx["scale"]; inpl = ref_ctx["inplane"]; ref_slice = ref_ctx["slice_idx"]
    refp = int(np.floor(float(ref_ctx["z_canon"]) + 0.5))          # round-half-up, matches assign_canonical_z
    slot_plane = np.array([p[0] for p in picks])
    slot_of_plane = [int(np.argmin(np.abs(slot_plane - p))) for p in range(D_CANON)]
    has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]
    has_slot[refp] = True                                          # reference plane always has (swept) input
    fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))
    def canon(im): return to_canonical_inplane(np.clip((im-vmin)/(vmax-vmin), 0, 1), inpl).numpy()
    # faithful input: (NF,12,256,256)
    IN = np.zeros((NF, D_CANON, 256, 256), np.float32)
    for p in range(D_CANON):
        if not has_slot[p]:
            continue
        if p == refp:                                   # swept reference plane
            for ti, f in enumerate(fr): IN[ti, p] = canon(cine[f, ref_slice])
        else:                                           # static: this slot's fixed frame
            _, si, f, _ = picks[slot_of_plane[p]]
            IN[:, p] = canon(cine[f, si])[None]
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
        fig.suptitle(f"{name}, ref=z{refp} — FAITHFUL input (ref swept, others static) vs recon. frame {t}/{NF-1}", fontsize=12)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(f"result/gif_{name}_faithful_inputrecon_V1.gif", frames, duration=0.16, loop=0)
    print(f"saved result/gif_{name}_faithful_inputrecon_V1.gif  (ref z{refp}, S={S})", flush=True)
print("DONE", flush=True)
