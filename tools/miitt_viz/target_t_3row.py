"""Legacy target_t model (t59w6nqy = 218747856_..._z_no_t): static inputs, animate by sweeping
target_t (phase_sweep). 3-row GIF for V1 & V2: real-time beat | STATIC input | recon(target_t)."""
import numpy as np, sys, gc, torch
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.inference import phase_sweep
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import to_canonical_inplane, assign_canonical_z, percentile_scale, D_CANON, GRID_SHAPE
from vggt.models.vggt import VGGT
from vggt.models.aggregator import ZIndexEmbedder, TIndexEmbedder

dev = "cuda"; NF = 24
CKPT = "scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
print("loading target_t model (num_freqs=6 embedders) ...", flush=True)
model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False,
             use_target_t_pose_embedding=True, train_on_residual_dvf=True,
             enable_refiner=False, refiner_use_coverage=False, grid_shape=GRID_SHAPE)
model.aggregator.z_embedder = ZIndexEmbedder(embed_dim=1024, num_freqs=6)         # match ckpt (13-dim)
model.aggregator.target_t_embedder = TIndexEmbedder(embed_dim=1024, num_freqs=6)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ck["model"] if "model" in ck else ck
missing, unexpected = model.load_state_dict(state, strict=False)
crit = [k for k in missing if any(s in k for s in ("aggregator", "point_head", "z_embedder", "target_t_embedder"))]
assert not crit, f"missing critical: {crit[:5]}"
print(f"  loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
model = model.to(dev).eval()

for subj in ["Volunteer1", "Volunteer2"]:
    Vtag = {"Volunteer1": "V1", "Volunteer2": "V2"}[subj]
    ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
    cine = ad.load(); vmin, vmax = percentile_scale(cine); inpl = ad.inplane_mm()
    batch, S, picks = ad.build_batch(np.random.default_rng(0), dev, continuous_z=False)   # static 1-frame/slice
    slot_plane = np.array([p[0] for p in picks])
    slot_of_plane = [int(np.argmin(np.abs(slot_plane - p))) for p in range(D_CANON)]
    has_slot = [min(abs(slot_plane - p)) < 0.5 for p in range(D_CANON)]
    zmap = assign_canonical_z(ad.slice_positions_mm(), False)
    inp_slice = [min(zmap, key=lambda t: abs(t[0]-p))[1] for p in range(D_CANON)]
    def canon(im): return to_canonical_inplane(np.clip((im-vmin)/(vmax-vmin), 0, 1), inpl).numpy()

    vols, _ = phase_sweep(model, batch, n_phases=NF, device=dev)   # target_t sweep
    RE = np.stack(vols)                                            # (NF,12,256,256)
    fr = list(np.linspace(0, cine.shape[0]-1, NF).astype(int))
    BEAT = np.zeros((NF, D_CANON, 256, 256), np.float32); STAT = np.zeros((NF, D_CANON, 256, 256), np.float32)
    for p in range(D_CANON):
        if not has_slot[p]: continue
        for ti, f in enumerate(fr): BEAT[ti, p] = canon(cine[f, inp_slice[p]])
        _, si, ff, _ = picks[slot_of_plane[p]]; STAT[:, p] = canon(cine[ff, si])[None]   # static, all frames identical
    print(f"{Vtag}: S={S}, recon motion/plane={np.round(RE.std(0).mean((1,2)),4)}", flush=True)

    bvmax = np.percentile(BEAT, 99.5); ivmax = np.percentile(STAT, 99.5); rvmax = np.percentile(RE, 99.5); blank = np.zeros((256, 256), np.float32)
    rows = [("real-time\n(each slice beats)", BEAT, bvmax, True), ("input\n(STATIC, 1 frame/slice)", STAT, ivmax, True), ("recon\n(target_t sweep)", RE, rvmax, False)]
    frames = []
    for t in range(NF):
        fig, axs = plt.subplots(3, D_CANON, figsize=(24, 6.6), dpi=95)
        for ri, (lab, arr, vm, bl) in enumerate(rows):
            for p in range(D_CANON):
                axs[ri, p].imshow(arr[t, p] if (has_slot[p] or not bl) else blank, cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
                if ri == 0: axs[ri, p].set_title(f"z{p}" + ("" if has_slot[p] else "\n(no input)"), fontsize=8)
                if p == 0: axs[ri, p].text(-0.42, 0.5, lab, transform=axs[ri, p].transAxes, rotation=90, va="center", fontsize=8)
        fig.suptitle(f"target_t model (no reference) on {subj} — real beat | STATIC input | recon via target_t sweep. frame {t}/{NF-1}", fontsize=12)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(f"result/gif_targett_3row_{Vtag}.gif", frames, duration=0.16, loop=0)
    np.savez_compressed(f"result/targett_allslices_{Vtag}.npz", recon=RE, stat=STAT, beat=BEAT, slot_plane=slot_plane)
    print(f"saved result/gif_targett_3row_{Vtag}.gif", flush=True)
print("DONE", flush=True)
