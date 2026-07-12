"""Fix the in-dist input row: blank planes with no real input slot (z0/z11 stand-ins).
Recompute has_slot from subj0's slice_indices; re-render the 3 GIFs from saved npz (CPU)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "training"); sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29597")
    dist.init_process_group("gloo", rank=0, world_size=1)
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
for r, fn in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
              ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
    try: OmegaConf.register_new_resolver(r, fn)
    except Exception: pass
with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
    cfg = compose(config_name="mri_volume")
mri_ds = instantiate(cfg.data.val, _recursive_=False).dataset.base_dataset.datasets[0]
data = mri_ds.get_data(seq_index=0, img_per_seq=mri_ds.num_slices)
slice_z = np.asarray(data["slice_indices"]).astype(np.float32)   # (S,) per-slot canonical z
D = mri_ds.gt_grid_shape[0]   # z-plane count (12), NOT H
has_slot = [bool(np.min(np.abs(slice_z - p)) < 0.5) for p in range(D)]
print(f"per-slot z (rounded): {sorted(set(np.round(slice_z).astype(int).tolist()))}", flush=True)
print(f"planes WITH real input: {[p for p in range(D) if has_slot[p]]}  (blanked in input row: {[p for p in range(D) if not has_slot[p]]})", flush=True)

AMP = "breathing ON: per-slot |disp| mean 3.9mm, max 8.8mm"
for name in ["control0", "gather05", "s20contz"]:
    d = dict(np.load(f"result/indist_{name}_S0.npz"))
    GT, RE, IN, rz = d["gt"], d["recon"], d["inp"], int(d["ref_zmid"])
    T_total = GT.shape[0]
    gv, rv, iv = [np.percentile(x, 99.5) for x in (GT, RE, IN)]; blank = np.zeros((256, 256), np.float32)
    rows = [("GT (real beat)", GT, gv, False), ("input (breathing)", IN, iv, True), ("recon", RE, rv, False)]
    ycen = [0.74, 0.45, 0.16]
    frames = []
    for t in range(T_total):
        fig, axs = plt.subplots(3, D, figsize=(2*D, 6.8), dpi=95)
        for ri, (lab, arr, vm, bl) in enumerate(rows):
            for p in range(D):
                img = arr[t, p] if (has_slot[p] or not bl) else blank
                axs[ri, p].imshow(img, cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
                if ri == 0: axs[ri, p].set_title(f"z{p}" + ("*REF" if p == rz else ("" if has_slot[p] else "\n(no input)")), fontsize=8)
        for yc, (lab, _, _, _) in zip(ycen, rows):
            fig.text(0.012, yc, lab, rotation=90, va="center", ha="center", fontsize=11)
        fig.suptitle(f"{name}  |  IN-DIST CMRxRecon VAL subj0 (Train_P053)  |  {AMP}  |  cardiac phase {t}/{T_total-1}", fontsize=12)
        fig.subplots_adjust(left=0.035, right=0.997, top=0.90, bottom=0.01, wspace=0.04, hspace=0.10)
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(f"result/gif_indist_{name}_3row_S0.gif", frames, duration=0.18, loop=0)
    print(f"saved result/gif_indist_{name}_3row_S0.gif", flush=True)
print("DONE", flush=True)
