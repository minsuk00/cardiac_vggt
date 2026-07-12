"""(1) Verify subj0 is really in the VAL split. (2) Recompute the exact breathing displacement
applied (resp_disp_mm) to prove breathing was ON + quantify amplitude. (3) Re-render the 3 in-dist
GIFs from saved npz with breathing amplitude in the caption + non-clipped row labels."""
import os, sys, numpy as np, torch
sys.path.insert(0, "training"); sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29594")
    dist.init_process_group("gloo", rank=0, world_size=1)
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
for r, fn in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
              ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
    try: OmegaConf.register_new_resolver(r, fn)
    except Exception: pass

with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
    cfg = compose(config_name="mri_volume")
mri_ds = instantiate(cfg.data.val, _recursive_=False).dataset.base_dataset.datasets[0]
resp_cfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)

# ---- (1) VAL verification ----
print(f"split attr on dataset = {mri_ds.split!r}", flush=True)
subs = [str(s) for s in mri_ds.subjects]
print(f"n subjects = {len(subs)}; subj0 = {os.path.basename(subs[0])}", flush=True)
# cross-check against the [val] section of the split file
sf = mri_ds.split_file; valset = []; cur = None
for line in open(sf):
    line = line.strip()
    if line.startswith("[") and line.endswith("]"): cur = line[1:-1].lower()
    elif cur == "val" and line: valset.append(line)
b0 = os.path.basename(subs[0])
print(f"split_file={os.path.basename(sf)}; [val] has {len(valset)} entries; subj0 in [val]? "
      f"{any(b0 in v or v in b0 for v in valset)} (first val entries: {valset[:3]})", flush=True)

# ---- (2) breathing amplitude actually applied ----
data = mri_ds.get_data(seq_index=0, img_per_seq=mri_ds.num_slices)
def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
batch = {"images": st("images").permute(0,1,4,2,3).contiguous()/255.0,
         "scanner_coords": st("scanner_coords"), "z_indices": st("z_indices"), "t_indices": st("t_indices"),
         "phases": torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).unsqueeze(0),
         "timesteps": st("timesteps", np.int64), "slice_indices": st("slice_indices", np.float32),
         "seq_index": torch.tensor([[0]], dtype=torch.int64)}
batch = gpu_augment_batch(batch, None, "cpu", respiratory_cfg=resp_cfg, train=False)
disp = batch["resp_disp_mm"][0].cpu().numpy()   # (S,3) mm, canonical (z,y,x)
mag = np.linalg.norm(disp, axis=1)
print(f"\nresp.enable={resp_cfg.enable}  resp_disp_mm per slot (S={disp.shape[0]}):", flush=True)
print(f"  |disp| mm: mean={mag.mean():.1f} max={mag.max():.1f}  per-axis(z,y,x) max=({np.abs(disp[:,0]).max():.1f},{np.abs(disp[:,1]).max():.1f},{np.abs(disp[:,2]).max():.1f})", flush=True)
AMP = f"breathing ON: per-slot |disp| mean {mag.mean():.1f}mm, max {mag.max():.1f}mm"

# ---- (3) re-render GIFs from saved npz, fixed labels + amplitude caption ----
T_total = mri_ds.gt_grid_shape[0]
for name in ["control0", "gather05", "s20contz"]:
    d = dict(np.load(f"result/indist_{name}_S0.npz"))
    GT, RE, IN, ref_zmid = d["gt"], d["recon"], d["inp"], int(d["ref_zmid"])
    D = GT.shape[1]
    gvmax = np.percentile(GT, 99.5); rvmax = np.percentile(RE, 99.5); ivmax = np.percentile(IN, 99.5)
    rows = [("GT (real beat)", GT, gvmax), ("input (breathing)", IN, ivmax), ("recon", RE, rvmax)]
    ycen = [0.74, 0.45, 0.16]
    frames = []
    for t in range(T_total):
        fig, axs = plt.subplots(3, D, figsize=(2*D, 6.8), dpi=95)
        for ri, (lab, arr, vm) in enumerate(rows):
            for p in range(D):
                axs[ri, p].imshow(arr[t, p], cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
                if ri == 0: axs[ri, p].set_title(f"z{p}" + ("*REF" if p == ref_zmid else ""), fontsize=8)
        for yc, (lab, _, _) in zip(ycen, rows):
            fig.text(0.012, yc, lab, rotation=90, va="center", ha="center", fontsize=11)
        fig.suptitle(f"{name}  |  IN-DIST CMRxRecon VAL subj0  |  {AMP}  |  cardiac phase {t}/{T_total-1}", fontsize=12)
        fig.subplots_adjust(left=0.035, right=0.997, top=0.90, bottom=0.01, wspace=0.04, hspace=0.10)
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(f"result/gif_indist_{name}_3row_S0.gif", frames, duration=0.18, loop=0)
    print(f"saved result/gif_indist_{name}_3row_S0.gif", flush=True)
print("DONE", flush=True)
