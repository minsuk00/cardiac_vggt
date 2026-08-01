"""In-distribution CMRxRecon val, breathing ON. 3-row GIF per model (control0/gather05/s20contz):
GT phases (real beat) | scattered breathing-corrupted INPUT (ref plane swept, others static) | recon(V_canon).
Animate over cardiac phase t. Replicates trainer._log_cardiac_cycle_filmstrip standalone."""
import os, sys, glob, gc
sys.path.insert(0, "training"); sys.path.insert(0, ".")
import numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from loss import compute_volume_intensity_loss
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from inference.inference import load_rtfb_model_reference

dev = "cuda"
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29593")
    dist.init_process_group("gloo", rank=0, world_size=1)
for r, fn in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
              ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
    try: OmegaConf.register_new_resolver(r, fn)
    except Exception: pass

with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
    cfg = compose(config_name="default")
val_wrap = instantiate(cfg.data.val, _recursive_=False)
mri_ds = val_wrap.dataset.base_dataset.datasets[0]
resp_cfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)
T_total = mri_ds.gt_grid_shape[0]; num_slices = mri_ds.num_slices
print(f"val ds ok: subjects={len(mri_ds.subjects)} num_slices={num_slices} T={T_total} "
      f"reference_slot={mri_ds.reference_slot} resp.enable={resp_cfg.enable}", flush=True)

CKPTS = {"control0": "216539845_*ftctrl_gather0*1frame*", "gather05": "216539845_*ftgather05*1frame*",
         "s20contz": "216949414_*s20contz*"}
SUBJ = 0

def run_model(name, pat):
    ck = glob.glob(f"scratch/logs/{pat}/ckpts/checkpoint_last.pt")[0]
    model = load_rtfb_model_reference(ck, refiner=False, device=dev)
    data = mri_ds.get_data(seq_index=SUBJ, img_per_seq=num_slices)
    def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(dev)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    S = imgs.shape[1]
    phases_bundle = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(dev)  # (T,D,H,W)
    batch = {"images": imgs, "scanner_coords": st("scanner_coords"), "z_indices": st("z_indices"),
             "t_indices": st("t_indices"), "phases": phases_bundle.unsqueeze(0),
             "timesteps": st("timesteps", np.int64), "slice_indices": st("slice_indices", np.float32),
             "seq_index": torch.tensor([[SUBJ]], dtype=torch.int64, device=dev)}
    bb = np.asarray(data["anatomy_bbox"]).astype(np.int64); ref_zmid = (int(bb[0]) + int(bb[1])) // 2
    D = phases_bundle.shape[1]; slice_z = batch["slice_indices"][0].cpu().numpy()  # (S,)
    slot_of_plane = [int(np.argmin(np.abs(slice_z - p))) for p in range(D)]
    grid_shape = tuple(mri_ds.gt_grid_shape); hw = imgs.shape[-1]
    GT = np.zeros((T_total, D, 256, 256), np.float32); RE = np.zeros_like(GT); IN = np.zeros_like(GT)
    for t in range(T_total):
        t_norm = (t / max(1, T_total)) * 2.0 - 1.0
        batch["target_t_indices"] = torch.full((1, S, 1), t_norm, dtype=torch.float32, device=dev)
        batch["timesteps"][:, 0] = t
        batch = gpu_augment_batch(batch, None, dev, respiratory_cfg=resp_cfg, train=False)  # breathing re-extract
        batch["gt_target_volume"] = phases_bundle[t].unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
            out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()},
                                                batch, grid_shape=grid_shape, tv_weight=0.0)
        RE[t] = out["V_canon"][0].float().cpu().numpy(); GT[t] = out["V_gt"][0].float().cpu().numpy()
        img256 = F.interpolate(batch["images"][0, :, 0][:, None], size=(256, 256), mode="bilinear", align_corners=True)[:, 0].cpu().numpy()
        for p in range(D): IN[t, p] = img256[slot_of_plane[p]]
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"[{name}] S={S} ref_zmid={ref_zmid} recon motion/plane={np.round(RE.std(0).mean((1,2)),4)}", flush=True)
    np.savez_compressed(f"result/indist_{name}_S0.npz", gt=GT, recon=RE, inp=IN, ref_zmid=ref_zmid)
    return GT, RE, IN, ref_zmid, D

def render(name, GT, RE, IN, ref_zmid, D):
    gvmax = np.percentile(GT, 99.5); rvmax = np.percentile(RE, 99.5); ivmax = np.percentile(IN, 99.5)
    rows = [("GT\n(real beat)", GT, gvmax), ("input\n(breathing)", IN, ivmax), ("recon", RE, rvmax)]
    frames = []
    for t in range(T_total):
        fig, axs = plt.subplots(3, D, figsize=(2*D, 6.6), dpi=95)
        for ri, (lab, arr, vm) in enumerate(rows):
            for p in range(D):
                axs[ri, p].imshow(arr[t, p], cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
                if ri == 0: axs[ri, p].set_title(f"z{p}" + ("*REF" if p == ref_zmid else ""), fontsize=8)
                if p == 0: axs[ri, p].text(-0.42, 0.5, lab, transform=axs[ri, p].transAxes, rotation=90, va="center", fontsize=9)
        fig.suptitle(f"{name} IN-DIST CMRxRecon val subj0, breathing ON — GT beat | input | recon. cardiac phase {t}/{T_total-1}", fontsize=12)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(f"result/gif_indist_{name}_3row_S0.gif", frames, duration=0.18, loop=0)
    print(f"saved result/gif_indist_{name}_3row_S0.gif", flush=True)

for name in ["gather05", "s20contz"]:   # control0 already done
    render(name, *run_model(name, CKPTS[name]))
print("DONE", flush=True)
