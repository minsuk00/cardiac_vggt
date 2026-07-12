"""Run one model on one in-dist VAL subject, breathing ON. 3-row (GT|input(blanked)|recon).
Usage: python run_indist_subj.py <name> <subj_idx>"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, "training"); sys.path.insert(0, ".")
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29599")
    dist.init_process_group("gloo", rank=0, world_size=1)
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss
from inference.inference import load_rtfb_model_reference
for r, fn in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
              ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
    try: OmegaConf.register_new_resolver(r, fn)
    except Exception: pass
NAME = sys.argv[1]; SUBJ = int(sys.argv[2]); dev = "cuda"
CKPTS = {"control0": "216539845_*ftctrl_gather0*1frame*", "gather05": "216539845_*ftgather05*1frame*",
         "s20contz": "216949414_*s20contz*"}
with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
    cfg = compose(config_name="mri_volume")
mri_ds = instantiate(cfg.data.val, _recursive_=False).dataset.base_dataset.datasets[0]
resp_cfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)
import dataclasses
TAG = ""
if len(sys.argv) > 3:   # optional amplitude override (mm) for EXTREME/out-of-distribution breathing
    amp = float(sys.argv[3]); resp_cfg = dataclasses.replace(resp_cfg, amplitude_mm=amp, amplitude_jitter=amp * 0.15)
    TAG = f"_amp{int(amp)}"; print(f"OVERRIDE amplitude_mm={amp} (train default 16±8)", flush=True)
ids = [str(p).split("/")[-2] for p in mri_ds.subjects]

model = load_rtfb_model_reference(glob.glob(f"scratch/logs/{CKPTS[NAME]}/ckpts/checkpoint_last.pt")[0], refiner=False, device=dev)
data = mri_ds.get_data(seq_index=SUBJ, img_per_seq=mri_ds.num_slices)
def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(dev)
imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0; S = imgs.shape[1]
phases = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(dev)
batch = {"images": imgs, "scanner_coords": st("scanner_coords"), "z_indices": st("z_indices"), "t_indices": st("t_indices"),
         "phases": phases.unsqueeze(0), "timesteps": st("timesteps", np.int64), "slice_indices": st("slice_indices", np.float32),
         "seq_index": torch.tensor([[SUBJ]], dtype=torch.int64, device=dev)}
bb = np.asarray(data["anatomy_bbox"]).astype(np.int64); rz = (int(bb[0]) + int(bb[1])) // 2
D = phases.shape[1]; T = phases.shape[0]; grid = tuple(mri_ds.gt_grid_shape)
slice_z = batch["slice_indices"][0].cpu().numpy()
sop = [int(np.argmin(np.abs(slice_z - p))) for p in range(D)]
has_slot = [bool(np.min(np.abs(slice_z - p)) < 0.5) for p in range(D)]
GT = np.zeros((T, D, 256, 256), np.float32); RE = np.zeros_like(GT); IN = np.zeros_like(GT); rd = None
for t in range(T):
    batch["target_t_indices"] = torch.full((1, S, 1), (t / max(1, T)) * 2 - 1, dtype=torch.float32, device=dev)
    batch["timesteps"][:, 0] = t
    batch = gpu_augment_batch(batch, None, dev, respiratory_cfg=resp_cfg, train=False)
    if rd is None: rd = batch["resp_disp_mm"][0].cpu().numpy()
    batch["gt_target_volume"] = phases[t].unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
        out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()}, batch, grid_shape=grid, tv_weight=0.0)
    RE[t] = out["V_canon"][0].float().cpu().numpy(); GT[t] = out["V_gt"][0].float().cpu().numpy()
    im = F.interpolate(batch["images"][0, :, 0][:, None], size=(256, 256), mode="bilinear", align_corners=True)[:, 0].cpu().numpy()
    for p in range(D): IN[t, p] = im[sop[p]]
mag = np.linalg.norm(rd, axis=1)
AMP = f"breathing |disp| mean {mag.mean():.1f}mm max {mag.max():.1f}mm (SI/z max {np.abs(rd[:,0]).max():.1f}mm)"
print(f"[{NAME} subj{SUBJ}={ids[SUBJ]}] {AMP}  recon motion/plane={np.round(RE.std(0).mean((1,2)),4)}", flush=True)
np.savez_compressed(f"result/indist_{NAME}_subj{SUBJ}{TAG}.npz", gt=GT, recon=RE, inp=IN, ref_zmid=rz, has_slot=np.array(has_slot))
gv, rv, iv = [np.percentile(x, 99.5) for x in (GT, RE, IN)]; blank = np.zeros((256, 256), np.float32)
rows = [("GT (real beat)", GT, gv, False), ("input (breathing)", IN, iv, True), ("recon", RE, rv, False)]; ycen = [0.74, 0.45, 0.16]
frames = []
for t in range(T):
    fig, axs = plt.subplots(3, D, figsize=(2*D, 6.8), dpi=95)
    for ri, (lab, arr, vm, bl) in enumerate(rows):
        for p in range(D):
            axs[ri, p].imshow(arr[t, p] if (has_slot[p] or not bl) else blank, cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
            if ri == 0: axs[ri, p].set_title(f"z{p}" + ("*REF" if p == rz else ("" if has_slot[p] else "\n(no input)")), fontsize=8)
    for yc, (lab, *_ ) in zip(ycen, rows): fig.text(0.012, yc, lab, rotation=90, va="center", ha="center", fontsize=11)
    fig.suptitle(f"{NAME} | IN-DIST VAL subj{SUBJ} ({ids[SUBJ]}) | {AMP} | cardiac phase {t}/{T-1}", fontsize=11)
    fig.subplots_adjust(left=0.035, right=0.997, top=0.90, bottom=0.01, wspace=0.04, hspace=0.10)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave(f"result/gif_indist_{NAME}_subj{SUBJ}{TAG}.gif", frames, duration=0.18, loop=0)
print(f"saved result/gif_indist_{NAME}_subj{SUBJ}{TAG}.gif\nDONE", flush=True)
