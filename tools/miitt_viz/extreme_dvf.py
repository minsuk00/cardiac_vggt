"""gather05, EXTREME breathing (amp 50mm), subj0, WITH DVF. 6-row GIF: GT|input|recon|Dx|Dy|Dz."""
import os, sys, glob, numpy as np, torch, dataclasses
sys.path.insert(0, "training"); sys.path.insert(0, ".")
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29601")
    dist.init_process_group("gloo", rank=0, world_size=1)
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss
from inference.inference import load_rtfb_model_reference
from inference.adapters.base import MM_PER_NORM
for r, fn in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
              ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
    try: OmegaConf.register_new_resolver(r, fn)
    except Exception: pass
dev = "cuda"; SUBJ = 0; AMP = 50.0
with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
    cfg = compose(config_name="mri_volume")
mri_ds = instantiate(cfg.data.val, _recursive_=False).dataset.base_dataset.datasets[0]
resp_cfg = dataclasses.replace(RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory),
                               amplitude_mm=AMP, amplitude_jitter=AMP * 0.15)
ids = [str(p).split("/")[-2] for p in mri_ds.subjects]
def to256(a): idx = np.linspace(0, a.shape[0]-1, 256).astype(int); return a[np.ix_(idx, idx)]

model = load_rtfb_model_reference(glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0], refiner=False, device=dev)
data = mri_ds.get_data(seq_index=SUBJ, img_per_seq=mri_ds.num_slices)
def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(dev)
imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0; S = imgs.shape[1]
phases = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(dev)
batch = {"images": imgs, "scanner_coords": st("scanner_coords"), "z_indices": st("z_indices"), "t_indices": st("t_indices"),
         "phases": phases.unsqueeze(0), "timesteps": st("timesteps", np.int64), "slice_indices": st("slice_indices", np.float32),
         "seq_index": torch.tensor([[SUBJ]], dtype=torch.int64, device=dev)}
sc = batch["scanner_coords"][0].cpu().numpy()   # (S,518,518,3)
bb = np.asarray(data["anatomy_bbox"]).astype(np.int64); rz = (int(bb[0]) + int(bb[1])) // 2
D = phases.shape[1]; T = phases.shape[0]; grid = tuple(mri_ds.gt_grid_shape)
slice_z = batch["slice_indices"][0].cpu().numpy()
sop = [int(np.argmin(np.abs(slice_z - p))) for p in range(D)]
has_slot = [bool(np.min(np.abs(slice_z - p)) < 0.5) for p in range(D)]
GT = np.zeros((T, D, 256, 256), np.float32); RE = np.zeros_like(GT); IN = np.zeros_like(GT)
DV = np.zeros((T, D, 256, 256, 3), np.float32); rd = None
for t in range(T):
    batch["target_t_indices"] = torch.full((1, S, 1), (t / max(1, T)) * 2 - 1, dtype=torch.float32, device=dev)
    batch["timesteps"][:, 0] = t
    batch = gpu_augment_batch(batch, None, dev, respiratory_cfg=resp_cfg, train=False)
    if rd is None: rd = batch["resp_disp_mm"][0].cpu().numpy()
    batch["gt_target_volume"] = phases[t].unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
        wp = preds["world_points"][0].float().cpu().numpy()   # (S,518,518,3)
        out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()}, batch, grid_shape=grid, tv_weight=0.0)
    RE[t] = out["V_canon"][0].float().cpu().numpy(); GT[t] = out["V_gt"][0].float().cpu().numpy()
    im = F.interpolate(batch["images"][0, :, 0][:, None], size=(256, 256), mode="bilinear", align_corners=True)[:, 0].cpu().numpy()
    for p in range(D):
        IN[t, p] = im[sop[p]]
        d = (wp[sop[p]] - sc[sop[p]]) * np.array(MM_PER_NORM)[None, None, :]
        DV[t, p] = np.stack([to256(d[..., k]) for k in range(3)], -1)
mag = np.linalg.norm(rd, axis=1)
AMPS = f"EXTREME breathing |disp| mean {mag.mean():.1f}mm max {mag.max():.1f}mm"
vlx, vly, vlz = [max(1.0, np.percentile(np.abs(DV[..., k]), 99)) for k in range(3)]
print(f"[gather05 subj0 amp{AMP}] {AMPS}  DVF vlims dx/dy/dz={vlx:.0f}/{vly:.0f}/{vlz:.0f}mm", flush=True)
np.savez_compressed("result/indist_gather05_subj0_amp50_dvf.npz", gt=GT, recon=RE, inp=IN, dvf=DV, has_slot=np.array(has_slot), ref_zmid=rz)

gv, rv, iv = [np.percentile(x, 99.5) for x in (GT, RE, IN)]; blank = np.zeros((256, 256), np.float32)
rows = [("GT", GT, "gray", 0, gv, False),
        ("input", IN, "gray", 0, iv, True), ("recon", RE, "gray", 0, rv, False),
        (f"Dx±{vlx:.0f}", DV[..., 0], "bwr", -vlx, vlx, True), (f"Dy±{vly:.0f}", DV[..., 1], "bwr", -vly, vly, True),
        (f"Dz±{vlz:.0f}mm", DV[..., 2], "bwr", -vlz, vlz, True)]
ycen = np.linspace(0.90, 0.03, 7)[:-1] - (0.90-0.03)/12
frames = []
for t in range(T):
    fig, axs = plt.subplots(6, D, figsize=(2*D, 11), dpi=85)
    for ri, (lab, arr, cm, lo, hi, bl) in enumerate(rows):
        for p in range(D):
            img = arr[t, p] if (has_slot[p] or not bl) else blank
            axs[ri, p].imshow(img, cmap=cm, vmin=lo, vmax=hi); axs[ri, p].axis("off")
            if ri == 0: axs[ri, p].set_title(f"z{p}" + ("*REF" if p == rz else ("" if has_slot[p] else "\n(no in)")), fontsize=7)
    for yc, (lab, *_ ) in zip(ycen, rows): fig.text(0.011, yc, lab, rotation=90, va="center", ha="center", fontsize=10)
    fig.suptitle(f"gather05 | IN-DIST VAL subj0 ({ids[SUBJ]}) | {AMPS} | GT|input|recon|DVF Dx/Dy/Dz | phase {t}/{T-1}", fontsize=11)
    fig.subplots_adjust(left=0.03, right=0.997, top=0.93, bottom=0.005, wspace=0.04, hspace=0.08)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave("result/gif_indist_gather05_subj0_amp50_dvf.gif", frames, duration=0.2, loop=0)
print("saved result/gif_indist_gather05_subj0_amp50_dvf.gif\nDONE", flush=True)
