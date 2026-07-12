"""Survey breathing amplitude across all 30 val subjects (deterministic, cheap), pick a big
breather, run gather05 on it, render 3-row (GT beat | breathing input z1-9 blanked | recon)."""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, "training"); sys.path.insert(0, ".")
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29598")
    dist.init_process_group("gloo", rank=0, world_size=1)
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig, sample_resp_disp
from loss import compute_volume_intensity_loss
from inference.inference import load_rtfb_model_reference
for r, fn in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
              ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
    try: OmegaConf.register_new_resolver(r, fn)
    except Exception: pass
dev = "cuda"
with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
    cfg = compose(config_name="mri_volume")
mri_ds = instantiate(cfg.data.val, _recursive_=False).dataset.base_dataset.datasets[0]
resp_cfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)
ids = [str(p).split("/")[-2] for p in mri_ds.subjects]; N = len(ids)

# ---- cheap survey: per-subject breathing amplitude via deterministic seq_index seeding ----
gid = (torch.arange(20).view(1, 20) % 12).long()  # representative z layout
rows = []
for i in range(N):
    disp, _ = sample_resp_disp(1, 20, resp_cfg, "cpu", train=False, seq_index=torch.tensor([[i]]), group_ids=gid)
    m = np.linalg.norm(disp[0].numpy(), axis=1)
    rows.append((i, float(m.max()), float(m.mean())))
rows.sort(key=lambda x: -x[1])
print("top breathers (seq, max|disp|mm, mean):", [(i, round(mx, 1), round(mn, 1)) for i, mx, mn in rows[:6]], flush=True)
SUBJ = rows[0][0]
print(f"chosen subj {SUBJ} = {ids[SUBJ]} (survey max {rows[0][1]:.1f}mm)", flush=True)

# ---- run gather05 on chosen subject ----
ck = glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0]
model = load_rtfb_model_reference(ck, refiner=False, device=dev)
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
GT = np.zeros((T, D, 256, 256), np.float32); RE = np.zeros_like(GT); IN = np.zeros_like(GT); real_disp = None
for t in range(T):
    batch["target_t_indices"] = torch.full((1, S, 1), (t / max(1, T)) * 2 - 1, dtype=torch.float32, device=dev)
    batch["timesteps"][:, 0] = t
    batch = gpu_augment_batch(batch, None, dev, respiratory_cfg=resp_cfg, train=False)
    if real_disp is None: real_disp = batch["resp_disp_mm"][0].cpu().numpy()
    batch["gt_target_volume"] = phases[t].unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
        out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()}, batch, grid_shape=grid, tv_weight=0.0)
    RE[t] = out["V_canon"][0].float().cpu().numpy(); GT[t] = out["V_gt"][0].float().cpu().numpy()
    im256 = F.interpolate(batch["images"][0, :, 0][:, None], size=(256, 256), mode="bilinear", align_corners=True)[:, 0].cpu().numpy()
    for p in range(D): IN[t, p] = im256[sop[p]]
mag = np.linalg.norm(real_disp, axis=1)
AMP = f"breathing |disp| mean {mag.mean():.1f}mm max {mag.max():.1f}mm (SI/z max {np.abs(real_disp[:,0]).max():.1f}mm)"
print(f"[gather05 subj {SUBJ}] {AMP}  recon motion/plane={np.round(RE.std(0).mean((1,2)),4)}", flush=True)
np.savez_compressed(f"result/indist_gather05_subj{SUBJ}.npz", gt=GT, recon=RE, inp=IN, ref_zmid=rz, has_slot=np.array(has_slot))

gv, rv, iv = [np.percentile(x, 99.5) for x in (GT, RE, IN)]; blank = np.zeros((256, 256), np.float32)
disp_rows = [("GT (real beat)", GT, gv, False), ("input (breathing)", IN, iv, True), ("recon", RE, rv, False)]; ycen = [0.74, 0.45, 0.16]
frames = []
for t in range(T):
    fig, axs = plt.subplots(3, D, figsize=(2*D, 6.8), dpi=95)
    for ri, (lab, arr, vm, bl) in enumerate(disp_rows):
        for p in range(D):
            axs[ri, p].imshow(arr[t, p] if (has_slot[p] or not bl) else blank, cmap="gray", vmin=0, vmax=vm); axs[ri, p].axis("off")
            if ri == 0: axs[ri, p].set_title(f"z{p}" + ("*REF" if p == rz else ("" if has_slot[p] else "\n(no input)")), fontsize=8)
    for yc, (lab, *_ ) in zip(ycen, disp_rows): fig.text(0.012, yc, lab, rotation=90, va="center", ha="center", fontsize=11)
    fig.suptitle(f"gather05 | IN-DIST VAL subj{SUBJ} ({ids[SUBJ]}) | {AMP} | cardiac phase {t}/{T-1}", fontsize=11)
    fig.subplots_adjust(left=0.035, right=0.997, top=0.90, bottom=0.01, wspace=0.04, hspace=0.10)
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
imageio.mimsave(f"result/gif_indist_gather05_bigbreath_subj{SUBJ}.gif", frames, duration=0.18, loop=0)
print(f"saved result/gif_indist_gather05_bigbreath_subj{SUBJ}.gif\nDONE", flush=True)
