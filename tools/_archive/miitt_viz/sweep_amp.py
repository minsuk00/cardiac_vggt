"""Does in-dist reconstruction accuracy DEGRADE as breathing goes clean->extreme?
gather05 on val subjects, sweep amplitude_mm, measure PSNR(recon, GT) over anatomy voxels."""
import os, sys, glob, numpy as np, torch, dataclasses
sys.path.insert(0, "training"); sys.path.insert(0, ".")
import torch.distributed as dist
if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29603")
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
dev = "cuda"; SUBJS = [0, 10, 26]; AMPS = [0.5, 8, 16, 24, 40, 60, 80]
with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
    cfg = compose(config_name="default")
mri_ds = instantiate(cfg.data.val, _recursive_=False).dataset.base_dataset.datasets[0]
base_resp = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)
ids = [str(p).split("/")[-2] for p in mri_ds.subjects]
model = load_rtfb_model_reference(glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0], refiner=False, device=dev)

def psnr(a, b, m):
    mse = float((((a - b) ** 2)[m]).mean()); return 20 * np.log10(1.0 / np.sqrt(mse + 1e-12))

def run(subj, amp):
    rc = dataclasses.replace(base_resp, amplitude_mm=amp, amplitude_jitter=amp * 0.15)
    data = mri_ds.get_data(seq_index=subj, img_per_seq=mri_ds.num_slices)
    def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(dev)
    phases = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(dev)
    batch = {"images": st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0,
             "scanner_coords": st("scanner_coords"), "z_indices": st("z_indices"), "t_indices": st("t_indices"),
             "phases": phases.unsqueeze(0), "timesteps": st("timesteps", np.int64),
             "slice_indices": st("slice_indices", np.float32), "seq_index": torch.tensor([[subj]], dtype=torch.int64, device=dev)}
    S = batch["images"].shape[1]; T = phases.shape[0]; grid = tuple(mri_ds.gt_grid_shape)
    ps, disp_max = [], 0.0
    for t in range(T):
        batch["target_t_indices"] = torch.full((1, S, 1), (t / max(1, T)) * 2 - 1, dtype=torch.float32, device=dev)
        batch["timesteps"][:, 0] = t
        batch = gpu_augment_batch(batch, None, dev, respiratory_cfg=rc, train=False)
        if t == 0: disp_max = float(np.linalg.norm(batch["resp_disp_mm"][0].cpu().numpy(), axis=1).max())
        batch["gt_target_volume"] = phases[t].unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
            out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()}, batch, grid_shape=grid, tv_weight=0.0)
        RE = out["V_canon"][0].float().cpu().numpy(); GT = out["V_gt"][0].float().cpu().numpy()
        m = GT > 0.05
        if m.any(): ps.append(psnr(RE, GT, m))
    return float(np.mean(ps)), disp_max

print(f"{'subj':16s} " + " ".join(f"amp{a:g}".rjust(11) for a in AMPS), flush=True)
for subj in SUBJS:
    res = [run(subj, a) for a in AMPS]
    cells = " ".join(f"{p:.1f}dB/{d:.0f}mm".rjust(11) for p, d in res)
    print(f"{ids[subj][:16]:16s} {cells}", flush=True)
print("\n(PSNR recon-vs-GT over anatomy voxels; /Nmm = max breathing disp that amp produced)", flush=True)
print("DONE", flush=True)
