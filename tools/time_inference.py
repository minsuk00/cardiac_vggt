"""Time the model forward pass per val subject. Reports:
  - dataset get_data + collation
  - host→GPU transfer
  - VGGT forward
  - V_canon construction (splat + loss)
Excludes figure rendering / disk IO.
"""
import os
import sys
import time
import random
import numpy as np
import torch
import nibabel as nib
import glob
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss

CKPT = "/home/minsukc/vggt/scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
N_TIMINGS = 5
N_WARMUP = 2


class IdentityRandom:
    def __init__(self, *a, **kw): pass
    def shuffle(self, x): pass
    def random(self): return 0.0


def build_batch(ds, seq_idx):
    orig_class, orig_shuffle = random.Random, random.shuffle
    random.Random = IdentityRandom
    random.shuffle = lambda x: None
    try:
        data = ds.get_data(seq_index=seq_idx, img_per_seq=12)
    finally:
        random.Random = orig_class
        random.shuffle = orig_shuffle
    def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    return {
        "images": imgs, "scanner_coords": st("scanner_coords"),
        "world_points": st("world_points"),
        "z_indices": st("z_indices"), "t_indices": st("t_indices"),
        "target_t_indices": st("target_t_indices"),
        "timesteps": torch.from_numpy(np.stack(data["timesteps"]).astype(np.int64)).unsqueeze(0),
        "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
        "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0)
                            if "gt_target_volume" in data else None,
        "t_target": torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0)
                    if "t_target" in data else None,
    }


def main():
    device = "cuda"
    print("Building model...")
    t0 = time.perf_counter()
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=True, use_target_t_pose_embedding=True,
                 train_on_residual_dvf=True).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)

    print(f"\nWarmup ({N_WARMUP} subjects)...")
    for i in range(N_WARMUP):
        b = build_batch(ds, i)
        b = {k: v.to(device) for k, v in b.items() if v is not None}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(b["images"], batch=b)
        torch.cuda.synchronize()

    print(f"\nTiming {N_TIMINGS} subjects (excludes JIT warmup, includes data load):\n")
    print(f"  {'subject':<14}  {'S':>3}  {'native':<14}   "
          f"{'data_ms':>8}  {'h→d_ms':>8}  {'fwd_ms':>8}  {'splat_ms':>9}  {'TOTAL_ms':>9}")
    rows = []
    for i in range(N_WARMUP, N_WARMUP + N_TIMINGS):
        # Data load (CPU)
        t0 = time.perf_counter()
        b = build_batch(ds, i)
        t_data = (time.perf_counter() - t0) * 1000

        # Host → device
        t0 = time.perf_counter()
        b = {k: v.to(device) for k, v in b.items() if v is not None}
        torch.cuda.synchronize()
        t_h2d = (time.perf_counter() - t0) * 1000

        # Forward
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(b["images"], batch=b)
        torch.cuda.synchronize()
        t_fwd = (time.perf_counter() - t0) * 1000

        # Splat + loss (the V_canon construction; what makes the actual output)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()},
                                            b, grid_shape=(12, 256, 256), tv_weight=0.1)
        torch.cuda.synchronize()
        t_splat = (time.perf_counter() - t0) * 1000

        S = b["images"].shape[1]
        sp = ds.subjects[i]
        nii0 = sorted(glob.glob(os.path.join(sp, "3d_recon", "sax_frame_*.nii.gz")))[0]
        native = nib.load(nii0).header.get_data_shape()[:3]
        name = os.path.basename(os.path.dirname(sp))
        total = t_data + t_h2d + t_fwd + t_splat
        rows.append((name, S, native, t_data, t_h2d, t_fwd, t_splat, total))
        print(f"  {name:<14}  {S:>3}  {str(tuple(native)):<14}   "
              f"{t_data:8.1f}  {t_h2d:8.1f}  {t_fwd:8.1f}  {t_splat:9.1f}  {total:9.1f}")

    arr = np.array([[r[3], r[4], r[5], r[6], r[7]] for r in rows])
    print("\n  mean    "
          f"{'':<14}        {'':>14}    "
          f"{arr[:,0].mean():8.1f}  {arr[:,1].mean():8.1f}  "
          f"{arr[:,2].mean():8.1f}  {arr[:,3].mean():9.1f}  {arr[:,4].mean():9.1f}")
    print(f"\n  Per-subject inference (data + h→d + forward + splat):  "
          f"{arr[:,4].mean():.0f} ms  ≈ {arr[:,4].mean()/1000:.2f}s")
    print(f"  Model forward alone:  {arr[:,2].mean():.0f} ms ({arr[:,2].mean()/arr[:,4].mean()*100:.0f}% of total)")


if __name__ == "__main__":
    main()
