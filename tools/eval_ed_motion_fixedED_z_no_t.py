"""LIVE ED motion/bbox/full PSNR for the in-flight single-phase z-only/no-t runs.

Evals the two new checkpoints (head-only vs aggft, both fixedED z-only no-t) at ED
over all 30 val subjects under one common protocol + the identity floor. t-OFF
(use_t_pose_embedding=false) to match how the runs were trained.

  identity       : Delta=0 splat floor
  headonly_ep75  : single-phase ED, point_head only (near-final, ep75/100)
  aggft_ep28     : single-phase ED, aggregator+head (PARTIAL, ep28/100 — lower bound)
"""
import os, sys, json
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss, compute_motion_mask

DATA_ROOT = "scratch/data/CMRxRecon2024/Cine_combined"
SPLIT = "training/splits/random_8_1_1.txt"
DEV = "cuda"
T_TARGET = 0  # ED

# t-OFF to match the runs (use_t_pose_embedding=false); z + target_t on.
FLAGS = dict(use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True)
MODELS = {
    "headonly_ep75": "scratch/live_eval/headonly_ep75.pt",
    "aggft_ep28":    "scratch/live_eval/aggft_ep28.pt",
}


def build_ds():
    conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                             "rescale_aug": False, "landscape_check": False,
                             "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(conf, DATA_ROOT, split="val", split_file=SPLIT,
                      mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)


def make_batch(ds, seq_index, t_target):
    ds.t_target_fixed = t_target
    data = ds.get_data(seq_index=seq_index, img_per_seq=12)
    st = lambda k, dt=np.float32: torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    batch = {
        "images": st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0,
        "scanner_coords": st("scanner_coords"), "world_points": st("world_points"),
        "z_indices": st("z_indices"), "t_indices": st("t_indices"),
        "target_t_indices": st("target_t_indices"),
        "timesteps": st("timesteps", np.int64), "slice_indices": st("slice_indices", np.int64),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
        "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0),
        "t_target": torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0),
    }
    phases = torch.from_numpy(data["phases"].astype(np.float32)).unsqueeze(0)
    return batch, phases


def psnr(a, b, m=None):
    if m is None:
        mse = ((a - b) ** 2).mean()
    else:
        m = m.float(); n = m.sum().clamp(min=1)
        mse = (((a - b) ** 2) * m).sum() / n
    return float(10 * torch.log10(1.0 / mse.clamp(min=1e-10)))


def load_model(ckpt):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             train_on_residual_dvf=True, **FLAGS).to(DEV)
    ck = torch.load(ckpt, map_location=DEV, weights_only=False)
    m.load_state_dict(ck["model"], strict=False)
    m.eval()
    return m


@torch.no_grad()
def eval_world_points(world_points, bd, phases):
    preds = {"world_points": world_points.float()}
    out = compute_volume_intensity_loss(preds, bd, grid_shape=(12, 256, 256), tv_weight=0.1)
    Vc = out["V_canon"][0].float().cpu(); Vg = out["V_gt"][0].float().cpu()
    mot = compute_motion_mask(phases)[0].cpu(); anat = (Vg > 1e-3)
    return dict(full=psnr(Vc, Vg), bbox=psnr(Vc, Vg, anat), motion=psnr(Vc, Vg, mot))


def main():
    ds = build_ds()
    n = len(ds.subjects)
    print(f"val subjects: {n}, target phase t={T_TARGET} (ED), t-OFF\n", flush=True)

    rows = {k: [] for k in ["identity", *MODELS]}
    batches = [make_batch(ds, s, T_TARGET) for s in range(n)]

    for batch, phases in batches:
        bd = {k: v.to(DEV) for k, v in batch.items()}
        rows["identity"].append(eval_world_points(bd["scanner_coords"], bd, phases))
        del bd; torch.cuda.empty_cache()
    print("identity done", flush=True)

    for mkey, ckpt in MODELS.items():
        model = load_model(ckpt)
        for batch, phases in batches:
            bd = {k: v.to(DEV) for k, v in batch.items()}
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = model(bd["images"], batch=bd)
            rows[mkey].append(eval_world_points(preds["world_points"], bd, phases))
            del bd, preds; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
        print(f"{mkey} done", flush=True)

    print(f"\n{'model':<16} {'motion':>16} {'bbox':>16} {'full':>16}   (mean +/- std, n={n}, t=ED, t-off)")
    summary = {}
    for k, per in rows.items():
        summary[k] = {m: [float(np.mean([r[m] for r in per])), float(np.std([r[m] for r in per]))]
                      for m in ["motion", "bbox", "full"]}
        f = summary[k]
        print(f"{k:<16} {f['motion'][0]:7.2f} +/- {f['motion'][1]:4.2f}  "
              f"{f['bbox'][0]:7.2f} +/- {f['bbox'][1]:4.2f}  "
              f"{f['full'][0]:7.2f} +/- {f['full'][1]:4.2f}")
    json.dump(summary, open("scratch/live_eval/ed_motion_z_no_t.json", "w"), indent=2)
    print("\nwrote scratch/live_eval/ed_motion_z_no_t.json", flush=True)


if __name__ == "__main__":
    main()
