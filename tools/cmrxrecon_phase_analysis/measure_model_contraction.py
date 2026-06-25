"""Measure the TRAINED model's contraction vs GT: does REFERENCE-SLICE conditioning
recover per-patient amplitude, or still regress toward the mean (under-contract)?

For each val subject we sweep target_t = 0..11, reconstructing V_canon at each. We
save both the predicted V_canon and the GT canonical phase V_gt as nnU-Net inputs. A
later seg + analysis step turns these into LV-volume curves -> predicted EF vs GT EF.

REFERENCE-SLICE CONTRACT (docs/25): slot 0 = a real target-phase slice at the
mid-ventricular plane (reference_slot=True), marked via VGGT's native camera_token
anchor (use_reference_token=True). The model reads the target phase from slot-0's
image CONTENT, not a target_t index. Consequently slot 0's image CHANGES across the
t-sweep (it IS the reference) while slots 1..S-1 stay bit-identical (seq_index-seeded,
independent of t_target). The cross-t guard therefore fingerprints slots 1..S-1 only.

Model: 217721337 (reference, resp-on, z-only, reference-token, aggft, DPT head).
"""
import os, sys, argparse
import numpy as np
import torch
import nibabel as nib
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss

CKPT = "/home/minsukc/vggt/scratch/logs/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
CANON_SPACING = (1.4, 1.4, 12.0)  # canonical Z relabeled 8->12mm true pitch (docs/27)
T = 12


def save_nnunet(vol_dhw, out_dir, tag, kind):
    """vol_dhw: canonical (D=12,H=256,W=256) splat-order (Z,Y,X) -> nnU-Net (X,Y,Z)."""
    assert "/" not in tag, f"tag must be a bare id, got {tag!r}"  # guard the abs-path bug
    arr = np.transpose(np.asarray(vol_dhw, np.float32), (2, 1, 0))   # (X,Y,Z)
    path = os.path.join(out_dir, f"{tag}_{kind}_0000.nii.gz")
    nib.save(nib.Nifti1Image(arr, np.diag([*CANON_SPACING, 1.0])), path)


def subj_id(path):
    # ds.subjects holds full paths ".../Cine_combined/<ID>/sax"  -> "<ID>"
    return os.path.basename(os.path.dirname(path))


def build_batch(data, device):
    def stack(key, dtype=np.float32):
        return torch.from_numpy(np.stack(data[key]).astype(dtype)).unsqueeze(0)
    imgs = stack("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    batch = {
        "images": imgs,
        "scanner_coords": stack("scanner_coords"),
        "world_points": stack("world_points"),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
        "z_indices": stack("z_indices"),
        "t_indices": stack("t_indices"),
        "target_t_indices": stack("target_t_indices"),
        "timesteps": torch.from_numpy(np.stack(data["timesteps"]).astype(np.int64)).unsqueeze(0),
        "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0),
        "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0),
        "t_target": torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0),
    }
    return {k: v.to(device) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_subjects", type=int, default=30)
    ap.add_argument("--ckpt", default=CKPT, help="checkpoint .pt (default: reference model)")
    ap.add_argument("--warp_head_type", default="dpt", choices=["dpt", "bspline"])
    ap.add_argument("--bspline_grid_size", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True, "rescale_aug": False,
        "landscape_check": False, "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518,
                    reference_slot=True)
    n_val = len(ds.subjects)
    print(f"{n_val} val subjects; measuring {min(args.n_subjects, n_val)}")

    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=False,
                 use_target_t_pose_embedding=False, use_reference_token=True,
                 train_on_residual_dvf=True,
                 warp_head_type=args.warp_head_type, bspline_grid_size=args.bspline_grid_size).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    assert len(missing) == 0 and len(unexpected) == 0, \
        f"checkpoint/model mismatch! missing={missing[:8]} unexpected={unexpected[:8]}"
    model.eval()

    for si in range(min(args.n_subjects, n_val)):
        subj = subj_id(ds.subjects[si % n_val])
        ref_t_idx = None
        for t in range(T):
            ds.t_target_fixed = t
            data = ds.get_data(seq_index=si, img_per_seq=12)
            batch = build_batch(data, device)
            # GUARD: scattered slots 1..S-1 are identical across the t-sweep (seq_index-
            # seeded, independent of t_target). Slot 0 is the target-phase REFERENCE and is
            # EXPECTED to change with t — so it's excluded from the fingerprint.
            # Fingerprint t + z + scanner_coords (per reviewer: t alone is weaker).
            cur = (batch["t_indices"][:, 1:].cpu().numpy().round(5).tobytes()
                   + batch["z_indices"][:, 1:].cpu().numpy().round(5).tobytes()
                   + batch["scanner_coords"][:, 1:].cpu().numpy().round(5).tobytes())
            if ref_t_idx is None:
                ref_t_idx = cur
            elif cur != ref_t_idx:
                raise RuntimeError(f"SCATTERED INPUT SLICES CHANGED across t for subj {subj}! "
                                   "fixed-inputs assumption violated.")
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = model(batch["images"], batch=batch)
            preds["world_points"] = preds["world_points"].float()
            out = compute_volume_intensity_loss(preds, batch, grid_shape=(T, 256, 256), tv_weight=0.1)
            V_canon = out["V_canon"][0].float().cpu().numpy()   # (D,H,W)
            V_gt = out["V_gt"][0].float().cpu().numpy()
            tag = f"{subj}_t{t:02d}"
            save_nnunet(V_canon, args.out_dir, tag, "pred")
            save_nnunet(V_gt,    args.out_dir, tag, "gt")
        ds.t_target_fixed = None
        print(f"[{si+1}] {subj}: 12 phases x (pred,gt) saved")

    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
