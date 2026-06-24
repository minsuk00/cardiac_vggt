"""ORACLE-SPLAT baseline: isolate the splat renderer from the model.

For each (subject, target_t) we splat the TRUE target-phase slices at their TRUE
positions (identity placement, no model, no motion error), same z-sparsity the model
saw. If oracle-EF tracks GT-EF (slope ~1) the splat PRESERVES per-patient contraction
-> the model's slope~0 is the conditioning/motion (architectural). If oracle-EF is also
flat (~constant) the SPLAT destroys EF -> renderer cap, conditioning question is moot.

No model is loaded. Uses static mode (all input slices at the target phase) + identity
world_points = scanner_coords.
"""
import os, sys, argparse
import numpy as np
import torch, nibabel as nib
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.datasets.mri_dataset import MRIDataset
from loss import compute_volume_intensity_loss

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
CANON_SPACING = (1.4, 1.4, 8.0); T = 12


def save_nnunet(vol_dhw, out_dir, tag, kind):
    assert "/" not in tag
    arr = np.transpose(np.asarray(vol_dhw, np.float32), (2, 1, 0))
    nib.save(nib.Nifti1Image(arr, np.diag([*CANON_SPACING, 1.0])),
             os.path.join(out_dir, f"{tag}_{kind}_0000.nii.gz"))


def subj_id(p):
    return os.path.basename(os.path.dirname(p))


def build_batch(data, device):
    def stack(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    imgs = stack("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    b = {"images": imgs, "scanner_coords": stack("scanner_coords"),
         "z_indices": stack("z_indices"), "t_indices": stack("t_indices"),
         "target_t_indices": stack("target_t_indices"),
         "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0),
         "t_target": torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0)}
    return {k: v.to(device) for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_subjects", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cc = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                           "rescale_aug": False, "landscape_check": False,
                           "augs": {"scales": [1.0, 1.0]}})
    # STATIC mode: every input slice is at the target phase (correct-phase, perfect data)
    ds = MRIDataset(cc, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="static", mri_mode="axial", num_slices=12, target_size=518)
    n_val = len(ds.subjects)
    print(f"{n_val} val subjects; oracle for {min(args.n_subjects, n_val)} (static mode, identity splat)")

    for si in range(min(args.n_subjects, n_val)):
        subj = subj_id(ds.subjects[si % n_val])
        for t in range(T):
            ds.t_target_fixed = t
            data = ds.get_data(seq_index=si, img_per_seq=12)
            batch = build_batch(data, device)
            # IDENTITY placement: world_points = scanner_coords (no motion, perfect)
            preds = {"world_points": batch["scanner_coords"].clone()}
            out = compute_volume_intensity_loss(preds, batch, grid_shape=(T, 256, 256), tv_weight=0.1)
            save_nnunet(out["V_canon"][0].float().cpu().numpy(), args.out_dir, f"{subj}_t{t:02d}", "oracle")
            save_nnunet(out["V_gt"][0].float().cpu().numpy(), args.out_dir, f"{subj}_t{t:02d}", "gt")
        ds.t_target_fixed = None
        print(f"[{si+1}] {subj} done")
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
