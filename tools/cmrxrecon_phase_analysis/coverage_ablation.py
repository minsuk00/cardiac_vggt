"""Coverage ablation: force k input slots to OBSERVE the target phase, re-measure EF.

Decides coverage-limit vs capacity-limit. For each n_forced in {1,6,12(all-z)} we
run the trained model with that many input slots placed at the target phase (same z
as the scattered baseline; only their phase changes), sweep target_t=0..11, save the
predicted volume for segmentation. n_forced=0 is the existing baseline (model_vols).

If pred-EF-vs-GT-EF slope -> 1 as n_forced -> all-z  => INFORMATION/COVERAGE limit.
If it stays ~0 even at all-z                          => CAPACITY limit.

Saves only PRED volumes (GT is identical across conditions; reuse model baseline gt).
Tags: {subj}_t{tt}_f{n}_pred_0000.nii.gz
"""
import os, sys, argparse
import numpy as np, torch, nibabel as nib
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss

CKPT = "/home/minsukc/vggt/scratch/logs/218643188_mri_volume_noresp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
CANON_SPACING = (1.4, 1.4, 8.0); T = 12
FORCE_LEVELS = [1, 6, 12]


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
    device = "cuda"

    cc = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                           "rescale_aug": False, "landscape_check": False,
                           "augs": {"scales": [1.0, 1.0]}})
    ds = MRIDataset(cc, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)
    n_val = len(ds.subjects)

    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=False,
                 use_target_t_pose_embedding=True, train_on_residual_dvf=True).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    assert len(miss) == 0 and len(unexp) == 0, (miss[:8], unexp[:8])
    model.eval()
    print(f"loaded ckpt clean; ablation over n_forced={FORCE_LEVELS} on {min(args.n_subjects,n_val)} subjects")

    for nf in FORCE_LEVELS:
        for si in range(min(args.n_subjects, n_val)):
            subj = subj_id(ds.subjects[si % n_val])
            # GUARD: verify the forced slots actually observe the target phase
            for t in range(T):
                ds.t_target_fixed = t
                data = ds.get_data(seq_index=si, img_per_seq=12, n_forced_target=nf)
                # check: first min(nf,S) input t_indices equal target_t (normalized)
                ti = np.stack(data["t_indices"]).ravel()
                tt = float((t / T) * 2 - 1)
                k = min(nf, len(ti))
                assert np.allclose(ti[:k], tt, atol=1e-4), \
                    f"forced slots not at target phase: {ti[:k]} vs {tt}"
                batch = build_batch(data, device)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    preds = model(batch["images"], batch=batch)
                preds["world_points"] = preds["world_points"].float()
                out = compute_volume_intensity_loss(preds, batch, grid_shape=(T, 256, 256), tv_weight=0.1)
                save_nnunet(out["V_canon"][0].float().cpu().numpy(), args.out_dir, f"{subj}_t{t:02d}_f{nf:02d}", "pred")
            ds.t_target_fixed = None
        print(f"  n_forced={nf}: {min(args.n_subjects,n_val)} subjects done")
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
