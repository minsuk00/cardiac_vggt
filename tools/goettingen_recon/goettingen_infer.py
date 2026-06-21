"""Run the trained joint-L1 VGGT model on Göttingen RT free-breathing data and render a
beating-heart GIF by sweeping the target cardiac phase.

The model is z-only for inputs (use_t_pose_embedding=false) + target_t for the query, with a
coverage-refiner, trained WITH respiratory aug. So we feed scattered real Göttingen slices
(z-conditioned, blind to input phase) and sweep target_t = 0..11 to reconstruct the cardiac cycle.

Run on a GPU node: micromamba run -n svr python tools/goettingen_recon/goettingen_infer.py
"""
import os, sys
import numpy as np
import torch
import nibabel as nib
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT

CKPT = "/home/minsukc/vggt/scratch/logs/218349151_mri_refiner_joint/ckpts/checkpoint_last.pt"
ROOT = "/home/minsukc/vggt/scratch/data/goettingen/canonical_subjects"
SPLIT = ROOT + "/gott_split.txt"
OUTDIR = "/home/minsukc/vggt/scratch/data/goettingen/inference"
SUBJECTS = ["vol0001_vis1", "vol0023_vis1", "vol0009_vis1"]
T = 12


def build_model(device):
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
        enable_refiner=True, refiner_use_coverage=True, grid_shape=(12, 256, 256),
    ).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"  ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
    # sanity: the loaded model must NOT have stray missing refiner/embedder weights
    bad = [k for k in missing if ("refiner" in k or "z_embedder" in k or "target_t_embedder" in k)]
    assert not bad, f"missing trained weights: {bad[:5]}"
    model.eval()
    return model


def get_batch(ds, seq_index, device):
    data = ds.get_data(seq_index=seq_index, img_per_seq=12)
    def stack(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    imgs = stack("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    batch = {"images": imgs, "scanner_coords": stack("scanner_coords"),
             "world_points": stack("world_points"),
             "z_indices": stack("z_indices"), "t_indices": stack("t_indices"),
             "target_t_indices": stack("target_t_indices"),
             "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0)}
    return {k: v.to(device) for k, v in batch.items()}, data


def norm8(x, hi=None):
    hi = hi if hi is not None else np.percentile(x, 99.5)
    return (np.clip(x / (hi + 1e-9), 0, 1) * 255).astype(np.uint8)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common_conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                                    "rescale_aug": False, "landscape_check": False,
                                    "augs": {"scales": [1.0, 1.0]}})
    ds = MRIDataset(common_conf, ROOT, split="val", split_file=SPLIT,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)
    print(f"subjects: {[os.path.basename(os.path.dirname(s)) for s in ds.subjects]}")
    model = build_model(device)

    for si, subj in enumerate(SUBJECTS):
        idx = next(i for i, s in enumerate(ds.subjects) if subj in s)
        batch, data = get_batch(ds, idx, device)
        S = batch["images"].shape[1]
        # sweep target_t -> V_refined(q)
        vols = []
        for q in range(T):
            tval = (q / T) * 2.0 - 1.0
            batch["target_t_indices"] = torch.full((1, S, 1), tval, dtype=torch.float32, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = model(batch["images"], batch=batch)
            V = preds.get("V_refined", preds.get("V_canon"))[0].float().cpu().numpy()  # (12,256,256)
            vols.append(V)
        vols = np.stack(vols)                          # (Tq=12, D=12, 256, 256)
        hi = np.percentile(vols, 99.5)
        np.save(f"{OUTDIR}/{subj}_Vrefined_sweep.npy", vols)

        # GIF: mid-D axial slice across query phases (the beating heart), 2x upscale
        midD = vols.shape[1] // 2
        frames = []
        for q in range(T):
            a = norm8(vols[q, midD], hi)
            im = Image.fromarray(a, "L").resize((512, 512), Image.BILINEAR)
            frames.append(im)
        frames[0].save(f"{OUTDIR}/{subj}_beating.gif", save_all=True, append_images=frames[1:],
                       duration=120, loop=0)

        # static montage: 6 query phases x 3 depths (apex/mid/base)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        qs = np.linspace(0, T - 1, 6).astype(int)
        ds_depths = [vols.shape[1] // 4, vols.shape[1] // 2, 3 * vols.shape[1] // 4]
        fig, ax = plt.subplots(3, 6, figsize=(15, 7.5))
        for r, d in enumerate(ds_depths):
            for c, q in enumerate(qs):
                ax[r, c].imshow(np.clip(vols[q, d] / hi, 0, 1), cmap="gray"); ax[r, c].axis("off")
                if r == 0: ax[r, c].set_title(f"target_t={q}", fontsize=9)
                if c == 0: ax[r, c].set_ylabel(f"z={d}", fontsize=9)
        plt.suptitle(f"{subj}: V_refined across target cardiac phase (z-only inputs, S={S})", fontsize=12)
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/{subj}_phase_montage.png", dpi=95, bbox_inches="tight"); plt.close()
        print(f"{subj}: wrote beating.gif + phase_montage.png  V range [{vols.min():.3f},{vols.max():.3f}]")


if __name__ == "__main__":
    main()
