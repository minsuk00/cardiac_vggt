"""Control: run the SAME joint-L1 model + same target_t sweep on IN-DISTRIBUTION CMRxRecon val
subjects, quantify cardiac motion the same way, and compare to the Göttingen (OOD) numbers.

If in-distribution beats much more strongly than Göttingen -> the weak Göttingen beat is a DOMAIN GAP.
If similar -> the weak beat is just the model's general behavior (no domain-gap claim).
"""
import os, sys
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT

CKPT = "/home/minsukc/vggt/scratch/logs/218349151_mri_refiner_joint/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
OUTDIR = "/home/minsukc/vggt/scratch/data/goettingen/inference"
N_SUBJ = 3
T = 12


def main():
    device = "cuda"
    cc = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                           "rescale_aug": False, "landscape_check": False, "augs": {"scales": [1.0, 1.0]}})
    ds = MRIDataset(cc, DATA_ROOT, split="val", split_file=SPLIT,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)
    print(f"CMRxRecon val subjects: {len(ds.subjects)}", flush=True)

    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
                 train_on_residual_dvf=True, enable_refiner=True, refiner_use_coverage=True,
                 grid_shape=(12, 256, 256)).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    print("model loaded", flush=True)

    results = []
    for si in range(min(N_SUBJ, len(ds.subjects))):
        data = ds.get_data(seq_index=si, img_per_seq=12)
        def stack(k, dt=np.float32):
            return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
        imgs = stack("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
        batch = {"images": imgs, "scanner_coords": stack("scanner_coords"), "world_points": stack("world_points"),
                 "z_indices": stack("z_indices"), "t_indices": stack("t_indices"),
                 "target_t_indices": stack("target_t_indices"),
                 "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0)}
        batch = {k: v.to(device) for k, v in batch.items()}
        S = batch["images"].shape[1]
        vols = []
        for q in range(T):
            batch["target_t_indices"] = torch.full((1, S, 1), (q / T) * 2 - 1, dtype=torch.float32, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = model(batch["images"], batch=batch)
            vols.append(preds.get("V_refined", preds.get("V_canon"))[0].float().cpu().numpy())
        vols = np.stack(vols)                                   # (12,12,256,256)
        tstd = vols.std(0)
        mv = float((tstd > 0.05 * vols.max()).mean())
        heart = float(tstd[:, 88:168, 88:168].mean()); full = float(tstd.mean())
        subj = os.path.basename(os.path.dirname(ds.subjects[si]))
        results.append((subj, mv, heart, full))
        print(f"  {subj}: moving-frac {mv:.4f}  heart-tstd {heart:.4f}  full-tstd {full:.4f}", flush=True)
        if si == 0:
            hi = np.percentile(vols, 99.5); midD = vols.shape[1] // 2
            fr = [Image.fromarray((np.clip(vols[q, midD] / hi, 0, 1) * 255).astype(np.uint8), "L").resize((512, 512))
                  for q in range(T)]
            fr[0].save(f"{OUTDIR}/INDIST_{subj}_beating.gif", save_all=True, append_images=fr[1:], duration=120, loop=0)

    mv = np.mean([r[1] for r in results]); ht = np.mean([r[2] for r in results])
    print(f"\n=== IN-DIST (CMRxRecon val, n={len(results)}): moving-frac {mv:.4f}  heart-tstd {ht:.4f} ===", flush=True)
    print("=== OOD (Göttingen): moving-frac ~0.002  heart-tstd ~0.006 (from earlier) ===", flush=True)
    print(f"=== in-dist / OOD motion ratio: moving-frac {mv/0.002:.1f}x  heart-tstd {ht/0.006:.1f}x ===", flush=True)


if __name__ == "__main__":
    main()
