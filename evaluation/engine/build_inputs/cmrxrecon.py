"""Smoke-test step 1/3 — build clean + breathing-corrupted per-phase SAX stacks for ONE
CMRxRecon val subject, plus GT + mask + a breathing manifest, on the canonical grid.

Everything downstream (SVRTK recon, scoring, GIF) reads these frozen files. The breathing
is the trainer's OWN sim (training/data/respiratory.py), deterministic per subject via a
stable name-hash seed (not the positional seq_index -> robust to split reordering).

Run: micromamba run -n svr python scratch/eval/cmrxrecon/build_inputs.py [subject_idx]
"""
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import nibabel as nib
import torch
from omegaconf import OmegaConf

VGGT = "/home/minsukc/vggt"
sys.path.insert(0, os.path.join(VGGT, "training"))
sys.path.insert(0, VGGT)

from data.datasets.mri_dataset import MRIDataset  # noqa: E402
from data.respiratory import RespiratoryConfig, sample_resp_disp, reslice_volume_vec  # noqa: E402

DATA_ROOT = f"{VGGT}/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = f"{VGGT}/training/splits/random_8_1_1.txt"
OUT_ROOT = f"{VGGT}/scratch/eval/cmrxrecon/out"
SPACING_XYZ = (1.4, 1.4, 12.0)   # mm, canonical grid (must match preprocess.py)
N_CANON_Z = 12
DATASET = "cmrxrecon"


def _affine(sp):
    return np.diag([sp[0], sp[1], sp[2], 1.0])


def name_seed(dataset, name):
    """Stable, split-order-robust breath seed = hash of '<dataset>/<name>'."""
    return int(hashlib.sha256(f"{dataset}/{name}".encode()).hexdigest(), 16) % (2 ** 31)


def build_respiratory_config():
    """Load RespiratoryConfig from the LIVE mri_volume.yaml (matches training)."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver(
        "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    cfgdir = os.path.join(VGGT, "training", "config")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="mri_volume")
    return RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory), cfg


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rcfg, cfg = build_respiratory_config()
    rcfg.per_slot = False   # per-SUBJECT breath amplitude (match training); phase stays per-slice
    print(f"RespiratoryConfig: enable={rcfg.enable} amp={rcfg.amplitude_mm} jit={rcfg.amplitude_jitter} "
          f"ap_ratio={rcfg.ap_ratio} group_by_burst={rcfg.group_by_burst} "
          f"tilt=({rcfg.tilt_min_deg},{rcfg.tilt_max_deg}) dir_jit={rcfg.direction_jitter_deg}", flush=True)
    assert rcfg.enable, "respiratory sim disabled in mri_volume.yaml?!"

    common = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False, "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=1, target_size=518,
                    t_target_fixed=0)
    subject_path = ds.subjects[idx % len(ds.subjects)]
    name = os.path.basename(os.path.dirname(subject_path))
    data = ds.get_data(seq_index=idx, img_per_seq=1)

    phases = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32))  # (T,D,H,W) splat order
    content_mask = np.asarray(data["content_mask"]).astype(np.uint8)          # (D,H,W)
    T, D, H, W = phases.shape
    assert D == N_CANON_Z, (D, N_CANON_Z)
    affine = _affine(SPACING_XYZ)

    # Deterministic breathing (one realization per subject; name-hash seed).
    seed = name_seed(DATASET, name)
    seq_index = torch.tensor([[seed]], dtype=torch.int64)
    disp, r = sample_resp_disp(1, N_CANON_Z, rcfg, "cpu", train=False, seq_index=seq_index)
    disp0 = disp[0]        # (12,3) mm (d_D,d_H,d_W) per z-plane
    r0 = r[0]              # (12,)
    mean_abs = disp0.norm(dim=-1).mean().item()
    print(f"[{idx}] {name}  T={T} D={D} H={H} W={W}  seed={seed}  mean|disp|={mean_abs:.3f}mm  "
          f"max|disp|={disp0.norm(dim=-1).max().item():.3f}mm", flush=True)

    subj_dir = os.path.join(OUT_ROOT, name)
    for sub in ("gt", "clean", "breath"):
        os.makedirs(os.path.join(subj_dir, sub), exist_ok=True)

    def save_xyz(dhw, path):
        nib.save(nib.Nifti1Image(np.ascontiguousarray(dhw.transpose(2, 1, 0)), affine), path)

    # mask (unshifted for both variants — acquisition geometry, not anatomy)
    save_xyz(content_mask.astype(np.float32), os.path.join(subj_dir, "mask.nii.gz"))
    # SVRTK recon ROI + per-phase heart seg: use the OFFICIAL GT whole-heart segmentation that
    # already sits beside the data on the canonical grid (do NOT re-segment). heart_roi_canonical
    # = binary whole-heart ROI (256,256,12); heart_seg_canonical = per-phase LV/MYO/RV (256,256,12,T).
    sax = os.path.join(DATA_ROOT, name, "sax")
    shutil.copyfile(os.path.join(sax, "heart_roi_canonical.nii.gz"),
                    os.path.join(subj_dir, "mask_heart.nii.gz"))
    shutil.copyfile(os.path.join(sax, "heart_seg_canonical.nii.gz"),
                    os.path.join(subj_dir, "heart_seg.nii.gz"))

    for t in range(T):
        Vt = phases[t]  # (D,H,W)
        # GT (clean, unshifted)
        save_xyz(np.asarray(Vt), os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz"))
        # clean stack == GT planes (the SVR upper-bound: nothing to correct)
        save_xyz(np.asarray(Vt), os.path.join(subj_dir, "clean", f"stack_t{t:02d}.nii.gz"))
        # breathing stack: each plane z resliced by its own disp[z], keep that plane
        breathed = torch.stack(
            [reslice_volume_vec(Vt, disp0[z])[z] for z in range(D)], dim=0)  # (D,H,W)
        save_xyz(breathed.numpy(), os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz"))

    manifest = {
        "dataset": DATASET, "subject": name, "subject_idx": idx, "seed": seed,
        "T": T, "D": D, "H": H, "W": W, "spacing_xyz_mm": list(SPACING_XYZ),
        "content_mask_frac": float(content_mask.mean()),
        "breath": {
            "mean_abs_disp_mm": mean_abs,
            "disp_dhw_mm": disp0.tolist(),   # (12,3) per z-plane
            "r_per_plane": r0.tolist(),
            "amplitude_mm": rcfg.amplitude_mm, "amplitude_jitter": rcfg.amplitude_jitter,
            "ap_ratio": rcfg.ap_ratio, "cos2n": rcfg.cos2n,
            "group_by_burst": rcfg.group_by_burst,
            "tilt_min_deg": rcfg.tilt_min_deg, "tilt_max_deg": rcfg.tilt_max_deg,
            "direction_jitter_deg": rcfg.direction_jitter_deg,
        },
    }
    json.dump(manifest, open(os.path.join(subj_dir, "manifest.json"), "w"), indent=2)
    print(f"done -> {subj_dir}  (gt/clean/breath x {T} phases + mask + manifest.json)", flush=True)


if __name__ == "__main__":
    main()
