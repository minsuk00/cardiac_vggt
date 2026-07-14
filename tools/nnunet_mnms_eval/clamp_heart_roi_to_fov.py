"""One-off data fix: clamp every CMRx canonical heart ROI to the native FOV.

WHY: `heart_roi_canonical.nii.gz` is the whole-heart seg dilated +6mm in-plane and +-1 z-plane
(build_heart_roi.py). On the CANONICAL (256,256,12) grid, subjects with <12 acquired slices are
zero-padded, and the +-1 z dilation spills the ROI onto padding planes that have NO acquired data
(GT is all zeros there). Scoring a reconstruction inside those planes is meaningless and drags the
metric (e.g. Train_P053 clean 20.2->27.5 dB once its empty z0 is dropped). Only CMRx has canonical
ROIs; native-space ROIs (ACDC/MIITT/OCMR/goettingen) aren't padded, so they don't spill.

FIX: overwrite each heart_roi_canonical.nii.gz with (roi & content_mask), where content_mask is the
canonical native-FOV mask that MRIDataset already computes (1=real data, 0=zero-pad) -- i.e. exactly
`build_roi(..., fov_mask=content)`. Keeps a heart_roi_canonical_prefov.nii.gz backup (idempotent:
always clamps the ORIGINAL, so re-running is safe). Read by training/loss.py (val diagnostic).

Run:  micromamba run -n svr python tools/nnunet_mnms_eval/clamp_heart_roi_to_fov.py [--limit N] [--subjects Name ...]
"""
import argparse, os, shutil, sys
import numpy as np
import nibabel as nib

VGGT = "/home/minsukc/vggt"
sys.path.insert(0, os.path.join(VGGT, "training"))
sys.path.insert(0, VGGT)
from data.datasets.mri_dataset import MRIDataset  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DATA_ROOT = f"{VGGT}/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = f"{VGGT}/training/splits/random_8_1_1.txt"
COMMON = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                           "rescale_aug": False, "landscape_check": False,
                           "augs": {"scales": [1.0, 1.0]}})


def content_mask_xyz(ds, idx):
    """(X,Y,Z) bool native-FOV mask for subject idx via MRIDataset (matches training)."""
    data = ds.get_data(seq_index=idx, img_per_seq=1)
    cm = np.asarray(data["content_mask"]).astype(bool)      # (D,H,W) = (Z,Y,X) splat order
    return np.transpose(cm, (2, 1, 0))                      # -> (X,Y,Z) = on-disk ROI order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N subjects (smoke test)")
    ap.add_argument("--subjects", nargs="*", default=None, help="restrict to these subject names")
    ap.add_argument("--dry_run", action="store_true", help="report only, do not overwrite")
    args = ap.parse_args()

    n_done = n_changed = n_missing = 0
    for split in ("train", "val", "test"):
        ds = MRIDataset(COMMON, DATA_ROOT, split=split, split_file=SPLIT_FILE, mode="dynamic",
                        mri_mode="axial", num_slices=1, target_size=518, t_target_fixed=0)
        for idx, subject_path in enumerate(ds.subjects):
            name = os.path.basename(os.path.dirname(subject_path))
            if args.subjects and name not in args.subjects:
                continue
            roi_path = os.path.join(DATA_ROOT, name, "sax", "heart_roi_canonical.nii.gz")
            if not os.path.exists(roi_path):
                n_missing += 1
                continue
            backup = os.path.join(DATA_ROOT, name, "sax", "heart_roi_canonical_prefov.nii.gz")
            # always clamp the ORIGINAL (idempotent): source = backup if it exists else current
            src = backup if os.path.exists(backup) else roi_path
            img = nib.load(src)
            roi0 = np.asarray(img.dataobj) > 0.5
            content = content_mask_xyz(ds, idx)
            new = (roi0 & content).astype(np.uint8)
            dropped = int(roi0.sum() - new.sum())
            dropped_planes = [z for z in range(roi0.shape[2])
                              if roi0[:, :, z].any() and not new[:, :, z].any()]
            tag = f"  {name:16s} roi {int(roi0.sum()):6d} -> {int(new.sum()):6d}  dropped {dropped:5d}"
            if dropped_planes:
                tag += f"  empty planes {dropped_planes}"
            if dropped > 0:
                n_changed += 1
            if not args.dry_run:
                if not os.path.exists(backup):
                    shutil.copyfile(roi_path, backup)                 # preserve original once
                nib.save(nib.Nifti1Image(new, img.affine), roi_path)
            print(tag + ("   [dry]" if args.dry_run else ""))
            n_done += 1
            if args.limit and n_done >= args.limit:
                break
        if args.limit and n_done >= args.limit:
            break
    print(f"\n{'DRY-RUN ' if args.dry_run else ''}done: {n_done} ROIs, {n_changed} changed "
          f"(had FOV spill), {n_missing} missing.")


if __name__ == "__main__":
    main()
