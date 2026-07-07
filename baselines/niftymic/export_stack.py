"""Export a clean, real per-phase SAX stack (+ body mask) as NIfTI for NiftyMIC.

NiftyMIC is a classical slice-to-volume-registration + super-resolution toolkit
(fetal-brain native, applied off-label here). Per docs/24 (_html/24_svr_baselines.html
section 4/5), the fair way to feed it our data is:

    - take the REAL, already-aligned per-phase stack from the canonical cache
      (`phases[t_target]`, D=12, H=256, W=256 at 1.4x1.4x12.0mm) — NOT our
      synthetic scattered/respiratory-corrupted VGGT-MRI training input. This IS
      the same tensor our own `V_gt` uses, so it's the fair "clean input" upper
      bound for a classical-SVR comparator (see docs/24 sec 3, DMCVR card).
    - NiftyMIC requires masks (--filenames-masks is mandatory), so we pass the
      canonical `content_mask` (native-FOV vs zero-pad) as the ROI mask.

Output per subject: <OUT_DIR>/<subject>_t<phase>_stack.nii.gz
                     <OUT_DIR>/<subject>_t<phase>_mask.nii.gz

Axis-order: cache tensors are in splat order (D=Z, H=Y, W=X); NIfTI/nibabel
wants (X, Y, Z), so we transpose(2, 1, 0) — the exact reverse of the
permute(0, 3, 2, 1) documented in MRIDataset.get_data (CLAUDE.md "Axis-order
gotcha"). Affine is a simple diagonal at the canonical (1.4, 1.4, 12.0) mm
spacing — this is a synthetic canonical grid, not a real-world RAS frame, so
only the spacing (not orientation) needs to be physically meaningful for
NiftyMIC's registration/SR math.
"""
import os
import sys

import nibabel as nib
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.datasets.mri_dataset import MRIDataset  # noqa: E402

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
OUT_DIR = "/home/minsukc/vggt/scratch/niftymic/data"
TARGET_PHASE = 0  # ED, matches the project's established ED-only convention
SUBJECT_INDICES = [0, 1]  # "a couple" of val subjects
SPACING_XYZ = (1.4, 1.4, 12.0)  # mm, must match TARGET_SPACING in preprocess.py


def _affine(spacing_xyz):
    return np.diag([spacing_xyz[0], spacing_xyz[1], spacing_xyz[2], 1.0])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(
        common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
        mode="dynamic", mri_mode="axial", num_slices=1, target_size=518,
        t_target_fixed=TARGET_PHASE,
    )
    affine = _affine(SPACING_XYZ)

    for idx in SUBJECT_INDICES:
        subject_path = ds.subjects[idx % len(ds.subjects)]
        name = os.path.basename(os.path.dirname(subject_path))
        data = ds.get_data(seq_index=idx, img_per_seq=1)

        phases = data["phases"]          # (T, D, H, W) float16, splat order
        content_mask = data["content_mask"]  # (D, H, W) uint8
        t_target = int(data["t_target"][0])
        assert t_target == TARGET_PHASE, (t_target, TARGET_PHASE)

        stack_dhw = np.asarray(phases[t_target]).astype(np.float32)  # (D, H, W)
        mask_dhw = np.asarray(content_mask).astype(np.uint8)         # (D, H, W)

        # splat (D=Z, H=Y, W=X) -> nibabel (X, Y, Z)
        stack_xyz = stack_dhw.transpose(2, 1, 0)
        mask_xyz = mask_dhw.transpose(2, 1, 0)

        stack_path = os.path.join(OUT_DIR, f"{name}_t{t_target}_stack.nii.gz")
        mask_path = os.path.join(OUT_DIR, f"{name}_t{t_target}_mask.nii.gz")
        nib.save(nib.Nifti1Image(stack_xyz, affine), stack_path)
        nib.save(nib.Nifti1Image(mask_xyz, affine), mask_path)

        print(f"[{idx}] {name}  t={t_target}  stack={stack_xyz.shape} "
              f"nonzero_mask_frac={mask_xyz.mean():.3f}")
        print(f"     -> {stack_path}")
        print(f"     -> {mask_path}")


if __name__ == "__main__":
    main()
