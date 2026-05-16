import os
import sys

import nibabel as nib
import numpy as np
import pytest
from omegaconf import OmegaConf

# Make training/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    """
    Creates a minimal Cine_combined-like directory:
      Train_P001/sax/3d_recon/sax_frame_{00,01,02}.nii.gz  — shape (32, 30, 4), spacing (1.344, 1.398, 8.0)
      Train_P001/sax/dvf_elastix/dvf_frame_{01,02}.nii.gz  — constant DVF [2.0, 1.5, 4.0] mm
      Val_P001/sax/...  — same structure
      splits/default.txt  — Train_P001 in [train], Val_P001 in [val]
      splits/overfit_p001.txt  — Train_P001 in both [train] and [val]
    """
    root = tmp_path_factory.mktemp("Cine_combined")
    W, H, Z, T = 32, 30, 4, 3
    spacing = (1.344, 1.398, 8.0)
    affine = np.diag([*spacing, 1.0])

    for subj in ["Train_P001", "Val_P001"]:
        recon = root / subj / "sax" / "3d_recon"
        dvf_d = root / subj / "sax" / "dvf_elastix"
        recon.mkdir(parents=True)
        dvf_d.mkdir(parents=True)

        rng = np.random.RandomState(42)
        for t in range(T):
            vol = (rng.rand(W, H, Z) * 1000).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, affine), str(recon / f"sax_frame_{t:02d}.nii.gz"))

        # Known constant DVF: [2.0, 1.5, 4.0] mm everywhere
        dvf = np.full((W, H, Z, 1, 3), [2.0, 1.5, 4.0], dtype=np.float32)
        for t in range(1, T):
            nib.save(nib.Nifti1Image(dvf, affine), str(dvf_d / f"dvf_frame_{t:02d}.nii.gz"))

    # Split files
    splits_dir = root / "splits"
    splits_dir.mkdir()
    (splits_dir / "default.txt").write_text(
        "# test split\n[train]\nTrain_P001\n\n[val]\nVal_P001\n\n[test]\n"
    )
    (splits_dir / "overfit_p001.txt").write_text(
        "[train]\nTrain_P001\n\n[val]\nTrain_P001\n\n[test]\nTrain_P001\n"
    )

    return str(root)


@pytest.fixture(scope="module")
def split_file(synthetic_root):
    return os.path.join(synthetic_root, "splits", "default.txt")


@pytest.fixture(scope="module")
def overfit_split_file(synthetic_root):
    return os.path.join(synthetic_root, "splits", "overfit_p001.txt")


@pytest.fixture(scope="module")
def common_conf():
    return OmegaConf.create({
        "img_size": 518,
        "patch_size": 14,
        "rescale": True,
        "rescale_aug": False,
        "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })


@pytest.fixture(scope="module")
def train_ds(synthetic_root, split_file, common_conf):
    from data.datasets.mri_dataset import MRIDataset
    return MRIDataset(
        common_conf, synthetic_root,
        split="train", split_file=split_file,
        mode="dynamic", mri_mode="axial",
        num_slices=3, target_size=518,
    )
