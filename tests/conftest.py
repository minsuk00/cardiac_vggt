import os
import sys

import nibabel as nib
import numpy as np
import pytest
from omegaconf import OmegaConf

# Make training/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: heavier tests (e.g. full-model construction)")

# Synthetic native shape/spacing. T=12 to match the canonical pipeline's
# NUM_PHASES. Spatial shape kept small for fast tests; it resamples into the
# fixed canonical (256, 256, 12) cube, occupying a centered sub-region.
SYN_W, SYN_H, SYN_Z, SYN_T = 64, 60, 8, 12
SYN_SPACING = (1.4, 1.4, 8.0)


@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory):
    """
    Creates a minimal Cine_combined-like directory:
      Train_P001/sax/3d_recon/sax_frame_{00..11}.nii.gz  — shape (64, 60, 8), spacing (1.4, 1.4, 8.0)
      Val_P001/sax/...  — same structure
      splits/default.txt       — Train_P001 in [train], Val_P001 in [val]
      splits/overfit_p001.txt  — Train_P001 in both [train] and [val]
    """
    root = tmp_path_factory.mktemp("Cine_combined")
    affine = np.diag([*SYN_SPACING, 1.0])

    for subj in ["Train_P001", "Val_P001"]:
        recon = root / subj / "sax" / "3d_recon"
        recon.mkdir(parents=True)
        rng = np.random.RandomState(42)
        for t in range(SYN_T):
            vol = (rng.rand(SYN_W, SYN_H, SYN_Z) * 1000).astype(np.float32)
            nib.save(nib.Nifti1Image(vol, affine), str(recon / f"sax_frame_{t:02d}.nii.gz"))

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
def monai_cache_dir(tmp_path_factory):
    """Isolated PersistentDataset cache per test session (don't pollute /tmp shared cache)."""
    return str(tmp_path_factory.mktemp("monai_cache"))


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
def train_ds(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    return MRIDataset(
        common_conf, synthetic_root,
        split="train", split_file=split_file,
        mode="dynamic", mri_mode="axial",
        num_slices=12, target_size=518,
        cache_dir=monai_cache_dir,
    )
