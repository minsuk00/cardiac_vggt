"""Train Dangi Stage A on the VGGT pooled cohort (all 12 cardiac phases, nnU-Net LV centroids).

Reads `training/splits/pooled_curated_v2.txt`-style split files (`[train]`/`[val]`/`[test]`
sections, one `<source_dir>/<subject>` per line under a data root). Each subject supplies
`sax/3d_recon/sax_frame_{t:02d}.nii.gz` (native grid) and `sax/heart_seg.nii.gz` (4D
nnU-Net segmentation on the same grid; LV cavity = label 1).

Differences from the ACDC loader (data.py): targets are nnU-Net pseudo-label centroids over
every phase rather than expert ED/ES labels, and an epoch draws ONE random transform of the
189-grid per slice (the paper's "one random augmentation per volume per epoch" scheme)
instead of exhausting the grid.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import GRID, augment
from .config import Config
from .data import centers_from_labels, normalize, preprocess

LV_LABEL = 1  # nnU-Net M&Ms convention: 1 = LV cavity, 2 = myocardium, 3 = RV


def read_split(split_file, section):
    subjects, current = [], None
    for line in Path(split_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            current = line[1:-1].lower()
        elif current == section.lower():
            subjects.append(line)
    if not subjects:
        raise ValueError(f'No subjects in section [{section}] of {split_file}')
    return subjects


def read_subject(sax_dir, config=Config(), lv_label=LV_LABEL, num_phases=12):
    """All phases of one subject -> normalized (T*Z,192,192) slices + (T*Z,2) centres."""
    sax_dir = Path(sax_dir)
    seg = nib.load(sax_dir / 'heart_seg.nii.gz')
    if len(seg.shape) != 4 or seg.shape[3] != num_phases:
        raise ValueError(f'{sax_dir}: expected 4D seg with {num_phases} phases, got {seg.shape}')
    seg_data = np.asarray(seg.dataobj)
    slices, centers, valid = [], [], []
    for t in range(num_phases):
        image = nib.load(sax_dir / '3d_recon' / f'sax_frame_{t:02d}.nii.gz')
        if image.shape != seg.shape[:3] or not np.allclose(image.affine, seg.affine, atol=1e-3):
            raise ValueError(f'Image/seg geometry mismatch: {sax_dir} phase {t}')
        x, _ = preprocess(image, config)
        mask, _ = preprocess(nib.Nifti1Image((seg_data[..., t] == lv_label).astype(np.uint8) * 3,
                                             seg.affine), config, order=0)
        c, v = centers_from_labels(mask, config.min_bp_pixels)
        slices.append(normalize(x))
        centers.append(c)
        valid.append(v)
    return np.concatenate(slices), np.concatenate(centers), np.concatenate(valid)


class PoolSlices(Dataset):
    """Every slice of every phase of every subject; one random 189-grid transform per draw.

    Slices are held in RAM as float16 (~5.5 GB for the 628-subject train section). The
    transform index comes from torch's RNG, which the DataLoader reseeds per worker and per
    epoch, so augmentation differs across workers and epochs. `cache_dir` stores one .npz per
    subject so repeated loads skip the NIfTI resampling.
    """

    def __init__(self, data_root, split_file, section, config=Config(), augmented=False,
                 limit=None, cache_dir=None):
        self.subjects = read_split(split_file, section)[:limit]
        images, centers = [], []
        for subject in self.subjects:
            cache = Path(cache_dir) / (subject.replace('/', '__') + '.npz') if cache_dir else None
            if cache and cache.exists():
                loaded = np.load(cache)
                x, y = loaded['images'], loaded['centers']
            else:
                x, y, _ = read_subject(Path(data_root) / subject / 'sax', config)
                x = x.astype(np.float16)
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(cache, images=x, centers=y)
            images.append(x)
            centers.append(y)
        self.images = np.concatenate(images)
        self.centers = np.concatenate(centers)
        self.augmented = augmented

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, center = self.images[index].astype(np.float32), self.centers[index]
        if self.augmented:
            image, center = augment(image, center, int(torch.randint(len(GRID), (1,))))
        return image[None].copy(), center.copy()
