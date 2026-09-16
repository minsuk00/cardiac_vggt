from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from torch.utils.data import Dataset

from .augment import GRID, augment
from .config import Config


def preprocess(volume, config=Config(), order=1):
    """NIfTI (X,Y,Z) -> (Z,Y,X), on an explicitly recorded physical grid.

    Resample in plane, then integer centre crop/pad. No slice-axis resampling.
    The returned affine maps output NIfTI voxel indices to physical coordinates.
    """
    if len(volume.shape) != 3:
        raise ValueError("Expected a 3D volume; select a cine frame first")
    spacing = np.linalg.norm(volume.affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Invalid voxel spacing")
    shape = np.asarray(volume.shape[:2])
    new_shape = np.maximum(1, np.rint(shape * spacing[:2] / config.spacing_mm).astype(int))
    start = np.where(new_shape >= config.size, (new_shape - config.size) // 2,
                     -((config.size - new_shape) // 2))
    ratio = config.spacing_mm / spacing[:2]
    transform = np.eye(4)
    transform[0, 0], transform[1, 1] = ratio
    transform[:2, 3] = start * ratio
    values = volume.get_fdata(dtype=np.float32)
    output = np.empty((volume.shape[2], config.size, config.size), np.float32)
    # Enforce the intermediate resampled extent when padding.
    coords = np.arange(config.size)
    valid_x = (coords + start[0] >= 0) & (coords + start[0] < new_shape[0])
    valid_y = (coords + start[1] >= 0) & (coords + start[1] < new_shape[1])
    for z in range(volume.shape[2]):
        plane = affine_transform(values[:, :, z], np.diag(ratio),
                                 offset=start * ratio,
                                 output_shape=(config.size, config.size),
                                 order=order, mode="constant", cval=0, prefilter=False)
        plane[~valid_x, :] = 0
        plane[:, ~valid_y] = 0
        output[z] = plane.T
    return output, volume.affine @ transform


def normalize(slices):
    """Assumption: independent slice min/max normalization; constant slices -> 0."""
    low = slices.min(axis=(-2, -1), keepdims=True)
    span = slices.max(axis=(-2, -1), keepdims=True) - low
    return np.divide(slices - low, span, out=np.zeros_like(slices), where=span > 0)


def centers_from_labels(labels, minimum=1):
    """Centroid of label 3, then nearest valid slice propagation for empty slices.

    Ties favor the valid slice closer to stack middle. A larger minimum is an
    optional ACDC adaptation; an entirely invalid stack raises instead of guessing.
    """
    centers = np.full((len(labels), 2), np.nan, np.float32)
    valid = np.zeros(len(labels), bool)
    for z, plane in enumerate(labels):
        y, x = np.nonzero(plane == 3)
        if len(x) >= minimum:
            centers[z] = [x.mean(), y.mean()]
            valid[z] = True
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise ValueError("No valid LV blood-pool centre in this stack")
    for z in np.flatnonzero(~valid):
        nearest = min(candidates, key=lambda k: (abs(k-z), abs(k-(len(labels)-1)/2)))
        centers[z] = centers[nearest]
    return centers, valid


def read_case(image_path, config=Config()):
    image_path = Path(image_path)
    label_path = image_path.with_name(image_path.name.replace('.nii.gz', '_gt.nii.gz'))
    image, labels = nib.load(image_path), nib.load(label_path)
    if image.shape != labels.shape or not np.allclose(image.affine, labels.affine):
        raise ValueError(f"Image/label geometry mismatch: {image_path}")
    slices, affine = preprocess(image, config)
    masks, _ = preprocess(labels, config, order=0)
    centers, valid = centers_from_labels(masks, config.min_bp_pixels)
    return normalize(slices), centers, valid, affine


def make_split(root, seed=2018):
    """80/10/10 patient split, stratified by ACDC Group; no frame leakage."""
    groups = {}
    for patient in sorted((Path(root) / 'training').glob('patient*')):
        info = dict(line.split(':', 1) for line in (patient / 'Info.cfg').read_text().splitlines() if ':' in line)
        groups.setdefault(info['Group'].strip(), []).append(patient.name)
    if sum(map(len, groups.values())) != 100:
        raise ValueError("Expected 100 ACDC training patients")
    rng = np.random.default_rng(seed)
    split = {key: [] for key in ('train', 'val', 'test')}
    for patients in groups.values():
        rng.shuffle(patients)
        nval = ntest = round(len(patients) * .1)
        split['val'].extend(patients[:nval])
        split['test'].extend(patients[nval:nval+ntest])
        split['train'].extend(patients[nval+ntest:])
    return {key: sorted(value) for key, value in split.items()}


def case_paths(root, patients, subset='training'):
    for patient in patients:
        for path in sorted((Path(root) / subset / patient).glob('*_frame*.nii.gz')):
            if not path.name.endswith('_gt.nii.gz'):
                yield path


class ACDCSlices(Dataset):
    def __init__(self, paths, config=Config(), augmented=False):
        images, centers = [], []
        for path in paths:
            x, y, _, _ = read_case(path, config)
            images.append(x)
            centers.append(y)
        if not images:
            raise ValueError("No labelled frame images found")
        self.images = np.concatenate(images)
        self.centers = np.concatenate(centers)
        self.augmented = augmented

    def __len__(self):
        return len(self.images) * (len(GRID) if self.augmented else 1)

    def __getitem__(self, index):
        source, transform = divmod(index, len(GRID)) if self.augmented else (index, 0)
        image, center = self.images[source], self.centers[source]
        if self.augmented:
            image, center = augment(image, center, transform)
        return image[None].copy(), center.copy()
