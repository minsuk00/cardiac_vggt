import glob
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


class MRIProcessor:
    def __init__(self, prefix="img_t"):
        """
        Generic MRI processor for handling multiple subjects and slice orientations.
        Automatically manages subject metadata and intensity normalization.
        """
        self.prefix = prefix
        self.subject_cache = {}  # {abs_path: {stats, t_total}}

    def _get_subject_info(self, data_dir):
        """Automated detection of sequence metadata and global stats from Frame 1."""
        abs_dir = os.path.abspath(data_dir)
        if abs_dir in self.subject_cache:
            return self.subject_cache[abs_dir]

        pattern = os.path.join(abs_dir, f"{self.prefix}*.nii.gz")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {abs_dir} with prefix {self.prefix}")

        # Intensity stats from the first volume only
        img = nib.load(files[0])
        data = img.get_fdata()

        self.subject_cache[abs_dir] = {"stats": (np.min(data), np.max(data)), "t_total": len(files)}
        return self.subject_cache[abs_dir]

    def get_slice(self, data_dir, timestep, mode="axial", idx=None, center=None, angles=None):
        """
        Fetch a normalized 2D slice.
        Returns: (slice_data, slice_index, rotation_angles)
        """
        info = self._get_subject_info(data_dir)
        v_min, v_max = info["stats"]

        file_path = os.path.join(data_dir, f"{self.prefix}{timestep:03d}.nii.gz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        mode = mode.lower()
        if mode in ["axial", "coronal", "sagittal"]:
            return self._get_orthogonal(file_path, timestep, info["t_total"], mode, idx, v_min, v_max)
        elif mode == "oblique":
            return self._get_oblique(file_path, center, angles, v_min, v_max)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _get_orthogonal(self, file_path, timestep, t_total, mode, manual_idx, v_min, v_max):
        """Internal helper for medically-aligned planes (Axial, Coronal, Sagittal)."""
        axis_map = {"sagittal": 0, "coronal": 1, "axial": 2}
        axis = axis_map[mode]

        img = nib.load(file_path)
        total_slices = img.header.get_data_shape()[axis]

        # Use equidistant traversal if no manual index provided
        idx = manual_idx if manual_idx is not None else int(np.round((timestep - 1) * (total_slices - 1) / (t_total - 1)))

        # Memory-efficient sampling via nibabel ArrayProxy
        if axis == 0:
            raw = np.array(img.dataobj[idx, :, :])
        elif axis == 1:
            raw = np.array(img.dataobj[:, idx, :])
        else:
            raw = np.array(img.dataobj[:, :, idx])

        return self._normalize(raw, v_min, v_max), idx, None

    def _get_oblique(self, file_path, center, angles, v_min, v_max, output_shape=(256, 256)):
        """Internal helper for 3D interpolation at arbitrary angles."""
        if center is None:
            raise ValueError("Oblique mode requires a 3D center tuple (x, y, z)")

        img = nib.load(file_path)
        data = img.get_fdata()  # Full load required for interpolation

        if angles is None:
            angles = tuple(np.random.uniform(0, 360, 3).round(1))

        rot = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        h, w = output_shape
        grid_x, grid_y = np.meshgrid(np.arange(-h // 2, h // 2), np.arange(-w // 2, w // 2), indexing="ij")

        # Combine grid, rotate, and translate to the heart center
        coords = rot @ np.stack([grid_x.flatten(), grid_y.flatten(), np.zeros_like(grid_x).flatten()])
        for i, c in enumerate(center):
            coords[i, :] += c

        # Interpolate sampled points from the 3D volume
        sampled = map_coordinates(data, coords, order=1, mode="constant", cval=v_min)
        return self._normalize(sampled.reshape(output_shape), v_min, v_max), None, angles

    def _normalize(self, data, v_min, v_max):
        """Helper to rescale raw data to [0, 1] range."""
        return (data - v_min) / (v_max - v_min + 1e-8)


if __name__ == "__main__":
    DATA_DIR = "./scratch/nifti/card_3.125x3.125x3.125mm_256x256x70x24x4_snr20_fa60_bh"
    mri = MRIProcessor()
    # Comprehensive Self-Test
    s, idx, ang = mri.get_slice(DATA_DIR, 1, mode="axial")
    print(f"Self-test: Mode Axial -> Shape {s.shape}, Index {idx}")
    s, idx, ang = mri.get_slice(DATA_DIR, 1, mode="oblique", center=(127, 148, 34))
    print(f"Self-test: Mode Oblique -> Shape {s.shape}, Angles {ang}")
