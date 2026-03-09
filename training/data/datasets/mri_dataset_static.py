import glob
import logging
import os
import random

import cv2
import nibabel as nib
import numpy as np
import torch
from data.base_dataset import BaseDataset
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


class MRIDataset(BaseDataset):
    def __init__(self, common_conf, data_root, split="train", mode="static", num_slices=5, target_size=518, mri_mode="mixed", **kwargs):
        """
        MRI Dataset for VGGT Fine-tuning.
        mri_mode:
            'axial': Standard axial stack traversal.
            'mixed': Random mix of Axial, Coronal, and Sagittal views in one sequence.
            'oblique': Arbitrary euler rotations centered on the heart.
        """
        super().__init__(common_conf=common_conf)

        self.data_root = os.path.abspath(data_root)
        logging.info(f"MRIDataset: data_root = {self.data_root}")
        self.mode, self.num_slices, self.target_size, self.mri_mode = mode, num_slices, target_size, mri_mode
        self.subjects = self._find_subjects()
        logging.info(f"MRIDataset: Found {len(self.subjects)} subjects.")
        self.len_train = 1000
        # Typical heart center for this dataset (can be randomized slightly if needed)
        self.heart_center = (127, 148, 34)

    def _find_subjects(self):
        if not os.path.isdir(self.data_root):
            return []
        if glob.glob(os.path.join(self.data_root, "img_t*.nii.gz")):
            return [self.data_root]
        return sorted(
            [os.path.join(self.data_root, d) for d in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, d)) and glob.glob(os.path.join(self.data_root, d, "img_t*.nii.gz"))]
        )

    def __len__(self):
        return self.len_train

    def get_data(self, seq_index=0, img_per_seq=None, **kwargs):
        S = max(2, img_per_seq or self.num_slices)
        sub_dir = self.subjects[seq_index % len(self.subjects)]

        # Load Volume (Phase 1/Static: use first timestep)
        nii_files = sorted(glob.glob(os.path.join(sub_dir, "img_t*.nii.gz")))
        nii_file = nii_files[0]
        img_obj = nib.load(nii_file)
        vol = img_obj.get_fdata()
        v_min, v_max = np.min(vol), np.max(vol)
        W, H, Z = vol.shape

        res = {k: [] for k in ["images", "world_points", "cam_points", "point_masks", "depths", "extrinsics", "intrinsics", "original_sizes", "frame_ids"]}

        for i in range(S):
            if self.mri_mode == "axial":
                axis = "axial"
                idx = int(np.round((Z - 1) * (i / (max(1, S - 1)))))
            elif self.mri_mode == "mixed":
                axis = random.choice(["axial", "coronal", "sagittal"])
                lim = {"axial": Z, "coronal": H, "sagittal": W}[axis]
                idx = random.randint(0, lim - 1)
            elif self.mri_mode == "oblique":
                axis = "oblique"
                idx = i
            else:
                axis = "axial"
                idx = i

            # --- 1. Extract Slice and set 3D Coords ---
            if axis == "axial":
                raw = vol[:, :, idx]
                r, c = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
                coords = np.stack([r, c, np.full_like(r, idx)], axis=-1)
            elif axis == "coronal":
                raw = vol[:, idx, :]
                r, c = np.meshgrid(np.arange(W), np.arange(Z), indexing="ij")
                coords = np.stack([r, np.full_like(r, idx), c], axis=-1)
            elif axis == "sagittal":
                raw = vol[idx, :, :]
                r, c = np.meshgrid(np.arange(H), np.arange(Z), indexing="ij")
                coords = np.stack([np.full_like(r, idx), r, c], axis=-1)
            elif axis == "oblique":
                # Sample at 256x256 resolution
                out_h, out_w = 256, 256
                angles = tuple(np.random.uniform(0, 360, 3))
                rot = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
                grid_x, grid_y = np.meshgrid(np.arange(-out_h // 2, out_h // 2), np.arange(-out_w // 2, out_w // 2), indexing="ij")

                # Rotate and translate to heart center
                flat_coords = rot @ np.stack([grid_x.flatten(), grid_y.flatten(), np.zeros_like(grid_x).flatten()])
                for j, c_val in enumerate(self.heart_center):
                    flat_coords[j, :] += c_val

                # Sample intensity
                sampled = map_coordinates(vol, flat_coords, order=1, mode="constant", cval=v_min)
                raw = sampled.reshape((out_h, out_w))
                # Reshape coords for world_points
                coords = flat_coords.T.reshape((out_h, out_w, 3))

            # --- 2. Normalize ---
            # Intensity [0, 255] - ComposedDataset will divide by 255
            img = np.repeat(((raw - v_min) / (v_max - v_min + 1e-8))[..., None], 3, -1).astype(np.float32)
            img = np.clip(img * 255.0, 0, 255)
            # World Positions [-1, 1]
            wp = coords.copy().astype(np.float32)
            wp[..., 0] = (wp[..., 0] / (W - 1)) * 2 - 1
            wp[..., 1] = (wp[..., 1] / (H - 1)) * 2 - 1
            wp[..., 2] = (wp[..., 2] / (Z - 1)) * 2 - 1

            # --- 3. Resize and Pad to target_size ---
            h, w = raw.shape
            sc = self.target_size / max(h, w)
            nw, nh = int(w * sc), int(h * sc)
            img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
            wp_r = cv2.resize(wp, (nw, nh), interpolation=cv2.INTER_LINEAR)

            # Volume Bounds Mask: Only True for pixels that map INSIDE the 3D volume grid.
            # coords has shape (H, W, 3) or (out_h, out_w, 3)
            vol_mask = (coords[..., 0] >= 0) & (coords[..., 0] < W) & (coords[..., 1] >= 0) & (coords[..., 1] < H) & (coords[..., 2] >= 0) & (coords[..., 2] < Z)
            vol_mask_r = cv2.resize(vol_mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0.5

            hp, wp_p = self.target_size - nh, self.target_size - nw
            pt, pb, pl, pr = hp // 2, hp - hp // 2, wp_p // 2, wp_p - wp_p // 2

            # Clip to [0, 255] to handle cv2.INTER_CUBIC overshoot
            res["images"].append(np.clip(np.pad(img_r, ((pt, pb), (pl, pr), (0, 0))), 0, 255))
            res["world_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=-2.0))
            res["cam_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=-2.0))

            # Final point_mask: must be inside volume AND not in padding
            m = np.zeros((self.target_size, self.target_size), bool)
            m[pt : pt + nh, pl : pl + nw] = vol_mask_r
            res["point_masks"].append(m)

            # Metadata
            res["depths"].append(np.zeros((self.target_size, self.target_size), np.float32))
            res["extrinsics"].append(np.eye(3, 4, dtype=np.float32))
            res["intrinsics"].append(np.array([[self.target_size, 0, self.target_size / 2], [0, self.target_size, self.target_size / 2], [0, 0, 1]], np.float32))
            res["original_sizes"].append(np.array([h, w], np.float32))
            res["frame_ids"].append(idx)

        return {
            **res,
            "seq_name": f"mri_{self.mri_mode}_{os.path.basename(sub_dir)}",
            "ids": np.array(res["frame_ids"], np.int64),
            "frame_num": S,
            "tracks": np.zeros((1, 1, 2), np.float32),
        }
