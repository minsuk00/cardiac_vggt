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
    def __init__(self, common_conf, data_root, split="train", mode="static", num_slices=5, target_size=518, mri_mode="mixed"):
        """
        MRI Dataset for VGGT.
        mode: 'static' (Phases 1/2) or 'dynamic' (Phase 3 with DVF)
        mri_mode: 'axial', 'mixed', or 'oblique'
        """
        super().__init__(common_conf=common_conf)
        self.data_root = os.path.abspath(data_root)
        logging.info(f"MRIDataset: data_root = {self.data_root}")
        self.mode, self.num_slices, self.target_size, self.mri_mode, self.split = mode, num_slices, target_size, mri_mode, split
        self.subjects = self._find_subjects()
        logging.info(f"MRIDataset: Found {len(self.subjects)} subjects.")
        self.len_train = 1000
        # self.heart_center = (127, 148, 34)

    def _find_subjects(self):
        search_root = os.path.join(self.data_root, self.split)
        if not os.path.isdir(search_root):
            return []
        # Return all subdirectories under data_root/split
        return sorted([os.path.join(search_root, d) for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))])

    def __len__(self):
        return self.len_train

    def get_data(self, seq_index=0, img_per_seq=None, **kwargs):
        S = max(2, img_per_seq or self.num_slices)
        sub_dir = self.subjects[seq_index % len(self.subjects)]

        # Determine paths (Images are in sub_dir/nifti/*/img_t*.nii.gz)
        nii_pattern = os.path.join(sub_dir, "nifti", "*", "img_t*.nii.gz")
        nii_files = sorted(glob.glob(nii_pattern))
        T_total = len(nii_files)
        
        # DVF files are in sub_dir root
        dvf_root = sub_dir

        res = {k: [] for k in ["images", "world_points", "cam_points", "point_masks", "depths", "extrinsics", "intrinsics", "original_sizes", "frame_ids", "timesteps", "slice_indices", "gt_dvfs", "scale_factors", "z_indices", "rotations"]}

        for i in range(S):
            # i=0 is always the reference (t=1). Static mode is also always t=1.
            if self.mode == "static" or i == 0:
                t_idx = 1
            else:
                t_idx = random.randint(1, T_total)
            
            # Find the specific image file path for t_idx
            img_path = [f for f in nii_files if f.endswith(f"img_t{t_idx:03d}.nii.gz")][0]
            img_obj = nib.load(img_path)
            vol = img_obj.get_fdata()
            v_min, v_max = np.min(vol), np.max(vol)
            W, H, Z = vol.shape

            # 1. Orientation & Slicing
            axis = self.mri_mode if self.mri_mode != "mixed" else random.choice(["axial", "coronal", "sagittal"])
            rot = np.eye(3, dtype=np.float32)

            if axis == "axial":
                # Equidistant traversal across the Z volume
                idx = int(np.round((Z - 1) * (i / (max(1, S - 1)))))
                raw = vol[:, :, idx]
                r, c = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
                coords = np.stack([r, c, np.full_like(r, idx)], axis=-1).astype(np.float32)
                res["rotations"].append(np.zeros(3, dtype=np.float32))
            elif axis == "coronal":
                idx = random.randint(0, H - 1)
                raw = vol[:, idx, :]
                r, c = np.meshgrid(np.arange(W), np.arange(Z), indexing="ij")
                coords = np.stack([r, np.full_like(r, idx), c], axis=-1).astype(np.float32)
                rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
                res["rotations"].append(np.zeros(3, dtype=np.float32))
            elif axis == "sagittal":
                idx = random.randint(0, W - 1)
                raw = vol[idx, :, :]
                r, c = np.meshgrid(np.arange(H), np.arange(Z), indexing="ij")
                coords = np.stack([np.full_like(r, idx), r, c], axis=-1).astype(np.float32)
                rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
                res["rotations"].append(np.zeros(3, dtype=np.float32))
            elif axis == "oblique":
                out_h, out_w = 256, 256
                angles = tuple(np.random.uniform(0, 360, 3))
                res["rotations"].append(np.array(angles, dtype=np.float32))
                rot = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
                gx, gy = np.meshgrid(np.arange(-out_h // 2, out_h // 2), np.arange(-out_w // 2, out_w // 2), indexing="ij")
                flat_coords = rot @ np.stack([gx.flatten(), gy.flatten(), np.zeros_like(gx).flatten()])

                # Use volume center instead of hardcoded heart center
                # center = (127, 148, 34)
                center = (W // 2, H // 2, Z // 2)
                for j, cv in enumerate(center):
                    flat_coords[j, :] += cv

                raw = map_coordinates(vol, flat_coords, order=1, mode="constant", cval=v_min).reshape((out_h, out_w))
                coords = flat_coords.T.reshape((out_h, out_w, 3)).astype(np.float32)
                idx = i

            # Silently provide z_indices for potential sinusoidal embedding
            if axis == "axial":
                z_idx_norm = (idx / max(1, Z - 1)) * 2.0 - 1.0
            else:
                z_idx_norm = 0.0
            res["z_indices"].append(np.array([z_idx_norm], dtype=np.float32))

            # 2. Dynamic DVF Ground Truth (Phase 3)
            gt_dvf = np.zeros_like(coords)
            if self.mode == "dynamic" and t_idx > 1:
                dvf_path = os.path.join(dvf_root, f"cardiac_dense_frame{t_idx}_dvf.nii.gz")
                if os.path.exists(dvf_path):
                    dvf_vol = nib.load(dvf_path).get_fdata()  # (W, H, Z, 3)
                    # Sample DVF at the spoke's static coordinates
                    dx = map_coordinates(dvf_vol[..., 0], coords.reshape(-1, 3).T, order=1).reshape(raw.shape)
                    dy = map_coordinates(dvf_vol[..., 1], coords.reshape(-1, 3).T, order=1).reshape(raw.shape)
                    dz = map_coordinates(dvf_vol[..., 2], coords.reshape(-1, 3).T, order=1).reshape(raw.shape)
                    # DVF points from Frame 1 to Frame t. To map current coords (Frame t) back to Frame 1, we subtract DVF.
                    gt_dvf = np.stack([dx, dy, dz], axis=-1)
                    coords -= gt_dvf

            # Metadata storage:
            res["frame_ids"].append(i)  # Sequence index
            res["timesteps"].append(t_idx - 1)
            res["slice_indices"].append(idx)
            img = np.clip(np.repeat(((raw - v_min) / (v_max - v_min + 1e-8))[..., None], 3, -1) * 255.0, 0, 255).astype(np.float32)
            wp = coords.astype(np.float32).copy()

            # Isotropic and Centered Normalization
            max_dim = max(W, H, Z) - 1
            scale_factor = max_dim / 2
            wp[..., 0] = (wp[..., 0] - (W - 1) / 2) / scale_factor
            wp[..., 1] = (wp[..., 1] - (H - 1) / 2) / scale_factor
            wp[..., 2] = (wp[..., 2] - (Z - 1) / 2) / scale_factor

            h, w = raw.shape
            sc = self.target_size / max(h, w)
            nw, nh = int(w * sc), int(h * sc)
            img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
            wp_r = cv2.resize(wp, (nw, nh), interpolation=cv2.INTER_LINEAR)
            gt_dvf_r = cv2.resize(gt_dvf.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)

            vol_mask = (coords[..., 0] >= 0) & (coords[..., 0] < W) & (coords[..., 1] >= 0) & (coords[..., 1] < H) & (coords[..., 2] >= 0) & (coords[..., 2] < Z)
            vol_mask_r = cv2.resize(vol_mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0.5
            wp_r[~vol_mask_r] = -2.0

            hp, wp_p = self.target_size - nh, self.target_size - nw
            pt, pb, pl, pr = hp // 2, hp - hp // 2, wp_p // 2, wp_p - wp_p // 2

            res["images"].append(np.clip(np.pad(img_r, ((pt, pb), (pl, pr), (0, 0))), 0, 255))
            res["world_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=-2.0))
            res["cam_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=-2.0))
            res["gt_dvfs"].append(np.pad(gt_dvf_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=0.0))
            res["scale_factors"].append(np.array([scale_factor], dtype=np.float32))

            m = np.zeros((self.target_size, self.target_size), bool)
            m[pt : pt + nh, pl : pl + nw] = vol_mask_r
            res["point_masks"].append(m)

            res["depths"].append(np.zeros((self.target_size, self.target_size), np.float32))
            # Camera Pose must be Identity for MRI world coordinate regression
            res["extrinsics"].append(np.eye(3, 4, dtype=np.float32))
            res["intrinsics"].append(np.array([[self.target_size, 0, self.target_size / 2], [0, self.target_size, self.target_size / 2], [0, 0, 1]], np.float32))
            res["original_sizes"].append(np.array([h, w], np.float32))
            # res["frame_ids"] already handled earlier in loop

        rel_path = os.path.relpath(sub_dir, self.data_root)
        seq_name = f"mri_{self.mri_mode}_{rel_path.replace(os.sep, '_')}"
        return {**res, "seq_name": seq_name, "ids": np.array(res["frame_ids"], np.int64), "frame_num": S, "tracks": np.zeros((1, 1, 2), np.float32)}
