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
    def __init__(self, common_conf, data_root, split="train", split_file=None, mode="static", num_slices=5, target_size=518, mri_mode="axial", dvf_dirname="dvf_elastix", gt_grid_shape=(12, 256, 256)):
        """
        MRI Dataset for VGGT — CMRxRecon2024 Cine_combined format.

        Data layout:
            data_root/
              {SubjectName}/sax/3d_recon/sax_frame_{t:02d}.nii.gz   (0-indexed, t=0 is reference)
              {SubjectName}/sax/{dvf_dirname}/dvf_frame_{t:02d}.nii.gz (t=1..T-1, mm units, T→0 convention)

        split_file: path to a .txt file with [train]/[val]/[test] sections listing subject folder names.
        mode: 'static' (always t=0) or 'dynamic' (reference t=0 + random dynamic frames)
        mri_mode: 'axial' only (coronal/sagittal/oblique not used currently)
        dvf_dirname: directory name for DVFs (e.g., 'dvf_elastix', 'dvf_carmen')
        """
        super().__init__(common_conf=common_conf)
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.split_file = os.path.abspath(split_file) if split_file else None
        self.mode = mode
        self.num_slices = num_slices
        self.target_size = target_size
        self.mri_mode = mri_mode
        self.dvf_dirname = dvf_dirname
        self.gt_grid_shape = tuple(gt_grid_shape)  # (D, H, W) canonical-grid shape for GT phase-0 volume

        self.subjects = self._find_subjects()
        logging.info(f"MRIDataset [{split}]: {len(self.subjects)} subjects from {self.split_file}")
        self.len_train = 1000

    def _find_subjects(self):
        """Parse split_file and return list of subject/sax paths for self.split."""
        if self.split_file is None or not os.path.exists(self.split_file):
            logging.warning(f"MRIDataset: split_file not found: {self.split_file}. No subjects loaded.")
            return []
        subjects = []
        current_split = None
        with open(self.split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current_split = line[1:-1].lower()
                elif current_split == self.split.lower():
                    path = os.path.join(self.data_root, line, "sax")
                    if os.path.isdir(path):
                        subjects.append(path)
                    else:
                        logging.warning(f"MRIDataset: subject path not found, skipping: {path}")
        return subjects  # preserves file order — user controls ordering

    def __len__(self):
        return self.len_train

    def get_data(self, seq_index=0, img_per_seq=None, **kwargs):
        sub_dir = self.subjects[seq_index % len(self.subjects)]

        # For val/test, fix randomness per (subject, seq_index) so metrics are reproducible
        if self.split != "train":
            rng = random.Random(seq_index * 1000 + hash(sub_dir) % 100000)
        else:
            rng = None  # use global random for train

        # Discover image files: sax_frame_00.nii.gz ... sax_frame_{T-1:02d}.nii.gz
        nii_pattern = os.path.join(sub_dir, "3d_recon", "sax_frame_*.nii.gz")
        nii_files = sorted(glob.glob(nii_pattern))
        T_total = len(nii_files)

        # Load cardiac mask (W, H, Z) once per subject call — used to restrict loss to heart region
        cardiac_mask_path = os.path.join(sub_dir, self.dvf_dirname, "mask_frame_00.nii.gz")
        cardiac_mask_vol = None
        if os.path.exists(cardiac_mask_path):
            cardiac_mask_vol = nib.load(cardiac_mask_path).get_fdata() > 0.5  # (W, H, Z)

        # ── Determine S = min(T_total, Z_total, img_per_seq) ─────────────
        # img_per_seq is set by the dataloader from max_img_per_gpu to prevent OOM.
        # Get Z_total from first file header (cheap, no full volume load)
        _hdr = nib.load(nii_files[0]).header
        Z_total = int(_hdr.get_data_shape()[2])
        S = min(T_total, Z_total, img_per_seq or self.num_slices)

        # ── Pre-sample t and z without replacement ────────────────────────
        # Slot 0: t=0 (ED reference, required for GT volume loading).
        # Slots 1..S-1: random non-zero t, random z. All distinct within batch.
        if self.mode == "static":
            t_sequence = [0] * S
        else:
            dynamic_ts = list(range(1, T_total))
            if self.split != "train":
                rng.shuffle(dynamic_ts)
            else:
                random.shuffle(dynamic_ts)
            t_sequence = [0] + dynamic_ts[: S - 1]

        all_z = list(range(Z_total))
        if self.split != "train":
            rng.shuffle(all_z)
        else:
            random.shuffle(all_z)
        z_sequence = all_z[:S]

        res = {
            k: []
            for k in [
                "images",
                "world_points",
                "cam_points",
                "point_masks",
                "geom_masks",
                "depths",
                "extrinsics",
                "intrinsics",
                "original_sizes",
                "frame_ids",
                "timesteps",
                "slice_indices",
                "gt_dvfs",
                "scale_factors",
                "z_indices",
                "t_indices",
                "rotations",
                "scanner_coords",
            ]
        }

        gt_phase0_volume = None  # built once per sample when slot 0 (t=0) is processed
        for i in range(S):
            t_idx = t_sequence[i]
            idx = z_sequence[i]

            img_path = [f for f in nii_files if f.endswith(f"sax_frame_{t_idx:02d}.nii.gz")][0]
            img_obj = nib.load(img_path)
            vol = img_obj.get_fdata()
            spacing = np.array(img_obj.header.get_zooms()[:3], dtype=np.float32)  # (sx, sy, sz) mm/voxel
            v_min = np.percentile(vol, 1)
            v_max = np.percentile(vol, 99.5)
            W, H, Z = vol.shape

            # ── GT canonical volume (phase 0 only): load full vol, resample once to gt_grid_shape ──
            # Same per-axis normalized [-1, 1] frame as scanner_coords so V_canon and V_gt are voxel-aligned.
            # Each canonical axis's [-1, 1] spans exactly that axis's native physical extent →
            # the canonical grid voxel sizes match native acquisition (~8mm Z, ~1.34mm X/Y).
            if t_idx == 0 and gt_phase0_volume is None:
                D_t, H_t, W_t = self.gt_grid_shape
                half_t = np.array(
                    [W * spacing[0] / 2, H * spacing[1] / 2, Z * spacing[2] / 2],
                    dtype=np.float32,
                )
                center_t = np.array(
                    [(W - 1) / 2 * spacing[0], (H - 1) / 2 * spacing[1], (Z - 1) / 2 * spacing[2]],
                    dtype=np.float32,
                )
                d_canon = np.linspace(-1, 1, D_t, dtype=np.float32)
                h_canon = np.linspace(-1, 1, H_t, dtype=np.float32)
                w_canon = np.linspace(-1, 1, W_t, dtype=np.float32)
                dd, hh, ww = np.meshgrid(d_canon, h_canon, w_canon, indexing="ij")
                x_vox = (center_t[0] + half_t[0] * ww) / spacing[0]
                y_vox = (center_t[1] + half_t[1] * hh) / spacing[1]
                z_vox = (center_t[2] + half_t[2] * dd) / spacing[2]
                # vol is (W, H, Z); map_coordinates needs coords in (W, H, Z) order
                vox_coords = np.stack([x_vox.ravel(), y_vox.ravel(), z_vox.ravel()])
                gt_phase0_volume = map_coordinates(vol, vox_coords, order=1, cval=0.0).reshape(D_t, H_t, W_t)
                gt_phase0_volume = np.clip((gt_phase0_volume - v_min) / (v_max - v_min + 1e-8), 0, 1).astype(np.float32)

            # ── Axial slicing only ──────────────────────────────────────────
            raw = vol[:, :, idx]
            r, c = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
            coords_vox = np.stack([r, c, np.full_like(r, idx)], axis=-1).astype(np.float32)
            res["rotations"].append(np.zeros(3, dtype=np.float32))
            z_idx_norm = (idx / max(1, Z - 1)) * 2.0 - 1.0
            res["z_indices"].append(np.array([z_idx_norm], dtype=np.float32))
            t_idx_norm = (t_idx / max(1, T_total - 1)) * 2.0 - 1.0
            res["t_indices"].append(np.array([t_idx_norm], dtype=np.float32))

            # ── Convert voxel coords → physical mm ────────────────────────
            # Elastix registered in mm space; DVF is in mm. Working in mm gives
            # isotropic normalization and consistent DVF units across subjects.
            coords = coords_vox * spacing  # (H, W, 3) in mm

            # scanner_coords = physical position of each pixel in frame_t (before DVF)
            sc_wp = coords.copy()

            # ── Load DVF (mm, T→0 convention) ─────────────────────────────
            gt_dvf = np.zeros_like(coords)
            if self.mode == "dynamic" and t_idx > 0:
                dvf_path = os.path.join(sub_dir, self.dvf_dirname, f"dvf_frame_{t_idx:02d}.nii.gz")
                if os.path.exists(dvf_path):
                    dvf_obj = nib.load(dvf_path)
                    dvf_vol = dvf_obj.get_fdata()  # (W, H, Z, 1, 3) or (W, H, Z, 3), in mm
                    if dvf_vol.ndim == 5:
                        dvf_vol = dvf_vol[..., 0, :]  # squeeze to (W, H, Z, 3)

                    # map_coordinates needs voxel-index positions for indexing into dvf_vol
                    vox_pts = coords_vox.reshape(-1, 3).T  # use original voxel coords
                    dx = map_coordinates(dvf_vol[..., 0], vox_pts, order=1).reshape(raw.shape)
                    dy = map_coordinates(dvf_vol[..., 1], vox_pts, order=1).reshape(raw.shape)
                    dz = map_coordinates(dvf_vol[..., 2], vox_pts, order=1).reshape(raw.shape)

                    gt_dvf = np.stack([dx, dy, dz], axis=-1)  # mm displacement (T→0)
                    coords += gt_dvf  # coords now = world position in frame_0 (mm)

            # ── Physical (mm) normalization ────────────────────────────────
            # Use physical extent so all axes are treated consistently regardless
            # of voxel anisotropy or per-subject differences in spacing/FOV.
            # Per-axis normalization: each axis's [-1, 1] spans that axis's full physical extent,
            # so canonical voxels match native acquisition resolution (~8mm Z, ~1.34mm X/Y).
            half_extent = np.array(
                [W * spacing[0] / 2, H * spacing[1] / 2, Z * spacing[2] / 2],
                dtype=np.float32,
            )
            # Backward-compat: scale_factor used by viz / loss for un-normalization (mm-space DVF).
            # Use mean half-extent as a representative scalar; only matters for DVF visualization
            # in the legacy supervised path (the unsupervised volume loss does not use it).
            scale_factor = float(half_extent.mean())

            center_mm = np.array(
                [(W - 1) / 2 * spacing[0], (H - 1) / 2 * spacing[1], (Z - 1) / 2 * spacing[2]],
                dtype=np.float32,
            )

            wp = coords.copy()
            for p_map in [wp, sc_wp]:
                p_map[..., 0] = (p_map[..., 0] - center_mm[0]) / half_extent[0]
                p_map[..., 1] = (p_map[..., 1] - center_mm[1]) / half_extent[1]
                p_map[..., 2] = (p_map[..., 2] - center_mm[2]) / half_extent[2]

            # ── Volume mask (use voxel coords for bounds check) ────────────
            # coords is in mm; convert back to voxels to check if still inside volume
            coords_vox_post = coords / spacing
            geom_mask = (
                (coords_vox_post[..., 0] >= 0)
                & (coords_vox_post[..., 0] < W)
                & (coords_vox_post[..., 1] >= 0)
                & (coords_vox_post[..., 1] < H)
                & (coords_vox_post[..., 2] >= 0)
                & (coords_vox_post[..., 2] < Z)
            )

            # point_masks = cardiac region only (used for loss/MAE)
            # geom_masks  = full in-bounds region (used for visualization)
            vol_mask = geom_mask.copy()
            if cardiac_mask_vol is not None:
                cardiac_slice = cardiac_mask_vol[:, :, idx]  # (W, H)
                vol_mask = vol_mask & cardiac_slice

            # ── Resize to target_size ──────────────────────────────────────
            h, w = raw.shape
            sc = self.target_size / max(h, w)
            nw, nh = int(w * sc), int(h * sc)

            img_norm = np.clip(
                np.repeat(((raw - v_min) / (v_max - v_min + 1e-8))[..., None], 3, -1) * 255.0,
                0,
                255,
            ).astype(np.float32)

            img_r = cv2.resize(img_norm, (nw, nh), interpolation=cv2.INTER_CUBIC)
            wp_r = cv2.resize(wp, (nw, nh), interpolation=cv2.INTER_LINEAR)
            sc_wp_r = cv2.resize(sc_wp, (nw, nh), interpolation=cv2.INTER_LINEAR)
            gt_dvf_r = cv2.resize(gt_dvf.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
            vol_mask_r = cv2.resize(vol_mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0.5
            geom_mask_r = cv2.resize(geom_mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0.5

            wp_r[~geom_mask_r] = -2.0
            sc_wp_r[~geom_mask_r] = -2.0

            # ── Pad to target_size × target_size ──────────────────────────
            hp = self.target_size - nh
            wp_p = self.target_size - nw
            pt, pb = hp // 2, hp - hp // 2
            pl, pr = wp_p // 2, wp_p - wp_p // 2

            res["images"].append(np.clip(np.pad(img_r, ((pt, pb), (pl, pr), (0, 0))), 0, 255))
            res["world_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=-2.0))
            res["cam_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=-2.0))
            res["scanner_coords"].append(np.pad(sc_wp_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=-2.0))
            res["gt_dvfs"].append(np.pad(gt_dvf_r, ((pt, pb), (pl, pr), (0, 0)), constant_values=0.0))
            res["scale_factors"].append(np.array([scale_factor], dtype=np.float32))

            m = np.zeros((self.target_size, self.target_size), bool)
            m[pt : pt + nh, pl : pl + nw] = vol_mask_r
            res["point_masks"].append(m)

            gm = np.zeros((self.target_size, self.target_size), bool)
            gm[pt : pt + nh, pl : pl + nw] = geom_mask_r
            res["geom_masks"].append(gm)

            res["depths"].append(np.zeros((self.target_size, self.target_size), np.float32))
            res["extrinsics"].append(np.eye(3, 4, dtype=np.float32))
            res["intrinsics"].append(
                np.array(
                    [[self.target_size, 0, self.target_size / 2], [0, self.target_size, self.target_size / 2], [0, 0, 1]],
                    np.float32,
                )
            )
            res["original_sizes"].append(np.array([h, w], np.float32))
            res["frame_ids"].append(i)
            res["timesteps"].append(t_idx)
            res["slice_indices"].append(idx)

        rel_path = os.path.relpath(sub_dir, self.data_root)
        seq_name = f"mri_{self.mri_mode}_{rel_path.replace(os.sep, '_')}"
        out = {
            **res,
            "seq_name": seq_name,
            "ids": np.array(res["frame_ids"], np.int64),
            "frame_num": S,
            "tracks": np.zeros((1, 1, 2), np.float32),
        }
        if gt_phase0_volume is not None:
            out["gt_phase0_volume"] = gt_phase0_volume  # (D, H, W) float32 in [0, 1]
        return out
