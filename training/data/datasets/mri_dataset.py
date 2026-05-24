import glob
import logging
import os
import random
import zlib

import cv2
import nibabel as nib
import numpy as np
import torch
from data.base_dataset import BaseDataset
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


class MRIDataset(BaseDataset):
    def __init__(self, common_conf, data_root, split="train", split_file=None, mode="static", num_slices=5, target_size=518, mri_mode="axial", gt_grid_shape=(12, 256, 256), t_target_fixed=None):
        """
        MRI Dataset for VGGT — CMRxRecon2024 Cine_combined format.

        Data layout:
            data_root/
              {SubjectName}/sax/3d_recon/sax_frame_{t:02d}.nii.gz   (0-indexed)

        split_file: path to a .txt file with [train]/[val]/[test] sections listing subject folder names.
        mode: 'static' (all slots at t_target) or 'dynamic' (slot 0 at t_target + other slots at varying t).
        mri_mode: 'axial' only (coronal/sagittal/oblique not used currently).
        t_target_fixed: if None (default), sample t_target uniformly per call (train: random; val: rng-seeded).
                        If int, force t_target to that fixed phase every call — use 0 to reproduce the
                        original ED-only pipeline.
        """
        super().__init__(common_conf=common_conf)
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.split_file = os.path.abspath(split_file) if split_file else None
        self.mode = mode
        self.num_slices = num_slices
        self.target_size = target_size
        self.mri_mode = mri_mode
        self.gt_grid_shape = tuple(gt_grid_shape)  # (D, H, W) canonical-grid shape for GT volume
        self.t_target_fixed = None if t_target_fixed is None else int(t_target_fixed)

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
            # zlib.crc32 (not Python's hash()) so the seed is deterministic across processes.
            rng = random.Random(seq_index * 1000 + zlib.crc32(sub_dir.encode()) % 100000)
        else:
            rng = None  # use global random for train

        # Discover image files: sax_frame_00.nii.gz ... sax_frame_{T-1:02d}.nii.gz
        nii_pattern = os.path.join(sub_dir, "3d_recon", "sax_frame_*.nii.gz")
        nii_files = sorted(glob.glob(nii_pattern))
        T_total = len(nii_files)

        # ── Determine S = min(T_total, Z_total, img_per_seq) ─────────────
        # img_per_seq is set by the dataloader from max_img_per_gpu to prevent OOM.
        # Get Z_total from first file header (cheap, no full volume load)
        _hdr = nib.load(nii_files[0]).header
        Z_total = int(_hdr.get_data_shape()[2])
        S = min(T_total, Z_total, img_per_seq or self.num_slices)

        # ── Sample target phase ───────────────────────────────────────────
        # Slot 0 is anchored to t_target; V_gt is loaded from the t_target NIfTI.
        # If t_target_fixed is set, use it (e.g., 0 reproduces the original ED-only pipeline).
        # Otherwise — train: uniform random; val/test: STRATIFIED via seq_index % T_total
        #   so per-phase val metrics get balanced sample counts (e.g., 30 val subjects ÷ 12
        #   phases ≈ 3 subjects/phase for t=0..5, 2 subjects/phase for t=6..11). Deterministic.
        if self.t_target_fixed is not None:
            t_target = self.t_target_fixed % max(1, T_total)
        elif self.split != "train":
            t_target = seq_index % T_total
        else:
            t_target = random.randrange(T_total)

        # ── Pre-sample t and z without replacement ────────────────────────
        # Train: random t/z per slot (subject to slot 0 = t_target).
        # Val/test: diagonal acquisition pattern — slot i = ((t_target + i) mod T, z=i).
        #   This matches real-life sequential slice acquisition (one slice per heartbeat,
        #   scanner walks through z while cardiac phase advances). Deterministic given t_target.
        if self.split != "train":
            if self.mode == "static":
                t_sequence = [t_target] * S
            else:
                t_sequence = [(t_target + i) % T_total for i in range(S)]
            z_sequence = list(range(S))
        else:
            if self.mode == "static":
                t_sequence = [t_target] * S
            else:
                dynamic_ts = [t for t in range(T_total) if t != t_target]
                random.shuffle(dynamic_ts)
                t_sequence = [t_target] + dynamic_ts[: S - 1]
            all_z = list(range(Z_total))
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
                "z_indices",
                "t_indices",
                "rotations",
                "scanner_coords",
            ]
        }

        gt_target_volume = None  # built once per sample when slot 0 (t=t_target) is processed
        ref_v_min = ref_v_max = None  # cached t_target percentiles, reused for every slot
        for i in range(S):
            t_idx = t_sequence[i]
            idx = z_sequence[i]

            img_path = [f for f in nii_files if f.endswith(f"sax_frame_{t_idx:02d}.nii.gz")][0]
            img_obj = nib.load(img_path)
            vol = img_obj.get_fdata()
            spacing = np.array(img_obj.header.get_zooms()[:3], dtype=np.float32)  # (sx, sy, sz) mm/voxel
            # Use t_target percentiles for V_gt AND every input slot so the unsupervised
            # |V_canon - V_gt| loss isn't biased by per-phase intensity drift.
            # Slot 0 is always t=t_target (see t_sequence construction above), so the cache
            # is populated before any other slot needs it.
            if ref_v_min is None:
                assert t_idx == t_target, "slot 0 must be t=t_target so reference percentiles come from the GT frame"
                ref_v_min = np.percentile(vol, 1)
                ref_v_max = np.percentile(vol, 99.5)
            v_min, v_max = ref_v_min, ref_v_max
            W, H, Z = vol.shape

            # ── GT canonical volume (t_target only): load full vol, resample once to gt_grid_shape ──
            # Same per-axis normalized [-1, 1] frame as scanner_coords so V_canon and V_gt are voxel-aligned.
            # Each canonical axis's [-1, 1] spans exactly that axis's native physical extent →
            # the canonical grid voxel sizes match native acquisition (~8mm Z, ~1.34mm X/Y).
            if t_idx == t_target and gt_target_volume is None:
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
                # vol is (W, H, Z); map_coordinates needs coords in (W, H, Z) order.
                # Option A (current): mode='nearest' for boundary canonical voxels.
                #   half_extent uses N·spacing/2 (full physical FOV), so canonical
                #   d=0 and d=D-1 query z_vox = -0.5 and Z-0.5 — half a voxel past
                #   the native data on each side. With the previous cval=0 fill,
                #   scipy returned literal 0 (no partial interpolation), creating
                #   phantom V_gt=0 boundary voxels that the splat (working off
                #   scanner_coords with the SAME half_extent) does write into.
                #   'nearest' replicates the edge native voxel instead, so V_gt
                #   has real anatomy at the boundaries — consistent with what the
                #   splat puts there.
                # TODO Option B (cleaner, next retrain): shrink half_extent to
                #   (N-1)/2·spacing for ALL three axes here AND in the scanner_coords
                #   normalization at the bottom of get_data. Canonical [-1,+1] then
                #   spans native voxel centers exactly; no phantom region; input z=0
                #   maps exactly to canonical d=0. Invalidates current checkpoints
                #   because it changes the scanner_coords scale the model trained on.
                vox_coords = np.stack([x_vox.ravel(), y_vox.ravel(), z_vox.ravel()])
                gt_target_volume = map_coordinates(vol, vox_coords, order=1, mode="nearest").reshape(D_t, H_t, W_t)
                gt_target_volume = np.clip((gt_target_volume - v_min) / (v_max - v_min + 1e-8), 0, 1).astype(np.float32)

            # ── Axial slicing only ──────────────────────────────────────────
            raw = vol[:, :, idx]
            r, c = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
            coords_vox = np.stack([r, c, np.full_like(r, idx)], axis=-1).astype(np.float32)
            res["rotations"].append(np.zeros(3, dtype=np.float32))
            z_idx_norm = (idx / max(1, Z - 1)) * 2.0 - 1.0
            res["z_indices"].append(np.array([z_idx_norm], dtype=np.float32))
            # Cyclic encoding: divide by T_total (not T_total-1) so the wrap point
            # lands at +1 (outside the data range). With sin/cos(2^i · π · t_norm),
            # t=0 and t=T-1 are now close-but-distinct rather than collapsed to the
            # same feature point. See TIndexEmbedder in vggt/models/aggregator.py.
            t_idx_norm = (t_idx / max(1, T_total)) * 2.0 - 1.0
            res["t_indices"].append(np.array([t_idx_norm], dtype=np.float32))

            # ── Convert voxel coords → physical mm ────────────────────────
            # Elastix registered in mm space; DVF is in mm. Working in mm gives
            # isotropic normalization and consistent DVF units across subjects.
            coords = coords_vox * spacing  # (H, W, 3) in mm

            # scanner_coords = physical position of each pixel in frame_t.
            # world_points is identical (GT DVF supervision was removed; the model's
            # predicted Δ is added to scanner_coords inside vggt.py, then splatted).
            sc_wp = coords.copy()

            # ── Physical (mm) normalization ────────────────────────────────
            # Use physical extent so all axes are treated consistently regardless
            # of voxel anisotropy or per-subject differences in spacing/FOV.
            # Per-axis normalization: each axis's [-1, 1] spans that axis's full physical extent,
            # so canonical voxels match native acquisition resolution (~8mm Z, ~1.34mm X/Y).
            half_extent = np.array(
                [W * spacing[0] / 2, H * spacing[1] / 2, Z * spacing[2] / 2],
                dtype=np.float32,
            )
            center_mm = np.array(
                [(W - 1) / 2 * spacing[0], (H - 1) / 2 * spacing[1], (Z - 1) / 2 * spacing[2]],
                dtype=np.float32,
            )

            wp = coords.copy()
            for p_map in [wp, sc_wp]:
                p_map[..., 0] = (p_map[..., 0] - center_mm[0]) / half_extent[0]
                p_map[..., 1] = (p_map[..., 1] - center_mm[1]) / half_extent[1]
                p_map[..., 2] = (p_map[..., 2] - center_mm[2]) / half_extent[2]

            # ── Volume mask ────────────────────────────────────────────────
            # coords_vox is constructed entirely inside the native volume
            # ([0, W-1] × [0, H-1] × {idx}), so the per-pixel bounds check is
            # trivially True. point_masks/geom_masks are kept in the batch
            # because downstream code (compute_point_loss, _apply_batch_repetition,
            # visualizations) still reads them.
            geom_mask = np.ones(coords_vox.shape[:-1], dtype=bool)
            vol_mask = geom_mask.copy()

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
            vol_mask_r = cv2.resize(vol_mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0.5
            geom_mask_r = cv2.resize(geom_mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0.5

            wp_r[~geom_mask_r] = -2.0
            sc_wp_r[~geom_mask_r] = -2.0

            # ── Pad to target_size × target_size ──────────────────────────
            hp = self.target_size - nh
            wp_p = self.target_size - nw
            pt, pb = hp // 2, hp - hp // 2
            pl, pr = wp_p // 2, wp_p - wp_p // 2

            # Pad with 0 (interior point). Padded pixels are dropped by the splat's
            # intensity gate (Mask B) since image padding is also 0, so the position
            # value at padded pixels has no effect on V_canon. The earlier -2.0
            # sentinel was vestige from the supervised point-loss pipeline and only
            # served to inflate the TV penalty at anatomy/padding boundaries.
            res["images"].append(np.clip(np.pad(img_r, ((pt, pb), (pl, pr), (0, 0))), 0, 255))
            res["world_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0))))
            res["cam_points"].append(np.pad(wp_r, ((pt, pb), (pl, pr), (0, 0))))
            res["scanner_coords"].append(np.pad(sc_wp_r, ((pt, pb), (pl, pr), (0, 0))))

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
        if gt_target_volume is not None:
            out["gt_target_volume"] = gt_target_volume  # (D, H, W) float32 in [0, 1]
        out["t_target"] = np.array([t_target], dtype=np.int64)
        return out
