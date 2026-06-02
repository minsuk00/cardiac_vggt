"""MRIDataset — VGGT-MRI dataset, canonical-grid edition.

Each subject's 12 cine phases live on disk as `sax_frame_{tt:02d}.nii.gz`. A
monai `PersistentDataset` preprocess pipeline (see `training/data/preprocess.py`)
resamples every native NIfTI to a fixed (1.4, 1.4, 8.0) mm spacing and crops
/zero-pads to (256, 256, 12) voxels, geometric-center-aligned. The cached
output is a single `(T=12, 1, X=256, Y=256, Z=12)` float16 tensor per subject,
plus a `(1, X, Y, Z)` content mask that tracks which voxels came from native
data vs zero-pad. Cache lives in `/tmp` (node-local NVMe, fast).

At training time `get_data` just looks up the cached bundle, permutes to splat
order `(T, D=12, H=256, W=256)`, samples (t_target, S=(t,z)-slots) per the
multi-phase contract, and produces:

    images          (S, 518, 518, 3)  float32, [0, 255]   — bilinear-upsampled
                                                            canonical slices, no
                                                            letterbox, no padding
    scanner_coords  (S, 518, 518, 3)  float32, [-1, +1]   — purely geometric:
                                                            (px, py, z_i) →
                                                            (x_norm, y_norm, z_norm)
                                                            same formula for every
                                                            subject
    world_points    same as scanner_coords (DVF supervision removed)
    cam_points      same as scanner_coords (legacy field)
    z_indices       (S, 1)   z_i / (D-1) * 2 - 1, D=12
    t_indices       (S, 1)   t_i / T * 2 - 1, T=12  (cyclic — wraps at +1)
    gt_target_volume (D, H, W) = phases_splat[t_target]
    anatomy_bbox    (6,) int64  — (z0, z1, y0, y1, x0, x1) from content_mask
    content_mask    (D, H, W) uint8  — 1 = native FOV reached, 0 = zero-pad
    phases          (T, D, H, W) float16 — full canonical bundle, needed by
                                          the Phase 4 GPU aug to augment all 12
                                          phases consistently then re-extract
                                          slices + V_gt + bbox.
    t_target        (1,) int64
    point_masks     (S, 518, 518) bool  all True (no letterbox padding region)
    geom_masks      (S, 518, 518) bool  all True (legacy field, kept for compat)

Drops (vs the legacy implementation):
    - scipy.ndimage.map_coordinates / cv2.resize / np.pad of inputs
    - per-subject `half_extent` / `center_mm` normalization
    - DVF NIfTI loading + `gt_dvfs` / `scale_factors` (deprecated)
    - cardiac_mask_vol loading (intensity mask was a side-effect of letterbox)
"""

from __future__ import annotations

import glob
import logging
import os
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from data.base_dataset import BaseDataset
from data.preprocess import (
    NUM_PHASES,
    TARGET_SHAPE,
    TARGET_SPACING,
    build_data_dicts,
    cache_signature,
    compute_geometric_bbox,
    default_cache_dir,
    get_canonical_transforms,
)

try:
    from monai.data import PersistentDataset
except ImportError:  # pragma: no cover — monai is a hard dep after this refactor
    PersistentDataset = None


# ──────────────────────────────────────────────────────────────────────────────
# Canonical-grid constants (single source of truth; mirror preprocess.py)
# ──────────────────────────────────────────────────────────────────────────────
# Splat-order shape (D, H, W) — used internally and by the splat. monai stores
# in (X, Y, Z) order, which transposes to splat (D=Z, H=Y, W=X).
GRID_SHAPE_SPLAT = (TARGET_SHAPE[2], TARGET_SHAPE[1], TARGET_SHAPE[0])  # (12, 256, 256)
INPUT_IMG_SIZE = 518  # DINOv2 patch_embed expects 518×518 (37 × 14)


class MRIDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        data_root,
        split="train",
        split_file=None,
        mode="static",
        num_slices=12,
        target_size=INPUT_IMG_SIZE,
        mri_mode="axial",
        dvf_dirname="dvf_elastix",     # legacy — accepted but unused
        gt_grid_shape=GRID_SHAPE_SPLAT,  # legacy override; must match preprocess.py
        t_target_fixed=None,
        t_target_phases=None,
        cache_dir=None,
    ):
        """
        Args mirrors the legacy MRIDataset for Hydra-config compatibility.
        New args:
            cache_dir: where monai PersistentDataset stores cached tensors.
                       Defaults to /tmp/vggt-mri_<USER>_monai_cache.
        Legacy args kept but no longer load anything:
            dvf_dirname: DVF supervision was removed; this is ignored.
            gt_grid_shape: must equal `GRID_SHAPE_SPLAT` (canonical grid is fixed).
        """
        super().__init__(common_conf=common_conf)
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.split_file = os.path.abspath(split_file) if split_file else None
        self.mode = mode
        self.num_slices = num_slices
        self.target_size = target_size
        self.mri_mode = mri_mode
        self.t_target_fixed = t_target_fixed
        self.t_target_phases = list(t_target_phases) if t_target_phases is not None else None
        if self.t_target_phases is not None and len(self.t_target_phases) == 0:
            raise ValueError("t_target_phases must be a non-empty list of phase indices, or null.")

        if tuple(gt_grid_shape) != GRID_SHAPE_SPLAT:
            raise ValueError(
                f"gt_grid_shape must match canonical {GRID_SHAPE_SPLAT}; got {tuple(gt_grid_shape)}. "
                "The canonical grid is fixed by training/data/preprocess.py."
            )
        # Stored for trainer diagnostics (identity baseline, cardiac filmstrip) that
        # read mri_ds.gt_grid_shape. Always equals the canonical GRID_SHAPE_SPLAT.
        self.gt_grid_shape = tuple(gt_grid_shape)

        # Legacy ignored arg — surface a warning so people don't think it does something.
        if dvf_dirname not in (None, "dvf_elastix"):
            logging.info(
                f"MRIDataset [{split}]: dvf_dirname={dvf_dirname!r} ignored "
                f"(DVF supervision was removed from the live data path)."
            )

        # ── Subject discovery (same split-file format as before) ──────────
        self.subjects = self._find_subjects()
        logging.info(f"MRIDataset [{split}]: {len(self.subjects)} subjects from {self.split_file}")
        self.len_train = 1000

        # ── monai PersistentDataset cache ─────────────────────────────────
        if PersistentDataset is None:
            raise RuntimeError(
                "monai is required for canonical-grid MRIDataset — pip install monai>=1.4,<1.5"
            )
        # Subdir keyed by content-defining params (spacing/shape/normalization) so a
        # normalization change routes to a fresh cache instead of silently reusing a
        # stale one (PersistentDataset hashes only the input dict, not the transform).
        cache_dir = os.path.join(cache_dir or default_cache_dir(), cache_signature())
        os.makedirs(cache_dir, exist_ok=True)
        data_dicts = build_data_dicts(self.subjects, num_phases=NUM_PHASES)
        self.cache = PersistentDataset(
            data=data_dicts,
            transform=get_canonical_transforms(
                target_spacing=TARGET_SPACING,
                target_shape=TARGET_SHAPE,
                num_phases=NUM_PHASES,
            ),
            cache_dir=cache_dir,
        )
        logging.info(
            f"MRIDataset [{split}]: PersistentDataset cache_dir={cache_dir}  "
            f"target_spacing={TARGET_SPACING}  target_shape={TARGET_SHAPE}"
        )

    # ── Subject discovery ────────────────────────────────────────────────
    def _find_subjects(self):
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
        return subjects

    def __len__(self):
        return self.len_train

    # ── Main get_data ────────────────────────────────────────────────────
    def get_data(self, seq_index=0, img_per_seq=None, **kwargs):
        subj_idx = seq_index % len(self.subjects)
        sub_dir = self.subjects[subj_idx]

        # Val/test determinism: the val branch below makes NO random calls (the
        # (t, z) slots are a pure function of seq_index + the subject's fixed
        # bbox) and the val loader runs shuffle=False, so val get_data is fully
        # reproducible across epochs and runs without any per-sample seeding.

        # ── Cache lookup → splat-order tensors ────────────────────────────
        cached = self.cache[subj_idx]
        # ConcatItemsd(dim=0) stacks 12 × (1, X, Y, Z) → (T=12, X=256, Y=256, Z=12)
        # (the per-phase channel dim is absorbed into T). content_mask keeps its
        # channel dim: (1, X=256, Y=256, Z=12).
        phases = cached["phases"]                # (T, X, Y, Z)  [or (T, 1, X, Y, Z) if shape ever changes]
        content_mask = cached["content_mask"]    # (1, X, Y, Z)
        if phases.ndim == 5:                     # defensive: tolerate a channel dim
            phases = phases.squeeze(1)
        # Axis-order conversion site (ONLY here). monai (X,Y,Z) → splat (D=Z,H=Y,W=X).
        phases_splat = phases.permute(0, 3, 2, 1).contiguous()              # (T, D=12, H=256, W=256)
        mask_splat = content_mask.squeeze(0).permute(2, 1, 0).contiguous()  # (D, H, W)
        T_total, D, H_can, W_can = phases_splat.shape
        assert (D, H_can, W_can) == GRID_SHAPE_SPLAT, (D, H_can, W_can)

        # ── Geometric anatomy bbox (computed BEFORE z sampling) ───────────
        # Used to restrict z sampling to canonical planes that carry real data
        # (i.e., inside the subject's native FOV). Without this, small-Z subjects
        # waste many slots on zero-padded Z planes — see explanation in the
        # `z_sequence` block below.
        anatomy_bbox = compute_geometric_bbox(mask_splat).cpu().numpy().astype(np.int64)  # (6,)
        bbox_z0, bbox_z1 = int(anatomy_bbox[0]), int(anatomy_bbox[1])
        bbox_z_size = max(1, bbox_z1 - bbox_z0)  # at least 1 for fallback

        # ── Pick t_target ─────────────────────────────────────────────────
        # Priority: single fixed phase > restricted phase pool > all T phases.
        if self.t_target_fixed is not None:
            t_target = int(self.t_target_fixed) % T_total
        elif self.t_target_phases is not None:
            pool = [int(t) % T_total for t in self.t_target_phases]
            if self.split != "train":
                t_target = pool[seq_index % len(pool)]   # deterministic cycle for stable val
            else:
                t_target = random.choice(pool)
        elif self.split != "train":
            t_target = seq_index % T_total
        else:
            t_target = random.randrange(T_total)

        # ── S = min(T, in-bbox z count, requested) ────────────────────────
        # Cap by the subject's in-FOV z extent (bbox_z_size), NOT the padded
        # canonical D=12. A small-FOV subject thus gets FEWER than 12 input
        # slices rather than wrapping z back over already-sampled planes
        # (e.g. slot (t=10,z=10) → (t=11,z=0)). t still cycles mod T (phases are
        # cyclic, so distinct phases), but z never repeats within a sample.
        # Before the canonical-grid refactor `D` was the subject's native Z, so
        # this min naturally shrank S; padding Z to 12 silently pinned S=12.
        S = min(T_total, bbox_z_size, img_per_seq or self.num_slices)

        # ── Build (t, z) slot sequences ───────────────────────────────────
        # IMPORTANT: z is sampled from WITHIN the anatomy bbox (in-FOV z planes
        # only). Otherwise small-Z subjects (e.g., native Z=6 → bbox z=[3,9])
        # waste up to half their slots on zero-padded Z planes, where input
        # intensity = 0 → splat writes 0 → matches V_gt = 0 → loss = 0 (harmless
        # but useless compute, and uneven data efficiency across subjects).
        #
        # DECOUPLED TARGET PHASE: the reconstruction target phase t_target is
        # injected globally via `target_t_indices` (broadcast to every slot, built
        # below) — NOT by forcing slot 0 to be a slice at t_target. So input-slice
        # phases are sampled INDEPENDENTLY of t_target; slot 0 is no longer special.
        # A slice that happens to land on t_target still acts as a free anchor, but
        # it's no longer guaranteed (the honest sparse/unobserved-phase regime).
        #
        # Train AND val draw from the SAME distribution: each slot t = random phase
        # WITH replacement (independent of t_target); z = random WITHOUT replacement
        # within the bbox (distinct planes). They differ ONLY in the RNG source:
        #   Train → the global `random` module: varies every epoch (the model should
        #           see fresh scattered acquisitions, like augmentation).
        #   Val   → a LOCAL `random.Random(seq_index)`: random-LOOKING but fully
        #           reproducible across runs AND epochs (same seq_index → same draw),
        #           and a PRIVATE generator that NEVER perturbs the global RNG stream
        #           the train split draws from. Each val seq_index is therefore a
        #           distinct, reproducible scattered acquisition — no rigid pattern and
        #           no bit-identical duplicates when (subject, t_target) recur.
        # t_target itself is chosen above (a deterministic cycle in val) for balanced
        # per-phase coverage; only the INPUT (t, z) is randomized here.
        # Static mode: all slots == t_target (z still drawn as below).
        rng = random if self.split == "train" else random.Random(seq_index)
        if self.mode == "static":
            t_sequence = [t_target] * S
        else:
            t_sequence = [rng.randrange(T_total) for _ in range(S)]
        in_bbox_z = list(range(bbox_z0, bbox_z1))
        if len(in_bbox_z) >= S:
            rng.shuffle(in_bbox_z)
            z_sequence = in_bbox_z[:S]
        else:
            # Fewer in-bbox z planes than requested slots → sample z with replacement.
            # (Unreachable under the S = min(..., bbox_z_size, ...) cap above; kept as a
            # defensive guard.) t is independent per slot, so coverage stays diverse.
            z_sequence = rng.choices(in_bbox_z, k=S)

        # ── Build per-slot tensors ────────────────────────────────────────
        images_list = []
        scanner_coords_list = []
        z_indices_list = []
        t_indices_list = []
        target_t_indices_list = []
        rotations_list = []
        frame_ids_list = []
        timesteps_list = []
        slice_indices_list = []
        depths_list = []
        extrinsics_list = []
        intrinsics_list = []
        original_sizes_list = []

        # Per-pixel canonical (x, y, z) coords for a 518×518 input image.
        # Bilinear resize 256→518 with align_corners semantics: pixel (py, px) of
        # 518×518 corresponds to source 256×256 voxel index (py·255/517, px·255/517).
        # Normalized [-1, +1]: y_norm = py/517·2 - 1; same for x. Constant across
        # subjects → compute once.
        py_grid, px_grid = np.meshgrid(np.arange(INPUT_IMG_SIZE), np.arange(INPUT_IMG_SIZE), indexing="ij")
        x_norm = (px_grid.astype(np.float32) / (INPUT_IMG_SIZE - 1)) * 2.0 - 1.0
        y_norm = (py_grid.astype(np.float32) / (INPUT_IMG_SIZE - 1)) * 2.0 - 1.0

        # Pre-resize ALL S canonical slices in one batched F.interpolate call.
        # `to_resize` shape (S, 1, 256, 256) float32; output (S, 1, 518, 518).
        slot_indices = torch.tensor(z_sequence, dtype=torch.long)
        slot_ts = torch.tensor(t_sequence, dtype=torch.long)
        # phases_splat[slot_ts, slot_indices] would do fancy indexing; use it.
        canon_slices = phases_splat[slot_ts, slot_indices].float()  # (S, H=256, W=256)
        canon_slices = canon_slices.unsqueeze(1)                    # (S, 1, 256, 256)
        upsampled = F.interpolate(
            canon_slices, size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
            mode="bilinear", align_corners=True,
        )                                                            # (S, 1, 518, 518)
        # Match ComposedDataset's `/255` contract — keep images in [0, 255].
        upsampled = (upsampled.squeeze(1) * 255.0).clamp(0, 255).cpu().numpy()  # (S, 518, 518)

        for i in range(S):
            t_idx = t_sequence[i]
            z_i = z_sequence[i]
            # RGB-replicate to match VGGT model contract (3-channel input).
            img = np.repeat(upsampled[i, ..., None], 3, axis=-1).astype(np.float32)
            images_list.append(img)

            # scanner_coords: per-pixel canonical (x_norm, y_norm, z_norm) for this z.
            z_val = (z_i / max(1, D - 1)) * 2.0 - 1.0
            sc = np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], axis=-1).astype(np.float32)
            scanner_coords_list.append(sc)

            # z / t indices (per-slot scalar embeddings).
            z_indices_list.append(np.array([z_val], dtype=np.float32))
            t_val = (t_idx / max(1, T_total)) * 2.0 - 1.0  # cyclic, wrap at +1
            t_indices_list.append(np.array([t_val], dtype=np.float32))
            # target_t index: the GLOBAL reconstruction target phase, same value for
            # every slot (broadcast query). Normalized identically to t_indices so the
            # separate target_t_embedder sees the same cyclic domain.
            target_t_val = (t_target / max(1, T_total)) * 2.0 - 1.0
            target_t_indices_list.append(np.array([target_t_val], dtype=np.float32))

            rotations_list.append(np.zeros(3, dtype=np.float32))
            frame_ids_list.append(i)
            timesteps_list.append(t_idx)
            slice_indices_list.append(z_i)

            # Legacy-shape filler fields (kept so ComposedDataset and downstream
            # code paths don't have to special-case the MRI dataset).
            depths_list.append(np.zeros((self.target_size, self.target_size), np.float32))
            extrinsics_list.append(np.eye(3, 4, dtype=np.float32))
            intrinsics_list.append(
                np.array(
                    [[self.target_size, 0, self.target_size / 2],
                     [0, self.target_size, self.target_size / 2],
                     [0, 0, 1]],
                    np.float32,
                )
            )
            original_sizes_list.append(np.array([H_can, W_can], np.float32))

        # ── Masks (all-True; no letterbox padding region in the new pipeline) ─
        all_true = np.ones((INPUT_IMG_SIZE, INPUT_IMG_SIZE), dtype=bool)
        point_masks_list = [all_true.copy() for _ in range(S)]
        geom_masks_list = [all_true.copy() for _ in range(S)]

        # ── world_points = cam_points = scanner_coords (DVF supervision removed) ─
        world_points_list = [sc.copy() for sc in scanner_coords_list]
        cam_points_list = [sc.copy() for sc in scanner_coords_list]

        # ── V_gt + full phases bundle (for Phase 4 aug) ───────────────────
        # `anatomy_bbox` was already computed above (used to constrain z sampling).
        gt_target_volume = phases_splat[t_target].float().cpu().numpy()  # (D, H, W) [0, 1] float32
        # phases_full is the full (T, D, H, W) canonical bundle. Kept in float16 to
        # keep batch transfer cheap; the trainer casts to float32 inside aug.
        phases_full = phases_splat.cpu().numpy()  # (T, D, H, W) float16
        content_mask_np = mask_splat.cpu().numpy().astype(np.uint8)  # (D, H, W)

        rel_path = os.path.relpath(sub_dir, self.data_root)
        seq_name = f"mri_{self.mri_mode}_{rel_path.replace(os.sep, '_')}"

        return {
            "images": images_list,
            "world_points": world_points_list,
            "cam_points": cam_points_list,
            "scanner_coords": scanner_coords_list,
            "point_masks": point_masks_list,
            "geom_masks": geom_masks_list,
            "depths": depths_list,
            "extrinsics": extrinsics_list,
            "intrinsics": intrinsics_list,
            "original_sizes": original_sizes_list,
            "frame_ids": frame_ids_list,
            "timesteps": timesteps_list,
            "slice_indices": slice_indices_list,
            "z_indices": z_indices_list,
            "t_indices": t_indices_list,
            "target_t_indices": target_t_indices_list,
            "rotations": rotations_list,
            "seq_name": seq_name,
            "ids": np.array(frame_ids_list, np.int64),
            "frame_num": S,
            "tracks": np.zeros((1, 1, 2), np.float32),
            "gt_target_volume": gt_target_volume,
            "t_target": np.array([t_target], dtype=np.int64),
            "anatomy_bbox": anatomy_bbox,
            "content_mask": content_mask_np,
            "phases": phases_full,
        }
