"""MRIDataset — VGGT-MRI dataset, canonical-grid edition.

Each subject's 12 cine phases live on disk as `sax_frame_{tt:02d}.nii.gz`. A
monai `PersistentDataset` preprocess pipeline (see `training/data/preprocess.py`)
resamples every native NIfTI to a fixed (1.4, 1.4, 12.0) mm spacing and crops
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

from torch.utils.data import Dataset

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


class MRIDataset(Dataset):
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
        reference_slot=False,
        continuous_z=False,
        one_frame_per_slice=False,
        z_jitter=0.5,
        cache_dir=None,
        ef_val_sweep=False,
        cardiac_phase_csv=None,
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
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.split_file = os.path.abspath(split_file) if split_file else None
        self.mode = mode
        self.num_slices = num_slices
        self.target_size = target_size
        self.mri_mode = mri_mode
        self.t_target_fixed = t_target_fixed
        self.t_target_phases = list(t_target_phases) if t_target_phases is not None else None
        # Reference-slice conditioning (docs/25): when True, slot 0 is forced to OBSERVE the
        # target phase at the mid-ventricular plane (z = bbox z-center), and the remaining
        # slots are scattered with that plane excluded. The model reads the target phase from
        # slot-0's image content (via the native camera_token anchor) instead of a target_t
        # index. Default False → legacy decoupled sampling (slot 0 not special).
        self.reference_slot = bool(reference_slot)
        # Continuous physical z (docs/28): when True, each non-reference input slot is sampled at
        # a CONTINUOUS z (its nominal in-bbox plane + bounded jitter ∈ [-z_jitter, +z_jitter]),
        # extracted by linear interpolation between the two bracketing planes. Teaches the model
        # z as a continuous physical coordinate so off-grid inference slices splat correctly.
        # Slot 0 (reference, when reference_slot) stays on its integer plane. Default False →
        # integer planes (numerically identical to the discrete-grid pipeline).
        self.continuous_z = bool(continuous_z)
        # One-frame-per-slice (the sparse-acquisition extreme): when True, S is forced per subject
        # to the subject's in-FOV plane count so every in-bbox z plane appears EXACTLY once (no
        # multi-frame extras, n_extra=0). Overrides the incoming img_per_seq / num_slices budget;
        # safe because each batch is a single subject (batch_size=1), so the per-slot count can
        # vary across iterations without any cross-subject padding. Default False → the fixed-S
        # multi-frame sampler (bit-identical to before). Composes with reference_slot (slot 0 =
        # target-phase z_mid = that plane's one frame) and continuous_z (per-plane off-grid jitter).
        self.one_frame_per_slice = bool(one_frame_per_slice)
        self.z_jitter = float(z_jitter)
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

        # ── EF val sweep (opt-in, val-only): reconstruct each subject at its GT ED and ES ──
        # instead of the coupled seq_index%T phase. Builds an explicit (subj_idx, t_target) list
        # of length 2*N (all ED first, then all ES) from cardiac_phase.csv, so seq_index deterministically
        # enumerates a 30x{ED,ES} sweep. Needed for the predicted-EF metric (docs: EF-aware val).
        self.val_targets = None
        self.cardiac_phase_csv = None
        if ef_val_sweep and self.split.lower() == "val" and self.subjects:
            csv_path = cardiac_phase_csv or os.path.normpath(
                os.path.join(self.data_root, "..", "..", "whs", "cardiac_phase.csv"))
            self.val_targets = self._build_val_targets(csv_path)
            self.cardiac_phase_csv = csv_path
            logging.info(f"MRIDataset [val]: ef_val_sweep ON — {len(self.val_targets)} "
                         f"(subject, t_target) pairs from {csv_path}")

        # ── monai PersistentDataset cache ─────────────────────────────────
        if PersistentDataset is None:
            raise RuntimeError(
                "monai is required for canonical-grid MRIDataset — pip install monai>=1.6,<1.7"
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

    def _build_val_targets(self, csv_path):
        """Explicit (subj_idx, t_target) sweep: each subject at its GT ED, then each at its GT ES.
        Keyed by subject id (basename of the subject's parent dir) against the CSV `subject` column."""
        import csv as _csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ef_val_sweep needs cardiac_phase.csv; not found: {csv_path}")
        ed_es = {}
        with open(csv_path) as f:
            for row in _csv.DictReader(f):
                ed_es[row["subject"]] = (int(row["ED"]), int(row["ES"]))
        ed_list, es_list, missing = [], [], []
        for i, path in enumerate(self.subjects):
            sid = os.path.basename(os.path.dirname(path))  # ".../<ID>/sax" -> "<ID>"
            if sid not in ed_es:
                missing.append(sid); continue
            ed, es = ed_es[sid]
            # Fail loud if a subject's ED/ES falls outside the canonical phase count — else
            # get_data's `% T_total` would silently reconstruct the WRONG phase. All CMRx val
            # subjects have T=12 (ED/ES < 12); this guards a future split that adds T!=12 data.
            if not (0 <= ed < NUM_PHASES and 0 <= es < NUM_PHASES):
                raise ValueError(f"ef_val_sweep: {sid} has ED={ed}/ES={es} outside [0,{NUM_PHASES}); "
                                 "incompatible with the canonical 12-phase grid.")
            ed_list.append((i, ed)); es_list.append((i, es))
        if missing:
            raise KeyError(f"ef_val_sweep: {len(missing)} val subjects missing from {csv_path}: {missing}")
        return ed_list + es_list   # all ED, then all ES  ->  seq_index < N => ED, else ES

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    # ── Main get_data ────────────────────────────────────────────────────
    def get_data(self, seq_index=0, img_per_seq=None, **kwargs):
        forced_t = None
        if self.val_targets is not None:
            subj_idx, forced_t = self.val_targets[seq_index % len(self.val_targets)]
        else:
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
        # Priority: EF-sweep forced phase > single fixed phase > restricted phase pool > all T phases.
        if forced_t is not None:
            t_target = int(forced_t) % T_total
        elif self.t_target_fixed is not None:
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

        # ── S = requested slot budget (multi-frame; NOT capped by T or bbox) ──
        # Multi-frame-per-slice allows phase reuse (t with replacement) and plane
        # reuse (LV-weighted extras), so S is no longer clamped to T_total or the
        # in-FOV z extent. Full z-coverage is GUARANTEED in the slot-building block
        # below (every in-bbox plane appears ≥once), so V_canon has no coverage
        # holes and the full-volume L1 loss stays valid.
        S = img_per_seq or self.num_slices

        # ── Build (t, z) slot sequences — multi-frame, full coverage ──────
        # Matches real multi-frame-per-slice acquisition (+ classical SVR, + closer
        # to original VGGT's many-view input):
        #   • Every in-bbox z plane is covered AT LEAST once → full V_canon coverage.
        #   • The remaining slots are EXTRA frames whose planes are drawn UNIFORMLY at
        #     random (with replacement) over the in-bbox planes.
        #   • t per slot is a random phase WITH replacement, used ONLY to extract the
        #     slice's image CONTENT — input cardiac phase is NEVER a model input
        #     (t_indices/target_t are inert; real-time CMR carries no phase label).
        #
        # Train vs val differ ONLY in the RNG source (determinism + no global-RNG
        # leak): train → global `random` (fresh each epoch); val → a private
        # `random.Random(seq_index)` (reproducible across epochs/runs).
        #
        # REFERENCE-SLOT MODE (self.reference_slot, docs/25): slot 0 OBSERVES the
        # target phase at z_mid; the model reads the target phase from slot-0's image
        # content via the native camera_token anchor (NOT a target_t index). z_mid is
        # still covered by other slots/extras — multi-frame redundancy there is desired.
        rng = random if self.split == "train" else random.Random(seq_index)

        z_mid = (bbox_z0 + bbox_z1) // 2
        in_bbox_z = list(range(bbox_z0, bbox_z1)) or [z_mid]   # guard degenerate/empty bbox

        if self.reference_slot:
            z_sequence = [z_mid]                                # slot 0 = target-phase reference
            coverage = [z for z in in_bbox_z if z != z_mid]     # cover the remaining planes once
        else:
            z_sequence = []
            coverage = list(in_bbox_z)                          # cover all planes once

        if self.one_frame_per_slice:
            # Force S to the in-FOV plane count → every plane covered exactly once, n_extra=0
            # (the sparse one-frame-per-slice extreme). Ignores the incoming budget S.
            S = len(z_sequence) + len(coverage)

        room = S - len(z_sequence)
        if len(coverage) > room:                                # S < #planes (e.g. img_per_seq < bbox_z_size)
            rng.shuffle(coverage)
            coverage = coverage[: max(0, room)]                 # can't fully cover; subsample
        n_extra = max(0, room - len(coverage))

        # Extra frames: uniform random over the in-bbox planes (with replacement). LOCAL rng →
        # val deterministic, global RNG stream never perturbed.
        extras = rng.choices(in_bbox_z, k=n_extra) if n_extra else []

        tail = coverage + extras
        rng.shuffle(tail)                                       # order is irrelevant to the
        z_sequence += tail                                      # set-attention model; keeps val
        #                                                         inputs varying per seq_index and
        #                                                         interleaves extras with coverage.
        # len(z_sequence) == S; slot 0 (if reference) stays the z_mid anchor.

        # ── t per slot (extraction-only; never a model conditioning input) ──
        if self.mode == "static":
            t_sequence = [t_target] * S
        else:
            t_sequence = [rng.randrange(T_total) for _ in range(S)]
        # ABLATION HOOK (gated, default no-op): force the first n_forced_target slots
        # to OBSERVE the target phase. Inference-only; training never passes it.
        _n_forced = int(kwargs.get("n_forced_target", 0))
        for _i in range(min(_n_forced, S)):
            t_sequence[_i] = t_target
        if self.reference_slot:
            t_sequence[0] = t_target                            # slot 0 observes the target phase

        # ── Continuous physical z (gated; default OFF → integer planes) ────
        # Jitter each non-reference slot's nominal integer plane into a CONTINUOUS physical z
        # so the model learns z as a continuous coordinate (real inference slices land off the
        # discrete grid). Slot 0 (reference) stays on its integer plane so the filmstrip's
        # integer reference gather is unaffected. Bounded to ±z_jitter and clamped to
        # [0, D-1-eps] so the 2-plane blend and the splat in-bounds gate stay valid. Uses the
        # LOCAL rng → val-deterministic, no global-RNG leak.
        if self.continuous_z:
            eps = 1e-3
            z_sequence = [
                z if (self.reference_slot and i == 0)
                else float(min(max(0.0, z + rng.uniform(-self.z_jitter, self.z_jitter)), D - 1 - eps))
                for i, z in enumerate(z_sequence)
            ]

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
        slot_ts = torch.tensor(t_sequence, dtype=torch.long)
        if self.continuous_z:
            # Continuous z → linear blend between the two bracketing planes. eps-clamp above
            # guarantees floor ≤ D-2, so z0/z1 are valid indices. Reduces to the exact integer
            # plane when z is integer-valued (frac = 0), so the OFF path is numerically identical.
            z_f = torch.tensor(z_sequence, dtype=torch.float32)
            z0 = torch.floor(z_f).long().clamp(0, D - 1)
            z1 = (z0 + 1).clamp(0, D - 1)
            frac = (z_f - z0.float()).view(-1, 1, 1)                 # (S, 1, 1)
            s0 = phases_splat[slot_ts, z0].float()                  # (S, H, W)
            s1 = phases_splat[slot_ts, z1].float()
            canon_slices = (1.0 - frac) * s0 + frac * s1
        else:
            slot_indices = torch.tensor(z_sequence, dtype=torch.long)
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

            original_sizes_list.append(np.array([H_can, W_can], np.float32))

        # ── V_gt + full phases bundle (for Phase 4 aug) ───────────────────
        # `anatomy_bbox` was already computed above (used to constrain z sampling).
        gt_target_volume = phases_splat[t_target].float().cpu().numpy()  # (D, H, W) [0, 1] float32
        # phases_full is the full (T, D, H, W) canonical bundle. Kept in float16 to
        # keep batch transfer cheap; the trainer casts to float32 inside aug.
        phases_full = phases_splat.cpu().numpy()  # (T, D, H, W) float16
        content_mask_np = mask_splat.cpu().numpy().astype(np.uint8)  # (D, H, W)

        # Anatomy whole-heart ROI (nnU-Net seg, union-over-phases + dilation) resampled
        # onto the same canonical grid as the phases — a val-only metric mask that is
        # shared with the SVR baselines (vs the intensity-derived motion mask). Same axis
        # convention as content_mask: on-disk (X, Y, Z) → splat order (D, H, W). Loaded
        # from disk each call (~7 KB gz, ~6 ms); absent for synthetic-test / non-CMRx
        # subjects → the key is simply omitted and the metric skips those samples.
        heart_roi_path = os.path.join(sub_dir, "heart_roi_canonical.nii.gz")
        heart_roi_np = None
        if os.path.exists(heart_roi_path):
            roi_xyz = np.asarray(nib.load(heart_roi_path).dataobj)          # (X, Y, Z)
            heart_roi_np = np.ascontiguousarray(
                np.transpose(roi_xyz, (2, 1, 0)) > 0).astype(np.uint8)      # (D, H, W)

        rel_path = os.path.relpath(sub_dir, self.data_root)
        seq_name = f"mri_{self.mri_mode}_{rel_path.replace(os.sep, '_')}"

        return {
            "images": images_list,
            "scanner_coords": scanner_coords_list,
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
            "gt_target_volume": gt_target_volume,
            "t_target": np.array([t_target], dtype=np.int64),
            "anatomy_bbox": anatomy_bbox,
            "content_mask": content_mask_np,
            **({"heart_roi_canonical": heart_roi_np} if heart_roi_np is not None else {}),
            "phases": phases_full,
            # Stable per-sample id → deterministic val respiratory seeding (mirrors
            # the val `random.Random(seq_index)` z/t determinism). See gpu_aug.py.
            "seq_index": np.array([seq_index], dtype=np.int64),
        }
