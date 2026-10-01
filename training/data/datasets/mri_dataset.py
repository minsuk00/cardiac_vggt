"""MRIDataset — VGGT-MRI dataset on the in-plane canonical grid with native z (docs/58).

Each subject's 12 cine phases live on disk as `sax_frame_{tt:02d}.nii.gz`. A
monai `PersistentDataset` preprocess pipeline (see `training/data/preprocess.py`)
resamples every native NIfTI's IN-PLANE spacing to a fixed 1.4 mm and crops
/zero-pads X/Y to 256×256 — but does NOT resample z: each subject keeps its own
native z spacing/plane count (`dz_mm`, `D`). The cached output is a single
`(T=12, X=256, Y=256, Z=D)` float16 tensor per subject, plus a `(1, X, Y, Z)`
content mask and `dz_mm` (that subject's own z spacing). Cache lives in `/tmp`
(node-local NVMe, fast).

At training time `get_data` just looks up the cached bundle, permutes to splat
order `(T, D, H=256, W=256)`, samples (t_target, S=(t,z)-slots) per the
multi-phase contract, and produces:

    images          (S, R, R, 3)  float32, [0, 255]       — bilinear-upsampled
                                                            canonical slices, no
                                                            letterbox, no padding
                                                            (R = target_size; omitted
                                                            when defer_input_images)
    scanner_coords  (S, R, R, 3)  float32, [-1, +1]       — purely geometric:
                                                            (px, py, z_i) →
                                                            (x_norm, y_norm, z_norm);
                                                            x/y use the same formula
                                                            for every subject, z is
                                                            PHYSICAL (z_mm/Z_HALF_MM)
                                                            so it's also comparable
                                                            across subjects despite D
                                                            varying
    z_indices       (S, 1)   (z_i - (D-1)/2) * dz / Z_HALF_MM
    gt_target_volume (D, H, W) = phases_splat[t_target]
    anatomy_bbox    (6,) int64  — (z0, z1, y0, y1, x0, x1) from content_mask
    content_mask    (D, H, W) uint8  — 1 = native FOV reached, 0 = zero-pad (x/y only)
    phases          (T, D, H, W) float16 — full canonical bundle, needed by
                                          the GPU aug (gpu_aug.py) to augment all 12
                                          phases consistently then re-extract
                                          slices + V_gt + bbox.
    t_target        (1,) int64
    dz_mm           (1,) float32 — this subject's own native z spacing
    z_scale         (1,) float32 — Z_HALF_MM / dz_mm; required by splat.py
"""

from __future__ import annotations

import logging
import os
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from monai.data import PersistentDataset
from torch.utils.data import Dataset

from training.data.preprocess import (
    NUM_PHASES,
    TARGET_SHAPE,
    TARGET_SPACING,
    Z_HALF_MM,
    build_data_dicts,
    cache_signature,
    compute_geometric_bbox,
    default_cache_dir,
    get_canonical_transforms,
)


# ──────────────────────────────────────────────────────────────────────────────
# Canonical-grid constants (single source of truth; mirror preprocess.py)
# ──────────────────────────────────────────────────────────────────────────────
# In-plane shape (H, W) — FIXED for every subject. There is no fixed D: each
# subject's canonical grid is (D, 256, 256) with D = its own native slice count.
CANONICAL_HW = (TARGET_SHAPE[1], TARGET_SHAPE[0])  # (256, 256)
INPUT_IMG_SIZE = 518  # default model-input resolution (37×14; any multiple of 14 runs)


def validate_patch_grid(target_size, patch_size):
    target_size = int(target_size)
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError(f"patch_size={patch_size} must be positive.")
    if target_size <= 0 or target_size % patch_size:
        raise ValueError(
            f"target_size={target_size} must be a positive multiple of patch_size={patch_size}."
        )
    return target_size, patch_size


def to_tensors(batch):
    """`get_data`'s numpy dict -> the torch sample the trainer consumes (dtypes are part of
    the contract). `images` is absent under `defer_input_images` — gpu_augment_batch builds
    it on GPU instead; that missing key IS the signal, so it is passed straight through."""
    sample = {"seq_name": batch["seq_name"]}
    if "images" in batch:
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # [0, 255] HWC -> [0, 1] CHW
        sample["images"] = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)
    sample["scanner_coords"] = torch.from_numpy(np.stack(batch["scanner_coords"]).astype(np.float32))
    sample["z_indices"] = torch.from_numpy(np.stack(batch["z_indices"]).astype(np.float32))
    sample["timesteps"] = torch.from_numpy(np.stack(batch["timesteps"]).astype(np.int64))
    # float32: the re-extraction paths (respiratory grid_sample, gpu_aug) take float z.
    sample["slice_indices"] = torch.from_numpy(np.stack(batch["slice_indices"]).astype(np.float32))
    sample["gt_target_volume"] = torch.from_numpy(batch["gt_target_volume"].astype(np.float32))
    sample["t_target"] = torch.from_numpy(batch["t_target"].astype(np.int64))
    sample["seq_index"] = torch.from_numpy(batch["seq_index"].astype(np.int64))
    sample["dz_mm"] = torch.from_numpy(batch["dz_mm"].astype(np.float32))
    sample["z_scale"] = torch.from_numpy(batch["z_scale"].astype(np.float32))
    sample["anatomy_bbox"] = torch.from_numpy(batch["anatomy_bbox"].astype(np.int64))
    sample["content_mask"] = torch.from_numpy(batch["content_mask"].astype(np.uint8))
    if "heart_roi_canonical" in batch:
        sample["heart_roi_canonical"] = torch.from_numpy(batch["heart_roi_canonical"].astype(np.uint8))
    # Full (T, D, H, W) canonical bundle, kept float16 to keep batch transfer cheap.
    sample["phases"] = torch.from_numpy(np.asarray(batch["phases"]))
    return sample


class MRIDataset(Dataset):
    def __init__(
        self,
        common_conf=None,       # unused; kept so positional callers keep working
        data_root=None,
        split="train",
        split_file=None,
        mode="static",
        num_slices=12,
        target_size=INPUT_IMG_SIZE,
        mri_mode="axial",
        t_target_fixed=None,
        reference_slot=False,
        one_frame_per_slice=False,
        cache_dir=None,
        ef_val_sweep=False,
        cardiac_phase_csv=None,
        defer_input_images=False,
        intensity_percentiles=(0.5, 99.9),
        patch_size=14,
    ):
        """
        Args:
            cache_dir: where monai PersistentDataset stores cached tensors.
                       Defaults to /tmp/vggt-mri_<USER>_monai_cache.
        """
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.split_file = os.path.abspath(split_file) if split_file else None
        self.mode = mode
        self.num_slices = num_slices
        # Input-image resolution R (config `img_size`). `get_data`'s own resize and the
        # `scanner_coords` normalization read it here, and `gpu_aug`/`respiratory` extraction
        # derive R from the batch's own scanner_coords (`R = batch["scanner_coords"].shape[-2]`),
        # so every `images` builder matches. Any multiple of the configured patch size runs —
        # the pretrained position embeddings are interpolated dynamically. Changing R starts
        # a fresh numeric series — it is not a free knob for comparisons.
        self.target_size, self.patch_size = validate_patch_grid(target_size, patch_size)
        # See the block in get_data: skip building `images` because the trainer's
        # gpu_augment_batch re-extracts them on GPU regardless. Training sets this true;
        # anything that calls get_data WITHOUT going through gpu_augment_batch must leave
        # it false (the default) or it will get a batch with no `images` key.
        self.defer_input_images = bool(defer_input_images)
        if len(intensity_percentiles) != 2:
            raise ValueError("intensity_percentiles must be [lower, upper].")
        self.intensity_percentiles = tuple(float(x) for x in intensity_percentiles)
        lower, upper = self.intensity_percentiles
        if not 0 <= lower < upper <= 100:
            raise ValueError(
                f"intensity_percentiles must satisfy 0 <= lower < upper <= 100, got "
                f"{self.intensity_percentiles}."
            )
        self.mri_mode = mri_mode
        self.t_target_fixed = t_target_fixed
        # Reference-slice conditioning: when True, slot 0 is forced to OBSERVE the
        # target phase at the mid-ventricular plane (z = bbox z-center), and the remaining
        # slots are scattered with that plane excluded. The model reads the target phase from
        # slot-0's image content (via the native camera_token anchor). Default False → slot 0
        # is not special.
        self.reference_slot = bool(reference_slot)
        # One-frame-per-slice (the sparse-acquisition extreme): when True, S is forced per subject
        # to the subject's in-FOV plane count so every in-bbox z plane appears EXACTLY once (no
        # multi-frame extras, n_extra=0). Overrides the incoming img_per_seq / num_slices budget;
        # safe because each batch is a single subject (batch_size=1), so the per-slot count can
        # vary across iterations without any cross-subject padding. Default False → the fixed-S
        # multi-frame sampler. Composes with reference_slot (slot 0 = target-phase z_mid = that
        # plane's one frame).
        self.one_frame_per_slice = bool(one_frame_per_slice)

        # ── Subject discovery ─────────────────────────────────────────────
        self.subjects = self._find_subjects()
        logging.info(f"MRIDataset [{split}]: {len(self.subjects)} subjects from {self.split_file}")
        # Exactly one pass over the cohort per epoch. A fixed epoch length not equal to N
        # (e.g. `max(1000, N)`) would draw the same `seq_index % N` residual subjects twice
        # every epoch, a bias that never averages out over seeds or epochs.
        self.len_train = len(self.subjects)

        # ── EF val sweep (opt-in, val-only): reconstruct each subject at its GT ED and ES ──
        # instead of the coupled seq_index%T phase. Builds an explicit (subj_idx, t_target) list
        # of length 2*N (all ED first, then all ES) from cardiac_phase.csv, so seq_index deterministically
        # enumerates an N x {ED,ES} sweep. Needed for the predicted-EF metric.
        self.val_targets = None
        self.cardiac_phase_csv = None
        if ef_val_sweep and self.split.lower() == "val" and self.subjects:
            csv_path = cardiac_phase_csv or os.path.normpath(
                os.path.join(self.data_root, "..", "..", "whs", "cardiac_phase.csv"))
            self.val_targets = self._build_val_targets(csv_path)
            self.cardiac_phase_csv = csv_path
            logging.info(f"MRIDataset [val]: ef_val_sweep ON — {len(self.val_targets)} "
                         f"(subject, t_target) pairs from {csv_path}")
            # An epoch must enumerate SWEEP ENTRIES, not subjects. `val_targets` is
            # `[(i, ED) for every subject] + [(i, ES) for every subject]`, i.e. 2N long, and
            # `get_data` indexes it by `seq_index % len(val_targets)`. Leaving len_train at N
            # caps `__len__` (and therefore the dataloader) at N, so `seq_index` only ever
            # reaches 0..N-1 — the ED half — and EVERY ES entry is silently unreachable
            # (`trainer.py`'s `limit_val_batches` cannot rescue it: a dataloader cannot yield
            # more samples than the dataset declares).
            self.len_train = len(self.val_targets)

        # ── monai PersistentDataset cache ─────────────────────────────────
        # Subdir keyed by content-defining params (spacing/shape/normalization) so a
        # normalization change routes to a fresh cache instead of silently reusing a
        # stale one (PersistentDataset hashes only the input dict, not the transform).
        self.cache_signature = cache_signature(lower, upper)
        cache_dir = os.path.join(cache_dir or default_cache_dir(), self.cache_signature)
        os.makedirs(cache_dir, exist_ok=True)
        data_dicts = build_data_dicts(self.subjects, num_phases=NUM_PHASES)
        self.cache = PersistentDataset(
            data=data_dicts,
            transform=get_canonical_transforms(
                target_spacing=TARGET_SPACING,
                target_shape=TARGET_SHAPE,
                num_phases=NUM_PHASES,
                lower=lower,
                upper=upper,
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
        missing = []
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
                        missing.append(path)
                        logging.warning(f"MRIDataset: subject path not found, skipping: {path}")
        # Post-condition: the split file is the contract for how many subjects
        # this run trains/evaluates on. Warn-and-skip alone means a rename or a GPFS mount
        # hiccup silently shrinks the cohort, with only a startup warning that is easy to lose
        # — and every downstream number (epoch length, val means) would quietly change.
        if missing:
            raise FileNotFoundError(
                f"MRIDataset [{self.split}]: {len(missing)} of "
                f"{len(subjects) + len(missing)} subjects listed in {self.split_file} are "
                f"missing on disk (data_root={self.data_root}). First few: {missing[:5]}. "
                "Fix the paths or edit the split file — do not train on a silently smaller cohort."
            )
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
            # get_data's `% T_total` would silently reconstruct the WRONG phase.
            if not (0 <= ed < NUM_PHASES and 0 <= es < NUM_PHASES):
                raise ValueError(f"ef_val_sweep: {sid} has ED={ed}/ES={es} outside [0,{NUM_PHASES}); "
                                 "incompatible with the canonical 12-phase grid.")
            ed_list.append((i, ed)); es_list.append((i, es))
        if missing:
            raise KeyError(f"ef_val_sweep: {len(missing)} val subjects missing from {csv_path}: {missing}")
        return ed_list + es_list   # all ED, then all ES  ->  seq_index < N => ED, else ES

    def __len__(self):
        return self.len_train

    def __getitem__(self, seq_index):
        """One DataLoader sample: `get_data` at the slot budget `num_slices`, as tensors.

        The budget is passed explicitly (not left to `get_data`'s fallback) because that is
        what arms the `S > budget` guard in `_sample_slots`.
        """
        return to_tensors(self.get_data(seq_index=seq_index, img_per_seq=self.num_slices))

    # ── Main get_data ────────────────────────────────────────────────────
    def get_data(self, seq_index=0, img_per_seq=None):
        forced_t = None
        if self.val_targets is not None:
            subj_idx, forced_t = self.val_targets[seq_index % len(self.val_targets)]
        else:
            subj_idx = seq_index % len(self.subjects)
        sub_dir = self.subjects[subj_idx]

        # Val/test determinism: non-train splits draw only from a private
        # `random.Random(seq_index)` (see `rng` below) — never the global RNG — and the
        # val loader runs shuffle=False, so val get_data is reproducible across epochs/runs.

        # ── Cache lookup → splat-order tensors ────────────────────────────
        cached = self.cache[subj_idx]
        # ConcatItemsd(dim=0) stacks 12 × (1, X, Y, Z) → (T=12, X=256, Y=256, Z=D)
        # (the per-phase channel dim is absorbed into T; D = this subject's own native
        # slice count). content_mask keeps its channel dim: (1, 256, 256, D).
        phases = cached["phases"]                # (T, X, Y, Z)
        content_mask = cached["content_mask"]    # (1, X, Y, Z)
        dz = float(cached["dz_mm"])               # this subject's own native z spacing (mm)
        z_scale = Z_HALF_MM / dz
        # Axis-order conversion site (ONLY here). monai (X,Y,Z) → splat (D=Z,H=Y,W=X).
        phases_splat = phases.permute(0, 3, 2, 1).contiguous()              # (T, D, H=256, W=256)
        mask_splat = content_mask.squeeze(0).permute(2, 1, 0).contiguous()  # (D, H, W)
        T_total, D, H_can, W_can = phases_splat.shape
        assert (H_can, W_can) == CANONICAL_HW, (H_can, W_can)  # D varies per subject; H/W don't

        # ── Geometric anatomy bbox (computed BEFORE z sampling) ───────────
        # Used to restrict z sampling to canonical planes that carry real data
        # (i.e., inside the subject's native FOV). Without this, small-Z subjects
        # waste many slots on zero-padded Z planes — see `_sample_slots`.
        anatomy_bbox = compute_geometric_bbox(mask_splat).cpu().numpy().astype(np.int64)  # (6,)
        bbox_z0, bbox_z1 = int(anatomy_bbox[0]), int(anatomy_bbox[1])

        # RNG order is part of the data contract: in train, t_target is drawn from the global
        # `random` BEFORE any slot draw.
        t_target = self._pick_t_target(seq_index, forced_t, T_total)
        t_sequence, z_sequence = self._sample_slots(
            seq_index, t_target, T_total, bbox_z0, bbox_z1, img_per_seq)
        slots = self._slot_arrays(phases_splat, t_sequence, z_sequence, D, dz)

        # ── V_gt + full phases bundle (for the GPU aug) ───────────────────
        # `anatomy_bbox` was already computed above (used to constrain z sampling).
        gt_target_volume = phases_splat[t_target].float().cpu().numpy()  # (D, H, W) [0, 1] float32
        # phases_full is the full (T, D, H, W) canonical bundle. Kept in float16 to
        # keep batch transfer cheap; the trainer casts to float32 inside aug.
        phases_full = phases_splat.cpu().numpy()  # (T, D, H, W) float16
        content_mask_np = mask_splat.cpu().numpy().astype(np.uint8)  # (D, H, W)
        heart_roi_np = self._heart_roi(sub_dir, (D, H_can, W_can))

        rel_path = os.path.relpath(sub_dir, self.data_root)
        seq_name = f"mri_{self.mri_mode}_{rel_path.replace(os.sep, '_')}"

        return {
            **slots,
            "seq_name": seq_name,
            "gt_target_volume": gt_target_volume,
            "t_target": np.array([t_target], dtype=np.int64),
            "anatomy_bbox": anatomy_bbox,
            "content_mask": content_mask_np,
            **({"heart_roi_canonical": heart_roi_np} if heart_roi_np is not None else {}),
            "phases": phases_full,
            # This subject's own native z spacing (mm) and the derived voxel-index scale
            # (z_scale = Z_HALF_MM / dz) — required by splat.py's push/pull, loss.py's direct
            # splat/sample_volume call sites, and the respiratory mm->voxel conversion.
            "dz_mm": np.array([dz], dtype=np.float32),
            "z_scale": np.array([z_scale], dtype=np.float32),
            # Stable per-sample id → deterministic val respiratory seeding (mirrors
            # the val `random.Random(seq_index)` z/t determinism). See gpu_aug.py.
            "seq_index": np.array([seq_index], dtype=np.int64),
        }

    def _pick_t_target(self, seq_index, forced_t, T_total):
        """Priority: EF-sweep forced phase > single fixed phase > all T phases. Only the
        train branch draws (global `random`); val cycles `seq_index % T`."""
        if forced_t is not None:
            return int(forced_t) % T_total
        if self.t_target_fixed is not None:
            return int(self.t_target_fixed) % T_total
        if self.split != "train":
            return seq_index % T_total
        return random.randrange(T_total)

    def _sample_slots(self, seq_index, t_target, T_total, bbox_z0, bbox_z1, img_per_seq):
        """-> (t_sequence, z_sequence), one entry per input slot.

        Draw order (do not reorder — it defines the data stream): coverage shuffle (only if
        it must be subsampled) → extras `choices` (only if any) → tail shuffle (always) →
        per-slot phase `randrange` (only when mode != "static").
        """
        # ── S = requested slot budget (multi-frame; NOT capped by T or bbox) ──
        # Multi-frame-per-slice allows phase reuse (t with replacement) and plane
        # reuse (extras), so S is not clamped to T_total or the in-FOV z extent.
        # Full z-coverage is GUARANTEED below (every in-bbox plane
        # appears ≥once), so V_canon has no coverage holes and the full-volume L1 loss
        # stays valid.
        S = img_per_seq or self.num_slices

        # ── Build (t, z) slot sequences — multi-frame, full coverage ──────
        # Matches real multi-frame-per-slice acquisition (+ classical SVR, + closer
        # to original VGGT's many-view input):
        #   • Every in-bbox z plane is covered AT LEAST once → full V_canon coverage.
        #   • The remaining slots are EXTRA frames whose planes are drawn UNIFORMLY at
        #     random (with replacement) over the in-bbox planes.
        #   • t per slot is a random phase WITH replacement, used ONLY to extract the
        #     slice's image CONTENT — input cardiac phase is NEVER a model input
        #     (real-time CMR carries no phase label).
        #
        # Train vs val differ ONLY in the RNG source (determinism + no global-RNG
        # leak): train → global `random` (fresh each epoch); val → a private
        # `random.Random(seq_index)` (reproducible across epochs/runs).
        #
        # REFERENCE-SLOT MODE (self.reference_slot): slot 0 OBSERVES the target phase at
        # z_mid; the model reads the target phase from slot-0's image content via the
        # native camera_token anchor. z_mid may still be drawn again as an extra.
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
            #
            # ⚠️ z is never zero-padded, so `anatomy_bbox` z-range is always [0, D) ⇒
            # **S == D exactly**. Memory is therefore not a knob here: batch size is pinned
            # to 1 in data/loader.py, because two subjects with the same D but different
            # pitch would collate SILENTLY and share one z_scale. To cut memory, cut D (or
            # the model). `num_slices` survives as the S cap enforced just below.
            budget = S
            S = len(z_sequence) + len(coverage)
            # S is set by the DATA, so nothing else bounds it: a subject with a larger D
            # than the budget was sized for must fail loudly instead of OOM-ing mysteriously.
            #
            # Gated on `img_per_seq is not None`, i.e. only when the caller passed the budget
            # explicitly (`__getitem__` always does). A bare `get_data()` — tools, tests, the
            # identity gate — falls back to `self.num_slices`, whose default (12) is NOT the
            # training budget (20), so enforcing it there would reject valid D>12 subjects.
            if img_per_seq is not None and S > budget:
                raise ValueError(
                    f"one_frame_per_slice needs S={S} slots for this subject (D={S}), "
                    f"exceeding the configured budget of {budget} (num_slices). Raise "
                    f"num_slices, or exclude the subject."
                )

        room = S - len(z_sequence)
        if len(coverage) > room:                                # S < #planes (e.g. img_per_seq < in-bbox plane count)
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
        if self.reference_slot:
            t_sequence[0] = t_target                            # slot 0 observes the target phase
        return t_sequence, z_sequence

    def _slot_arrays(self, phases_splat, t_sequence, z_sequence, D, dz):
        """Per-slot arrays: `images` (unless deferred), `scanner_coords`, `timesteps`,
        `slice_indices`, `z_indices` — each a list with one entry per slot."""
        S = len(z_sequence)
        images_list = []
        scanner_coords_list = []
        z_indices_list = []
        timesteps_list = []
        slice_indices_list = []

        # Per-pixel canonical (x, y, z) coords for an R×R input image (R = target_size).
        # Bilinear resize 256→`self.target_size` with align_corners semantics: pixel
        # (py, px) of the R×R input corresponds to source 256×256 voxel index
        # (py·255/(R-1), px·255/(R-1)). Normalized [-1, +1]: y_norm = py/(R-1)·2 - 1;
        # same for x. Constant across subjects → compute once.
        R = int(self.target_size)
        py_grid, px_grid = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
        x_norm = (px_grid.astype(np.float32) / (R - 1)) * 2.0 - 1.0
        y_norm = (py_grid.astype(np.float32) / (R - 1)) * 2.0 - 1.0

        # Pre-resize ALL S canonical slices in one batched F.interpolate call.
        # `to_resize` shape (S, 1, 256, 256) float32; output (S, 1, R, R).
        slot_ts = torch.tensor(t_sequence, dtype=torch.long)
        slot_indices = torch.tensor(z_sequence, dtype=torch.long)
        canon_slices = phases_splat[slot_ts, slot_indices].float()  # (S, H=256, W=256)
        # `defer_input_images` (training default): the trainer re-extracts every input slice
        # on GPU anyway — respiratory needs the breathing-displaced reslice, affine needs the
        # warped volume — and `gpu_augment_batch` overwrites `images` wholesale. Building it
        # here first costs S·R·R·3 float32 (≈64 MB at S=20, R=518) of worker CPU, collate, and
        # host→device transfer per sample, every step, for a tensor that is immediately
        # discarded. So skip it and let the trainer produce it. `gpu_augment_batch` treats a
        # MISSING `images` key as "extract unconditionally", including on the no-augmentation
        # path, so this can never silently yield a stale/placeholder tensor.
        # Default False ⇒ standalone callers (tools/, tests, baselines) still get images.
        if not self.defer_input_images:
            canon_slices = canon_slices.unsqueeze(1)                # (S, 1, 256, 256)
            upsampled = F.interpolate(
                canon_slices, size=(R, R),
                mode="bilinear", align_corners=True,
            )                                                        # (S, 1, R, R)
            # Match to_tensors' `/255` contract — keep images in [0, 255].
            upsampled = (upsampled.squeeze(1) * 255.0).clamp(0, 255).cpu().numpy()  # (S, R, R)

        for i in range(S):
            t_idx = t_sequence[i]
            z_i = z_sequence[i]
            if not self.defer_input_images:
                # RGB-replicate to match VGGT model contract (3-channel input).
                img = np.repeat(upsampled[i, ..., None], 3, axis=-1).astype(np.float32)
                images_list.append(img)

            # scanner_coords: per-pixel canonical (x_norm, y_norm, z_norm) for this z.
            # z_norm is PHYSICAL (z_mm / Z_HALF_MM), NOT a fraction of D — D varies per
            # subject, but Z_HALF_MM is the same ruler for everyone.
            # z is measured from THIS subject's own mid-plane ((D-1)/2), at its own native
            # spacing dz.
            z_val = (z_i - (D - 1) / 2.0) * dz / Z_HALF_MM
            # A real `raise`, NOT an `assert` (asserts are stripped under `python -O`), and
            # headroom is small. If this were skipped, |z_norm| > 1 would flow silently into
            # ZIndexEmbedder, whose sinusoids have period 2 — two planes of one subject would
            # alias to the SAME embedding with no crash.
            if abs(z_val) > 1.0 + 1e-4:
                raise ValueError(
                    f"z_norm {z_val:.4f} exceeds Z_HALF_MM={Z_HALF_MM} half-span "
                    f"(dz={dz}, D={D}, z_i={z_i}) — raise Z_HALF_MM, do not crop the stack."
                )
            sc = np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], axis=-1).astype(np.float32)
            scanner_coords_list.append(sc)

            # z index (per-slot scalar embedding).
            z_indices_list.append(np.array([z_val], dtype=np.float32))

            timesteps_list.append(t_idx)
            slice_indices_list.append(z_i)

        return {
            # Omitted entirely when `defer_input_images` — an absent key is the signal
            # gpu_augment_batch keys off, so a stale tensor can never masquerade as input.
            **({} if self.defer_input_images else {"images": images_list}),
            "scanner_coords": scanner_coords_list,
            "timesteps": timesteps_list,
            "slice_indices": slice_indices_list,
            "z_indices": z_indices_list,
        }

    def _heart_roi(self, sub_dir, shape):
        """(D, H, W) uint8 heart ROI, or None when the subject has none (or a stale one)."""
        D, H_can, W_can = shape
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
            roi_candidate = np.ascontiguousarray(
                np.transpose(roi_xyz, (2, 1, 0)) > 0).astype(np.uint8)      # (D, H, W)
            # Some heart_roi_canonical.nii.gz files may be on a stale grid — warn-and-skip
            # rather than assert so a stale ROI never crashes a batch.
            if roi_candidate.shape == (D, H_can, W_can):
                heart_roi_np = roi_candidate
            else:
                logging.warning(
                    f"MRIDataset: heart_roi_canonical shape {roi_candidate.shape} != "
                    f"expected ({D}, {H_can}, {W_can}) for {sub_dir} — likely a stale "
                    f"pre-native-z ROI; skipping (heart_roi_canonical metric omitted this sample)."
                )
        return heart_roi_np
