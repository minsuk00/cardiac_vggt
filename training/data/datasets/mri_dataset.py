"""MRIDataset — VGGT-MRI dataset, native-z canonical-grid edition (docs/58).

Each subject's 12 cine phases live on disk as `sax_frame_{tt:02d}.nii.gz`. A
monai `PersistentDataset` preprocess pipeline (see `training/data/preprocess.py`)
resamples every native NIfTI's IN-PLANE spacing to a fixed 1.4 mm and crops
/zero-pads X/Y to 256×256 — but does NOT resample z: each subject keeps its own
native z spacing/plane count (`dz_mm`, `D`). The cached output is a single
`(T=12, 1, X=256, Y=256, Z=D)` float16 tensor per subject, plus a `(1, X, Y, Z)`
content mask and `dz_mm` (that subject's own z spacing). Cache lives in `/tmp`
(node-local NVMe, fast).

At training time `get_data` just looks up the cached bundle, permutes to splat
order `(T, D, H=256, W=256)`, samples (t_target, S=(t,z)-slots) per the
multi-phase contract, and produces:

    images          (S, 518, 518, 3)  float32, [0, 255]   — bilinear-upsampled
                                                            canonical slices, no
                                                            letterbox, no padding
    scanner_coords  (S, 518, 518, 3)  float32, [-1, +1]   — purely geometric:
                                                            (px, py, z_i) →
                                                            (x_norm, y_norm, z_norm);
                                                            x/y use the same formula
                                                            for every subject, z is
                                                            PHYSICAL (z_mm/Z_HALF_MM)
                                                            so it's also comparable
                                                            across subjects despite D
                                                            varying (docs/58)
    z_indices       (S, 1)   (z_i - (D-1)/2) * dz / Z_HALF_MM
    t_indices       (S, 1)   t_i / T * 2 - 1, T=12  (cyclic — wraps at +1)
    gt_target_volume (D, H, W) = phases_splat[t_target]
    anatomy_bbox    (6,) int64  — (z0, z1, y0, y1, x0, x1) from content_mask
    content_mask    (D, H, W) uint8  — 1 = native FOV reached, 0 = zero-pad (x/y only now)
    phases          (T, D, H, W) float16 — full canonical bundle, needed by
                                          the Phase 4 GPU aug to augment all 12
                                          phases consistently then re-extract
                                          slices + V_gt + bbox.
    t_target        (1,) int64
    dz_mm           (1,) float32 — this subject's own native z spacing
    z_scale         (1,) float32 — Z_HALF_MM / dz_mm; required by splat.py

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
    Z_HALF_MM,
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
# In-plane shape (H, W) — FIXED for every subject (native-z only stops resampling Z;
# X/Y are still resampled to this same 256×256 grid for everyone). There is no
# fixed D anymore: each subject's canonical grid is (D, 256, 256) with D = that
# subject's own native slice count (docs/58).
CANONICAL_HW = (TARGET_SHAPE[1], TARGET_SHAPE[0])  # (256, 256)
INPUT_IMG_SIZE = 518  # default model-input resolution (37×14; any multiple of 14 runs — docs/73)


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
        t_target_fixed=None,
        t_target_phases=None,
        reference_slot=False,
        continuous_z=False,
        one_frame_per_slice=False,
        z_jitter=0.5,
        cache_dir=None,
        ef_val_sweep=False,
        cardiac_phase_csv=None,
        defer_input_images=False,
        intensity_percentiles=(0.5, 99.9),
        patch_size=14,
    ):
        """
        Args mirrors the legacy MRIDataset for Hydra-config compatibility.
        New args:
            cache_dir: where monai PersistentDataset stores cached tensors.
                       Defaults to /tmp/vggt-mri_<USER>_monai_cache.
        """
        super().__init__()
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.split_file = os.path.abspath(split_file) if split_file else None
        self.mode = mode
        self.num_slices = num_slices
        # Input-image resolution R (config `img_size`). Fully threaded since the native-splat
        # port (2026-08-13): `get_data`'s own resize and the `scanner_coords` normalization
        # read it here, and `gpu_aug`/`respiratory` extraction derive R from the batch's own
        # scanner_coords (`R = batch["scanner_coords"].shape[-2]`), so every `images`
        # builder matches. Any
        # multiple of the configured patch size runs — the pretrained position embeddings
        # are interpolated dynamically. 224 is trained/validated for DINOv2 (docs/72); changing R starts
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

        # ── Subject discovery (same split-file format as before) ──────────
        self.subjects = self._find_subjects()
        logging.info(f"MRIDataset [{split}]: {len(self.subjects)} subjects from {self.split_file}")
        # Exactly one pass over the cohort per epoch. NOT `max(1000, N)` (docs/59 F6): with
        # N=940 that gave 1000 draws indexed `seq_index % 940`, so the 60-subject residual
        # `subj_idx ∈ 0..59` was drawn TWICE every epoch — a set that is invariant to seed and
        # epoch, so it never averages out. Since the split file is written `sorted()` and
        # `ACDC_sax/…` sorts first, those 60 were all ACDC, oversampling the finest-pitch,
        # pathology-labelled source 1.60× (14.5% of samples on 9.0% of subjects) — precisely
        # the imbalance pooling was meant to remove.
        self.len_train = len(self.subjects)

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
            # An epoch must enumerate SWEEP ENTRIES, not subjects. `val_targets` is
            # `[(i, ED) for every subject] + [(i, ES) for every subject]`, i.e. 2N long, and
            # `get_data` indexes it by `seq_index % len(val_targets)`. Leaving len_train at N
            # caps `__len__` (and therefore the dataloader) at N, so `seq_index` only ever
            # reaches 0..N-1 — the ED half — and EVERY ES entry is silently unreachable.
            # EF = (EDV - ESV)/EDV, so that leaves the predicted-EF metric with no ES volume
            # and nothing to compute from; `trainer.py`'s `limit_val_batches = len(val_targets)`
            # cannot rescue it, because a dataloader cannot yield more samples than the dataset
            # declares. Measured before this line existed: ES half reached 0/133, and only 133
            # of the expected 266 volumes were written to `ef_tmp/pred/`.
            self.len_train = len(self.val_targets)

        # ── monai PersistentDataset cache ─────────────────────────────────
        if PersistentDataset is None:
            raise RuntimeError(
                "monai is required for canonical-grid MRIDataset — pip install monai>=1.6,<1.7"
            )
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
        # Post-condition (docs/59 F17): the split file is the contract for how many subjects
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
            idx_N: Tuple containing (seq_index, img_per_seq). `img_per_seq` is the slot
                BUDGET (the docs/59 F19 cap), not the slot count — under
                `one_frame_per_slice` the count is this subject's own plane count D.
                (A third `aspect_ratio` element was dropped 2026-08-01: it was always 1.0
                and landed unread in `get_data(**kwargs)`.)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq = idx_N
        return self.get_data(seq_index=seq_index, img_per_seq=img_per_seq)

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
        # ConcatItemsd(dim=0) stacks 12 × (1, X, Y, Z) → (T=12, X=256, Y=256, Z=D)
        # (the per-phase channel dim is absorbed into T; D = this subject's own native
        # slice count under native-z). content_mask keeps its channel dim: (1, 256, 256, D).
        phases = cached["phases"]                # (T, X, Y, Z)  [or (T, 1, X, Y, Z) if shape ever changes]
        content_mask = cached["content_mask"]    # (1, X, Y, Z)
        dz = float(cached["dz_mm"])               # this subject's own native z spacing (mm)
        z_scale = Z_HALF_MM / dz
        if phases.ndim == 5:                     # defensive: tolerate a channel dim
            phases = phases.squeeze(1)
        # Axis-order conversion site (ONLY here). monai (X,Y,Z) → splat (D=Z,H=Y,W=X).
        phases_splat = phases.permute(0, 3, 2, 1).contiguous()              # (T, D, H=256, W=256)
        mask_splat = content_mask.squeeze(0).permute(2, 1, 0).contiguous()  # (D, H, W)
        T_total, D, H_can, W_can = phases_splat.shape
        assert (H_can, W_can) == CANONICAL_HW, (H_can, W_can)  # D varies per subject; H/W don't

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
            #
            # ⚠️ CONSEQUENCE (docs/59 F9): under native-z, z is never zero-padded, so
            # `anatomy_bbox` z-range is always [0, D) ⇒ **S == D exactly**. That makes
            # `num_slices` / `img_nums` stale as descriptions of the slot budget, and — the
            # operationally important part — the memory budget is no longer a knob at all:
            # `max_img_per_gpu` was DELETED (docs/59 F9) and batch size is pinned to 1 in
            # dynamic_dataloader, because under native-z two subjects with the same D but
            # different pitch collate SILENTLY and would share one z_scale. To cut memory, cut
            # D (or the model). `img_nums` survives as the S cap enforced just below.
            budget = S
            S = len(z_sequence) + len(coverage)
            # docs/59 F19: S is now set by the DATA, so nothing else bounds it. Max D in the
            # current train/val split is 18, but the pool holds D=19/20/21 subjects (all in
            # test today) — a re-seeded split would silently request more slots than the
            # memory budget was sized for. Fail loudly instead of OOM-ing mysteriously.
            #
            # Gated on `img_per_seq is not None`, i.e. only when the REAL dataloader supplied
            # the budget (from img_nums). Standalone construction — tools, tests, the identity
            # gate — falls back to `self.num_slices`, whose default (12) is NOT the training
            # budget (20), so enforcing it there would reject perfectly valid D>12 subjects.
            if img_per_seq is not None and S > budget:
                raise ValueError(
                    f"one_frame_per_slice needs S={S} slots for this subject (D={S}), "
                    f"exceeding the configured budget of {budget} (img_nums). Raise img_nums, "
                    f"or exclude the subject."
                )

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
        timesteps_list = []
        slice_indices_list = []

        # Per-pixel canonical (x, y, z) coords for a 518×518 input image.
        # Bilinear resize 256→`self.target_size` with align_corners semantics: pixel
        # (py, px) of the R×R input corresponds to source 256×256 voxel index
        # (py·255/(R-1), px·255/(R-1)). Normalized [-1, +1]: y_norm = py/(R-1)·2 - 1;
        # same for x. Constant across subjects → compute once.
        R = int(self.target_size)
        py_grid, px_grid = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
        x_norm = (px_grid.astype(np.float32) / (R - 1)) * 2.0 - 1.0
        y_norm = (py_grid.astype(np.float32) / (R - 1)) * 2.0 - 1.0

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
            # Match ComposedDataset's `/255` contract — keep images in [0, 255].
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
            # subject under native-z, but Z_HALF_MM is the same ruler for everyone (docs/58).
            # z is measured from THIS subject's own mid-plane ((D-1)/2), at its own native
            # spacing dz.
            z_val = (z_i - (D - 1) / 2.0) * dz / Z_HALF_MM
            # A real `raise`, NOT an `assert` (docs/59 F18): asserts are stripped under
            # `python -O`, and there is only 5.9% headroom here (max half-span over the
            # pooled cohort is 85.0 mm of 90). If this were skipped, |z_norm| > 1 would flow
            # silently into ZIndexEmbedder, whose sinusoids have period 2 — two planes of one
            # subject would alias to the SAME embedding with no crash. One protocol step away:
            # D=20 @10mm = 190mm, or D=17 @12mm = 192mm.
            if abs(z_val) > 1.0 + 1e-4:
                raise ValueError(
                    f"z_norm {z_val:.4f} exceeds Z_HALF_MM={Z_HALF_MM} half-span "
                    f"(dz={dz}, D={D}, z_i={z_i}) — raise Z_HALF_MM, do not crop the stack."
                )
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

            timesteps_list.append(t_idx)
            slice_indices_list.append(z_i)

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
            roi_candidate = np.ascontiguousarray(
                np.transpose(roi_xyz, (2, 1, 0)) > 0).astype(np.uint8)      # (D, H, W)
            # heart_roi_canonical.nii.gz files predate native-z and may still be on the
            # OLD fixed (256, 256, 12) grid — warn-and-skip rather than assert, so the ROI
            # regeneration (workstream: extend assemble_whs.py to all 5 sources) can lag
            # behind this code change without crashing every batch in the meantime.
            if roi_candidate.shape == (D, H_can, W_can):
                heart_roi_np = roi_candidate
            else:
                logging.warning(
                    f"MRIDataset: heart_roi_canonical shape {roi_candidate.shape} != "
                    f"expected ({D}, {H_can}, {W_can}) for {sub_dir} — likely a stale "
                    f"pre-native-z ROI; skipping (heart_roi_canonical metric omitted this sample)."
                )

        # ARM corseg-dice: per-phase GT labels at THIS sample's t_target, for the CorSeg
        # soft-Dice loss (training/corseg_dice.py). On-disk (X, Y, Z, T=12) uint8, labels
        # 1=LV_cav / 2=LV_myo / 3=RV (nnU-Net convention — corseg_dice remaps). Same
        # (D, H, W) splat-order transpose and same warn-and-skip shape policy as the ROI
        # above; the loss raises when corseg_weight > 0 and the key is absent.
        heart_seg_path = os.path.join(sub_dir, "heart_seg_canonical.nii.gz")
        heart_seg_np = None
        if os.path.exists(heart_seg_path):
            seg_xyzt = np.asarray(nib.load(heart_seg_path).dataobj)
            if seg_xyzt.ndim == 4 and t_target < seg_xyzt.shape[3]:
                seg_candidate = np.ascontiguousarray(
                    np.transpose(seg_xyzt[..., t_target], (2, 1, 0))).astype(np.uint8)  # (D, H, W)
                if seg_candidate.shape == (D, H_can, W_can):
                    heart_seg_np = seg_candidate
                else:
                    logging.warning(
                        f"MRIDataset: heart_seg_canonical phase shape {seg_candidate.shape} != "
                        f"expected ({D}, {H_can}, {W_can}) for {sub_dir} — skipping "
                        f"(heart_seg_t omitted this sample)."
                    )
            else:
                logging.warning(
                    f"MRIDataset: heart_seg_canonical ndim/T unexpected "
                    f"(shape {seg_xyzt.shape}, t_target {t_target}) for {sub_dir} — skipping."
                )

        rel_path = os.path.relpath(sub_dir, self.data_root)
        seq_name = f"mri_{self.mri_mode}_{rel_path.replace(os.sep, '_')}"

        return {
            # Omitted entirely when `defer_input_images` — an absent key is the signal
            # gpu_augment_batch keys off, so a stale tensor can never masquerade as input.
            **({} if self.defer_input_images else {"images": images_list}),
            "scanner_coords": scanner_coords_list,
            "timesteps": timesteps_list,
            "slice_indices": slice_indices_list,
            "z_indices": z_indices_list,
            "t_indices": t_indices_list,
            "target_t_indices": target_t_indices_list,
            "seq_name": seq_name,
            "gt_target_volume": gt_target_volume,
            "t_target": np.array([t_target], dtype=np.int64),
            "anatomy_bbox": anatomy_bbox,
            "content_mask": content_mask_np,
            **({"heart_roi_canonical": heart_roi_np} if heart_roi_np is not None else {}),
            **({"heart_seg_t": heart_seg_np} if heart_seg_np is not None else {}),
            "phases": phases_full,
            # This subject's own native z spacing (mm) and the derived voxel-index scale
            # (z_scale = Z_HALF_MM / dz) — required by splat.py's push/pull, loss.py's direct
            # splat/sample_volume call sites, and the respiratory mm->voxel conversion (docs/58).
            "dz_mm": np.array([dz], dtype=np.float32),
            "z_scale": np.array([z_scale], dtype=np.float32),
            # Stable per-sample id → deterministic val respiratory seeding (mirrors
            # the val `random.Random(seq_index)` z/t determinism). See gpu_aug.py.
            "seq_index": np.array([seq_index], dtype=np.int64),
        }
