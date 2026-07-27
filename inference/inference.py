"""Shared RTFB inference primitives (batch -> reconstructed volume).

One place for: building the z-only + target_t VGGT model, a single target-phase forward
(with the universal V_refined -> V_canon -> splat(world_points) precedence), and the
target_t phase sweep that produces the beating-heart stack. Dataset-agnostic — the adapter
already normalized the batch.
"""
import numpy as np
import torch
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local
from vggt.utils.splat import splat_predictions
from inference.adapters.base import (
    GRID_SHAPE, DEFAULT_CKPT, DEFAULT_CKPT_REFERENCE, INPUT_IMG_SIZE, to_canonical_inplane,
)


def load_rtfb_model(ckpt=DEFAULT_CKPT, *, refiner=False, device="cuda"):
    """RETIRED — the legacy target_t-index model. Use `load_rtfb_model_reference` instead.

    The target_t conditioning this loader was built for (`TIndexEmbedder` /
    `target_t_embedder`, `use_t/target_t_pose_embedding`) was removed in docs/48: it is the
    content-free phase index that regressed every patient's EF to the cohort mean (docs/24,
    docs/25), replaced by the reference-slot camera-token anchor. `VGGT.__init__` now absorbs
    the retired flags via `**kwargs`, so this function would still *construct* and *load* —
    but the resulting model reads no phase query at all, and `phase_sweep()` would return 12
    IDENTICAL volumes (a non-beating "beating heart") with no error. Its missing-weights guard
    is likewise vacuous, because the model no longer has a `target_t_embedder` to be missing.
    Failing loudly instead of returning that model silently.
    """
    raise NotImplementedError(
        "load_rtfb_model() is retired: target_t-index conditioning was removed in docs/48, so "
        "this model would ignore the target phase and phase_sweep() would yield 12 identical "
        "volumes. Use load_rtfb_model_reference() (reference-slot conditioning) instead."
    )


def load_rtfb_model_reference(ckpt=DEFAULT_CKPT_REFERENCE, *, refiner=False, device="cuda"):
    """Reference-slot z-only VGGT-MRI model (docs/25 + docs/28): slot 0 is the native
    camera_token anchor (`use_reference_token`) instead of a content-free `target_t` index —
    the query is read from slot-0's real image content via `reference_sweep`, not a scalar."""
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False,
        use_target_t_pose_embedding=False, use_reference_token=True, train_on_residual_dvf=True,
        enable_refiner=refiner, refiner_use_coverage=refiner, grid_shape=GRID_SHAPE,
    )
    ck = torch.load(stage_checkpoint_to_local(ckpt), map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if any(s in k for s in
           ("aggregator", "point_head", "refiner", "z_embedder"))]
    if bad:
        raise RuntimeError(f"missing critical weights: {bad[:5]} ...")
    print(f"  loaded {ckpt}  (reference-slot, refiner={refiner}, missing={len(missing)}, unexpected={len(unexpected)})",
          flush=True)
    return model.to(device).eval()


@torch.no_grad()
def forward(model, batch, *, target_t=-1.0, want=("V",), device="cuda", grid_shape=GRID_SHAPE):
    """One target-phase query. Returns a dict with the requested keys (each batched-out `[0]`).

    `want` is any subset of {"V","V_canon","V_refined","world_points","coverage"}.
    "V" = V_refined if the model produced it, else V_canon (splatted from world_points when
    the no-refiner model doesn't emit it) — the universal per-script `_forward` result.

    (No V_gt: these OOD datasets are prospectively acquired and have no ground-truth volume.
    For the in-dist val path, GT comes from the training loss, not this helper.)
    """
    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), target_t, dtype=torch.float32, device=device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)

    wp = preds["world_points"].float()
    V_canon = preds.get("V_canon")
    cov = preds.get("coverage")
    if V_canon is None:                                  # no-refiner model: splat here
        V_canon, cov = splat_predictions({"world_points": wp}, batch, grid_shape)
    V_ref = preds.get("V_refined")

    out = {}
    if "V_canon" in want:
        out["V_canon"] = V_canon[0].float().cpu().numpy()
    if "V_refined" in want:
        out["V_refined"] = V_ref[0].float().cpu().numpy() if V_ref is not None else None
    if "V" in want:
        out["V"] = (V_ref if V_ref is not None else V_canon)[0].float().cpu().numpy()
    if "world_points" in want:
        out["world_points"] = wp[0].cpu().numpy()
    if "coverage" in want:
        out["coverage"] = cov[0].cpu().numpy() if cov is not None else None
    return out


@torch.no_grad()
def phase_sweep(model, batch, *, n_phases=12, return_world_points=False,
                device="cuda", grid_shape=GRID_SHAPE):
    """Sweep target_t over `n_phases` -> (vols, wp_by_t|None). Same inputs, varying query.

    t_norm = t / max(1, n_phases) * 2 - 1  (ED phase 0 -> -1.0), matching training +
    the original reconstruct_cycle / goettingen_infer loops.
    """
    want = ("V", "world_points") if return_world_points else ("V",)
    vols, wp_by_t = [], []
    for t in range(n_phases):
        t_norm = t / max(1, n_phases) * 2.0 - 1.0
        r = forward(model, batch, target_t=t_norm, want=want, device=device, grid_shape=grid_shape)
        vols.append(r["V"])
        if return_world_points:
            wp_by_t.append(r["world_points"])
    return vols, (wp_by_t if return_world_points else None)


@torch.no_grad()
def reference_sweep(model, batch, ref_ctx, *, return_world_points=False,
                    device="cuda", grid_shape=GRID_SHAPE):
    """Reference-slot analogue of `phase_sweep` (docs/25 + docs/28): sweeps slot 0 over REAL
    acquired frames at the reference z-plane instead of a synthetic `target_t` scalar, mirroring
    the trainer's cardiac-cycle filmstrip (`_log_cardiac_cycle_filmstrip`) but fed from the
    dataset's own real-time cine rather than the cached phase bundle.

    -> (vols, wp_by_t|None, frame_indices) — `ref_ctx["frame_indices"]` (set by
    `build_batch_multiframe`'s `frames_for_reference`), the single source of truth for which
    real frames get swept.
    """
    frame_indices = ref_ctx["frame_indices"]
    vmin, vmax = ref_ctx["scale"]
    inplane = ref_ctx["inplane"]
    cine_slice = ref_ctx["cine_slice"]  # (n_frames, H, W)

    want = ("V", "world_points") if return_world_points else ("V",)
    vols, wp_by_t = [], []
    for f in frame_indices:
        norm = np.clip((cine_slice[f] - vmin) / (vmax - vmin), 0.0, 1.0)
        canon = to_canonical_inplane(norm, inplane)
        up = F.interpolate(canon[None, None], size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                           mode="bilinear", align_corners=True)[0, 0].numpy()
        batch["images"][:, 0] = torch.from_numpy(up).to(device).repeat(3, 1, 1)
        r = forward(model, batch, want=want, device=device, grid_shape=grid_shape)
        vols.append(r["V"])
        if return_world_points:
            wp_by_t.append(r["world_points"])
    return vols, (wp_by_t if return_world_points else None), frame_indices
