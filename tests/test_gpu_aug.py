"""Tests for the batchaug GPU augmentation pipeline (training/data/gpu_aug.py).

Run on CPU — batchaug's pytorch backend works without CUDA.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import data.gpu_aug as gpu_aug
from data.gpu_aug import (
    build_gpu_transforms,
    extract_slices_from_phases,
    gpu_augment_batch,
    recompute_bbox_gpu,
)
from data.respiratory import RespiratoryConfig

DEVICE = "cpu"


def _fake_batch(B=2, T=12, D=12, H=256, W=256, S=8):
    """Synthetic batch. `D` is parametrizable because under native-z (docs/58) every
    subject keeps its own slice count (5-21 across the pooled cohort) — the aug path must
    be D-agnostic. The z-extent of the content mask/bbox is derived from `D`, and
    `slice_indices` are wrapped into range, so the default D=12 reproduces the pre-native-z
    values exactly (mask z 1:11, bbox z (1,11), slice_indices 0..S-1)."""
    z0, z1 = 1, D - 1
    phases = torch.rand(B, T, D, H, W, dtype=torch.float16)
    content_mask = torch.zeros(B, D, H, W, dtype=torch.uint8)
    content_mask[:, z0:z1, 30:230, 5:251] = 1
    return {
        "phases": phases,
        "content_mask": content_mask,
        "gt_target_volume": torch.rand(B, D, H, W),
        "anatomy_bbox": torch.tensor([[z0, z1, 30, 230, 5, 251]] * B, dtype=torch.int64),
        "t_target": torch.tensor([[0], [5]][:B], dtype=torch.int64),
        "timesteps": torch.tensor([list(range(S))] * B, dtype=torch.int64),
        "slice_indices": torch.tensor([[i % D for i in range(S)]] * B, dtype=torch.int64),
        "images": torch.rand(B, S, 3, 518, 518),
        "scanner_coords": torch.rand(B, S, 518, 518, 3),
        "seq_index": torch.tensor([[7], [9]][:B], dtype=torch.int64),
        # dz_mm = 12.0 matches SPACING_MM's D-axis default exactly, so respiratory
        # numerics here are unchanged from before native-z (D=12 in this synthetic batch).
        "dz_mm": torch.tensor([[12.0]] * B, dtype=torch.float32),
    }


def _resp_cfg(enable=True, **kw):
    return RespiratoryConfig(enable=enable, **kw)


# ── build_gpu_transforms ──────────────────────────────────────────────────────

def test_build_returns_none_when_disabled():
    assert build_gpu_transforms(OmegaConf.create({"enable": False})) is None
    assert build_gpu_transforms(None) is None

def test_build_returns_compose_when_enabled():
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "conservative"}))
    assert t is not None

def test_build_moderate_tier_builds():
    """Moderate tier: in-plane only, ±180° rotation. Gaussian noise is DISABLED (wrong artifact
    model) and flip is AGGRESSIVE-ONLY as of 2026-08-01 (it was briefly on in every tier,
    2026-07-31; moderate is the arm docs/46 §3 C2 measured and shipped, which had no flip), so
    3 active transforms: affine, contrast, bias-field."""
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "moderate"}))
    assert t is not None
    assert len(t.transforms) == 3
    assert type(t.transforms[0]).__name__ == "RandAffined"


def test_flip_is_aggressive_only():
    """Flip is a vector-field symmetry (needs a coupled Δx sign negation), so it is confined to
    the aggressive tier; conservative/moderate must stay flip-free."""
    for tier in ("conservative", "moderate"):
        t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": tier}))
        names = [type(x).__name__ for x in t.transforms]
        assert "RandFlipd" not in names, f"{tier} tier must not flip"
    agg = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "aggressive"}))
    assert type(agg.transforms[0]).__name__ == "RandFlipd"

def test_build_unknown_tier_raises():
    with pytest.raises(ValueError):
        build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "bogus"}))


# ── gpu_augment_batch identity passthrough ────────────────────────────────────

def test_identity_passthrough_when_none():
    batch = _fake_batch()
    pre = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = gpu_augment_batch(batch, None, DEVICE)
    for k in ["phases", "images", "gt_target_volume", "anatomy_bbox", "content_mask"]:
        assert torch.equal(out[k], pre[k]), f"{k} changed under identity passthrough"


# ── gpu_augment_batch with conservative pipeline ──────────────────────────────

def test_aug_preserves_shapes_and_ranges():
    batch = _fake_batch()
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "conservative"}))
    out = gpu_augment_batch(batch, t, DEVICE)
    B, S = 2, 8
    assert out["phases"].shape == (B, 12, 12, 256, 256)
    assert out["images"].shape == (B, S, 3, 518, 518)
    assert out["gt_target_volume"].shape == (B, 12, 256, 256)
    assert out["anatomy_bbox"].shape == (B, 6)
    assert out["content_mask"].shape == (B, 12, 256, 256)
    # Images re-extracted and normalized to [0, 1].
    assert float(out["images"].min()) >= 0.0 and float(out["images"].max()) <= 1.0

@pytest.mark.parametrize("D", [5, 7, 12, 21])
@pytest.mark.parametrize("tier", ["conservative", "moderate", "aggressive"])
def test_aug_is_native_z_agnostic(D, tier):
    """Under native-z every subject has its own D (5-21 across the pooled cohort, docs/58),
    and augmentation is ON by default as of 2026-07-31 — so the aug path must carry D
    through unchanged for every tier, not just the legacy D=12 cube."""
    batch = _fake_batch(D=D)
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": tier}))
    out = gpu_augment_batch(batch, t, DEVICE)
    B, S = 2, 8
    assert out["phases"].shape == (B, 12, D, 256, 256)
    assert out["gt_target_volume"].shape == (B, D, 256, 256)
    assert out["content_mask"].shape == (B, D, 256, 256)
    assert out["images"].shape == (B, S, 3, 518, 518)
    # Recomputed bbox must stay inside this subject's own depth, not a hardcoded 12.
    for b in range(B):
        z0, z1 = out["anatomy_bbox"][b][:2].tolist()
        assert 0 <= z0 < z1 <= D, f"bbox z ({z0},{z1}) outside D={D}"


def test_aug_recomputes_bbox_validly():
    batch = _fake_batch()
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "conservative"}))
    out = gpu_augment_batch(batch, t, DEVICE)
    for b in range(out["anatomy_bbox"].shape[0]):
        z0, z1, y0, y1, x0, x1 = out["anatomy_bbox"][b].tolist()
        assert 0 <= z0 < z1 <= 12
        assert 0 <= y0 < y1 <= 256
        assert 0 <= x0 < x1 <= 256


def test_conservative_tier_has_no_through_plane_spatial_op():
    """Regression guard for the through-plane rotation bug.

    The conservative tier must never move intensity ACROSS Z (D) planes: no
    through-plane rotation, translation, or scale. We confine content to a few
    D-planes, apply the spatial aug 10× at prob=1, and assert no mass leaks into
    the empty planes. (batchaug's rotate_range is positional by plane-of-rotation,
    so a wrong slot silently produces through-plane rotation — this catches it.)
    """
    import batchaug as B
    import numpy as np

    keys = ["phases"]
    mode = {"phases": "bilinear"}
    # Spatial-only conservative ops (drop photometric, which don't move mass).
    spatial = B.Compose(transforms=[
        B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
        B.RandAffined(keys=keys, prob=1.0,
                      rotate_range=(float(np.deg2rad(5)), 0.0, 0.0),
                      translate_range=(0.0, 4.0, 4.0),
                      scale_range=(0.0, 0.05, 0.05),
                      padding_mode="zeros"),
    ], lazy=True, mode=mode)

    content_planes = list(range(3, 9))
    empty_planes = [0, 1, 2, 9, 10, 11]
    for _ in range(10):
        phases = torch.zeros(1, 12, 12, 256, 256)
        phases[:, :, 3:9, 60:200, 60:200] = 1.0
        out = spatial({"phases": phases})["phases"]
        leak = out[0, 0][empty_planes].abs().sum().item()
        assert leak < 1.0, f"through-plane leak detected: {leak:.1f} mass in empty Z-planes"


# ── helpers ───────────────────────────────────────────────────────────────────

def test_recompute_bbox_gpu_tight():
    mask = torch.zeros(12, 256, 256)
    mask[2:9, 50:200, 10:240] = 1
    bb = recompute_bbox_gpu(mask)
    assert bb.tolist() == [2, 9, 50, 200, 10, 240]

def test_recompute_bbox_gpu_empty_fallback():
    bb = recompute_bbox_gpu(torch.zeros(12, 256, 256))
    assert bb.tolist() == [0, 12, 0, 256, 0, 256]

def test_extract_slices_shapes_and_indexing():
    B, T, D, H, W, S = 2, 12, 12, 256, 256, 8
    phases = torch.rand(B, T, D, H, W)
    t_seq = torch.tensor([list(range(S))] * B, dtype=torch.int64)
    z_seq = torch.tensor([list(range(S))] * B, dtype=torch.int64)
    imgs = extract_slices_from_phases(phases, t_seq, z_seq)
    assert imgs.shape == (B, S, 518, 518, 3)
    assert float(imgs.max()) <= 255.0 and float(imgs.min()) >= 0.0


def test_extract_slices_float_z_exact_at_integer_and_interpolates():
    """continuous-z support: float z_seq must (a) be accepted and (b) match the integer gather
    EXACTLY at integer-valued z (so the discrete-grid pipeline is numerically unchanged), and
    (c) linearly interpolate between bracketing planes at fractional z."""
    B, T, D, H, W, S = 1, 12, 12, 256, 256, 4
    phases = torch.rand(B, T, D, H, W)
    t_seq = torch.zeros(B, S, dtype=torch.int64)
    # (a)+(b): integer-valued FLOAT z must equal the int64 gather byte-for-byte.
    z_int = torch.tensor([[2, 5, 7, 9]], dtype=torch.int64)
    out_int = extract_slices_from_phases(phases, t_seq, z_int)
    out_flt = extract_slices_from_phases(phases, t_seq, z_int.float())
    assert torch.allclose(out_int, out_flt), "float z at integer values must match the integer gather"
    # (c): z=5.5 must be the mean of planes 5 and 6 (linear blend), checked on the raw plane.
    z_half = torch.tensor([[5.5]], dtype=torch.float32)
    blended = extract_slices_from_phases(phases, torch.zeros(1, 1, dtype=torch.int64), z_half)
    expect = 0.5 * phases[0, 0, 5] + 0.5 * phases[0, 0, 6]   # (H, W)
    expect_up = torch.nn.functional.interpolate(
        (expect * 255.0).clamp(0, 255)[None, None], size=(518, 518), mode="bilinear", align_corners=True
    )[0, 0]
    assert torch.allclose(blended[0, 0, :, :, 0], expect_up, atol=1e-3), "fractional z must linearly blend planes"


# ── respiratory integration in gpu_augment_batch ──────────────────────────────

_REF_KEYS = ["phases", "gt_target_volume", "anatomy_bbox", "content_mask", "scanner_coords"]


def test_resp_disabled_is_identity():
    """respiratory_cfg.enable=False with affine off → batch returned unchanged, no
    resp_disp_mm surfaced (training stays bit-identical when breathing is off)."""
    batch = _fake_batch()
    pre = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=_resp_cfg(enable=False), train=True)
    for k in ["phases", "images", "gt_target_volume", "anatomy_bbox", "content_mask"]:
        assert torch.equal(out[k], pre[k]), f"{k} changed with respiratory disabled"
    assert "resp_disp_mm" not in out  # no diagnostic key leaks when breathing is off


def test_resp_on_changes_only_images():
    """HEADLINE: respiratory overwrites ONLY images; the reference fields stay put."""
    batch = _fake_batch()
    pre = {k: batch[k].clone() for k in _REF_KEYS}
    pre_images = batch["images"].clone()
    g = torch.Generator(device=DEVICE).manual_seed(0)
    out = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=_resp_cfg(), train=True, resp_generator=g)
    for k in _REF_KEYS:
        assert torch.equal(out[k], pre[k]), f"{k} must stay at the reference under respiratory"
    # Prove breathing actually MOVED content: the output must differ from the
    # zero-displacement (clean) reslice of the SAME phases — not merely from the
    # random seed images — AND a nonzero breath must have been drawn.
    clean = extract_slices_from_phases(out["phases"].float(), out["timesteps"], out["slice_indices"])
    clean = clean.permute(0, 1, 4, 2, 3).contiguous() / 255.0
    assert not torch.equal(out["images"], clean)               # breathing shifted the anatomy
    assert out["images"].shape == pre_images.shape
    assert float(out["images"].min()) >= 0.0 and float(out["images"].max()) <= 1.0
    # The per-slot displacement is surfaced for diagnostics (captions + scalar).
    B, S = pre_images.shape[0], pre_images.shape[1]
    assert out["resp_disp_mm"].shape == (B, S, 3)
    assert float(out["resp_disp_mm"].abs().max()) > 0.0        # a real (nonzero) breath
    assert torch.isfinite(out["resp_disp_mm"]).all()


def test_resp_val_deterministic_per_seq_index():
    cfg = _resp_cfg()
    batch = _fake_batch()                                       # same volume both passes
    a = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=cfg, train=False)["images"].clone()
    b = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=cfg, train=False)["images"].clone()
    assert torch.equal(a, b)                                    # same seq_index → identical breath


def test_resp_val_requires_seq_index():
    batch = _fake_batch()
    del batch["seq_index"]
    with pytest.raises(ValueError):
        gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=_resp_cfg(), train=False)


def test_resp_train_iid_across_calls():
    cfg = _resp_cfg()
    g = torch.Generator(device=DEVICE).manual_seed(123)
    batch = _fake_batch()                                       # same volume both passes
    a = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=cfg, train=True, resp_generator=g)["images"].clone()
    b = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=cfg, train=True, resp_generator=g)["images"].clone()
    assert not torch.equal(a, b)                                # generator advances → fresh breath


def test_affine_plus_resp_single_extraction(monkeypatch):
    """With both augs on, images are extracted exactly once (the respiratory path);
    the affine slice-extractor must NOT also run (wasted + discarded)."""
    calls = {"plain": 0, "resp": 0}
    real_plain = gpu_aug.extract_slices_from_phases
    real_resp = gpu_aug.extract_slices_with_respiratory_vec

    def spy_plain(*a, **k):
        calls["plain"] += 1
        return real_plain(*a, **k)

    def spy_resp(*a, **k):
        calls["resp"] += 1
        return real_resp(*a, **k)

    monkeypatch.setattr(gpu_aug, "extract_slices_from_phases", spy_plain)
    monkeypatch.setattr(gpu_aug, "extract_slices_with_respiratory_vec", spy_resp)

    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "conservative"}))
    g = torch.Generator(device=DEVICE).manual_seed(1)
    out = gpu_augment_batch(_fake_batch(), t, DEVICE, respiratory_cfg=_resp_cfg(), train=True, resp_generator=g)
    # ONE resp extraction, at native resolution (-> images_splat); the model input is a
    # resample of it. The affine slice-extractor must still NOT run (wasted + discarded).
    assert calls["resp"] == 1 and calls["plain"] == 0
    # gt/bbox were re-derived by affine; images carry breathing.
    assert out["images"].shape == (2, 8, 3, 518, 518)
    assert out["images_splat"].shape == (2, 8, 256, 256)


# ── defer_input_images contract ───────────────────────────────────────────────
# The dataset may omit `images` entirely (`defer_input_images`, the training default)
# because gpu_augment_batch re-extracts every slice on GPU anyway. The contract is that a
# MISSING key means "extract unconditionally" — on EVERY path, including no-augmentation.
# If this ever regresses, a batch reaches the model with no input at all.

def _deferred_batch(**kw):
    b = _fake_batch(**kw)
    del b["images"]
    return b


@pytest.mark.parametrize("aug,resp", [(False, False), (False, True), (True, False), (True, True)])
def test_missing_images_is_always_rebuilt(aug, resp):
    """All four augmentation combinations must yield a usable `images` tensor."""
    batch = _deferred_batch()
    transforms = build_gpu_transforms(
        OmegaConf.create({"enable": True, "tier": "conservative"})) if aug else None
    out = gpu_augment_batch(batch, transforms, DEVICE,
                            respiratory_cfg=_resp_cfg(enable=resp), train=True)
    assert "images" in out, f"images missing with aug={aug} resp={resp}"
    B, S = out["timesteps"].shape
    assert out["images"].shape == (B, S, 3, 518, 518)
    assert torch.isfinite(out["images"]).all()
    assert float(out["images"].max()) <= 1.0 + 1e-5, "images must be normalized to [0,1]"


def test_deferred_matches_nondeferred_when_no_aug():
    """With every augmentation off, the GPU-rebuilt images must equal what the dataset
    would have produced — i.e. deferring changes nothing about the model's input."""
    ref = _fake_batch()          # ONE batch — building two would compare different volumes
    built = extract_slices_from_phases(
        ref["phases"].float(), ref["timesteps"], ref["slice_indices"],
    ).permute(0, 1, 4, 2, 3).contiguous() / 255.0
    deferred = {k: v for k, v in ref.items() if k != "images"}
    out = gpu_augment_batch(deferred, None, DEVICE,
                            respiratory_cfg=_resp_cfg(enable=False), train=True)
    assert torch.allclose(out["images"], built, atol=1e-6)
