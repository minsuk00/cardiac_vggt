"""Native-z (per-subject slice pitch) tests for the respiratory simulation — docs/59 F4.

WHY THIS FILE EXISTS. docs/58 changed `respiratory.py` so a breathing shift in mm is
converted to slice indices using **this subject's own `dz`** instead of the hardcoded
`SPACING_MM[0] = 12.0`. That fix was real but **completely untested**: every fixture in
`test_respiratory.py` / `test_gpu_aug.py` uses `dz = 12.0`, and 12.0 is ALSO the old
hardcoded value — so `d/dz` and `d/12.0` compute the identical number in every existing
test, and reverting the fix could not fail anything. Same for `n_planes`: the burst tests
never pass it and use plane ids <= 6, where `clamp(0, 11)` and `clamp(0, D-1)` are both
no-ops.

Every test here is therefore built to FAIL if the pitch/plane-count plumbing regresses to
the legacy constants. Two of them (`test_pitch_fault_injection_is_detectable`,
`test_n_planes_fault_injection_is_detectable`) exist purely to prove the others have
teeth — they assert that the legacy behaviour is *measurably different*, so a silent
revert cannot pass.

Run on CPU — grid_sample works without CUDA.
"""

import pytest
import torch

from data.respiratory import (
    N_CANON_PLANES,
    SPACING_MM,
    RespiratoryConfig,
    extract_slices_with_respiratory_vec,
    sample_resp_disp,
)

DEVICE = "cpu"

# Real pitches from the pooled cohort (training/splits/manifest.csv spans 5.0-12.0 mm).
# 12.0 is included ONLY as the degenerate control where old and new agree.
REAL_PITCHES = [5.0, 6.0, 8.0, 8.8, 9.6, 10.0, 12.0]


def _plane_coded_phases(B, T, D, H=8, W=8):
    """(B, T, D, H, W) volume whose every voxel on plane z holds the value z/100.

    Reading an output intensity back out therefore RECOVERS THE (fractional) PLANE INDEX
    the reslice actually sampled — which is exactly the quantity the pitch conversion
    decides. /100 keeps the values well under the extractor's `* 255` + clamp(0, 255)
    tail, so nothing saturates and the decode stays exact.
    """
    z = torch.arange(D, dtype=torch.float32).view(1, 1, D, 1, 1) / 100.0
    return z.expand(B, T, D, H, W).contiguous()


def _resliced_plane(phases, z_i, disp_dhw, spacing):
    """Reslice one slot and decode which plane index the sampler actually landed on."""
    t_seq = torch.zeros(1, 1, dtype=torch.int64)
    z_seq = torch.tensor([[z_i]], dtype=torch.int64)
    out = extract_slices_with_respiratory_vec(
        phases, t_seq, z_seq, disp_dhw, spacing=spacing
    )                                     # (1, 1, 518, 518, 3) in [0, 255]
    centre = float(out[0, 0, 259, 259, 0])
    return centre / 255.0 * 100.0         # undo the *255 tail and the /100 encoding


# ── 1. The core invariant: a d-mm shift lands at exactly d/dz planes ──────────

@pytest.mark.parametrize("dz", REAL_PITCHES)
@pytest.mark.parametrize("D", [5, 12, 21])
def test_shift_lands_at_exact_plane_for_every_pitch(dz, D):
    """d mm of through-plane breathing must move the sampled plane by EXACTLY d/dz.

    This is the assertion the legacy hardcoded 12.0 breaks for every dz != 12.
    """
    phases = _plane_coded_phases(B=1, T=2, D=D)
    z_i = D // 2
    # Pick a shift that stays inside the stack at every pitch (<= 1.5 planes deep).
    d_mm = 1.5 * dz
    disp = torch.tensor([[[d_mm, 0.0, 0.0]]], dtype=torch.float32)   # (B=1, S=1, 3) D-axis only

    landed = _resliced_plane(phases, z_i, disp, spacing=(dz, 1.4, 1.4))
    expected = z_i + d_mm / dz                                       # == z_i + 1.5

    assert landed == pytest.approx(expected, abs=2e-3), (
        f"dz={dz} D={D}: {d_mm} mm should land at plane {expected}, landed at {landed}"
    )


@pytest.mark.parametrize("dz", [5.0, 8.0, 9.6])
def test_pitch_fault_injection_is_detectable(dz):
    """PROOF THE TEST ABOVE HAS TEETH: passing the LEGACY 12.0 pitch instead of the
    subject's own dz must produce a MEASURABLY different landing plane.

    Without this, `test_shift_lands_at_exact_plane_for_every_pitch` could be passing for
    a trivial reason. `dz` is deliberately restricted to non-12 pitches — at dz == 12 the
    fix and the bug are numerically identical, which is precisely the blind spot that let
    F4 exist.
    """
    D = 21
    phases = _plane_coded_phases(B=1, T=2, D=D)
    z_i = 8
    # 3 planes deep: large enough that the native-vs-legacy gap, 3*(1 - dz/12) planes,
    # clears half a slice even at the closest real pitch (9.6 mm → 0.6).
    d_mm = 3.0 * dz
    disp = torch.tensor([[[d_mm, 0.0, 0.0]]], dtype=torch.float32)

    correct = _resliced_plane(phases, z_i, disp, spacing=(dz, 1.4, 1.4))
    legacy = _resliced_plane(phases, z_i, disp, spacing=(SPACING_MM[0], 1.4, 1.4))

    assert correct == pytest.approx(z_i + 3.0, abs=2e-3)
    assert legacy == pytest.approx(z_i + d_mm / SPACING_MM[0], abs=2e-3)
    # The whole point: at a real non-12 pitch the two DISAGREE by over half a slice.
    assert abs(correct - legacy) > 0.5, (
        f"dz={dz}: legacy 12mm and native-z landings are indistinguishable "
        f"({legacy} vs {correct}) — this test could not catch a revert"
    )


def test_shift_direction_and_zero_shift():
    """Sign convention + the disp=0 identity, at a non-12 pitch.

    d > 0 samples DEEPER (higher plane index); d < 0 samples shallower; d = 0 must return
    the untouched plane. Guards against a sign flip hiding inside the pitch division.
    """
    dz, D = 5.0, 21
    phases = _plane_coded_phases(B=1, T=2, D=D)
    z_i = 10

    def landed(d_mm):
        disp = torch.tensor([[[d_mm, 0.0, 0.0]]], dtype=torch.float32)
        return _resliced_plane(phases, z_i, disp, spacing=(dz, 1.4, 1.4))

    assert landed(0.0) == pytest.approx(z_i, abs=2e-3)
    assert landed(2 * dz) == pytest.approx(z_i + 2, abs=2e-3)
    assert landed(-2 * dz) == pytest.approx(z_i - 2, abs=2e-3)


def test_subvoxel_shift_interpolates_at_native_pitch():
    """A shift smaller than one slice must interpolate to a FRACTIONAL plane, using the
    subject's pitch. At dz=5 a 2 mm shift is 0.4 planes; under the legacy 12 mm it would
    be 0.167 — a 2.4x understatement that no integer-plane test could see."""
    dz, D, z_i = 5.0, 21, 10
    phases = _plane_coded_phases(B=1, T=2, D=D)
    disp = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)
    landed = _resliced_plane(phases, z_i, disp, spacing=(dz, 1.4, 1.4))
    assert landed == pytest.approx(z_i + 0.4, abs=2e-3)


# ── 2. In-plane axes must NOT be scaled by dz ─────────────────────────────────

@pytest.mark.parametrize("dz", [5.0, 12.0])
def test_inplane_shift_uses_inplane_spacing_not_dz(dz):
    """The H/W components use the 1.4 mm in-plane spacing regardless of the z pitch.

    A plausible bad fix for F4 is to scale ALL three axes by dz; this pins that the
    through-plane pitch never leaks into the in-plane conversion.
    """
    D, H, W = 8, 16, 16
    # Encode the H index instead of the z index this time.
    h = torch.arange(H, dtype=torch.float32).view(1, 1, 1, H, 1) / 100.0
    phases = h.expand(1, 2, D, H, W).contiguous()

    t_seq = torch.zeros(1, 1, dtype=torch.int64)
    z_seq = torch.tensor([[D // 2]], dtype=torch.int64)
    disp = torch.tensor([[[0.0, 1.4, 0.0]]], dtype=torch.float32)   # +1 voxel along H
    out = extract_slices_with_respiratory_vec(phases, t_seq, z_seq, disp, spacing=(dz, 1.4, 1.4))

    # Sample a pixel well inside the field so the 256->518 upsample is not edge-affected.
    # Output row r corresponds to input row r*(H-1)/517; row 259 -> input row 7.5, +1 voxel -> 8.5.
    landed = float(out[0, 0, 259, 259, 0]) / 255.0 * 100.0
    assert landed == pytest.approx(8.5, abs=0.05), (
        f"dz={dz} must not affect the in-plane shift; got {landed}"
    )


# ── 3. group_by_burst must use the subject's real D, not the legacy 12 ────────

def test_group_by_burst_respects_n_planes_beyond_twelve():
    """With n_planes=D>12, planes 12..D-1 are INDEPENDENT breaths.

    The legacy `clamp(0, N_CANON_PLANES-1)` collapsed every plane >= 12 onto plane 11, so
    a 21-slice subject's whole basal third shared one breath. Existing burst tests use
    plane ids <= 6 and so cannot see this.
    """
    D = 21
    cfg = RespiratoryConfig(enable=True, direction_jitter_deg=30.0, group_by_burst=True)
    gids = torch.arange(D, dtype=torch.int64).view(1, D)          # one slot per plane, 0..20
    seq = torch.tensor([[7]], dtype=torch.int64)

    _, r = sample_resp_disp(1, D, cfg, DEVICE, train=False, seq_index=seq,
                            group_ids=gids, n_planes=D)

    deep = r[0, N_CANON_PLANES:]                                  # planes 12..20
    assert deep.numel() == D - N_CANON_PLANES
    # Independent breaths ⇒ these must not all be the same value.
    assert float(deep.max() - deep.min()) > 1e-3, (
        "planes >= 12 all share one breath phase — n_planes is being ignored"
    )
    # And no deep plane may simply echo plane 11 (the legacy clamp target).
    assert not torch.allclose(deep, r[0, N_CANON_PLANES - 1].expand_as(deep), atol=1e-6)


def test_n_planes_fault_injection_is_detectable():
    """PROOF THE TEST ABOVE HAS TEETH: omitting n_planes (the legacy default) must
    collapse every plane >= 12 onto plane 11's breath."""
    D = 21
    cfg = RespiratoryConfig(enable=True, direction_jitter_deg=30.0, group_by_burst=True)
    gids = torch.arange(D, dtype=torch.int64).view(1, D)
    seq = torch.tensor([[7]], dtype=torch.int64)

    _, r_legacy = sample_resp_disp(1, D, cfg, DEVICE, train=False, seq_index=seq,
                                   group_ids=gids, n_planes=None)

    deep = r_legacy[0, N_CANON_PLANES:]
    # Legacy clamp ⇒ planes 12..20 all read plane 11.
    assert torch.allclose(deep, r_legacy[0, N_CANON_PLANES - 1].expand_as(deep), atol=1e-6), (
        "the legacy n_planes=None path no longer clamps — this fault injection is stale"
    )


def test_group_by_burst_shares_one_breath_within_a_deep_plane():
    """Slots on the SAME plane still share one breath, including for planes >= 12."""
    D = 21
    cfg = RespiratoryConfig(enable=True, direction_jitter_deg=30.0, group_by_burst=True)
    gids = torch.tensor([[18, 3, 18, 20, 18, 20]], dtype=torch.int64)
    seq = torch.tensor([[7]], dtype=torch.int64)

    disp, r = sample_resp_disp(1, 6, cfg, DEVICE, train=False, seq_index=seq,
                              group_ids=gids, n_planes=D)
    for p in (18, 20):
        idx = (gids[0] == p).nonzero().flatten()
        assert torch.allclose(r[0, idx], r[0, idx[0]].expand_as(r[0, idx]), atol=1e-6)
        assert torch.allclose(disp[0, idx], disp[0, idx[0]].expand_as(disp[0, idx]), atol=1e-6)
    assert float(r[0, 0]) != pytest.approx(float(r[0, 1]), abs=1e-6)   # plane 18 vs plane 3


def test_group_by_burst_clamps_to_subject_D_for_thin_stacks():
    """A thin subject (D=5) must clamp plane ids into [0, 4] — the legacy constant would
    have allowed indices up to 11 and indexed past the sampled table."""
    D = 5
    cfg = RespiratoryConfig(enable=True, direction_jitter_deg=30.0, group_by_burst=True)
    gids = torch.tensor([[0, 4, 9, 11, 4]], dtype=torch.int64)      # 9, 11 are out of range
    seq = torch.tensor([[7]], dtype=torch.int64)

    _, r = sample_resp_disp(1, 5, cfg, DEVICE, train=False, seq_index=seq,
                            group_ids=gids, n_planes=D)
    # Out-of-range ids clamp to D-1 = 4, so slots 1, 2, 3, 4 all share plane 4's breath.
    for i in (2, 3, 4):
        assert float(r[0, i]) == pytest.approx(float(r[0, 1]), abs=1e-6)


# ── 4. The gpu_aug integration actually threads dz/D through ──────────────────

@pytest.mark.parametrize("dz,D", [(5.0, 21), (9.6, 11), (12.0, 12)])
def test_gpu_aug_passes_native_pitch_and_plane_count(monkeypatch, dz, D):
    """End of the chain: `gpu_augment_batch` must hand the RESLICER this subject's own
    `dz` and `D`, not the module constants.

    This is the integration half of F4 — the unit tests above prove the reslicer is
    correct GIVEN the right spacing; this proves the trainer path supplies it.
    """
    import data.gpu_aug as gpu_aug

    seen = {}
    real_extract = gpu_aug.extract_slices_with_respiratory_vec
    real_sample = gpu_aug.sample_resp_disp

    def spy_extract(phases, t_seq, z_seq, disp, spacing):
        seen["spacing"] = tuple(float(s) for s in spacing)
        return real_extract(phases, t_seq, z_seq, disp, spacing)

    def spy_sample(*args, **kwargs):
        seen["n_planes"] = kwargs.get("n_planes")
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(gpu_aug, "extract_slices_with_respiratory_vec", spy_extract)
    monkeypatch.setattr(gpu_aug, "sample_resp_disp", spy_sample)

    B, T, S, H, W = 1, 12, 4, 256, 256
    batch = {
        "phases": torch.rand(B, T, D, H, W, dtype=torch.float16),
        "content_mask": torch.ones(B, D, H, W, dtype=torch.uint8),
        "gt_target_volume": torch.rand(B, D, H, W),
        "anatomy_bbox": torch.tensor([[0, D, 0, H, 0, W]] * B, dtype=torch.int64),
        "t_target": torch.zeros(B, 1, dtype=torch.int64),
        "timesteps": torch.zeros(B, S, dtype=torch.int64),
        "slice_indices": torch.tensor([[i % D for i in range(S)]] * B, dtype=torch.int64),
        "images": torch.rand(B, S, 3, 518, 518),
        "scanner_coords": torch.rand(B, S, 518, 518, 3),
        "seq_index": torch.tensor([[7]] * B, dtype=torch.int64),
        "dz_mm": torch.tensor([[dz]] * B, dtype=torch.float32),
    }

    gpu_aug.gpu_augment_batch(
        batch, transforms=None, device=DEVICE, train=False,
        respiratory_cfg=RespiratoryConfig(enable=True, direction_jitter_deg=30.0),
    )

    assert seen["spacing"][0] == pytest.approx(dz), (
        f"reslicer got pitch {seen['spacing'][0]}, expected this subject's {dz}"
    )
    assert seen["spacing"][1:] == pytest.approx((1.4, 1.4))   # in-plane must stay canonical
    assert seen["n_planes"] == D


def test_gpu_aug_rejects_mixed_pitch_batch():
    """docs/59 F7 guard, exercised through the real aug entry point: one scalar dz is
    applied to the whole batch, so a batch mixing pitches must RAISE, not silently
    breathe row 1 at row 0's scale."""
    import data.gpu_aug as gpu_aug

    B, T, D, S, H, W = 2, 12, 10, 4, 256, 256
    batch = {
        "phases": torch.rand(B, T, D, H, W, dtype=torch.float16),
        "content_mask": torch.ones(B, D, H, W, dtype=torch.uint8),
        "gt_target_volume": torch.rand(B, D, H, W),
        "anatomy_bbox": torch.tensor([[0, D, 0, H, 0, W]] * B, dtype=torch.int64),
        "t_target": torch.zeros(B, 1, dtype=torch.int64),
        "timesteps": torch.zeros(B, S, dtype=torch.int64),
        "slice_indices": torch.tensor([[i % D for i in range(S)]] * B, dtype=torch.int64),
        "images": torch.rand(B, S, 3, 518, 518),
        "scanner_coords": torch.rand(B, S, 518, 518, 3),
        "seq_index": torch.tensor([[7], [9]], dtype=torch.int64),
        "dz_mm": torch.tensor([[5.0], [12.0]], dtype=torch.float32),   # same D, different pitch
    }
    with pytest.raises(RuntimeError, match="dz_mm is not uniform"):
        gpu_aug.gpu_augment_batch(
            batch, transforms=None, device=DEVICE, train=False,
            respiratory_cfg=RespiratoryConfig(enable=True, direction_jitter_deg=30.0),
        )
