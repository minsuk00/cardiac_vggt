"""Tests for the respiratory-motion simulation (training/data/respiratory.py).

Run on CPU — grid_sample works without CUDA.
"""

import numpy as np
import pytest
import torch

from data.gpu_aug import extract_slices_from_phases
from data.respiratory import (
    RespiratoryConfig,
    extract_slices_with_respiratory,
    extract_slices_with_respiratory_vec,
    lujan_displacement,
    reslice_volume,
    sample_displacement_vectors,
    sample_displacements,
    sample_resp_disp,
)

DEVICE = "cpu"


# ── 1. Lujan waveform ─────────────────────────────────────────────────────────
def test_lujan_endpoints():
    A = 12.0
    assert abs(float(lujan_displacement(0.0, A))) < 1e-6          # end-expiration = rest
    assert abs(float(lujan_displacement(1.0, A))) < 1e-5          # back to exhale
    assert abs(float(lujan_displacement(0.5, A, n=2)) - A) < 1e-5  # peak inspiration = A

def test_lujan_monotone_on_first_half_and_vectorized():
    r = torch.linspace(0, 0.5, 50)
    d = lujan_displacement(r, 12.0, n=2)
    assert d.shape == r.shape                                      # vectorizes
    assert torch.all(torch.diff(d) >= -1e-6)                      # non-decreasing 0→0.5
    assert float(d[0]) == pytest.approx(0.0, abs=1e-6)


# ── 2. d=0 identity (primary equivalence with the existing extractor) ─────────
def test_zero_displacement_matches_baseline_extractor():
    B, T, D, H, W, S = 1, 2, 8, 16, 16, 3
    phases = torch.rand(B, T, D, H, W, dtype=torch.float32)
    t_seq = torch.tensor([[0, 1, 0]], dtype=torch.int64)
    z_seq = torch.tensor([[2, 5, 7]], dtype=torch.int64)
    zero = torch.zeros(B, S)
    a = extract_slices_with_respiratory(phases, t_seq, z_seq, zero, zero)
    b = extract_slices_from_phases(phases, t_seq, z_seq)
    assert a.shape == b.shape == (B, S, 518, 518, 3)
    assert torch.allclose(a, b, atol=1e-3)


# ── 3. Known shift moves content by the expected number of voxels ─────────────
def test_known_si_shift_one_voxel():
    D, H, W = 8, 4, 4
    V = torch.arange(D, dtype=torch.float32).view(D, 1, 1).expand(D, H, W).contiguous()  # V[z]=z
    out = reslice_volume(V, d_si_mm=8.0, d_ap_mm=0.0)   # +1 voxel along D (8mm spacing)
    for z in range(D - 1):
        assert float(out[z, 0, 0]) == pytest.approx(z + 1, abs=1e-4)  # samples plane z+1
    assert float(out[D - 1, 0, 0]) == pytest.approx(0.0, abs=1e-4)    # off-stack → padding 0

def test_known_ap_shift_one_voxel():
    D, H, W = 4, 8, 8
    V = torch.arange(H, dtype=torch.float32).view(1, H, 1).expand(D, H, W).contiguous()  # V[:,h,:]=h
    out = reslice_volume(V, d_si_mm=0.0, d_ap_mm=1.4, ap_axis="H")    # +1 voxel along H (1.4mm)
    for h in range(H - 1):
        assert float(out[0, h, 0]) == pytest.approx(h + 1, abs=1e-4)


# ── 4. Axis correctness (guards the (x,y,z)=(W,H,D) grid ordering trap) ────────
def test_ap_axis_selects_correct_inplane_axis():
    D, H, W = 4, 8, 8
    ramp_h = torch.arange(H, dtype=torch.float32).view(1, H, 1).expand(D, H, W).contiguous()
    # AP along H shifts the H-ramp; AP along W leaves the H-ramp unchanged.
    out_h = reslice_volume(ramp_h, 0.0, 1.4, ap_axis="H")
    out_w = reslice_volume(ramp_h, 0.0, 1.4, ap_axis="W")
    assert float(out_h[0, 3, 0]) == pytest.approx(4.0, abs=1e-4)   # h=3 → 4 (shifted)
    assert float(out_w[0, 3, 0]) == pytest.approx(3.0, abs=1e-4)   # unchanged


# ── 5. Batched == per-slot loop ───────────────────────────────────────────────
def test_batch_equals_loop():
    B, T, D, H, W, S = 2, 3, 8, 12, 12, 4
    phases = torch.rand(B, T, D, H, W, dtype=torch.float32)
    t_seq = torch.randint(0, T, (B, S))
    z_seq = torch.randint(0, D, (B, S))
    d_si = torch.rand(B, S) * 16 - 8
    d_ap = torch.rand(B, S) * 4 - 2
    batched = extract_slices_with_respiratory(phases, t_seq, z_seq, d_si, d_ap)
    for b in range(B):
        for s in range(S):
            one = extract_slices_with_respiratory(
                phases[b:b + 1], t_seq[b:b + 1, s:s + 1], z_seq[b:b + 1, s:s + 1],
                d_si[b:b + 1, s:s + 1], d_ap[b:b + 1, s:s + 1],
            )
            assert torch.allclose(batched[b, s], one[0, 0], atol=1e-4)


# ── 6. Padding leaks zeros, not edge replication ──────────────────────────────
def test_offstack_shift_reads_zero():
    D, H, W = 8, 8, 8
    V = torch.rand(D, H, W) + 1.0                # strictly positive content
    out = reslice_volume(V, d_si_mm=8.0 * 20, d_ap_mm=0.0)   # 20 voxels → fully off-stack
    assert float(out.abs().max()) < 1e-4


# ── 7. Determinism of the sampler ─────────────────────────────────────────────
def test_sample_displacements_deterministic_with_seed():
    cfg = RespiratoryConfig(enable=True, seed=123)
    a_si, a_ap, _ = sample_displacements(2, 6, cfg, DEVICE)
    b_si, b_ap, _ = sample_displacements(2, 6, cfg, DEVICE)
    assert torch.equal(a_si, b_si) and torch.equal(a_ap, b_ap)
    c_si, _, _ = sample_displacements(2, 6, RespiratoryConfig(enable=True, seed=999), DEVICE)
    assert not torch.equal(a_si, c_si)
    # AP is the configured ratio of SI.
    assert torch.allclose(a_ap, cfg.ap_ratio * a_si, atol=1e-5)

def test_per_slot_flag_controls_amplitude_sharing(monkeypatch):
    """per_slot=False → ONE breath depth (amplitude A) shared across all slots;
    per_slot=True → independent A per slot. Patch lujan to return A directly (bypass
    the per-slot r modulation) so the amplitude broadcast is observable in isolation."""
    import data.respiratory as R
    # d := A (broadcast to r's shape), isolating the amplitude from the r draw.
    monkeypatch.setattr(
        R, "lujan_displacement",
        lambda r, A, n=3: torch.as_tensor(A, dtype=torch.float32) * torch.ones_like(r))

    shared = RespiratoryConfig(enable=True, amplitude_jitter=8.0, per_slot=False, seed=5)
    d_shared, _, _ = R.sample_displacements(1, 6, shared, DEVICE)
    assert torch.allclose(d_shared, d_shared[:, :1].expand_as(d_shared))  # all slots == one A

    perslot = RespiratoryConfig(enable=True, amplitude_jitter=8.0, per_slot=True, seed=5)
    d_ps, _, _ = R.sample_displacements(1, 6, perslot, DEVICE)
    assert not torch.allclose(d_ps, d_ps[:, :1].expand_as(d_ps))  # independent A per slot


# ── 8. fp16 input is handled ──────────────────────────────────────────────────
def test_fp16_phases_ok():
    B, T, D, H, W, S = 1, 2, 8, 16, 16, 2
    phases = torch.rand(B, T, D, H, W, dtype=torch.float16)
    t_seq = torch.zeros(B, S, dtype=torch.int64)
    z_seq = torch.tensor([[3, 4]], dtype=torch.int64)
    d_si = torch.tensor([[4.0, -4.0]])
    d_ap = torch.tensor([[1.0, -1.0]])
    out = extract_slices_with_respiratory(phases, t_seq, z_seq, d_si, d_ap)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 255.0


# ── 9. Direction randomization (3-vector + rotation) ──────────────────────────
def test_zero_tilt_reduces_to_si_ap():
    """direction_jitter_deg=0 → the 3-vector is exactly (d_si, d_ap_on_axis, 0)."""
    cfg = RespiratoryConfig(enable=True, direction_jitter_deg=0.0, ap_axis="H", seed=42)
    v, _ = sample_displacement_vectors(2, 6, cfg, DEVICE)         # (B,S,3)
    d_si, d_ap, _ = sample_displacements(2, 6, cfg, DEVICE)       # same seed → same draw
    assert torch.allclose(v[..., 0], d_si, atol=1e-6)            # D = SI
    assert torch.allclose(v[..., 1], d_ap, atol=1e-6)            # H = AP
    assert torch.allclose(v[..., 2], torch.zeros_like(v[..., 2]), atol=1e-6)  # W = 0


def test_zero_tilt_vec_matches_scalar_extractor():
    """With zero tilt the vec core output equals the scalar (SI+AP) shim bit-for-bit."""
    B, T, D, H, W, S = 1, 2, 8, 16, 16, 4
    phases = torch.rand(B, T, D, H, W, dtype=torch.float32)
    t_seq = torch.randint(0, T, (B, S))
    z_seq = torch.randint(0, D, (B, S))
    d_si = torch.rand(B, S) * 12 - 6
    d_ap = torch.rand(B, S) * 4 - 2
    scalar = extract_slices_with_respiratory(phases, t_seq, z_seq, d_si, d_ap, ap_axis="H")
    disp = torch.stack([d_si, d_ap, torch.zeros_like(d_si)], dim=-1)
    vec = extract_slices_with_respiratory_vec(phases, t_seq, z_seq, disp)
    assert torch.allclose(scalar, vec, atol=1e-4)


def test_tilt_preserves_mm_magnitude_and_adds_lr():
    """Rotation is rigid → mm magnitude preserved; tilt injects a nonzero LR (W)
    component and shrinks the D component vs the untilted vector (same seed)."""
    base = RespiratoryConfig(enable=True, direction_jitter_deg=0.0, ap_axis="H", seed=7)
    tilt = RespiratoryConfig(enable=True, direction_jitter_deg=30.0, ap_axis="H", seed=7)
    v0, _ = sample_displacement_vectors(4, 8, base, DEVICE)      # (d_si, d_ap, 0)
    vt, _ = sample_displacement_vectors(4, 8, tilt, DEVICE)      # rotated
    # Same seed → identical d_si/d_ap draw → identical mm magnitude per slot.
    assert torch.allclose(v0.norm(dim=-1), vt.norm(dim=-1), atol=1e-4)
    # Tilt actually moved the vector off the D/H plane (nonzero W somewhere).
    assert float(vt[..., 2].abs().max()) > 1e-3
    # The D (SI) component is no larger than the untilted one (tilt bleeds into plane).
    assert float(vt[..., 0].abs().max()) <= float(v0[..., 0].abs().max()) + 1e-4


# ── 10. sample_resp_disp determinism (train vs val) ───────────────────────────
def test_sample_resp_disp_val_deterministic_per_seq_index():
    cfg = RespiratoryConfig(enable=True, direction_jitter_deg=30.0)
    seq = torch.tensor([[3], [4]], dtype=torch.int64)
    a, a_r = sample_resp_disp(2, 6, cfg, DEVICE, train=False, seq_index=seq)
    b, b_r = sample_resp_disp(2, 6, cfg, DEVICE, train=False, seq_index=seq)
    assert torch.equal(a, b) and torch.equal(a_r, b_r)           # reproducible (disp + phase)
    assert a.shape == (2, 6, 3) and a_r.shape == (2, 6)
    assert float(a_r.min()) >= 0.0 and float(a_r.max()) < 1.0    # r ∈ [0,1)
    # Row-permuted batch (same per-row seq_index) → per-row vectors track the index.
    seq_swapped = torch.tensor([[4], [3]], dtype=torch.int64)
    c, _ = sample_resp_disp(2, 6, cfg, DEVICE, train=False, seq_index=seq_swapped)
    assert torch.equal(a[0], c[1]) and torch.equal(a[1], c[0])  # per-ROW, not per-batch
    # Distinct seq_index → distinct breathing.
    assert not torch.equal(a[0], a[1])


def test_sample_resp_disp_val_requires_seq_index():
    cfg = RespiratoryConfig(enable=True)
    with pytest.raises(ValueError):
        sample_resp_disp(1, 4, cfg, DEVICE, train=False, seq_index=None)


def test_sample_resp_disp_train_uses_generator():
    cfg = RespiratoryConfig(enable=True, direction_jitter_deg=30.0)
    g1 = torch.Generator(device=DEVICE).manual_seed(11)
    g2 = torch.Generator(device=DEVICE).manual_seed(11)
    a, _ = sample_resp_disp(2, 6, cfg, DEVICE, train=True, generator=g1)
    b, _ = sample_resp_disp(2, 6, cfg, DEVICE, train=True, generator=g2)
    assert torch.equal(a, b)                                     # same seed → same draw
    # Advancing the same generator → a different draw (iid across steps).
    c, _ = sample_resp_disp(2, 6, cfg, DEVICE, train=True, generator=g1)
    assert not torch.equal(a, c)
