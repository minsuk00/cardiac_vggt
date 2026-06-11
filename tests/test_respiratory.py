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
    lujan_displacement,
    reslice_volume,
    sample_displacements,
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
    a_si, a_ap = sample_displacements(2, 6, cfg, DEVICE)
    b_si, b_ap = sample_displacements(2, 6, cfg, DEVICE)
    assert torch.equal(a_si, b_si) and torch.equal(a_ap, b_ap)
    c_si, _ = sample_displacements(2, 6, RespiratoryConfig(enable=True, seed=999), DEVICE)
    assert not torch.equal(a_si, c_si)
    # AP is the configured ratio of SI.
    assert torch.allclose(a_ap, cfg.ap_ratio * a_si, atol=1e-5)

def test_per_subject_amplitude_shares_depth_across_slots():
    cfg = RespiratoryConfig(enable=True, amplitude_jitter=0.0, per_slot=False, seed=7)
    d_si, _ = sample_displacements(1, 5, cfg, DEVICE)
    # Same A and ap_ratio across slots; r differs → d differs, but max ≤ amplitude_mm.
    assert float(d_si.max()) <= cfg.amplitude_mm + 1e-4


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
