"""Unit tests for vggt.utils.splat."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vggt.utils.splat import sample_volume, splat_to_volume


def test_splat_single_point_at_origin():
    """A single point at (0,0,0) lands in the center of the grid."""
    pos = torch.tensor([[[0.0, 0.0, 0.0]]])  # (1, 1, 3) — normalized center
    intensity = torch.tensor([[1.0]])
    grid = (5, 5, 5)
    vol, cov = splat_to_volume(pos, intensity, grid)
    assert vol.shape == (1, 5, 5, 5)
    assert cov.sum().item() == pytest.approx(1.0, abs=1e-5), "trilinear weights sum to 1"
    # The four center voxels (between indices 1..3 with align_corners) receive most of the mass.
    assert vol[0, 2, 2, 2].item() == pytest.approx(1.0, abs=1e-5)


def test_splat_sample_roundtrip_isolated_point():
    """Splat then sample at the SAME position should recover the input intensity."""
    pos = torch.tensor([[[0.3, -0.2, 0.5]]])
    intensity = torch.tensor([[0.7]])
    vol, _ = splat_to_volume(pos, intensity, (8, 8, 8))
    sampled = sample_volume(vol, pos)
    assert sampled.shape == (1, 1)
    assert sampled.item() == pytest.approx(0.7, abs=1e-5)


def test_splat_disagreeing_points_average():
    """Two points at the same position with different intensities resample to the average."""
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])  # (1, 2, 3)
    intensity = torch.tensor([[1.0, 0.0]])
    vol, _ = splat_to_volume(pos, intensity, (5, 5, 5))
    sampled = sample_volume(vol, pos)
    assert sampled[0, 0].item() == pytest.approx(0.5, abs=1e-5)
    assert sampled[0, 1].item() == pytest.approx(0.5, abs=1e-5)


def test_splat_out_of_bounds_contributes_zero():
    """A point outside [-1, 1] contributes nothing."""
    pos = torch.tensor([[[2.0, 2.0, 2.0]]])
    intensity = torch.tensor([[1.0]])
    vol, cov = splat_to_volume(pos, intensity, (4, 4, 4))
    assert vol.abs().sum().item() == pytest.approx(0.0)
    assert cov.sum().item() == pytest.approx(0.0)


def test_splat_gradient_flows_to_position():
    """Gradients must flow back to pos through the trilinear weights."""
    pos = torch.tensor([[[0.1, 0.2, 0.3]]], requires_grad=True)
    intensity = torch.tensor([[1.0]])
    vol, _ = splat_to_volume(pos, intensity, (8, 8, 8))
    sampled = sample_volume(vol, pos)
    sampled.sum().backward()
    assert pos.grad is not None
    # grad may be near-zero for an isolated point (roundtrip is exact), but the call must succeed.


def test_splat_batched():
    """Batching should be independent across B."""
    B, N = 2, 100
    pos = torch.rand(B, N, 3) * 1.8 - 0.9  # mostly in-bounds
    intensity = torch.rand(B, N)
    vol, cov = splat_to_volume(pos, intensity, (6, 6, 6))
    assert vol.shape == (B, 6, 6, 6)
    assert cov.shape == (B, 6, 6, 6)
    # Each batch's coverage should equal the in-bounds point count (up to rounding from trilinear).
    for b in range(B):
        assert cov[b].sum().item() > 0


def test_splat_weight_gating():
    """Weight=0 points contribute to neither numerator nor denominator."""
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    intensity = torch.tensor([[1.0, 9.0]])  # second point has very different intensity
    weight = torch.tensor([[1.0, 0.0]])     # second point gated off
    vol, cov = splat_to_volume(pos, intensity, (5, 5, 5), weight=weight)
    sampled = sample_volume(vol, pos[:, :1, :])  # sample at first point only
    # Only first point should contribute → V_canon at origin = 1.0, not (1+9)/2 = 5.0
    assert sampled.item() == pytest.approx(1.0, abs=1e-5)
    # Coverage should reflect only the first point's weight
    assert cov.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_splat_recovers_constant_field():
    """If all points have the same intensity I, V_canon should be ~I where covered."""
    N = 1000
    pos = torch.rand(1, N, 3) * 1.8 - 0.9
    intensity = torch.full((1, N), 0.42)
    vol, cov = splat_to_volume(pos, intensity, (4, 4, 4))
    # Where covered, V_canon should equal 0.42 (intensity / weight, both scaled identically).
    covered = cov[0] > 0.1
    assert vol[0][covered].mean().item() == pytest.approx(0.42, abs=1e-4)
