import pytest
import torch

from baselines.fcsvr_cardiac.cardiac import (
    augment_reference_inplane,
    augment_slice_intensity,
    compensate_motion,
    extract_respiratory_slices_256,
    motion_metrics,
    native_to_stage1,
    released_coarse_motion_metrics,
    reconstruct_native,
    stage1_flow_to_native_mm,
)
from data.respiratory import reslice_volume_vec


@pytest.mark.parametrize("depth", [6, 8, 10])
def test_stage1_depth_padding_round_trip(depth):
    volume = torch.arange(depth * 16 * 16, dtype=torch.float32).reshape(depth, 16, 16)
    mask = torch.ones_like(volume)
    disp = torch.zeros(depth, 3)

    inputs, target, meta = native_to_stage1(volume, mask, disp)

    padded_depth = max(16, 2 * depth)
    assert inputs.shape == (2, padded_depth, 60, 60)
    assert target.shape == (4, padded_depth, 60, 60)
    assert meta.stage1_depth == 2 * depth
    assert inputs[:, : meta.pad_before].count_nonzero() == 0
    assert inputs[:, meta.pad_before + 2 * depth :].count_nonzero() == 0
    assert target[3, : meta.pad_before].count_nonzero() == 0
    assert target[3, meta.pad_before + 2 * depth :].count_nonzero() == 0


def test_mm_voxel_conversion_and_axis_order():
    depth = 8
    _, _, meta = native_to_stage1(
        torch.ones(depth, 16, 16), torch.ones(depth, 16, 16), torch.zeros(depth, 3)
    )
    # Released model channels are external (x, y, z), in 3-mm slab voxels.
    high_flow_xyz = torch.zeros(1, 3, 4 * depth, 120, 120)
    high_flow_xyz[:, 0] = 2.0
    high_flow_xyz[:, 1] = -1.0
    high_flow_xyz[:, 2] = 0.5

    native_dhw = stage1_flow_to_native_mm(high_flow_xyz, meta)

    expected = torch.tensor([1.5, -3.0, 6.0])  # (D,H,W) mm
    assert native_dhw.shape == (1, depth, 16, 16, 3)
    torch.testing.assert_close(native_dhw[0], expected.expand(depth, 16, 16, 3))


def test_native_conversion_preserves_dense_spatial_motion():
    depth = 8
    _, _, meta = native_to_stage1(
        torch.ones(depth, 16, 16), torch.ones(depth, 16, 16), torch.zeros(depth, 3)
    )
    flow = torch.zeros(1, 3, 4 * depth, 120, 120)
    flow[:, 0, :, :, 60:] = 2.0

    native = stage1_flow_to_native_mm(flow, meta)

    assert native[0, :, :, :7, 2].abs().max() == 0
    assert native[0, :, :, 9:, 2].min() == pytest.approx(6.0)


def test_positive_motion_splats_toward_positive_axis():
    volume = torch.zeros(3, 5, 5)
    volume[1, 2, 2] = 1
    flow = torch.zeros(1, 3, 3)
    flow[:, 1, 0] = 12.0

    recon, _ = reconstruct_native(volume, flow, spacing_mm=(12.0, 1.4, 1.4))

    assert recon[0, 2, 2, 2] == pytest.approx(1.0)
    assert recon[0, 1, 2, 2] == 0


def test_zero_motion_is_identity():
    generator = torch.Generator().manual_seed(4)
    volume = 0.1 + 0.9 * torch.rand(1, 6, 17, 19, generator=generator)
    flow = torch.zeros(1, 6, 3)

    recon, coverage = reconstruct_native(volume, flow, spacing_mm=(12.0, 1.4, 1.4))

    torch.testing.assert_close(recon, volume)
    torch.testing.assert_close(coverage, torch.ones_like(coverage))


def test_dense_zero_motion_is_identity():
    volume = torch.rand(1, 4, 11, 13) + 0.1
    dense_flow = torch.zeros(1, 4, 11, 13, 3)

    recon, _ = reconstruct_native(volume, dense_flow, spacing_mm=(12.0, 1.4, 1.4))

    torch.testing.assert_close(recon, volume)


def test_motion_metrics_report_mm_and_slab_voxels():
    pred = torch.zeros(1, 2, 3, 4, 3)
    target = torch.zeros(1, 2, 3)
    pred[..., 0] = 3.0
    mask = torch.ones(1, 2, 3, 4)

    metrics = motion_metrics(pred, target, mask, slab_spacing_mm=3.0)

    # Released l22 averages the three squared components; Appendix A.2 sums
    # the squared vector per foreground point. Report both without ambiguity.
    assert metrics["metric_motion_component_mse_mm2"] == pytest.approx(3.0)
    assert metrics["metric_motion_paper_mse_mm2"] == pytest.approx(9.0)
    assert metrics["metric_motion_epe_mm"] == pytest.approx(3.0)
    assert metrics["metric_motion_component_mse_slab_vox2"] == pytest.approx(1 / 3)
    assert metrics["metric_motion_paper_mse_slab_vox2"] == pytest.approx(1.0)
    assert metrics["metric_motion_epe_slab_vox"] == pytest.approx(1.0)


def test_coarse_metrics_are_exact_released_loss_functions():
    from baselines.fcsvr_cardiac.models.losses import (
        l21_loss_affine_invariant,
        l22_loss_affine_invariant,
    )

    generator = torch.Generator().manual_seed(7)
    prediction = torch.randn(1, 3, 4, 5, 6, generator=generator)
    target = torch.randn(1, 4, 4, 5, 6, generator=generator)
    target[:, 3] = 1

    metrics = released_coarse_motion_metrics(prediction, target)

    assert metrics["metric_released_coarse_l22_component_vox2"] == pytest.approx(
        l22_loss_affine_invariant(prediction, target, eps=0).item()
    )
    assert metrics["metric_released_coarse_l21_epe_vox"] == pytest.approx(
        l21_loss_affine_invariant(prediction, target, eps=0).item()
    )


def test_compensation_removes_global_translation_and_preserves_relative_errors():
    target = torch.zeros(1, 3, 4, 3, 3)
    target[:, 0, 1] = 0.5
    target[:, 1, 2] = -0.25
    mask = torch.ones(1, 1, 4, 3, 3)
    predicted = target + torch.tensor([2.0, -3.0, 1.0]).view(1, 3, 1, 1, 1)

    compensated = compensate_motion(predicted, torch.cat([target, mask], dim=1))

    torch.testing.assert_close(compensated, target, atol=2e-5, rtol=0)
    before = predicted[:, :, 1] - predicted[:, :, 0]
    after = compensated[:, :, 1] - compensated[:, :, 0]
    torch.testing.assert_close(after, before, atol=2e-5, rtol=0)


def test_native_respiratory_extraction_matches_vggt_reslice_geometry():
    phases = torch.arange(2 * 4 * 7 * 9, dtype=torch.float32).reshape(1, 2, 4, 7, 9)
    phases /= phases.max()
    t = torch.tensor([[0, 1, 0]])
    z = torch.tensor([[0, 2, 3]])
    disp = torch.tensor([[[0.0, 0.0, 0.0], [3.0, -1.4, 0.7], [-2.0, 0.5, -0.2]]])
    spacing = (12.0, 1.4, 1.4)

    actual = extract_respiratory_slices_256(phases, t, z, disp, spacing)
    expected = torch.stack([
        reslice_volume_vec(phases[0, int(t[0, i])], disp[0, i], spacing)[int(z[0, i])]
        for i in range(3)
    ]).unsqueeze(0)

    assert actual.shape == (1, 3, 7, 9)
    torch.testing.assert_close(actual, expected)


def test_oracle_motion_reconstruction_improves_over_zero_motion():
    depth, height, width = 4, 17, 19
    clean = torch.zeros(depth, height, width)
    clean[:, 6:11, 7:12] = torch.linspace(0.2, 1.0, depth)[:, None, None]
    phases = clean[None, None]
    z = torch.arange(depth).view(1, depth)
    t = torch.zeros_like(z)
    displacement = torch.zeros(1, depth, 3)
    displacement[:, :, 1] = 1.4
    corrupt = extract_respiratory_slices_256(
        phases, t, z, displacement, spacing_mm=(12.0, 1.4, 1.4)
    )
    foreground = torch.ones_like(corrupt)

    zero_recon, _ = reconstruct_native(
        corrupt, torch.zeros_like(displacement),
        spacing_mm=(12.0, 1.4, 1.4), foreground_mask=foreground,
    )
    oracle_recon, _ = reconstruct_native(
        corrupt, displacement,
        spacing_mm=(12.0, 1.4, 1.4), foreground_mask=foreground,
    )

    zero_mse = (zero_recon - clean).square().mean()
    oracle_mse = (oracle_recon - clean).square().mean()
    assert oracle_mse < zero_mse * 1e-6


def test_heart_mask_controls_loss_while_content_mask_remains_model_input():
    volume = torch.ones(6, 16, 16)
    content = torch.ones_like(volume)
    heart = torch.zeros_like(volume)
    heart[:, 4:12, 5:11] = 1

    inputs, target, meta = native_to_stage1(volume, content, torch.zeros(6, 3), heart)

    assert inputs[1, meta.pad_before : meta.pad_before + meta.stage1_depth].all()
    assert 0 < target[3].count_nonzero() < target[3].numel()


def test_model_intensity_is_zero_outside_input_foreground():
    volume = torch.ones(6, 16, 16)
    foreground = torch.zeros_like(volume); foreground[:, 4:12, 5:11] = 1

    inputs, _, _ = native_to_stage1(volume, foreground, torch.zeros(6, 3), foreground)

    assert inputs[0].masked_select(inputs[1] == 0).count_nonzero() == 0


def test_author_augmentations_are_seeded_and_keep_masks_binary():
    phases = torch.linspace(0, 1, 2 * 4 * 16 * 16).reshape(2, 4, 16, 16)
    content = torch.ones(4, 16, 16)
    heart = torch.zeros_like(content); heart[:, 5:11, 6:10] = 1

    def run():
        generator = torch.Generator().manual_seed(19)
        p, c, h = augment_reference_inplane(phases, content, heart, generator)
        return augment_slice_intensity(p[0], generator), c, h

    first = run(); second = run()
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)
    assert set(first[1].unique().tolist()) <= {0.0, 1.0}
    assert set(first[2].unique().tolist()) <= {0.0, 1.0}
