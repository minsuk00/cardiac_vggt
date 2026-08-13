"""Native-render splat + input-resolution threading (docs/72 port)."""
import torch
import torch.nn.functional as F

from data.gpu_aug import extract_slices_from_phases
from data.respiratory import extract_slices_with_respiratory_vec
from loss import _splat_preds_native
from vggt.utils.splat import splat_predictions, splat_to_volume


def _phases(seed=0, B=1, T=3, D=4, H=64, W=64):
    return torch.rand(B, T, D, H, W, generator=torch.Generator().manual_seed(seed))


def test_extract_out_size_and_native_identity():
    phases = _phases()
    t, z = torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 3]])
    assert extract_slices_from_phases(phases, t, z, out_size=28).shape == (1, 3, 28, 28, 3)
    # at out_size == H the resize is an identity -> exact native content
    native = extract_slices_from_phases(phases, t, z, out_size=64)[..., 0] / 255.0
    for s in range(3):
        assert torch.allclose(native[0, s], phases[0, t[0, s], z[0, s]], atol=1e-6)


def test_resp_native_is_corrupted_not_clean():
    phases = _phases(seed=1)
    t, z = torch.tensor([[0, 1]]), torch.tensor([[1, 2]])
    spacing = (12.0, 1.4, 1.4)
    disp = torch.full((1, 2, 3), 0.0); disp[..., 1] = 8.0        # 8 mm AP shift
    corrupted = extract_slices_with_respiratory_vec(
        phases, t, z, disp, spacing, out_size=64)[..., 0] / 255.0
    clean = torch.stack([phases[0, t[0, s], z[0, s]] for s in range(2)]).unsqueeze(0)
    assert not torch.allclose(corrupted, clean, atol=1e-3)
    at_rest = extract_slices_with_respiratory_vec(
        phases, t, z, torch.zeros(1, 2, 3), spacing, out_size=64)[..., 0] / 255.0
    # 1e-5: the *255->clamp->/255 roundtrip + grid_sample identity reslice leave ~1e-7
    assert torch.allclose(at_rest, clean, atol=1e-5)


def test_native_splat_matches_manual_and_falls_back():
    g = torch.Generator().manual_seed(2)
    S, D, hm, hn = 3, 4, 28, 64
    px = torch.linspace(-1, 1, hm)
    gy, gx = torch.meshgrid(px, px, indexing="ij")
    wp = torch.zeros(1, S, hm, hm, 3)
    wp[..., 0], wp[..., 1] = gx, gy
    for s in range(S):
        wp[0, s, :, :, 2] = (s - (D - 1) / 2) / ((D - 1) / 2)
    wp += 0.01 * torch.rand(wp.shape, generator=g)
    batch = {"images": torch.rand(1, S, 3, hm, hm, generator=g),
             "images_splat": torch.rand(1, S, hn, hn, generator=g)}
    grid, z_scale = (D, hn, hn), (D - 1) / 2.0

    V, _ = _splat_preds_native({"world_points": wp}, batch, grid, z_scale)
    x = F.interpolate(wp.permute(0, 1, 4, 2, 3).reshape(S, 3, hm, hm),
                      size=(hn, hn), mode="bilinear", align_corners=True)
    wp_n = x.reshape(1, S, 3, hn, hn).permute(0, 1, 3, 4, 2)
    inten = batch["images_splat"].reshape(1, -1)
    Vm, _ = splat_to_volume(wp_n.reshape(1, -1, 3), inten, grid, z_scale,
                            weight=(inten > 1e-3).float())
    assert V.shape == (1,) + grid and torch.allclose(V, Vm, atol=1e-6)

    b2 = {k: v for k, v in batch.items() if k != "images_splat"}
    V2, _ = _splat_preds_native({"world_points": wp}, b2, grid, z_scale)
    V3, _ = splat_predictions({"world_points": wp}, b2, grid, z_scale)
    assert torch.equal(V2, V3)
