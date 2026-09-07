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


def _field_and_native(seed=3, S=3, D=4, hm=28, hn=64):
    """A smooth-ish point field at model res `hm` and native slices at `hn`, plus the model
    input the trainer derives from them (native -> hm bilinear, RGB-replicated, [0,1])."""
    g = torch.Generator().manual_seed(seed)
    px = torch.linspace(-1, 1, hm)
    gy, gx = torch.meshgrid(px, px, indexing="ij")
    wp = torch.zeros(1, S, hm, hm, 3)
    wp[..., 0], wp[..., 1] = gx, gy
    for s in range(S):
        wp[0, s, :, :, 2] = (s - (D - 1) / 2) / ((D - 1) / 2)
    wp += 0.01 * torch.rand(wp.shape, generator=g)
    native = torch.rand(1, S, hn, hn, generator=g)
    images = F.interpolate(native.reshape(S, 1, hn, hn), size=(hm, hm), mode="bilinear",
                           align_corners=True).view(1, S, 1, hm, hm).expand(1, S, 3, hm, hm)
    return wp, native, images.contiguous()


def test_splat_res_none_is_noop_and_native_res_is_identity():
    wp, native, images = _field_and_native()
    S, D, hn = 3, 4, 64
    batch = {"images": images, "images_splat": native}
    grid, z_scale = (D, hn, hn), (D - 1) / 2.0
    V0, _ = _splat_preds_native({"world_points": wp}, batch, grid, z_scale)
    V1, _ = _splat_preds_native({"world_points": wp}, batch, grid, z_scale, splat_res=None)
    V2, _ = _splat_preds_native({"world_points": wp}, batch, grid, z_scale, splat_res=hn)
    assert torch.equal(V0, V1) and torch.equal(V0, V2)


def test_splat_res_at_model_res_reproduces_old_model_res_splat():
    """splat_res == the model's own resolution must reproduce the pre-docs/73 path
    (`splat_predictions` on `batch["images"]`, no field resample) — that is the training
    signal every pre-2026-08-13 checkpoint (e.g. the old 518 arm) was trained against."""
    wp, native, images = _field_and_native()
    S, D, hm, hn = 3, 4, 28, 64
    batch = {"images": images, "images_splat": native}
    grid, z_scale = (D, hn, hn), (D - 1) / 2.0
    V_new, cov_new = _splat_preds_native({"world_points": wp}, batch, grid, z_scale, splat_res=hm)
    V_old, cov_old = splat_predictions({"world_points": wp}, {"images": images}, grid, z_scale)
    assert torch.allclose(V_new, V_old, atol=1e-6) and torch.allclose(cov_new, cov_old, atol=1e-6)
    # and it is NOT the native render (different point count -> different coverage)
    V_nat, _ = _splat_preds_native({"world_points": wp}, batch, grid, z_scale)
    assert not torch.allclose(V_nat, V_old, atol=1e-3)


def test_splat_res_supersample_matches_manual():
    """splat_res above native: field AND native image are both bilinearly resampled
    (align_corners=True, the extraction convention) to splat_res², then splatted."""
    wp, native, images = _field_and_native()
    S, D, hm, hn, R = 3, 4, 28, 64, 96
    batch = {"images": images, "images_splat": native}
    grid, z_scale = (D, hn, hn), (D - 1) / 2.0
    V, _ = _splat_preds_native({"world_points": wp}, batch, grid, z_scale, splat_res=R)
    x = F.interpolate(wp.permute(0, 1, 4, 2, 3).reshape(S, 3, hm, hm), size=(R, R),
                      mode="bilinear", align_corners=True)
    wp_r = x.reshape(1, S, 3, R, R).permute(0, 1, 3, 4, 2)
    im_r = F.interpolate(native.reshape(S, 1, hn, hn), size=(R, R), mode="bilinear",
                         align_corners=True).reshape(1, -1)
    Vm, _ = splat_to_volume(wp_r.reshape(1, -1, 3), im_r, grid, z_scale,
                            weight=(im_r > 1e-3).float())
    assert torch.allclose(V, Vm, atol=1e-6)


def test_splat_res_threads_through_loss_config():
    """`loss.volume.splat_res` reaches the splat: the loss must differ from the native default
    (different point set) and equal the model-res path when splat_res == model res."""
    from loss import compute_volume_intensity_loss
    wp, native, images = _field_and_native()
    S, D, hm, hn = 3, 4, 28, 64
    g = torch.Generator().manual_seed(4)
    V_gt = torch.rand(1, D, hn, hn, generator=g)
    batch = {"images": images, "images_splat": native, "gt_target_volume": V_gt,
             "z_scale": torch.tensor([(D - 1) / 2.0])}
    preds = {"world_points": wp}
    l_nat = compute_volume_intensity_loss(preds, batch, tv_weight=0.0)["loss_volume"]
    l_none = compute_volume_intensity_loss(preds, batch, tv_weight=0.0, splat_res=None)["loss_volume"]
    l_mod = compute_volume_intensity_loss(preds, batch, tv_weight=0.0, splat_res=hm)["loss_volume"]
    V_old, _ = splat_predictions(preds, {"images": images}, (D, hn, hn), (D - 1) / 2.0)
    assert torch.equal(l_nat, l_none)
    assert torch.allclose(l_mod, (V_old - V_gt).abs().mean(), atol=1e-6)
    assert not torch.allclose(l_mod, l_nat, atol=1e-4)
