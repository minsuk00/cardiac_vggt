"""torch.compile-traceable 3D fused-SSIM.

Ported directly from MRI2CT (`common/fused_ssim_compat.py`).
The upstream `fused_ssim` package calls the raw pybind CUDA kernels
(`fused_ssim_cuda.fusedssim3d` / `fusedssim_backward3d`) directly inside a
`torch.autograd.Function`. Those pybind functions are not registered with the
PyTorch dispatcher (no schema, no meta/fake kernel), so TorchDynamo marks them
"function marked as skipped" (graph-break gb0007).

This module wraps the two kernel calls as `torch.library.custom_op`s with fake
kernels so the SAME autograd.Function traces cleanly (0 graph breaks). Numerically
identical to upstream (verified bit-exact on loss and gradient). 3D, "same"
padding, fp32.

Drop-in for `from fused_ssim import fused_ssim3d`.
"""
import torch

try:
    from fused_ssim_cuda import fusedssim3d as _fusedssim3d
    from fused_ssim_cuda import fusedssim_backward3d as _fusedssim_backward3d
    _HAS_FUSED_SSIM = True
except ImportError:
    _HAS_FUSED_SSIM = False

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


if _HAS_FUSED_SSIM:
    # --- opaque kernels wrapped as dispatcher-registered custom ops (traceable) ---
    @torch.library.custom_op("vggt_fused_ssim::forward3d", mutates_args=())
    def _ssim_forward3d(img1: torch.Tensor, img2: torch.Tensor, train: bool) -> list[torch.Tensor]:
        # kernel returns (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12), all fp32.
        # train=False -> the three dm_* buffers come back empty (shape (0,)).
        return [t.contiguous() for t in _fusedssim3d(_C1, _C2, img1, img2, train)]


    @_ssim_forward3d.register_fake
    def _(img1, img2, train):
        smap = torch.empty_like(img1, dtype=torch.float32)
        if train:
            dm = [torch.empty_like(img1, dtype=torch.float32) for _ in range(3)]
        else:
            dm = [img1.new_empty(0, dtype=torch.float32) for _ in range(3)]
        return [smap, *dm]


    @torch.library.custom_op("vggt_fused_ssim::backward3d", mutates_args=())
    def _ssim_backward3d(img1: torch.Tensor, img2: torch.Tensor, dL_dmap: torch.Tensor,
                         dm_dmu1: torch.Tensor, dm_dsigma1_sq: torch.Tensor,
                         dm_dsigma12: torch.Tensor) -> torch.Tensor:
        dL_dmap = dL_dmap.contiguous()
        return _fusedssim_backward3d(_C1, _C2, img1, img2, dL_dmap,
                                     dm_dmu1, dm_dsigma1_sq, dm_dsigma12).contiguous()


    @_ssim_backward3d.register_fake
    def _(img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12):
        return torch.empty_like(img1, dtype=torch.float32)


    class _FusedSSIM3D(torch.autograd.Function):
        @staticmethod
        def forward(ctx, img1, img2, train=True):
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = torch.ops.vggt_fused_ssim.forward3d(
                img1, img2, train)
            ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
            return ssim_map

        @staticmethod
        def backward(ctx, opt_grad):
            img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
            grad = torch.ops.vggt_fused_ssim.backward3d(
                img1, img2, opt_grad, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
            return grad, None, None


    def fused_ssim3d(img1, img2, padding="same", train=True):
        """1 - handled by caller; returns mean SSIM over the "same"-padded map. 3D, fp32.

        Matches upstream `fused_ssim.fused_ssim3d` for padding="same" (the only mode used here)."""
        assert padding == "same", f"fused_ssim_compat supports padding='same' only, got {padding!r}"
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        ssim_map = _FusedSSIM3D.apply(img1, img2, train)
        return ssim_map.mean()
else:
    def fused_ssim3d(img1, img2, padding="same", train=True):
        raise NotImplementedError("fused_ssim_cuda is not installed")
