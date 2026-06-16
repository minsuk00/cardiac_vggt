"""Optional 3D UNet refiner for the splatted volume.

Refines `V_canon` (the coverage-averaged splat) → `V_refined`, to recover the high-frequency
detail the splat smooths away (see `_html/08_breathing_failure_mode.html`: ~75% of the
reconstruction blur is the splat renderer). Small, self-contained, no new deps.

Design:
  - Anisotropic: pools ONLY H/W (D=12 at 8mm is already coarse; keep it). Convs are 3x3x3.
  - Residual: `V_refined = V_canon + Δ`, with the output conv zero-initialized so Δ≈0 at init
    ⇒ the refiner starts as the identity (gentle warm-up, no early disruption to V_canon).
  - Input channels: V_canon, optionally + coverage (tells the net where data is trustworthy vs
    under-covered, so it deblurs covered regions and stays conservative in sparse ones).
  - fp32: runs under autocast(enabled=False) so V_refined matches V_gt dtype for the L1/PSNR
    and the small 3D convs / GroupNorm stay numerically stable (the splat is already fp32).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3d -> GroupNorm -> GELU) x2 (wolny/pytorch-3dunet style)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        g = min(8, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class VolumeRefiner(nn.Module):
    """Anisotropic residual 3D UNet: (V_canon[, coverage]) -> V_refined (B, D, H, W)."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16, levels: int = 2,
                 use_coverage: bool = False):
        super().__init__()
        self.use_coverage = use_coverage
        chs = [base_channels * (2 ** i) for i in range(levels + 1)]  # e.g. [16,32,64]

        # Encoder: a DoubleConv per level, then anisotropic H/W pool between levels.
        self.enc = nn.ModuleList()
        prev = in_channels
        for i in range(levels):
            self.enc.append(DoubleConv(prev, chs[i]))
            prev = chs[i]
        self.bottleneck = DoubleConv(chs[levels - 1], chs[levels])

        # Decoder: upsample H/W, concat the skip, DoubleConv. in = below_ch + skip_ch.
        self.dec = nn.ModuleList()
        for i in reversed(range(levels)):
            self.dec.append(DoubleConv(chs[i + 1] + chs[i], chs[i]))

        self.out_conv = nn.Conv3d(chs[0], 1, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)   # Δ ≈ 0 at init → starts as identity
        nn.init.zeros_(self.out_conv.bias)

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, V_canon: torch.Tensor, coverage: torch.Tensor = None) -> torch.Tensor:
        # V_canon: (B, D, H, W). coverage: (B, D, H, W) or None.
        with torch.cuda.amp.autocast(enabled=False):
            Vc = V_canon.float()
            x = Vc.unsqueeze(1)  # (B, 1, D, H, W)
            if self.use_coverage:
                cov = coverage.float() if coverage is not None else torch.zeros_like(Vc)
                x = torch.cat([x, cov.unsqueeze(1)], dim=1)  # (B, 2, D, H, W)

            skips = []
            for enc in self.enc:
                x = enc(x)
                skips.append(x)
                x = self.pool(x)
            x = self.bottleneck(x)
            for dec, skip in zip(self.dec, reversed(skips)):
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
                x = dec(torch.cat([x, skip], dim=1))

            delta = self.out_conv(x).squeeze(1)  # (B, D, H, W)
            return Vc + delta
