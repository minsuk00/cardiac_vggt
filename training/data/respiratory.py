"""Respiratory-motion simulation for VGGT-MRI (SI/AP rigid translation).

Implements the model scoped in `docs/01_respiratory_motion_simulation.md`: a
**rigid translation per input slice**, parameterized by one scalar displacement
`d` (mm) — superior-inferior (SI, through-plane along D) + a smaller anterior-
posterior (AP, in-plane), no left-right, no rotation, no scaling.

Applied by **deform-then-reslice**: the 3D volume is shifted by `d(r)` and the
input slice is re-sampled at its FIXED canonical plane, so the slice images
different anatomy as the heart moves (the physical through-plane content change).
The reconstruction target and `scanner_coords` stay at the unshifted end-
expiration reference — the model learns to *correct* breathing (blind to `r`).

Geometry (splat order, matching `gpu_aug.py` / `mri_dataset.py`):
    phases  (B, T=12, D=12, H=256, W=256)   spacing (D=Z=8.0, H=Y=1.4, W=X=1.4) mm
    D = SI (through-plane)   H/W = in-plane (AP vs LR not recoverable → AP axis
    is configurable, default H).

`extract_slices_with_respiratory` is a drop-in for `gpu_aug.extract_slices_from_phases`
(identical I/O contract), so the training hook is a thin add (see docs §5).

Single source of truth: the trainer GPU path AND `tools/render_respiratory_examples.py`
import these functions, so the visualized motion is exactly what training applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

INPUT_IMG_SIZE = 518          # DINOv2 input — must match MRIDataset.target_size
CANON_HW = 256
CANON_D = 12
SPACING_MM = (8.0, 1.4, 1.4)  # (D=Z, H=Y, W=X) mm — canonical cube


# ──────────────────────────────────────────────────────────────────────────────
# Lujan respiratory waveform
# ──────────────────────────────────────────────────────────────────────────────
def lujan_displacement(r, amplitude_mm, n: int = 2):
    """Lujan respiratory waveform: SI displacement (mm) vs respiratory phase `r`.

        d(r) = amplitude_mm * sin(pi * r) ** (2n)

    This is the Lujan cos^{2n} family phase-shifted so r=0 is end-expiration
    (sin^{2n}(πr) = cos^{2n}(π(r − ½))):

    r = 0   → end-expiration (rest)      → d = 0
    r = 0.5 → peak inspiration            → d = amplitude_mm
    r → 1   → back to end-expiration      → d = 0

    The even power makes the curve flat (≈ 0) over a WIDE range near r = 0/1 — i.e.
    it **dwells at end-expiration** (the rest position) and passes more quickly
    through inspiration, the physiological breathing trace. Larger `n` → longer
    exhale dwell (n=2 → sin^4).

    `r` and `amplitude_mm` may be Python floats or broadcastable tensors; returns
    the same shape/type. Pure torch (runs on GPU).
    """
    if not torch.is_tensor(r):
        r = torch.as_tensor(r, dtype=torch.float32)
    s = torch.sin(torch.pi * r)
    return amplitude_mm * s.pow(2 * n)


# ──────────────────────────────────────────────────────────────────────────────
# Config + per-slot displacement sampling
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RespiratoryConfig:
    """Config for the respiratory-motion augmentation (see docs §4 for numbers)."""
    enable: bool = False
    amplitude_mm: float = 12.0     # mean SI breath-depth A (mm); ~10-15mm literature
    amplitude_jitter: float = 4.0  # +/- uniform jitter on A (mm); 0 disables
    cos2n: int = 2                 # n in the Lujan cos^{2n} waveform
    ap_ratio: float = 0.35         # k: AP displacement = k * SI
    ap_axis: str = "H"             # in-plane axis carrying AP ("H" or "W")
    per_slot: bool = True          # independent breath per input slice (scattered regime)
    seed: int | None = None        # int → deterministic sampling (val/report); None → global RNG

    @classmethod
    def from_cfg(cls, cfg):
        """Build from an OmegaConf node / dict / object, falling back to defaults
        for any missing key (tolerant — ready for the deferred yaml block)."""
        if cfg is None:
            return cls()
        def g(key, default):
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)
        return cls(
            enable=bool(g("enable", cls.enable)),
            amplitude_mm=float(g("amplitude_mm", cls.amplitude_mm)),
            amplitude_jitter=float(g("amplitude_jitter", cls.amplitude_jitter)),
            cos2n=int(g("cos2n", cls.cos2n)),
            ap_ratio=float(g("ap_ratio", cls.ap_ratio)),
            ap_axis=str(g("ap_axis", cls.ap_axis)),
            per_slot=bool(g("per_slot", cls.per_slot)),
            seed=g("seed", cls.seed),
        )


def sample_displacements(B, S, cfg: RespiratoryConfig, device, generator=None):
    """Sample per-slot SI and AP displacement (mm).

    Returns (d_si_mm, d_ap_mm), each `(B, S)` float32 on `device`.

    - r ~ U[0,1) per (B,S) — respiratory phase, decoupled from cardiac t.
    - A = amplitude_mm + U(-jitter, +jitter); shape (B,1) broadcast over S if
      `per_slot=False` (one breath depth per subject), else (B,S).
    - d_si = lujan(r, A, cos2n);  d_ap = ap_ratio * d_si.

    Determinism: pass a `generator`, or set `cfg.seed` (a generator is then built
    on `device`). Train leaves both unset → global RNG.
    """
    if generator is None and cfg.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))

    def rand(shape):
        return torch.rand(shape, device=device, generator=generator, dtype=torch.float32)

    r = rand((B, S))                                  # respiratory phase
    amp_shape = (B, S) if cfg.per_slot else (B, 1)
    A = cfg.amplitude_mm + (rand(amp_shape) * 2.0 - 1.0) * cfg.amplitude_jitter
    A = A.clamp_min(0.0)                              # depth can't be negative
    d_si = lujan_displacement(r, A, n=cfg.cos2n)      # (B,S), A broadcasts
    d_ap = cfg.ap_ratio * d_si
    return d_si.to(torch.float32), d_ap.to(torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Deform-then-reslice core (grid_sample)
# ──────────────────────────────────────────────────────────────────────────────
def _norm_delta(d_mm, spacing_mm, size):
    """mm displacement → normalized [-1,1] grid delta (align_corners=True:
    one voxel = 2/(size-1) in normalized coords)."""
    return (d_mm / spacing_mm) * (2.0 / (size - 1))


def extract_slices_with_respiratory(
    phases, t_seq, z_seq, d_si_mm, d_ap_mm,
    ap_axis: str = "H", spacing=SPACING_MM,
):
    """Drop-in alternative to `gpu_aug.extract_slices_from_phases`, with a per-slot
    SI (through-plane) + AP (in-plane) shift applied at reslice time.

    Args:
        phases:   (B, T, D, H, W) float (any dtype; cast to f32 internally)
        t_seq:    (B, S) int64 — cardiac t per slot
        z_seq:    (B, S) int64 — canonical z plane per slot
        d_si_mm:  (B, S) float — SI displacement (mm, along D)
        d_ap_mm:  (B, S) float — AP displacement (mm, along ap_axis)
        ap_axis:  "H" or "W"
        spacing:  (D, H, W) mm

    Returns:
        (B, S, 518, 518, 3) float in [0, 255] — RGB-replicated, IDENTICAL contract
        to `extract_slices_from_phases` (ready for `permute(0,1,4,2,3)/255`).

    Sign convention: d > 0 samples the volume at plane `z + d_vox` (so anatomy from
    deeper planes appears at `z`). Consistent for SI and AP and with `reslice_volume`.
    At d = 0 the grid reduces to the integer-plane reslice → output matches
    `extract_slices_from_phases` (pinned by tests).
    """
    B, T, D, H, W = phases.shape
    S = t_seq.shape[1]
    device = phases.device
    pf = phases.float()

    # Gather per-slot D-stacks (over T only — need full D to interpolate through-plane).
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, S)
    vols = pf[b_idx, t_seq]                               # (B, S, D, H, W)
    N = B * S
    inp = vols.reshape(N, 1, D, H, W)                    # (N, C=1, D, H, W)

    # In-plane identity base grid (output pixel → same input pixel), shape (H, W).
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    y_base = (ys / (H - 1) * 2.0 - 1.0).view(1, H, 1).expand(1, H, W)   # (1,H,W)
    x_base = (xs / (W - 1) * 2.0 - 1.0).view(1, 1, W).expand(1, H, W)   # (1,H,W)

    # Depth base per slot + SI shift.
    z_base = (z_seq.to(torch.float32) / (D - 1) * 2.0 - 1.0).reshape(N)     # (N,)
    dz_norm = _norm_delta(d_si_mm.reshape(N), spacing[0], D)                # (N,)
    z_coord = (z_base + dz_norm).view(N, 1, 1, 1).expand(N, 1, H, W)        # (N,1,H,W)

    # AP shift along the chosen in-plane axis.
    if ap_axis == "H":
        d_ap_norm = _norm_delta(d_ap_mm.reshape(N), spacing[1], H).view(N, 1, 1, 1)
        y_coord = y_base.unsqueeze(0).expand(N, 1, H, W) + d_ap_norm
        x_coord = x_base.unsqueeze(0).expand(N, 1, H, W)
    elif ap_axis == "W":
        d_ap_norm = _norm_delta(d_ap_mm.reshape(N), spacing[2], W).view(N, 1, 1, 1)
        x_coord = x_base.unsqueeze(0).expand(N, 1, H, W) + d_ap_norm
        y_coord = y_base.unsqueeze(0).expand(N, 1, H, W)
    else:
        raise ValueError(f"ap_axis must be 'H' or 'W', got {ap_axis!r}")

    # grid last-dim order is (x, y, z) = (W, H, D) — grid_sample's REVERSE of tensor dims.
    grid = torch.stack([x_coord, y_coord, z_coord], dim=-1)   # (N, 1, H, W, 3)
    resliced = F.grid_sample(
        inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
    ).view(N, 1, H, W)                                        # (N,1,256,256)

    # Tail mirrors extract_slices_from_phases: upsample 256→518, scale, RGB-replicate.
    up = F.interpolate(resliced, size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                       mode="bilinear", align_corners=True)   # (N,1,518,518)
    up = up.view(B, S, INPUT_IMG_SIZE, INPUT_IMG_SIZE)
    up = (up * 255.0).clamp(0.0, 255.0)
    return up.unsqueeze(-1).expand(B, S, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3)


def reslice_volume(V, d_si_mm, d_ap_mm, ap_axis: str = "H", spacing=SPACING_MM):
    """Shift a whole single-phase volume `V` (D,H,W) by a constant (SI, AP)
    displacement and resample onto the canonical grid (one grid_sample, D_out=D).

    Used by the example renderer for coronal/sagittal/axial views — same math as
    `extract_slices_with_respiratory`, no upsample/RGB. Returns (D, H, W) float32.
    """
    D, H, W = V.shape
    device = V.device
    inp = V.float().view(1, 1, D, H, W)

    zs = torch.arange(D, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    z_base = zs / (D - 1) * 2.0 - 1.0
    y_base = ys / (H - 1) * 2.0 - 1.0
    x_base = xs / (W - 1) * 2.0 - 1.0

    dz = _norm_delta(float(d_si_mm), spacing[0], D)
    z_coord = (z_base + dz).view(D, 1, 1).expand(D, H, W)

    if ap_axis == "H":
        d_ap = _norm_delta(float(d_ap_mm), spacing[1], H)
        y_coord = (y_base + d_ap).view(1, H, 1).expand(D, H, W)
        x_coord = x_base.view(1, 1, W).expand(D, H, W)
    elif ap_axis == "W":
        d_ap = _norm_delta(float(d_ap_mm), spacing[2], W)
        x_coord = (x_base + d_ap).view(1, 1, W).expand(D, H, W)
        y_coord = y_base.view(1, H, 1).expand(D, H, W)
    else:
        raise ValueError(f"ap_axis must be 'H' or 'W', got {ap_axis!r}")

    grid = torch.stack([x_coord, y_coord, z_coord], dim=-1).unsqueeze(0)  # (1,D,H,W,3)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=True)                              # (1,1,D,H,W)
    return out.view(D, H, W)
