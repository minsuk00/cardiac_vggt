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

import math
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
def lujan_displacement(r, amplitude_mm, n: int = 3):
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
    exhale dwell (default n=3 → sin^6).

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
    amplitude_mm: float = 16.0     # mean SI breath-depth A (mm); peak at full inspiration
    amplitude_jitter: float = 8.0  # +/- uniform jitter on A (mm) → ~8-24mm (tidal → deep)
    cos2n: int = 3                 # n in the Lujan sin^{2n} waveform (n=3 → sin^6, longer exhale dwell)
    ap_ratio: float = 0.35         # k: AP displacement = k * SI
    ap_axis: str = "H"             # in-plane axis carrying AP ("H" or "W")
    per_slot: bool = True          # independent breath per input slice (scattered regime)
    direction_jitter_deg: float = 30.0  # max random tilt (deg) of the SI+AP vector off the D axis;
                                        # 0 → pure SI+AP (no randomization). Handles the SAX-stack
                                        # tilt (D is the LV long axis, ~20-45° off true SI).
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
            direction_jitter_deg=float(g("direction_jitter_deg", cls.direction_jitter_deg)),
            seed=g("seed", cls.seed),
        )


def sample_displacements(B, S, cfg: RespiratoryConfig, device, generator=None):
    """Sample per-slot SI and AP displacement (mm).

    Returns (d_si_mm, d_ap_mm, r), each `(B, S)` float32 on `device` (r is the
    respiratory phase in [0,1) — 0/1=end-exhale, 0.5=peak inspiration).

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
    return d_si.to(torch.float32), d_ap.to(torch.float32), r.to(torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Canonical displacement VECTOR (SI + AP, optionally tilted off the D axis)
# ──────────────────────────────────────────────────────────────────────────────
def _build_disp_dhw(d_si, d_ap, ap_axis: str):
    """Stack scalar SI/AP displacements into a canonical (..., 3) = (d_D, d_H, d_W)
    mm vector. SI → D; AP → the chosen in-plane axis; the other in-plane axis → 0."""
    zeros = torch.zeros_like(d_si)
    if ap_axis == "H":
        return torch.stack([d_si, d_ap, zeros], dim=-1)
    if ap_axis == "W":
        return torch.stack([d_si, zeros, d_ap], dim=-1)
    raise ValueError(f"ap_axis must be 'H' or 'W', got {ap_axis!r}")


def _rotate_disp(v, theta, phi):
    """Rodrigues-rotate canonical mm vectors `v` (..., 3) = (d_D, d_H, d_W) by angle
    `theta` (...) about an axis lying in the in-plane (H-W) plane at azimuth `phi`.

    The axis `k = (0, -sinφ, cosφ)` is chosen so that θ tilts the D (SI) direction
    toward the in-plane direction `(cosφ along H, sinφ along W)`. Rotation is rigid →
    **mm magnitude is preserved**; θ=0 → identity (v unchanged). Applied to the WHOLE
    SI+AP vector (SI and AP tilt together as a coupled rigid unit), in PHYSICAL mm
    space (so the anisotropic 8 mm D vs 1.4 mm in-plane spacing is handled later, by
    per-axis normalization in the reslice core — NOT here).
    """
    kD = torch.zeros_like(phi)
    kH = -torch.sin(phi)
    kW = torch.cos(phi)
    k = torch.stack([kD, kH, kW], dim=-1)             # (..., 3) unit axis
    cos = torch.cos(theta).unsqueeze(-1)
    sin = torch.sin(theta).unsqueeze(-1)
    kxv = torch.cross(k, v, dim=-1)
    kdotv = (k * v).sum(dim=-1, keepdim=True)
    return v * cos + kxv * sin + k * kdotv * (1.0 - cos)


def sample_displacement_vectors(B, S, cfg: RespiratoryConfig, device, generator=None):
    """Sample per-slot canonical displacement vectors. Returns `(v, r)` where
    `v` is `(B, S, 3)` = (d_D, d_H, d_W) mm and `r` is `(B, S)` respiratory phase.

    Builds the SI+AP vector via `sample_displacements`, then (if
    `cfg.direction_jitter_deg > 0`) tilts each slot's vector by a random
    θ~U(0, jitter) about a random azimuth φ~U(0, 2π) — the randomized translation
    direction. All draws share ONE generator so determinism (val/report) is exact.
    """
    # Resolve the generator ONCE so SI/AP and θ/φ draw from the same stream
    # (otherwise a cfg.seed would only seed the SI/AP draw, breaking determinism).
    if generator is None and cfg.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(cfg.seed))

    d_si, d_ap, r = sample_displacements(B, S, cfg, device, generator=generator)  # (B,S) each
    v = _build_disp_dhw(d_si, d_ap, cfg.ap_axis)                                   # (B,S,3)

    if cfg.direction_jitter_deg and cfg.direction_jitter_deg > 0:
        def rand(shape):
            return torch.rand(shape, device=device, generator=generator, dtype=torch.float32)
        theta = rand((B, S)) * math.radians(float(cfg.direction_jitter_deg))
        phi = rand((B, S)) * (2.0 * math.pi)
        v = _rotate_disp(v, theta, phi)
    return v.to(torch.float32), r


def sample_resp_disp(B, S, cfg: RespiratoryConfig, device, *, train: bool,
                     seq_index=None, generator=None):
    """Determinism wrapper used by the trainer GPU path. Returns `(B, S, 3)` mm.

    Returns `(disp, r)`: `disp` is `(B, S, 3)` mm, `r` is `(B, S)` respiratory phase.

    - **train** → one batched draw from the passed private `generator` (iid per epoch;
      never perturbs the global RNG stream).
    - **val** → a per-ROW generator seeded from `seq_index[b]`, so breathing is fully
      reproducible across epochs/runs regardless of how rows are grouped into batches
      (mirrors the dataset's `random.Random(seq_index)` z/t determinism). Per-ROW (not
      per-batch) because `DynamicBatchSampler` groups variable rows per batch.
    """
    if train:
        return sample_displacement_vectors(B, S, cfg, device, generator=generator)

    if seq_index is None:
        raise ValueError(
            "respiratory val augmentation requires seq_index for deterministic sampling"
        )
    seq = seq_index.reshape(-1).tolist()
    v_rows, r_rows = [], []
    for b in range(B):
        g = torch.Generator(device=device).manual_seed(int(seq[b]))
        v_b, r_b = sample_displacement_vectors(1, S, cfg, device, generator=g)
        v_rows.append(v_b[0]); r_rows.append(r_b[0])
    return torch.stack(v_rows, dim=0), torch.stack(r_rows, dim=0)  # (B,S,3), (B,S)


# ──────────────────────────────────────────────────────────────────────────────
# Deform-then-reslice core (grid_sample)
# ──────────────────────────────────────────────────────────────────────────────
def _norm_delta(d_mm, spacing_mm, size):
    """mm displacement → normalized [-1,1] grid delta (align_corners=True:
    one voxel = 2/(size-1) in normalized coords)."""
    return (d_mm / spacing_mm) * (2.0 / (size - 1))


def extract_slices_with_respiratory_vec(phases, t_seq, z_seq, disp_dhw, spacing=SPACING_MM):
    """Reslice S input slices per batch element with a per-slot CANONICAL 3-vector
    displacement (the general, rotation-aware core). Drop-in for
    `gpu_aug.extract_slices_from_phases` (identical output contract).

    Args:
        phases:   (B, T, D, H, W) float (any dtype; cast to f32 internally)
        t_seq:    (B, S) int64 — cardiac t per slot
        z_seq:    (B, S) int64 — canonical z plane per slot
        disp_dhw: (B, S, 3) float — per-slot (d_D, d_H, d_W) mm (canonical axes)
        spacing:  (D, H, W) mm

    Returns:
        (B, S, 518, 518, 3) float in [0, 255] — RGB-replicated (ready for
        `permute(0,1,4,2,3)/255`).

    Sign convention: d > 0 samples the volume at coord `+ d_vox` (so deeper anatomy
    appears at the fixed plane). At disp = 0 the grid reduces to the integer-plane
    reslice → output matches `extract_slices_from_phases` (pinned by tests).
    """
    B, T, D, H, W = phases.shape
    S = t_seq.shape[1]
    device = phases.device
    pf = phases.float()
    disp = disp_dhw.to(device=device, dtype=torch.float32).reshape(B * S, 3)

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

    # Per-slot, per-axis mm → normalized deltas (each axis uses its own spacing).
    dz_norm = _norm_delta(disp[:, 0], spacing[0], D)       # (N,)
    dy_norm = _norm_delta(disp[:, 1], spacing[1], H)       # (N,)
    dx_norm = _norm_delta(disp[:, 2], spacing[2], W)       # (N,)

    z_base = (z_seq.to(torch.float32) / (D - 1) * 2.0 - 1.0).reshape(N)     # (N,)
    z_coord = (z_base + dz_norm).view(N, 1, 1, 1).expand(N, 1, H, W)        # (N,1,H,W)
    y_coord = y_base.unsqueeze(0).expand(N, 1, H, W) + dy_norm.view(N, 1, 1, 1)
    x_coord = x_base.unsqueeze(0).expand(N, 1, H, W) + dx_norm.view(N, 1, 1, 1)

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


def extract_slices_with_respiratory(
    phases, t_seq, z_seq, d_si_mm, d_ap_mm,
    ap_axis: str = "H", spacing=SPACING_MM,
):
    """Scalar (SI + AP, no tilt) convenience shim over
    `extract_slices_with_respiratory_vec`. Kept for the renderer + existing tests.

    `d_si_mm` / `d_ap_mm` are `(B, S)`; builds the canonical 3-vector with zero tilt
    (SI→D, AP→ap_axis, other in-plane→0) and delegates to the vector core.
    """
    if not torch.is_tensor(d_si_mm):
        d_si_mm = torch.as_tensor(d_si_mm, dtype=torch.float32)
    if not torch.is_tensor(d_ap_mm):
        d_ap_mm = torch.as_tensor(d_ap_mm, dtype=torch.float32)
    disp = _build_disp_dhw(d_si_mm.to(torch.float32), d_ap_mm.to(torch.float32), ap_axis)
    return extract_slices_with_respiratory_vec(phases, t_seq, z_seq, disp, spacing)


def reslice_volume_vec(V, disp_dhw, spacing=SPACING_MM):
    """Shift a whole single-phase volume `V` (D,H,W) by a constant canonical 3-vector
    `disp_dhw` = (d_D, d_H, d_W) mm and resample onto the canonical grid (one
    grid_sample, D_out=D). Returns (D, H, W) float32."""
    D, H, W = V.shape
    device = V.device
    inp = V.float().view(1, 1, D, H, W)
    disp = torch.as_tensor(disp_dhw, dtype=torch.float32, device=device).reshape(3)

    zs = torch.arange(D, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    z_base = zs / (D - 1) * 2.0 - 1.0
    y_base = ys / (H - 1) * 2.0 - 1.0
    x_base = xs / (W - 1) * 2.0 - 1.0

    dz = _norm_delta(disp[0], spacing[0], D)
    dy = _norm_delta(disp[1], spacing[1], H)
    dx = _norm_delta(disp[2], spacing[2], W)
    z_coord = (z_base + dz).view(D, 1, 1).expand(D, H, W)
    y_coord = (y_base + dy).view(1, H, 1).expand(D, H, W)
    x_coord = (x_base + dx).view(1, 1, W).expand(D, H, W)

    grid = torch.stack([x_coord, y_coord, z_coord], dim=-1).unsqueeze(0)  # (1,D,H,W,3)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros",
                        align_corners=True)                              # (1,1,D,H,W)
    return out.view(D, H, W)


def reslice_volume(V, d_si_mm, d_ap_mm, ap_axis: str = "H", spacing=SPACING_MM):
    """Scalar (SI + AP, no tilt) shim over `reslice_volume_vec`. Used by the example
    renderer for coronal/sagittal/axial views. Returns (D, H, W) float32."""
    disp = _build_disp_dhw(
        torch.as_tensor(float(d_si_mm)), torch.as_tensor(float(d_ap_mm)), ap_axis)
    return reslice_volume_vec(V, disp, spacing)
