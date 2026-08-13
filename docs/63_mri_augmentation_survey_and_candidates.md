# docs/63 — MRI Augmentation Survey: Cross-Scanner/Vendor Generalization Candidates

**Date:** 2026-08-01
**Context:** GPU aug pipeline (`training/data/gpu_aug.py`) currently has: in-plane affine (rotate ±180°, translate ±16 px, scale ±5% anisotropic), gamma contrast, bias field, and Lujan respiratory motion sim. This document surveys what else could be added for OOD robustness, why, and how easy each is to add.

---

## 1. Why Cross-Scanner Generalization Is Hard in MRI

Unlike CT (Hounsfield units = physical densities), **MRI intensity values are non-quantitative**. They vary with:

- **Hardware / Vendor**: field strength (1.5T vs 3.0T), RF coil geometry (Siemens vs Philips vs GE)
- **Pulse sequence**: TE, TR, flip angle, receiver bandwidth, reconstruction filter
- **Protocol**: matrix size, voxel size, slice thickness, phase-encode direction, acceleration factor (GRAPPA/SENSE)

Neural networks overfit to site-specific "fingerprints" (contrast distributions, noise, coil shading). The CMRxRecon2024 training set is a single-site, single-protocol, single-vendor cohort — any OOD evaluation (ACDC, MIITT, OCMR, real-time free-breathing cine) crosses scanner or protocol boundaries.

---

## 2. Current Pipeline (Active Default: `moderate` Tier)

From `training/data/gpu_aug.py` and `training/config/default.yaml`:

| Transform | Setting | Purpose |
| :--- | :--- | :--- |
| `RandAffined` rotate | ±180° in-plane, prob=0.6 | Patient orientation spread (MIITT ~180° off CMRx mode) |
| `RandAffined` translate | ±16 px H/W, D frozen | Patient positioning variability |
| `RandAffined` scale | ±5% anisotropic H/W | Small zoom variation |
| `RandAdjustContrastd` | gamma (0.7–1.5), prob=0.6 | Contrast/protocol variation |
| `RandBiasFieldd` | degree=3, coeff ±0.5, prob=0.5 | B0/B1 RF coil shading inhomogeneity |
| `RandRicianNoise` | disabled | (see §3.6 — not worth adding) |
| `RandGaussianNoise` | disabled | Real degradation is structured, not i.i.d. |
| Respiratory motion (Lujan) | always on (train+val) | Free-breathing SI+AP deform |

Rotation (full-circle ±180°) was the single biggest OOD driver — +0.48 dB MIITT, +0.47 dB ACDC (docs/46 §3 C2). The moderate tier is the shipped default as of 2026-08-01 (docs/46, docs/58).

---

## 3. Augmentation Candidates Surveyed

### 3.1 Fourier Amplitude Mixup (FDA)

**What it is:**
The 2D Fourier transform of an image has two components:
- **Phase spectrum**: WHERE structures are (edges, shapes — anatomy). Domain-invariant.
- **Amplitude spectrum**: low-level intensity statistics, texture, contrast style. Scanner-specific.

FDA swaps the **low-frequency center region** of the amplitude spectrum between two subjects while keeping each subject's phase spectrum. Result: Patient A's heart anatomy under Patient B's scanner contrast profile.

**Why relevant for VGGT-MRI:**
DINOv2 (our patch embed + aggregator backbone) is sensitive to global low-frequency contrast distributions across vendors. Standard gamma/bias-field augmentation shifts the global curve but preserves the overall texture character. FDA goes further by literally adopting another subject's acquisition style.

**Implementation:**
```python
def apply_fda(phases, beta=0.08):
    """phases: (B, T, D, H, W). Style donor = torch.roll(phases, 1, 0)."""
    trg = torch.roll(phases, shifts=1, dims=0)   # adjacent subject in batch
    fft_s = torch.fft.fftshift(torch.fft.fft2(phases.float()), dim=(-2,-1))
    fft_t = torch.fft.fftshift(torch.fft.fft2(trg.float()),    dim=(-2,-1))
    amp_s, amp_t = torch.abs(fft_s), torch.abs(fft_t)
    phase_s = torch.angle(fft_s)
    H, W = phases.shape[-2:]
    ch, cw = int(H * beta), int(W * beta)
    h2, w2 = H // 2, W // 2
    amp_mix = amp_s.clone()
    amp_mix[..., h2-ch:h2+ch, w2-cw:w2+cw] = amp_t[..., h2-ch:h2+ch, w2-cw:w2+cw]
    return torch.fft.ifft2(
        torch.fft.ifftshift(amp_mix * torch.exp(1j * phase_s), dim=(-2,-1))
    ).real.clamp(0,1).to(phases.dtype)
```

**Cost:** ~0.26 ms/call (GPU A40, 50-repeat timing 2026-08-01).
**Not in batchaug** — custom code in `gpu_aug.py`.
**beta parameter:** 0.05–0.10. Larger beta swaps more of the amplitude, including structural edges.

**References:**
- Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation", CVPR 2020. arXiv:2004.05498
- Xu et al., "FreeSDG: Frequency-mixed Single-source Domain Generalization for Medical Image Segmentation", MICCAI 2023.

---

### 3.2 Simulated Low Resolution

**What it is:**
Randomly downsamples a 2D/3D image by a factor s in [0.5, 0.85], then upsamples back to the original size (nearest downsample, trilinear upsample). Result: blurred image with visible pixelation/blocking, simulating lower-resolution acquisition.

**Why relevant for VGGT-MRI:**
CMRxRecon2024 has ~1.4 mm in-plane voxels. ACDC ranges 1.25–1.9 mm, MIITT varies by protocol, real-time cine often has 2.0–2.5 mm in-plane (larger voxels for speed). A DINOv2 backbone trained only on sharp 1.4 mm data will over-rely on fine edge features that disappear at lower resolution.

**Implementation:**
Available natively in batchaug as `RandSimulateLowResolutiond`. Drop-in one-liner:
```python
_B.RandSimulateLowResolutiond(
    keys=["phases"], prob=0.4,
    zoom_range=(0.5, 0.85),
    downsample_mode="nearest",
    upsample_mode="trilinear",
    align_corners=True,
)
```

**Cost:** ~0.54 ms/call (GPU A40).
**In batchaug:** Yes.
**Note:** Apply to `phases` only (not `content_mask`) — the mask should stay crisp.

**References:**
- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods 2021. DOI: 10.1038/s41592-020-01008-z

---

### 3.3 Gibbs Ringing

**What it is:**
When k-space is truncated (not fully sampled or zero-padded before IFFT), the image reconstruction is equivalent to convolving with a sinc function — oscillating ripples near sharp tissue boundaries (Gibbs phenomenon). Controlled by `alpha`: 0 = full k-space (identity), 1 = DC only (maximum ringing).

**Implementation:**
Available in batchaug as `RandGibbsNoised`:
```python
_B.RandGibbsNoised(keys=["phases"], prob=0.3, alpha=(0.5, 0.75))
```
Internally: 3D FFT → fftshift → spherical k-space mask → ifftshift → IFFT.

**Cost:** ~0.41 ms/call (GPU A40).
**In batchaug:** Yes.

**References:**
- Pérez-García et al., "TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning", Comput. Methods Programs Biomed. 2021. DOI: 10.1016/j.cmpb.2021.106236

---

### 3.4 Phase-Encode Ghosting

**What it is:**
Different from Gibbs ringing. Caused by **periodic intensity variation** in k-space along the phase-encode direction — e.g., pulsatile aortic blood flow, or heartbeat-to-heartbeat variation in non-gated acquisitions. Result: faint **duplicate ghost copies** of the anatomy offset by FOV/N along the phase-encode axis (where N is the periodicity of the corruption).

**Why relevant for VGGT-MRI:**
Extremely common in cardiac cine. Every acquisition line sampled during a different phase of the cardiac or respiratory cycle contributes to ghosting. Real-time free-breathing cine (our inference target) will have **much worse** ghosting than gated breath-hold CMRx data — the model currently never sees this artifact during training.

Note: Gibbs ringing and phase-encode ghosting are distinct artifacts often confused:
- Gibbs: resolution truncation → ripples at edges (symmetric, tied to structure geometry)
- Ghosting: periodic k-space line corruption → displaced ghost copy (offset by FOV/N, often from blood flow)

**Implementation:**
Not in batchaug — ~20 lines custom PyTorch:
```python
def apply_ghosting(phases, prob=0.5, num_ghosts=2, intensity=0.35):
    """Attenuate periodic phase-encode k-space lines → ghost copies."""
    if torch.rand(1).item() > prob:
        return phases
    ksp = torch.fft.fft2(phases.float())
    step = phases.shape[-1] // num_ghosts   # phase-encode axis = W
    ksp[..., ::step] *= (1.0 - intensity)
    return torch.fft.ifft2(ksp).real.clamp(0, 1).to(phases.dtype)
```

**Cost:** ~0.3–0.5 ms estimated (two 2D FFTs).
**In batchaug:** No — custom implementation required.

---

### 3.5 Isotropic Heart Scale (`RandZoomd`)

**What it is:**
Current `RandAffined(scale_range=(0.0, 0.05, 0.05))` samples H and W scaling **independently** — LV cross-section can distort into an ellipse. `RandZoomd` samples a **single scalar** z ~ U(min, max) and applies it equally to all spatial axes — LV stays circular.

**Why relevant:**
Real patient hearts vary in size: small/normal vs dilated cardiomyopathy, pediatric vs adult. Different operators also select different FOV zoom levels. ±15% isotropic zoom covers this spread without distortion.

**Lazy fusion with RandAffined:**
`RandZoomd`'s inner `RandZoom` has a `to_affine` method. batchaug's `Compose(lazy=True)` (compose.py:140-151) fuses consecutive geometric affines via matrix multiplication and materializes everything in **one single `grid_sample` call**. So adding `RandZoomd` before `RandAffined` costs zero extra grid_sample calls.

**Implementation:**
```python
# Before RandAffined in the transforms list:
_B.RandZoomd(
    keys=keys,
    prob=0.5, min_zoom=0.85, max_zoom=1.15,
    mode={"phases": "bilinear", "content_mask": "nearest"},
    padding_mode={"phases": "zeros", "content_mask": "zeros"},
),
# RandAffined: remove scale_range
_B.RandAffined(keys=keys, prob=0.9,
    rotate_range=(float(np.deg2rad(180)), 0.0, 0.0),
    translate_range=(0.0, 20.0, 20.0),
    scale_range=None,   # removed; isotropic zoom handled by RandZoomd above
    padding_mode="zeros"),
```

**In batchaug:** Yes.

---

### 3.6 Rician Noise (Rejected)

Considered but **not recommended**. Rician noise is the physically correct noise model for MRI magnitude images:
  output = sqrt((signal + n1)^2 + n2^2),   n1, n2 ~ Gaussian(0, sigma)

batchaug has `RandRicianNoised` ready to use. However, the same rationale that disabled Gaussian noise applies equally:

- Real OOD degradation in cardiac cine MRI is **structured** (aliasing from undersampling, off-resonance blur, ghosting from blood flow, eddy current shading), not spatially i.i.d. per-pixel noise.
- Rician adds a noise floor in background regions (physically correct for 1.5T vs 3.0T SNR differences) but this is minor compared to the structured artifacts above.
- Bias field already covers the dominant scanner-to-scanner intensity variation.

The advantage of Rician over Gaussian is purely the distribution shape (never negative, Rayleigh background floor), not the spatial structure. Neither belongs in our pipeline.

---

## 4. GPU Timing Summary (A40, 50 repeats, 2026-08-01)

All timings on a single (1, 1, 1, 246, 256) float32 tensor (one SAX slice as 5D batchaug input):

| Augmentation | Time | % of ~3960 ms train step |
| :--- | :--- | :--- |
| Rician noise | 0.165 ms | 0.004% |
| FDA amplitude mixup | 0.260 ms | 0.007% |
| Gibbs ringing | 0.406 ms | 0.010% |
| Simulated low-res | 0.536 ms | 0.014% |

All are negligible. Adding all four simultaneously would cost ~1.4 ms — well under 0.1% of a training step.

Visualization script: `tools/render_aug_comparison.py` → `result/aug_comparison.png`

---

## 5. Recommended Additions to `gpu_aug.py`

Priority order based on expected impact for VGGT-MRI OOD generalization:

| Priority | Augmentation | Why | Implementation |
| :--- | :--- | :--- | :--- |
| 1 | Isotropic zoom (`RandZoomd`) | Real heart-size variability without LV distortion | 1 line + remove scale_range |
| 2 | Simulated Low-Res | Voxel-size/resolution protocol differences — directly relevant to real-time cine | 1 line (batchaug) |
| 3 | Phase-Encode Ghosting | Most clinically relevant artifact for non-gated cine | ~20 lines custom |
| 4 | Gibbs Ringing | K-space truncation artifact — common in accelerated imaging | 1 line (batchaug) |
| 5 | FDA Amplitude Mixup | Cross-vendor contrast style transfer | ~20 lines custom |

### Suggested aggressive tier update

```python
transforms = [
    _B.RandZoomd(keys=keys, prob=0.5, min_zoom=0.85, max_zoom=1.15, ...),  # NEW isotropic scale
    _B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
    _B.RandAffined(
        keys=keys, prob=0.9,
        rotate_range=(float(np.deg2rad(180)), 0.0, 0.0),
        translate_range=(0.0, 20.0, 20.0),
        scale_range=None,           # removed — RandZoomd above replaces this
        padding_mode="zeros",
    ),
    _B.RandAdjustContrastd(keys=["phases"], prob=0.75, gamma=(0.6, 1.7)),
    _B.RandBiasFieldd(keys=["phases"], prob=0.7, degree=3, coeff_range=(-0.5, 0.5)),
    _B.RandSimulateLowResolutiond(                                          # NEW
        keys=["phases"], prob=0.4,
        zoom_range=(0.5, 0.85),
        downsample_mode="nearest",
        upsample_mode="trilinear",
        align_corners=True,
    ),
    _B.RandGibbsNoised(keys=["phases"], prob=0.3, alpha=(0.5, 0.75)),      # NEW
    # apply_ghosting(batch["phases"], ...)   # NEW custom, applied after Compose in gpu_augment_batch
    # apply_fda(batch["phases"], ...)        # NEW custom, applied after Compose in gpu_augment_batch
]
```

Note: FDA and ghosting operate on `batch["phases"]` directly (not via batchaug's dict wrapper) and should be applied in `gpu_augment_batch()` after the batchaug Compose call, before slice re-extraction.

---

## 6. Key Design Constraints (Not to Violate)

- **In-plane only**: No through-plane (D-axis) rotation, elastic deform, or scale. Z spacing is 5–12 mm (native-z, docs/58) — coarse and anisotropic vs 1.4 mm in-plane.
- **Apply to `phases` only** for intensity transforms: `content_mask` must stay binary for bbox recomputation.
- **No `scanner_coords` update needed**: coords are a pure geometric mapping from pixel index, decoupled from image content.
- **Post-aug consistency**: After any spatial aug, `gt_target_volume`, `anatomy_bbox`, and `images` must be re-derived from the augmented `phases`. This is already done in `gpu_augment_batch()`.
- **Val never affine-augments**: respiratory motion applies in val (deterministic per seq_index); spatial affine does not.
- **Bias field clamp**: photometric ops can push intensities outside [0,1]; `phases_aug` is clamped before deriving `gt_target_volume` to prevent unlearnable L1 residuals.
