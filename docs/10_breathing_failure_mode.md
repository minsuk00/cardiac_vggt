# 10 — The breathing reconstruction failure mode (splat vs breathing blur)

> **TL;DR & takeaway**
> Dissects *why* the breathing-corrupted reconstruction (the real-time free-breathing deployment case)
> is blurry, with measured proofs (snapshot 2026-06-16, `tools/measure_sharpness.py`, n=12 val each).
> **The recon carries only 0.65× the ground truth's high-frequency detail.** Decomposing it:
> **~75% of that blur is the SPLAT renderer itself, ~25% is the breathing.** Even on *perfect clean
> inputs* the splat alone already drops sharpness to 0.74× GT (the ~75%); breathing inputs add the rest
> down to 0.65×. Critically, the **trained model's sharpness ≈ the raw-splat sharpness** — the point
> head fixes *where* anatomy goes (PSNR climbs well above identity) but adds essentially no
> high-frequency detail over the raw splat, because the coverage-averaging splat caps it. It is **blur,
> not black holes**: the cube stays filled (coverage_frac ≈ 0.713 ≥ the ~0.69 tissue fraction); only
> ~5–6% of tissue voxels are under-covered (the localized dark spots). **Implication & fix, biggest
> lever first:** a learned decoder / 3D UNet refiner on the splat attacks the ~75% chunk *without
> solving motion* — this is the entire motivation for the refiner (impl: `version_history/v2_unet_refiner.md`;
> results: `docs/11`). Better motion correction attacks the smaller ~25% chunk. Source report:
> `_html/08_breathing_failure_mode.html`; companion: `_html/07` / `docs/05` (the model-vs-model
> respiratory-variant comparison).

The recon under scrutiny is the model **fed breathing-corrupted inputs** (the real-time
free-breathing deployment case the project goal hinges on). Sharpness here = mean in-plane gradient
magnitude over the anatomy bbox (anatomy voxels only), normalized to GT: **1.0 = as sharp as GT**;
below 1.0 = high-frequency detail lost = blur. If the recon were sharp but mis-placed (a geometry
error, not blur), sharpness would stay ≈ 1.0 and error would show only as edge mismatches.

## 1. The blur decomposition (splat vs breathing)

| stage | sharpness ÷ GT | lost | share of the blur |
|---|---|---|---|
| ground truth | 1.00 | — | — |
| splat alone (perfect/clean inputs) | 0.74 | 0.26 | **~75%** |
| + breathing inputs (deployment) | 0.65 | 0.09 | **~25%** |

So the blur is **~75% the splat renderer, ~25% the breathing**. The splat smears even with perfect
inputs (§2 explains why); breathing adds a quarter on top by making slices image *shifted* anatomy
that then gets averaged together. The splat is the larger, **motion-independent** lever.

*(Aside — this is a pipeline limit, not the resp model being a bad reconstructor: on identical inputs
the resp and no-resp models are within ~2% sharpness of each other, and the resp model is more
accurate. The model-vs-model comparison is in `docs/05` / `_html/07`.)*

## 2. Why the splat caps sharpness

The model never outputs a volume directly. It outputs, per input-slice pixel, a 3-D position
(`world_points = scanner_coords + Δ`) in the normalized canonical cube. Those (position, intensity)
pairs are splatted into the voxel grid — each pixel drops its intensity into its 8 surrounding voxels,
trilinear-weighted (`vggt/utils/splat.py:splat_to_volume`):

```
volume   = Σ over pixels of (trilinear_weight · intensity)   # numerator
coverage = Σ over pixels of (trilinear_weight)               # denominator
V_canon  = volume / (coverage + 1e-6)                        # = coverage-weighted average
```

A voxel's value is the **coverage-weighted average of every input pixel that landed near it**. The
model only chooses *where* pixels land (via Δ); the splat does the rendering. Contributing pixels that
disagree get averaged → blur. This is why the splat — not the model — caps how sharp the output can be.

## 3. Coverage, and whether black voxels contribute

- `coverage[voxel]` = accumulated trilinear weight at that one voxel (a per-voxel soft count of "how
  much input landed here"). Can be 0, 0.4, 2.8, …
- **`coverage_frac` ≈ 0.713** = fraction of cube voxels whose coverage > 0 (what fraction of the box got
  hit by *any* input). NOT an average coverage value.
- `coverage_mean` = average `coverage[voxel]` over voxels (≈ how many input pixels stack per covered
  voxel) — a different quantity.
- `gt_coverage_frac` ≈ 0.69 = fraction of voxels where the *target* has tissue (`V_gt > 1e-3`) — a
  property of the answer key: a heart-in-a-chest fills only ~69% of a rectangular box; the rest is
  air/zero-padding that should be black.

**Do black voxels contribute? No.** Before splatting, each input pixel is gated:
`splat_weight = (intensity > 1e-3)` (`training/loss.py:342`). A black pixel (zero-padding, or a slice
that drifted off-FOV) has weight 0, so it adds to *neither* the value numerator *nor* the coverage
denominator. A voxel that receives no non-black input → `0 / (0 + 1e-6) ≈ 0` → a true black hole. Black
input never dilutes or fills anything.

## 4. Proof: it's blur, and the splat is the bottleneck

`tools/measure_sharpness.py`, resp (v2) and no-resp (v4) models, 12 val samples each:

| model | protocol | recon ÷ GT | raw-splat ÷ GT | coverage_frac | bbox PSNR |
|---|---|---|---|---|---|
| var2 resp_no_t | clean | 0.736 | 0.754 | 0.719 | 30.86 |
| var2 resp_no_t | breathing | 0.649 | 0.635 | 0.713 | 26.26 |
| var4 noresp_no_t | clean | 0.751 | 0.754 | 0.720 | 30.00 |
| var4 noresp_no_t | breathing | 0.641 | 0.635 | 0.705 | 23.14 |

Three reads: (1) recon ÷ GT < 1 everywhere → genuinely blurrier than GT. (2) Even *clean* is only
~0.74× → most of the blur is the splat itself, present without any breathing. (3) **recon ≈ raw-splat
sharpness** → the trained model adds essentially no high-freq over the raw splat; it corrects *position*
(PSNR climbs well above identity) but the splat caps *detail*. Breathing adds a second, smaller blur on
top (the clean → breathing drop).

## 5. Blur vs black holes — which dominates the dark spots you see

**coverage is a holes detector** — it only tells you whether a voxel received any input (filled vs
empty); it says **nothing about blur**. A voxel can be fully covered and blurry (its contributing
pixels disagreed and got averaged). Blur is proven only by the §4 sharpness number (0.65× GT).

Black holes require a voxel to receive zero corrected coverage. But `coverage_frac ≈ 0.713 ≥` the
~0.69 tissue fraction — the inputs fill at least as much of the box as the target occupies — so at
current amplitudes (~8–24 mm ≈ 1–3 planes at 8 mm spacing) there are **no gaping holes**; the cube is
filled, just blurrily. The dark spots are real but **localized: only ~5–6% of tissue voxels are
under-covered**, concentrated at specific z-planes and tissue edges. Holes *would* emerge with larger
motion or sparser input (fewer slices → planes that lose all corrected coverage).

## 6. The fix, biggest lever first

1. **Learned decoder / 3D UNet refiner on the splat** — attacks the ~75% (motion-independent) chunk
   without solving motion. → implemented in `version_history/v2_unet_refiner.md`, results in `docs/11`.
2. **Better motion correction** — attacks the smaller ~25% breathing chunk (e.g. respiratory
   conditioning; see `docs/04`, `docs/07`).

Reproduce: `tools/measure_sharpness.py`.
