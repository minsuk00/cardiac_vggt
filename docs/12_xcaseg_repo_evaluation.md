# 12 — xcaseg repo evaluation (rejected)

> **TL;DR & takeaway**
> Evaluated `github.com/liamjwang/xcaseg` (commit `151e951`, 2025-05-01, no license) as a possible
> source for the respiratory-motion-simulation direction in `docs/01`. **Verdict: DON'T ADOPT — nothing
> to drop in.** Despite the name, **xcaseg ≠ XCAT**: it is a synthetic **X-ray Coronary Angiography
> seg**mentation data generator (a UMich EECS-545 course project; internal codename `synthca`), not a
> cardiac-MRI or motion tool. Its single motion component (`CardiacWarpTransform`) simulates *cardiac
> contraction* — which our real CMRxRecon cine already contains for free as 12 measured phases — **not
> the respiratory motion we actually need to simulate** (grep for `respir|breath|diaphragm` → nothing).
> The warp is also an **uncalibrated artistic Perlin-noise deformation** (no mm, no strain, no
> literature grounding, no spacing awareness), welded to a NeRF ray-marching X-ray renderer
> (`nerfstudio`) on continuous query positions — not our discrete monai voxel grid. **The only takeaway
> is a design-pattern confirmation we already reached independently in `docs/01`:** compose the motion
> warp *before* sampling ("deform-then-reslice"). Source report: `_html/05_xcaseg_evaluation.html`.
> This is the X-ray-angiography analogue of the MRXCAT/XCAT rejection in `docs/01 §6`.

## What the repo actually is

- **`xcaseg` = X-ray Coronary Angiography segmentation**, not XCAT. Goal: generate synthetic
  coronary-angiogram videos with ground-truth vessel masks to train a 2D vessel-segmentation network
  without real labeled angiograms. The real work lives in `pkgs/xcaseg/xcaseg/datagens/`; the rest is a
  heavyweight monorepo vendoring a dozen unrelated packages (`nerfstudio`, `dust3r`, `mvdust3r`,
  Video-Swin, apex, a CroCo fork).
- **Maturity signals it's a student deliverable, not a library:** single squashed commit, no license,
  leftover scratch files (`asdfsadf.py`, `dataset_synthca copy.py`), `TODO: disable cardiac warp`
  comments in the hot path.

## The generation pipeline (`synthca_v3`)

One synthetic example is built entirely from a generic CT plus procedural geometry — no real angiogram
in the loop: (1) load a CT as a static density volume; (2) grow a synthetic vessel tree
(`treegen.generate_tree`); (3) apply a time-dependent `CardiacWarpTransform` to both volumes via a
`SequentialSampleTransform` chain; (4) differentiable X-ray rendering — march C-arm camera rays through
the warped volumes, accumulate density, apply Beer–Lambert `exp(−Σdensity)` to get a 2D projection;
(5) write per-frame X-ray PNG/npz + depth + poses, tree channel doubling as the segmentation label.
Everything below the warp (vessel trees, mass attenuation, C-arm geometry, NeRF renderer) is
**X-ray-specific and irrelevant to MRI volume reconstruction**.

## The one relevant component: `CardiacWarpTransform`

`datagens/components/cardiac_warp.py` — a `torch.nn.Module` mapping `(positions, t) → (warped_positions, t)`,
the only motion model in the repo:

- **Periodic time drive.** `t ∈ [0,1)` feeds harmonic signals (`sin(2πt)+0.4·sin(4πt)` for a global
  rotation angle; a 3-harmonic signal for the contraction phase) so motion loops over a cardiac cycle.
- **Radial contraction.** Core displacement is `vec_to_center · sq` — a crude myocardial squeeze.
- **Perlin-noise spatial modulation.** A fixed random 3-channel 3D fractal-noise field multiplies the
  displacement → spatially non-uniform, randomized per instance.
- Plus a small global rigid rotation (≈0.05 rad) and translation (≈0.1 normalized units).

**It is artistic, not anatomical.** Magnitudes are hand-tuned to look plausible in a 2D projection: no
mm calibration, no strain model, no literature reference, no spacing awareness (grep: zero `mm` /
`strain` / `spacing` / `diaphragm`). It is domain-randomization motion (maximize variety so a segmenter
generalizes) — the **opposite** of the calibrated, reproducible motion `docs/01` requires. And there is
**no respiratory motion anywhere** — the actual gap in our pipeline.

## Why it doesn't fit VGGT-MRI

| Dimension | xcaseg provides | We need | Fit |
|---|---|---|---|
| Imaging domain | 2D X-ray angiography (Beer–Lambert, C-arm) | 3D cine-MRI magnitude volumes | ✗ |
| Motion type | synthetic cardiac contraction | already have real cardiac motion (12 phases); need **respiratory** | ✗ |
| Respiratory model | none | rigid SI translation ~10–15 mm, decoupled clock (`docs/01`) | ✗ |
| Calibration | uncalibrated Perlin/artistic | literature-validated, mm-calibrated, reproducible | ✗ |
| Data structure | continuous query positions in a NeRF Field | discrete canonical voxel grid `(256,256,12)` | ~ |
| Deform-then-sample pattern | `SampleTransform` chain warps positions before sampling | exactly our "deform-then-reslice" (`docs/01 §7`) | ✓ idea only |
| Engineering cost | drags in `nerfstudio` + huge env; no license | a ~50-line rigid-translate-then-reslice on the cache | — |

**Conclusion:** don't adopt. The deform-then-reslice ordering it confirms is the only transferable
idea, and `docs/01` already specifies it. Respiratory simulation was instead implemented from scratch
(`training/data/respiratory.py`; see `docs/01`).
