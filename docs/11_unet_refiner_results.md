# 11 — 3D UNet refiner: results (frozen / joint + SSIM term)

> **TL;DR & takeaway**
> Results for the optional 3D UNet refiner on the splat (implementation:
> `version_history/v2_unet_refiner.md`; motivation: `docs/10` — ~75% of the breathing blur is the splat
> renderer). **The refiner conclusively helps on a fixed geometry: +~1.0 dB motion / +~1.5 dB bbox** on
> held-out breathing val (frozen run, n=100). That frozen V_refined motion (20.07) **beats the best
> no-refiner baseline** (var2, 19.28 at a more-trained geometry) by ~+0.8 dB — i.e. **deblurring the
> splat beats training the geometry further; the splat, not the geometry, was the bottleneck.** BUT the
> gain is almost all **dark-spot / coverage correction, NOT deblurring**: sharpness rises only
> 0.668 → ~0.69× GT (~6–8% of the gap closed). Under L1 (mean-seeking) that's expected. Adding a **2D
> per-slice SSIM term** (`L_post = L1 + 0.1·(1−SSIM_2D)`) reaches the sharpness ceiling *faster*
> (0.692× by ep26 vs L1's ~ep62) at **no PSNR cost**, but **plateaus at the same ~0.69× ceiling** — the
> splat discards detail neither loss can invent. **Verdict:** black spots — solved; blur — only
> marginally improved by either loss. SSIM is worth keeping (faster, free), but the real blur lever is
> *upstream* (higher-res scatter, learned splat, coverage-aware loss). Source reports: `_html/10`
> (first snapshot), `_html/11` (motion-led full analysis), `_html/12` (SSIM). Eval: `tools/eval_refiner.py`,
> `tools/measure_sharpness.py`.

## The runs

| run | job | geometry (VGGT) | refiner | role |
|---|---|---|---|---|
| **frozen** | 51876098 | FROZEN at var2 ep-59 seed (`t59w6nqy`: resp, z, no-t, aggft) | trains | isolate the pure splat-deblur gain (geometry fixed) |
| **joint** | 51876099 | finetunes from VGGT-1B base | trains | geometry + refiner co-adapt |
| **var2** | 51862799 | finetunes (resumed ep59→) | none | no-refiner control (V_canon only) |
| **frozen-SSIM** | 51950141 | same frozen seed | trains (SSIM loss) | controlled L1-vs-SSIM comparison |

All apply the **identical deterministic breathing corruption at val** (per-`seq_index` seed), so
V_refined(frozen) vs V_refined(joint) vs V_canon(var2) vs identity are on **one yardstick** (unlike
report 07's cross-task issue — see `docs/05`). Caveat: they validate at **unequal epochs**.

## Headline comparison — motion primary (n=100 held-out breathing val, `_html/11`)

| model (output) | motion PSNR | Δ motion vs var2 | bbox PSNR | sharp/GT | status |
|---|---|---|---|---|---|
| **frozen — V_refined** | **20.07** | **+0.79** | **27.94** | 0.688 | decelerating, still rising (lower bound) |
| joint — V_refined | 19.64 | +0.36 | 27.37 | 0.651 | NOT converged (ep ~22/200) |
| var2 — V_canon (best no-refiner) | 19.28 | 0.00 | 26.73 | — | roughly plateaued (~ep 85) |
| frozen — V_canon (pinned geom) | 19.04 | −0.24 | 26.37 | 0.667 | frozen geometry (refiner input) |
| joint — V_canon | 19.02 | −0.26 | 26.30 | 0.623 | co-adapting |
| identity (do-nothing) | 16.59 | −2.69 | 23.23 | — | floor |

On the **frozen** run the refiner lifts motion **19.04 → 20.07 (+1.02)** and bbox **26.37 → 27.94
(+1.57)**.

### Lever-efficiency read (the key result)

Frozen's V_canon (motion 19.04) comes from the *older, frozen* var2-ep59 geometry — yet a refiner on
top of it reaches **20.07 motion, +0.79 dB above var2's V_canon (19.28)**, and var2 trained that
geometry **~25 epochs further with no refiner**. So **deblurring the splat > training the geometry
more**: the splat (`docs/10`), not the geometry, was the motion bottleneck. *(Caveat: the two
geometries differ → this is a lever-efficiency comparison, not a strict head-to-head.)*

## Honesty flags (read first)

- **frozen V_canon is exactly flat** (geometry frozen, slope +0.0000/1k-step). V_refined is
  *decelerating but still slowly rising* (19.97 → 20.03 over last 10 epochs, +0.0084/1k-step) → the
  frozen number is a **lower bound, not a converged plateau**.
- **joint is NOT converged** (ep ~22/200, V_refined slope +0.0598/1k-step, ~7× faster than frozen).
  "frozen > joint now" will likely narrow as joint's geometry catches up.
- var2 roughly plateaued (motion fluctuating 19.28–19.55, slope +0.0070/1k-step).
- **Sharpness gain is modest** — under L1 (mean-seeking) the refiner mostly fixes accuracy/coverage,
  not high-frequency detail.

## Is it a real deblur? Sharpness scorecard

Sharpness = in-plane gradient energy ÷ GT (1.0 = as sharp as GT). If V_refined merely *smoothed*,
sharpness would *fall*; instead it **rises** in both runs (so it's not blurring) — but the rise is
**modest** (frozen 0.667 → 0.688; joint 0.623 → 0.651). Most of the PSNR gain is accuracy/coverage-error
correction, not high-frequency recovery — exactly what L1 predicts.

| original goal (from `docs/10`) | fixed? | evidence |
|---|---|---|
| Dark / dim under-covered spots (coverage-averaging artifacts) | ✅ **yes, substantially** | this is where ~all of the +1.0 dB motion / +1.5 dB bbox comes from — V_refined fills/brightens coverage-starved voxels and smooths seams |
| Blurriness (lost in-plane high frequency) | ❌ **mostly NOT** | only ~6% (frozen) / ~8% (joint) of the blur-gap to GT is closed; V_refined still only ~0.69× GT sharpness |

The refiner is currently a **coverage / intensity corrector more than a deblurrer**, because L1 rewards
getting the average intensity right (fills dark spots) but does not reward sharp edges (won't deblur).

## The SSIM term (`_html/12`)

To attack the blur, a structural-similarity term is added to the refiner loss (V_refined only; the
point head's L1 supervision is untouched):

```
L_post = 1.0 · L1(V_refined, V_gt) + 0.1 · (1 − SSIM_2D(V_refined, V_gt))
```

- **2D per-slice, in-plane** (Y–X): `(B,D,H,W) → (B·D,1,H,W)`; the through-plane Z is never mixed. The
  cube is anisotropic (8 mm Z over 12 slices vs 1.4 mm in-plane) and the blur we fight is in-plane, so
  a 2D window is correct — a 3D window would mix the coarse Z axis.
- **Reference-based / two-sided:** penalizes blur (too smooth vs GT) *and* hallucinated/misplaced edges
  (sharp where GT is smooth). Can't be gamed by cranking contrast. **NB: this is NOT total variation** —
  TV penalizes *any* gradient and would blur *more*; SSIM/gradient losses reference GT and reward
  matching its edges. (Recipe per Zhao et al. 2017; `fused-ssim` already in the env.)
- **L1 stays in the mix** because SSIM is contrast-normalized (blind to a uniform brightness offset);
  L1 nails absolute intensity, SSIM nails structure.

Controlled frozen comparison (same seed, same frozen geometry, byte-identical V_canon = 26.41 bbox /
19.07 motion; only the loss differs):

| metric (n=60) | raw splat | L1 @ep95 | SSIM @ep26 |
|---|---|---|---|
| sharpness / GT | 0.668 | 0.694 | 0.692 |
| bbox PSNR (dB) | 26.49 | 28.16 | 27.69 |
| motion PSNR (dB) | 19.14 | 20.26 | 19.91 |

- **SSIM is active and costs no accuracy:** a tiny PSNR "tax" early (optimizing structure not MSE),
  crossing over ~step 25k, then ties L1 at matched steps.
- **SSIM reaches the sharpness ceiling faster** (0.692× by ep26 vs L1 needing ~ep62 for 0.688) — but
  **both plateau at the same ~0.69× GT ceiling** (~8% of the gap). The splat discards detail neither
  loss can invent; the refiner can only recover what is still latent in `[V_canon, coverage]`.
- The epoch gap (L1 ep95 vs SSIM ep26, 3.6× more trained) means SSIM's lower *current* PSNR is mostly
  immaturity, not a deficit — at matched steps they tie.

## Verdict & next steps

- **Black spots: solved** by the refiner (either loss).
- **Blur: only marginally improved.** SSIM helps a bit more and a lot faster, but both losses hit a
  ~0.69× GT ceiling **set by the splat**.
- **SSIM is worth keeping** — faster convergence, no PSNR cost, may still edge ahead once mature.
- **Decisive matched-epoch test pending:** SSIM @ep50 vs L1's saved `checkpoint_50`, and again at ep95.
- **If blur is the priority, the real lever is upstream:** fix the splat/coverage (higher-res scatter,
  learned splat, coverage-aware loss) or add a gradient-matching term — the information SSIM needs to
  sharpen further isn't in the splat anymore. (See `version_history/v2_unet_refiner.md` "refiner vs
  learned decoder.")
- **Hallucination check status:** no fabricated anatomy visible in the n=2 qualitative panels (verified,
  not assumed); a broader OOD audit on held-out subjects is future work.

Reproduce: `tools/eval_refiner.py` (last-checkpoint re-eval, sharpness + panels), trajectories parsed
from `slurm_logs/*refiner*`, report builders `_html/build_refiner_results_report.py`,
`_html/build_refiner_full_analysis.py`, `_html/build_ssim_analysis.py`.
