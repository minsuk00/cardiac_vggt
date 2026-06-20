# 09 — Aggregator-finetune & per-phase characterization

> **TL;DR & takeaway**
> Two 2026-06-10 multi-phase studies on the canonical-grid pipeline, read on the **motion-PSNR**
> metric (dynamic heart voxels only, ~8% of the bbox; identity/do-nothing floor ≈ 20.6 dB).
> **(1) Finetuning the aggregator — not just the point head — is what corrects motion.** The two
> aggregator-finetune ("aggft") runs reach **+2.7 to +3.2 dB** over the no-correction baseline; the
> head-only run adds only **+0.6 dB** and is *at/below the baseline at ED* (it barely moves the heart).
> **(2) Training on all 12 target phases is free** — `allphases_aggft` ties the `{0,7}` specialist on
> t0/t7, so one model covers the whole cardiac cycle at no quality cost. **(3) The z/t Fourier
> embeddings are worth ~0.8 dB of motion PSNR** (larger than the ~0.5 dB it looked on bbox); dropping
> them also bleeds content into the out-of-FOV padded z-planes (full PSNR drops ~3 dB). **(4) There is a
> ~1.7 dB mid-systole dip** — quality is best at ED, worst at peak contraction (t4). **(5) Encouraging
> info-contract signal:** the `no_zt` model, which removes per-slice input-phase conditioning, still
> reconstructs at ~22.8 dB motion — the network infers enough phase/geometry from the images themselves.
> Source reports: `_html/01_four_run_comparison.html`, `_html/02_two_model_analysis.html`. Inference
> script: `tools/analyze_two_models.py`.

All runs warm-start from the 4-day baseline and train the canonical-grid unsupervised intensity
pipeline (see `CLAUDE.md`). **Primary metric = motion PSNR** (the `val_motion` panel): PSNR over only
the dynamic voxels — the ~5–8% of the cube that moves across the cardiac cycle. Static tissue is
excluded, so it is the honest measure of whether the model corrects motion *where it matters*.
bbox/full PSNR are dominated by stationary anatomy and overstate everything. Identity (Δ=0, no motion
correction) motion baseline ≈ **20.5–20.7 dB** (t0 ≈ 19.65, t7 ≈ 21.20). The motion mask (τ=0.05 on
intensity swing across the 12 phases) covers **7.9%** of the bbox, tightly localized on the
myocardium/blood-pool.

## Part A — Four-run comparison (`_html/01`)

### Headline numbers

| Run | Target phases | Trained | Steps | ★ motion PSNR | Δ vs ~20.6 | bbox | full | SSIM |
|---|---|---|---|---|---|---|---|---|
| `t0t7_aggft` (`warrwlv8`) | {0,7} | aggregator+head | ~173k | **23.72** | **+3.22** | 32.66 | 34.03 | 0.975 |
| `allphases_aggft` (`fc8d065g`) | all 12 | aggregator+head | ~173k | **23.44** | **+2.74** | 32.63 | 33.95 | 0.976 |
| `t0t7_aggft_no_zt` (`vry47r4f`) | {0,7} | aggregator+head, **z/t OFF** | ~171k | 22.92 | +2.42 | 32.01 | 30.87 | 0.955 |
| `allphases_headonly` (`bnwfjav6`) | all 12 | **point_head only** | 200k | 21.29 | +0.59 | 30.64 | 30.07 | 0.957 |

The mean column is **not** comparable across the phase split (the {0,7} runs average only phases 0+7;
the all-phases runs average all 12). Use the apples-to-apples view below.

### Apples-to-apples: phases 0 (ED) and 7 (~ES) — the only phases all four runs are scored on

| Run | motion t0 | motion t7 | bbox t0 | bbox t7 |
|---|---|---|---|---|
| `t0t7_aggft` | 23.38 | 24.06 | 32.70 | 32.62 |
| `allphases_aggft` | 23.14 | 24.27 | 32.86 | 32.88 |
| `t0t7_aggft_no_zt` | 22.56 | 23.28 | 32.00 | 32.02 |
| `allphases_headonly` | 19.70 | 21.94 | 29.61 | 30.45 |
| identity baseline | 19.65 | 21.20 | 29.80 | 30.68 |

### Findings

- **Aggregator-finetune is the decisive lever.** At ED, head-only = 19.70 ≈ identity 19.65 —
  *statistically no motion correction at ED*, and only +0.7 at t7. The bbox view (29.6) hides this
  because static tissue carries it. Aggft = **+3.5 dB over head-only at ED** (23.1–23.4 vs 19.7).
- **Phase coverage is free.** `allphases_aggft` ≈ `t0t7_aggft` on motion (23.1/24.3 vs 23.4/24.1) —
  tied within noise. One 12-phase model is as good per-phase as a 2-phase specialist.
- **z/t embeddings help the dynamic region ~0.8 dB** (22.56/23.28 vs 23.1–23.4) — more than the ~0.5 dB
  it looked on bbox.
- **The `no_zt` damage lives in the padding.** bbox/motion barely move (~0.5–0.8 dB), but **full PSNR
  drops ~3 dB**: without z/t conditioning the model bleeds content into the out-of-FOV (zero-padded)
  z-planes that should be empty. In-FOV content reconstructs almost as well as aggft.
- **⚠️ `no_zt` removes z AND input-t jointly** — it measures their *joint* effect and cannot attribute
  anything to z alone. (An earlier version of the report wrongly blamed the z-embedder specifically;
  not supported.)
- **Qualitative ceiling:** even aggft predictions are visibly *smoother than GT* (fine wall/texture
  lost) — the ~23 dB motion PSNR is "partial correction," not solved. This is the blur that motivates
  doc 10 / the refiner (doc 11).

## Part B — Two-model per-phase characterization (`_html/02`)

Independent characterization (not a head-to-head), fresh inference on the val set, target phase swept
**with the input slice set held fixed** (verified identical across `t_target` 0–11) — so each
per-phase montage is a clean "same inputs, different requested phase" demonstration. The model is never
given a slice *at* most target phases; it must synthesize the queried phase from slices acquired at
other phases.

- **M1 = `allphases_aggft` (`fc8d065g`)** — z+t+target_t on, trained on all 12 phases, evaluated at
  every phase 0–11.
- **M2 = `no_zt` (`vry47r4f`)** — z and input-t off (target_t on), trained on {0,7}, evaluated at t=0,7.

Summary metrics (this inference, 8 val subjects, mean [min–max]):

| Model | phases | z/t emb | motion PSNR | bbox PSNR | full PSNR |
|---|---|---|---|---|---|
| M1 allphases | 0–11 | on | 23.45 [19.8–26.3] | 31.96 [29.0–35.3] | 33.49 [30.0–37.5] |
| M2 no_zt | 0, 7 | OFF | 22.75 [18.0–24.9] | 31.09 [27.2–34.1] | 30.19 [27.0–34.0] |

(Slightly different sampling/averaging than the wandb panels → numbers differ by tenths of a dB but
agree on the picture. SSIM omitted — GPU SSIM kernel unavailable in that run.)

### Findings

- **M1 — a clear mid-systole dip.** Motion PSNR is highest at ED (t0/t1 ≈ 24.3 dB) and dips to its
  minimum at peak contraction (**t4 ≈ 22.6 dB**), a **~1.7 dB swing**, recovering toward end-systole
  (t7/t8). bbox and full track the same U-shape. "Works on all phases" is true, but **quality is
  phase-dependent**: best at rest, weakest at peak systole (where displacement from the cycle-average
  shape is largest). Error concentrates on the moving myocardium/blood-pool and grows through systole.
- **M2 — works on its trained phases with the no_zt signature.** Flat ~23.0 (t0) / 22.5 (t7) motion,
  good in-FOV anatomy (bbox ≈ 31.1), but **full (30.2) sits below bbox (31.1)** — the out-of-FOV padding
  bleed. A working 2-phase reconstructor that leaves dirt in the empty planes.
- **Why M2 matters for the real-time free-breathing goal (info contract, see `docs/04`).** In a true
  real-time free-breathing acquisition we would *not know each input slice's cardiac phase t* (no ECG
  gating) nor its respiratory phase r. The `no_zt` ablation removes exactly that per-slice input-phase
  conditioning (keeping only the target query, which is always available — we choose which phase to
  render). That it still reconstructs at **~22.8 dB motion** is encouraging evidence the approach can
  tolerate not knowing input-slice phase — the network appears to infer enough phase/geometry from the
  images themselves via attention. **Caveats:** (i) `no_zt` also dropped the depth embedder z, which we
  *do* have at deployment (slice z is known), so a realistic "unknown-t" model would keep z and likely
  do better; (ii) trained on clean gated cine → does not yet test the real domain shift (bSSFP
  transient, single-shot artifacts, respiratory motion); (iii) the padding-bleed wrinkle remains. So: a
  promising signal that the information contract is viable, **not** proof of real-time transfer.

## Relationship to other docs

- The motion-metric-primary framing here is the basis for the headline metric used in `docs/05`
  (respiratory variants), `docs/07` (predicted DVF), and `docs/08` (OOD paradox).
- The "predictions are smoother than GT" ceiling observed here is decomposed quantitatively in
  **`docs/10`** (the blur is ~75% the splat renderer) and attacked in **`docs/11`** (the UNet refiner).
- The `no_zt` info-contract signal feeds the design stance in **`docs/04`**.
