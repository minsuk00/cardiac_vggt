# 13 — VGGT+refiner: limitations decomposition & proven improvement directions

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Question:** what are the *inherent* limits of the VGGT+refiner slice-to-volume model (fixed input
> contract: ~10 scattered single-frame-per-slice slices, z-only), and which improvement directions are
> actually worth funding? Tested end-to-end on the **current best model** (joint refiner, ckpt epoch ~76)
> with a ladder of **training-free oracle reconstructions** + a **4-expert debate panel**. MOTION PSNR
> (dynamic heart voxels) is the headline metric throughout (n=30, breathing-val).
>
> **The limits split into two independent levers + one transfer risk:**
> 1. **Accuracy (motion PSNR) is GEOMETRY-limited, not renderer-limited.** Perfect motion/placement with
>    the *same splat* reaches **34.98 dB** motion; the model is at **20.62** (floor 16.66) → a **~14 dB gap
>    in motion estimation**. The refiner/SSIM work polished the wrong lever. *(Honest caveat: the oracle
>    feeds true target-phase planes, so part of the 14 dB is the warp-only appearance wall + the r/t-blind
>    information floor and is unreachable; the achievable ceiling is between ~21 and ~35 dB. Direction is
>    unambiguous though — the model is well below any conservative ceiling, matching doc 07's under-correction.)*
> 2. **Sharpness IS renderer-limited, and the culprit is the 256→518→256 RESIZE, not the trilinear kernel.**
>    Perfect-placement splat = **0.768× GT** sharpness; the same splat at **native 256 (no resize) = 0.997×**.
>    Nearest≈native-trilinear (kernel innocent). **No loss can break this — it's a concrete renderer bug.**
> 3. **The true binding constraint for the stated goal is the DOMAIN GAP + no real-data metric.** Feeding
>    true single-shot R=8-aliased input drops the model **−3.03 dB** motion (20.62→17.59, near the floor):
>    brittle to artifacts it never trained on. And every number is on the model's *own simulator*.
>
> **Reassuring:** the model did **not** memorize the breathing simulator — under an OOD respiratory waveform
> its gain over identity *survives and grows* (+3.96→+4.88 dB).
>
> **Proven improvements (ranked):** (A) **native-resolution splat** — sharpness 0.648→0.721 with the model's
> own geometry, ceiling 0.997 (training-free evidence); (B) **multi-draw test-time ensemble** — **+1.76 dB**
> motion at K=8, free (but blurs); (C) **degraded-input augmentation** — proven necessary (the 3 dB cliff)
> and precedented (respiratory-sim generalized). **Ranked DOWN by evidence:** fancier/perceptual losses
> (sharpness is renderer-not-loss-limited), data-consistency loss (already ~34 dB satisfied), unfreezing the
> backbone (grad-norm ~0.014), k-space input (incompatible with frozen RGB DINOv2). **Biggest gap = a
> real-data, motion-region metric.**

**Date:** 2026-06-20
**Status:** Analysis complete (training-free; no model change). Full report with figures:
[`_html/14_limitations_and_improvements.html`](../_html/14_limitations_and_improvements.html).
**Artifacts:** `tools/limits_decomposition.py` (ladder + OOD-breathing + data-consistency),
`tools/improvements_test.py` (native-splat + multi-draw), `tools/kspace_singleshot_toy.py` (domain gap),
`tools/qual_panel.py`, `tools/make_limits_figures.py`, `_html/build_limits_report.py`,
`result/limits_eval/*.json`.

---

## 1. Method

Pipeline transports input pixel intensities (`world_points = scanner_coords + Δ` → trilinear splat →
`V_canon` → 3D-UNet refiner → `V_refined`). To localize the error, a **ladder of reconstructors** each
removes one error source, all scored identically vs the target-phase volume on the breathing-val protocol:
`identity` (Δ=0 floor) → `model V_canon/V_refined` → `oracle_perfect` (true target planes, Δ=0 = perfect
placement upper bound) → `oracle_native256 / nearest / super2x` (renderer-attribution variants). The oracle
uses `batch["phases"]` (real cached cine phase volumes — **NOT** the on-disk elastix/carmen DVFs, which are
unreliable, produced by another model). Diagnostics: data-consistency (re-slice recon at input geometry),
OOD-breathing (different respiratory params than training), single-shot k-space (R=8 aliasing).

## 2. Headline numbers (n=30, breathing-val, motion PSNR primary)

| reconstructor | MOTION | bbox | sharp/GT |
|---|---|---|---|
| identity (floor) | 16.66 | 23.32 | 0.633 |
| model V_canon | 19.65 | 27.13 | 0.648 |
| model V_refined (best) | **20.62** | 28.65 | 0.673 |
| oracle (perfect placement) | **34.98** | 41.84 | 0.768 |
| oracle native-256 (no resize) | 104.6 | 68.5 | **0.997** |
| oracle nearest (no tent) | 93.1 | 67.6 | 0.997 |
| oracle super-2× grid | 37.1 | 44.3 | 0.801 |

Data-consistency (recon re-sliced at input geometry): model V_refined **33.5 dB** → DC loss low-headroom.
Coverage ~70% (oracle same) → coverage is a full/bbox issue, not the motion bottleneck.
Single-shot R=8: motion 20.62→17.59 (**−3.03**). OOD-breathing gain: +3.96 (IND) → +4.88 (OOD).
Improvements: native-splat sharp 0.648→0.721 (model geom); multi-draw motion 20.62→22.38 (+1.76, K=8).

## 3. The debate panel

Four adversarial subagents (MRI physicist, neural-rendering architect, pragmatic experimentalist, red-team).
**Convergence:** all three builders independently nominated a *training-free decomposition* as the top
next step — exactly what this executes. **Adjudication by experiment:** red-team right that motion>bbox is
honest (gap survived, 14 dB); architect's resize hypothesis confirmed (0.768→0.997); physicist's DC loss
real but low-headroom (already 34 dB); pragmatist's "fancy losses are a trap" corroborated. The architect's
**feature-splat + tiny 3D decoder** (to break the warp-only appearance wall) and a **degraded-input
fine-tune** are the two scoped-but-unrun training experiments (shared-cluster time).

## 4. Recommended sequence

1. **Build a real-data motion-region metric** (held-out multi-frame OCMR reference, score `psnr_motion` at
   matched phases vs nearest-frame / conditional-mean baselines) — without it every dB is suspect.
2. **Degraded-input augmentation** (single-shot k-space undersampling + transient/SNR) — the transfer lever.
3. **Attack motion estimation** (the ~14 dB lever): low-rank/basis-DVF motion subspace on the point head
   to regularize the under-constrained through-plane Δ; and/or content-inferred cardiac-phase conditioning.
4. **Native-resolution splat** (sharpness win, needs head re-fit). 5. **Multi-draw ensemble** (free).
