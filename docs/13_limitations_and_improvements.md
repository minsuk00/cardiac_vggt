# 13 — VGGT+refiner: limitations decomposition & improvement roadmap

> **⚠️ CORRECTION (2026-06-22, see [[19_motion_correction_warp_ceiling]]):** the claim below that
> "the ~14 dB motion gap is GEOMETRY/MOTION-ESTIMATION-limited" is **WRONG**. A `/toy-experiment`
> partition (directly optimizing Δ on the real task = the warp-only ceiling) showed a
> perfectly-optimized warp tops out at **~21 dB ≈ the trained model (20.6)**, ~14 dB below the
> oracle. The oracle (35) is **unreachable by any warp** — that gap is the **appearance wall**
> (warping other-phase intensities can't synthesize target-phase appearance), NOT motion estimation
> (which has ~0.5 dB headroom). The right fix is **appearance synthesis (learned renderer/decoder)**,
> not better motion estimation. Read doc 19 for the corrected conclusion; the renderer/coverage and
> sharpness findings below still stand.

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Question:** what are the *inherent* limits of the VGGT+refiner slice-to-volume model (fixed input
> contract: ~10 scattered single-frame-per-slice slices, z-only), and which improvement directions are
> actually worth funding? Tested end-to-end on the **current best model** (joint refiner, ckpt epoch ~76)
> with a ladder of **training-free oracle reconstructions** + a **4-expert debate panel**. MOTION PSNR
> (dynamic heart voxels) is the headline metric throughout (n=30, breathing-val).
>
> **Conclusion: the dominant lever is MOTION ESTIMATION.** Everything decomposes into two model-side levers
> (plus a transfer risk that is *out of scope* per the project owner — we assume clean input images for now):
> 1. **Accuracy (motion PSNR) is GEOMETRY/MOTION-limited, NOT renderer-limited — this is the headline.**
>    Perfect motion/placement with the *same splat* reaches **34.98 dB** motion; the model is at **20.62**
>    (floor 16.66) → a **~14 dB gap that lives in motion estimation.** This is the lever to push for better
>    results. *(Honest caveat: the oracle feeds true target-phase planes, so part of the 14 dB is the
>    warp-only appearance wall + the r/t-blind information floor and is unreachable; achievable ceiling is
>    between ~21 and ~35 dB. Direction is unambiguous — the model under-corrects, doc 07 slope 0.42.)*
> 2. **Sharpness IS renderer-limited, and the culprit is the 256→518→256 RESIZE, not the trilinear kernel.**
>    Perfect-placement splat = **0.768× GT**; the same splat at **native 256 (no resize) = 0.997×**.
>    Nearest≈native-trilinear (kernel innocent). Fix = render the splat at native 256 ("518 in, 256 out",
>    §4). **Sharpness-only and PSNR-neutral at today's geometry** — it buys crispness, not the 14 dB.
> 3. *(Out of scope per owner)* The gated→real-time **domain gap** is real (single-shot R=8 input drops the
>    model −3.03 dB to near the floor) but we **assume good input images for now**, so it is deferred.
>
> **Improvement roadmap (ranked, given "good images" assumption):** the prize is **motion estimation** —
> (1) **motion-weighted loss** (the model is barely supervised on the 3–5% dynamic voxels), (2) **low-rank /
> smooth Δ** (B-spline/basis field — regularizes the ill-posed through-plane motion that drives the
> shrinkage), (3) **auxiliary cardiac-phase head** (content-infer phase within the contract; helps cardiac,
> not respiratory), (4) **keep/accelerate the aggregator fine-tune** (a *proven* lever — see correction
> below), (5) **oversample large-motion cases** (directly attacks the under-correction). Plus the cheap
> **native-resolution splat** for sharpness (§4). **Demoted:** multi-draw ensembling (needs extra frames →
> against the fast-acquisition goal; diagnostic only); fancier losses (sharpness is renderer- not
> loss-limited); data-consistency loss (already ~34 dB satisfied); k-space input (frozen RGB backbone).

**Date:** 2026-06-20 (rev. 2 — incorporates the aggregator-unfreeze correction, the renderer-fix recipe,
the motion roadmap, and the "assume good images" scoping from the follow-up discussion).
**Status:** Analysis complete (training-free; no model change yet). Full report with figures:
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
Single-shot R=8: motion 20.62→17.59 (**−3.03**, *out of scope*). OOD-breathing gain: +3.96 (IND) → +4.88 (OOD).
Native-splat (model geom, no retrain): sharp 0.648→0.721. Multi-draw motion 20.62→22.38 (+1.76, K=8) — *demoted, see §6*.

## 3. The debate panel

Four adversarial subagents (MRI physicist, neural-rendering architect, pragmatic experimentalist, red-team).
**Convergence:** all three builders independently nominated a *training-free decomposition* as the top
next step — exactly what this executes. **Adjudication by experiment:** red-team right that motion>bbox is
honest (gap survived, 14 dB); architect's resize hypothesis confirmed (0.768→0.997); physicist's DC loss
real but low-headroom (already 34 dB); pragmatist's "fancy losses are a trap" corroborated. The architect's
**feature-splat + tiny 3D decoder** (to break the warp-only appearance wall) and a **low-rank motion
subspace** are scoped-but-unrun training experiments.

## 4. The renderer fix (sharpness): "518 in, 256 out"

**Why 518 exists:** DINOv2 (frozen backbone) has patch_size=14 and was pretrained on 518² images (518/14=37
patches), so its *input* must be 518. That is the only reason 518 is in the pipeline — it is not the real
data resolution (the slice is 256). The bug: that 518 leaks downstream — the DPT head outputs Δ at 518 and
the splat renders from the **518-upsampled intensity**. Scattering ~518² points into the 256² grid puts ~4
points per voxel; trilinear-averaging them is a low-pass blur (256→518→[avg back to 256]). The splat is a
*coordinate scatter*, not an image resize — point count ⟂ grid size — so native-256 points go in 1-per-voxel
and stay sharp (0.997) while 518² points blur (0.768). Confirmed the *kernel* is innocent: nearest = native
trilinear = 0.997.

**The fix:** keep DINOv2's 518 input, but render the splat at native 256.
- **Option 1 (minimal, no architecture change):** leave the head at 518; in `splat_predictions` downsample
  `world_points` 518→256 and use the **native-256 image** intensity. Splat 256² points.
- **Option 2 (clean):** change the DPT head's final `custom_interpolate` target (dpt_head.py ~229–231) from
  518 to **256** so it outputs Δ@256 directly; build `scanner_coords` at 256; splat native-256 intensity.

**Option 1 ≈ Option 2 (measured):** Δ is spatially smooth — it round-trips 518→256→518 with only **0.61%**
error — so downsampling Δ is ~lossless and the two options are equivalent (<1%); Option 2 may be a hair
better after re-fit (no train/render mismatch), never worse. Both need a **head re-fit** (the head learned Δ
against the 518 splat). Companion: the output cube is already 256×256×12, so the refiner shape is unchanged.

**Honest scope:** this is **sharpness-only and PSNR-neutral at today's geometry** — proof: native-splat with
the model's own (imperfect) Δ gave sharp 0.648→0.721 but motion PSNR 19.65→19.49 (flat). PSNR payoff is
*gated by geometry* — a crisp edge in the wrong place doesn't lower error. The ceiling is 0.997 but the model
won't reach it until placement improves; realistic post-refit ~0.8–0.9.

## 5. The motion-estimation roadmap (the ~14 dB lever — the real prize)

Ranked by impact × feasibility. The recoverable part of the 14 dB is ill-posedness + weak supervision +
under-correction (the r-blind part is unreachable by contract).

1. **Motion-weighted loss (cheapest, do first).** Loss is full-volume L1, but the dynamic heart is ~3–5% of
   voxels, so its gradient is drowned by static tissue. Upweight the loss inside `compute_motion_mask`
   (e.g. 5–10×). *Test:* short fine-tune vs current → motion PSNR. *Risk:* slightly worse static PSNR (the
   inflated metric anyway).
2. **Low-rank / smooth Δ (biggest structural fix).** Replace free per-pixel 3-ch Δ with a coarse control
   grid (B-spline/FFD) or learned smooth basis-deformations × per-slice coeffs. Free per-pixel 3D Δ is
   wildly under-determined — *especially through-plane (z)* — so it regresses to the mean → the doc-07
   under-correction (slope 0.42). A low-rank smooth field bakes in "tissue moves coherently," cutting DOF
   and shrinkage. (STINR-MR / subspace-INR, doc 02.) *Test:* basis-Δ vs free-Δ overfit on a few subjects.
   *Risk:* too few bases → can't represent sharp local motion (DOF sweep).
3. **Auxiliary cardiac-phase head (better conditioning, within contract).** Add a small head predicting each
   input slice's *cardiac* phase as an **auxiliary** loss (GT t is a free train-time label). At inference t
   is NOT fed — the point is to force the features to encode phase. Different from *feeding* t (doc 05 found
   that unhelpful). *Test:* linear-probe phase from frozen features first; if recoverable, wire the aux loss.
   *Caveat:* the deep-breath headroom is mostly **respiratory** (r), which is NOT content-inferable from a
   cropped SAX slice → helps cardiac/ES quality, bounded upside.
4. **Keep/accelerate the aggregator fine-tune (proven lever — see §6).** Unfreezing pays off with long
   training. Push it: more epochs, or faster feature adaptation (higher LR on the aggregator / LoRA-adapters
   / unfreeze only the last N blocks) so the payoff arrives sooner. Low risk, already working.
5. **Oversample large-motion cases (cheap).** The model specifically shrinks on *large* displacements
   (deep breaths, ES). Oversample large-amplitude examples or curriculum small→large so it gets more gradient
   on the hard cases. Sampling change only.

## 6. Corrections / updates to earlier conclusions

- **CORRECTION — unfreezing the aggregator is a REAL lever (earlier claim was wrong).** Rev. 1 said
  "unfreeze backbone = low ROI (grad-norm 0.014, ~no movement)." That was based on the report-10/11
  *snapshots* at joint epoch ~22 (unconverged, behind frozen) plus the instantaneous grad-norm. With more
  training the joint (unfrozen) model **overtook** the frozen-refiner-only model: joint V_refined motion went
  19.64 (ep22) → **20.62 (ep76)**, above frozen's ~20.07. The grad-norm 0.014 is *slow-but-real* adaptation
  (the right way to fine-tune a big pretrained backbone), not "stuck." So aggregator fine-tuning is moved
  from "ranked down" to roadmap item #4, and the rev-1 claim that "the geometry lever is only about how Δ is
  predicted, not the backbone" is withdrawn — **better features help too.**
- **DEMOTION — multi-draw ensembling is a diagnostic, not a deployable win.** The measured +1.76 dB used K
  genuinely-different *frame draws* per slice ⇒ ~K× more acquired frames, which contradicts the
  one-frame-per-slice / fast-acquisition purpose. Its value is the negative result it gives (the gap shrinks
  only modestly with more frames ⇒ residual error is genuine motion estimation, not variance). A
  deployment-legal TTA variant (same frames, permutations) exists but gives much less.
- **SCOPING — domain gap deferred.** Per the project owner, assume clean input images for now, so the
  single-shot −3.03 dB finding (§2) and degraded-input augmentation are deferred (kept on record for when
  real-time transfer is back in scope).

## 7. Recommended sequence (given "good images" assumption)

1. **Motion-weighted loss** (§5.1) — cheapest attack on the 14 dB lever.
2. **Low-rank / smooth Δ** (§5.2) — the biggest structural fix for the ill-posed through-plane motion.
3. **Keep/accelerate the aggregator fine-tune** (§5.4) — proven; let it run / speed it up.
4. **Auxiliary cardiac-phase head** (§5.3) — probe-then-commit; bounded by r-blindness.
5. **Native-resolution splat** (§4) — cheap sharpness win; needs a head re-fit; PSNR-neutral.
6. *(deferred)* real-data motion metric + degraded-input augmentation — when real-time transfer is in scope.
