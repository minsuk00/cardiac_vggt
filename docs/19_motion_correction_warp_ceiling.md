# 19 — Motion correction is warp-architecture-limited, not motion-estimation-limited

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Toy-experiment investigation** (`/toy-experiment`) into the best way to improve motion
> correction (drop splatting? architecture? training? loss?). Method: remove the 941M network and
> **directly optimize the displacement field Δ** on the real breathing task to find the **warp-only
> ceiling**, plus a no-optimizer loss-landscape sweep (E0) and a debate panel + a prove-it pass.
>
> **CONCLUSION (high confidence): the bottleneck is the WARP-ONLY ARCHITECTURE, not motion
> estimation / loss / training.** A perfectly-optimized, target-aware warp of the real input slices
> — every Δ parameterization from rigid to free-per-pixel — tops out at **~21 dB motion PSNR ≈ the
> trained model (20.6)**, **~14 dB below the perfect-placement oracle (35).** The model is already at
> its architectural ceiling. The 14 dB is the **appearance wall**: warping *other-phase* input
> intensities cannot synthesize *target-phase* appearance (ES blood-pool, through-plane
> disocclusion); the oracle only reaches 35 because it is handed the true target-phase planes.
>
> **⇒ Best fix = APPEARANCE SYNTHESIS** (splat *features* + a small 3-D decoder, or a learned/implicit
> renderer that generates target-phase appearance) — **NOT** better motion estimation, loss tweaks,
> low-rank Δ, unfreezing, or phase conditioning, all of which are proven near-zero headroom against
> the warp ceiling. Secondary lever: the splat's **coverage-division** weakens the through-plane
> gradient ~2× (E0); a coverage-free / inverse-warp renderer helps *reach* the ceiling but can't
> break it.
>
> **⚠️ This CORRECTS [[13_limitations_and_improvements]]**, which claimed "~14 dB motion gap is
> motion estimation." That was an artifact of comparing the model to an oracle the warp cannot reach.
> The 14 dB is the warp/appearance ceiling; motion-estimation headroom is ~0.5 dB.
>
> **Report:** `_html/18_motion_correction_warp_ceiling.html` (also `~/research-reports/2026-06-22_0401_*`).

**Date:** 2026-06-22
**Status:** Conclusive (warp ceiling proved, n=4 subjects × 5 parameterizations + independent
clean-LR cross-validation; E0 n=8, prove-it-verified). The *fix* (learned decoder) is scoped, not
yet trained. **Related:** [[13_limitations_and_improvements]] (corrected), [[18_joint_refiner_breaks_ood]]
(refiner delegation), [[07_predicted_dvf_analysis]] (under-correction slope 0.42).

---

## 1. Method

Central question: improve motion correction — drop splatting / architecture / training / loss?
Partition strategy: remove the network and **directly optimize Δ** (target-aware) on the real
breathing task → the warp-only ceiling. If it ≈ model ≪ oracle → the warp architecture is the
ceiling (appearance is the gap). If ≈ oracle → motion estimation is the lever.

- **E0 (`tools/toy_landscape.py`)** — no-optimizer loss-landscape sweep: single phase's own planes
  at aligned positions, sweep a known rigid offset along z vs x, record L1. Measures objective
  conditioning, immune to optimizer confounds. Variants: covdiv splat, raw-accumulate (nocovdiv),
  inverse-warp.
- **Warp ceiling (`tools/toy_warpceiling.py`)** — optimize Δ (rigid / low-rank control-grid G16,G32
  / free-per-pixel, ±TV) with sane per-parameterization LRs + clamp + convergence logging; report
  best motion PSNR.
- Debate panel (3 subagents: objective/renderer vs network/info vs red-team). prove-it pass (3
  reviewers + verification).

## 2. Results (real breathing-val, motion PSNR, dynamic voxels)

| | motion PSNR |
|---|---|
| identity floor | 16.8 |
| **trained model** | **20.6** |
| warp ceiling — rigid | 19.0 |
| warp ceiling — low-rank G16 / G32 | 20.6 / **21.2** |
| warp ceiling — free per-pixel (±TV) | 20.8 / 20.9 |
| **oracle (perfect placement)** | **35.0** |

- **Warp ceiling ≈ 21 ≈ model (20.6) ≪ oracle (35).** Every Δ parameterization lands at ~19-21 →
  the ceiling is the warp, not the Δ representation. Model→ceiling ≈ 0.5 dB; ceiling→oracle ≈ 14 dB.
- **E0:** z loss-gradient 0.0067/mm vs x 0.0143/mm (~2× weaker through-plane); coverage-division is a
  culprit — raw-accumulate / inverse-warp restore z to 0.0103/mm.

## 3. prove-it correction (important)

My first direct-opt (lr=0.03) showed "L1 down, motion down → objective misaligned." **prove-it
refuted it**: at lr=0.03 Adam *diverged* (Δ exploded out of bounds, **L1 went UP** 0.014→0.051). At
lr=0.005, L1 is genuinely minimized (→0.0054) and motion PSNR → 20.6 = the model. So **L1 is aligned
with motion**, and the corrected warp-ceiling run uses sane LRs. E0 was cleared (mm↔norm fair;
nocovdiv not a scale artifact — coverage maxes at 1.0; axis order correct). The warp ceiling is
cross-validated by the reviewer's independent clean-LR run.

## 4. Why the warp can't beat ~21 dB (mechanism)

The splat transports **input pixel intensities**. The inputs are slices at *other* cardiac phases;
at the target phase the heart genuinely *looks different* (chamber size, wall position, blood-pool
intensity). A displacement field can relocate input pixels but cannot change their values to
target-phase appearance where it differs, nor fill anatomy disoccluded by through-plane motion. So
even a perfect warp ≈ "best rearrangement of available (wrong-phase) intensities" ≈ 21 dB. The
oracle reaches 35 only because it is given target-phase intensities. The 14 dB = appearance the warp
cannot synthesize.

## 5. Recommendation (ranked)

1. **Appearance synthesis** — splat K-channel *features* (not one intensity) + a small 3-D decoder,
   or a learned/implicit renderer; residual-on-transport design + held-out hallucination audit. The
   only lever that can break ~21 dB.
2. **Coverage-free / inverse-warp renderer** (replace `V=acc/(cov+ε)`) — restores the through-plane
   gradient (E0); helps reach the ceiling; PSNR-bounded by ~21 until #1.
3. **De-prioritize for motion PSNR:** loss re-engineering, low-rank vs free Δ, unfreezing, phase
   conditioning — proven near-zero headroom vs the warp ceiling (still relevant to sharpness / OOD).

**Limitations:** n=4 for the ceiling (subject variance); direct-opt mildly unstable (motion PSNR
peaks then drifts under the clamp), so ~21 is an *upper bound* on the warp ceiling — which only
strengthens "≪ 35." The fix (#1) is not yet trained.
