# 21 — Motion PSNR is acquisition-coverage-limited: the ONLY free lever is proximity sampling (+0.8 dB)

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Toy-experiment** (`/toy-experiment`) testing doc-20's *prescription* — "relax the input contract" — to find
> what actually raises MOTION PSNR beyond the ~21 dB warp wall, and by how much. Training-free probes (rebuild
> input compositions from the cached phase bundle; no retraining), 24 + 8 held-out subjects, clean protocol,
> S=8, paired, prove-it-clean on all 3 load-bearing scripts.
>
> **CONCLUSION (high confidence): there is NO large lever inside the fast-acquisition contract. Motion PSNR is
> bound by how much of the TARGET PHASE you DIRECTLY observe, and the model does NOT propagate that appearance
> to planes it didn't measure.**
> 1. **Lever A (count at target).** Adding K input slices *at* the target phase raises *full-volume* motion PSNR
>    18.2→20.1 — but that is an **OBSERVATION LEAK** (you score the planes where you injected a true target
>    slice). The leak-free **held-out-z** motion PSNR (dynamic voxels only on planes given NO target slice)
>    *FALLS* 18.2→13.6. ⇒ the model fills only what you directly measure; it cannot synthesize the target phase
>    on unobserved planes. (Held-out-z metric is genuinely leak-free: identity-splat z-bleed ~1e-16, prove-it.)
> 2. **Lever B (proximity, leak-free, fixed budget).** Companions near the target phase (t±1) beat random by
>    **+0.8 dB** (18.2→19.0). The **warp CEILING** gain under the same compositions is **+0.84 dB** (19.1→19.9)
>    ⇒ the current model is **already at its proximity ceiling — a retrain buys ~nothing.** Real, cheap,
>    goal-compatible (no extra frames, just *which* phases you sample), but ~+0.8 dB, and still below 21.
> 3. Everything else lands within ~1 dB of the wall: renderer/decoder +0.03 (doc 20), splat-res −0.13 (doc 13),
>    input-t ~0 (doc 05). The big dB (28.1 dense-temporal-interp, 35 oracle) needs information the sparse,
>    phase-blind, ~1-frame-per-slice contract does not contain.
>
> **⇒ What to do for better motion PSNR:** (1) **proximity sampling** is the only free win (~+0.8 dB) — bias the
> scattered acquisition/query toward the target phase band, no retrain. (2) For a LARGE gain you must **break
> the contract** — directly observe more of the target phase (more frames/slice near target, a temporal stream,
> or a 5-D motion-resolved reference) — a deliberate **speed↔quality trade**, and even then it helps only the
> planes you measure. (3) Stop tuning the model/renderer for motion PSNR.
>
> **The ~21 dB wall is an ACQUISITION-COVERAGE wall, not a model wall.** This **completes doc 20** (which named
> "relax the contract" but never measured it): the contract lever is real but the cheap part is small and the
> big part costs acquisition speed. Diagnosis of docs 19/20 stands.
>
> **Report:** `_html/21_motion_psnr_contract_levers.html` ·
> https://github.com/minsuk00/research-reports/blob/main/2026-06-22_1258_motion-psnr-contract-wall.html

**Date:** 2026-06-22
**Status:** Conclusive (training-free, held-out, prove-it-clean ×3, two cross-validating experiments).
Bounds: in-distribution simulator, clean (no-respiratory) protocol, S=8; a deliberately
propagation-trained model is unverified (ceiling-bounded ⇒ expected small).
**Related:** [[20_appearance_synthesis_test]] (prescription this tests), [[19_motion_correction_warp_ceiling]]
(warp ceiling), [[13_limitations_and_improvements]] (splat-res/native-256 dead lever),
[[04_inference_information_contract]] (the contract that caps recoverability),
[[05_respiratory_variants_eval]] (input-t neutral).

---

## 1. Method

Question: doc 20 concluded "relax the input contract" but never measured *which* relaxation raises motion PSNR.
The input is decoupled (target phase is a *query*; input slices are random-phase/z) → probe the contract
**without retraining** by rebuilding compositions from each subject's cached `phases` bundle and reading motion
PSNR. All clean protocol, fixed budget S=8, fixed per-subject z-set, paired.

- **Lever A — count at target** (`tools/toy_contract_levers.py`): K of 8 slices at the target phase (distinct
  z), rest at fixed random other phases; sweep K=0,1,2,4,8. Per K, four readouts:
  - C1 identity-Δ on only the K target slices (pure observation, model-free),
  - C2 identity-Δ on the full 8 (splat-only floor),
  - C3 model on the full 8 (realizable),
  - **C4 model, motion PSNR on HELD-OUT z-planes only** (planes given no target slice) = leak-free propagation.
  (The red-team flagged that C1/C2/C3 score planes where target slices were injected → oracle leak; C4 fixes it.)
- **Lever B — proximity** (same script): K=0 (no exact target frame), identical z-set, companions = random vs
  t±2 vs t±1. Leak-free by construction; isolates *which phases* vs *how many frames*.
- **Warp ceiling by composition** (`tools/toy_proximity_ceiling.py`): direct-opt a free per-pixel Δ (sane lr,
  clamp, TV — the corrected warp-ceiling loop from doc 19) on random vs near±1 → tells whether proximity's small
  model gain is a *floor* (higher ceiling ⇒ retrain pays) or *capped* (+0.8 is the whole prize).
- Debate: 4 agents (H1 more/target frames; H3 proximity; H2 input-t; red-team). prove-it: 3 reviewers.

## 2. Results (clean val, motion PSNR, dynamic voxels, n=24 / ceiling n=8)

**Lever A (count at target):**

| K | C1 obs-only | C2 id-full | C3 model (full, LEAKY) | C4 held-out-z (leak-free) |
|---|---|---|---|---|
| 0 | – | 15.90 | 18.18 | 18.18 |
| 1 | 10.72 | 16.10 | 18.42 | 17.98 |
| 2 | 11.29 | 16.30 | 18.66 | 17.76 |
| 4 | 12.65 | 16.74 | 19.10 | 17.10 |
| 8 | 17.91 | 17.91 | 20.15 | **13.55** |

Full rises +2 dB; held-out-z **falls** → the gain is leak; no propagation to unobserved planes.

**Lever B (proximity, leak-free, K=0):**

| companions | mean \|Δt\| | C2 identity | C3 model |
|---|---|---|---|
| random | 3.17 | 15.92 | 18.24 |
| near ±2 | 1.53 | 16.47 | 18.62 |
| near ±1 | 1.00 | 16.88 | **19.04** |

**Warp ceiling by composition (n=8):** random identity 16.14 / ceiling 19.08; near±1 identity 17.09 / ceiling
**19.92**. Ceiling gain = **+0.84 dB** ≈ the model's realized +0.80 ⇒ model already at the proximity ceiling.

## 3. Debate vs results

H1 ("more/target-phase frames", predicted steep rise to ~35) — **refuted**: the rise is leak; held-out-z falls.
H3 ("proximity", predicted ≥+5 dB) — **partially confirmed but small**: +0.8 dB, not +5 (the 28.1 temporal-interp
anchor used the subject's *dense* t±1 at *every* plane; 8 scattered near-phase slices realize far less). H2
(input-t) — not retested; prior null stands (doc 05), and proximity needs no phase labels anyway. Red-team's
held-out-z + ceiling controls were decisive (turned an OOD-confounded curve into a clean verdict).

## 4. prove-it

3 reviewers, all clean. (a) `toy_contract_levers.py`: held-out-z mask correct & leak-free (splat z-bleed
~1e-16), composition control fixed-z, mpsnr scale [0,1] correct, Lever B fair. (b) custom `build_batch`
faithful to the real pipeline to **6e-8** (×255→clamp→÷255 round-trip is identity since phases pre-clamped to
[0,1]; phase/z indexing, scanner_coords, normalization all match formula-for-formula; t_indices inert under
use_t=False). (c) `toy_proximity_ceiling.py`: +0.84 robust — lr=0.005 converges, 1500 steps past plateau
(2.7× more = +0.016 dB), RNG symmetric, 256-vs-518 splat applied equally to both arms.

## 5. Recommendation

1. **Proximity sampling** — the only free lever (~+0.8 dB, no retrain). Bias the scattered acquisition/query
   toward the target cardiac-phase band (±1 phase).
2. **For a large gain, break the contract** — directly observe more target-phase appearance (more frames/slice
   near target, temporal stream, or 5-D reference). Speed↔quality trade; helps only measured planes.
3. **Stop tuning model/renderer for motion PSNR** — renderer/decoder/splat-res/input-t all ≤0.1 dB.
