# 34 — Breathing through-plane (SI) correction: mechanism + implementation brief (HANDOFF)

> **TL;DR & takeaway** *(design brief — NOT yet implemented; hand this to an executing agent)*
>
> **Goal:** improve the model's correction of respiratory through-plane (SI) shift. Current: DVF
> slope 0.35 (applied SI vs predicted Δz), corr 0.52, ~54 % of deep breaths (≥12 mm) get |Δz|<2 mm
> (ignored), residual ~1.93 dB in a ~12 % deep-breath tail. This is a **secondary** metric — EF (the
> clinical headline) is recovered and unaffected by any of this.
>
> **The honest ceiling:** the realistic *contract-preserving* best is **slope ~0.5** (NOT 1.0). The
> weakness factors as: **coverage-division 1.55×** (renderer, *fixable*) × **12 mm slice-pitch 1.39×**
> (fundamental, *not* fixable in-contract) × a **blind-inference information floor** (corr 0.52). Only
> the 1.55× is a real lever; expect **+0.5–1 dB / slope 0.35→~0.45–0.5**, and it is **unproven** (my
> coverage-free direct-opt probe was degenerate — see §5).
>
> **Ranked levers:** (B) coverage-free **auxiliary placement loss** [best ROI, add-a-loss, no renderer
> rewrite] → (A) coverage-free **renderer retrain** [swaps the splat, riskier] → (C) oversample deep
> breaths [cheap add-on] → (D) explicit cross-slice-consistency term [promising because sim breaths are
> differential, but overlaps B] → (E) add information (LAX / navigator) [large gain, **breaks the
> 1-frame contract** — defer]. Everything is a **RETRAIN** (changes the learnable Δz gradient; an
> inference-time renderer swap on the trained model does nothing).
>
> **Before any retrain, run the two training-free go/no-go gates in §6.** If they fail, **accept** the
> limit — it is the price of the fast, gating-free, single-frame single-orientation acquisition.

**Date:** 2026-07-04. **Status:** design brief, not implemented. **Model context:** 4wokxzov
(reference-slot, DPT, 1-frame), see [[33_4wok_conclusive_analysis]].

---

## 1. Measured baseline (reproduce before changing anything)
| quantity | value | source |
|---|---|---|
| breathing DVF slope (applied SI → pred Δz, scattered slots) | **0.35** | `tools/exp_4wok_analysis.py` → `result/analysis_4wok/summary.json` (`A_breathing_dvf`) |
| corr(applied \|SI\|, pred Δz) | 0.52 | same |
| deep (≥12 mm): applied ~17.6 → predicted ~5.7 mm | under-corrected 3× | same |
| fraction of deep breaths with \|Δz\|<2 mm (ignored) | **0.54** | same (`frac_deep_ignored_dz_lt2mm`) |
| raw breathing cost (do-nothing) | 2.79 dB | `B_breathing_cost` |
| residual the model fails to remove (model_clean − model_breathed) | **1.93 dB** | same |
| direct-opt (no network, WITH V_gt) under covdiv | slope **0.44** | `result/covfree_diag/summary.json` |

Breathing sim: `training/data/respiratory.py` (rigid SI+AP per-slice shift, Lujan sin⁶, amp 16±8 mm,
`group_by_burst=true`). Applied in `training/data/gpu_aug.py`. Config `training/config/mri_volume.yaml`.

## 2. Why through-plane correction is weak — 3-factor decomposition (all measured)

**E0 loss-gradient landscape** (`tools/toy_landscape.py` → `result/limits_eval/toy_landscape.json`,
d(L1)/d(offset), no optimizer, no network):

| axis / renderer | rise per mm |
|---|---|
| in-plane x (any renderer) | 0.01434 |
| through-plane z, **coverage-free** (nocovdiv ≡ invwarp) | 0.01032 |
| through-plane z, **covdiv (current splat)** | 0.00668 |

- **Factor 1 — coverage-division: 1.55×** (0.01032 / 0.00668). `vggt/utils/splat.py:90`
  `V = acc/(cov+1e-6)`. In low-coverage (~1 slice/plane) regions the per-voxel normalization makes the
  rendered intensity ~insensitive to the exact coverage weight → damps ∂V/∂Δz. **Renderer-fixable.**
- **Factor 2 — 12 mm slice pitch: 1.39×** (0.01432 / 0.01032). A 1 mm z-shift is 1/12 of a plane, so it
  moves intensity a small fraction toward the next voxel → intrinsically shallow gradient. **Fundamental
  to SAX geometry; NOT fixable in-contract.** (coverage-free does NOT touch this.)
- Product = 2.15× total z-vs-in-plane weakness. Coverage ≈ 57 % of the log-gap, pitch ≈ 43 %.

**Is the loss a blocker or a drag?** A *drag*. The loss IS full-volume L1 (`training/loss.py:368`,
un-masked — the coverage-masked version at :367 was removed to stop ghost-blob over-prediction), so
holes ARE penalized. BUT: (a) the damping is only 1.55×, and (b) **direct-opt under covdiv still reaches
slope 0.44** — a sighted optimizer pushes through the hole-penalty. Full z-coverage sampling (every
in-bbox plane ≥once, `mri_dataset.get_data`) means a slice that moves is re-covered by others, so no
*permanent* hole. **Do NOT coverage-mask the loss** — it removes the pressure to cover anatomy →
degenerate (model leaves hard voxels uncovered for free). The fix must keep the full-volume loss and
remove the hole/normalization coupling a different way (§3).

## 3. Key clarifications (get these right)

- **Supervised (0.44) vs blind (0.35).** Direct-opt has V_gt and reaches 0.44 (the covdiv-renderer
  ceiling *given the answer*). The trained model is blind at inference (no V_gt) and reaches 0.35. So
  ~0.09 is "info the model isn't extracting that's recoverable given V_gt," and 0.44 itself is capped by
  the covdiv renderer. A coverage-free renderer should lift BOTH the 0.44 ceiling (×~1.55 gradient) and,
  via a cleaner training gradient, the blind 0.35.
- **Differential vs global breathing.** `respiratory.py:146` draws **one breath per z-plane** (`gather`
  over P planes) → breaths are **per-plane iid (differential)**; there is essentially **no coherent
  global component** in the sim. (The debate agent's "global breath unobservable" is a real-world
  caveat, but **does not apply to this sim** — the whole breath is differential.) **Consequence:** the
  breathing is, in principle, **recoverable from cross-slice coherence** (a mis-placed plane is
  incoherent with its neighbors + with V_gt). The blind ceiling (corr 0.52) is therefore an
  *inference/architecture* limit (the model under-exploits coherence), not a hard information wall.
- **Can it tell 12 vs 18 mm?** Partially (corr 0.52). The canonical cube is FIXED; a deeper breath fills
  the fixed plane with ~6 mm-deeper anatomy — a subtle content change, and the crop removed the SI
  landmarks. So single-slice content-inference is weak; the leverage is cross-slice coherence (above).

## 4. Levers (implementation-ready, ranked by ROI)

### (B) Coverage-free AUXILIARY placement loss ⭐ [best first move — add a term, no renderer rewrite]
**Idea:** add a loss that samples V_gt AT each predicted world position and matches the input intensity
— a *gather*-supervised placement term with NO coverage-division damping and strong z-sensitivity.
- **Impl:** in `compute_volume_intensity_loss` (`training/loss.py`), add
  `L_place = |sample_volume(V_gt, world_points_flat) − input_intensity_flat|` using
  `vggt/utils/splat.py:124 sample_volume` (grid_sample; differentiable in `world_points`). Weight it
  small (e.g. 0.1–0.3) as an **auxiliary alongside** the existing splat L1 (`loss_volume`) + the TV/
  diffusion smoothness. Wire via `MultitaskLoss` (`loss.py:108`) + a `mri_volume.yaml` weight.
- **Why it helps z:** moving a slice in z changes where V_gt is sampled → a clean, un-damped z-gradient
  the covdiv splat suppresses. Recovers the 1.55× without swapping the renderer.
- **DEGENERACY RISK (must guard):** the gather term alone is under-constrained (a pixel of intensity I is
  "happy" at any V_gt voxel = I) — this is exactly why my Step-1 invwarp collapsed. So it MUST be
  auxiliary (small weight) with the splat L1 as primary + the smoothness reg holding Δ coherent. Ablate
  the weight; watch for Δ blowing up / PSNR dropping.
- **Expected:** slope 0.35 → ~0.45–0.5 (unproven). **Validate:** re-run `exp_4wok_analysis.py` breathing
  slope + deep-breath bins; must rise on the DEEP bin, not just PSNR.

### (A) Coverage-free RENDERER retrain [bigger change, riskier]
Replace `V = acc/(cov+1e-6)` with a coverage-free renderer (fixed-normalization raw-accumulate, or a
proper inverse-warp gather) in the training forward. E0 says +1.55× z-gradient. **Risk:** (i) raw-
accumulate over-brightens overlaps (needs a fixed/consistent normalization, not per-voxel); (ii) a naive
gather is degenerate (Step-1). Prefer **(B)** first — it captures most of the benefit as an added loss
without destabilizing the renderer. Same expected gain (~+0.5–1 dB), same pitch+info ceiling.

### (C) Oversample deep breaths [cheap add-on]
The deep tail is ~12 % of planes and rare per epoch → gradient-starved. Bias the per-plane breath draw
in `respiratory.sample_displacements` toward larger amplitudes during training (or add a re-weighting).
**Expected:** +0.2–0.5 dB; helps the model *reach* the renderer/info ceilings on the tail, can't exceed
them. Watch shallow-breath calibration doesn't regress. Complementary to B/A, not standalone.

### (D) Explicit cross-slice-consistency term [promising but overlaps B]
Because sim breaths are differential (§3), coherence is genuinely informative. But the existing
splat-L1-vs-V_gt already supplies the coherence signal (V_gt is coherent); an extra term mainly helps if
it is **more z-sensitive** than the covdiv L1 — which is exactly what (B) is. So treat (D) as *subsumed
by (B)*, unless you design a self-supervised coherence loss usable when V_gt is dropped (the "fully
unsupervised" direction in CLAUDE.md) — separate, larger project.

### (E) Add information: LAX view / navigator / self-gating [large gain, BREAKS CONTRACT — defer]
A breathing SI shift is *in-plane* for a long-axis view → directly observable, no pitch penalty. Or a
1-D SI navigator / bellows measures the breath. **Large gain (could approach oracle),** but all break
the fast, gating-free, single-frame single-orientation contract that is the project's novelty (docs 04,
31, 33 §7). Simulate LAX by reslicing the cached 4-D `phases` bundle (same `grid_sample` primitive as
respiratory reslice). **Only pursue if a through-plane clinical endpoint (regional wall-motion / strain)
becomes the goal.**

## 5. What Step-1 actually showed (so the next agent doesn't repeat my mistake)
`tools/exp_covfree_diag.py` direct-optimized per-plane Δz on breathed inputs under covdiv vs a custom
gather ("invwarp"). **Result: covdiv converged to slope 0.44; the custom coverage-free renderer was
DEGENERATE (slope 0.001 — it flattened the L1 landscape so the optimizer sat at Δz≈0 and still got low
loss by borrowing neighbor content via gather interpolation).** So that script does NOT prove a
coverage-free advantage. The robust evidence is E0 (§2, 1.55×). **Lesson:** a coverage-free renderer/loss
is easy to make degenerate; always keep the splat-L1 as primary and validate on the DVF slope, not PSNR.

## 6. Recommended execution sequence (with training-free GO/NO-GO gates FIRST)

1. **Gate 1 (training-free):** fix the Step-1 coverage-free renderer so it is non-degenerate (keep splat-
   L1 as the objective; only swap how V_canon is normalized — fixed-K, not per-voxel cov), and re-run
   `exp_covfree_diag.py`. **GO if** coverage-free direct-opt slope > covdiv's 0.44 by a clear margin.
   **NO-GO** (accept) if it doesn't beat 0.44 → the covdiv renderer isn't the limiter, and (B)/(A) won't
   help beyond noise.
2. **Gate 2 (training-free):** confirm the pitch floor — repeat E0 (`toy_landscape.py`) at finer z (e.g.
   interpolate V_gt to 24 planes) to confirm the 1.39× is pitch, bounding the max achievable.
3. **If both GO:** implement **(B)** (auxiliary placement loss), finetune from 4wok (short, warm-start —
   it already has EF/in-plane; only needs to learn larger Δz), higher LR on the point head. Add **(C)**.
4. **Validate the retrain** on: breathing DVF slope + **deep-breath (≥12 mm) recovered Δz** (the tail is
   the target), residual dB (`exp_4wok_analysis.py` B-block), and **confirm EF/in-plane did NOT regress**
   (`exp_4wok_p95.py`, the EF pipeline). Report robustly; the gain is ~1 dB on a 12 % tail — small.

## 7. Files / functions inventory
- Loss: `training/loss.py` — `compute_volume_intensity_loss` (:320), `loss_volume` (:368), `MultitaskLoss` call (:108).
- Renderer: `vggt/utils/splat.py` — `splat_to_volume` (:7, the `/cov` at :90), `splat_predictions` (:94), **`sample_volume` (:124, the gather primitive for lever B)**.
- Breathing sim: `training/data/respiratory.py` (per-plane draw :146), applied in `training/data/gpu_aug.py`.
- Sampling / full-coverage: `training/data/datasets/mri_dataset.py` `get_data`.
- Config: `training/config/mri_volume.yaml` (+ `mri_volume_diffusion.yaml` for loss weights).
- Measurement: `tools/exp_4wok_analysis.py` (breathing slope + cost), `tools/measure_dvf_z_correction.py`, `tools/exp_covfree_diag.py` (Step-1), `tools/toy_landscape.py` (E0).
- Prior context: docs 04 (blind-r contract), 07 (breathing DVF), 19 (E0 origin), 28 (multi-frame), 31 (single-orientation underdetermined), 33 (this model's full analysis).
