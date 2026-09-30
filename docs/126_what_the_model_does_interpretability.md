# What does CardioStitch do? Causal interpretability of the final model (final518_diff1000)

> **TL;DR & takeaway** — Causal interventions on the final model's cross-slice (global)
> attention show it handles the two sub-problems at different depths.
> 1. **Cardiac state is read from the reference slice in essentially one layer.**
>    - In global-attention layer 3 (the 4th of 24), the other slices ("companions") read the
>      reference slice's heart-region tokens. The read is biased toward each query's own in-plane
>      location. Four of the 16 heads carry it (h2, h10, h12, h14).
>    - **Necessity:** blocking this read in that one layer leaves **2%** of cardiac-state transfer.
>      Every one of 180 subjects is below 10% (max 7.6%).
>    - **Sufficiency:** swapping only what companions read from the reference in that layer moves
>      the reconstruction **97.5%** of the way to another phase (per-subject minimum 0.92). The 4
>      heads alone give 93.8% on held-out subjects.
>    - **Specificity:** hiding the reference's heart tokens leaves 4%. Hiding its background leaves
>      102%.
>    - **Paper metric (nnU-Net, n=60):** companion-plane EF error goes from 8.8 to 48.7 pp; the naive
>      stack is 57.2.
> 2. **Breathing is aligned by comparing slices with each other in the middle layers.**
>    - It needs cross-slice attention in layers 6–17, and layers 12–17 are indispensable. Removing
>      12–17 gives slope 0.87 → 0.001 and EPE 0.96 → 4.92 mm, the naive stack's value; every
>      subject has |slope| < 0.1.
>    - Layers 18–23 are not needed, and neither is the reference (EPE unchanged).
>    - The correction uses many slices, not only neighbours.
>    - On simulated inputs it is relative (a shift common to all slices is invisible) and
>      one-sided, as the training prior d_SI = A·sin⁶ ≥ 0 predicts.
> 3. **The dissociation is asymmetric.** Blocking the layer-3 reference read leaves breathing
>    intact. Blocking layers 12–17 keeps cardiac transfer largely but not fully (85%; EF error 9.9 vs
>    8.8 pp, n.s.).
>
> Evidence: 180 test subjects (pooled_curated_v2 test split; E4 n=60, E1 scan n=40), paired
> Wilcoxon + bootstrap CIs, simulated gated-phase scatter + val-style breathing.
> **Real data (E8, §7):** on the paper's real free-breathing RT scans (AF patient + volunteer), the
> same layer-3 knockout removes 97% of the LV volume swing of the non-reference slices (91.8 → 2.4
> and 52.0 → 1.7 mL). Removing layers 12–17 keeps it (98–109%).
> **Not shown:** that the model "does not estimate phase". No phase probe was run.
> **Paper sentence** (§4): *"The model receives no phase label. It takes the cardiac state from the
> reference slice's heart region through four heads in one early global layer, and aligns breathing
> by comparing slices in the middle layers."*

## 1. Question and approach

The paper says (Sec. 3/4.4) that CardioStitch "jointly reasons over the stack" and "propagates the
cardiac state observed in the reference slice to the rest of the heart". This doc asks *how*.
The literature sweep is docs/125.

Following current practice, we treat attention maps as hypothesis generators only and base every
claim on **causal interventions**:
- knockouts via attention masks;
- sufficiency via activation patching;
- dissociations.

The only pathway between slices is global attention: frame attention is per-slice and the DPT head
is per-frame. So interventions on global attention are complete for "what crosses slices".

**Harness** (`tools/interp/harness.py`):
- Rebuilds the trainer's val input path (`get_data` → `extract_slices_with_respiratory_vec` →
  `_resize_to_model_res`) with controlled per-slot phase, breathing shift, z-token and slot order.
- Monkeypatches every `aggregator.global_blocks[i].attn`. This supports group- or token-level
  boolean masks, chunked attention statistics (no (S·1374)² materialisation), and slot-0 key/value
  activation patching by layer, head and token.
- Verified on GPU:
  - The hooked forward is bit-identical to the original.
  - An all-true mask gives the same output to within 2e-4 (normalized units).
  - Self-patching reproduces the unpatched output exactly (0.0, 360/360 runs).
  - Permuting the companions only permutes the output (3e-4).
  - SDPA uses True = attend.
  - On real RT data, the harness reproduces the published recon to within 5e-7 (§7).

**Protocol:**
- Per test subject: companions at random gated phases (seed 1000+idx) and val-style breathing
  (seed 2000+idx), the same draw for every experiment. The reference (slot 0, mid plane) is swept
  over the 12 phases.
- **Cardiac transfer G** = mean_t NCC(V_t, GT_t) − mean_{t≠t'} NCC(V_t, GT_t'), computed in the
  heart ROI on companion planes, on a volume splatted **from companion slots only**. Without this,
  the reference slice's own pixels leak into neighbouring planes through its predicted Δz; that
  leakage adds a 16.5% floor (§9).
  - G = 0 exactly for any output that does not depend on the reference (naive stack: 0).
  - Absolute G is small (full model 0.050). Its functional meaning is given by the LV curves (§6).
- **Breathing:** through-plane EPE (companions, per-subject demeaned, as in the paper) and the
  slope of predicted vs applied shift.

## 2. Cardiac state: reference read in global layer 3 (E1, E1b, E5, E7, E3)

**Necessity** (E1b, n=180, companion-only G; mean G as % of the full model; per-subject quantiles):

| intervention (companions → reference) | cardiac transfer | per-subject | breathing EPE (mm) |
|---|---|---|---|
| full model | 100% (G=0.050) | | 0.98 |
| block reference, **layer 3 only** | **2%** | median 1.9%, max 7.6%, 180/180 < 10% | 0.97 |
| block reference, all 24 layers | 0% | | 0.98 |
| hide reference **heart** tokens, layer 3 | **4%** | median 2.9%, 93% < 10% | 0.98 |
| hide reference **background** tokens, layer 3 | 102% | +1.6%, tiny but consistent | 0.98 |
| hide reference heart tokens, layers 0–5 | 2% | | 0.97 |
| no cross-slice attention, layers 12–17 | 85% | median 87%, min 26% | **4.77** |
| naive stack (Δ = 0) | 0% | | 4.77 |

- **E1 single-layer scan** (n=40, full-splat G, leakage floor 16.5%): blocking the reference in any
  one of layers 0, 1, 2, 4, 5 leaves 98–100%. Layer 3 alone leaves 18.9%, i.e. 2.4 pp above the
  floor. Layers 6–23 leave ~100%.
- The reference's special tokens (camera + reference + z token, and registers) are not what is
  read: blocking them in all layers leaves 100%.
- **Removing the reference designation** (`NoRefToken`: slot 0 gets the "other" camera and
  register tokens) abolishes transfer (11.9% on full-splat G, below the floor). The model needs the
  token to know which slice to read.
- **Moving the reference token** to a plane about D/4 from mid (`ref_offmid`, never trained): 73%
  of transfer, with a wide spread (p5 = 0.06, p25 = 0.60).

**Sufficiency** (E5, n=180, companion-only splat). Two runs differ only in the reference frame (ED
vs the phase least similar to ED). The patched run uses the source input, but in the listed layers
companions read the donor's reference keys/values. Slot 0's own queries keep live keys/values.
State index: 0 = source, 1 = donor; both directions averaged.

| patched | index | 95% CI | per subject |
|---|---|---|---|
| **layer 3 only** | **0.975** | [0.973, 0.977] | median 0.977, min 0.919 (per direction min 0.87) |
| all layers except 3 | 0.025 | [0.023, 0.027] | max 0.085 |
| any single other layer 0–11 | ≤ 0.026 | | layer 2 is the largest |
| layers 12–17 / 18–23 | 0.000 | | |
| layer 3, heads {2,10,12,14} (held-out n=177) | **0.938** | [0.935, 0.942] | min 0.854 |
| layer 3, other 12 heads (held-out) | 0.034 | [0.031, 0.037] | |
| single heads h2 / h10 / h12 / h14 | 0.29 / 0.15 / 0.18 / 0.17 | | none dominates |

- The TOP4 heads were chosen on idx 0–2 and are evaluated held-out.
- Only one phase pair (ED vs its least-similar phase) was tested. The donor keys/values are real
  activations, and companions carry almost no reference information before layer 3 (layers 0–2
  patch: 0.026). So the patched state is close to in-distribution; this is the strongest single
  piece of evidence.

**Spatial bias** (E7, n=180): only the reference's layer-3 keys/values on one half of the heart
are patched.
- Companion in-plane displacement changes 2.08 mm on the patched side vs 1.00 mm on the opposite
  side. The full donor effect is 3.02 mm; 180/180 subjects show this.
- A 3×3-token (≈29 mm) patch's influence decays with distance: 0.68 mm at 0–5 mm, 0.39 at 20–25,
  0.13 at 40–45.
- So the read is **biased toward nearby locations at chamber scale, not a same-pixel lookup**. The
  output smoothness (diffusion regulariser, DPT head) also spreads the effect; the two cannot be
  separated.

**Attention** (E3, n=180, descriptive; unmodified model; heart-ROI queries on companions, weighted
by query count):
- Companion→reference attention relative to uniform-over-slots is about 1× in every global layer
  except layer 3. The maximum outside layer 3 is 1.35×, and layer 3 is the peak layer in 94% of
  subjects.
- In layer 3, the causal heads average 6.1× (h2 3.6, h10 6.5, h12 6.8, h14 7.6); the other 12
  average 2.7×.
- 24% (causal) / 11% (other) of the reference mass falls in the 3×3-token neighbourhood of the
  query's own location, against 0.66% chance.
- **h0 and h8 attend about 6× to the reference but are not causal** (patching each transfers
  <2%). Attention maps alone would have misidentified the mechanism.
- Figure: `figs/interp/fig_attention.png`.

**Feature onset** (E9, n=60 = every 3rd test subject; `tools/interp/e9_features.py`). Two runs
differ only in the reference phase (ED vs least-ED-like). We measure the relative change of
companion tokens after every frame and global block:
- It is **< 1% through global layer 2**, then **jumps to 43% after global layer 3** in companion
  heart tokens. Background tokens change 6% and special tokens 1%.
- It then decays with depth (to 2% at the last layer) as the information is turned into
  displacement.
- The per-token change map puts the change on the companion's heart. This independently confirms,
  at the feature level, the layer and the tokens found by the knockouts. Figure:
  `figs/interp/fig_features.png`.
- VGGT-Ω-style joint PCA feature maps (`fig_pca.png`) were also rendered. They are **not
  informative**: colours shift per layer and per slice with no cardiac-specific structure. Not used.

## 3. Breathing: cross-slice comparison in the middle layers (E2, E6)

**E2** (n=180 × 3 breathing draws, val-style sampler; companions). Full model: slope 0.871
[0.858, 0.884], EPE 0.96 mm, in-plane AP slope 0.99.

| no cross-slice attention in | SI slope | EPE mm | AP slope | same-sign subjects |
|---|---|---|---|---|
| layers 0–5 | 0.74 | 1.50 | 0.97 | 88% (EPE) |
| layers 6–11 | 0.53 | 2.55 | 0.42 | 98% |
| **layers 12–17** | **0.001** | **4.92** | 0.01 | 100%; every subject has \|slope\| < 0.09 |
| layers 18–23 | 0.88 | 0.95 | — | null (a same-size control for "any mid-network disruption") |
| all layers | 0.003 | 4.91 | — | 100% |

| slices visible (all layers) | slope | EPE mm |
|---|---|---|
| all except the reference | 0.88 | 0.96 (Δ +0.001, p = 0.92) |
| only itself + reference | 0.10 | 4.46 |
| only itself + adjacent planes | 0.58 | 3.54 |
| all except adjacent planes | 0.77 | 1.77 |

- **Many slices, not neighbours.** Neighbours alone recover 0.58; removing only the neighbours
  still leaves 0.77.
- **The reference is neither required nor privileged**, but it is not excluded. With all
  companions shifted −24 mm, the model "corrects" the reference (+14.8 mm). With all companions
  shifted +24 mm, it corrects the companions (+12.3 mm). The least-shifted slices act as the anchor.
- **Relative + one-sided (a property of the training prior, not an independent mechanism).**
  Training shifts are d_SI = A·sin⁶ ≥ 0, and most slices sit near end-expiration.
  - The model predicts only Δz ≥ 0 and anchors on the least-displaced slices.
  - A shift common to *every* slice, δ ∈ [−24, 24] mm, is essentially not recovered (mean
    |pred| ≤ 1.65 mm).
  - Absolute position is recovered only while the stack satisfies the prior (mean applied 5.1 mm,
    residual offset −0.45 mm).
  - The z-token is used one-sidedly: labelling a slice 24 mm too low yields +15.0 mm; labelling it
    too high yields ≈0.
  - **Implication (untested on real data):** correction should degrade if most slices are not
    near end-expiration.
- Supplementary, OOD: a single shifted slice with all others unshifted is recovered
  non-monotonically. A shift of exactly one pitch duplicates the neighbouring plane and is poorly
  recovered.

**E6** (n=180, descriptive): shifting one slice changes its mid-layer attention *routing* only
slightly. The comparison does not show up as re-routed attention; it is carried in the values. No
breathing sufficiency (patching) or head-level localisation was attempted. The breathing evidence is
knockouts only.

## 4. Interpretation (for the paper)

Draft paragraph, revised after the 3-agent debate (§10):

> *What does CardioStitch learn?* We probed the final model with causal interventions on its
> cross-slice (global) attention, the only pathway between slices, on simulated inputs from the
> test set (n = 180). The model receives no phase label. It takes the cardiac state from the
> reference slice in a single early layer.
>
> In global attention layer 3 (of 24), four heads let every other slice read the reference slice's
> heart region, preferentially near their own in-plane location (Fig. Xa–c).
> - Blocking this read in that one layer removes 98% of the cardiac-state transfer to the other
>   slices (every subject below 10%). Their LV volume swing drops to near that of the naive stack
>   (0.11 vs 0.81 of the GT swing on slices ≥2 away from the reference; nnU-Net, n = 60, our own
>   heart crop).
> - Conversely, transplanting only this read from a reference at another cardiac phase moves the
>   reconstruction 97.5% of the way to that phase.
>
> Breathing, in contrast, is aligned in the middle layers by comparing the slices with each other.
> - Removing cross-slice attention in layers 12–17 abolishes the through-plane correction (EPE
>   equal to the naive stack) while keeping most of the cardiac-state transfer (85%).
> - Blocking the reference leaves the breathing correction unchanged.
>
> The same holds on the real free-breathing scans of Sec. 4.4: blocking the layer-3 reference read
> removes 97% of the LV volume swing of the non-reference slices in both subjects (Fig. Xk–l).
>
> The two sub-problems are thus handled at different depths of the network. This is consistent with
> how a single forward pass separates cardiac from respiratory motion without a phase input or
> gating.

**Limitations sentence** (belongs in the paper): *"Because the breathing correction is relative, a
displacement shared by all slices cannot be recovered; the absolute end-expiratory frame is
inherited from the respiratory model used in training."* It also suggests softening
`method_lx.tex`'s "canonical end-expiratory configuration".

Simple one-liner: *"one early layer copies the heartbeat from the reference; the middle layers line
the slices up against each other."*

## 5. Caveats

- **Simulation.** All E1–E7 interventions use gated on-grid phases with val-style breathing, not
  the AF test simulation. Real RT data (E8) confirms the cardiac pathway on 2 subjects. The
  breathing pathway cannot be scored on real data (no ground truth).
- **Phase estimation.** Not tested. A cross-subject donor (same phase, different anatomy) or a
  phase probe would be needed to separate "copies anatomy" from "reads a phase code".
- **Attention ≠ mechanism.** h0 and h8 attend ~6× to the reference but are not causal.
- **Weight drift.** Drift from VGGT-1B is 30–58% relative Frobenius in every block (weight decay
  over 300 epochs), so it does not localise anything. Not used.
- **Base VGGT-1B** was not probed, so we cannot say whether the layer-3 read is created by
  fine-tuning or repurposed. It sits in a layer that VGGT follow-ups call early and positional
  (docs/125). A same-(x,y) lookup is consistent with that.
- **E4** uses its own padded ROI (in-plane 10 mm + ±1 plane from `heart_roi_canonical`), not the
  paper's `ef_dice` mask, and the full (not companion-only) splat. Its companion-plane numbers
  therefore include some reference leakage; the far-plane column reduces it.
- **p-values.** At n = 180, p = 2.7e-31 is the Wilcoxon floor when every subject agrees. We report
  effect sizes and per-subject quantiles instead. Effects of ~0.01–0.03 mm EPE or +1.6% G are
  "significant" but trivial.
- **E1b heart/background token sets** use the nominal-plane ROI, while the reference itself is
  breathing-shifted (up to 25 mm). The ROI is dilated and the heart-hide result is decisive, so
  this is judged harmless.

## 6. LV volume (paper metric) — E4

nnU-Net Task114 3d_fullres (the paper's segmenter) on every 3rd test subject (n=60). Reference
sweep over 12 phases; LV volume counted on (a) all planes, (b) companion planes, and (c) planes
≥2 from the reference.

| arm | EF err (pp), companion planes | LV swing / GT swing, companion planes | same, far planes |
|---|---|---|---|
| full | 8.8 | 0.83 | 0.81 |
| no cross-slice attn L12–17 | 9.9 (Δ n.s., p=0.16) | 0.79 (p=0.014) | 0.76 |
| **reference read blocked, layer 3 only** | **48.7** (60/60 worse, p=1.6e-11) | **0.14** | **0.11** |
| naive stack | 57.2 | 0.05 | 0.05 |

The naive stack's far-plane curve is not exactly flat. Its input there is constant, but the 3D
segmenter sees the reference plane, so this metric's swing floor is about 0.05. The example subject
in `figs/interp/fig_lv.png` is chosen as typical on both the full-model error and the naive-stack
swing.

## 7. Real free-breathing RT data — E8

`tools/interp/e8_real_rt.py` + `e8_score.py`. Real free-breathing ungated RT scans of the paper's
Sec. 4.4 subjects (AF patient, Volunteer1; neither is in training), under the paper's frame-0 cine
protocol (docs/124): companions frozen at frame 0, reference = mid slice swept over all 180 real
frames.
- `full` reproduces the published `vggt_frame0` recon to 4.8e-7.
- LV volume uses the reference slab excluded (2D nnU-Net Task114, docs/124 method). The swing is
  p95−p5 over 4.5 s.

| arm | AF patient: swing (mL) / r vs ref-slice LV area | Volunteer1: swing / r |
|---|---|---|
| full | 91.8 / 0.98 | 52.0 / 0.98 |
| **reference read blocked, layer 3 only** | **2.4 (3%)** / 0.94 | **1.7 (3%)** / 0.79 |
| no cross-slice attention, layers 12–17 | 90.3 (98%) / 0.98 | 56.9 (109%) / 0.98 |
| naive stack (frame k of every slice) | 66.6 / −0.06 | 37.4 / −0.55 |

- **On real data, blocking one layer's reference read removes 97% of the heartbeat of the rest of
  the heart in both subjects.** This makes the paper's Sec. 4.4 statement ("propagates the cardiac
  state observed in the reference slice") causal.
- Removing mid-layer cross-slice attention leaves the beat intact (98–109%).
- The residual 2 mL still correlates with the reference (r = 0.94 / 0.79). This is most likely the
  reference slab's own content reaching adjacent planes via the splat (full splat here; cf. the
  simulation leakage, §1). It is negligible in size.
- Breathing correction cannot be scored on real data (no ground truth).
- n = 2 subjects, so this confirms the simulation result rather than being a statistical result.
- Figure: `figs/interp/fig_real.png`.

## 7b. Paper figures and the follow-up experiments E10–E16 (2026-09-26)

The paper gets two appendix figures (decided with the user over several iterations; the draft
multi-panel figures `fig_mechanism`, `fig_paper_main`, `fig_appendix`, `fig_final`, `fig_attn_*`,
`fig_B_af` / `fig_B_hrv` are superseded):

- **Fig A — where the heartbeat is read** (`figs/interp/fig_A_inferno.{png,pdf}`,
  `make_figures.fig_A(scale="inferno")`). One query point (x) on a non-reference slice of test
  subject s090; its global-attention weights over the reference slice and three other
  non-reference slices, in layer 3 vs layer 8. Mean of all 16 heads (no head selection), one shared
  linear colour scale, low attention transparent. Layer 3: the reference slice's heart lights up;
  layer 8: attention is spread evenly (each patch ≤ 1% of the colour-scale top; only 4.9% of layer
  8's attention falls inside the displayed crops). Population statement for the caption: layer 3 is
  the peak reference-attention layer in 94% of 180 subjects (E3).
- **Fig B — what breaks** (`figs/interp/fig_B_final_e15.{png,pdf}`, `make_figures.fig_B_final()`).
  CMRx24_Test_P017 under the paper's own AF simulator, 48 frames (4 s): columns (a) full model,
  (b) w/o reference attention (layer 3), (c) w/o cross-slice attention (layers 12–17). Row 1: nnU-Net
  LV volume of the non-reference slices (GT dashed); swings 75 / 12 / 77 mL vs GT 111 mL. Row 2: per
  slice, the engine's through-plane breathing shift (bars) vs the model's predicted shift (dots):
  (a) and (b) track it, (c) ≈ 0.

What was tried and why the final form looks this way:

| exp | script | what | outcome |
|---|---|---|---|
| E10 | `e10_visual_dump.py` | full-FOV recons at ED/ES, 4 arms, 12 subjects | image-grid drafts; ED/ES crops were too subtle to carry the heartbeat claim |
| E11 | `e11_breathing_demo.py` | same-phase stacks with some / all slices shifted | illustrates that a shift shared by all slices goes uncorrected (s075: 0.1 mm of 15); s117 partly recovered it (5 of 8 mm) — illustration only, not used |
| E12 | `e12_attn_gallery.py`, `render_attn_gallery.py`, `render_attn_grid.py` | full maps (24 layers × 16 heads) for named points and for a grid of heart points | the layer-3 hot spot on the reference follows the query location; layers 4–8 and 13–23 give nearly query-independent maps (map similarity 0.9–1.0), so an earlier "same (x,y) in layers 4–10" claim was wrong; query-specific layers are mainly 3 and 11 |
| E12b | inline (E3 stats) | attention received vs breathing shift | layers 12–17 attend more to breathing-shifted slices (median ρ = +0.30, 81% of 180 subjects) but the extra mass sits on image borders (sink-like); correlational, not used |
| E13 | `e13_full_matrix.py` (+ `harness._record_full`) | full token × token head-mean matrix, layers 3/8, s090 (6×6 pooled) | layer 3: a band at the reference slice (18% over all tokens incl. background) plus a weaker band at the far end slice; layer 8 uniform. Hard to read; not used |
| E14 | `e14_rt_breath_frames.py` | real volunteer RT: frame 20 of every slice (paper Fig. 6 input) and a max-breathing frame search | full arm reproduces the paper's recon (2e-4); neither cut changes the long-axis view (2.1–2.4% vs 6.4% for the naive stack); predicted through-plane shifts only 0–1.7 mm, so the volunteer's visible stair-steps are mostly not breathing. Real data cannot show the breathing effect |
| E15 | `e15_af48_cuts.py` | paper engine AF48 / HRV48 bundles, every frame under the 3 arms, nnU-Net 3d_fullres on the `mask_heart_pad10` crop | the Fig B heartbeat row. Candidates: P017 AF48 (chosen), R8V0Y4 AF48, P017/P002/E3L8U8/P015 HRV48 |
| E16 | `e16_breath_readout.py` | per-slot applied (engine `breathing_model`) vs predicted through-plane shift, one frame | the Fig B breathing row; long-axis images could not show one-slice-sized shifts (≤ 15 mm at 10–12 mm pitch) |

**AF12 spot check (paper test protocol, 3 subjects: CMRx24_Test_P001/P013/P014; `scratch/interp/af12chk/`).**
Same E15/E16 pipeline on the paper's `cmrx2024_af12` bundles. Mean LV swing / GT: full 0.85, w/o reference
attention 0.06, w/o cross-slice attention 0.80; EF error 8.7 / 56.6 / 9.6 pp; breathing EPE 0.67 / 0.67 /
3.54 mm. Same pattern as the population table (E4 gated sweep n = 60, E1b/E2 n = 180), so the paper table
uses those numbers as is, with one sentence noting the AF check. Scoring here is simplified (non-reference
slices, EF from the 12-frame min/max, EPE at one frame, no rigid registration) — not the ef_dice scorer.

Notes:
- 48-frame bundles (`*_af48`, `*_hrv48`, the latter a new `hrv48` arm in `build_af_bundle.py`) live in
  `scratch/interp/eval48/` (built there with `--eval-root`; never in `scratch/eval`). They were first
  built into `scratch/eval` by mistake and moved.
- Subject choice: several subjects' 12-phase GT cines have a single-phase ED spike or an
  unphysiological cycle (P017: ejection over 7 of 12 phases). The curves inherit this through the
  reference slice. P002's cycle is physiological but its (b) panel wobbles.
- An earlier "HRV" draft re-timed 12 already-measured LV values instead of reconstructing each
  frame; it was discarded. Every Fig B curve is from real per-frame reconstructions.

## 8. Reproduce

```
PYTHONPATH=training:. python tools/interp/<e1_reference|e1b_dissociation|e2_breathing|e3_attention|
    e4_seg_dump|e5_patching|e6_breath_attention|e7_locality>.py --out temp/interp/<exp> --shard k --nshard 3
python tools/interp/e4_score.py scratch/interp/e4          # after nnUNet_predict -m 3d_fullres on e4/in
PYTHONPATH=training:. python tools/interp/e8_real_rt.py --out scratch/interp/e8
python tools/interp/analyze.py                             # all tables
python tools/interp/make_figures.py                        # figs/interp/{fig_mechanism,fig_attention,fig_breathing,fig_lv}
# paper figures (§7b)
PYTHONPATH=training:. python tools/build_af_bundle.py --source cmrx2024 --subjects CMRx24_Test_P017 \
    --arms af48 --eval-root scratch/interp/eval48      # needs a symlink eval48/cmrx2024 -> scratch/eval/cmrx2024
CUDA_VISIBLE_DEVICES=k PYTHONPATH=training:. python tools/interp/e15_af48_cuts.py --eval-root scratch/interp/eval48 \
    --cohort cmrx2024_af48 --subject CMRx24_Test_P017 --out scratch/interp/e15 --shard k --nshard 3
nnUNet_predict -i scratch/interp/e15/seg_in -o scratch/interp/e15/seg -t 114 -m 3d_fullres -tr nnUNetTrainerV2_MMS
PYTHONPATH=training:. python tools/interp/e16_breath_readout.py --eval-root scratch/interp/eval48 \
    --cohort cmrx2024_af48 --subject CMRx24_Test_P017 --out scratch/interp/e15
python -c "import make_figures as M; M.fig_A(scale='inferno'); M.fig_B_final()"   # from tools/interp
```

Checkpoint: `scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt`.

## 9. prove-it mid-check (2026-09-25)

Four reviewers each read all the code, with lenses on attention mechanics, input geometry, metrics
and statistics, and experiment design. Runtime checks ran on an RTX 6000 Ada. The mechanics,
geometry, signs and units were confirmed correct.

Fixed before the affected runs:
1. E1b `ref_offmid` breathing was scored against un-permuted displacements.
2. The reference slice's own splat leakage inflated knockout G; E1b and E5 now use companion-only
   scoring. This made the knockouts stronger.
3. The E7 p-value double-counted two halves per subject.
4. The TOP4 head-selection subjects were included in the head stats.
5. A relative split path silently returned 0 subjects outside the repo root.
6. A latent crash when combining head and token patching.
7. The E4 docstring overclaimed that it matched the paper's protocol.
8. Wording: the reference's residual stream is "untouched" only up to the patched layer.

## 11. Story ranking for the paper (2-agent debate, 2026-09-25)

A clinical/MICCAI reviewer persona and an ML-interpretability reviewer persona ranked the findings
by importance × simplicity × intuitiveness. They agreed on:
1. Two jobs at two depths, with the asymmetric dissociation stated honestly.
2. The real-data knockout.
3. Layer-3 sufficiency.
4. Feature onset at layer 3 on the heart.
5. Relative breathing, as a Limitations sentence only.
6. Heads, locality and attention maps go to the appendix; PCA is dropped.

They differed only on the image panel; both are kept in `figs/interp/fig_paper_main.png` (5 panels).
Self-contained HTML summary: `figs/interp/interp_report.html` (`tools/interp/make_report.py`).

## 10. 3-agent debate (2026-09-25)

Three independent agents each read the doc, code and raw data (CPU only) with different lenses.

**Correct?** An independent recomputation from raw files reproduces every headline number. Fixed:
- the blanket "all 180 subjects same direction" claims (false for several conditions);
- E3 ×-uniform averaging (now weighted by heart-query count: 6.1× vs 5.65× before);
- the leakage floor (16.5%, not 19%);
- the AP slope (0.99);
- E4 using the full splat, now stated;
- an atypical LV example subject, now replaced.

**Supported?** The layer-3 result survives the out-of-distribution attack: the matched
heart/background token control, sufficiency via real donor activations, and breathing intact under
the knockout. Overreach, now reworded:
- "does not estimate phase" (untested);
- "same location / locally" (the non-local share is 33%);
- "breathing in 12–17" (6–11 also matter);
- "cardiac intact" (a 15% drop);
- "independent of the reference" (the reference is an ordinary slice);
- relative/one-sided presented as a mechanism (it is the sin⁶ ≥ 0 prior);
- simulation-only evidence.

**Significant?**
- "Reads the reference" is trivial by design; what is non-trivial is *where* and *how narrowly*:
  one layer, 4 heads, heart tokens, spatial bias.
- Both main findings hold per subject: the layer-3 knockout max is 7.6%, the layer-3 patch min is
  0.92, and every subject has |slope| < 0.09 without layers 12–17.
- The breathing locus agrees with VGGT literature (middle layers do matching). A critical read in
  early layer 3 extends the "early layers are positional" picture.
- Recommendation: main paper = one paragraph plus a compact panel (layer-3 necessity/sufficiency +
  asymmetric dissociation + LV consequence). Appendix = heads, locality, attention maps, off-mid.
  Limitations = relative/one-sided breathing.
- Highest-value upgrade: the real-data knockout (E8).

**Rebuttal round:** all three accept the revised §4 paragraph for simulated inputs. The final
wording edits are applied: "is consistent with", "without a phase input", and the far-plane LV
swing with the naive value. The "correct?" debater re-verified every new number and both
re-rendered figures; its nits are fixed (E1b label, colour-scale cap 8, E6 n). Both "supported?"
and "significant?" made a positive E8 the condition for any real-data wording. E8 was positive
(97% swing removed in both subjects) and is now in §4.
