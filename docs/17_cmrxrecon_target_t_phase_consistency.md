# 17 — Is CMRxRecon's `target_t = k/12` a fixed cardiac state?

> **TL;DR & takeaway.** **No.** CMRxRecon's 12 cine phases are a **resampling of each
> subject's own R–R interval** (native acquired phase count varies **14–40**, *never* 12,
> yet every subject is stored as exactly 12), so `target_t = k/12` is **normalized
> fractional cycle time**, not a fixed physiological state. Proven two ways over **all 301
> subjects**: (1) exact structural fact — 301/301 have 12 frames, native `TemporalPhase`
> 14–40 never 12; (2) gold-standard — nnU-Net Task114 (M&Ms) LV blood-pool volume in all
> **301×12 = 3,612** phase volumes. Only `target_t = 0` (end-diastole) is an anchored state
> (LV-volume maximum for **97%** of subjects). **End-systole drifts across frames 3–8 (frac
> 0.25–0.67, std 0.72)**; the cross-subject state spread is ≈0 at the ED anchor and peaks in
> **mid-diastole** (the loosest phase, because diastolic duration absorbs heart-rate
> variation). The ambiguity is **bounded** within this single breath-hold cohort (IQR t5–6,
> 97% at t5–7) but **real**, and its tails reach the full ACDC range. CMRxRecon has **no ES
> label**, so physiological ED↔ES phase normalization is impossible. **Practical reading:**
> ignorable for CMRxRecon-only training (the model learns a usable soft `target_t→state`
> map), but it is *not* a fixed-state guarantee and *not* a license to mix a wider-HR /
> cardiomyopathy cohort (e.g. ACDC), which would widen bounded → contradictory. Report:
> `_html/16_cmrxrecon_target_t_phase_consistency.html`.

## Motivation

The model conditions reconstruction on `target_t`, the fractional position in the cardiac
cycle. A critique (originally raised on ACDC) is that a *fraction* of the cycle is not a
fixed cardiac *state*: heart rate and systole/diastole timing vary, so end-systole lands at
different fractions across subjects, giving contradictory supervision for a fixed `target_t`.
This doc settles whether that ambiguity exists in **CMRxRecon** specifically, with
conclusive full-dataset evidence rather than a transfer argument from ACDC.

## Method

`tools/cmrxrecon_phase_analysis/`:

- **`structural_facts.py`** — exact on-disk frame count per subject + native
  `TemporalPhase` from each `cine_sax_info.csv` + an intensity periodicity proxy
  (resample-vs-truncate discriminator). No segmentation.
- **`prep_phases.py`** — rewrites the 12 native `sax_frame_{tt}.nii.gz` per subject to
  nnU-Net inputs, preserving each subject's **true per-subject voxel spacing**.
- **`analyze_phases.py`** — reads the Task114 segmentations, computes **LV blood-pool
  volume per phase** (`count(label==1) × voxel_mL`), derives ED (max LV), ES (min LV), EF,
  a robust **parabolic sub-frame ES** (immune to integer-grid quantization on the flat
  trough), and the **per-phase cross-subject state spread** (std of contraction-fraction
  `cf` and relative volume `v_rel`).
- **`build_report.py`** — self-contained HTML with embedded plots.

Segmentation: `nnUNet_predict -t 114 -m 2d -tr nnUNetTrainerV2_MMS` (M&Ms 5-fold 2D
ensemble, validated LV Dice ≈ 0.95) in the isolated `nnunet` env (see `docs/15`). `svr`
untouched.

## Results (all 301)

| Quantity | Value |
|---|---|
| On-disk frames / subject | **12 for 301/301** |
| Native `TemporalPhase` | 14–40 (mean 22), **never 12** |
| Diff-from-ED curve | peaks t5, **falls for 99.7%** of subjects ⇒ full-cycle resample (not truncation); corr(ES-proxy, native count) = +0.31 |
| EF (sanity) | **64 ± 6 %** (normal) |
| ED = frame 0 (gating anchor) | **97 %** of subjects |
| ES frame (argmin LV) | mean 5.9, **std 0.72, range 3–8**, IQR 5–6, frac **0.25–0.67** |
| Robust sub-frame ES | std **0.68 frames** (drift survives de-quantization ⇒ real, not trough noise) |
| Per-phase state spread (cf_std) | ≈0.02 at ED → **peaks 0.20 at t9 (mid-diastole)** |

The example contraction curves make it visceral: `Test_P044` reaches ES at **t3** while
`Val_P002` is still contracting at **t8** — both anchored at t0, same `target_t` meaning a
different heart.

## ACDC comparison — does mixing widen it? (yes, empirically)

ACDC (150 subjects, the candidate dataset to mix in) run through the **identical** pipeline
(resample each subject to 12 ED-aligned phases → same Task114 seg → argmin-LV), plus its
ground-truth ED/ES labels as a cross-check. The nnU-Net argmin method **self-validates**:
on ACDC it reproduces the GT ES distribution (nnU-Net frac 0.40 vs GT 0.376; GT exactly
matches the literature 0.38 ± 0.08, range 0.25–0.69).

| | CMRxRecon (nnU-Net /12) | ACDC (same nnU-Net /12) | ACDC GT |
|---|---|---|---|
| ES frac mean | 0.49 | 0.40 | 0.376 |
| ES frac std | **0.060** | **0.100** (1.67×) | 0.077 |
| ES range | 0.25–0.67 | 0.25–0.75 | 0.25–0.69 |
| EF | 64 ± 6 % | 45 ± 19 % | 48 ± 21 % |
| ED = frame 0 | 97 % | 71 % | — |

Two independent mixing hazards, both measured not argued: (1) ACDC's ES distribution is
**1.67× wider** (driven by pathology: GT ES frac **DCM 0.435 → 0.69 @ EF 20%**, RV 0.394,
MINF 0.364, HCM/NOR 0.345); (2) the two cohorts **peak at different `target_t`** (0.49 vs
0.40) — so a fixed `target_t` is a *systematically* different cardiac state depending on
which dataset a sample came from, on top of being noisier. Per-phase state spread: ACDC sits
above CMRxRecon across systole/mid-cycle (peak ~0.25 vs ~0.20). ACDC also re-anchors `k/N`
at a different native frame count (NbFrame 12–35). **Verdict: mixing ACDC without
physiological ED↔ES normalization injects contradictory `target_t→state` supervision —
confirmed.** (CMRxRecon's later mean ES, 0.49, partly reflects its dataset-provided native→12
resampling, which is not under our control; the *spread* and *peak-offset* conclusions are
robust to this.) Tools: `acdc_gt_analysis.py`, `prep_acdc_phases.py`.

## Why bounded-but-real, and where it bites

- **ED is anchored** (R-wave trigger ⇒ frame 0 = LV max for 97%). Near-ES is also
  relatively tight because ES is a velocity turning point (flat LV trough) **and** systolic
  duration is relatively heart-rate-invariant.
- **Diastole is the loose end**: diastolic duration absorbs HR variation, so a fixed
  `target_t` in mid-diastole (t9) maps to the widest range of states (cf_std 0.20). This is
  invisible if one only checks ES.
- **No ES label** in CMRxRecon ⇒ only half the physiological anchor exists ⇒ can't time-warp
  ED↔ES to make `target_t` mean a consistent state; EF itself requires segmenting all 12
  phases and taking argmin-LV (done here).

## Caveats (from a 4-agent prove-it review of the analysis code)

One confirmed bug was fixed (a constant-LV subject would NaN-poison the whole per-phase
aggregate; now skipped). Methodological caveats folded into the report: (1) the LV trough is
physiologically flat (~2–3 frames) so the *integer* argmin-ES has ±1-frame noise — mitigated
by the sub-frame estimator and the continuous state-spread curve, both of which exceed the
trough width; (2) the contraction-fraction is endpoint-pinned (cf≡1 at ED, ≡0 at ES per
subject) so its std is read at interior frames only, with `v_rel` (un-normalized) reported
alongside; (3) Task114 is M&Ms-trained on thinner slices — a segmentation bias that is
constant across a subject's 12 phases cancels in ES-frame / EF / cf (all relative), so it
cannot manufacture the qualitative drift; EF≈64% confirms the segmenter is in-regime.

Geometry/axis/spacing/label correctness was independently cleared (native frames are already
`(X,Y,Z)`; no transpose needed; LV=1/MYO=2/RV=3 consistent).
