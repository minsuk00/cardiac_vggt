# 95 — Inter-slice misalignment re-measured (reviewed, v3.1): 9.0% of adjacent slice pairs / 38.3% of subjects have a >5 mm in-plane discontinuity (21.3% >8 mm); docs/94's 65.6% was inflated by its metric and its 4 mm threshold

> ➡ **The final curated split built on this score (automatic clean+mild cut + the user's by-eye overrides → `training/splits/pooled_curated_v1.txt`, 898/1337) is documented in docs/96.** This doc is the measurement and its rule history.

> **TL;DR & takeaway** — Re-measured on **all 1337 pooled subjects × 12 phases** with two independent
> per-slice estimators that must agree (epicardial-centroid geometry from the nnU-Net seg, and NCC
> image registration to the neighbouring slices; both phase-median), adversarially reviewed by two
> agents, re-run, and then **re-run again (v3.1, 2026-09-11) after a by-eye catch exposed two
> consensus-rule bugs** (§"v3.1"). **Median slice offset 1.2 mm (≈1 pixel); 9.0% of adjacent pairs have a
> >5 mm discontinuity (4.7% >8 mm); 38.3% of subjects have at least one >5 mm discontinuity, 21.3%
> >8 mm, 62.9% >3 mm** (v3 said 35.8 / 17.4 / 61.2 — the fixes moved 80 subjects UP a band, 26 of them
> out of "clean"). Offsets are phase-invariant (cross-phase scatter 0.5 mm) — a per-slice breath-hold
> property. Present in every centre/vendor group (MNMs centre 1 Siemens 64% >5 mm worst, CMRx25 Siemens
> Vida sites 10–28% cleanest). docs/94 is overstated, not absurd: like-for-like (its interior-slice
> column vs our severity at its 4 mm threshold) 50.9% vs 48.1%, r=0.57; its whole-heart-centroid
> polynomial residual additionally flags 22.1% of subjects with no discontinuity above 3 mm, and "the
> median subject is misaligned" does not hold (median worst discontinuity 3.9 mm). **Curation:** subject-
> level severity lists that filter **train, val AND test** (the val/test GT carries the same corruption and
> would penalise a correct reconstruction): `temp/misalign_v2/curated_sev8mm.txt` keeps 729/935 train,
> 107/133 val, 220/269 test; `curated_sev5mm.txt` 574 / 91 / 167. Every train centre/vendor group is
> retained, and val/test centre shares move by <1 point (only the worst-misaligned groups, MNMs centre 1
> and CMRx25 Aera, shrink); 18 unmeasurable short-stack subjects kept and named in the header.
> **Slice-level masking is NOT recommended** (a shifted block has no single bad slice; the
> neighbours of a slip inherit half its spike). Don't cut at 3 mm (63%), don't re-register. docs/38 on an
> ablation remains the arbiter. Report: **`_html/95_slice_misalignment_reanalysis.html`**; galleries of
> every measurable subject per band `_html/95d_gallery_*.html`; code `tools/misalign_v2/`; nothing under
> `scratch/data/`, `evaluation/`, splits, or caches touched.

## v3.1 (2026-09-11): a by-eye catch, two rule bugs, full re-sweep

The user, skimming the *clean* gallery, spotted `ACDC_patient081` (reported severity 2.4 mm) with an
unmistakable ~28 mm jump between z7 and z8 — everything below aligned on one side, everything above on
the other. Two bugs, both in how the "both estimators must agree" rule handled a non-answer:

1. **Registration search window too narrow, and "no answer" treated as "disagrees".** The NCC search
   covered ±20 mm. For a 28 mm shift the peak sat pinned at the bound, was discarded as degenerate
   (NaN), and the consensus rule then dropped the slice — throwing away the segmentation's correct
   27.9 mm answer exactly as if the image had said "no shift". Fix: window 20→40 mm
   (`MAXSHIFT_MM`), and `metric_v2.py` now records *why* a registration failed (`img_pin_*`: pinned
   at the bound = "offset ≥ window"; `img_flat_*`: featureless ROI = genuinely no answer). A pinned
   registration is treated as agreeing with any large segmentation estimate — the seg value is used
   as-is (`spike_src`/`step_src == "seg_fallback"`, marked † in the galleries): 91 slices, 74 pairs,
   116 subjects. A flat registration still leaves the slice unscored.
2. **Step direction check zeroed pairs that were large in both estimators.** The step rule required
   the seg and img vectors to agree in direction (`|seg−img| ≤ max(2.5, 0.5·min)`), and on failure set
   the pair to **0.0**. That is right when one estimate is small (the segmentation false-positive
   pattern the rule exists for — of the pairs where the seg claims >5 mm and the image has a valid
   answer, 36% are contradicted by it) but wrong when both are large: `CMRx25_R1test_Center001_UIH_30T_umr780_P003`
   z1 had seg 20.2 / img 15.2 mm and scored 0.0. Fix: if both magnitudes are ≥8 mm the pair keeps the
   conservative min(seg, img) (`step_src == "both_large"`, 101 pairs, 85 subjects); otherwise it is
   still zeroed (`"contradict"`).
3. **Why the first review missed it.** Fault injection only tested shifts inside the window; the
   prove-it review checked code against its own contract, not the window's coverage; the sensitivity
   analysis looked at aggregate rates, never at individual dropped slices; and the clean bucket was
   never audited by eye. `tools/misalign_v2/test_fault_injection.py` now covers the gap: 16 checks on a
   clean Z=12 subject (in-window +5.4/−2.8 mm slip; a 28 mm slip under BOTH the 40 mm window (recovered
   by consensus: img 27.9 vs 28.2 injected) and the old 20 mm window (registration pinned → seg fallback
   28.2 mm); an 8.1 mm 3-slice block (edge steps 7.7/7.0, inside 0.3/0.2, spike halved 4.0); synthetic
   direction disagreements (both-large kept at 10.6, one-small zeroed)). All pass.
4. **Effect.** 84 of 1319 measurable subjects changed band (80 up, 4 down; 26 left "clean", 52 became
   "severe"; transitions clean→mild 13, clean→moderate 7, clean→severe 6, mild→moderate 7, mild→severe
   17, moderate→severe 29). >5 mm subjects 35.8% → 38.3%, >8 mm 17.4% → 21.3%. Per-subject list:
   `temp/misalign_v2/band_changes_vs_v3.csv`; the biggest movers are rendered at the bottom of
   `_html/95c_misalignment_examples_by_category.html`. Previous outputs and panels kept under
   `temp/misalign_v2/_v3_superseded/`.
5. **Is spike still needed next to step?** 63 subjects would land in a different band with step alone
   (265 have spike > step), so both stay. (v3 said 81; part of that was bug 2.)
6. **Cost of the wider window.** More spurious far NCC peaks on consensus slices (144 slices with image
   >10 mm but segmentation <3 mm, vs 26 at 20 mm); the seg-vs-img correlation drops 0.78 → 0.52 (0.63
   with both clipped at 15 mm, 0.76 on mid-ventricular NCC>0.6 slices). The scores are unaffected —
   min(seg, img) takes the segmentation's small value there — but it is why r fell.
7. **Terminology.** "kink" → **spike** (single-slice deviation from its two neighbours), paired with
   **step** (adjacent-pair jump vs the subject's own median drift). Same formula; all columns renamed.

## What the first version got wrong (prove-it review, 2 reviewers + fault injection)

The first version of this doc/report (same day) said 22.8% of subjects >5 mm and recommended masking
568 individual slices. Both were wrong, in ways worth recording:

1. **The spike metric halves a shifted block.** `c_z − (c_{z−1}+c_{z+1})/2` reads `s/2` at the edges of a
   block of slices shifted together by `s` and 0 inside it; a 9 mm block scored "mild". Fault injection
   on a clean subject: a 3-slice block shifted 8.1 mm → edge spikes 3.4–5.4 mm. Fix: add an adjacent-pair
   **step** = `(c_{z+1}−c_z) − median_z(c_{z+1}−c_z)` (subject's own drift removed), seg and image must
   agree; severity = max(spike, step). >5 mm subject rate 20.7% (spike) → 35.8% (spike or step, v3).
2. **"~half of severe cases are interleaved odd/even" was the operator talking.** A single slip gives
   `(−s/2, +s, −s/2)` on three slices — alternating by construction. Requiring a ≥4-slice alternating run:
   12.0% of severe subjects (v3.1).
3. **Slice masking is the wrong remedy.** 189 of the 542 slices with spike >5 mm are the half-magnitude
   halo of a bigger neighbour; a block shift has no single wrong slice. Withdrawn (`_v1_superseded/`).
4. **The docs/94 comparison was not like-for-like** (our interior-only max vs their end-slice-inclusive
   max): 42%/r=0.28 became 22%/r=0.57 against their own interior column.
5. **Image-registration sign** was the negative of the seg spike (magnitudes, hence all v1 numbers,
   unaffected); degenerate registrations were not gated; sensitivity numbers mixed gated/ungated
   slices. All fixed in code; every number in the HTML comes from `summary.json`.

## Method (`tools/misalign_v2/`)

- `metric_v2.py` (GPU, 1337 subjects in 2126 s at the 40 mm window, 0 errors): per slice z, per phase
  t — epicardial (labels 1+2) and LV centroid; NCC translation registration (±40 mm, sub-pixel, conv2d
  `ncc_fast.py` verified equal to brute force to 5e-4 px) of slice z to z−1 and z+1 inside `heart_roi`,
  sign convention = seg's; pinned/flat bookkeeping per side. Spike and one-sided displacements reduced
  to the median vector over 12 phases + RMS cross-phase scatter; phase-median centroid stored for the
  step metric. `analyze_arrays()` is the disk-free entry point used by the fault-injection test.
- `analyze.py`: gates (epicardial mask ≥400 mm², ≥6 phases, both registrations NCC ≥0.4, or seg
  fallback when pinned → 7586 slices, 9523 pairs; 1319/1337 subjects measurable), consensus rules above,
  subject severity, per source/centre tables, like-for-like docs/94 comparison, band-change table vs the
  previous sweep, curated lists, figures, `summary.json`. `gate_spikes()` / `compute_steps()` are
  importable.
- **Val/test curation and centre balance.** The lists filter all three sections. Curating val/test does
  not create a centre-dominance problem: at 8 mm the per-centre share of val+test moves by under one
  point for every large group (ACDC 16.2→15.8%, CMRx24 Fudan 14.7→15.5%, MNMs Canon 12.4→13.3%, MNMs GE
  12.2→11.8%); MNMs centre 1 (5.0→2.8%, 9/20 kept) and CMRx25 Aera (1.0→0.3%, 1/4) shrink because they
  are the worst-misaligned. The pre-existing weakness is the opposite one: most CMRx25 centres have only
  1–7 val/test subjects before any cut, so per-centre eval on them was never meaningful; fixing that
  needs a centre-stratified re-split, which costs comparability with every existing run — a separate
  decision, not made here.
- `test_fault_injection.py` — run it after any change to the metric or the rules.
- `render_slab.py` + `build_band_gallery.py` → **`_html/95d_gallery_{clean,mild,moderate,severe}.html`:
  every measurable subject (1319 total: 489/325/224/281), one gallery per severity band**;
  `build_categorized_examples.py` → `_html/95c_misalignment_examples_by_category.html`: **4 examples per
  band chosen automatically at fixed within-band severity quantiles (low edge → high edge) plus the
  subjects v3.1 moved up the most — start here**; `build_html.py` → the main report. All use a no-gap,
  no-interpolation reslice: every native SAX slice stacked as an equal-height flat band back to back (a
  true-physical-spacing version was too thin/sparse to read); epicardial extent ticked per slice; red ▶
  for consensus spike >5 mm (no step marker — by request, "just show me the slice"); † in the number
  column for seg-only values; real LAX frames beside CMRx23/24 panels, reference only — no geometry.
  `_html/95b_misalignment_severe_gallery.html` (`build_gallery.py`) is the earlier severe-only layout.

## Validation

- Estimators agree: r=0.52 (all consensus slices; 0.63 clipped at 15 mm), 0.76 (mid-ventricular,
  NCC>0.6); median vector disagreement 1.1 mm (spike), 1.6 mm (step).
- Cross-phase: scatter 0.48/0.70 mm (seg/img) overall, 0.67/1.01 mm on flagged slices; 0.6% of flagged
  slices have scatter larger than their offset.
- Sensitivity of the >5 mm subject rate (same gated slices, same denominator): consensus-min spike
  22.0%, mean 34.9%, seg-only 26.5%, img-only 41.5%, spike-or-step 38.3% (headline).
- Fault injection: 16/16 checks (§v3.1 item 3).
- Geometry: a 5 mm spike at 10 mm pitch needs a 10 mm radius of curvature; tilt gives zero.
- Visual: every measurable subject rendered and browsable per band. Post-v3.1 by-eye audit of a random
  48 clean + 24 mild subjects (contact sheets, `temp/misalign_v2/audit/`): no missed jump. One weak spot
  found: 38 subjects (29 clean, 7 mild, 1 moderate, 1 severe) have *every* scored pair zeroed as a
  seg/img contradiction, so their step score is uninformative and severity rests on the spike consensus
  alone — flagged as `weak_step_evidence` in `subjects_v3.csv`, not excluded.

## Caveats

Only in-plane offsets are measurable (LAX NIfTIs carry no scanner geometry); end slices and slices
outside the ventricles are unscored and their loss weight was not measured; seg-only (†) values rest
on one estimator (rare, and only when the registration says ≥40 mm); the wider window costs estimator
correlation but not scores; 3/5/8 mm thresholds come from pixel size, wall thickness and the training
respiratory-sim AP amplitude (4–9 mm), not from a model ablation.
