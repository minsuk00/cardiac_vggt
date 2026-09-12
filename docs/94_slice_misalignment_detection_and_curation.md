# 94 — Inter-slice breath-hold misalignment in the training GT: measured, and a staged detect+train-through-it plan

> ⚠️ **Superseded on the numbers by docs/95 (2026-09-05, reviewed + rerun):** the whole-heart-centroid polynomial metric below over-reads (23.5% of its flagged subjects have no discontinuity >3 mm) and 4 mm ≈ 3 pixels is too tight a threshold. Re-measured with two agreeing estimators (v3.1, 2026-09-11): 9.0% of adjacent pairs and 38.3% of subjects have a >5 mm in-plane discontinuity (21.3% >8 mm); the median subject's worst discontinuity is 3.9 mm. See `docs/95_slice_misalignment_reanalysis.md` and `_html/95_slice_misalignment_reanalysis.html`.

> **TL;DR & takeaway** — Confirmed real: acquisition-time in-plane inter-slice misalignment (different
> breath-hold position per slice) is present in the on-disk GT across all 5 pooled sources
> (n=30 audit, `heart_seg`-centroid smoothness): mean max-jump **9.0mm** (median 7.0mm), **67%** of
> subjects exceed a 4mm flag threshold, visually confirmed as a myocardial "staircase" kink on
> reslice. Severity is source-correlated, not uniform — ACDC cleanest (33% flagged, 4.0mm mean),
> MNMs/CMRxRecon2025 worst (83%, 10.6–13.4mm) — which matters because it is the key fact a
> 3-agent adversarial debate converged on: this is *not* simple sparse-outlier noise a robust loss
> can assume away, but it also isn't clearly a "drop 2/3 of the 1337-subject pool" problem either,
> since the metric hasn't been validated against actual model failure yet. **Recommended sequence**
> (no camp won outright; revised after the user challenged the original ordering — see §5a):
> **(0)** extend the existing read-only audit script to score the full 1337-subject pool at ED
> (not just n=30) — mandatory, zero risk, informs everything downstream. Scoring every phase was
> the original plan but is now deprioritized: since one breath-hold acquires a slice's whole
> cardiac cycle, the misalignment is a per-slice property shared across phases, not an independent
> per-phase draw — a small cross-phase spot-check (not a full 12-phase sweep) is enough to confirm
> flagged slices are real rather than segmentation noise. **(1a)**
> try plain **split-file exclusion first**: this repo already partitions subjects via a text list
> (`training/splits/pooled.txt`'s `[train]/[val]/[test]` sections, which already excludes 6
> duplicate subjects this same way) — write filtered variants at 2-3 thresholds, point
> `default.yaml`'s `split_file` at each, and judge by `docs/38`. Zero bytes of `scratch/data/` or
> `training/loss.py` touched; purely a config pointer. Caveat that survived debate: the corruption
> is per-**z-plane** (1-3 bad slices out of 9-16 in a typical flagged subject), so subject-level
> exclusion is cruder than the failure mode and can discard many good slices to remove few bad ones
> — this is why it's run as an ablation to be judged, not adopted on cheapness alone. **(1b)**
> conditional — if 1a only partially wins or the pool-shrinkage cost looks too high, try a
> data-untouched loss change instead/in addition: swap `loss_volume`'s `.abs()` for a
> Charbonnier/truncated-L1 kernel, and/or wire the point head's currently-unused 4th ("confidence")
> channel into a per-**z-plane** Kendall-&-Gal heteroscedastic term reusing the existing splat —
> same `docs/38` gate. **(2)** only if severity score still correlates with per-subject eval failure
> after 1a/1b (or both fail outright), do targeted curation-by-registration: re-register only the
> severity-flagged tail (likely MNMs, CMRxRecon2025) by reusing the already-installed NiftyMIC/SVRTK
> baselines' rigid registration step, write corrected volumes to a **new** `preprocess.py`
> `cache_signature` (original cache and `scratch/data/` untouched), A/B as a controlled ablation —
> drops out entirely if 1a alone saturates `docs/38`. **Nothing in this doc was implemented** — this
> is the research/design record; `scratch/data/`,
> `evaluation/`, and the existing cache are untouched. All measurement is read-only in
> `temp/misalign_audit/` (script, CSV, figures — reproducible).
>
> **Update 2026-09-05 — Stage 0 run (§5a): the 4mm threshold does not identify a minority.**
> Full-cohort sweep (935 `[train]`-split subjects, all 12 phases, 63s, 0 errors,
> `temp/data_curation/`) reproduced the aggregate ED flag rate almost exactly (65.6% vs n=30's 67%)
> but **overturned the per-source ranking** — ACDC was NOT uniquely clean at scale (53.0% flagged,
> not 33%; MNMs/CMRxRecon2025 remained worst). More importantly: **at 4mm, every exclusion
> criterion tried flags a majority of the pool** (ED-flag 65.6%, ≥50%-of-phases-flagged 80.1%,
> any-phase-flagged 98.0%) — the median subject's own ED jump (6.2mm) is already above the
> "misalignment" cutoff. The 4mm number (Tarroni et al. 2020, UK Biobank) doesn't transfer: it was
> measured with a different metric on a single tightly-protocoled cohort, not our
> segmentation-centroid-smoothness metric on a heterogeneous multi-center research pool. Also: the
> "misalignment is phase-invariant" acquisition-physics argument in §5's original Stage 0 (below)
> was **too strong** — flag rate isn't constant across phases, it forms a smooth cycle-shaped curve
> peaking near end-systole (79.6% at phase 6) and lowest at ED (65.6%), consistent with
> segmentation-noise inflation during peak motion rather than a real per-phase-varying breath-hold
> offset, but real enough that ED-clean subjects still get their other phases flagged 52.8% of the
> time on average. **Net effect on the plan**: Stage 1a's split-file exclusion should sweep
> **percentile-based cuts** (worst 10/20/33%), not the absolute 4mm/7mm values originally proposed
> — and the scale of what "majority-flagged" means here is a point in favor of taking Stage 1b
> (robust/heteroscedastic loss) seriously in parallel, since a strict severity-minority premise
> looks shakier than assumed. Full write-up: `temp/data_curation/findings.md`.

## 1. The problem, stated precisely

The pooled training cohort (`training/splits/pooled.txt`, 1337 subjects: CMRxRecon2025×359,
M&MS_sax×341, CMRxRecon2024×294, CMRxRecon2023×195, ACDC_sax×148) is classical clinical SAX cine —
acquired as multiple separate 2D slice acquisitions, each its own (or a shared few-slice)
breath-hold. Because the diaphragm/heart sits at a different position during each slice's
breath-hold, the "3D volume" GT built by stacking these slices per cardiac phase can have real
in-plane (x/y) discontinuities across z: anatomy that should drift smoothly with z instead jumps.
This is **distinct** from the z-order/z-roll issues already documented and fixed (docs/56, docs/58
§10 — which end of the stack is apex vs base); this is about in-plane content misalignment between
adjacent slices that are already in the correct order.

This matters specifically for VGGT-MRI because the model is trained **supervised** against these
stacked volumes (`loss_volume = (V_canon - V_gt).abs().mean()`, `training/loss.py:228`) to correct
exactly this class of misalignment (plus respiratory + cardiac motion). If the GT itself contains
an un-modeled instance of the same failure mode the model exists to fix, training risks either (a)
teaching the model to reproduce/hallucinate the GT's own artifact, or (b) simply adding
uninformative variance to supervision that the model has no way to resolve (the input contract
gives it no way to know the true intended position).

## 2. Empirical measurement (this session, verified)

Method (`temp/misalign_audit/measure_misalignment.py`): for n=30 subjects (6 per pooled source,
seeded sample), load the native (unresampled) phase-0 volume and its real per-phase `heart_seg.nii.gz`
(nnU-Net M&Ms Task114 mask, docs/39). Per z-slice, take the heart-mask centroid (y,x); fit a robust
(IRLS/Huber) degree-1/2 polynomial trend through the centroid trajectory vs z; the **residual** (in
mm, via each subject's own in-plane spacing) is the jump signature — true anatomy drifts smoothly
with z, so a large single-slice residual against its own subject's smooth trend is not explained by
normal LV tilt.

**Verified against a real confound before trusting it**: CMRxRecon2024's `heart_seg.nii.gz` was
flagged in this repo's own README as derived from the pre-ESPIRiT-fix v1 recon and "moved to
archive" during the 2026-07-27 v2 rebuild — if the script had picked up stale v1-geometry masks
against v2 images, that alone would fake a jump signal. Checked: the live `heart_seg.nii.gz` files
have mtime 2026-09-01 (after the v2 rebuild), consistent with docs/39's nnU-Net-based regeneration
across the whole pool — not stale. Confound ruled out.

**Results:**

| source | mean max-jump | flag rate (>4mm) |
|---|---|---|
| ACDC_sax | 4.0mm | 33% |
| CMRxRecon2023 | 7.2mm | 67% |
| CMRxRecon2024 | 9.9mm | 67% |
| CMRxRecon2025 | 10.6mm | 83% |
| MNMs_sax | 13.4mm | 83% |
| **all (n=30)** | **9.0mm (median 7.0mm)** | **67%** (63% excluding edge slices) |

Visually confirmed: coronal/sagittal reslices with the measured centroid overlaid show the
myocardium/blood-pool boundary visibly kinking exactly at flagged z (`CMRx23_Train_P013`,
`MNMs_B4O3V3`), vs. smooth monotonic drift with no kink in clean cases (`ACDC_patient038`,
`CMRx23_Train_P118`) — the last pair shows the artifact is per-acquisition, not purely per-source,
even though the per-source rates above are real and fairly wide (33% to 83%).

Caveats (from the measuring agent, not smoothed over): small D (5–16 slices in some subjects)
limits polynomial degrees of freedom; a few of the largest jumps sit at edge slices with small mask
area (reported separately as `interior_max_jump_mm`, and the pattern survives excluding them);
heart-seg mask accuracy itself wasn't independently re-validated beyond the mtime/provenance check
above; only in-plane translation was modeled, not zoom/FOV differences between slices; n=6/source
is too small to do more than suggest the per-source ordering.

Artifacts: `temp/misalign_audit/{measure_misalignment.py,render_examples.py,jump_metrics.csv,figures/}`.
All read-only against `scratch/data/`; nothing there or in `evaluation/` was touched.

## 3. Literature sweep

### 3a. Detection & correction (dataset-side)

- **UK Biobank cardiac QC** (Tarroni et al. 2020, "Large-scale Quality Control of Cardiac Imaging in
  Population Studies: Application to UK Biobank", *Sci Rep*, PMC7015892 — verified by direct fetch
  of the paper, 2026-09-05; an earlier pass of this doc misattributed it to "Ruijsink et al."):
  hybrid random-forest QC over 19,265 SAX stacks scores inter-slice motion as the average in-plane
  misalignment (mm) of each SAX slice relative to a reference position from the long-axis views;
  **16.0% (2,977 stacks) flagged at ≥3.4mm**, the threshold found to best match expert visual
  assessment of "noticeable motion corruption" — an external, citable severity reference in the same
  units our own metric produces.
- **Segmentation-centroid trajectory smoothness** (Chandler et al. 2008, *JCMR*; "Cine Cardiac MRI
  Slice Misalignment Correction Towards Full 3D LV Segmentation", PubMed 30294064) — exactly the
  method our own audit script implements: fit a smooth centroid curve across z from a per-slice
  segmentation, flag/correct large residuals.
- **Classical iterative SVR correction** (Kuklisova-Murgasova 2012; NiftyMIC/Ebner 2020 — Huber-L2
  outlier-robust data term; SVRTK — EM-based robust statistics rejecting 9–19% of slices in
  published fetal-cardiac work; NeSVoR/SVoRT — transformer-regressed slice poses + continuous
  outlier weight). **All three are already installed and wired as baselines in this repo**
  (`baselines/`, `evaluation/src/engine/run_nesvor.sh`, `run_svrtk3d.sh`) — their per-slice
  registration-residual/outlier-weight output is obtainable largely for free, without adopting
  their full reconstruction as our GT.
- **Cardiac-specific rigid correction**: Chandler 2008 (NMI 6-DOF registration to a reference
  volume, cut simulated displacement 7.0mm→1.6mm); MICCAI-STACOM workshop 2017 (LAX+SAX groupwise
  registration, 7.33mm→1.96mm median on severe cases).
- **Exclusion-based curation**: flag-and-drop above a threshold — zero registration risk, but pure
  information loss, and leaves sub-threshold residual noise untouched.

### 3b. Training-time robustness (model-side)

- **RobustNeRF** (Sabour et al., CVPR 2023, arXiv:2302.00833): IRLS/trimmed-least-squares — rank
  per-pixel residuals, down-weight/zero a spatially-smoothed outlier mask, optimize only the inlier
  residual. Built explicitly for a **violated static-scene assumption** from transient content —
  structurally the closest published analogy to our problem (a violated "consistent 3D anatomy"
  assumption from transient breath-hold state). Pure loss-function change.
- **Heteroscedastic uncertainty** (Kendall & Gal, NeurIPS 2017; medical adaptation arXiv:2001.11927):
  network predicts per-output variance; loss `∝ exp(-s)·|residual| + s` lets the network inflate `s`
  (and thus down-weight gradient) where it can't/shouldn't fit. **Notable architectural fit**: the
  point head already emits a 4th "confidence" channel that CLAUDE.md documents as **unused** —
  wiring it into this loss reuses an existing tensor, adds no parameters.
- **Joint/self-calibrating slice-pose correction** (SVoRT, arXiv:2206.10802; "Fully Convolutional
  Slice-to-Volume Reconstruction", CVPR 2024, arXiv:2312.03102 — its "motion-compensating loss"
  removes a best-fit rigid transform before computing the loss): the theoretically "correct" fix,
  but those methods correct **input** slice poses feeding a fused volume; our corruption is baked
  into `V_gt` at cache-build time, a different point in the pipeline, so porting the idea needs a
  new differentiable warp module and degenerate-solution guardrails — genuinely expensive, unproven
  here.
- **Generic noisy-label ML** (co-teaching, curriculum): a 2024 survey (arXiv:2406.15982) explicitly
  flags this as an open gap — noisy-label literature is overwhelmingly classification-focused and
  doesn't transfer cleanly to continuous per-voxel regression; the useful sub-idea (down-weight by
  current residual, relax over training) collapses to a scheduling wrapper on the robust-loss idea
  above, not an independent lever.

## 4. Three-way adversarial debate (full transcripts not reproduced; positions summarized)

Three agents argued, in parallel, for **detect-and-curate**, **train-through-it**, and a **staged/hybrid**
position, each explicitly rebutting the other two's likely objections.

**Detect-and-curate's strongest point**: the per-source spread (33% ACDC vs 83% MNMs on the *same*
detector) is evidence this is a removable, source-correlated defect, not a universal training-time
nuisance — a fixed robust-loss hyperparameter has no way to learn "distrust MNMs subjects more than
ACDC ones" without literally encoding source identity, which is worse practice than fixing the data.
Concrete plan: tier by severity (severe → exclude; moderate → re-register via NiftyMIC/SVRTK reused
on just the flagged z-range; below-threshold → pass through), route through a new `cache_signature`
in `training/data/preprocess.py`, threshold picked by sweeping 4/6/8mm against `docs/38`'s existing
ship metrics rather than fixed a priori.

**Train-through-it's strongest point**: the centroid-jump metric structurally **cannot distinguish**
a breath-hold slip from genuine large cardiac excursion at a hyperdynamic phase — exactly the "hard
but real" motion this project exists to reconstruct — so a hard exclusion/threshold risks teaching
the model to under-fit real motion, which is worse than a bit of hallucinated smoothness and is a
failure mode this project has hit before (`project_vggt_flat_ef_amplitude` memory). A **learned**
per-z-plane uncertainty is information-preserving and falsifiable in a way a fixed threshold isn't:
if the learned `s` doesn't spatially correlate with the independently-measured jump score, that's
direct evidence the mechanism found a shortcut rather than the real artifact, and the fix reverts
to a simpler fixed-`tau` Charbonnier kernel.

**Staged position's strongest point, and why it's adopted as the primary recommendation**: both pure
positions are committing real resources on evidence that doesn't support the commitment yet.
Curate-first bets the *whole pool composition* on an n=30, phase-0-only measurement never checked
against actual model failure. Robust-loss-only leans on an IRLS/RobustNeRF-style "corruption is a
minority of residual mass" premise that our own numbers (67% of *subjects* flagged, 83% for the two
worst sources) put real pressure on — even though, as train-through-it's rebuttal correctly notes,
the *per-plane* corrupted fraction within a subject is still typically small (1–3 flagged z-planes
out of 9–16), so the premise may still approximately hold at the unit the loss actually reweights.
That disagreement is itself unresolved by current evidence — which is precisely the staged
position's point: resolve it empirically, in order, before committing further.

## 5. Recommended plan

**Stage 0 — full-cohort severity scoring, ED-primary (revised after a user challenge to the
original "score every phase" framing).** Extend `temp/misalign_audit/measure_misalignment.py` from
n=30 to the full 1337-subject pool (`training/splits/manifest.csv`) at **phase 0 (ED)**, not all 12
phases. Reasoning (acquisition physics, not measured — worth stating explicitly since it reverses
an earlier unverified claim in this doc that ED was "plausibly a lower bound" on severity): in
classical breath-hold cine, **one breath-hold acquires the entire cardiac cycle for a given slice**
(ECG-gated segmented k-space fills all ~12 phases across that one held breath); the next slice is a
*separate* breath-hold. So the breath-hold-position offset that causes inter-slice misalignment is a
**per-slice property inherited identically by every phase of that slice**, not an independent draw
per phase — there is no acquisition-physics reason to expect a materially different (worse or
different) set of misaligned slices at other phases. What *does* vary by phase is segmentation
reliability (motion blur near systole makes the mask, and therefore the centroid, noisier) — so the
useful reason to look at other phases is a **cross-phase consistency check**, not a search for
hidden worse cases: a jump that appears at ED and persists at 1-2 spot-checked other phases (e.g.
ES) is real; a jump that appears at only one noisy phase and not at others is segmentation-noise.
Score ED across the full pool as the primary pass; spot-check ES (or another phase) on a subset
(e.g. the ED-flagged subjects, plus a clean control sample) to validate the flag is real rather than
scoring all 12 phases × 1337 subjects by default. `loss_volume` is still computed against every
target phase during multi-phase training, but a per-slice offset that's constant across phases means
ED's flag already tells you which slices corrupt every phase's assembled GT volume. Aggregate
per-subject and keep the ED-vs-spot-check comparison for the correlation check in Stage 1→2.
Output stays in `temp/misalign_audit/` (or `tools/` if promoted to a standing script, per the
project's own convention — user's call). This produces the denominator every downstream decision
needs and both debate camps agreed is missing today.

**Stage 1a (revised — do this first) — split-file exclusion ablation.** After a follow-up
challenge from the user ("just flag the clean ones in a txt file and use that when loading — isn't
that what we already do?"), all three debate agents were re-engaged and converged on reordering the
plan around this. The detect-and-curate agent fully conceded it as strictly simpler than its own
tiered/re-registration proposal; the hybrid agent adopted it as a more *direct* causal test than a
loss change (it asks "does removing these subjects help" rather than inferring the answer from a
reweighted loss run). Concretely: this repo already partitions subjects purely via a plain-text
list — `training/splits/pooled.txt`'s `[train]`/`[val]`/`[test]` sections, consumed by
`MRIDataset._find_subjects()`/`split_file`, and it already excludes 6 duplicate subjects this exact
way. So: run Stage 0's full-cohort audit, write one or more filtered variants (**sweep 2-3
thresholds, e.g. 4mm and 7mm, as separate split files rather than committing to one cutoff** — the
hybrid agent's point that "cheap to implement ≠ cheap to decide," since a threshold is still a
choice, not a fact, and 4mm is borrowed from a different population-QC use case, not derived from
this model's failure modes), point `default.yaml`'s `split_file` at each for an ablation train,
judge by the standing `docs/38` gate. **Zero bytes of `scratch/data/` touched, zero lines of
`training/loss.py` touched** — a config pointer, fully reversible.

**One caveat that survived the challenge (train-through-it agent's rebuttal, not dismissed):** the
measured corruption is **per z-plane**, not per-subject — a typical flagged subject in the worst
sources has 1–3 bad slices out of 9–16, not a wholesale-bad volume. Split-file exclusion is
all-or-nothing per subject, so it necessarily discards good slices to remove bad ones, and at
67–83% flagged rates in MNMs/CMRxRecon2025 a naive threshold could discard most of those sources to
launder a minority-plane problem. This is a real cost, not just a theoretical one, and is the
reason Stage 1a is run as an **ablation to be judged empirically**, not adopted as the default on
its cheapness alone — per the hybrid agent's asymmetry argument: if the filtered arm (trained on
far less data) still wins on `docs/38`, that itself is strong evidence the misalignment was
actively harmful, since less data normally hurts.

**Stage 1b (conditional) — the loss-side fix, only if 1a is a partial win.** If Stage 1a helps but
doesn't saturate `docs/38`, or if the pool-shrinkage cost looks too high relative to the gain
(exactly the per-plane-granularity concern above — a loss reweight can in principle keep the 8–14
good slices in a flagged subject and only discount the 1–2 bad ones, which a split file structurally
cannot express), try the loss-side change instead of or in addition to exclusion: swap
`loss_volume`'s `.abs()` (`training/loss.py:228`) for a Charbonnier/truncated-L1 kernel, and/or wire
the point head's unused 4th ("confidence") channel into a Kendall-&-Gal heteroscedastic term pooled
**per z-plane** (not per-voxel) — the granularity that matches the measured failure (a whole
breath-hold slice is offset together). Guard the degenerate "infinite uncertainty everywhere"
solution with an `s`-regularizer and a clamp. Judge by the same `docs/38` gate.

**Gate 1→2 (falsifiable, not vibes-based).** After Stage 1a/1b, correlate Stage 0's per-subject
severity score against that run's own `val_per_subject.csv` (`psnr_3d_motion`, `recov_frac_heart`).
Two outcomes license Stage 2:
  - Stage 1 fails the `docs/38` win condition outright, **or**
  - Stage 1 passes overall, but severity score still correlates with per-subject error (e.g. worst-
    quartile-severity subjects measurably worse than best-quartile at matched source) — meaning
    neither fix is fully absorbing the corruption and it's concentrated in identifiable subjects.

  If neither holds — Stage 1 clears `docs/38` cleanly and severity uncorrelates with error — **stop
  here**; curation-by-registration would be solving a problem that isn't measurably costing the
  model anything. (Per the hybrid agent: Stage 2 also drops out entirely if Stage 1a alone
  saturates `docs/38` — no reason to build registration tooling to fix subjects you can just
  exclude.)

**Stage 2 (conditional only) — targeted, reversible correction.** Re-register only the
severity-flagged tail of the worst-scoring source(s) (Stage 0 suggests MNMs/CMRxRecon2025 as
priors, not a foregone conclusion until Stage 0 runs at full scale) by reusing the already-installed
NiftyMIC/SVRTK rigid registration step — not their full super-resolution reconstruction, and only on
the flagged z-range within flagged subjects, to bound cost. Gate the correction itself behind a
post-hoc QC check (does the corrected stack's own centroid-smoothness score actually improve, with
no new discontinuity introduced — SVR tools tuned on fetal/other anatomy are not guaranteed to
behave well on cardiac SSFP contrast without checking). Write the result to a **new**
`training/data/preprocess.py` `cache_signature` (e.g. `<base>_curated_v1`) — the monai
`PersistentDataset` cache is already content-addressed by signature, so this is a genuinely new
cache path; the current cache and `scratch/data/` are never touched. Train a with/without ablation
on exactly that corrected slice of the pool and judge again by `docs/38`, not by assumption.

## 5a. Stage 0 result (2026-09-05) — run, and it changes the threshold plan

Ran `temp/data_curation/full_sweep.py`: all 935 `[train]`-split subjects (not the full 1337, which
includes val/test — matching what the n=30 probe also sampled from), all 12 phases per subject
(cheap — `heart_seg.nii.gz` is a single already-fully-loaded 4D file, so looping phases costs no
extra I/O). 63 seconds, 0 errors. Full write-up with all tables: `temp/data_curation/findings.md`.

**The aggregate rate reproduced; the per-source ranking partly didn't.** Overall ED flag rate:
65.6% (vs. n=30's 67% — good agreement). But ACDC, which the n=30 sample suggested was uniquely
clean (33%), came in at **53.0%** at full scale — essentially tied with CMRxRecon2023 (53.2%) and
CMRxRecon2024 (54.0%), not a standout-clean source. MNMs (80.7%) and CMRxRecon2025 (76.6%) stayed
the worst two. This is the exact failure mode the n=30 caveat ("too small to claim more than a
suggestive ordering") warned about, and it did happen for one of the five sources — worth
remembering as a concrete example next time an n=6/source claim carries weight in a debate.

**Misalignment is not phase-invariant, but the deviation looks like measurement noise, not new
physical corruption.** The "one breath-hold covers a slice's whole cardiac cycle, so the offset is
phase-invariant" argument used to deprioritize a full 12-phase sweep predicted near-zero phase
dependence; measured phase dependence is real (Pearson r=0.50 between ED jump and the mean of the
other 11 phases — moderate, not near-1.0) but has a **shape** that argues against a genuine
per-phase respiratory effect: flag rate is a smooth, cyclic curve — 65.6% at ED (phase 0), rising
to a plateau of 76-80% around phases 4-9 (roughly end-systole per docs/17's "ES drifts frames
4-7"), falling back toward 69.8% at phase 11 (adjacent to the ED wraparound). There is no
acquisition-physics reason for a *breath-hold* (respiratory) artifact to specifically track
*cardiac* contraction timing — this pattern is consistent with segmentation quality degrading
during peak motion blur and inflating the apparent jump metric there, riding on top of a real,
closer-to-phase-invariant underlying offset. Not proven, since this sweep can't fully separate
"real varying misalignment" from "segmentation noise" — but it's a plausible, evidence-consistent
read, not a guess. Practically: ED-flagged subjects stay flagged elsewhere 85.8% of the time
(reliable signal); ED-*not*-flagged subjects still get 52.8% of their other phases flagged on
average (so "clean at ED" is not the same as "clean everywhere," which matters because training
samples `t_target` across all 12 phases) — ED alone is a reasonable, likely-somewhat-conservative
proxy, not a complete picture.

**The threshold-recalibration finding, and why it matters most.** At the borrowed 4mm cutoff,
*every* exclusion criterion tried flags a majority: ED-flag 65.6%, ≥50%-of-phases-flagged 80.1%,
any-phase-flagged 98.0%. The median subject's own ED jump is 6.2mm — already above "misaligned."
Reaching an actual minority tail (worst 20-33%) needs ~10-14mm, not 4mm (p67≈10.1mm, p80≈14.4mm on
ED; similar on the all-phase mean). The 4mm number was measured by Tarroni et al. 2020 with a
*different* metric (SAX-slice offset vs. a long-axis-derived reference) on UK Biobank's single,
tightly-protocoled cohort — not our segmentation-centroid-across-z smoothness residual on a
heterogeneous, multi-center, multi-vendor pooled research cohort. This sweep can't tell whether
that gap is "our cohort really is more misaligned" or "our metric reads systematically higher for
reasons unrelated to true severity," and doesn't need to for the next decision either way: **a
hard 4mm exclusion is not viable as cheap curation — it would discard most of the training pool,
not a removable minority.**

**Revision to Stage 1a**: sweep **percentile-based cuts** (e.g. worst 10/20/33% by
`mean_max_jump_mm` or `frac_phases_flagged`, both already computed in
`temp/data_curation/full_sweep_per_subject.csv`) instead of the absolute 4mm/7mm values originally
proposed — those were anchored to a threshold this measurement shows doesn't transfer. This also
means the train-through-it camp's original objection ("robust losses assume corruption is a
minority of residual mass") deserves more weight than the debate initially gave it, since by any
severity cut tried here "clean" is a minority, not "corrupted" — Stage 1b is worth trying in
parallel with Stage 1a now, not strictly after it.

## 6. Explicit non-actions (scope of this doc)

Per the task this doc answers ("do a thorough research... do not touch any of our data or overwrite"):
**nothing above was implemented.** No change to `training/loss.py`, `training/data/preprocess.py`,
or any cache. No subject was excluded from any split file. No registration tool was re-run as a
correction pass. **Stage 0 (§5a) has since been run** — it is pure measurement (read-only,
`temp/data_curation/`) and changes nothing about training or the data itself. Stage 1a/1b and Stage
2 remain follow-up implementation work, to be done only on further direction, each as its own
small, surgical, `docs/38`-validated change — not bundled together.
