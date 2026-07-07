# 35 — Self-gating methods for single-orientation SAX RTFB → SVRTK

> **TL;DR & takeaway**
>
> To feed SVRTK's `reconstructCardiac` (the SVR baseline, doc 34) we must turn ungated
> single-orientation SAX real-time free-breathing (RTFB) cine slices into a **per-frame cardiac
> phase θ + per-slice R-R interval**, purely from the reconstructed image time series (we have
> magnitude AND complex; no raw k-space, no navigator). A deep literature sweep (5 angles, 21
> primary sources, 25 adversarially-verified claims) settled the method choice:
>
> **The gater = two jobs, two methods:**
> 1. **Per-slice R-R period → temporal-Fourier ("x-f") FFT** on an ROI-mean signal, pick the
>    dominant peak in the adult cardiac band. This is exactly what the peer-reviewed SVRTK-adjacent
>    pipeline **fetal_cmr_4d (van Amerom, MRM 2019)** already does per-slice — the strongest
>    cite-able precedent, and the code is already in `baselines/fetal_cmr_4d/`. Robust at **every**
>    slice level (the beat is detectable even at apex; "how fast" is easy).
> 2. **Per-slice ED anchor (which frame is θ=0) → LV blood-pool AREA** from a CNN segmentation:
>    ED = local maximum of LV cross-sectional area (±3-frame rule), ES = area minimum between two
>    EDs. This is exactly what **Åkesson et al. 2025** (*Clin Physiol Funct Imaging*) does — **the
>    one on-point paper**: ECG-free, image-based retrospective synchronization of 2D RTFB **SAX**
>    cine, per-slice. We substitute our public **nnU-Net Task114 (M&Ms)** segmenter (in-distribution
>    for SAX; Åkesson's Bai-2018 weights are private, no code released).
>
> **Then:** assemble continuous θ = 2π·frac((f − f_ED)/T_s) per frame → the slice-major `-cardphase`
> file; per-slice T_s → `-rrintervals`; hand to SVRTK. Anchoring every slice's ED to θ=0 **is** the
> cross-slice synchronization — no spatial overlap needed, which is why it substitutes for the
> interslice-overlap step SAX geometry breaks (doc 34's central failure).
>
> **Why LV-area is the best anchor of all researched methods (the sharp reason):** every other
> method (x-f Fourier phase, manifold learning, PCA/SSA, blood-pool intensity) gives a *periodic
> signal* — great for the **period** and **relative** phase, but it does **not** tell you which
> point in the cycle is ED. LV-area is the **only** surrogate that is **anatomically ED by
> construction** (max cavity = max fill = ED). That is precisely what breaks doc 34's **106° ED
> scatter** — a Fourier signal has no ED landmark, so signal methods *cannot* anchor.
>
> **Complex data does NOT help (defensible default = magnitude).** The one paper deriving cardiac
> signal from phase (Seo et al., MRM 2017) uses a *velocity-encoded projection navigator*, not
> reconstructed-image phase; every established image-based method is magnitude-based.
>
> **Base/apex is a PHYSICAL limit, not a wrong-method choice.** LV-area fails at the extreme slices
> because of atrioventricular-plane descent (base plane moves ~1–2 cm out of slice in systole → area
> tracks through-plane motion, not filling) + a vanishing apical cavity. **No researched method
> fixes this** — it's the anatomy, not the algorithm. Åkesson admits the same. Handle by
> down-weighting those slices (SVRTK robust statistics already do this) — don't trust their anchor.
>
> **One open design choice, reduced to a measurement (a 2-agent debate converged here):** do we also
> need Åkesson's separate systole/diastole resampling? For SVRTK, **NO to image resampling** — the
> temporal PSF (sinc×Tukey) + per-slice `-rrintervals` replaces it. Whether we need her
> *ES-alignment effect* (linear θ pins ED but lets ES drift across slices with different HR) hinges
> on **one nearly-free number: the cross-slice ES-phase scatter** (ES = LV-area min, same masks as
> ED). Default to **ED-anchored linear θ** (the faithful van Amerom choice); if measured ES scatter
> ≫ ~15°, upgrade to **two-anchor piecewise-linear θ *labels*** (not image resampling) on the
> reliable mid-slices.
>
> **Status:** research + design complete; **nothing implemented yet.** Next = write the gater
> (x-f rate + nnU-Net LV-area anchor + θ assembly) and, before touching SVRTK, measure on one MIITT
> subject whether the 106° ED scatter collapses (proof-of-work) + the ES-scatter decision gate.

---

## 1. The problem (from doc 34)

SVRTK's `reconstructCardiac` does **not** self-gate — it *trusts* a cardiac-phase label handed to
it (doc 34 §3.1). To run the SVR baseline on our SAX RTFB data we must produce, from the
reconstructed image series alone:

| SVRTK input | Shape | Meaning |
|---|---|---|
| `-cardphase` | one θ per frame, **slice-major/frame-minor** | continuous θ∈[0,2π], **θ=0 ≡ R-trigger ≈ ED**. This is the entire self-gating output; SVRTK trusts it **blindly** — all cross-slice coherence lives here. |
| `-rrintervals` | one per slice | that slice's R-R period (HR drifts over the ~1-min scan) |
| `-rrinterval` | one scalar | mean R-R = output cine cycle length |

**Constraint:** image-domain only. We have magnitude and complex reconstructed images; **no raw
k-space, no navigator/pilot-tone**. So this is *retrospective image-based self-gating*.

**The hard part (doc 34's finding):** the paper's cross-slice coherence comes from **interslice
sync** — correlating slices where they **overlap in 3D**, which only happens between different
orientations. Parallel SAX slices never overlap → interslice sync is impossible → using the raw
intra-slice phases gives **circular-std = 106° ED scatter** across slices → the reconstructed LV
blends ED/ES states across z. We need a per-slice **absolute** anchor to replace interslice sync.

## 2. Research method + scope

Deep-research harness: 5 search angles, 21 primary sources fetched, 94 claims extracted, top 25
adversarially verified (3-vote, need 2/3 to refute) → **25/25 confirmed, 0 refuted.** Angles:
established self-gating families; signal separation & frequency bands; reference toolboxes; does
complex/phase help; per-slice gating & ED anchoring.

## 3. The method families (comparison)

The decisive axis is **usability under the image-only constraint**, then **can it anchor ED**.

| Method | Key cite | Signal produced | Usable image-only? | Can anchor ED? |
|---|---|---|---|---|
| **x-f temporal Fourier on image ROI** | van Amerom, MRM 2019 (fetal_cmr_4d) | per-slice **rate/period** | ✅ (already in our repo) | ❌ gives rate, not anchor |
| **LV blood-pool AREA** | **Åkesson, CPFI 2025** | per-slice **ED/ES** | ✅ | ✅ **anatomically ED by construction** |
| Manifold learning on image series | Usman, JMRI 2015 (PMID 25124545) | 1D cardiac (<20 ms SD vs ECG) + respiratory | ✅ | ❌ periodic signal, ambiguous extreme |
| Central-k-space echo peak | **Larson, MRM 2004** (seminal, >500 cites) | cardiac trigger | ❌ needs k-space | (n/a) |
| SSA-FARY (PCA on Hankel embedding) | Rosenzweig, IEEE TMI 2020 (BART impl) | cardiac+resp components | ❌ needs ACS/k-space | (n/a) |
| XD-GRASP frequency-band coil select | Feng, MRM (PMC4583338) | cardiac+resp | ❌ navigator/k-space | (n/a) |
| BSS / ICA on k-space center | von Kleist, MRI 2021 (bias 0.0 bpm) | cardiac | ❌ k-space center | (n/a) |

**Two observations that drive the whole decision:**

**(a) The classic self-gating canon is unusable here.** Larson 2004, SSA-FARY, XD-GRASP, BSS/ICA are
the most-cited "self-gating" methods, but **all derive their signal from raw k-space center /
navigator data we don't have.** They are background/citation context, not usable tools. Their
transferable ideas (frequency bands ~0.1–0.5 Hz respiratory, ~0.5–2.5 Hz cardiac; PCA
component-selection) survive; the mechanisms don't.

**(b) Signal methods can't anchor ED — only LV-area can.** x-f Fourier, manifold learning, PCA/SSA,
and blood-pool intensity all yield a clean *periodic signal*: excellent for the **period** and for
**relative** phase, but they don't tell you which cycle point is ED. You'd have to *assume* "the max
is ED," and that assumption's relationship to true ED varies by slice level and method. **This is not
hypothetical — doc 34's 106° scatter came from exactly this** (fetal_cmr_4d's intra-slice Fourier
phases used as the anchor). LV blood-pool **area** is the *only* surrogate that is **ED by
construction** (max cavity = max fill = ED, where the LV is a clean single cavity). That is why
Åkesson — the one paper doing single-orientation SAX RTFB retrospective sync — uses it, and why it
is what breaks the scatter that the signal methods provably cannot.

⇒ **Split the two jobs:** x-f Fourier for the **period** (robust everywhere), LV-area for the **ED
anchor** (anatomical, breaks the scatter).

## 4. The on-point precedent: Åkesson et al. 2025

**Åkesson et al., "Retrospectively synchronized time-resolved ventricular cine images from 2D
real-time exercise cardiac MRI," *Clin Physiol Funct Imaging* 2025** (doi 10.1111/cpf.70027; open
mirror PMC12406293). It is almost exactly our task: **ECG-free, image-based, retrospective
synchronization of 2D RTFB short-axis cine, per-slice.** Their recipe, verbatim:

- **Cardiac gating:** (1) segment LV with a CNN (**Bai et al. 2018** architecture, trained on their
  internal rest dataset, Berggren 2020); (2) LV **cross-sectional area** per frame; (3) **ED =
  timeframes with larger LV area than their 3 nearest frames each direction**; **ES = area minimum
  between two EDs.**
- **Respiratory gating (separate):** diaphragm ROI → **manifold learning** reduces the 2D ROI to a
  1D respiratory-position curve (validated in Edlund 2022) → user manually picks the end-expiratory
  peak with the most timeframes near it.
- **Cross-slice sync:** *"aligning the identified ED timeframes across slice positions,"* then
  **temporally down-sample systole and diastole *separately*** (MATLAB bilinear interp) to a common
  per-interval frame count = the minimum across slices. Then **stack** (no super-resolution).

**Code availability — verbatim:** *"The data that support the findings of this study are available
on request from the corresponding author. The data are not publicly available due to privacy or
ethical restrictions."* Software is a **MATLAB plugin for *Segment*** (Heiberg lab, free-for-
research); the CNN weights are **private**; **no GitHub, no released source.** → cite-and-reimplement
reference, not a drop-in. And **Åkesson does no SVR** — she resamples-and-stacks at native 10 mm
through-plane; her separate systole/diastole resampling substitutes for a PSF engine she lacks.

**We swap the segmenter:** their Bai-2018 net (private) → our **nnU-Net Task114 (M&Ms)**, public and
in-distribution for SAX, already wired in `tools/detect_ood_ed_*` and validated in docs/15. Same
algorithm (segment LV → area curve → ED = area max), better+public net.

## 5. Complex vs magnitude — magnitude is the defensible default

The only source deriving a cardiac signal from **phase** is Seo et al., "Self-gated cardiac cine
imaging using phase information," MRM 2017 (doi 10.1002/mrm.26204). But it uses a **velocity-encoded
projection navigator** (SI bipolar-gradient-amplified aortic projection phase; LR abdominal
projection phase for respiratory) — **not the phase of a reconstructed cine image.** So it does
**not** show that the phase channel of our already-reconstructed SAX cine adds cardiac information
over magnitude. Every established image-based method (Larson, XD-GRASP, manifold learning,
fetal_cmr_4d x-f, LV-area) is **magnitude-based**.

⇒ **Use magnitude for the primary gating signal; treat complex/phase as an optional exploratory
add-on, not the baseline.** (Open question, §8.)

## 6. Base/apex: a physical limit, not a method choice

LV-area's logic — biggest cavity = maximally filled = ED — holds **only where the slice cuts a clean
single LV cavity** (the mid-ventricle). At the extreme slices it breaks for **physical** reasons:

- **Base:** the base plane **descends ~1–2 cm toward the apex in systole** (atrioventricular-plane
  descent; apex fixed, base moves). A *fixed* basal slice sees different anatomy pass through it over
  the cycle (LV cavity ↔ left atrium / LVOT), so the measured area tracks **through-plane motion,
  not filling** → max-area ≠ ED.
- **Apex:** the cavity is a few pixels, often pinched shut at ES → signal near the noise floor →
  max unreliable.

**No researched method fixes this** — the anatomy genuinely isn't there to segment, and the signal
methods (Fourier/manifold) are contaminated by the same through-plane motion *and* still can't
anchor ED. Åkesson admits it: *"did not function well at basal and apical levels… due to through-
plane motion."* This is the **field's ceiling**, not our failure. Handling, in order:

1. **Down-weight** — SVRTK's robust statistics (doc 34 §2 E, `p_jk`) reject mis-synced base/apex
   frames; the coherent mid slices (where the LV volume / EF signal lives) dominate the solve.
2. **Optional calibrated fallback** — where a base/apex anchor is still wanted: use a signal method
   (Fourier/manifold) *there*, calibrating its "which extreme = ED" offset against LV-area on the
   mid slices (where both are available). Approximate but grounded.
3. **Accept the residual** — base/apex contribute little to EF and are through-plane-blurry
   single-orientation anyway (doc 34).

**Why we can't just propagate the good mid-slice clock to base/apex:** each slice is a *separate*
ungated acquisition with *no shared trigger*, and there is *no time-overlap* between slice windows to
count the integer beats across the gap — so the absolute inter-slice phase offset is genuinely
unrecoverable from timing. That is *why* each slice needs its own anchor, and why base/apex can't
inherit the mid clock for free.

## 7. The one open design choice: linear θ vs two-anchor (a 2-agent debate)

**Question:** after ED-anchoring, do we also need Åkesson's separate systole/diastole normalization?
Two subagents debated opposing sides; both **independently converged on the same falsifier and the
same ~3 EF-point threshold** — the tell that this is settleable by one measurement.

**Code-verified common ground:**
- The faithful SVRTK pipeline (van Amerom) computes **linear** per-slice phase:
  `cardsync_intraslice.m` → `θ = 2π·frac((t − t_trigger)/RR)`. No systole/diastole warp anywhere.
- Its interslice sync is a **single rigid scalar offset per slice** (`thetaOffset`/`fouriershift`),
  not a nonlinear warp.
- **Åkesson resamples images only because she has no PSF engine.** For SVRTK you would **never**
  resample images — if you need ES alignment you encode it as piecewise-linear phase **labels**.
  ⇒ **The literal answer to "do we need the MATLAB bilinear systole/diastole resampling?" is NO** —
  SVRTK's temporal PSF (sinc×Tukey) + per-slice `-rrintervals` replaces it.

**The genuine disagreement — does linear θ let ES drift?**
- **Physics (Debater B, correct and unrefuted):** systole is ~HR-invariant (~330 ms); diastole
  absorbs R-R variation. So per-slice RR scaling is a **uniform stretch that preserves each slice's
  systolic fraction — it never equalizes them.** With linear θ, ED aligns but ES lands at a
  slice-specific fraction. Over a 55–90 bpm cross-slice spread the ES-phase scatter could reach
  ~68° ≫ one PSF bin (~14.4°), biasing ES/EF (our headline metric).
- **Debater A's strongest point + B's decisive counter:** A argued "the faithful pipeline uses
  linear θ, so it's fine." B correctly countered: the faithful pipeline's linear θ **is rescued by
  interslice sync via spatial overlap** — the exact mechanism SAX single-orientation lacks. Remove
  that crutch and linear θ's structural ES scatter is *uncorrected*.
- **But** B's 68° assumes a wide cross-slice HR spread; within one subject's ~1-min scan the real
  spread may be ±5–10 bpm, putting the scatter under B's own 15° falsifier. **Magnitude unknown.**

**Resolution — reduce to one nearly-free measurement:**
- Both agree: measure the cross-slice **ES-phase scatter** = circular std of
  θ_ES = 2π·(t_ES − t_ED)/RR across slices under linear θ. ES = LV-area **minimum** from the **same**
  nnU-Net masks already run for ED — costs nothing extra.
- **Decision gate:**
  - scatter **< ~15°** → ship **ED-anchored linear θ** (faithful, minimal, van Amerom). Done.
  - scatter **≫ ~15°** → upgrade to **two-anchor piecewise-linear θ labels** (θ 0→π over ED→ES,
    π→2π over ES→next-ED; using detected ES), **restricted to reliable mid-slices**; then A/B recon
    test on EF (concede threshold: reproducible > 3 EF-points).
- **Default = linear θ** (respects simplicity + faithfulness; earn the two-anchor complexity only if
  the data demands it). Note the ES-scatter number is a *prerequisite* of the gater anyway, and is
  the same family as doc 34's 106° ED-scatter proof-of-work.

## 8. Recommended build + validation

**Gater (all pieces already in-repo or trivial):**
1. Per-slice R-R period `T_s` ← x-f FFT peak, adult band (`baselines/fetal_cmr_4d`, band already
   patched to `[45,110]` bpm). Robust at all slice levels. → `-rrintervals`.
2. Per-slice ED frame `f_ED,s` ← nnU-Net Task114 LV-area local max (Åkesson ±3 rule), reliable
   mid-ventricle; base/apex down-weighted (§6). → the anchor.
3. Continuous phase per frame `θ_s(f) = 2π·frac((f − f_ED,s)·dt / T_s)` (dt = 25 ms for MIITT). →
   slice-major `-cardphase`. Mean `T_s` → `-rrinterval`.
4. Feed SVRTK `reconstructCardiac` (doc 34 §3.1).

**Validation (before touching SVRTK), on one MIITT subject:**
- **Proof-of-work:** re-measure doc 34's per-slice **ED-phase scatter** after ED-anchoring — expect
  the 106° to **collapse toward < 25°** for the mid slices. If mid collapses, the sync works.
- **Decision gate:** measure the **ES-phase scatter** (§7) → picks linear vs two-anchor θ.

**Respiratory (secondary):** SVRTK treats respiration as rigid drift to *correct*, not to bin (doc
34 §2), so a respiratory *phase* is not strictly required. If wanted: Åkesson's diaphragm-ROI +
manifold-learning, or Usman's two-temporal-resolution ML.

## 9. Open questions

1. Does the phase channel of an **already-reconstructed** complex SAX cine (≠ velocity-encoded
   projection navigator) add cardiac-gating SNR over magnitude? No source tests this directly.
2. Is per-slice ED-anchor + R-R the *sufficient* substitute for SVRTK, or does `reconstructCardiac`
   still expect the cross-slice interslice alignment fetal_cmr_4d does via spatial overlap? (Test
   empirically — the scatter-collapse check answers most of this.)
3. What adult cardiac band / HR prior best replaces the fetal (110–180 bpm) default, and how
   sensitive is x-f peak-picking to free-breathing respiratory harmonics near the cardiac
   fundamental? (doc 34 §6: 3–4 MIITT slices are respiratory-drift-dominated → need a cardiac
   bandpass first.)
4. The linear-vs-two-anchor gate (§7) — resolved by measurement, not yet measured.

## 10. Implementation (2026-07-05, in progress)

### 10.1 Architecture — what runs single-orientation, and where LV-seg slots in
fetal_cmr_4d's full 5-stage pipeline (A acquire → B static MC → C cardiac sync → D dynamic MC →
E 4D recon) **does not run single-orientation** — 3 of 5 stages segfault or are inapplicable without
multi-orientation overlap (doc 34 §5: stack-stack reg, staged static MC, **inter-slice sync**). What
survives *is* **self-gate → `reconstructCardiac`** (the latter fuses D+E via its internal
register↔reconstruct iterations). So "following fetal_cmr_4d" single-orientation **is** "self-gate →
SVRTK once" — they are the same pipeline, not two tiers.

Our nnU-Net LV-area does **inter-slice sync's job** (make all slices agree on a common cardiac clock
= a per-slice phase **offset**) by a different route: a per-slice **absolute ED anchor** (no spatial
overlap needed). In the code the offset was hardcoded **identity** (→ 106° scatter); we replace it
with the LV-area-ED offset. Data flow:
```
per slice:  x-f FFT (authors, MATLAB, already run) → R-R period T_s  +  intra-slice phase θ_intra (θ=0 at frame 0)
            nnU-Net Task114 LV-area (NEW)          → ED frame f_ED,s
            re-anchor: θ_new = (θ_intra − circmean{θ_intra at ED frames}) mod 2π   (θ=0 at ED)
            → slice-major -cardphase + -rrintervals → reconstructCardiac (authors' engine + params)
```
The re-anchoring is **pure arithmetic on the authors' own x-f phases** — it keeps their R-R and
their per-frame phase verbatim, changing only the per-slice constant offset. The MATLAB cardphase is
**wrapped to [0,2π]** (verified: sawtooths at the 49.5-frame cycle), so re-anchoring wraps too.

### 10.2 Parameter provenance — "same as the authors" (user directive)
`reconstructCardiac` runs at the authors' `recon_cine_vol.bash` values **verbatim** (reverting the
speed cuts flagged in `DEVIATIONS.md §C`):

| Param | Authors / ours | Note |
|---|---|---|
| `-resolution` | **1.25 mm** | was run 1.5 for speed — reverted |
| `-iterations` | **4** | author default |
| `-rec_iterations` | **10** | was 7 — reverted |
| `-rec_iterations_last` | **20** | was 12 — reverted |
| robust statistics | **ON** (no `-no_robust_statistics`) | was OFF — reverted |
| `-numcardphase` | **25** | author default (our gated GT has 30; output length is chosen, not data-derived, doc 34 §3.1) |
| recon mask | tight heart-derived `mask_cine_vol` | not the oversized chest mask |

**Two FORCED gating deviations remain** (unavoidable, doc 34/35, `DEVIATIONS.md §A`): adult HR band
`[45,110]` bpm (A1 — fetal `[105,180]` would miss the adult beat) and the **LV-area ED anchor
replacing inter-slice sync** (A6 — inter-slice sync is mathematically impossible single-orientation).
These are the doc-34/35 findings, not tuning choices.

### 10.3 The baseline roster the self-gater feeds
The self-gater is **shared infrastructure** — every SVR baseline needs cardiac phase labels (the 4D
engine takes continuous phase; the 3D-per-phase engines take hard bins derived from the same phase):

| # | Baseline | 3D/4D | Temporal coupling | Engine |
|---|---|---|---|---|
| 1 | **SVRTK `reconstructCardiac`** | 4D | ✅ temporal PSF (joint) | SVRTK — **primary, building now** |
| 2 | SVRTK `reconstruct` per-phase | 3D+t | ❌ | SVRTK |
| 3 | NiftyMIC per-phase | 3D+t | ❌ | NiftyMIC (doc 29, clean→self-gated) |
| 4 | NeSVoR per-phase | 3D+t | ❌ | NeSVoR (doc 32, clean→self-gated) |
| 5 | Stack + through-plane interp | 3D+t | ❌ | none (floor, doc 31) |

**SVRTK vs NiftyMIC/NeSVoR:** SVRTK is the only one with a **native 4D cardiac** solve
(`reconstructCardiac`, temporal PSF couples phases); NiftyMIC and NeSVoR are **static 3D** → cardiac
only via per-phase hard-binning. #1-vs-#2/3/4 is a clean **4D-joint vs 3D-per-phase** ablation, and
self-gating upgrades the existing NiftyMIC/NeSVoR baselines from clean-gated to real free-breathing.

**Deliberately excluded:** SVRTK `reconstructFFD` (deformable/non-rigid) — corrects non-rigid
*spatial* motion, NOT cardiac phase. Cardiac motion is **resolved onto the temporal/phase axis**
(you keep the phases); using deformable to "correct cardiac motion" would warp phases onto each other
and *collapse* the cine — wrong goal, and non-standard for these tools (doc 34 §2: cardiac=resolved,
respiratory/bulk=rigid-corrected). `reconstructCardiacVelocity` also excluded (4D-flow/velocity only,
useless for magnitude, doc 30).

### 10.4 Code
- `baselines/fetal_cmr_4d/selfgate_lvarea_extract.py` — RT stack → all (slice,frame) nnU-Net Task114
  inputs on the canonical grid (all Z, unlike `tools/detect_ood_ed_extract.py`'s mid-slice-only).
- nnU-Net: `nnUNet_predict -t 114 -m 2d -tr nnUNetTrainerV2_MMS` in the `nnunet` env.
- `baselines/fetal_cmr_4d/selfgate_lvarea_assemble.py` — segs → per-slice LV-area → ED/ES (Åkesson
  ±3 rule) → re-anchor authors' phases → `cardphases_lvanchor_cardsync.txt` + scatter metrics + plot.

### 10.5 Gating validation result (Volunteer1, 2026-07-05)
nnU-Net Task114 segmented all 13×180 slices; per-slice LV-area → ED/ES → re-anchor. Findings:
- **Base/apex (z0,1,11,12): no LV at all** (area ≈ 0) → nnU-Net finds no cavity → excluded. This is
  the predicted **physical** limit (doc 35 §6), not an algorithm failure. **9/13 slices anchored.**
- **z9,z10: low LV coverage** (0.42, 0.16 — segmentation dropout on the apex side) → anchored but
  excluded from the metric. **7/13 high-confidence** (z2–z8, 100% coverage).
- **Naive local-maxima peak-picking over-fired** (8–9 "EDs" vs ~4 beats) → ES scatter a spurious
  99.8°. **Confirms doc 34/35's "needs robust detection, not naive peak-picking."** Fixed with
  savgol smoothing + `find_peaks(distance ≈ 0.6·R-R, prominence)` → ~one ED/beat.
- **Result (high-confidence set):** ED scatter **BEFORE (frame-0 anchored) = 75°** (reproduces
  doc-34's ~106° desync — the detector sees it); after LV-anchoring the ED sits at θ=0 and the
  **independent ES landmark lands consistently: ES scatter = 15.6°**, systolic fraction
  **0.319 ± 0.037** (physiological, tight).
- **Decision-gate outcome (§7):** ES scatter 15.6° ≈ the 15° threshold ⇒ **linear θ is justified**
  (systolic-fraction std 0.037 → physiological ES scatter ≈ 13°). Also matches "same as authors"
  (van Amerom uses linear phase). **No two-anchor needed.**

⇒ **Self-gating validated:** the desync the naive pipeline suffers (75–106°) collapses once each
slice's own ED anchors θ=0, on the slices where the LV is segmentable (mid-ventricle); base/apex are
the acknowledged physical ceiling, handled by exclusion + SVRTK robust stats.

**Confirmed on Volunteer2 (2026-07-05):** same pattern, even cleaner — ED scatter BEFORE 121.7° →
ES scatter AFTER **6.7°** (systolic fraction 0.457 ± 0.019; 9/13 anchored, 7/13 high-conf). Two
subjects now: V1 15.6°, V2 6.7° — the collapse is consistent, not a one-off. Both linear-θ (gate met).

Outputs: `<recon>/Volunteer1/cardsync/{cardphases_lvanchor_cardsync.txt,selfgate_lvarea.json}`,
`<recon>/Volunteer1/selfgate_lvarea.png`.

### 10.6 Recon ROI mask — segmentation is the WRONG tool for RTFB (2026-07-05)
The recon is confined to the `-mask` ROI, so it must (a) **fully cover the heart on all slices** and
(b) be **identical across all baselines** (else tools reconstruct different regions → metrics not
comparable — a first-class protocol rule). Key finding: **a whole-heart *segmentation* model is the
wrong tool here.**
- Whole-heart labeling (atria + vessels + chambers) is inherently a **3D** task (TotalSegmentator-MR,
  MM-WHS are 3D, near-isotropic). Our RT SAX slices are **2D, 10 mm-thick, highly anisotropic** → 3D
  models fit poorly. The 2D SAX models that *do* work (nnU-Net Task114, `-m 2d`) are **ventricle-only
  and fail at base/apex.** There is **no good 2D whole-heart SAX model** (SAX cine isn't
  whole-heart-labeled clinically).
- **Empirical check:** the LV+MYO+RV **union** is EMPTY on 4 base/apex slices (z0,1,11,12) and
  ventricle-only → a *worse* ROI than the generous spectral auto-mask, which covers all 12
  heart-bearing slices.
- **Decision:** the recon ROI is a **generous non-segmentation region** — the authors used a manual
  heart ROI; our spectral cardiac-band auto-mask (`s01_mask_heart`, doc 34 §B1) is the auto-analog.
  Use ONE such mask per subject, **identical across all baselines**, scored within it (`_anat`
  convention). nnU-Net stays reserved for gating + EF (precise, 2D, ventricles). QC (`qc_heartmask.png`)
  shows the raw spectral mask is a bit noisy (a spurious lower blob z1–z5) → a largest-CC + fill +
  modest-dilation cleanup is the refinement for the final comparison (not needed for the test run).

### 10.7 Memory / compute reality (doc 34 §7 confirmed)
The faithful author-param run (`-resolution 1.25`, robust ON, 25 phases) **OOM-kills at `mem=32G`**
(exit -9) — reconstructCardiac memory scales ~cubically with 1/resolution. My interactive alloc is
`mem=32G` (doc 34 §7's exact cap). ⇒ **the faithful 1.25 mm run must go to a fresh `standard` job with
`--mem≥128G`** (deferred). For the **end-to-end test run** (user-sanctioned to tune speed/mem knobs,
not fidelity), a **coarse `-resolution 3.0`, `-iterations 3`, `-rec_iterations 5/8`** (robust ON,
temporal PSF, LV-anchored cardphase, tight mask all UNCHANGED) fits 32 G and demonstrates the
pipeline end-to-end. `selfgate_cine_test/` = coarse test; `selfgate_cine/` reserved for the faithful run.

### 10.8 End-to-end demonstration — IT WORKS (Volunteer1, coarse 3 mm test, 2026-07-05)
The full pipeline ran end-to-end: **LV-area self-gating → author-logic `reconstructCardiac` →
coherent 4D cine.** Output `selfgate_cine_test/cine.nii.gz` = `(66,43,40,25)` isotropic 3 mm × 25
cardiac phases.
- **Qualitative (`vis_sax_montage.png`, `vis_sax.gif`):** a clean, recognizable LV cavity +
  myocardium, **consistent across all 25 phases** — a coherent beating heart, NOT the phase-scrambled
  blur the un-synced (106°) pipeline produces. Self-gating visibly worked.
- **Quantitative — LV volume vs phase (nnU-Net Task114, `recon_ef.png`):** a **smooth textbook
  cardiac cycle** — systolic descent to ES at phase 10, diastolic recovery to ED at phase 23 (ED near
  the θ=0 anchor ✓; systolic fraction ≈0.40, consistent with the gater's 0.32). **EF = 20.6%**,
  temporal contrast 0.206.
- **The low EF (20.6% vs a true ~60%) is the expected single-orientation attenuation, not a failure:**
  it matches doc 34's prior observation (temporal contrast ~0.19–0.21 recon vs 0.38 gated GT), and is
  compounded here by the coarse 3 mm test resolution + temporal-PSF soft-binning. **This is precisely
  the limitation the project thesis targets** — classical single-orientation SVR recovers *coherent
  but amplitude-attenuated* contraction; VGGT's learned prior is meant to recover the amplitude
  (docs 24/25/33). So this baseline result *supports* the narrative.

⇒ **Demonstration complete:** self-gating (validated §10.5) + `reconstructCardiac` (author logic/params
§10.2) produces a coherent, correctly-timed beating-heart 4D cine from single-orientation SAX RTFB.

Code: `baselines/fetal_cmr_4d/{selfgate_lvarea_extract,selfgate_lvarea_assemble,run_selfgate_recon,
recon_ef,recon_ef_pick}.py|sh`. Faithful run: `sbatch/selfgate_recon_faithful.sh` (1.25 mm, 128 G).

**Status:** self-gating + SVRTK demonstrated end-to-end on V1 (coarse test). Remaining:
(1) faithful 1.25 mm author-param run — sbatch ready, **blocked by the account job-submission limit**,
submit when a `standard` slot frees; (2) scale to more volunteers; (3) the other roster baselines
(§10.3) fed by the same gater; (4) mask refinement (manual/verified ROI) for the final comparison.

## 11. Operational profile (for the VGGT baseline comparison)

Measured on the **faithful author-param run** (`reconstructCardiac`, 1.25 mm, iterations 4,
rec_iterations 10, rec_iterations_last 20, robust ON, temporal PSF), Volunteer1, 2026-07-05.

| Aspect | Value |
|---|---|
| Hardware | 1 CPU node (gl3348), **4 cores** (CPU-only; no GPU path — SVRTK is CPU) |
| CPU utilization | ~361% (≈ all 4 cores; OpenMP-parallel) |
| Input | 1 SAX stack, **13 slice-locations × 180 real-time frames = 2340 images** |
| Frame dt | 25 ms (MIITT RT) |
| Output | **25 cardiac phases**, isotropic **1.25 mm**, heart ROI **651 cc** |
| **Wall time / motion-correction iteration** | **~58 min** (at rec_iterations=10); the final iteration (rec_iterations_last=20) ≈ 2× |
| **Wall time / subject (total)** | **~4.5–5 h** on 4 cores (4 iterations + final SR pass) |
| **Peak memory** | **~33 GB** |

**Memory correction (supersedes §10.7's estimate):** the real peak is **~33 GB**, NOT the 80–150 GB
extrapolated from the coarse probe (cubic-in-resolution scaling badly over-predicts here — the tight
651 cc mask bounds the reconstructed voxels). The 32 GB interactive alloc OOM-killed because the peak
sits *just above* 32 GB. **Safe requirement: 48–64 GB.** 128 GB is overkill.

**Speed:** the bottleneck is **core count**, not memory. At 4 cores it's ~4.5 h/subject; SVRTK's
registration+SR is OpenMP-parallel, so a 16–32-core alloc would cut this several-fold. **Recommended
for future / multi-subject sweeps: ~16–32 CPU cores, 48–64 GB RAM, CPU partition** (no GPU — SVRTK is
CPU-only). RAM peak was ~33 GB; 48 GB gives margin for bigger-heart ROIs (memory scales with mask
volume). Request more cores, not more memory.

**Comparison-fairness knobs to hold fixed across the roster** (§10.3): same 25 output phases, same
1.25 mm resolution, same heart ROI mask, same self-gated cardphase — so SVRTK-4D vs SVRTK-3D+t vs
NiftyMIC vs NeSVoR differ only in the recon engine, not the problem setup.

## References
- Deep-research report (this session): 21 sources, 25/25 verified claims.
- **Åkesson et al. 2025**, *Clin Physiol Funct Imaging* — doi 10.1111/cpf.70027; PMC12406293 (the
  on-point SAX RTFB retrospective-sync method; LV-area ED anchor + per-slice ED sync; **no code**).
- **van Amerom et al. 2019**, *MRM* 82:1055 — fetal_cmr_4d (x-f per-slice gating; the SVRTK
  pipeline). Local code `baselines/fetal_cmr_4d/matlab/cardsync_*`; doc 34.
- Usman et al. 2015, *JMRI* 41:1521 (PMID 25124545) — manifold learning, dual cardiac+respiratory.
- Larson et al. 2004, *MRM* (PMC2396326) — seminal self-gating (k-space echo peak).
- Rosenzweig et al. 2020, *IEEE TMI* (PMID 32275585) — SSA-FARY (BART: github.com/mrirecon/SSA-FARY).
- Feng et al., *MRM* (PMC4583338) — XD-GRASP frequency-band separation.
- von Kleist et al. 2021, *Magn Reson Imaging* (PMID 33905835) — BSS/ICA.
- Seo et al. 2017, *MRM* — doi 10.1002/mrm.26204 (phase-based; projection navigator, not image phase).
- doc 34 — fetal_cmr_4d methodology + single-orientation failure (106° scatter); doc 15 — nnU-Net
  Task114 EF; docs 24/25 — EF theme; doc 23 — MIITT (SAX-only).
