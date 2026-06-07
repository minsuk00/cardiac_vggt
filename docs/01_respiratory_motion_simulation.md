# 01 — Respiratory motion simulation for cardiac MRI

> **TL;DR & takeaway** *(human-facing; the rest of this doc is the detailed record for agents)*
>
> **The idea is valid and literature-standard.** We can take a breath-hold cardiac-phase volume
> and apply a parametric transform to put the heart where breathing would, then reconstruct back
> to a breath-hold reference. Two rules make it correct, not hand-wavy:
> 1. **Decouple the clocks** — sample a respiratory phase *per input slice, independently of
>    cardiac phase* (this is exactly how the standard XCAT phantom works).
> 2. **Deform-then-reslice** — translate/deform the whole 3D volume, *then* re-slice at the fixed
>    slice positions, so a slice correctly images different anatomy as the heart moves.
>
> **Model:** rigid superior-inferior translation **~10–15 mm** (heart tracks the diaphragm at
> ~0.6×), smaller front-back component, end-of-exhale as the rest position. That's 1–2.5× our
> 8 mm slice spacing — i.e. breathing is a real *multi-slice* shift, not a tiny nudge.
> **Status: not implemented** — this is the research + design that justifies building it.
> **Open decision before coding:** should the model *correct* breathing back to a fixed reference,
> or be *told* the breathing phase? **Reference code worth reusing:** NeSVoR/SVRTK (slice-to-volume
> reconstruction), MRXCAT / LGE_CMRI_Simulation (cardiac motion phantoms).

**Date:** 2026-06-07
**Status:** Research + design scoping. **Not implemented.** No code written yet.
**Goal:** decide whether (and how) to simulate breathing motion on top of our breath-hold
gated cine, to push the scattered-slice reconstruction pipeline toward real-time
free-breathing transfer (the headline direction in `CLAUDE.md → Future enhancements`).

---

## 1. The question

We currently train on **breath-hold gated cine**: per subject, 12 cardiac-phase 3D volumes
(`3d_recon/sax_frame_{00..11}.nii.gz`) with **no respiratory motion** (each acquired in a held
breath). The reconstruction target is the same breath-hold volume at a queried cardiac phase.

Real-time free-breathing acquisition — the regime we ultimately want to generalize to — adds
respiratory motion: every acquired slice is captured at a different point in the breathing
cycle, so the heart sits at a different position. The proposal under evaluation:

> Take a (breath-hold) phase volume and apply a parametric geometric transform that places the
> heart where breathing would have put it. Sample the respiratory state freely (e.g. one
> simulation puts a slice at mid-inspiration, another at end-expiration), per a formula.
> **Is this valid?**

**Short answer from this research: yes, it is a literature-standard approach**, with two
design constraints that must be respected (see §6).

---

## 2. Process / methodology

Two parallel investigations on 2026-06-07:

1. **Data exploration** (this repo, `/scratch/data/.../Cine_combined`) — inspected on-disk NIfTI
   shapes/affines/intensities, the monai canonical-grid preprocess (`training/data/preprocess.py`),
   and the dataset sampler (`training/data/datasets/mri_dataset.py`) to pin down exactly what a
   respiratory transform would operate on. Findings in §3.

2. **Deep web research** — ran the `deep-research` multi-agent workflow (run id
   `wf_d9a713b5-f5d`): 5 search angles → 20 sources fetched → 88 candidate claims extracted →
   **25 claims adversarially verified by 3-vote panels (2/3 refutes required to kill) → 25
   confirmed, 0 refuted.** Angles: (a) parametric heart respiratory-motion models & numbers,
   (b) respiratory waveform models, (c) digital cardiac-respiratory phantoms, (d) simulating
   free-breathing slices for DL super-resolution / slice-to-volume reconstruction (SVR),
   (e) open-source code. Findings in §4–5. Full source list in §8.

**Confidence note:** the numbers below are verified against primary sources, but a few carry
caveats — the 0.6 tracking factor is a *population mean* (subject range 0.31–1.2); the `cos^{2n}`
waveform citation our sweep verified (Eiben 2020) is a *lung* XCAT application (the waveform is
general, the canonical cardiac origin is Lujan); and the left-atrium component magnitudes were
the only claim with a split panel vote (2-1, still confirmed).

---

## 3. What our data actually is (the substrate for any simulation)

### On-disk (per subject)
```
Cine_combined/Train_P###/sax/
├── 3d_recon/sax_frame_{00..11}.nii.gz   ← USED: 12 cardiac-phase 3D SAX volumes
├── 4d_recon.nii.gz                       ← same, stacked (X,Y,Z,T)=(256,246,11,12)
├── cine_sax.mat                          ← complex/raw CMRxRecon source
├── cine_sax_info.csv                     ← acquisition params
└── dvf_elastix/ , dvf_carmen/            ← legacy DVFs+masks (archived, unused)
```
- **Per-phase volume:** native `(256, H∈{162,204,246}, Z∈{6..14})`, `float32` **magnitude**.
  Intensity is a tiny-magnitude normalized recon (max ~1.3e-3); the preprocess fixes scale via
  0.5/99.9 percentiles, so absolute magnitude is irrelevant.
- **Spacing:** `(~1.34, ~1.40, 8.0)` mm. **Affine is diagonal / axis-aligned** (LPS, X negative):
  the recon was already reoriented to an axis-aligned grid, so **the true oblique SAX orientation
  in patient space is baked out**. Canonical Z = the short-axis stack direction (≈ SAX normal).
- **Cardiac phases:** native `TemporalPhase=26` resampled to a fixed **12 phases**.
  bSSFP, TR 47.9 ms, TE 1.5 ms, FA 44°, 2.9 T, `SliceThickness=8`.
- 301 subjects; 240/30/31 train/val/test via `training/splits/random_8_1_1.txt`.

### Canonical cube (what the model consumes)
`preprocess.py` resamples every subject to one fixed cube: spacing `(1.4, 1.4, 8.0)` mm,
shape `(256, 256, 12)` voxels (358.4 × 358.4 × 96 mm), geometric-center aligned, percentile-
normalized. Output per subject: `phases (T=12, 256, 256, 12) float16` + `content_mask`
(1 = native FOV, 0 = zero-pad). `mri_dataset.get_data` permutes to splat order
`(T, D=12, H=256, W=256)` and samples S ≤ 12 input slices, each at an `(t, z)` pair,
bilinear-upsampled to 518² for DINOv2.

### Three facts that shape the respiratory design
1. **No true SI axis is recoverable** (affine is axis-aligned). Practically: model SI motion as
   **translation along canonical Z** (the SAX normal — the dominant through-plane direction), a
   well-justified proxy.
2. **A per-slot scalar-embedder pattern already exists** (`z_indices`, `t_indices` → Fourier
   embedders). A **respiratory-phase embedder** drops in identically — one more per-slot scalar
   `r ∈ [0,1)`.
3. **Decoupled-target-query plumbing already half-exists** — `get_data` already emits
   `target_t_indices` (broadcast to every slot); same mechanism would query "reconstruct at the
   end-expiration reference."

---

## 4. Research findings — how cardiac respiratory motion is modeled

### 4.1 Parametric motion magnitudes (verified)

| Quantity | Value | Source |
|---|---|---|
| Heart bulk SI respiratory translation | **~10–15 mm typical; can exceed 20 mm** at deep breaths | Shechter 2004; Faranesh 2013 |
| RCA respiratory vs cardiac displacement | 5.0 mm (resp) vs 14.4 mm (cardiac) | Shechter 2004 |
| Left-atrium components | **SI 16.5 mm, AP 5.8 mm, LR 3.1 mm** (2-1 panel vote) | Moghari/Faranesh 2013 |
| Heart-to-diaphragm tracking factor | **mean 0.6; subject range 0.31–1.2** | Wang 1995 |

**Direction:** SI-dominated, with a smaller AP component (~0.35× SI) and minor LR / rotation.

**Two amplitude regimes to distinguish:** (a) *bulk* respiratory displacement of the heart
(10–20 mm SI, above) — the position the heart sits at during a free-breathing acquisition; vs.
(b) *residual inter-slice misalignment* after gating/correction — a cardiac misalignment-
simulation paper fit this as **Gaussian, mean 2.3 mm / std 0.87 mm** (from UK Biobank), and also
simulated 0× and 4× that. (a) is what we add per-slice; (b) is the smaller residual a correction
model is ultimately judged on.

**Model complexity (rigid → affine → deformable):**
- **Rigid translation** (SI + smaller AP) captures the bulk; what most navigator methods assume.
- **Affine** better fits the right-coronary region; **rigid** suffices for the left tree
  (Shechter 2004) — different heart regions warrant different model orders.
- **Deformable / DVF** is the high-fidelity end (subject-specific 3D-motion ← 1D surrogate).

**Implication for us:** SI amplitude (10–20 mm) is **1–2.5× our 8 mm Z spacing** → respiratory
motion is a genuine *multi-slice through-plane* shift in the canonical cube, not a sub-voxel
nudge. Rigid SI(+AP) translation is a defensible first model; affine is the documented upgrade.

### 4.2 Respiratory waveform models (verified)

- **Lujan model:** `z(t) = z0 − A·cos^{2n}(πt/τ − φ)`. The `cos^{2n}` (n≈2–3) makes the trace
  dwell **longer at end-expiration** (the rest/reference position). Standard analytic breathing
  curve. (Verified cite Eiben 2020 is lung-XCAT; canonical cardiac origin = Lujan 1999/2003.)
- **Hysteresis** (inspiration path ≠ expiration path — the heart traces a loop):
  - **Elliptical / subject-specific** — Burger & Meintjes 2013, *MRM* (relates 3D heart motion
    to the SI surrogate with an explicit inspiration≠expiration loop).
  - **Fractional-polynomial** — PROCO 2019.
- **End-expiration = reference rest state** (most reproducible; breath-holds are acquired here).
  → the natural canonical *target* respiratory state for a motion-correction model.

### 4.3 Digital phantoms that couple cardiac + respiratory motion (verified)

- **4D-XCAT** (Segars 2010, *Med Phys*) — reference cardiac-respiratory phantom. **Two
  independent time curves** (cardiac, respiratory) explicitly coupled; default tidal breath =
  **5 s (2 s in / 3 s out)**; diaphragm drives heart displacement. *This independent-two-clocks
  design is exactly the cardiac ⊥ respiratory decoupling we need.*
- **MRXCAT** (Wissmann 2014, *JCMR*) — wraps XCAT into **MR cine + perfusion** numerical
  phantoms with cardiac + respiratory motion and realistic MR contrast/coils. Code is **email-
  gated** (request a download link from the authors at `biomed.ee.ethz.ch/mrxcat.html`, MATLAB,
  non-commercial). Closest thing to "synthetic CMRxRecon with breathing." The
  `sinaamirrajab/LGE_CMRI_Simulation` repo (§5) is a public MRXCAT *extension* that already
  simulates cardiac slice-misalignment artifacts.
- **STINR-MR** (arXiv 2308.09771) — simulates respiratory motion from 4D-XCAT, represents it as
  **Elastix DVFs reduced by PCA** to a low-dim respiratory manifold.
- **M-DIP** — learns deformation fields for cardiac-respiratory motion.

### 4.4 How SVR / DL papers simulate scattered free-breathing slices (verified)

- **Lowther 2018** (*Phys Med Biol*, PMID 29472089) — the canonical recipe: build a
  motion-including 4D-XCAT, then **extract individual slices at different respiratory positions**
  to synthesize free-breathing acquisitions for training. = per-slice respiratory-phase
  assignment + **deform-then-reslice**.
- Fetal/cardiac **SVR super-resolution** (PMC6370029, PMC10193526) — simulate scattered
  motion-corrupted slices by sampling a moving volume at slice positions, each slice frozen at
  its own motion state, then train the network to recover the static volume.

The dominant paradigm is **phantom-based** (XCAT gives ground-truth motion for free). Simulating
from *real breath-hold cine* (our setting) is less common precisely because you must *invent* the
motion — which is the gap this direction fills.

---

## 5. Open-source code

**All links below were individually fetched and verified to exist (2026-06-07).** Note: the
deep-research sweep's "open-source" angle was biased toward lung/phantom tooling and **missed
the DL slice-to-volume reconstruction toolkits** (NeSVoR/SVoRT, SVRTK), which are the closest
architectural analogs to our pipeline — added below. Treat this list as a good starting set,
**not exhaustive**.

| Repo / link | What it provides | Lang / license | Relevance |
|---|---|---|---|
| **MRXCAT** — `biomed.ee.ethz.ch/mrxcat.html` | XCAT→MR cine/perfusion phantom w/ cardiac+resp motion. **NOT a public download** — email subject "MRXCAT Download Link" to the authors. | MATLAB, non-commercial | Phantom w/ GT motion |
| **sinaamirrajab/LGE_CMRI_Simulation** | **MRXCAT extension simulating cardiac LGE with slice-misalignment (respiratory) artifacts** — directly our problem domain. | MATLAB (2018b), CC-BY-NC-ND | **High** — cardiac slice misalignment |
| **daviddmc/NeSVoR** | GPU slice-to-volume reconstruction (rigid + **deformable**) via implicit neural rep + **SVoRT** registration transformers. Fetal brain, but the DL SVR machinery is our analog. | Python/CUDA, **MIT** | **High** — DL SVR architecture |
| **SVRTK/SVRTK** (KCL) | Classical SVR + super-resolution incl. **4D whole-fetal-heart** reconstruction from motion-corrupted stacks. | C++ (MIRTK), Apache-2 | **High** — cardiac SVR reference |
| **nadeemlab/SeqX2Y** (RMSim) | DL respiratory-motion simulator: learns time-varying DVFs from 4D-CT, modulated by a 1D breathing trace. Lung/CT. | Python, non-commercial (Commons Clause) | Med — DVF-sim pattern |
| **UCL/SuPReMo** | Surrogate-Parameterised Respiratory Motion Modelling — fit/apply a respiratory model from a surrogate signal. Lung / MR-guided RT. | C++, BSD-3 | Med — motion-model fitting |
| (no public **cardiac** Lujan-2003 parametric implementation found) | — | — | — |

---

## 6. Design implications for our pipeline

The literature **validates the core idea** and pins down the design:

1. **Two independent clocks** (XCAT-style): sample respiratory phase `r` **per input slice,
   independent of cardiac phase `t`**. The user's "slide one window" framing must be pushed down
   to per-slice: scattered slices are acquired across many heartbeats/breaths, so within one
   simulated acquisition they span the *whole* respiratory cycle — not a contiguous 1 s window.
2. **First motion model:** rigid translation, **SI ~10–15 mm** (sample per-subject/breath, since
   the 0.6 factor varies 0.31–1.2), **AP ≈ 0.35 × SI**, optional small rotation. Affine /
   deformable is the documented next rung if rigid underfits.
3. **Waveform:** Lujan `cos^{2n}` with **end-expiration as `r = 0` reference**; optionally add
   elliptical hysteresis (inspiration ≠ expiration).
4. **Mechanism:** **deform-then-reslice** (Lowther 2018) — translate/deform the full 3D `phases`
   bundle, then re-slice at the fixed canonical z. This is what produces the physical
   *through-plane content change* (a fixed slice position images different anatomy as the heart
   moves). Translating 2D slices in-plane would be wrong. Natural insertion point: between the
   cached `phases` lookup and slice extraction in `get_data` (mirrors how the existing GPU aug
   already re-derives `gt_target_volume` + re-extracts slices after an affine).
5. **Reference frame:** SI ≈ translation along canonical **Z** (affine is axis-aligned, §3).

### Two build options
- **(A) Augment real CMRxRecon breath-hold cine** with the parametric respiratory transform.
  Cheap, keeps real anatomy/contrast. *= the user's original idea, now literature-backed.*
- **(B) MRXCAT phantom** for ground-truth motion + a fully controllable validation set. More
  work (MATLAB), but gives exact motion GT to *validate* (A) — i.e. confirm the model actually
  corrects a *known* motion field.

Suggested plan: implement **(A)** as the training augmentation; optionally use **(B)/MRXCAT** as
a controlled benchmark.

---

## 7. Open design questions (decide before coding)

1. **Correction vs. conditioning.** Should the model **correct** breathing back to the
   end-expiration reference (blind to `r`, robustness target) — or be **conditioned** on `r` (a
   second cyclic Fourier embedder, navigator-style known phase)? This changes both the target and
   whether we add the embedder.
2. **Target volume under motion.** If inputs carry per-slice respiratory offsets, `V_gt` stays at
   the end-expiration reference (the current breath-hold volume) for the correction framing.
3. **Amplitude distribution / hysteresis fidelity** — rigid-only first, or include hysteresis +
   small rotation from the start?
4. **Augment-real (A) vs. phantom (B)** as the primary training substrate.

---

## 8. Sources

Primary sources fetched and (where claims were drawn) 3-vote verified:

**Parametric models & numbers**
- Faranesh et al. 2013 — `pmc.ncbi.nlm.nih.gov/articles/PMC3579864/`
- Shechter et al. 2004 — `onlinelibrary.wiley.com/doi/10.1002/mrm.24502`
- `onlinelibrary.wiley.com/doi/10.1002/mrm.27681`
- `pmc.ncbi.nlm.nih.gov/articles/PMC9139421/`

**Waveform / hysteresis models**
- Lujan-type `cos^{2n}` — `sciencedirect.com/science/article/abs/pii/S036030160500711X`
- Burger & Meintjes 2013 (*MRM*) — `ncbi.nlm.nih.gov/pmc/articles/PMC4218740/`
- PROCO 2019 — `pmc.ncbi.nlm.nih.gov/articles/PMC11643223/`
- `pmc.ncbi.nlm.nih.gov/articles/PMC11014047/`

**Phantoms**
- Segars 2010 (4D-XCAT) — `pmc.ncbi.nlm.nih.gov/articles/PMC2941518/`
- Wissmann 2014 (MRXCAT) — `pmc.ncbi.nlm.nih.gov/articles/PMC4422262/` ; code `biomed.ee.ethz.ch/mrxcat.html`
- `sciencedirect.com/science/article/abs/pii/S112017971730635X`
- `iopscience.iop.org/article/10.1088/1361-6560/ab8533`
- STINR-MR — `arxiv.org/pdf/2308.09771`

**SVR / DL free-breathing-slice simulation**
- Lowther 2018 (PMID 29472089) — `pmc.ncbi.nlm.nih.gov/articles/PMC10193526/`
- `pmc.ncbi.nlm.nih.gov/articles/PMC6370029/`

**Code** (all individually fetched & confirmed to exist, 2026-06-07)
- `github.com/daviddmc/NeSVoR` (NeSVoR + SVoRT, MIT)
- `github.com/SVRTK/SVRTK` (SVR toolkit incl. 4D fetal heart, Apache-2)
- `github.com/sinaamirrajab/LGE_CMRI_Simulation` (cardiac LGE slice-misalignment sim)
- `github.com/nadeemlab/SeqX2Y` (RMSim, lung/CT)
- `github.com/UCL/SuPReMo` (surrogate respiratory motion modelling)
- MRXCAT — `biomed.ee.ethz.ch/mrxcat.html` (email-gated download)

**Follow-up search** (cardiac slice-misalignment + SVR toolkits, 2026-06-07): found the UK
Biobank inter-slice misalignment distribution (mean 2.3 / std 0.87 mm) and the NeSVoR/SVRTK
toolkits the original sweep missed.
- `ncbi.nlm.nih.gov/pmc/articles/PMC2292180/` (slice-to-volume registration for misaligned CMR)
- `sciencedirect.com/science/article/abs/pii/S0895611124000661` (end-to-end DL motion-correction + super-resolution, multi-slice CMR)

> Provenance: deep-research workflow run `wf_d9a713b5-f5d` (2026-06-07), 102 agents,
> 88 claims extracted → 25 verified (25 confirmed / 0 refuted). Caveats in §2. Open-source code
> links subsequently re-verified by direct fetch; NeSVoR/SVRTK added (sweep miss).
