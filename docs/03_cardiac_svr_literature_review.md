# 03 — Classical & DL cardiac SVR literature: focused sweep

> **TL;DR & takeaway** *(human-facing; the rest of this doc is the detailed record for agents)*
>
> **The true intersection is essentially empty** — no published method, classical or DL, reconstructs
> a 3D cardiac volume at a chosen target phase from genuinely scattered single-frame-per-slice `(t,z)`
> 2D acquisitions. This is a confirmed open gap, not a search failure.
>
> The closest non-DL predecessor is **Jantsch et al. (IEEE ISBI 2013)**: assembles free-breathing
> 2D real-time SAX cine stacks across z-depths into a coherent 3D+t volume via iterative respiratory
> motion correction. It hits all four criteria except it needs *many* frames per slice (not one) and
> has no target-phase query. That's the classical baseline to cite and beat.
>
> The closest DL analog is **Chen et al. (CMIG 2024)**: end-to-end motion-correction +
> through-plane super-resolution → 3D cardiac volume from SAX stacks. Same gaps: full per-slice cine
> required, no target-phase query, no free-breathing across z.
>
> **Confirmed out-of-scope** (commonly cited but not SVR): XD-GRASP = per-slice 2D reconstruction
> (no cross-z assembly); TetHeart = mesh output (no intensity volume). **DMCVR** (from `docs/02`
> first sweep) also **did not survive** this sweep's adversarial verification (1-2 twice) — treat
> with caution.
>
> The problem statement = inter-slice respiratory misalignment (**Chandler JCMR'08**, **Tarroni
> '18**, **Dangi MICCAI'18**) + single-frame scarcity + phase conditioning. Nobody has done all
> three together.

**Date:** 2026-06-07
**Status:** Literature sweep (deep-research harness: 5 search angles, 23 sources fetched, 98
claims extracted, 25 verified, **12 confirmed / 13 killed** — 52% kill rate, more rigorous than
`docs/02` sweep). No design decision attached.
**Goal:** Fill in the classical/MR-physics cardiac SVR literature that the first sweep (`docs/02`)
missed due to a DL/INR skew. This sweep explicitly targeted the narrow intersection of
{real-time OR free-breathing} × {cardiac} × {slice-to-volume reconstruction}.

---

## The ranked list

### Tier 1 — True cardiac slice-to-volume reconstruction (the rarest class)

**These actually assemble 2D slices at different z-depths into a 3D cardiac volume.**

#### 1. Jantsch et al. — Free-breathing multi-stack 2D cine to 3D+t via iterative motion correction
- **Authors / venue:** Jantsch, Rueckert, Price, Hajnal — *IEEE ISBI* 2013.
  ([IEEE](https://ieeexplore.ieee.org/document/6556599/))
- **Focus:** `MR/Classical` — iterative non-rigid registration, no deep learning.
- **Criteria hit:** cardiac ✅, free-breathing ✅, 2D slices → 3D volume ✅, real-time cine ✅
- **Criteria missed:** needs many frames per slice (not one); no target-phase query.
- **Summary:** Takes multiple stacks of 2D real-time SAX images acquired during normal
  free-breathing, estimates respiratory deformations iteratively from the data, and registers them
  into a coherent spatiotemporal 3D+t volume — entirely classical (2013, pre-DL registration).
- **Why it matters:** This is the **only peer-reviewed confirmed paper** in either sweep that
  achieves slice-to-volume assembly across z-depths under free-breathing for cardiac cine. It is
  the **classical baseline to cite and aim to surpass** with our learning-based approach.
- **Key gap:** "Many frames per slice" requirement — needs a full cine video per z-position to
  estimate respiratory motion; does not address the single-frame-per-slice regime.

#### 2. Chen et al. — End-to-end DL motion correction + through-plane SR → 3D cardiac volume
- **Authors / venue:** Chen et al., *Computerized Medical Imaging and Graphics (CMIG)*, April 2024.
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0895611124000661),
  GitHub: CMR_MC_SR_End2End)
- **Focus:** `DL` — CNN motion correction + through-plane super-resolution, end-to-end.
- **Criteria hit:** cardiac ✅, 2D slices → 3D volume ✅
- **Criteria missed:** breath-hold stacks (not real-time/free-breathing); full per-slice cine
  required (not single-frame); same-phase inputs only (no scattered-phase inputs, no target-phase
  query — see note).
- **Summary:** Takes multi-slice SAX cine stacks, corrects inter-slice motion (reduces
  misalignment from 3.33±0.74 mm to 1.36±0.63 mm), applies through-plane super-resolution, and
  outputs a high-resolution 3D cardiac volume. Peer-reviewed, public code. *Confidence: medium
  (quantitative figures behind paywall, verified plausible).*
- **Why it matters:** The **strongest DL slice-to-3D cardiac volume baseline** in either sweep.
  The two-stage design (motion correction → super-resolution → volume) is a good ablation foil for
  our end-to-end learned approach.
- **Key gap (important):** Like all classical cardiac SVR, this method operates in the
  *same-phase* regime — input slices are ECG-gated frames pre-selected to the same cardiac phase
  before reconstruction. The "phase query" is trivial (just pick the right frame from each cine).
  Our problem is strictly harder: inputs are at *scattered, arbitrary* cardiac phases and the model
  must infer how to reconstruct at any queried target phase. This is the core novelty gap.

---

### Tier 2 — Problem definition: inter-slice misalignment (the core challenge SVR solves)

**These confirm and quantify the problem that motivates all cardiac SVR work.**

#### 3. Chandler et al. — Quantifying inter-slice misalignment in breath-hold CMR
- **Authors / venue:** Chandler et al., *JCMR* 2008. ([PMC2292180](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2292180/))
- **Focus:** `MR/Classical`.
- **Summary:** Foundational paper quantifying breath-hold inconsistency as the primary source of
  inter-slice misalignment; reports LV volume errors of 5.8±4.1 ml from misaligned stacks.
  *Note: the specific NMI registration numbers (7.01→1.56 mm) were refuted by adversarial
  verification — do not cite those figures.*
- **Why it matters:** The problem-definition reference. Our simulation using gated breath-hold
  data sidesteps this problem, but it's exactly what our model must handle in true real-time
  free-breathing cine (the target domain).

#### 4. Tarroni et al. (Rueckert group) — Classical PSM-based slice alignment
- **Authors / venue:** Tarroni et al., 2018. ([arXiv:1810.02201](https://arxiv.org/abs/1810.02201))
- **Focus:** `MR/Classical` — random decision forests + rigid registration.
- **Summary:** Generates probabilistic segmentation maps (PSMs) of the LV cavity via random
  decision forests, then rigidly registers each slice's PSM to a target PSM derived from LAX
  images. Corrects inter-slice misalignment in ~28% of subjects where it's clinically significant.
- **Why it matters:** The canonical classical (non-DL) misalignment correction method from the
  Rueckert group (who also produced Jantsch 2013). Establishes the correction layer that precedes
  any volumetric reconstruction in a classical pipeline.

#### 5. Dangi et al. — CNN-based inter-slice misalignment correction
- **Authors / venue:** Dangi et al., *MICCAI workshop* 2018. ([PMC6168009](https://pmc.ncbi.nlm.nih.gov/articles/PMC6168009/))
- **Focus:** `DL` — CNN regression for in-plane rigid misalignment prediction.
- **Summary:** CNN reduces median inter-slice misalignment from 3.13 to 2.07 pixels (p=1.617e-76)
  on 97 patients. Residual 2.07 px error is non-trivial. Earliest DL method in the confirmed set
  for cardiac inter-slice correction.
- **Why it matters:** DL counterpart to Tarroni et al.; together they bracket the pre-volume,
  correction-only literature.

#### 6. Stolt-Ansó et al. — Intensity-agreement-based 3D rigid slice alignment (no anatomy needed)
- **Authors / venue:** Stolt-Ansó, Sideri-Lampretsa, Dannecker, Rueckert — *ISBI* 2024.
  ([arXiv:2404.00767](https://arxiv.org/pdf/2404.00767))
- **Focus:** `MR/Classical`.
- **Summary:** Maximizes pairwise intensity agreement at SAX/LAX slice intersections to solve a
  3D rigid correction (rotation + translation) per slice, without segmentation or anatomical
  knowledge. Output is aligned 2D slices, **not a 3D volume**.
- **Why it matters:** State-of-art classical slice alignment (2024); the intensity-intersection
  criterion is a candidate evaluation metric for our volume pipeline's geometric consistency.
  **Explicitly not SVR** — correction step only.

---

### Tier 3 — Real-time cine regimes that generate the input data

**These characterize the acquisition modes our pipeline must eventually handle.**

#### 7. Bhat et al. — Retrospective cardiac-phase binning of real-time free-breathing 2D cine
- **Authors / venue:** Bhat et al., *JCMR* 2013. ([PMC3842803](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3842803/))
- **Focus:** `MR`.
- **Summary:** Retrospective cardiac-phase binning of ~16-20s real-time free-breathing 2D cine
  acquisitions yields 30 cardiac phases per slice at 1.3-1.8×1.8-2.1 mm², 34.3±9.1 ms temporal
  resolution. **Per-slice only — no cross-z assembly.**
- **Why it matters:** Defines the per-slice acquisition regime our simulation targets. It shows
  that real-time free-breathing cine *can* yield multi-phase per-slice data (i.e., you can get
  30 frames per slice from 16-20s of real-time), but the unresolved step is assembling those
  per-slice cines across z-depths into a 3D+t volume — exactly our problem. *Note: image quality
  non-inferiority claims were refuted; the acquisition spec figures are reliable.*

---

### Confirmed out-of-scope (commonly cited as SVR — they are not)

These surfaced as candidates but were refuted by adversarial verification. Do **not** cite as
cardiac SVR baselines.

| Paper | Why not SVR |
|---|---|
| **XD-GRASP** (Feng et al., MRM 2016) | Per-slice 2D reconstruction (each slice independently binned into cardiac × respiratory phase space from golden-angle k-space). No cross-z assembly. The "3D XD-GRASP" is abdominal DCE-MRI with kz sampling, not cardiac cine. |
| **TetHeart** (arXiv:2509.12090, 2025) | Outputs 4D deformable tetrahedral **meshes** (shape/motion), not voxel intensity volumes. Cannot benchmark with PSNR/SSIM. Out-of-scope for intensity reconstruction. |
| **DMCVR** (arXiv:2308.09223, MICCAI'23) | Listed in `docs/02` as a Tier 2 finding; **did not survive this sweep's adversarial verification** (1-2 votes twice). The claims about through-plane SR / 3D-from-sparse-stacks were disputed. **Treat with caution.** |

---

## Synthesis: what this sweep adds to docs/02

| From `docs/02` (DL/INR sweep) | From this sweep (classical cardiac SVR sweep) |
|---|---|
| NeSVoR, SVoRT — best analogues for scattered-slice→volume (fetal, rigid) | **Jantsch 2013** — only confirmed classical cardiac SVR across z-depths (free-breathing, many frames/slice) |
| Kettelkamp 5D-MoCo, M-DIP — phase-DVF template paradigm | **Chen 2024** — only confirmed DL cardiac slice→3D volume (breath-hold, full cine, no phase query) |
| Real-time cine regime (INR/subspace, diffusion) | **Bhat 2013** — acquisition spec for real-time per-slice cine; **Chandler/Tarroni/Dangi** — the misalignment problem |
| DMCVR as Tier 2 DL candidate | **DMCVR refuted** here — downgrade to uncertain |

## The confirmed open gap (from both sweeps combined)

All existing cardiac SVR methods — classical and DL — share a hidden assumption: **input slices
are pre-selected to the same cardiac phase** (via ECG gating + retrospective frame selection from
a full cine). The phase query is therefore trivial (just pick the right frame). No published method
handles inputs at *scattered, arbitrary* cardiac phases and reconstructs at a queried target phase.

No published method does all five simultaneously:
- ✅ Cardiac (not fetal/brain)
- ✅ 2D slices → 3D volume across z-depths (SVR, not per-slice reconstruction)
- ✅ Free-breathing / real-time (not breath-hold)
- ✅ Single or few frames per slice (not full cine video)
- ✅ Scattered input phases → queried target phase (not same-phase inputs)

**Jantsch 2013 misses the last three.** Chen 2024 misses the last four. Everything else misses at
least three. The same-phase assumption alone rules out essentially the entire literature.
This combination is the novelty claim.

## Caveats

- **52% adversarial kill rate** — this sweep's verifiers were more skeptical than the first sweep
  (which killed 0/25). Surviving findings are more trustworthy; `docs/02` findings should be
  read with slightly lower confidence where they overlap with refuted claims here (especially
  DMCVR).
- Chen et al. 2024 is **medium confidence** due to a paywall; the quantitative figures (3.33→1.36
  mm) are plausible but not directly verified.
- Jantsch 2013 is behind IEEE paywall; the core methodology was confirmed 2-1 and 3-0 on
  complementary claims.
- **Search window:** no classical cardiac SVR paper post-2020 was confirmed in the set — either
  the field stalled (DL took over and mostly ignored the classical formulation) or there is
  post-2020 work not indexed in the search sources. Likely some of both.
