# 02 — Related work: ranked literature review

> **TL;DR & takeaway** *(human-facing; the rest of this doc is the detailed record for agents)*
>
> **Nobody has published exactly what we're doing** — reconstruct a 3D heart volume at a *chosen*
> cardiac phase from genuinely *scattered single-frame-per-slice* acquisitions (one frame per
> arbitrary `(t, z)`). That's a real novelty gap. But four families of prior art surround it and
> each nails one piece of our design:
>
> 1. **INR slice-to-volume reconstruction (SVR)** — **NeSVoR** (TMI'23) and **SVoRT** (MICCAI'22)
>    are the strongest *deep-learning* analogues for "scattered 2D slices → continuous 3D volume."
>    SVoRT is a **transformer over slices with cross-slice attention** — the closest published
>    cousin to what we do with VGGT. (Both are fetal-brain *rigid* SVR, not cardiac deformable —
>    that's the gap we fill.)
> 2. **Phase-conditioned template deformation** — **Kettelkamp 5D MoCo-DL** (MICCAI'23),
>    **M-DIP** (MRM'25), **MoCo-INR** (2025) all "deform one canonical template per phase via a
>    network-predicted DVF." This is *exactly* our `world_points = scanner_coords + Δ(t)` + splat
>    paradigm. **These are the papers to cite for our method's lineage.**
> 3. **Real-time free-breathing cine** — **Subspace-INR** (Rueckert, '24), untrained
>    **Fourier-feature INR** (Heckel, '23), **diffusion spiral** (MRM'25), **zero-shot PG-DL**
>    ('23). These define the **gated→real-time domain gap** we must cross (our central transfer
>    risk), and several echo our Fourier z/t embedders.
> 4. **Classical 5D MR-physics baseline** — **XD-GRASP 5D** (MRM'18) + **free-running self-gated**
>    (MRM'19): the retrospective-sort + compressed-sensing pipeline *we are trying to replace*.
>    **DMCVR** (MICCAI'23) is a diffusion 3D-from-sparse-stacks method for the same through-plane
>    problem.
>
> **Read-first shortlist:** NeSVoR, SVoRT, Kettelkamp 5D-MoCo, M-DIP. **Method-lineage to cite:**
> the family-2 template-deformation papers. **Status:** literature sweep only; no code/design
> decision attached. Findings below passed unanimous 3-0 adversarial verification; the geometry-
> foundation and motion-estimation entries (family 5) are surfaced-but-not-deep-verified — treat as
> leads.

**Date:** 2026-06-07
**Status:** Literature sweep (deep-research harness: 6 search angles, 26 sources fetched, 112
claims extracted, 25 verified 3-0, 12 synthesized). No design decision attached yet.
**Goal:** map the prior art around our core problem — *real-time cine → few/single frames per
slice → reconstruct the volume at a target cardiac phase* — and rank it by relevance so future
work knows what to read, cite, and benchmark against.

---

## How to read the ranking

Each entry is scored on three axes the request asked for:
- **(a) Setup match** — how close to *single/few-frame-per-slice → volume at a target phase*.
- **(b) Focus** — `DL` (deep-learning method) vs `MR` (MR-physics/reconstruction) vs `DL+MR`.
- **(c) Recency/impact.**

Ranking is **conceptual proximity, not measured transfer** — no source benchmarks our exact
scattered-`(t,z)` setup, so this is "what to read and why," not "what will work."

---

## The ranked list

### Tier 1 — Direct structural analogues (slice→volume + phase-conditioned deform)

**These map onto our architecture piece-for-piece. Read and cite these first.**

#### 1. NeSVoR — Implicit Neural Representation for SVR
- **Authors / venue:** Xu et al., *IEEE TMI* 2023. (arXiv 2210.… / [PMC10287191](https://pmc.ncbi.nlm.nih.gov/articles/PMC10287191/))
- **Focus:** `DL+MR` (INR method with an explicit MR slice-acquisition forward model).
- **Summary:** Learns a *continuous, resolution-agnostic* 3D volume function from multiple
  motion-corrupted 2D slices, jointly modeling rigid inter-slice motion, point-spread function,
  and bias fields. The trained continuous field is sampled at arbitrary resolution at inference.
- **Why it matters to us:** This is the canonical "scattered 2D slices → one continuous canonical
  volume" formulation. Resolution-agnostic sampling = our "query the canonical cube at a target
  grid." The explicit slice-acquisition forward model is the principled version of our differentiable
  splat. **Main gap:** rigid (fetal-brain) motion, not cardiac deformable phase warping.

#### 2. SVoRT — Slice-to-Volume Registration Transformer
- **Authors / venue:** Xu et al., *MICCAI* 2022. ([arXiv 2206.10802](https://arxiv.org/abs/2206.10802))
- **Focus:** `DL` (transformer).
- **Summary:** Treats multiple stacks of 2D slices as a **sequence with cross-slice attention** —
  predicts each slice's slice-to-volume transform using information from the other slices — and
  *jointly* estimates the underlying 3D volume, alternating volume and transform updates.
- **Why it matters to us:** The **closest published analogue to the VGGT adaptation**: attention
  across a set of slices to infer geometry + a shared volume. The set-attention "each slice's
  prediction depends on the others" is exactly the behavior we rely on (and saw in our own frame-0
  / companion-slice analysis). **Gap:** rigid fetal SVR, no target-phase query.

#### 3. Motion-Compensated Unsupervised DL for 5D MRI (Kettelkamp et al.)
- **Authors / venue:** Kettelkamp et al., *MICCAI* 2023. ([arXiv 2309.04552](https://arxiv.org/pdf/2309.04552))
- **Focus:** `DL+MR` (unsupervised, k-space forward model).
- **Summary:** Reconstructs motion-resolved 5D cardiac MRI from 3D radial data by modeling every
  cardiac/respiratory phase as a **single deformed 3D template**, where the deformation maps are
  produced by a **CNN driven by the physiological phase**; template + deformations are jointly
  estimated.
- **Why it matters to us:** This is **our core paradigm, published.** Phase-conditioned DVF on a
  canonical template == our `t_embedder`-conditioned residual `Δ`, with `world_points =
  scanner_coords + Δ`. **The primary method-lineage citation.** **Gap:** k-space forward model +
  full 3D radial acquisition, vs our image-domain splat from sparse 2D slices.

#### 4. M-DIP — Multi-dynamic Deep Image Prior for cardiac MRI
- **Authors / venue:** Vornehm et al., *MRM* 2025. ([PMC12501714](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12501714/))
- **Focus:** `DL+MR` (instance-specific / scan-specific, trained from scratch per measurement).
- **Summary:** Unsupervised reconstruction of accelerated **real-time free-breathing** cardiac MRI
  that **factorizes dynamics** into (i) a *spatial dictionary* synthesizing a time-dependent
  intermediate "template" image and (ii) *time-dependent deformation fields* explicitly modeling
  cardiac **and** respiratory motion.
- **Why it matters to us:** Template-synthesis + explicit deformation factorization is a near-twin
  of our canonical-volume + phase-DVF design, and it operates in the **real-time free-breathing**
  regime we're targeting — including the two-clock (cardiac × respiratory) structure from our
  `docs/01`. **Gap:** per-scan optimization (no learned generalization across subjects); k-space.

#### 5. STINR-MR — Spatial-Temporal INR for dynamic cine
- **Authors / venue:** *Phys. Med. Biol.* 2023. ([arXiv 2308.09771](https://arxiv.org/abs/2308.09771))
- **Focus:** `DL+MR`.
- **Summary:** Joint reconstruction + deformable registration: a **reference frame as a spatial
  INR**, dynamics encoded via a **temporal INR + basis DVFs**, reaching <100 ms temporal / 3 mm
  spatial resolution from undersampled k-space.
- **Why it matters to us:** Same reference-template + temporal-DVF decomposition; the basis-DVF
  trick (low-dimensional motion subspace) is a candidate regularizer if our per-pixel `Δ` proves
  under-constrained. **Gap:** 3D cine from continuous k-space, not scattered 2D slices.

---

### Tier 2 — Real-time regime & the gated→real-time domain gap

**These define the *target domain* we must generalize to (our central transfer risk) and supply a
method toolbox. Less structural overlap, high strategic relevance.**

#### 6. Subspace Implicit Neural Representations for Real-Time Cardiac Cine
- **Authors / venue:** Huang, Hammernik, Küstner, Rueckert, Dec 2024. ([arXiv 2412.12742](https://arxiv.org/abs/2412.12742))
- **Focus:** `DL+MR`.
- **Summary:** Two MLPs learn **separate spatial and temporal subspace bases** exploiting cine's
  low-rank structure, enabling **continuously sampled, ungated** imaging that **eliminates
  retrospective binning** and captures beat-to-beat variation.
- **Why it matters to us:** This is the cleanest statement of the **no-retrospective-gating,
  real-time** regime — exactly the pipeline-replacement we pitch. Subspace/low-rank factorization
  is a regularization idea we could adopt.

#### 7. MoCo-INR — INR + motion-compensated reconstruction
- **Authors / venue:** Nov 2025. ([arXiv 2511.11436](https://arxiv.org/abs/2511.11436))
- **Focus:** `DL+MR` (unsupervised).
- **Summary:** Combines INR with a classical motion-compensated framework for highly undersampled
  k-t data, yielding a cardiac-motion decomposition + high-quality recon; **evaluated on
  prospective real-acquired free-breathing scans.**
- **Why it matters to us:** INR + *explicit* motion modeling on **real free-breathing** data — both
  the method blend and the real-data evaluation we ultimately need. (Self-reported eval; very recent.)

#### 8. DMCVR — Morphology-guided Diffusion for 3D Cardiac Volume Reconstruction
- **Authors / venue:** *MICCAI* 2023. ([arXiv 2308.09223](https://arxiv.org/abs/2308.09223))
- **Focus:** `DL`.
- **Summary:** Morphology-conditioned diffusion model that synthesizes dense slices from **sparse
  2D cine stacks** to produce high-resolution 3D cardiac volumes — directly targeting the
  through-plane / slice-to-volume resolution problem.
- **Why it matters to us:** A generative alternative to our splat for the 3D-from-sparse-2D-stacks
  problem; a natural baseline / ablation for "diffusion inpainting vs explicit splat." **Gap:**
  doesn't handle arbitrary scattered `(t,z)` sampling or a target-phase query.

#### 9. Untrained INR + Fourier-feature real-time cine
- **Authors / venue:** Kunz, Ruschke, Heckel, 2023/2024. ([arXiv 2305.06822](https://arxiv.org/abs/2305.06822))
- **Focus:** `DL`.
- **Summary:** Represents the beating heart with an **untrained per-scan MLP + Fourier features**
  as a signal prior; reconstructs a real-time video from highly undersampled data with **no ECG /
  biosensors / training data.**
- **Why it matters to us:** Validates **Fourier-feature embeddings** as a strong prior for cardiac
  dynamics — echoes our sinusoidal Fourier `z/t` embedders. **Gap:** 2D real-time *video*, not 3D
  volume at a target phase.

#### 10. Diffusion-based real-time free-breathing spiral cine
- **Authors / venue:** Schad et al., *MRM* 2025. ([PMC12309890](https://pmc.ncbi.nlm.nih.gov/articles/PMC12309890/))
- **Focus:** `MR+DL` (score-based/diffusion posterior sampling on real spiral data).
- **Summary:** Reconstructs highly undersampled **real-time free-breathing spiral bSSFP** (13
  spiral arms/frame, ~48 ms/frame at 1.5T), covering the whole heart in <1 min.
- **Why it matters to us:** A **concrete spec of the real-time free-breathing acquisition** we aim
  to generalize to — useful for grounding the simulated→real domain gap with real numbers
  (temporal resolution, artifact character). Mostly a reconstruction reference, not a method analogue.

#### 11. Zero-shot self-supervised PG-DL for real-time cine
- **Authors / venue:** Demirel et al., 2023. ([PMC9948950](https://pmc.ncbi.nlm.nih.gov/articles/PMC9948950/))
- **Focus:** `MR+DL`.
- **Summary:** Subject-specific **zero-shot self-supervised physics-guided DL** for highly
  accelerated real-time Cartesian cine — **no training database, no ground truth.**
- **Why it matters to us:** A pointer toward truly self-supervised real-time recon (relevant to our
  "drop `gt_target_volume`" future direction). k-space recon, not slice-to-volume.

---

### Tier 3 — Classical MR-physics baseline (the pipeline we replace)

#### 12. XD-GRASP 5D whole-heart + free-running self-gated framework
- **Authors / venue:** Feng et al., *MRM* 2018 ([5D whole-heart](https://onlinelibrary.wiley.com/doi/10.1002/mrm.27898)); Di Sopra et al., *MRM* 2019 ([free-running, self-gated](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26745)).
- **Focus:** `MR` (compressed sensing; self-gating signal extraction).
- **Summary:** Continuous non-ECG-triggered 3D radial golden-angle bSSFP, **retrospectively sorted**
  into 5D (3 spatial + cardiac + respiratory) motion-resolved volumes via a self-extracted
  respiratory signal (+ recorded ECG), reconstructed with compressed sensing. Di Sopra removes
  external gating entirely (**fully self-gated**).
- **Why it matters to us:** This **is** the retrospective-sort + SVR/CS baseline our pitch replaces,
  and the **self-gating** route is exactly the bridge for the gated→real-time gap (recover
  `(cardiac, respiratory)` per readout). Cite as the established comparator. **Gap:** no learning;
  long acquisition; the cost we're trying to remove.

---

### Tier 4 — Geometry foundation models (the VGGT lineage) & motion-estimation nets

**Surfaced by the sweep but *not* in the deep-verified set (budget-dropped or my own
identification). Treat as leads, verify titles before citing.** The synthesis explicitly flagged
sub-topic 6 (multi-view-3D geometry transformers) as under-covered.

- **VGGT — Visual Geometry Grounded Transformer.** Wang et al., *CVPR* 2025
  ([arXiv 2503.11651](https://arxiv.org/abs/2503.11651)). `DL`. **The base model this project
  adapts.** Feed-forward multi-view transformer predicting cameras + per-pixel point maps. Our
  whole approach is "VGGT point head → world points → splat," so this is the architectural parent.
- **DUSt3R — Geometric 3D Vision Made Easy.** Wang et al., *CVPR* 2024
  ([arXiv 2312.14132](https://arxiv.org/abs/2312.14132)). `DL`. Predicts dense pointmaps directly
  from uncalibrated image pairs — the conceptual ancestor of VGGT's pointmap regression; explains
  *why* a geometry transformer can map 2D views to 3D points without explicit camera calibration
  (we exploit exactly this: no real cameras, just `(z,t)` embeddings).
- **Additional geometry-transformer sources** (surfaced, unverified — likely DUSt3R/MASt3R/VGGT
  follow-ups and a medical-imaging geometry-transformer): [arXiv 2503.01661](https://arxiv.org/abs/2503.01661),
  [arXiv 2507.08448](https://arxiv.org/abs/2507.08448), [arXiv 2507.14501](https://arxiv.org/html/2507.14501v1),
  [PMC12227771](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/). Verify titles before relying on these.
- **Cardiac motion estimation / DVF registration networks** (sub-topic 5; surfaced, not
  deep-verified) — relevant for the warping/DVF half of the design and as priors/regularizers:
  [arXiv 2209.00726](https://arxiv.org/pdf/2209.00726),
  [MedIA S1361841522003103](https://www.sciencedirect.com/science/article/pii/S1361841522003103),
  [PMC11466156](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466156/),
  [Comput. Biol. Med. S0010482523004663](https://www.sciencedirect.com/science/article/abs/pii/S0010482523004663),
  [arXiv 2103.16695](https://arxiv.org/pdf/2103.16695) (likely a biomechanics-informed registration
  net), [PMC7586816](https://pmc.ncbi.nlm.nih.gov/articles/PMC7586816/). These cover learned DVF
  prediction and biomechanics-informed regularization — directly applicable if our per-pixel `Δ`
  needs smoothness/physically-plausible constraints.

---

## Synthesis: how the families map onto our pipeline

| Our component | Closest prior art | What to borrow |
|---|---|---|
| Scattered 2D slices → canonical volume | NeSVoR, SVoRT | Forward model (NeSVoR); slice-set attention (SVoRT) |
| `world_points = scanner_coords + Δ(t)` (phase-DVF on a template) | **Kettelkamp 5D-MoCo, M-DIP, STINR-MR** | The exact template-deform paradigm; basis-DVF / dictionary regularization |
| Arbitrary target-phase query (our `target_t_indices`) | Phase-conditioned deformation (family 2) | Conditioning a single template on a queried phase |
| Fourier `z/t` embedders | Heckel Fourier-feature INR, NeRF positional enc. | Fourier features as a cardiac-dynamics prior |
| Splat → V_canon | NeSVoR forward model; DMCVR (diffusion alt.) | Explicit forward model vs generative inpainting (ablation) |
| gated→real-time transfer | Subspace-INR, MoCo-INR, XD-GRASP/self-gated, diffusion-spiral | Self-gating; real free-breathing eval; ungated low-rank priors |
| VGGT backbone | VGGT, DUSt3R | Why pointmap regression works without cameras |

## Open questions (from the verification stage)

1. **Genuine novelty gap:** no verified source reconstructs a 3D cardiac volume at a *chosen* phase
   from genuinely scattered **single-frame-per-slice** `(t,z)` data. Closest are full-per-slice cine
   or continuous k-space. This is likely our publishable delta.
2. **Quantified gated→real-time transfer** is unmeasured anywhere — our central risk, and an
   evaluation nobody has reported.
3. **Splat vs INR vs k-space forward model** for volumetric cardiac recon — no head-to-head exists;
   an ablation worth running (ties to the "UNet ablation — replace the splat" future item).
4. **Geometry-transformer ↔ medical SVR bridge** is thin in the literature — adapting VGGT to slices
   is relatively unexplored territory.

## Caveats on this sweep

- All 12 Tier 1–3 findings are from primary peer-reviewed (TMI, MRM, PMB, MICCAI) or primary arXiv
  sources and passed **unanimous 3-0 adversarial verification**. Tier 4 is **surfaced-but-not-deep-
  verified** — verify exact titles/authors before citing.
- Relevance ranking is **interpretive** (conceptual proximity), not benchmarked on our setup.
- Several "real-time applicability" claims (MoCo-INR, M-DIP, subspace-INR) are the **authors' own
  evaluations** without independent replication.
- DL methods skew 2023–2025 (fast-moving — newer work may exist past the search window); the
  classical 5D MR-physics references (2017–2019) are foundational and stable.
