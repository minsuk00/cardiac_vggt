# 99 — Paper Methods section: project explanation, literature sweep of methods-writing, and the two-agent debate

> **TL;DR & takeaway**
>
> **What this is.** It has three parts, all bound to the existing ICLR 2027 draft in `_paper/`:
> - a detailed, code-verified explanation of the project for a paper-writing agent (§1);
> - a sweep of how ~40 well-known ML and medical-imaging papers write their Method sections (§2);
> - a two-agent debate (medical-imaging vs vision/ML tradition) on how to write *our* Section 3,
>   starting from the problem formulation (§3), plus the ruling.
>
> **Ruling.** Section 3 is a **hybrid**. It opens with a one-sentence task statement, then a
> *two-line* image-level observation model with an explicit known/unknown split. That is the
> precedent of every SVR paper, including FC-SVR at CVPR. Then, in this order:
> - the learned per-pixel displacement + differentiable forward splat;
> - the training objective as an expectation over the simulator;
> - the simulator, written as sampling the latent variables, with a "not modelled" paragraph;
> - inference.
>
> Presentation rules:
> - **The architecture is not a contribution.** Give a short VGGT recap, then an explicit
>   reuse/change/remove diff (MASt3R pattern).
> - **No noise term** in the observation model (the simulator adds none; artefacts corrupt inputs *and*
>   target).
> - **Never write "unsupervised".** Say "supervised by gated target-state volumes; the displacement
>   field is latent".
>
> The drop-in outline is §4 and the drop-in LaTeX problem formulation is §5.
>
> **Draft-vs-code check (§6), after author review 2026-09-13.** Most flags were resolved as fine:
> - The deformable (non-rigid) predicted displacement is the correction.
> - Table 1 describes inference.
> - The 5-frames rows are kept for now; the reference choice and model selection are accepted.
>
> The splat point-density mismatch is also fixed: `run_vggt.py` now renders with the run's own
> `loss.volume.splat_res`.
>
> **Status.** Writing guidance only. No paper file was edited. All pre-v2 numbers must be re-measured.

Sources for this doc (debate files preserved, gitignored): `temp/paper_methods_debate/`
(`project_brief.md`, `debate_A_round1.md`, `debate_B_round1.md`, `debate_A_round2.md`,
`debate_B_round2.md`). Draft read: `_paper/main.tex`, `sections/Intro_lx.tex` (byte-identical to
`introduction.tex`), `method.tex` (placeholder), `experiments.tex`, `related_work.tex`, `brainstorm.md`,
`math_commands.tex`, `reference.bib`.

---

## 1. The project, explained for another agent (code-verified 2026-09-13, HEAD `d0c2854`)

### 1.1 One paragraph
**Input:**
- A **free-breathing, ungated short-axis (SAX) stack** with **one 2D frame per slice plane**. Each
  frame was acquired at an **unknown cardiac phase and unknown respiratory state**.
- There is no ECG, no navigator, and no repeated frames for self-gating. The only per-frame metadata
  is the prescribed through-plane position `z`.
- One frame, at the central plane, is designated the **reference**. Its *image content* defines the
  target cardiac contraction state.

**Output:** the **target-state volume**, i.e. the heart at that cardiac state and at end expiration,
on the acquisition grid.

**Method:**
- A transformer initialised from **VGGT-1B** (Wang et al., CVPR 2025) attends jointly over all
  frames.
- For every input pixel it predicts a **residual 3D displacement** from where the pixel was acquired
  to where that tissue lies in the target state.
- A **differentiable, coverage-normalised trilinear forward splat** renders the moved intensities into
  a volume.

**Training:**
- Image-domain losses against **ECG-gated breath-hold cine** at the target phase.
- No motion, registration or phase labels, and no DVF ground truth.
- Training data are **simulated** from gated cine: random per-plane phase, plus a rigid respiratory
  translation model and augmentation.

**Evaluation:**
- Against SVRTK / NeSVoR (and planned FC-SVR etc.) on **byte-identical simulated inputs**.
- Qualitatively on real real-time free-breathing MIITT scans, including an AFib patient.

### 1.2 Why (the gap)
- **Standard cine is costly and fragile.** It uses ECG gating plus one breath-hold per slice, which
  fails in arrhythmia, dyspnoea, children and fetuses.
- **Real-time free-breathing cine removes gating, but slices become mismatched.** Each shows a
  different cardiac state and is displaced by breathing, mostly through-plane.
- **Classical SVR needs redundancy** (multiple stacks/orientations/frames).
  - Single-stack SVRTK recovers **slope ≈ 0** of injected per-plane motion on all axes.
  - A 2-overlapping-stack positive control recovers 96–103%.
  - Single-stack SVRTK bakes ~85% of breathing displacement into the volume (docs/40, 41).
- **Self-gating needs a temporal stream per slice**, which the one-frame regime removes (docs/04 §4).
- **Novelty is the combination** of gating-free + respiratory-corrupted + phase-scattered single
  stack. Neither "single frame" alone nor "unobserved phase" is novel (docs/02, 03).

### 1.3 Information contract (docs/04, docs/25)
**Known per frame:** plane position `z_k`. **Unknown:** cardiac phase `τ_k` and respiratory state
`r_k`, independent across planes.

**Target phase comes from the reference frame's content, not an index.**
- A content-free phase index made every patient's EF regress to the cohort mean (pred-EF slope ≈ −0.03).
- An oracle splat gave slope 1.03, so the pipeline was sound and this was an information limit
  (docs/24).
- `k/12` is fractional R–R time, not a fixed cardiac state (ES drifts frames 3–8, docs/17).
- **Consequence:** the method reconstructs cardiac states that are *observed* (at the reference plane),
  not arbitrary unobserved ones.

**Respiration is corrected, not resolved.**
- The target is fixed at end expiration, which is ≈ the breath-hold reference.
- Reason: cardiac state is visible in one SAX frame; respiratory state largely is not.

**Input `t` is never a model input.** The dataset emits `t_indices`; the aggregator ignores them.

### 1.4 Data
**Training cohort: `training/splits/pooled_curated_v2.txt`.**
- 898 subjects, split 628/90/180.
- Sources: CMRxRecon 2023/24/25, ACDC, M&Ms-1; gated bSSFP SAX cine.
- Stratified by (source, vendor, centre, pathology).
- **Canon and CMRx25 Center012 (Philips)** are unseen-vendor holdouts: val/test only (docs/97).

**Curation (docs/95–96).**
- Inter-slice breath-hold misalignment measured on 1337 subjects × 12 phases with two agreeing
  estimators; 38.3% have ≥1 jump >5 mm.
- Kept ≤5 mm, plus 114 by-eye overrides.
- An uncurated ablation arm (1066 train, same val/test) is **PENDING**.

**Data engineering the paper can mention (appendix):**
- In-house ESPIRiT SENSE reconstruction of CMRxRecon k-space, with an input-domain bug fixed (docs/54).
- True slice pitch = thickness + gap, e.g. CMRx 8+4 = 12 mm (docs/27, 54).
- The odd-Z −1 slice roll in the challenge release fixed (docs/56).
- Slice order standardised apex-at-z0 per subject (docs/58 §10).

**Temporal format.**
- T = 12 phases over each subject's own R–R interval (CMRx resampled from native 14–40).
- ED = phase 0 for 97% of subjects.
- ED/ES labels come from nnU-Net LV volume curves.

**Held-out evaluation only:**
- **MIITT:** 13 subjects (10 volunteers + 3 cardiomyopathy), *paired* gated (1.5 mm) and real-time
  spiral (2.3 mm, 25 ms/frame, 180 frames × 13 slices), dz = 10 mm.
- **OCMR:** 8 gated stacks.
- Göttingen radial RT (68 volumes) for qualitative use.

**Preprocessing (`training/data/preprocess.py`).**
- LPS orientation.
- **In-plane only:** resample to 1.4 mm, crop/pad to 256²; **z never resampled** (native `D` = 5–21,
  pitch 5–12 mm).
  - Forcing a shared z grid caps even a Δ=0 identity splat at 25.4 dB vs exact 120 dB (docs/58).
- Intensity: all phases normalised to [0,1] using phase-0's (0.5, 99.9) percentiles.
- Physical z: `z_norm = z_mm / 90`. The 90 mm constant avoids aliasing of the period-2 Fourier
  features (docs/58).

### 1.5 Simulation of a sparse free-breathing stack (per training sample)
Code: `mri_dataset.py:377-505`, `gpu_aug.py:372-544`, `respiratory.py`.

1. **Target phase.** `τ* ~ U{0..11}`. Val instead uses a deterministic ED + ES sweep per subject
   (180 = 90 × 2).
2. **Slots.**
   - Slot 0 is the reference `(τ*, z_ref)`, with `z_ref = (bbox_z0+bbox_z1)//2 = ⌊D/2⌋`, the
     geometric centre (only *approximately* mid-ventricular).
   - Each other plane appears **exactly once** with `τ_k ~ U{0..11}` i.i.d., so **S = D**.
   - The reference plane is observed only via the reference.
   - (Legacy multi-frame sampler, `one_frame_per_slice=false`: cover every plane once, then fill a
     budget S with extra frames at uniformly random planes, with replacement. It is **not**
     "F frames per plane".)
3. **Augmentation** (train only; "aggressive" tier).
   - One draw per subject, shared across all 12 phases, the content mask and the heart ROI.
   - Ops: flip, affine (rotation ±180°, translation ±32 px), isotropic zoom, gamma, bias field,
     simulated low-res, Gibbs, phase-encode ghosting. All in-plane.
   - **Applied to the 4D tensor before both the target and the inputs are extracted**, so every
     artefact appears in the target too.
   - **No additive noise** (commented out). The model learns *invariance*, not denoising
     (`gpu_aug.py:435-468`).
4. **Respiration** (train and val; val deterministic).
   - Rigid per-plane translation by deform-then-reslice (shift the volume, resample the fixed plane).
   - SI displacement `d(r) = A·sin^6(π r)` (Lujan waveform).
   - Per subject: `A ~ 18.8 + U(−7.35, 7.35)` mm (Burger & Meintjes, MRM 2013), AP = 0.35·SI, tilt
     `θ ~ U(0°, 45°)` plus azimuth.
   - Per plane: breath phase `r_k ~ U[0,1)`.
   - **Applied to every input frame including the reference; the target is not shifted.**
   - The exact displacement is kept as `resp_disp_mm` for evaluation only.
   - **Not modelled:** rotation, non-rigid deformation, LR motion, slice profile, bSSFP transients,
     undersampling artefacts, R–R variability.

### 1.6 Network (`vggt/models/{vggt,aggregator}.py`)
**Initialisation and freezing.**
- Initialised from VGGT-1B (~941 M parameters).
- **Frozen:** DINOv2 ViT-L/14-reg patch embedding (~304 M).
- **Trained:** 24 frame + 24 global alternating-attention blocks, camera/register tokens, z-embedder,
  DPT head (~636 M).
- Aggregator fine-tuning is required: +3.4 dB motion PSNR vs head-only (docs/26).

**Input.** Each native 256² frame is resized to R² (R = 518 in the final recipe), tiled to 3 channels
and ImageNet-normalised; this gives 37² patch tokens.

**Per-frame special token:** `camera_token[a_k] + ZEmb(z_k/90mm)`.
- `ZEmb(z) = Linear([z, sin 2^iπz, cos 2^iπz]_{i=0..2})`.
- `a_k = 0` for the reference, 1 otherwise.
- The **4 register tokens are also two-way**, so they mark the reference as well.
- The result is permutation-equivariant except for the reference.

**Head.**
- DPT head on layers 4/11/17/23 outputs a linear residual displacement `δ ∈ R³` (normalised units) plus
  an unused confidence.
- `p = c + δ`, where `c = (x_norm, y_norm, z_norm)` is the acquired scanner coordinate.
- The map is **forward** (acquired pixel → target position), not a backward warp.
- Camera/depth/track heads and SfM losses are removed (docs/48). A B-spline head is a variant only.

### 1.7 Rendering (`vggt/utils/splat.py:12-145`, `training/loss.py:151-189`)
**Formula.**
`V̂(v) = Σ g·k(v − π(p))·I / (Σ g·k(v − π(p)) + 1e−6)`
- `k` is the trilinear kernel.
- `π` maps physical z to plane index via the subject's pitch.
- `g = 1[I > 1e−3]` excludes padding.
- Computed in fp32.

**Point density (training vs evaluation; the output grid is identical).**
- Both scatter into the subject's native `D×256×256` grid.
- The final training recipe sets `loss.volume.splat_res = 518`. The predicted field and the native
  frame are both bilinearly upsampled to 518² *points* before scattering (~4.1× denser), into the same
  native grid. The alternative is `null`: one point per native pixel.
- **Evaluation matches training (fixed 2026-09-13).** `evaluation/src/engine/run_vggt.py` reads
  `splat_res` from the checkpoint's own `run_meta.jsonl` config, passes it to `_splat_preds_native`, and
  stamps it in the arm metadata. Verified that all four `final518_*` runs record `splat_res = 518`.
- The real-time MIITT harness `evaluation/src/engine/run_vggt_rt.py` also uses the run's `splat_res`
  (fixed by the author; verified by grep, every `_splat_preds_native` call in `evaluation/src` passes it).
- Paper: one sentence, "the splat scatters R² points per frame (R = 518) at training and test time".

**Properties.**
- Each covered voxel is a **convex combination of acquired intensities**, so there is no appearance
  synthesis.
- Uncovered voxels are holes.
- A stop-gradient on the denominator restores the through-plane gradient but tears holes (−4 dB), so it
  was rejected (docs/37).

### 1.8 Objective (`training/loss.py:192-375`)
`L = L_vol + 0.5·L_heart + 0.5·L_gather`

- **`L_vol`:** L1 over the whole `D×256×256` grid, deliberately unmasked (a masked variant painted
  ghost intensity where the target is 0).
- **`L_heart`:** L1 inside a dilated **nnU-Net whole-heart ROI** (union over phases). The heart is ~5%
  of voxels, so this is ~11× per-voxel weight. It sped the contraction-amplitude transition from
  ~43.9k to ~8.9k steps (docs/71). **Segmentation labels are used in training.**
- **`L_gather`:** `mean g·|V_gt(p) − I|`, sampling the target at each predicted position and comparing
  with the acquired pixel (model-resolution pixels).
  - It restores the per-pixel through-plane placement gradient that coverage normalisation flattens.
    Breathing EPE is 1.37 mm with it vs 2.89 mm without (docs/44).
  - It is the SVR observation-consistency residual.
  - It is exact only for tissue whose appearance does not change between `τ_k` and `τ*`.
- **No smoothness regulariser in the paper model.** The no-reg arm won EF ranking (r 0.792 → 0.837).
  The L2 diffusion weight 1000 and motion-weighted L1 exist only as arms.

### 1.9 Optimisation and inference
**Optimisation (`sbatch/train_final_518.sh`).**
- AdamW (fused), peak LR 5e−5, 5% warmup then cosine, weight decay 0.05.
- bf16 compute (splat in fp32), gradient clip 1.0.
- Batch = 1 subject; 628 steps/epoch; 300 epochs; seed 42.
- One L40S at ~30 min/epoch (~6 days).
- LR ≥ 2e−4 kills the DPT head through a dying ReLU (docs/64).
- **Model selection:** best val `metric_psnr_3d_heartseg`, measured on the same ROI as `L_heart`, so it
  must be disclosed.

**Inference.**
- One forward pass plus splat: ~100 ms per volume on an A40.
- A cine is produced by sweeping the reference over the real-time frames at the reference plane, with
  the other frames fixed (docs/87). This is the only place repeated frames enter.

### 1.10 Evaluation (docs/83–98)
**Inputs.**
- Frozen v2 val/test bundles from 7 sources.
- The `scatter/` bundle freezes VGGT's own phase + breathing draw, so **SVRTK/NeSVoR get byte-identical
  input** (verified to 6e−8).

**Baselines.**
- SVRTK (3D per-phase) and NeSVoR (native build).
- FC-SVR, Dangi, Stolt-Ansó, Fetal CMR 4D, CiNeVol appear in the draft tables; their runners are
  **PENDING**.

**Scoring.**
- 6-DOF NCC registration plus slab PSF (per-subject **thickness**), reporting registered and raw.
- **NCC primary** (intensity-gauge invariant); PSNR and windowed per-slice SSIM secondary.
- nnU-Net Task114 on the **heart-ROI crop for every arm, GT included** → Dice/HD95 and EDV/ESV/EF/LVM.
- Demeaned breathing EPE.
- Runtime.
- Real RT (MIITT): qualitative only.

### 1.11 Evidence so far (pre-v2 checkpoints unless noted; all must be re-measured on v2)
| finding | number | doc |
|---|---|---|
| Single-stack SVR cannot recover motion | slope ≈ 0; 2-stack control 96–103% | 40, 41 |
| Under breathing VGGT > SVRTK (CMRx n=30) | 20.6 vs 17.6 dB; cost −1.5 vs −10.6 dB; 29/30 wins | 42 |
| On clean coherent stacks SVRTK > VGGT | ≈ +3.2 dB | 83 |
| Reference slice recovers per-patient EF | slope ~0 → 0.77–0.79 (Spearman ~0.55) | 24, 25, 33 |
| Pooled EF slope across 7 config arms | 0.74–0.83; 518 input best PSNR (+0.26 dB) and EF MAE | 86 |
| Residual EF bias | −12.6 pts (ED under-expansion + half-rendered wall) | 92 |
| Breathing simulation necessary | +3.33 dB bbox / +2.24 dB motion | 05 |
| Gather auxiliary | breathing EPE 1.37 vs 2.89 mm | 44 |
| Augmentation | +0.41 dB pooled OOD (n=61), −0.13 in-dist | 46 |
| Aggregator fine-tune needed | +3.4 dB motion vs head-only | 26 |
| Appearance wall | ~90% of in-dist error shared across variants; unobserved-plane target appearance not in the inputs | 19–22, 44 |

**PENDING:**
- the final 518/518 five-arm sweep (base / diff1000 / motion10 / hw2 / motion10_hw2);
- the curated-vs-uncurated ablation;
- the v2 SVRTK/NeSVoR generation + scoring campaign (launched 2026-09-13).

### 1.12 Limitations the paper must scope
1. Quantitative results are on **simulated** acquisitions derived from gated data; real RT results are
   qualitative. The single-shot gap was measured at −3.03 dB (docs/13).
2. Respiration is modelled as rigid SI + AP translation.
3. A target-phase reference frame is needed; only observed cardiac states are reconstructed. Per-voxel
   appearance at unobserved planes is information-limited, so EF/volumes are the meaningful per-patient
   outputs.
4. No through-plane super-resolution; single SAX orientation.
5. EF is under-estimated by ~12 pts, while per-patient ranking is preserved.
6. Classical SVR wins on clean coherent stacks. The claim is robustness to gating-free, breathing-
   corrupted, phase-scattered input, plus ~10³× speed.

---

## 2. How famous papers write their Method section (literature sweep)

**Coverage.**
- **Debater A** (medical imaging): 19 papers. Full text for most, via arXiv PDF or PMC XML. Access
  gaps: MoDL §II, nnU-Net NM published version, TransMorph §3.3–3.4, and CineVN §2.1 were not read;
  NeSVoR/XD-GRASP/CineVN equations were stripped by the HTML extraction.
- **Debater B** (vision/ML): 21 papers. Raw full text for MASt3R and Softmax Splatting; the other 19
  were read through a summarising fetch tool, so quotes may be lightly normalised and several venues
  were mis-stated by the tool and corrected from the publication record.
- Per-paper records (verbatim headings, formulation quotes, loss style, length share) are in the
  round-1 files.

### 2.1 Medical imaging / MR reconstruction (A)
U-Net, nnU-Net, VoxelMorph, SynthMorph, SynthSeg, TransMorph, MoDL, E2E-VarNet, Kuklisova-Murgasova
(SVRTK), NiftyMIC, SVoRT, NeSVoR, FC-SVR, DMCVR, fetal_cmr_4d, XD-GRASP, Qin 2018, Bai 2018, CineVN.

1. **Reconstruction papers state an operator acquisition model before any network (10/10).**
   Placement varies:
   - first Methods subsection (Kuklisova, NeSVoR, fetal_cmr_4d);
   - "Mathematical Preliminaries" (FC-SVR, CVPR);
   - Background (VarNet) or Introduction (CineVN);
   - inline (SVoRT, NiftyMIC).

   Registration/segmentation papers instead pose function estimation `g_θ(f,m)=u` plus an energy-style
   loss.
2. **SVR papers state what is known vs unknown.**
   - Kuklisova: "PSF … known, but the positions and orientations … have to be estimated".
   - fetal_cmr_4d: "slice position and acquisition times are known, … displacement … and the cardiac
     phase are not".
3. **Architecture is subordinated when the contribution is formulation or data.**
   - VoxelMorph: "the exact architecture is not our focus".
   - SynthSeg: "network architecture is not a focus".
   - SynthMorph puts it under Implementation Details.
4. **Load-bearing synthetic data is written in Methods as sampling statements** `θ ~ U(a,b)`, with
   values in a table (SynthSeg §3.1, SynthMorph §II-C, fetal_cmr_4d "Simulation").
5. **Loss = one display equation `L_sim + λ L_reg`** with classical-energy lineage.
6. **Learned estimators are tied back to the inverse problem:**
   - VoxelMorph "Amortized Optimization Interpretation";
   - FC-SVR "iterations of classical SVR as … splat and slice operations";
   - NiftyMIC's Nadaraya–Watson scattered-data initialisation (the classical analogue of our splat).
7. **Ill-posedness is confronted in words.** NeSVoR: "ill-posed … insufficient ROI coverage". FC-SVR:
   "Optimization-based methods do not work on single stacks".
8. **Evaluation carries clinical endpoints and statistics** (EF/strain + Holm correction, Bland–Altman),
   a simulated-with-GT arm plus a real arm, and a stated baseline-tuning protocol (NeSVoR).

### 2.2 Vision / ML (B)
VGGT, DUSt3R, MASt3R, CUT3R, MonST3R, Fast3R, NeRF, 3DGS, pixelNeRF, LRM, D-NeRF, Nerfies,
Deformable-3DGS, CoTracker, RAFT, ViT, DETR, MAE, SAM, Softmax Splatting, SynSin.

- **P1. The first Method unit is a task definition:** inputs → outputs, one named function, light
  notation. Examples: VGGT "3.1 Problem definition and notation", Fast3R, D-NeRF "Problem Formulation",
  SAM "Task".
- **P2. The reference/anchor is part of the problem definition.** VGGT: "first image … reference frame
  … permutation equivariant for all but the first frame".
- **P3. Building on a foundation model → a short recap, then an explicit diff** (MASt3R "3.1 The
  DUSt3R framework" + contributions in blue; MonST3R; pixelNeRF).
- **P4. The differentiable renderer is an equation in the main text; engineering goes to the appendix**
  (NeRF, 3DGS, Softmax Splatting Eq. 11 "average splatting", SynSin).
- **P5. One total loss equation with named weighted terms**, one short paragraph each.
- **P6. Data placement follows its role.** Off-the-shelf data goes in Experiments; generated or adapted
  data goes in Method (SAM Data Engine, MonST3R, VGGT 3.4).
- **P7. One implementation paragraph in the main text; tables in the appendix.**
- **P8. Dynamic reconstruction = per-point displacement into a shared frame, with direction stated**
  (D-NeRF `Ψ_t: (x,t) → Δx`, Nerfies, Softmax Splatting forward vs backward).
- **P9. The supervision regime is one sentence** ("no 3D supervision").
- **P10. Maths density tracks novelty.** Inherited parts are prose.

### 2.3 The one paper that sits on both sides
**FC-SVR (Young et al., CVPR 2024, arXiv 2312.03102)** is the closest prior art in method shape and is
in the draft's baseline table. Its method is single-stack, feed-forward, dense per-slice motion, and it
splats intensities. It opens with *"3 Mathematical Preliminaries: 3.1 Slice Acquisition Model"*
(`f_k(x,y) = ε + ∫(v∘u_k)(x,y,z) h(s_k−z) dz`) at a vision venue. This decided the hybrid.

---

## 3. The debate

### 3.1 Round 1 positions
- **A (medical imaging).**
  - Venue: TMI.
  - Methods opens with the forward model `I_k = S_{z_k}[W_{r_k} V(·,t_k)] + ε_k` as an inverse
    problem.
  - Simulator written as "training-data generation".
  - Splat presented as scattered-data interpolation.
  - A separate evaluation-setup section with clinical endpoints.
  - Main attack on B: "set of views → 3D" misstates identifiability. There is no pose problem and no
    correspondence; the scene is a conditioned state of a 4D object; physical sampling is first-order.
- **B (vision/ML).**
  - Venue: CVPR/ICCV.
  - Methods opens with a VGGT-style task definition `f_θ`, with at most a 2-line observation model.
  - Architecture as a MASt3R-style diff.
  - Contributions: task + rendering supervision + simulation engine + recoverability argument.
  - Main attack on A: no measurement operator exists to invert. The observation model only generates
    data, training is amortised regression with no regulariser, and full physics invites
    unmodelled-term questions.

### 3.2 Mid-debate injection: the draft in `_paper/` is binding
- **Venue: ICLR 2027 template.** Title: *"Learning Reference-Conditioned Cardiac Volume Reconstruction
  from Free-Breathing Ungated MRI"*.
- **Vocabulary is fixed:** "reference-conditioned cardiac volume reconstruction", "reference", "target
  cardiac contraction state", "asynchronous stack", "cardiac-state mismatch", "target-state volume",
  "end expiration", "gather loss".
- **Contributions** are formulation / *adapted* feed-forward model / validation.
- **Planned Methods figures:** method overview, breathing simulation.
- **Notation:** Goodfellow dlbook (`math_commands.tex`).

This mooted both venue recommendations and A's terms "anchor frame" / "reference standard".

### 3.3 Round 2: concessions
**A conceded:**
- ICLR ML-style Method;
- architecture as a recap + diff;
- the register tokens also mark the reference (verified, `aggregator.py:169,292`);
- the 518² training splat density goes in Method;
- Softmax Splatting average splatting as the renderer citation;
- "reference" and "gather loss";
- the forward-vs-backward argument;
- no "unsupervised";
- a direct volume-regression ablation as a reviewer gap.

**B conceded:**
- an observation model with an explicit known/unknown split inside §3.1;
- the simulation written as sampling the model's latent variables, with a "not modelled" paragraph;
- the learning problem as an expectation over `p_sim`;
- two lineages for the splat (Nadaraya–Watson/SVR and Softmax Splatting);
- "corrected not resolved" respiration;
- A's experiments: random-init ablation, phase-oracle SVRTK, stress tests outside `p_sim`, baseline
  tuning protocol, EF with the reference plane excluded.

### 3.4 Residual disputes and my rulings (each checked against code)
| dispute | A | B | ruling | evidence |
|---|---|---|---|---|
| noise term `ε_k` in the observation model | keep | drop | **Drop.** List noise/RT artefacts under "not modelled"; describe augmentation as appearance randomisation of `V` (SynthSeg-style). | No additive noise; all artefacts applied to the 4D tensor before target *and* inputs are cut (`gpu_aug.py:435-468`, noise commented out in all tiers) |
| does the observation model appear in the method, or only the simulator? | gather loss = SVR residual | only simulator | **A, one clause.** `L_gather = |V*(p) − I|` evaluates the target at the estimated position against the measurement. That is the Kuklisova residual under a phase-invariance assumption. State it and its bias. | `loss.py:353-362` |
| "what transfers from VGGT is set attention, not geometry" | claimed | unverified | **Don't claim.** No random-init ablation exists; the frozen 304 M DINOv2 embedding is reused. | `default.yaml:510`; docs/26 only shows head-only is insufficient |
| subsection order | simulation 2nd | simulation after objective | **Formulation → network → splat → objective → simulation → inference.** Matches the figure float order in `method.tex` (overview before breathing), the ML convention, and `p_sim` as a forward reference. | `_paper/sections/method.tex:3-15` |
| "point map" wording | avoid | one analogy sentence | **One sentence allowed**, but the primary term is "residual 3D displacement", which the draft intro already uses. | `Intro_lx.tex:18` |
| symbol for slice pitch | `h` | `Δz` | **`h`.** The draft's breathing figure uses `Δz` for the breathing shift. | `method.tex:13` |
| "unordered set of posed slices" | reject | used in R1 | **Reject.** Say "equivariant to permutations of non-reference frames"; frames carry `z`. | — |

---

## 4. Final recommended Section 3 outline (drop-in for `_paper/sections/method.tex`, ≈2.5–3 pp)

```latex
\section{Methodology}\label{sec:method}
% Fig. method_overview float first (already placed)
\subsection{Problem formulation}\label{sec:formulation}          % ~0.6 p — §5 below
\subsection{Reference-conditioned reconstruction network}\label{sec:model}  % ~0.45 p
\subsection{Differentiable forward splatting}\label{sec:splat}   % ~0.35 p
\subsection{Training objective}\label{sec:objective}             % ~0.3 p
\subsection{Simulating free-breathing ungated acquisitions}\label{sec:simulation} % ~0.5 p + Fig. breathing_simulation
\subsection{Inference}\label{sec:inference}                      % ~0.15 p
```

**3.2 Network.**
- *Recap* (2–3 sentences, prose): VGGT tokenises each image with DINOv2, alternates frame-wise and
  global attention, and marks the first frame with dedicated camera/register tokens; DPT heads.
- *Diff* (list):
  - frames resized to 518² and tiled to 3 channels;
  - per-frame token `e^cam_{a_k} + ZEmb(z_k/Z_0)`, with `Z_0 = 90` mm physical;
  - the two-way camera and register tokens repurposed to mark the reference, initialised from their
    pretrained values;
  - DPT head → linear residual displacement;
  - camera/depth/track heads removed;
  - patch embedding frozen, everything else fine-tuned.
- *Emphasise, without calling it novel:* conditioning on reference-frame **content** rather than a phase
  index. Point to the index-vs-reference EF ablation (re-measure on v2).

**3.3 Splatting.**
- Equation (average splatting, trilinear kernel, coverage normalisation, intensity gate).
- `π` maps physical z to plane index via `h`.
- Lineage: Nadaraya–Watson scattered-data approximation (NiftyMIC / FC-SVR) + Softmax Splatting.
- **Why forward:** no target-state volume exists to warp backward from.
- Convex-combination property; holes → motivates the gather loss.
- One sentence on point density: R² points per frame (R = 518), the same at training and test time,
  into the native output grid.
- Numerics → appendix.

**3.4 Objective.**
- `\Ls = \Ls_{vol} + λ_h \Ls_{heart} + λ_g \Ls_{gather}`, one paragraph per term.
- `\Ls_{heart}`: the ROI is an automatic nnU-Net whole-heart segmentation, used in training only.
- Gather loss:
  - observation-consistency clause;
  - "exact only for tissue whose appearance is unchanged between `τ_k` and `τ*`";
  - uses model-resolution pixels.
- "No smoothness regulariser; the L2 diffusion penalty is ablated in App. X."
- Optimiser details → §4 setup / appendix.

**3.5 Simulation.**
- Sampling statements, one per random variable: `τ*`, `τ_k`, `k_ref = ⌊D/2⌋`, per-subject in-plane
  geometry + appearance draw shared by inputs *and* target, per-subject `A`, `θ`, `φ`, per-plane `r_k`,
  `d = A sin^6(π r_k)` [Lujan], AP = 0.35 d, deform-then-reslice on **every input frame including the
  reference**, target unshifted.
- "Not modelled" paragraph.
- Val/test use the same simulator with fixed seeds.
- Parameter table → appendix.
- Citations missing from `reference.bib`: Lujan 1982, Burger & Meintjes 2013.

**3.6 Inference.**
- One forward pass + splat (~0.1 s).
- A cine = sweeping the reference over real-time frames at the reference plane.
- **State how the reference plane/frame is chosen on real RT data.**
- No mask, no phase, no optimisation.

**Goes in §4 setup or the appendix, not §3:**
- data, splits, holdouts, MIITT/OCMR;
- byte-identical inputs, registration gauge + slab PSF (per-subject thickness), metric definitions;
- EF also reported with the reference plane excluded;
- preprocessing, curation, optimiser and compute;
- model-selection metric disclosure;
- multi-frame sampling rule.

**Bibliography gaps** (not in `reference.bib`): DINOv2, DPT, Softmax Splatting (Niklaus & Liu 2020),
SynthSeg, VoxelMorph, SVoRT, nnU-Net, Lujan 1982, Burger & Meintjes 2013.

---

## 5. Final problem-formulation draft (LaTeX, draft vocabulary, dlbook notation)

Merged from A-R2 §6 and B-R2 §6 under the §3.4 rulings (≈650 words of prose). Add
`\def\vphi{{\bm{\phi}}}` and `\def\vdelta{{\bm{\delta}}}` to `math_commands.tex`, or inline `\bm{}`.

```latex
\subsection{Problem formulation}
\label{sec:formulation}

We study \emph{reference-conditioned cardiac volume reconstruction}: given an asynchronous
short-axis stack in which one frame is designated as the reference, recover the 3D volume of the
heart in the cardiac contraction state shown by the reference, at end expiration
(\Figref{fig:method_overview}).

\paragraph{Observation model.}
Let $V(\vx,\tau)$ denote the image intensity of a subject's heart at end expiration, where
$\vx=(x,y,z)\in\R^3$ is the patient coordinate with $z$ along the short-axis normal and
$\tau\in[0,1)$ is the cardiac phase as a fraction of the R--R interval. A short-axis acquisition
prescribes $D$ parallel slices at positions $z_k=\big(k-\tfrac{D-1}{2}\big)h$, $k=0,\dots,D-1$,
where the slice pitch $h$ (thickness plus gap) and $D$ vary across subjects. In the free-breathing,
ungated setting, each slice contributes a single 2D frame
\begin{equation}
  \mI_k=\gS_{z_k}\!\left[\,V\!\left(\bm{\phi}_{r_k}^{-1}(\cdot),\,\tau_k\right)\right]\in\R^{H\times W},
  \label{eq:observation}
\end{equation}
where $\gS_{z}$ samples the in-plane pixel grid $\sU$ at height $z$ and $\bm{\phi}_{r}$ is the
respiratory displacement at breathing state $r$ ($\bm{\phi}$ is the identity at end expiration).
Because breathing moves the heart through the fixed imaging plane, $\mI_k$ generally depicts tissue
that lies elsewhere, in-plane and through-plane, at end expiration.

\paragraph{Known and unknown quantities.}
The slice positions $z_k$ and the in-plane geometry are known. The cardiac phase $\tau_k$ and
respiratory state $r_k$ of each frame are not: slices are acquired in different heartbeats and
breaths, so these states vary independently across the stack (cardiac-state mismatch), and no ECG,
respiratory, or self-gating signal is available. To specify which cardiac state to reconstruct, the
frame at the central slice $k_{\mathrm{ref}}=\floor{D/2}$ (approximately mid-ventricular) is
designated the \newterm{reference}; the \newterm{target cardiac contraction state}
$\tau^\star\equiv\tau_{k_{\mathrm{ref}}}$ is conveyed only through its image content, never as a
phase value. The reference is itself displaced by breathing. The model input is
$\sS=\{(\mI_k,z_k,a_k)\}_{k=0}^{D-1}$ with $a_k=\1[k\neq k_{\mathrm{ref}}]$.

\paragraph{Target.}
We seek the \newterm{target-state volume}, the heart at cardiac state $\tau^\star$ and end
expiration, sampled on the acquisition grid $\sG=\sU\times\{z_k\}_{k=0}^{D-1}$:
\begin{equation}
  \tV^\star=V(\cdot,\tau^\star)\big|_{\sG}\in\R^{D\times H\times W}.
  \label{eq:target}
\end{equation}
Respiratory motion is corrected rather than resolved: the cardiac state is visible in a single
short-axis frame, whereas the respiratory state largely is not. $\sG$ keeps each subject's native
slice positions; we do not resample through-plane.

\paragraph{Why a learned prior is needed.}
\Eqref{eq:observation} provides one frame per slice, each at a different unknown $(\tau_k,r_k)$, so
no anatomy is observed twice and the redundancy that slice-to-volume reconstruction exploits to
estimate motion is absent~\citep{kuklisovamurgasova2012reconstruction,xu2023nesvor}. Moreover, for
$k\neq k_{\mathrm{ref}}$ the target-state appearance $\gS_{z_k}[V(\cdot,\tau^\star)]$ is never
acquired. We therefore learn the required anatomical and motion prior from a population.

\paragraph{Reconstruction by displacement and splatting.}
A network $f_{\vtheta}$ jointly processes the stack and predicts, for every pixel $\vu\in\sU$ of every
frame, a residual 3D displacement $\bm{\delta}_k(\vu)$ from its acquired scanner coordinate
$\vc_k(\vu)=(u_x,u_y,z_k)$ to the position of the same tissue in the target state. A fixed
differentiable splatting operator $\gR$ (\secref{sec:splat}) renders the displaced intensities:
\begin{equation}
  \{\bm{\delta}_k\}_{k}=f_{\vtheta}(\sS),\qquad
  \hat{\tV}=\gR\Big(\big\{\big(\vc_k(\vu)+\bm{\delta}_k(\vu),\,\mI_k(\vu)\big)\big\}_{k,\vu};\,\sG\Big).
  \label{eq:transport}
\end{equation}
Like the point maps of DUSt3R and VGGT~\citep{wang2024dust3r,wang2025vggt}, $\vc_k+\bm{\delta}_k$
expresses every pixel in a shared frame, here the target cardiac state rather than a camera; slice
positions are known and nothing is triangulated. $f_{\vtheta}$ is equivariant to permutations of the
non-reference frames, and $\tau_k$, $r_k$ are never inputs or outputs.

\paragraph{Learning problem.}
Free-breathing ungated acquisitions have no volumetric ground truth, but ECG-gated breath-hold cine
provides $V(\cdot,\tau)$ on $\sG$ at $T=12$ phases, which we take as end expiration. We sample the
latent variables of \eqref{eq:observation} from a simulation distribution $p_{\mathrm{sim}}$
(\secref{sec:simulation}) and train
\begin{equation}
  \vtheta^\star=\argmin_{\vtheta}\;
  \E_{V\sim\train}\,
  \E_{(\tau^\star,\{\tau_k\},\{r_k\})\sim p_{\mathrm{sim}}}
  \Big[\Ls\big(\hat{\tV}_{\vtheta},\,\tV^\star\big)\Big].
  \label{eq:learning}
\end{equation}
Supervision comes only from the gated target-state volume; the displacement field is latent, and no
displacement, registration, or phase label is used. An automatically segmented heart region
reweights the loss during training (\secref{sec:objective}); inference requires no mask and no
per-subject optimization.

\paragraph{What is recoverable.}
Because $\gR$ normalizes by coverage, each covered voxel of $\hat{\tV}$ is a convex combination of
acquired intensities: the model relocates observed tissue but does not synthesize appearance. At
slices whose target-state appearance was never acquired, fidelity is therefore bounded by how much of
the change between cardiac states is explained by motion.
```

---

## 6. Draft statements that contradict the code — fix before submission

| # | where | draft says | code/evidence | fix |
|---|---|---|---|---|
| 1 | `Intro_lx.tex:10` | breathing "induces **non-rigid** spatial displacements" | **Resolved (author):** the predicted per-pixel displacement is dense/deformable, and real breathing is non-rigid. Only the *simulated corruption* is a rigid per-slice translation. | Keep the intro. §3.5 describes the simulator as a per-slice rigid translation (factual) |
| 2 | Table 1 | "**No external mask**" ✓ for Ours | **Resolved (author):** Table 1 describes inference, which is mask-free | Mention the training-time heart ROI in §3.4 only |
| 3 | `Intro_lx.tex:18` | "no explicit cardiac or respiratory phase information" | The reference conveys `τ*` implicitly by content | Keep "explicit"; add "beyond the reference frame" |
| 4 | `Intro_lx.tex:14` | "one acquired **mid-ventricular** frame" | Reference = geometric centre plane `⌊D/2⌋` (`mri_dataset.py:422`) | **Resolved (author):** fine as written |
| 5 | `Intro_lx.tex:14,18` | "without repeated temporal observations" | True per volume; the real-RT cine figure sweeps the reference over 180 frames at one plane (docs/87) | State in §3.6 |
| 6 | `experiments.tex:105-120` | ablation rows "Train frames per slice = 5" | No 5-frames-per-plane training sampler or model found. The legacy multi-frame sampler is a slot budget with random extras. `tools/exp_multiframe_quick.py` is an *eval-time* fps∈{1,5} probe on 4wok | **Resolved (author):** keep as is for now |
| 7 | Methods (to be written) | — | Training uses a 518²-point splat (`train_final_518.sh:65`). **Fixed (author, 2026-09-13):** `run_vggt.py` and `run_vggt_rt.py` now render at the run's own `splat_res`. | One sentence in §3.3 |
| 8 | experiments | "ground-truth EF" | Gated volumes are multi-breath-hold, curated ≤5 mm, 12-phase R–R-resampled | Define once as "gated target-state volume"; report residual misalignment stats |
| 9 | model selection | — | Best ckpt picked on val `psnr_3d_heartseg`, the same ROI as `L_heart` | **Resolved (author):** fine |

Not an error, verified: "thick (typically 6–8 mm) and potentially gapped" refers to *thickness*. The
method uses pitch; the eval PSF uses thickness. Never call `h` thickness.

---

## 7. Reviewer objections to pre-empt (union of both debaters, deduplicated)

1. **"Only simulated evidence; sim-to-real unproven."** Pre-empt with:
   - the "not modelled" paragraph;
   - the measured single-shot gap;
   - stress tests outside `p_sim` (larger amplitude, rotation, LR, non-rigid);
   - unseen-vendor holdouts;
   - MIITT real-time vs gated EF agreement on the same subject, if feasible.
2. **"VGGT + z embedding = limited novelty."** Keep contributions as formulation / adaptation /
   validation, show the explicit diff, and back each change with an ablation: aggregator fine-tuning,
   reference vs index conditioning, gather loss, breathing simulation.
3. **"Prior-dominated / hallucinated average heart."** Pre-empt with:
   - the convex-combination property;
   - identity floor and oracle transport bound;
   - per-patient EF agreement (slope / Bland–Altman), reporting the −12 pt bias up front;
   - EF with the reference plane excluded;
   - pathology subgroups (ACDC, M&Ms).
4. **"Straw-man baselines; SVR not designed for this."** Pre-empt with:
   - byte-identical inputs and a gauge-fixing registration;
   - the clean-input arm where SVRTK wins;
   - a phase-oracle SVRTK arm;
   - a learned baseline (FC-SVR);
   - a stated tuning protocol.
5. **"Why displacement + splat and not direct volume regression? Why a 941 M natural-image model?"**
   Pre-empt with a direct-regression ablation and a random-init ablation. **Neither exists yet.**

---

## 8. Open items
- Author review 2026-09-13 resolved §6 #1, #2, #4, #6 and #9 as fine.
- #7 is fixed in both `run_vggt.py` and `run_vggt_rt.py`.
- Add the missing bib entries (§4).
- Re-measure every §1.11 number on v2 once the final sweep and the v2 campaign finish.
