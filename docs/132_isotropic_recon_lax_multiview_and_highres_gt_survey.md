# 132 — Isotropic reconstruction + LAX/multi-view: slab forward model, LAX data inventory, and a sweep for high-res ground truth

> **TL;DR & takeaway**
> (2026-10-01, design + literature + inventory session; **nothing run, no code changed**. Companion: `docs/131_cardiac_respiratory_motion_physiology_and_sim_realism.md`.)
> 1. **Slab forward model is a standard recipe, and it handles variable spacing.** SVRTK and NeSVoR both model each slice as a **Gaussian slab along the slice normal with σ_z = thickness / 2.355** (in-plane σ ≈ 0.51 × pixel size), in mm world coordinates; **thickness and gap are separate inputs** (thickness sets the PSF, gap only places slices). Adopting it in our observation loss is a modest change. It adds **no information** — the 4 mm gaps stay unconstrained (weight at the gap midpoint is ≈0.21 of peak for 8 mm slabs; a calculation, not a measurement) — only priors / extra views fill them.
> 2. **CMRx LAX = 3 single slices per subject (2ch/3ch/4ch, 12 cine phases) + a separate `lvot` file, one slice per view is the standard clinical protocol (SCMR).** But **CMRx2023/2024/2025 LAX carry NO geometry** (diagonal affine, no position/orientation; README: "LAX can never be registered to SAX here") ⇒ using it needs LAX↔SAX registration, weakly constrained (sparse intersections, separate breath-holds). **OCMR has real geometry** (positions + orientation cosines from ISMRMRD) but only 18 SAX+LAX exams (8 gated with 4-D GT). Every SVR tool we checked assumes **several multi-slice stacks in different orientations**; 1 SAX stack + 3–4 LAX planes is untested by them.
> 3. **There is NO large public isotropic cardiac CINE ground truth** (sweep over MRI, CT, phantoms, literature). On disk: **FRF (1 subject, 1.5 mm, 20 cardiac × 5 resp)**, HVSMR-2.0 (60 static CHD volumes), fiss k-space (5). Public: ~17 raw-k-space volunteers (OSU 7, Lausanne 5+5), static HVSMR/MM-WHS/ImageCAS, simulators (MRXCAT 2.0 / XCAT) with a large appearance gap. The big lead (Imperial UKDHP, 1,331 subjects 3D cine at 1.2×1.2×2 mm) is public **as ED/ES label maps only**.
> 4. **Implication:** direct supervised high-res training at scale is not possible. Realistic plan = **forward-model (observation) loss on all subjects + LAX-plane supervision where LAX is registered/geometry-known**, and use the small isotropic sets **for evaluation only** (see §6). Do not claim "isotropic output" without a real isotropic test.
> 5. **Direction under consideration (§7, undecided):** *simulate* the isotropic dynamic GT — **(A) segmentation → synthetic MR**, or **(B) static 3D MRI (HVSMR/MM-WHS) → dynamic MRI via a motion field**. Risks: appearance gap (docs/20, LISynSeg), and the motion source is invented/transferred. Proposed first step: a small sim-trained model tested on real FRF/LAX before committing.

**Evidence marks.** `READ` = opened by us/a subagent (often abstract only; many publisher pages were 403) ; `RECALLED` = background knowledge ; `INFER` = our inference ; `[M]` = measured from a file/header this session. All web/literature content was gathered by subagents — re-open before citing numbers.

## 1. Goal and the questions this doc answers
Output a **high-resolution (isotropic) volume** on top of CardioStitch's motion-corrected, target-phase reconstruction. Questions: how to treat a thick slice (slab) in the model; do we need more slices or views (LAX); is there high-res GT; what can be used for evaluation.

## 2. The slab forward model (read from the SVRTK and NeSVoR source)

**"Forward model"** = the simulation volume → observed slice: place the plane (pose incl. motion) → average over the slab with the PSF → sample at the slice pixel grid (+ optional per-slice intensity scale/bias, noise). Reconstruction is the **inverse** problem; the loss compares *simulated* slices to *observed* slices. The paper's Eq. 1 `I_i = A_i(V) + n` is this; our splat is a direct slices→volume mapping, and the paper's volume-to-slice sampler is the (currently PSF-free) forward operator used in the observation loss.

### 2.1 NeSVoR (`baselines/nesvor/NeSVoR/nesvor/`, READ)
- PSF sigmas `utils/psf.py:6-35`: `GAUSSIAN_FWHM = 1/(2√(2 ln2)) ≈ 0.4247`; σ_z = 0.4247·thickness; in-plane σ = `SINC_FWHM·res` with `SINC_FWHM = 1.2067·0.4247` (≈0.512·res). Axis-aligned 3D Gaussian in the slice's own frame, thick along the normal only.
- INR path: **256 Monte-Carlo points per observed pixel** drawn `N(0, σ_psf)` in the slice frame, mapped through the slice's current rigid pose, INR queried, mean taken (`inr/models.py:384-429`). Explicit-operator path: discrete 3D Gaussian kernel (`psf.py:38-79`), sparse (pixel, voxel) coefficient matrix (`slice_acq_torch.py:95-179`), output = Σw·V / Σw.
- Loss (`models.py:375-454`): heteroscedastic Gaussian NLL between simulated and observed pixel with **learned per-pixel and per-slice variance** and a **learned per-slice intensity scale**; per-slice 6-DoF pose optimised jointly (AdamW, ~6000 iters). Edge-preserving image regulariser; no hard outlier masking in the INR path.
- **Thickness vs gap are separate:** `Stack.thickness` (default = gap; override `--thicknesses`) sets `resolution_z` for the PSF; `gap` only places slices (`transform.py:324-331`). The PSF does not enforce coverage.
- Variable spacing: all in mm world coords; per-slice in-plane res, thickness, pose stored (`inr/data.py`); INR queried on any grid; output grid is **isotropic** (`--output-resolution`, default 0.8 mm). No assumption of fixed slice count in the INR path (the explicit-operator path asserts isotropic in-plane/out voxels).
- No temporal/cardiac-phase axis in the main path (`--deformable` = per-slice non-rigid warp latent, not tied to time).
- Not read: the CUDA kernel, `svr/registration.py`, `svort/`.

### 2.2 SVRTK (SVRTK/SVRTK master on GitHub, `include/svrtk/Parallel.h`, `src/Reconstruction.cc`, `ReconstructionCardiac4D.cc`, READ; no local copy)
- PSF (`Parallel::CoeffInit`, ~Parallel.h 899-917): 3D Gaussian in slice space, σ_x,y = 1.2·d/2.3548, **σ_z = dz/2.3548** (FWHM = thickness). `dz` is `-thickness` (per stack) / `-default_thickness`, defaulting to the header z voxel size (so the true CMRx pitch 12 mm ≠ thickness 8 mm must be passed deliberately).
- Forward: `sim = Σ p.value·V / Σ p.value` over the PSF-weighted sparse coefficients (`SimulateSlices`). Recon: Gaussian initialisation `V = Σ c·I / Σ c` (`GaussianReconstruction`), then iterative **backprojection** super-resolution (`Superresolution`: V += α·addon; not a solved least squares), edge-preserving regularisation, per-slice rigid SVR (NMI/NCC), robust EM voxel/slice weights, closed-form per-slice scale + smooth bias field.
- Output grid isotropic (`-resolution`, default min header voxel; cardiac tool default 0.75 mm). The 4D cardiac variant needs **externally computed cardiac phase per frame (`-cardphase`) and R–R intervals** (no self-gating) and weights slices into output phases with a sinc×Tukey temporal kernel.
- `INFER`: the model has no gap parameter, so for 8 mm / 12 mm pitch the gaps stay under-covered unless other stacks/phases fill them.

### 2.3 What to adopt for us
Slab-aware splat (each pixel spreads over several fine-z voxels with σ_z ≈ 0.4247·thickness ≈ 3.4 mm for 8 mm) **and** a matching blur-then-sample forward model in the observation loss (otherwise splat and loss are inconsistent); thickness (8 mm) and pitch (12 mm) as separate per-dataset inputs; optional per-slice scale/variance (robust-loss idea we lack). **Ground truth at the native pitch only:** our `V_gt` exists only at 12 mm pitch, so the reconstruction loss must also go through blur-and-subsample. A refinement stage after the splat is probably needed to fill the in-between z — `INFER`, untested (the paper has none).

## 3. LAX / multi-view: what we have and what SVR tools assume

### 3.1 Clinical norm
SCMR protocols (Kramer et al., JCMR 2020, READ): each LAX view (2-/3-/4-chamber, LVOT) is **one prescribed plane, not a stack**; SAX is a stack (6–8 mm thick, 0–4 mm gaps). CMRxRecon2024 (arXiv 2406.19043, READ): seven cine views (LAX 2/3/4ch, SAX, LVOT, aorta transversal + sagittal), 330 healthy volunteers, 3T. A search snippet says "a single slice was acquired for each LAX view" (consistent with our data; the paper's slice-count table lost its column headers in extraction).

### 3.2 Our data (docs + READMEs + `[M]` headers)
| Dataset | LAX content | Geometry |
|---|---|---|
| CMRxRecon2024 | `lax/` = 3 slices (2/3/4ch) + `lvot/`, `[M]` `lax/4d_recon.nii.gz` = (224,204,3,12), 6 mm thick, ~1.5 mm in-plane; 291/294 live subjects have LAX; info CSV `TemporalPhase=27` (acquired) vs 12 in recon (relation unchecked) | **none** — diagonal affine, origin 0; README: "NOT spatially reg. to SAX" |
| CMRxRecon2023 | 3 views packed in the slice dim (3ch/2ch/4ch) | **none** — README: "No patient-coordinate info at all — which is why LAX can never be registered to SAX here" |
| CMRxRecon2025 | LAX for some subjects only (one centre: 4/44 ship any LAX) | none; **view axis rolled −1** vs CMRx-300 in 7/7 subjects checked ⇒ not the documented 3ch,2ch,4ch order (docs/56) |
| **OCMR** | 18 exams with SAX+LAX; plane inventory SAX 25, 4CH 18, 2CH 15, 3CH/LVOT 3, axial 2, RVIT 2, RVOT 1; 8 gated (4-D GT) + 13 real-time free-breathing (no GT) | **yes** — slice positions + orientation cosines verbatim from ISMRMRD; same study UID ⇒ same patient, co-registered |
| M&Ms-2 | one 4-chamber LA cine slice (not a stack) | true oblique orientation exists |
| M&Ms-1, ACDC, MIITT | no LAX noted | n/a |
CMRx LAX/SAX are gated breath-hold cines at the same cardiac phases (docs/30).

### 3.3 Using CMRx LAX needs LAX↔SAX registration (unproven)
Risks: (1) a single LAX plane meets the SAX stack only along ~5–21 lines at 12 mm pitch (+4 mm gaps) — weakly constrains 6 DOF × 3 views; (2) separate breath-holds ⇒ per-view rigid shifts of a few mm confounded with pose; (3) pose error becomes label noise; (4) view identity/order (docs/56). **Feasibility check before building:** intersection-line intensity consistency after registration; all LAX planes through the LV long axis with plausible angles; convergence from different starts; inspect the raw `.mat` files for dropped position/orientation tags (not opened). A model using LAX needs a **plane-pose token** (6-DoF) in place of the z token and `scanner_coords=(x,y,z_norm)`.

### 3.4 What SVR tools assume (subagent literature read)
All checked SVR works use **several multi-slice stacks in different orientations**: van Amerom MRM 2019 (≥3 stacks, ≥30 slices total; READ), Roberts Nat Commun 2020 (≥5 non-coplanar stacks of 7–11 slices; READ), NeSVoR TMI 2023 (3–10 stacks real data; ablation: more orientations/stacks help, nothing tests one stack; READ local PDF), NiftyMIC (5-stack example; README), Kuklisova-Murgasova 2012 (8 stacks, 3 orthogonal planes; snippet), Tian MRM 2025 and Odille MICCAI 2015 (multi-slice stacks in several orientations; abstract/snippet only), CiNeVol (not opened). **None tests 1 SAX stack + 1 slice per LAX view; no source says it suffices or fails** (unverified, not evidence of failure). SVRTK/NiftyMIC/fetal_cmr_4d degrade to through-plane interpolation on single SAX (docs/31, docs/34) — the gap a learned prior is meant to fill.

## 4. Sweep for high-res (isotropic) ground truth — results
**Headline: no public, ready-made isotropic cardiac cine image dataset with many subjects exists.** Four parallel sweeps (local disk, public MRI, CT/phantoms/synthetic, GT-free literature). Not exhaustive: not searched or only partly — OpenNeuro, LVSC, York, EMIDEC, CAD-RADS sets; many publisher pages 403.

### 4.1 On disk (`[M]` unless noted)
| Path | What | N | Spacing | Cine? | Notes |
|---|---|---|---|---|---|
| `data/FRF/nifti_rlr/resp{0-4}_4d` | free-running 5D whole-heart (SubSpaceMoCo5D demo), 20 cardiac × 5 resp | **1** | 200×186×234×20, 1.50×1.505×1.496 mm | **cine** | Zenodo 15033956, CC-BY 4.0 [R]; healthy volunteer [R] |
| `data/FRF/sax_native`, `sax_8mm` | our reslice to SAX (1.5 mm iso; header by construction) and slab-averaged to 8 mm | same | 160×160×96×20 ; 1.5×1.5×8 | cine | ready-made GT/input pair |
| `data/hvsmr/cropped` | HVSMR-2.0 3D whole-heart SSFP, 1.5T, free-breathing nav | **60** | pat0 0.885×0.885×0.890 ; pat10 0.708×0.708×0.800 ([R] mean 0.73×0.73×0.81) | **static** | CHD patients, ages <1–52; license not in README (Figshare 10.6084/m9.figshare.c.7074755.v2, Sci Data 2024; check terms) |
| `data/fiss/inVivo.zip` | raw 3D radial k-space, 5 volunteers × 5 sequences, 2 mm iso [R] | 5 | 2.0 mm [R] | raw only | Zenodo 13868462, CC-BY 4.0 [R] |
| `data/fiss/recon_preview/cs_v01_final.npz` | our CS recon of V01 | 1 | (25,192,192,192), 2 mm | cine | README: "somewhat blurry/soft" — **not GT quality** |
| `data/goettingen` | radial RT free-breathing SAX | 68 vols [R] | 1.6 mm in-plane, **6 mm slices** [R] | RT | not fine through-plane |
| `scratch/fetal_cmr_4d/.../vol_p*`, `scratch/nesvor`, `scratch/niftymic` | **our own SVR reconstructions** (1.25 / 0.8 / 1.0 mm grids) | 1 / 2 / 12 dirs | resampled grids | cine/static | **reconstructions, not GT** (fine grid = output choice) |
| OCMR / CMRxRecon 2023/24/25 / MIITT | all 2D multi-slice (12 mm pitch; MIITT dz 10 mm) | — | — | — | no 3D / whole-heart tasks on disk |
`data/whs` = mask-stats manifest only (not image data). Not inspected: ACDC, M&Ms, CMAC, corseg, archive dirs.
**Verdict: true isotropic cardiac cine in hand = 1 subject (FRF).**

### 4.2 Public MRI (subagent web sweep)
**Isotropic + dynamic (all tiny, mostly raw k-space):**
- **OSU** Zenodo 12515230 (Arshad et al., MRM 2024; code github.com/OSU-MR/motion-robust-CMR): "Motion-robust **free-running** volumetric CMR"; 3D Cartesian bSSFP cine, self-gated undersampled k-space sorted into 20 cardiac bins; 1.3×1.3×1.0 → 2.1×2.3×2.0 mm; seven 3D-cine datasets, ages 23–75, 4 patients, 1.5T+3T; also 4D flow; **raw k-space**; CC-BY-4.0. (Free-breathing vs breath-hold is INFER from the title/self-gating; Zenodo page doesn't state release N.)
- **Lausanne** Zenodo 13868462 (5D whole-heart free-running, 1.5T, **5 volunteers × 5 sequences** [bSSFP, FISS, BORR, LIBRE, LIBOR]; raw, 34 GB, CC-BY-4.0) and Zenodo 7621356 (multi-echo GRE, 5 healthy volunteers V1–V5, 25.7 GB, pilot-tone+ECG, CC-BY-4.0). Resolution unconfirmed (typical ~1.1–1.5 mm is RECALLED).
- **Zenodo 15033956** = our FRF (1 subject, 1.5 mm iso, 37.7 GB, raw + recon, CC-BY-4.0).
- **KCL fetal 4D demo** (Figshare c.4689437, code github.com/mriphysics/fetal_cmr_4d GPL-3.0): ONE fetus, 1.25 mm iso, 25 phases; the ~141-fetus cohort is not public.
- **CMRxRecon2026 4D flow** (github.com/CmrxRecon/CMRx4DFlow2026, Synapse syn64545434): 400+ cases 3D+t, **embargoed until Dec 2026**, resolution/cardiac share unknown.
- **Imperial UK Digital Heart Project "Cardiac Super-resolution Label Maps"** (Mendeley pw87p286yx/1, CC BY 4.0): HR = 3D bSSFP cine 1.2×1.2×2 mm, 60 sections, 20 phases, single breath-hold, 1.5T; LR 1.8×1.8×8 mm; **1,331 healthy adults — but the public files are ED/ES label maps (LV pool, LV myo, RV pool); images not confirmed present.** Same cohort as Oktay 2016 and the motion-correction-SR paper (PMC11144076). Obtaining the images (e.g. asking the authors) is an **unexplored, unverified lead**.

**Isotropic but STATIC:** HVSMR-2.0 (60 CHD, ~0.73×0.73×0.81 mm, CC BY 4.0), HVSMR 2016/Part II (Zenodo 4575238; 20→90 scans, ~0.9 mm; Part II CC-BY-ND 4.0 = no derivatives), MM-WHS MR (60 MR, ~1 mm resampled, registration + no-redistribution agreement; project maintenance ended, successor CARE-WHS), LGE atrium sets (2018 LA challenge 154 scans 0.625 mm; LAScarQS; RAS Zenodo 10967867 50 scans CC-BY-4.0; LGE contrast, atrium only), MSD Heart (30 scans 1.25×1.25×2.7 mm).
**Not isotropic (coverage):** CMRxRecon 2023/24/25, CMRxMotion, OCMR, Harvard radial cine, UK Biobank (SAX 1.8×1.8×8 mm + 3 LAX; application + fee), ACDC, M&Ms/-2, Sunnybrook, Kaggle DSB 2015, MESA/DETERMINE, Zenodo 10117944 (15 volunteers 2D), 4D-flow phantom 4882572, cardiac MRF (in-vivo not public).
**Searched, nothing usable:** TCIA, Kaggle (reposts only), PhysioNet, IEEE DataPort (LA 2018 mirror only), Dryad, Stanford AIMI, fastMRI, ISMRM challenges, XD-GRASP data.

### 4.3 CT / phantoms / synthetic (subagent sweep)
**No public isotropic 4D (cardiac-phase-resolved) non-MRI image dataset.** Real CT is static: ImageCAS (1000 CCTA, ~0.3–0.4 mm in-plane, 0.25–0.45 mm slices; Kaggle; Apache 2.0 per a catalog; phase not stated), ASOCA (60, 0.625 mm slices), MM-WHS CT (agreement forbids sharing), MultiD4CAD (Zenodo 15148653, 118 CCTA, likely 10–20 phases, **restricted — signed DUA**), Duke PROMISE-based 4D library (Malin 2025; 32 patients; request-only), Ruijin 4D CT (18 pts; no download found), 4D-Lung (respiratory not cardiac, 3 mm slices).
**Simulators (4D, known GT, arbitrary resolution):** XCAT (Duke, request via CVIT/Segars; labels only, no MRI texture); **MRXCAT / MRXCAT 2.0** (Buoso et al., JCMR 2023; gitlab.ethz.ch/ibt-cmr-public/mrxcat-2.0, MIT; cine + respiratory motion + texturizer; EF 34–51%, LV-only shape model; **output voxel size / phases / XCAT dependency not verified**); CMRsim (Weine, MRM 2024; GPU Bloch; repo/licence not found); JEMRIS / KomaMRI.jl / SIMRI (Bloch simulators needing a moving phantom).
**Mesh cohorts (static ED, need rasterising + an appearance model):** Rodero 2021 (Zenodo 4506930: 1000 four-chamber meshes, CC BY 4.0, built from **19** CT hearts / 9 SSM modes; 4593739: 39 meshes with fibres), Van Santvliet (Zenodo 14261122), UKB biventricular twins (Ugurlu 2025, Zenodo 15649643: 1,423 meshes, CC0, 113 GB; built from anisotropic CMR), Strocchi 2020 (N=24), Cardiac Mesh Flow (arXiv 2605.01884; 3D+t meshes from UKB CMR; weights unconfirmed).
**Label-to-image synthesis:** SynthSeg is brain-only; LISynSeg (arXiv 2608.31073, static CARE-WHS CT/MR; synthetic-only underperformed real) is the closest cardiac template, no released weights yet; no cardiac-cine SynthSeg found.
**Generative cine priors:** all trained on 10 mm anisotropic ACDC / M&Ms-2 / UKB data (CardioDiT — weights "coming soon"; residual motion diffusion; Chain of Flow) — **none isotropic**.
**Domain gap:** large for every synthetic route; nobody has measured it for a slice-to-volume task.

### 4.4 How published methods get through-plane supervision (literature)
- **In-plane self-SR as target** (Jog 2018; **SMORE** Zhao TMI 2021; **ECLARE** Remedios 2026 — handles slice gaps up to tested 1.5 mm gap/5 mm thickness; CRIS arXiv 2606.15967): validated on brain/knee/mouse/synthetic abdomen, **never cardiac**; the in-plane≈through-plane statistics assumption is untested. `INFER` (subagent): **it very likely fails for SAX** (in-plane = circular LV cross-sections; through-plane reformats = long-axis-like) and the 5–8× pitch ratio exceeds where it was validated.
- **Simulate LR from a real HR cohort:** Oktay MICCAI 2016 (3D SAX 1.25×1.25×2 mm; 1,233 healthy ED; **930 train**; multi-input model 153 SAX+LAX pairs), Steeden JCMR 2020 (500 whole-heart bSSFP volumes; did not improve diagnostic accuracy), STRMSR arXiv 2607.07581 (86 WHS volumes, ×4/×8; LAX-guided), motion-correction SR paper PMC11144076 (1,331 label-map volumes; myocardium Dice 0.97 → 0.79 sim→real). Simulated LR is gap-free blur-only, so sim-to-real drops; HR is ~2 mm, not truly isotropic.
- **Held-out-slice self-supervision at scale:** CaLID arXiv 2508.13826 (UKB 11,360 train, 1.8×1.8×8 mm; PSNR 24.2 dB, SSIM 0.778; degrades at large gaps, atria not recovered; states no paired 3D cine GT exists), DMCVR (MICCAI 2023), DeepRecon arXiv 2206.07163 (UKB 810 train; no 3D GT).
- **Learned SVR data sizes:** SVoRT 68 training volumes (FeTA), cSVR 138; fetal volumes are already isotropic. **No paper found measuring how much HR data is enough for cardiac through-plane SR.**
- **Best-justified when isotropic GT is scarce (subagent):** held-out-slice / forward-model supervision + LAX as the only real through-plane check. Content in 12 mm-pitch gaps is only recoverable via priors.

## 5. Implications for the plan
- **Cannot** do direct supervised high-res training at scale (no data). Training signal options, weakest→strongest: forward-model loss on observed slices (constrains slabs only); + LAX-plane supervision on subjects with usable LAX geometry (~290 CMRx2024 after registration; OCMR small); + small isotropic sets for fine-tuning (untested how many suffice).
- **Mixed training is natural:** observation loss on all 628 train subjects, LAX loss masked to those with LAX. Whether ~290 LAX-supervised subjects suffice for through-plane detail is untested.
- **Honest claim scope:** without a real isotropic test, claim "fine-z output + LAX-plane improvement", not "high-resolution isotropic".

## 6. Candidate evaluation sets (we may use these for evaluation)
| Source | Role | Effort | Caveat |
|---|---|---|---|
| **FRF** (1 subject, on disk) | isotropic cine GT; real respiratory states; `sax_8mm` is already the thick-slab input | none | N=1, healthy |
| **OSU** (7), **Lausanne** (5+5), **fiss** (5) | more isotropic cine after reconstruction | reconstruct from raw k-space | "GT" = our recon of undersampled data (recon error); fiss preview is soft; Lausanne res unconfirmed; OSU free-breathing status inferred |
| **HVSMR-2.0** (60, on disk) | real isotropic geometry check | low | static single phase; paediatric CHD; different contrast ⇒ out-of-distribution |
| **Held-out LAX planes** | real through-plane check, large N | OCMR (8 gated) ready; CMRx needs registration | only 3–4 planes per subject |
| **Imperial UKDHP** | shape-level GT (labels) / images if obtainable | contact authors (unverified) | public = ED/ES labels |
| **MRXCAT 2.0 / XCAT** | 4D GT, known motion | access/setup | large appearance gap |
Also run SVRTK / NeSVoR on the same simulated thick-slice inputs for comparison.

## 7. Direction under consideration: simulate the high-res dynamic GT (user's proposal, 2026-10-01 — NOT decided, nothing built)
Since no isotropic cine GT exists (§4), generate it. Two routes, which can be combined:
- **Route A — segmentation → synthetic MR (label-to-image).** Take label maps (our nnU-Net `heart_seg*` on CMRx/ACDC/M&Ms, HVSMR/MM-WHS/CARE-WHS labels, UKB/Rodero meshes rasterised at any resolution) and render MRI-like images at arbitrary (isotropic) resolution with a SynthSeg-style generator (GMM intensities, bias field, blur/noise randomisation). Closest published template: LISynSeg (arXiv 2608.31073; static; synthetic-only underperformed real, so they mix); no cardiac-cine generator with released weights exists (§4.3). Motion must be added separately (mesh/label motion, e.g. 4D SSM, Cardiac Mesh Flow, XCAT/MRXCAT 2.0 phantoms).
- **Route B — static 3D MRI → dynamic MRI.** Take real isotropic static volumes (HVSMR-2.0 60 on disk, MM-WHS MR, FRF phases) and **warp them with a cardiac motion field + our respiratory sim** to make a 4D isotropic sequence with real anatomy and appearance. Needs a **motion source**: DVFs registered from real cine (the old DVF tooling `dvf_carmen`/`dvf_elastix` was archived with the removed supervised-DVF path, docs/58), mesh motion, or a phantom — and a way to transfer SAX-cine motion onto a different heart (alignment/template). HVSMR is paediatric CHD with a different contrast, so appearance and motion priors are both off-distribution (INFER).
- **Both routes:** the simulated thick slabs are then produced by the §2 forward model (slab blur + gaps + breathing) and supervised against the simulated isotropic dynamic volume — the Oktay/Steeden recipe, with simulated rather than real HR GT.
- **Known risks (evidence we already have):** (i) docs/20: models hallucinate subject-specific appearance that is absent from the inputs, and a population template was useless — sim-trained priors may learn simulator anatomy; (ii) LISynSeg: synthetic-only < real; (iii) the sim-to-real drop in the motion-correction-SR paper (myocardium Dice 0.97 → 0.79); (iv) the realism of the added motion is only as good as its source (see docs/131: our breathing sim is already a stress test); (v) static sources have no cardiac phase, so cardiac motion is wholly invented/transferred.
- **Cheap first test of the appearance gap (proposal):** train on a small simulated set, evaluate on the real isotropic cine we do have (FRF; OSU/Lausanne after recon) and on held-out LAX planes; compare to the forward-model-only baseline. Decide A vs B (or hybrid) only after that. Open choices: which label/mesh source, which motion source, whether real cine appearance can be used as the style source for the generator (not investigated).

## 8. Open items
1. Open the CMRx `.mat` files and check for dropped slice position/orientation tags; run the LAX↔SAX registration feasibility check on a few subjects (§3.3).
2. Decide whether reconstructing the ~17 raw-k-space volunteers is worth it (not scoped).
3. Verify MRXCAT 2.0 output specs and XCAT access if a phantom route is wanted.
4. Check Imperial UKDHP for image availability.
5. Check whether CMRx2023/2025 LAX counts and M&Ms LAX presence are usable (not counted).
6. Re-open key papers before citing (most reads were abstract-only).
