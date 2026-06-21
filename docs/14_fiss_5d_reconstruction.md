> **TL;DR & takeaway**
>
> The **N=5 FISS free-running k-space** set (Zenodo 13868462, Safarkhanlo & Bastiaansen) ships **raw Siemens TWIX `.dat` only — no recon code**; the paper used unreleased in-house MATLAB following the **Free-Running Framework (Di Sopra 2019)**. So we must **reimplement the recon**. This doc records the **on-node inspection that fully de-risked it at the data level** (2026-06-20) and the build plan.
>
> **What we confirmed by reading one `.dat` (`twixtools`, no install, in the `elastix` env):** 3D radial **spiral phyllotaxis**, **1877 interleaves × 24 spokes = 45048 spokes**, **20 coils**, 224 readout samples (2× oversampled), matrix **112³ @ FOV 220 mm → 2.0 mm isotropic** (matches the paper), transverse slab. **The first spoke of every interleave is the exact SI navigator** (`TAGFLAG1` flag, 1877 of them, every 59 ms). Trajectory is **not stored** in the file → must be generated analytically from the phyllotaxis formula. **Self-gating is feasible**: PCA+FFT of the navigator SI-projections shows sharp, separable spectral peaks in the respiratory band (~0.12–0.2 Hz / 7–12 per min) and cardiac band (~0.7 / 1.1 Hz). Target recon: **25 cardiac × 4 respiratory** motion-resolved 5D, CS (ADMM, TV + local low-rank).
>
> **Status (2026-06-20):** **entire recon front-end validated end-to-end** on V01 FISS — only the final CS solve remains. (1) **Trajectory correct**: static gridding recon of all 43k imaging spokes → clean whole-heart 112³ volume (heart+chambers, spine, mediastinum). (2) **Self-gating works**: navigator PCA → clean ~43 bpm cardiac sinusoid + realistic respiratory signal (after rejecting the first 8 s bSSFP transient). (3) **Motion resolved**: a diastole-vs-systole binned recon shows coherent myocardial-wall differences (diff RMS ~22% of signal). Remaining: coil maps → **CS recon** (ADMM TV+LLR, 25×4, needs GPU) → per-resp 4D NIfTI. Code in `tools/fiss_recon/`. Engine env = fresh disposable `fiss-recon` (sigpy + twixtools) — **`svr`/`elastix` untouched** (user constraint). SAX reslicing handled separately (sibling task; applies to both this and the N=1 demo).
>
> **CS recon stage (2026-06-20, runs end-to-end on the A40):** `tools/fiss_recon/cs_recon.py` — readout de-oversampling (224→112 in projection space), smooth-division coil maps, per-resp-bin PDHG solve (`sigpy.app.LinearLeastSquares`, GPU) over the 25-cardiac-phase variable with 4D TV. Produces a `(25,112,112,112)` motion-resolved volume. **Quality not yet there**: isotropic 4D TV over-smooths the cardiac dimension (phases nearly identical) and is spatially blurry → needs **separate spatial vs temporal/LLR regularization tuning** + more iterations + better coil maps. GPU env needs cupy 13.x (not 14) + the pip nvidia CUDA libs on `LD_LIBRARY_PATH` + a merged header dir as `CUDA_PATH` + a one-line monkeypatch of `sigpy.linop.Embed._apply` (it hardcodes `np.zeros`, breaking `Slice` on GPU). Use `tools/fiss_recon/run.sh` which sets all paths.
>
> **Key gotchas found (all fixed):** 3D-radial DCF must be **kr² not |kr|** (|kr| → blur); **exclude the 1877 SI-navigator spokes** from imaging (identical +z spokes streak the image); **reject the first ~8 s** approach-to-steady-state transient before self-gating (else it dominates the PCA); **the thorax (~360 mm) is wider than the nominal 220 mm FOV** → you MUST reconstruct on a grid that contains the whole body (≥180³, we use 192³ = 384 mm) and **crop the central 112³** afterwards — reconstructing directly at 112³ aliases the body onto the heart (bright FOV-edge ring, blobs). The "de-oversample the readout to recon at 112³" shortcut is WRONG for this reason and was removed. CS regularizer memory (TV duals on 25×grid³) is what bounds the grid size on a 46 GB A40; 192³ fits, 224³ OOMs.

---

## 1. Context — what these datasets are

Two free-running 5D whole-heart datasets were downloaded as candidate **Tier-B** evaluation references for the gated→real-time-free-breathing transfer claim (see `docs/06` / `_html/08_candidate_evaluation_datasets.html`):

- **N=1** — Zenodo **15033956** (cardiopulmonary demo). Ships **finished 5D recons** (`recon_nufft_5D1.mat`, `recon_RLR_2Ref1.mat`; 20 cardiac × 5 respiratory). Already converted to per-respiratory 4D NIfTI by `tools/convert_freerunning_5d_to_nifti.py` → `scratch/data/freerunning_demo_15033956/{nifti,nifti_rlr}/`. **Nothing to reconstruct.**
- **N=5** — Zenodo **13868462**, *"Evaluating rapid water excitation techniques for 5D whole-heart fat-suppressed free-running cardiac MRI at 1.5T"* (Safarkhanlo & Bastiaansen). Ships **raw Siemens TWIX `.dat` only**. On disk: `scratch/data/fiss/inVivo.zip` (34 GB; download complete — the curl `416` is just resume-past-EOF). 5 volunteers `V01–V05`, each with `bSSFP / FISS / BORR / LIBRE / LIBOR` acquisitions (the paper compares fat-suppression water-excitation pulses). **This is what needs reconstruction.**

## 2. Recon code availability — none shipped

- Zenodo record = data only, no code.
- Paper Data Availability: data in Zenodo; recon = *"in-house MATLAB code following the free-running framework described by Di Sopra et al."*, gpuNUFFT + CS ADMM (TV + local low-rank), MATLAB R2022b. **Code not public.**
- The CHUV/Lausanne **Free-Running Framework** itself is not publicly released (on-request only). No public GitHub found.

⇒ We reimplement the FRF recon from raw k-space.

## 3. What we learned from the raw `.dat` (inspection, 2026-06-20)

Read `V01/meas_MID00227_FID105839_FISS_NR4_TR2p47_a50_24x1877.dat` (~2.1 GB, extracted to `/tmp`) with `twixtools` 0.24 in the **`elastix`** env (it already has `twixtools` + numpy/scipy/nibabel/h5py — **no install needed for reading**).

| Property | Value | Source |
|---|---|---|
| Trajectory | 3D radial, spiral **phyllotaxis** | `sKSpace.ucTrajectory=2`, `ucDimension=4` |
| Total spokes | **45048** | `lRadialViews`; = #image-scan lines |
| Interleaves × spokes | **1877 × 24** | `TAGFLAG1` count = 1877; filename `24x1877` |
| Coils | **20** | line data shape `(20, 224)` |
| Readout samples | **224** (2× oversampled; base 112) | `lBaseResolution=112` |
| Matrix / FOV / res | **112³ / 220 mm / 2.0 mm iso** | `lBaseResolution`, `asSlice[0].dReadoutFOV=220` |
| Slab | transverse, normal +z, thick 220 mm | `asSlice[0].sNormal.dTra=1.0` |
| Slab center offset (mm) | sag −1.9, cor 39.1, tra 21.8 | `asSlice[0].sPosition` |
| Flip angle / module TR | 50° / 2.47 ms per readout | filename `a50`, `TR2p47`; `adFlipAngleDegree=50` |
| FISS module | NR4 = 4 readouts/module (`Seg = Lin//4`) | counters |
| Scan duration | ~111 s (1877 × 59.3 ms) | 24 × 2.47 ms/interleave |

**Navigator:** the first spoke of each interleave carries `TAGFLAG1` (and line 0 also `FIRSTSCANINSLICE`). In spiral-phyllotaxis free-running, that first readout is oriented exactly along **SI (+z)** and serves as the self-gating navigator — one SI projection every 59 ms (≈16.9 Hz sampling).

**Trajectory is NOT in the file** — `IceProgramPara` is empty and no per-spoke angles are stored. The phyllotaxis directions must be **computed analytically** (Piccini D et al., *Spiral phyllotaxis: the natural way to construct a 3D radial trajectory in MRI*, MRM 2011), parameterized by (interleave i, spoke-in-interleave j), with j=0 forced to (0,0,1). **This is the single riskiest piece to get exactly right** and must be validated (see §5).

**Inspection scripts** (throwaway, in `/tmp/fiss_inspect/`): `inspect*.py` (structure), `nav.py` (navigator SI projections → `nav_projections.png`), `selfgate.py` (PCA+FFT → `selfgate_pca.png`).

## 4. Self-gating feasibility — CONFIRMED

Extracted all 1877 navigator spokes, FFT along readout → SI projection vs time. PCA across SI position, FFT the temporal PCs:
- PC1 (89% var) + PC3: clear **respiratory** energy at ~0.12–0.2 Hz (7–12 breaths/min).
- PC2/PC4/PC6/PC8: sharp **cardiac** lines at ~0.7 Hz and ~1.1 Hz (≈43 / 66 bpm).

⇒ Both physiological signals are present and spectrally separable in the navigator — self-gating into 25 cardiac × 4 respiratory bins is feasible. (Exact cardiac band + which coils to weight gets pinned during implementation, per Di Sopra: bandpass + heart-region coil selection.)

## 5. Build plan (full CS recon)

Engine env: fresh disposable **`fiss-recon`** = `python=3.11 numpy scipy nibabel h5py matplotlib` + pip `sigpy twixtools`. **`svr` (frozen torch) and `elastix` are left untouched** — explicit user constraint.

1. **Read** `.dat` → k-space `(spoke, coil, readout)` + per-spoke (interleave, j) index. (`twixtools`.)
2. **Trajectory** — analytic spiral phyllotaxis (Piccini 2011), 1877×24, j=0 → SI. **Validate**: (a) navigator dirs == (0,0,1); (b) gridding-density sanity; (c) a single-frame NUFFT recon of all spokes → recognizable thorax at 112³ (proves trajectory orientation/scaling correct) *before* any binning.
3. **Coil sensitivities** — low-res gridding recon → ESPIRiT/Walsh (sigpy `mr.app.EspiritCalib`).
4. **Self-gating** — SI navigator projections → bandpass/PCA → cardiac + respiratory signals → assign each interleave (or spoke) to one of 25 cardiac × 4 respiratory bins (end-expiration as resp ref).
5. **Density compensation** + binned **NUFFT** (sigpy) → first a plain gridding 5D recon (validate motion appears) → then
6. **CS recon** — ADMM / PDHG with spatial TV + temporal/local-low-rank across the motion dims (sigpy `mri.app` or custom linop), 112³ × 25 × 4.
7. **Output** — magnitude, 2.0 mm iso, oriented per the slab quaternion → per-respiratory 4D NIfTI (mirror the N=1 converter layout) for the downstream SAX reslice.

**Open risks:** (i) exact phyllotaxis convention/half-vs-full-sphere and golden-angle interleave rotation — validate by image, not by faith; (ii) cardiac vs respiratory band disambiguation per subject; (iii) CS regularization weights (paper used TV + local-low-rank — tune to taste, GPU not required but helps); (iv) RF-spoiling/eddy-current phase between FISS modules. Quality target: visually motion-resolved whole-heart, good enough as a Tier-B SAX reference after reslicing — not a clinical-grade reproduction.

## 6. Pointers

- Data: `scratch/data/fiss/inVivo.zip` (raw); N=1 ready NIfTIs in `scratch/data/freerunning_demo_15033956/`.
- Reading needs no install (`elastix` has `twixtools`); engine env = `fiss-recon`.
- Dataset landscape: `docs/06`, `_html/08_candidate_evaluation_datasets.html`.
- Refs: Di Sopra et al., MRM 2019 (free-running framework); Piccini et al., MRM 2011 (spiral phyllotaxis); Safarkhanlo & Bastiaansen (the N=5 source paper, PMC12638339).
