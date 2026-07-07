# 33 — fetal_cmr_4d: methodology, MIITT adaptation, and the single-orientation findings

> **TL;DR & takeaway**
>
> `fetal_cmr_4d` (van Amerom et al., *MRM* 2019; `mriphysics/fetal_cmr_4d`) is the **only
> self-gating classical 4D cardiac pipeline** in our baseline roster. It turns many **ungated,
> free-running 2D real-time slices acquired in multiple orientations** into one **isotropic 4D
> (3D+cardiac-phase) cine**, by (1) self-gating the cardiac cycle from the images, (2) rigidly
> motion-correcting bulk/respiratory motion, and (3) super-resolution-reconstructing a 4D volume.
>
> **We adapted it to run on adult, single-orientation (SAX-only), already-reconstructed MIITT
> real-time cine — and the decisive finding is that it fundamentally does NOT work on
> single-orientation data**, for TWO independent reasons, both from the missing multi-orientation
> overlap:
> 1. **Through-plane resolution** → only interpolated (blur). No second orientation to fill the
>    gaps between SAX slices.
> 2. **Cross-slice cardiac synchronization** → impossible. Interslice sync aligns slices by
>    correlating where they **overlap in 3D**, which only happens between different orientations.
>    Parallel SAX slices never overlap. **Empirically proven on MIITT V1: per-slice ED phase
>    scatters with circular std = 106°** (synced would be <25°). The reconstructed 3D volume
>    therefore blends different cardiac states across z — the LV looks wrong, not just blurry.
>
> **So the V1 recon we produced (and the `_html/27` report) is a COMPROMISED result — the method
> run outside its valid regime.** It half-beats only because each through-plane position is
> dominated by its nearest self-consistent input slice.
>
> **Two ways to make single-orientation work:**
> - **Add a real-time LAX view (2 orthogonal stacks) — the FAITHFUL fix.** A LAX slice crosses
>   every SAX slice → gives the interslice sync its overlap (scatter would collapse) AND its sharp
>   in-plane axis fills the SAX through-plane gaps (fixes blur too). Needs RT (not gated) LAX;
>   ≥3 orientations is the paper's recommendation but 2 is enough to run. **MIITT is currently
>   SAX-only** → needs an RT LAX from Jesse.
> - **Per-slice ED self-gate — OUR modification (fixes sync only, not blur).** Anchor each slice to
>   its own detected ED (max blood-pool) instead of to other slices. **Confirmed feasible**: all 12
>   V1 slices carry a detectable cardiac oscillation (median cardiac-band SNR 0.74), but 3–4 slices
>   are respiratory-drift-dominated so it needs a cardiac bandpass first (not naive peak-picking).
>   Not yet implemented/verified end-to-end.
>
> **Status / recommendation:** as-is, fetal_cmr_4d is **not a valid single-orientation baseline**.
> Options: (1) report this as a limitation finding (truthful, supports the VGGT thesis directly),
> (2) get RT LAX and run it faithfully (2+ orientations), (3) add the ED self-gate (our method,
> sync-only), (4) drop it. **Decision pending.** All code is in `baselines/fetal_cmr_4d/`;
> per-change deviations from the author code are in `baselines/fetal_cmr_4d/DEVIATIONS.md`.

---

## 1. The paper & what it does

- **Paper:** J.F.P. van Amerom et al., "Fetal whole-heart 4D imaging using motion-corrected
  multi-planar real-time MRI," *Magn Reson Med* 82:1055–1072, 2019. PDF at
  `baselines/fetal_cmr_4d/fetal_cmr_4d_paper.pdf`. This is the **magnitude** 4D recon our pipeline
  uses; a companion paper (Roberts et al.) covers 4D **flow** (we don't use it).
- **Problem it solves:** you can't breath-hold or ECG-gate a fetus, so you acquire many **ungated
  real-time 2D bSSFP slices in multiple orientations** and *retrospectively* recover a gated 4D
  (3D+time) cine in software.
- **Output:** an **isotropic 3D volume × N cardiac phases** (author default `numcardphase=25`),
  covering the **heart region** (the mask), in the reference/target stack's world frame. Because
  it's isotropic you can **reslice into any plane** (SAX, 4-chamber, …) after the fact — the
  headline capability. Through-plane sharpness of those reformats comes from multi-orientation.
- **Fetal is hard** because of stacked motion: maternal respiration + maternal bulk + **unpredictable
  fetal bulk motion (any direction, sudden, large)** + a small fast heart (~120–160 bpm) + **no
  ECG**. Adult cardiac normally removes all of this (breath-hold + ECG + heart doesn't wander); our
  MIITT case deliberately drops gating/breath-hold to simulate real-time.

## 2. The 5-stage pipeline (and how each stage works)

Framework (paper Fig 1): A acquire → B static MC → C cardiac sync → D dynamic MC → E 4D recon.
Only manual inputs in the paper are heart ROIs + a chest VOI + a target stack.

- **A. Acquisition.** Stacks of parallel 2D real-time slices, in **multiple orientations** (ideally
  first 3 mutually orthogonal to the fetal trunk, ≥3 stacks / ≥30 slices). Each slice = a series of
  real-time frames (dense in space AND time). k-t SENSE recon (Philips ReconFrame) →
  **magnitude** images (phase is inconsistent across orientations, so magnitude is used).
- **B. Static motion correction.** Take the **temporal mean of each slice** (averages out the beat →
  a static image). Then rigid registration in stages: **stack-stack** (align stacks to a target
  stack) → **slice-volume** (register each slice to a static volume reconstructed from the means,
  interleaved with reconstruction). Gives a rough spatial alignment + initial transforms.
- **C. Cardiac synchronization (the self-gating).** Two steps:
  - *Intra-slice heart rate:* temporal Fourier transform of each slice's frame series; the peak of
    the spatial-mean spectrum **inside the heart ROI** (in the expected HR band) = the R-R interval.
    Each frame's time is known → phase = `(t − t_slice_start)/RR·2π`. **NB: the FFT magnitude gives
    the RATE, not the phase offset — the code assumes the R-trigger is at each slice's FIRST frame,
    so "phase 0" = that slice's arbitrary acquisition start.**
  - *Inter-slice sync:* reconstruct a per-slice cine, compute each slice's spatial footprint
    (volume weights), find where slices **overlap in 3D**, and estimate **one temporal offset
    `thetaOffset` per slice** (applied as a `fouriershift` = temporal Fourier phase shift) that
    **maximizes overlap-weighted Pearson correlation** with already-synced slices. Solved
    **greedily**: start from the max-overlap slice (offset 0), propagate outward. This makes all
    slices agree on a shared cardiac clock. **It does NOT find absolute ED — it makes slices
    consistent with each other; ED is identified afterward from the coherent cine.**
- **D. Dynamic motion correction.** With phases known, register each real-time **frame** to the 4D
  cine frame at its matching phase (`θ_h≈θ_k`), interleaved with 4D reconstruction. Iterate.
- **E. 4D super-resolution reconstruction.** Solve the inverse problem `X = argmin Σ p·‖e‖² + λR(X)`
  on the isotropic grid: forward model = PSF-blurred sampling of X; **robust statistics** reject
  outlier voxels/frames (`p_jk`); edge-preserving regularization `R(X)`; gradient descent.

**Two motions, treated differently (key concept):**
- **Cardiac** (periodic contraction) → **resolved** into phases (the cine).
- **Respiratory + bulk** (drift of the whole heart) → **corrected** as rigid displacement
  (registered out). **NOT binned. No respiratory phase, no end-expiration assumption** — the
  reference volume is just the position the registration converges to.

**Everything is RIGID.** Grep of the paper AND the codebase: zero `deformable`/`FFD`/`non-rigid`/
`affine` (only "rigid"). Spatial motion → rigid transforms; cardiac motion → the temporal (phase)
axis. (SVRTK ships a deformable `reconstructFFD`, but this pipeline never calls it.)

## 3. Where SVRTK vs MATLAB do the work

- **SVRTK** (`mirtk` C++ binaries, in the container): all **registration + reconstruction** —
  `reconstructCardiac` in `recon_ref_vol`/`recon_dc_vol`/`recon_slice_cine`/`recon_cine_vol`, plus
  mask ops. `reconstructCardiac` **consumes** cardiac-phase labels via `-cardphase` and does the
  4D **joint** solve with a **temporal PSF** (`sinc × Tukey`) coupling neighboring phases.
- **MATLAB** (`cardsync_*`, `preproc`): the **self-gating** + orchestration.

**Consequence (important for reuse):** you can generate cardiac phases with **any** pipeline and
feed them to `reconstructCardiac` — you still get the **4D-joint, temporally-consistent** recon,
because temporal consistency is a property of the *engine* (the joint solve + temporal PSF), not the
gater. You'd only lose it by reconstructing each phase **independently** (plain `mirtk reconstruct`
on hard bins). `numcardphase=25` is a chosen output length, not data-derived (our gated GT has 30).

### 3.1 Exactly what `reconstructCardiac` consumes (and what the phase values mean)

The final-recon call (`recon_cine_vol.bash`), with each input labelled:
```
mirtk reconstructCardiac  cine_vol.nii.gz  $NUMSTACK  $STACKS \
    -cardphase   $NUMFRAME  $CARDPHASES   # one cardiac phase per frame  ← THE SYNC (the cardiac content)
    -rrintervals $NUMSLICE  $RRINTERVALS  # R-R interval per slice
    -rrinterval  $MEANRR                  # mean R-R = target output-cine cycle length
    -numcardphase 25                      # number of OUTPUT phases
    -thickness $THICKNESS  -mask $MASK  -resolution 1.25 \
    -iterations 4 -rec_iterations 10 -rec_iterations_last 20 \
    -dofin <stack.dof>  -slice_transformations <dir>   # optional rigid-moco init (from dc_vol)
```
- **`-cardphase` is the sync.** A flat list of one phase per frame, **slice-major / frame-minor**,
  matching the stack (2340 = 13 slices × 180). SVRTK **trusts it blindly — it does NOT sync**;
  coherence lives entirely here. In the author pipeline this file is the **inter-slice-synced** one
  (`cardsync/cardphases_interslice_cardsync.txt`); our single-orientation run substitutes the
  **un-synced** intra-slice file, which is the exact one line that makes our output incoherent.
- **The phase values are FRACTIONAL, continuous [0, 2π] radians** (e.g. `0.000, 0.127, 0.254, …`
  up to ~6.22), **not integer bins.** The *output* is 25 discrete phases; the continuous→discrete
  mapping is the **temporal PSF** (soft binning).
- **What a phase means:** `θ = 2π·(t − t_trigger)/RR` — a **normalized fraction of *that patient's*
  cardiac cycle**, with **θ=0 = R-trigger ≈ ED** (max-filled), wrapping at 2π. So it is
  **per-patient / self-referential**: ED is pinned to 0 for everyone, but **ES and other events land
  at a *patient-specific* fraction** (systole/diastole ratio varies with HR — at higher HR diastole
  shortens more, so ES occurs later in the normalized cycle). ⇒ **"phase 1.0" is NOT the same
  cardiac state across patients.** It is a fixed state only *within one reconstruction*, where all
  slices share the ED=0 anchor — which is precisely the sync requirement. **Same caveat this project
  already documented for VGGT's `target_t`** (docs/25 / "target_t is normalized fractional time":
  ED-anchored fraction of each subject's own R-R, ES drifts across subjects). Harmless to the recon
  (each patient reconstructed independently with its own phases + R-R); it just means phase is not a
  cross-patient absolute cardiac state.
- **Why the R-R intervals are needed:** the temporal PSF operates in **physical time**, so SVRTK
  converts phase↔time via `time = (phase/2π)·RR`. `-rrintervals` (per slice) handles the fact that
  **HR drifts over the ~1-min scan** (each slice has its own period → scales its frames' temporal
  footprint correctly); `-rrinterval` (mean) sets the **output cine's cycle length** so all slices
  map onto one common heartbeat.
- **PSF = Point Spread Function = the blur kernel** of the imaging system. **Spatial PSF** = the
  slice-thickness profile (a point smears across the ~10 mm slice); the SR forward model convolves
  the true isotropic volume with it and **deconvolves to super-resolve** — this is what multiple
  orientations enable and single-orientation cannot (→ through-plane interpolation/blur).
  **Temporal PSF** (`sinc × Tukey`) = each frame is a *window* in the cycle, spread across
  neighboring output phases → soft binning + temporal smoothness.

## 4. What we built to run it on MIITT

All in `baselines/fetal_cmr_4d/` (upstream repo `scratch/fetal_cmr_4d/repo` is byte-for-byte
untouched — verified `git status` clean). See `DEVIATIONS.md` for the exhaustive change list.
Summary:
- `export_miitt.py` — MIITT real-time NIfTI → the pipeline's post-ktrecon entry (`data/s01_rlt_ab`,
  `s01_dc_ab`) + **auto heart/chest masks from cardiac-band spectral power** (replaces manual MITK).
- `bin/mirtk` — Singularity shim routing `mirtk` calls into `fetalsvrtk/svrtk.sif`. Requires
  `--cleanenv` (else the auto-mounted $HOME leaks the shim onto the container PATH and
  reconstructCardiac's child `mirtk` calls recurse). Stages the `.sif` to node-local `/tmp`.
- `matlab/miitt_preproc.m` — rebuilds the `S` struct from MIITT's known timing (25 ms/frame,
  slice-sequential), reading slice thickness from the NIfTI header — **no Philips `PARAM`**.
- `matlab/cardsync_intraslice.m` (shadowing fork, 3 edits): adult HR band `[45,110]` (was fetal
  `[105,180]`), read RT frames from NIfTI (not Philips `xtRcn.mat`), one `isvalid()` crash-guard.
- `matlab/cardsync_interslice.m` entry + `add_paths.m` + `miitt_gating.m`.
- `scripts/recon_dc_vol_miitt.bash` — single-stack patch (drop `-stack_registration`; capture slice
  dofs with `[0-9]*`; synthesize identity stack-transform). **Only needed for the staged path,
  which does not work single-orientation — see §5.**
- Environment: `fetalsvrtk/svrtk` Docker → Singularity `.sif` (no native SVRTK build). **All CPU.**
  MATLAB R2024b on the cluster. Interactive salloc is capped at `mem=32G` (SIGKILL'd 1.5 mm recons).

## 5. The single-orientation findings (the core result)

We attempted the faithful staged wrapper AND a direct `reconstructCardiac`. Findings, in order:

1. **The staged multi-stage pipeline collapses on single-orientation.** Three distinct breaks:
   - Stack-stack registration: N/A (one stack) — degenerate "avg weight 0" segfault; patched out.
   - `dc_vol` static MC (`numcardphase=1`): **segfaults on a single stack even after the patch** —
     single-stack *static* SVR is the degenerate case (the 4D `numcardphase=25` recon is fine).
   - `slice_cine` + **interslice sync**: **inapplicable** — need slice spatial overlap, which needs
     multiple orientations (§below). Plus a dof-naming incompatibility (`transformation0.dof` vs
     `transformation00000.dof`).
   So the only stage that runs single-stack is the final **direct `reconstructCardiac`** (4D).
2. **Through-plane resolution is only interpolated (blur).** One orientation → nothing fills the
   10 mm gaps between SAX slices → SR falls back to interpolation. Not a bug — a property of the
   input. (`-resolution` is the OUTPUT isotropic voxel size; input RT is 2.3×2.3×10 mm; the recon
   super-resolves in-plane 2.3→1.25 and interpolates through-plane 10→1.25 = 8×.)
3. **Cardiac synchronization is impossible single-orientation — PROVEN.** Interslice sync needs
   slices that **overlap in 3D**, i.e. from different orientations that cross. Parallel SAX slices
   sit at different depths and never overlap. Empirical proof on V1 (`selfgate_feasibility` /
   desync analysis): the cardiac phase at which each slice's blood pool peaks (≈ED), in our gating's
   own coordinate, **scatters 0–300° across slices, circular std = 106°** (synced would be <25°).
   So `reconstructCardiac` builds each output phase from slices at **different cardiac states** →
   the 3D LV is a blend of ED/ES → looks wrong, not merely soft.
4. **⇒ The V1 recon (and `_html/27`) is a compromised result.** It half-beats
   (temporal contrast 0.19 vs GT 0.38) only because each through-plane position is dominated by its
   **nearest self-consistent input slice**; the desynced neighbors + through-plane interpolation
   corrupt + attenuate it. **`_html/27`'s "coherent, just blurry" framing was written before the
   desync was found and is misleading — it needs correcting to state BOTH failure modes.**

## 6. The two fixes

- **Real-time LAX (2 orthogonal stacks) — FAITHFUL.** A LAX slice runs along the long axis and
  **crosses every SAX slice** → non-zero 3D overlap with all of them → interslice sync works (the
  106° scatter would collapse); the LAX's **sharp in-plane axis is the SAX through-plane axis**, so
  it also fills the gaps → fixes blur. Requirements: LAX must be **RT free-breathing** (a gated LAX
  won't help); stack-stack registration (patched out for single-stack) returns to correct
  SAX↔LAX respiratory drift; ≥3 orientations is the paper's recommendation but **2 is enough to
  run**. MIITT is **SAX-only** → needs an RT LAX from Jesse. This is strictly better than the
  self-gate (fixes sync AND blur, and stays faithful).
- **Per-slice ED self-gate — OUR modification (sync only).** Anchor each slice to its own detected
  ED (max blood pool in the cardiac-filtered signal) instead of to other slices → within-slice
  landmark, no overlap needed. **Confirmed feasible**: all 12 V1 slices have a detectable cardiac
  oscillation (median cardiac-band SNR 0.74). **Caveat:** z2/z3/z5 are dominated by ~13 bpm
  respiratory drift and z7 shows a harmonic → needs a **cardiac bandpass** (0.7–2.2 Hz) + HR-based
  harmonic resolution before ED detection (not naive peak-picking). Does **not** fix through-plane
  blur (that's unavoidable single-orientation). Not yet implemented/verified (implementing +
  re-checking the scatter collapse is the proof-of-work). Replaces the paper's sync stage → no
  longer faithful fetal_cmr_4d, but SVRTK's 4D-joint recon (with temporal consistency) is unchanged.

## 7. Practical / infra notes

- **CPU-only** end to end (masking, MATLAB gating, `reconstructCardiac`). No GPU path; shim doesn't
  pass `--nv`. Production runs belong on SLURM `standard`, not a GPU node.
- **Memory:** `reconstructCardiac` peak scales ~cubically with `1/resolution` and with mask (ROI)
  size. My oversized chest mask (2.7 L ROI) at 2 mm peaked >32 GB → SIGKILL under the interactive
  `mem=32G` cap. The author's **tight heart mask** reconstructs far fewer voxels; the correct
  faithful mask is heart-sized, and `--mem` should be sized from a coarse-res probe (extrapolate
  ×(res_coarse/res_fine)³).
- **`-remote`** (author flag): "run SVR registration as remote functions in case of memory issues"
  — a memory reducer; TaskSpooler (`tsp`) is NOT in the container, so it may need dropping (infra
  deviation, not algorithmic).
- **Comparison metric:** recon (from RT) and gated GT are SEPARATE acquisitions with different
  FOV/heart position → **no voxel correspondence → PSNR/SSIM are meaningless.** Use **ejection
  fraction** (nnU-Net LV segmentation, GPU; docs/15) — alignment-free, functional, ties to docs/24/25.
  BUT: EF needs a cardiac-COHERENT ED/ES volume across slices, which the desync currently breaks —
  so EF is only meaningful once sync is fixed (LAX or self-gate).

## 8. Open decision

fetal_cmr_4d is **not a valid single-orientation baseline as-is**. Pick: (1) report the limitation
(recommended — truthful, supports the thesis), (2) obtain RT LAX and run faithfully, (3) implement
the ED self-gate (sync-only, our method), (4) drop it. Also: **`_html/27` must be corrected** to
state both failure modes (resolution + sync) rather than "coherent, just blurry".

## References
- `baselines/fetal_cmr_4d/` — all code; `DEVIATIONS.md` — exhaustive per-change list.
- `_html/27_fetal_cmr_4d_methodology.html` — explainer report (**result section needs correcting**).
- `docs/31` — baseline roster; `docs/23` — MIITT (SAX-only); `docs/15` — nnU-Net EF; `docs/24`/`25` — EF theme.
- Paper: `baselines/fetal_cmr_4d/fetal_cmr_4d_paper.pdf` (van Amerom 2019).
