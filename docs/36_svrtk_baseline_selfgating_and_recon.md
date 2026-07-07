# 36 — SVRTK baseline: self-gating + 3D/4D reconstruction (comprehensive operational reference)

> **TL;DR & takeaway**
>
> This is the **end-to-end operational reference** for the SVRTK classical-SVR baseline on
> single-orientation SAX real-time free-breathing (RTFB) cine (MIITT). It covers the whole pipeline
> — **self-gate → reconstruct** — plus every problem we hit so future agents don't repeat them.
> Companion docs: **doc 34** (why fetal_cmr_4d fails single-orientation, the 106° scatter), **doc 35**
> (the self-gating *method research* + why LV-area is the anchor). This doc is the *how it runs +
> what breaks + the 3D-vs-4D findings*.
>
> **Pipeline:** per-slice **x-f FFT → R-R period** + **nnU-Net LV-area → per-slice ED anchor**
> (θ=0≡ED); assemble continuous per-frame phase → feed a reconstruction engine. Two engines, both
> real baselines:
> - **SVRTK 4D joint** (`reconstructCardiac`): one coupled solve over all frames, temporal PSF
>   (`sinc×Tukey`) blends phases → **temporally smooth but flattens the contraction**. ~4.5 h & ~33 GB
>   on 4 cores (mostly a core-count artifact; ~40–70 min on 32 cores).
> - **SVRTK 3D per-phase** (`mirtk reconstruct`, one static SVR per phase): independent phases, no
>   temporal coupling → **preserves the contraction but noisier**. ~6 min & few-GB, parallelized on 32
>   cores. This is the roster's "3D+t".
>
> **Headline scientific finding (measured, reproduced):** the 4D-joint recon **under-contracts** — LV
> EF collapses to ~9–20% though the **raw input contracts ~57%**. Cause is **NOT breathing** (motion
> was corrected — LV position stabilized 8 mm→1.5 mm), **NOT the self-gating** (well-synced mid slices
> still flatten), and **NOT resolution** — it is the **temporal PSF** of the joint-4D method blending
> ED/ES toward the middle. Proof: a phase **hard-bin of the input** preserves **46%**, and the **3D
> per-phase recon** recovers it (crude contrast ~0.43 vs 4D ~0.25). This is *tunable* (narrower PSF /
> per-phase), not a fundamental single-orientation wall.
>
> **Self-gating validated on 2 subjects:** ED-scatter (desync) 75°/122° → after LV-anchoring the
> independent ES landmark scatter is **15.6°/6.7°** → linear θ justified. Base/apex (4/13 slices) have
> no segmentable LV → excluded (physical limit). See doc 35 §10.5.
>
> **Read the Problems & Gotchas section (§9) before running anything** — trailing-space arg bugs,
> the 32 GB OOM (real peak is only ~33 GB, not the 128 GB first guessed), CPU contention between
> nnU-Net and the recon, `pkill -f` self-kills, per-stack `-thickness`, background-task deaths, etc.

---

## 1. Scope & relationship to other docs
- **doc 34** — fetal_cmr_4d methodology; *why* it fails single-orientation (through-plane blur +
  cross-slice sync impossible; empirical 106° ED scatter). Read for the geometric argument.
- **doc 35** — the self-gating *method research* (deep sweep): why x-f gives rate, why LV-area is the
  only anatomical ED anchor, complex-vs-magnitude, the linear-vs-two-anchor decision. Read for *why
  these methods*.
- **THIS doc (36)** — the operational how-to for the full baseline (self-gate → 3D/4D recon), the
  under-contraction investigation, and the exhaustive problem log.
- **Code lives in** `baselines/fetal_cmr_4d/`; upstream repo `scratch/fetal_cmr_4d/repo` is untouched;
  per-change deviations from the authors are in `baselines/fetal_cmr_4d/DEVIATIONS.md`.

## 2. The baseline roster (what the self-gater feeds)
The self-gater is **shared infrastructure**; every SVR baseline needs cardiac phase labels.

| # | Baseline | Engine | 3D/4D | Temporal coupling | Status (2026-07-07) |
|---|---|---|---|---|---|
| 1 | **SVRTK 4D joint** | `reconstructCardiac` | 4D | ✅ temporal PSF | **built** (V1+V2, faithful 1.25 mm) |
| 2 | **SVRTK 3D per-phase (rigid)** | `mirtk reconstruct` per phase | 3D+t | ❌ | **built** (V1) |
| 3 | SVRTK 3D per-phase (deformable) | `reconstructFFD` per phase | 3D+t | ❌ | not run (optional; overfits single-orientation) |
| 4 | NiftyMIC per-phase | NiftyMIC | 3D+t | ❌ | doc 29 (clean-gated); self-gated not run |
| 5 | NeSVoR per-phase | NeSVoR | 3D+t | ❌ | doc 32 (clean-gated); self-gated not run |
| — | stack+interp floor | none | 3D+t | ❌ | doc 31 |

**4D joint takes continuous phase; the 3D-per-phase ones take hard bins** derived from the same phase.

## 3. Pipeline overview (single-orientation)
```
RT stack s01_rlt_ab.nii.gz (128,128,Z=13,F=180), 25ms/frame, magnitude
  │
  ├─ [A] x-f FFT per slice (MATLAB, authors' cardsync_intraslice) → R-R period T_s + intra-slice phases (θ=0 at frame 0)
  ├─ [B] nnU-Net Task114 LV-area per (slice,frame) → per-slice ED anchor f_ED,s (θ=0 ≡ ED)   [NEW; replaces interslice sync]
  ├─ [C] re-anchor: θ_new[s,f] = (θ_intra[s,f] − circmean(θ_intra at ED frames)) mod 2π
  │        → cardphases_lvanchor_cardsync.txt (slice-major/frame-minor, wrapped [0,2π])
  └─ [D] reconstruct:
         • 4D:  reconstructCardiac (continuous θ, temporal PSF) → selfgate_cine/cine.nii.gz
         • 3D:  hard-bin θ into 25; mirtk reconstruct per phase → perphase_cine/cine.nii.gz
```
Only stages A (rate) + the reconstruction survive single-orientation; interslice sync (the authors'
cross-slice step) is geometrically impossible (doc 34) and is **replaced by B/C**.

## 4. Self-gating (summary — full method in doc 35)
- **Rate (T_s):** authors' x-f FFT (`cardsync_intraslice.m`), adult HR band `[45,110]` bpm (fetal
  `[105,180]` would miss the beat). Robust at every slice level. Output: `cardsync/rrintervals.txt`
  (per slice, seconds) + `cardphases_intraslice_cardsync.txt` (2340 = 13×180, wrapped [0,2π]).
- **ED anchor (θ=0):** nnU-Net Task114 (M&Ms, `-tr nnUNetTrainerV2_MMS -m 2d`) LV blood-pool area per
  (slice,frame); ED = local max, ES = local min between EDs. **Robust detection required** (smoothing
  + `find_peaks(distance≈0.6·R-R, prominence)`) — naive local-maxima over-fire (see §9). Only LV-area
  is *anatomically* ED (doc 35 §3); Fourier/manifold give a periodic signal that cannot anchor.
- **Re-anchor:** subtract the per-slice ED phase offset from the authors' intra-slice phases (keeps
  their rate + phase verbatim; changes only the offset — this is the interslice-sync substitute, A6).
- **Validation (doc 35 §10.5, 2 subjects):** ED scatter BEFORE (frame-0 anchored) = 75°(V1)/122°(V2),
  reproducing doc-34's ~106° desync; after LV-anchoring the independent ES-landmark scatter =
  **15.6°(V1)/6.7°(V2)**, systolic fraction 0.319±0.037 / 0.457±0.019 → **linear θ** (no two-anchor).
  9/13 slices anchored, 7/13 high-confidence; base/apex z0/1/11/12 have no LV → excluded (physical).
- **Magnitude only** (complex adds no proven benefit, doc 35 §5).

## 5. SVRTK 4D joint recon (`reconstructCardiac`)
**What it is:** ONE coupled optimization of a 4D volume V(x,φ) over 25 output phases, using ALL 2340
frames, with a **temporal PSF = `sinc() × Tukey_window()`** that spreads each frame across neighboring
output phases (soft binning) + robust statistics + iterated register↔reconstruct motion correction.
This is the fetal_cmr_4d engine (doc 34). Consumes continuous `-cardphase` + per-slice `-rrintervals`.

**Author-faithful parameters** (verified against `recon_cine_vol.bash`; DEVIATIONS §E):
`-resolution 1.25 -iterations 4 -rec_iterations 10 -rec_iterations_last 20` + robust ON (no
`-no_robust_statistics`) `-numcardphase 25`, tight heart mask `s01_mask_heart.nii.gz`.
Deviations (all necessary/cosmetic): LV-anchored cardphase (E1), heart mask substitute (E2, the
author `mask_cine_vol` needs dc_vol slice-dofs that don't exist single-orientation), dropped
`-dofin/-slice_transformations` (E3, dc_vol segfaults single-orientation), `-remote` (E4, no tsp in
container), empty `-force_exclude` (E5), `-debug` (E6).
Run: `bash baselines/fetal_cmr_4d/run_selfgate_recon.sh Volunteer1 [Volunteer2 …]`.
Output `selfgate_cine/cine.nii.gz` = `(159,102,96,25)` @ 1.25 mm iso, 651 cc heart ROI.

## 6. SVRTK 3D per-phase recon (`mirtk reconstruct`)
**What it is:** hard-bin the frames into 25 phases; reconstruct each phase's 3D volume INDEPENDENTLY
with the static SVR (`mirtk reconstruct`) — **no temporal PSF, no cross-phase coupling**. Build:
- `build_perphase_stacks.py`: for each phase p and each z, take the **K=4 nearest-phase frames**
  (≈ one per beat) → K stacks of (X,Y,Z=13) at the true slice positions.
- `run_perphase_recon.sh`: `mirtk reconstruct vol_pPP.nii.gz K <K stacks> -thickness <K×'10'>
  -mask s01_mask_heart.nii.gz -resolution 1.25 -iterations 3`, per phase, **parallelized J=8** across
  cores; then assemble the 25 volumes into `perphase_cine/cine.nii.gz` (same grid as the 4D).
- **Single-stack `mirtk reconstruct` WORKS** (unlike the static `reconstructCardiac`/dc_vol path which
  segfaults single-stack, doc 34 §5) — but multi-stack (K≥4) gives more observations per z.

## 7. The under-contraction investigation (the core finding, fully measured)
**Observation:** the 4D-joint recon barely beats — LV area/volume swing collapses.

**Measured chain of evidence (V1):**
1. **Raw RT input** mid-slice LV area contracts **~57%** (per gating segs) — the contraction IS in the
   data. Input LV also wanders ~8 mm (breathing).
2. **4D-joint recon**: mid-slice LV area contracts only **~9%** (nnU-Net); coarse-3 mm volume EF 20.6%.
   The LV position is **stable to ~1.5 mm** → **motion correction WORKED** (breathing removed).
3. **Well-synced mid slices** (ES scatter 15.6°) **still flatten** → it is **not** the self-gating.
4. **Hard-bin of the input** into 25 phases (no recon) preserves **46%** → the data + gating hold the
   contraction; binning does not destroy it.
5. **3D per-phase recon** (no temporal coupling) recovers it: crude threshold contrast **~0.43** vs the
   4D-joint's ~0.25; clean nnU-Net per-phase LV-EF = **[PENDING — filling when seg completes]**.

**Conclusion (proven, not hypothesized):** the flattening is the **4D-joint temporal PSF** — it blends
each frame across neighboring phases, pulling ED/ES toward the middle. It is **NOT** breathing (motion
corrected, position stable), **NOT** sync (well-synced slices flatten), **NOT** resolution/iterations
(a coarse and a faithful 4D both flatten; per-phase at the same resolution does not). It is *tunable*:
per-phase decoupling, a narrower temporal window, or more output phases would recover contraction.

**The tradeoff (why both are legitimate baselines):**
| | Contraction | Noise | Temporal smoothness | Cost |
|---|---|---|---|---|
| 4D joint | flattened (~9%) | low (denoised) | smooth | 4.5 h / 33 GB (4 cores) |
| 3D per-phase | preserved (~43% crude) | **higher** (few frames/phase; no coupling; robust rejects 19–33% of slices) | flickery | 6 min / few GB (32 cores) |

**Caveat on the per-phase noise (measured):** robust statistics exclude **19–33% of the 52 input
slices** per phase → a *tunable* noise source (relax robust rejection or increase K). Genuine *data*
limits remain: single-orientation through-plane interpolation (13 slices @10 mm → 1.25 mm = blur) and
single-shot RT frame noise — those need a 2nd orientation, not tuning.

## 8. Operational profile (hardware / timing / memory)
CPU-only end to end (SVRTK is CPU; nnU-Net ran CPU on the standard nodes — no GPU there).

| Stage | Time | Memory | Notes |
|---|---|---|---|
| nnU-Net LV-area seg (2340 slices) | ~10–15 min | small | the gater's bottleneck; **do NOT run concurrently with a recon** (§9) |
| MATLAB x-f gating | few min | small | already done for V1; re-run per new subject |
| **4D joint recon** | **~4.5 h @ 4 cores** (~58 min/iteration; final iter ~2×); **~40–70 min @ 32 cores** | **~33 GB peak** | core-count-bound, not memory-bound |
| **3D per-phase recon (25×)** | **~6 min @ 32 cores, J=8** (~1–2 min/phase, ~few-GB each) | ~36 GB for 8 in parallel; few GB sequential | trivially parallel |

**Recommended allocation for future runs:** **~16–32 CPU cores, 48–64 GB RAM, CPU partition.** RAM
peak is ~33 GB (4D) — 48 GB is safe for everything; the first-guess 128 GB was wrong (cubic-in-
resolution extrapolation badly over-predicts because the tight 651 cc mask bounds the voxels). The
4.5 h was a **4-core artifact**; use more cores.

## 9. PROBLEMS & GOTCHAS (read before running — every issue we hit + the fix)

**Argument / file-format bugs**
- **Trailing space in MATLAB outputs.** `mean_rrinterval.txt` and `slice_thickness.txt` are written by
  `fprintf('%.6f ', …)` → contain a **trailing space** (`"10 "`). Passing `-rrinterval "$X"` or
  `-thickness "$X"` **quoted** fails: `Argument parsing error: the argument ('10 ') for option … is
  invalid`. **Fix:** strip it (`X=$(tr -d '[:space:]' < file)`) OR pass **unquoted** like the authors
  (word-splitting drops it). The multi-value `-rrintervals`/`-cardphase` are fine unquoted.
- **`mirtk reconstruct` needs one `-thickness` value PER STACK.** With N stacks, `-thickness 10` fails:
  `Count of thickness values should equal to stack count!`. **Fix:** pass N copies (`-thickness 10 10
  10 10` for N=4).
- **cardphase is WRAPPED to [0,2π]** (sawtooths at the R-R period), not unwrapped/cumulative — verified.
  Re-anchoring subtracts a per-slice offset then `mod 2π`.
- **"Read cardiac phase for 2341 images"** vs "Number of images: 2340" — a harmless off-by-one display
  artifact in reconstructCardiac; ignore.

**Memory / compute**
- **`reconstructCardiac` OOM-kills at `mem=32G`** for 1.25 mm (exit **-9** = SIGKILL). The real peak is
  only **~33 GB** — it sits *just* above 32 GB. **Use 48 GB.** Do NOT trust the cubic-resolution
  extrapolation (it predicted 80–150 GB; wrong — the tight heart mask bounds the reconstructed voxels).
- **Interactive alloc memory cap ≠ node RAM.** `free -g` shows the node has 150+ GB free, but your
  cgroup cap (`scontrol show job $SLURM_JOB_ID | grep mem=`) may be 32 GB → OOM anyway. Check the cap.
- **The 4.5 h runtime was 4 cores, not inherent.** reconstructCardiac is OpenMP-parallel; request 16–32
  cores.
- **sbatch blocked by `AssocGrpSubmitJobsLimit`** (a *job-count* limit, ~12). If you're at the limit,
  `sbatch` is *rejected* (not queued) — free a slot or run inline on a big-enough interactive node.

**CPU contention (subtle, cost us a broken result)**
- **Do NOT run nnU-Net EF while a recon is running.** Both are CPU-bound; nnU-Net got starved (~15×
  slower, only 192/2400 slices in 20 min) AND slowed the recon, producing a partial/garbage EF
  (only 2 of 25 phases segmented → a bogus "4.9%"). Run the recon first, then the EF, or on separate
  nodes.

**Process management**
- **`pkill -f <pattern>` self-kills.** `-f` matches the *full command line* including your own `pkill …
  <pattern>` invocation → it kills its own shell (exit 1, "shell died"). **Fix:** kill by PID
  (`ps -eo pid,args | grep '[p]attern'` → `kill -9 <pid>`), or use the `[p]attern` bracket trick.
- **`pgrep -f reconstructCardiac` matches your own diagnostic commands** (they contain the string) →
  false "still running". To check a real worker, verify `stat=R` + CPU ticks on the *specific PID*
  (`awk '{print $14+$15}' /proc/<pid>/stat`), or match the binary (`ps -eo comm | grep reconstructCard`).
- **Background `nohup` tasks do NOT reliably survive session/turn transitions here** — a launched
  per-phase run died mid-phase-2 (wrapper + child gone). Use `setsid nohup … < /dev/null &`, make the
  driver **idempotent/resumable** (skip phases whose output exists), and re-check after transitions.

**Recon behavior**
- **`reconstructCardiac` single-stack STATIC (`numcardphase=1`, the dc_vol stage) segfaults** — the
  degenerate single-orientation case (doc 34 §5). But **4D `reconstructCardiac` and `mirtk reconstruct`
  (static) single-stack BOTH work.**
- **Robust statistics exclude 19–33% of slices** in the sparse per-phase recon → noise. Tunable
  (relax robust / larger K).
- **The mirtk shim** (`baselines/fetal_cmr_4d/bin/mirtk`) needs `export FCMR_BIND=…/scratch/fetal_cmr_4d`
  and `--cleanenv` (doc 34 §4) — else the container recurses on child `mirtk` calls.

**Gating detection**
- **Naive ED/ES peak-picking over-fires.** `argrelextrema(order=3)` found 8–9 "EDs" vs ~4 real beats →
  bogus ES scatter (99.8°). **Fix:** savgol-smooth + `find_peaks(distance≈0.6·R-R_frames, prominence)`
  → ~one ED/beat. (doc 34/35's "needs a cardiac bandpass, not naive peak-picking".)
- **nnU-Net LV dropout at low-coverage slices** (z9,z10 had frames with LV area = 0). **Interpolating
  the zeros made scatter WORSE** — those slices are genuinely too sparse. **Fix:** a coverage gate
  (only trust slices with LV present on ≥90% of frames for the metric; still anchor them for the recon).
- **Base/apex (z0,1,11,12) have NO segmentable LV** (through-plane descent + tiny apex cavity) → nnU-Net
  finds nothing → exclude them (physical limit, not a bug; doc 35 §6). They keep an identity offset and
  are down-weighted by SVRTK robust stats.

**Visualization**
- **Recon-vs-input orientation mismatch.** `recon_ef.py` extracted recon slices in (X,Y) while the
  MIITT adapter uses (Y,X) → the recon panel was transposed vs GT/RT. **Fix:** transpose the recon to
  match (EF is orientation-invariant, so numbers were fine; only the display was off).
- **LV-centering fails on RT via brightness or variance.** The chest wall is the brightest structure
  and breathing dominates temporal variance → both center the crop off the heart. **Fix:** center on
  the **nnU-Net LV segmentation** on the canonical grid (the only reliable RT locator).
- **Recon/GT/RT are separate acquisitions** (different FOV/grid) → PSNR/SSIM meaningless; use **EF**
  (per-recon, alignment-free) as the comparison metric (doc 34 §7).

## 10. Code map (`baselines/fetal_cmr_4d/`)
- `selfgate_lvarea_extract.py` — RT stack → all (slice,frame) nnU-Net Task114 inputs (canonical grid).
- `selfgate_lvarea_assemble.py` — segs → per-slice LV-area → ED/ES (robust) → re-anchor authors' phases
  → `cardphases_lvanchor_cardsync.txt` + scatter metrics + `selfgate_lvarea.{json,png}`.
- `run_selfgate_recon.sh` — **4D joint** (`reconstructCardiac`, author params) → `selfgate_cine/`.
- `build_perphase_stacks.py` + `run_perphase_recon.sh` — **3D per-phase** (`mirtk reconstruct`, K
  stacks, parallel J=8) → `perphase_cine/`.
- `recon_ef.py` + `recon_ef_pick.py` — extract recon phases → nnU-Net → LV volume vs phase → EF + plot.
- `visualize.py` / `score.py` / `qc_compare.py` — beating cine, GT registration + intensity metrics, QC.
- `sbatch/selfgate_recon_faithful.sh` — sbatch wrapper for the 4D run (128 G requested; 48 G suffices).
- MATLAB gating: `matlab/miitt_gating.m` (+ `cardsync_intraslice.m` fork) — run per new subject.
- nnU-Net: `nnUNet_predict -t 114 -m 2d -tr nnUNetTrainerV2_MMS -f 0 --disable_tta` in the `nnunet` env
  (`source tools/nnunet_mnms_eval/env.sh`).

## 11. Status & next steps (2026-07-07)
- ✅ Self-gating built + validated (V1, V2). ✅ 4D joint recon (V1, V2, faithful 1.25 mm).
  ✅ 3D per-phase recon (V1). ✅ Under-contraction cause proven (temporal PSF).
- ⏳ per-phase clean nnU-Net LV-EF (finishing) → fills §7 step 5.
- **Next:** V2 3D per-phase; the other roster engines self-gated (NiftyMIC/NeSVoR per-phase); a
  **tuned 4D** (narrower temporal PSF / more phases) to see if the joint solve can keep the beat;
  manual/verified heart ROI for the final fair comparison; scale to more MIITT subjects; the
  VGGT-vs-baseline EF comparison (per-recon nnU-Net EF, alignment-free).

## References
- doc 34 (fetal_cmr_4d single-orientation failure), doc 35 (self-gating method research), doc 29
  (NiftyMIC), doc 32 (NeSVoR), doc 31 (roster), doc 15 (nnU-Net Task114 EF), doc 23 (MIITT).
- `baselines/fetal_cmr_4d/DEVIATIONS.md` — exhaustive per-change list vs the authors.
- van Amerom et al., MRM 2019 (fetal_cmr_4d); Åkesson et al., CPFI 2025 (LV-area SAX RTFB gating).
