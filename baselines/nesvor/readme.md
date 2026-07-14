# NeSVoR — reference for how it works + our operational decisions

**What this is.** A ground-up explanation of the NeSVoR baseline (mechanism, forward model, CLI,
container) plus the design decisions for running it on our **single-orientation gated cardiac** data
(CMRxRecon + MIITT gated, breathing-simulated eval). Companion to `docs/32` (the first end-to-end run
— bugs, results, n=2 numbers) and `scratch/eval/README.md` (the shared baseline harness). Read this
to understand *why NeSVoR does what it does* and *how we invoke it*; read `docs/32` for the run log.

Paper: Xu et al., *NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI*,
IEEE TMI 2023 (`baselines/nesvor/nesvor.pdf`). Code: github.com/daviddmc/NeSVoR (MIT, effectively
dead upstream since ~2023). We run the pinned Docker image `junshenxu/nesvor:v0.5.0`.

---

## 1. What NeSVoR is (one paragraph)

A **per-scan slice-to-volume reconstruction (SVR)** method: same *class* as SVRTK/NiftyMIC (the
alternating register ⇄ reconstruct loop over a PSF forward model `y = D·B·M·X + noise`), but it
replaces the discrete voxel-grid solve with a **continuous implicit neural representation (INR)** — a
small MLP + multi-resolution hash-grid encoding (instant-NGP style) that maps a 3D coordinate → volume
intensity. It fits a fresh network from scratch on each scan by making its own forward simulation of
the acquired slices match the observed slices. It is a **fetal-brain** method; we run it **off-label**
on cardiac. It faces the identical single-orientation through-plane limit as every SVR tool (`docs/31`
§3) — it just interpolates/regularizes *better* within that limit, it does not escape it.

---

## 2. How it works (the mechanism)

### 2.1 The representation
The unknown volume is a **function** `V(x)`, not a voxel grid — an MLP fed a hash-grid encoding of the
continuous coordinate `x`. You query intensity anywhere; nothing is discretized until you sample the
final output. That's what "resolution-agnostic" means (train once, sample any resolution).

### 2.2 The network heads and per-slice parameters
Three MLP heads share the hash-grid encoding; per queried point `x` they emit:

| Output | Meaning | Depends on | Kept? |
|---|---|---|---|
| **`V(x)`** | underlying **volume intensity** | `x` only (slice-independent) | ✅ **the reconstruction** |
| **`B_i(x)`** | **bias field** — smooth *multiplicative* shading | `x` + slice `i` | nuisance (OFF by default) |
| **`σ²_i(x)`** | pixel-wise **noise variance** | `x` + slice `i` | nuisance |

Plus **per-slice trainable scalars/vectors** (not network outputs):
- **`T_i`** — the **6-DOF rigid pose** (3 translation + 3 rotation, axis-angle) placing slice `i` in 3D
- **`C_i`** — **slice scale** (one scalar/slice; global per-slice brightness)
- **`e_i`** — **slice embedding** (16-d latent, init standard normal)
- **`ν²_i`** — slice-wise variance (one scalar/slice)

**Only `V(x)` is kept.** `B`, `C`, `σ²`, `ν²` are acquisition-nuisance terms that get **stripped off
at readout** — they exist so the shared `V` comes out clean and harmonized.

**Slice embedding `e_i`** is a NeRF-in-the-Wild "appearance embedding": a per-slice latent code that
conditions the **bias** and **variance** heads but **not** `V` — so `V` stays one shared, slice-
independent anatomy while each slice can still have its own shading (`B_i`) and corruption level
(`σ²_i`). Even with bias off, `e_i` still drives the variance head.

**Slice scale `C_i` vs bias `B_i`:** `C_i` = one number for the *whole slice* (global brightness,
softmax-reparameterized so the `C_i` average to 1 → kills the global-scale ambiguity). `B_i(x)` =
smooth *spatial* shading that varies *within* a slice (fed only low-frequency hash levels so it can't
steal anatomy from `V`; regularized to `mean(log B_i)=0`). **`B_i` is OFF by default** (`--n-levels-bias 0`),
so for us `B_i ≡ 1` and the model simplifies.

### 2.3 The forward model (per acquired pixel)
For pixel `j` of slice `i`:
1. `p_ij` = the pixel's position in the slice's own 2D frame.
2. Draw **K samples** `u_k ~ N(0, Σ)` from a **3D anisotropic Gaussian = the PSF**, with
   `Σ = diag(σ_x², σ_y², σ_z²)`: **narrow in-plane** (`σ_x,σ_y` from pixel spacing ~1.4 mm), **wide
   through-plane** (`σ_z` from slice thickness ~8 mm). FWHM: in-plane ≈ 1.2×pixel, **through-plane FWHM
   = slice thickness** → this is why `--thicknesses` matters (it sets `σ_z`).
3. `x_k = T_i ∘ (u_k + p_ij)` — offset in the slice frame, then rigidly mapped by the pose into volume space.
4. Query and average: **`Ī_ij = C_i · mean_k[ V(x_k) ]`** (× `B_i(x_k)` inside the average if bias on).

**PSF = point spread function** = the blur kernel that models a thick slice integrating signal over
its thickness. NeSVoR approximates the true (sinc-ish) profile as a 3D anisotropic Gaussian; the
Monte-Carlo sampling is just how it approximates the convolution integral. So the sampling **is** the
thick-slice simulation.

### 2.4 The loss (self-supervised — no GT)
Gaussian negative-log-likelihood per pixel: **`(I_ij − Ī_ij)² / (2σ²_ij) + ½·log σ²_ij`** — i.e.
**`L2/var + logvar`**: heteroscedastic L2 (error down-weighted by predicted variance → automatic
outlier rejection; a corrupt slice gets a big `σ²` and is softly ignored) plus a log-variance term
that stops `σ²→∞`. Plus one image regularizer: edge-preserving **TV on `∇V`** (`--weight-image 1.0`).

**Crucial:** `Ī_ij` is compared to the **acquired *input* slice pixel `I_ij`**, NOT to any ground
truth. NeSVoR only ever sees the input slices; there is **no GT volume in its loss**. It drives `V` so
that re-simulating the slices reproduces the observed slices. (Our eval GT is separate — *we* use it
afterward to score the finished volume; NeSVoR never touches it.)

### 2.5 Optimization (everything at once)
One Adam optimizer, one loss, **all parameters jointly**: MLP weights, the hash grid, the per-slice
poses `T_i`, embeddings `e_i`, scales `C_i`, and log-variances. LR 5e-3, decays ×⅓ at 50/75/90% of
training; default **6000 iterations**, batch 4096, K=256 PSF samples, mixed precision.

So **registration happens *inside* reconstruction** — the poses are just more trainable parameters
refined every iteration. This joint non-convexity is why pose has a **limited capture range** (local
minima) — which is the entire reason a separate pose *initialization* exists (§2.6).

### 2.6 Registration = pose INITIALIZATION only
`--registration` sets the *initial* `T_i` before training; training then refines them by gradient
descent. Options:
- **`svort` / `svort-*`** — SVoRT, a learned transformer that predicts poses. **Trained on fetal brain
  → out of domain for cardiac. Do NOT use.**
- **`stack`** — stack-to-stack rigid registration: aligns *multiple stacks to each other*. The paper
  runs this only "when there are more than one input stacks." **On a single stack it's a literal
  no-op** → nominal header geometry, identical to `none`.
- **`none`** — init to the nominal slice geometry (header).

The output volume readout is bare `V` (× the output PSF only, Eq. 20) — **no `C_i`, no `B_i`.**

### 2.7 Deformable mode (not for us)
`--deformable` adds a **separate continuous neural displacement field** (its own coarse hash grid,
`--coarsest/finest-resolution-deform` 32→8 mm, its own per-slice deformation embedding) that warps the
query point before sampling `V`. It is **NOT** b-spline control points and **NOT** a stored dense DVF —
it's another INR mapping coordinate → displacement, regularized to be smooth. Meant for large non-
rigid ROIs (uterus/placenta), tagged experimental. **Our respiratory corruption is pure rigid → we use
rigid only; `--deformable` would let it fit motion that isn't in the data.**

---

## 3. The container (how we run it)

We run the pinned Docker image as a **Singularity/Apptainer** `.sif`
(`scratch/nesvor/sif/nesvor.sif`, 5.4 GB; Docker needs root, Singularity runs as a normal user → HPC
standard). **A container is not a VM and does not emulate hardware.** It is a **sealed filesystem
bundle** — a minimal Linux userland + Python + PyTorch + CUDA *userspace* libraries + tiny-cuda-nn +
the NeSVoR source + every dependency, frozen at the versions NeSVoR was built for. Its processes run
**natively on our host kernel, CPU, and A40 GPU** (near-native speed, no virtualization); Linux
namespaces just give them a different *view* of the filesystem. Think "chroot on steroids," not "a
computer inside a computer."

- **`--nv`** lends the container the host's NVIDIA driver + GPU devices (the container ships CUDA
  libraries but uses the host driver) — required, or it can't see the GPU.
- **`--bind <host>:<container>`** exposes our directories so it can read input NIfTIs and write recons.
- The **NeSVoR source ships inside** the image (that's how `docs/32` traced the mask bug by reading
  `nesvor/inr/data.py`); we invoke the `nesvor reconstruct` entrypoint, we don't have a separate clone.

---

## 4. Our operational decisions (single-orientation gated cardiac)

| Flag | Value | Why |
|---|---|---|
| `--registration` | **`none`** | single stack → `stack` is a no-op (nominal geometry either way); `none` is honest + marginally faster (~3 s). Poses still refined during training. |
| `--stack-masks` / `--sample-mask` | **whole-heart ROI** (`mask_heart.nii.gz`) | `--stack-masks` masks the INPUT slices (focus fit on heart); `--sample-mask` bounds the OUTPUT sampling. **`--sample-mask` is MANDATORY** (§5). |
| `--thicknesses` | **`8.0`** | the REAL slice thickness = PSF through-plane FWHM. NOT the 12 mm canonical *pitch* (8 mm thickness + 4 mm gap). |
| `--output-resolution` | **`1.4`** iso | match SVRTK (`-resolution 1.4`); both are resampled to the canonical `(1.4,1.4,12)` grid to score, so finer (0.8) gains nothing we measure. |
| `--device 0`, `--nv` | GPU | INR training needs the GPU. |
| everything else | **defaults** | don't tune a baseline: `--n-iter 6000`, bias off (`--n-levels-bias 0`), pixel/slice variance ON, edge-preserving TV, etc. |

**Intensity handling (PER-METHOD, evidence-driven — NOT a uniform rescale).** Measured in-ROI on
Train_P053 (`prep_recon` in `scratch/eval/engine/assemble_and_gif.py`):
- **SVRTK preserves `[0,1]`** (fit `k=1.05, c=−0.01` vs GT) → **score AS-IS**. Self-normalizing it
  *loses ~1.9 dB* of real reconstruction signal (29.85→27.93) — so a blanket uniform rescale is WRONG.
- **NeSVoR is an arbitrary 700-mean gauge** (`k≈12000`, negligible offset) and **NiftyMIC a +1.0 offset**
  (`c≈+1.0`) → **self-percentile `[0,1]`** using the recon's OWN 0.5/99.9 in-ROI percentiles (the GT
  recipe; no GT reference → no leak). NeSVoR −64.7→sane, NiftyMIC −0.15→sane.

**Do NOT use the old `k·gt+b` calibration** (`baselines/nesvor/score.py`): it regresses against GT →
leaks the answer and inflates PSNR (⚠️ `docs/29`). `--output-intensity-mean` alone won't fix it (pins
the mean, not the range). The rule is keyed on measured scale-preservation (`SELF_NORM_METHODS =
{"nesvor","niftymic"}`); a new method → measure its scale first, then decide.

---

## 5. Gotchas

- **`--sample-mask` is MANDATORY on our data** (`docs/32`). Omitted, NeSVoR estimates the output ROI by
  voxelizing slice points → Gaussian-blur → threshold; on a **single stack with ~6–8:1 anisotropy**
  that comes out **all-`False`** → `IndexError` in `Volume.resample()` **after ~7 min of otherwise-
  successful training** (easy to misdiagnose). Pass our heart ROI explicitly — a documented escape
  hatch, not a hack.
- **`--output-json` crashes v0.5.0** (`TypeError: Object of type dtype is not JSON serializable`) —
  but only *after* the volume + model are written, so the file-based success gate still promotes the
  recon. `engine/run_nesvor.sh` **omits `--output-json`** for a clean exit; poses (if ever needed for a
  motion metric) live in `model_t*.pt`.
- **Intensity scale** (§4) — must normalize; scoring raw output gives garbage PSNR.
- **Timing MEASURED (Train_P053, `--registration none`, defaults K=256/6000-iter):
  V100 = 195 s/fit; A40 = 462 s/fit → the V100 is 2.4× FASTER.** The v0.5.0 image's tinycudann is
  compiled for **cc70 (V100)**; on the A40 (cc86) the hash-grid kernels run via forward-compat at ~2.4×
  the cost (the "suboptimal for cc70" warning is a real 2.4× penalty, NOT minor — an earlier note
  dismissing it as a red herring was wrong, refuted by measurement). **Run NeSVoR on a V100**, not the
  A40. (V100 195 s even beats the paper's adult-brain 6.13 min on a V100 — that used K=128/high-res.)
- **ONE fit SATURATES the GPU (98% util, 2.8 GB)** → **single-GPU concurrency gives NO speedup**
  (measured: J=4 just time-shares, ~4× slower per fit). **J=1 is optimal per GPU; parallelize ACROSS
  GPUs** (sbatch array by subject; the runner is idempotent). Cost is fixed ~462 s/fit.
- **Per-phase 4D cost**: full both-cohorts ≈ **81 GPU-hours on V100** (CMRx 30×12×2 ≈ 39 h + MIITT
  13×30×2 ≈ 42 h) — was ~192 h on the A40. MIITT T=30 is the driver. Budget via a multi-GPU array or
  pilot first. K=128 (paper's value) would roughly halve this again.
- **Clean-variant finding:** default NeSVoR reconstructs **~6–7 dB BELOW SVRTK** on the clean round-trip
  (22.3 vs 29.9 dB, honest self-norm; calibration only +1.3 dB), uniformly across planes — INR+TV
  smoothing vs SVRTK's sharper solve in this coarse single-stack regime. The meaningful test is the
  *breath* variant (motion correction); on clean NeSVoR is a baseline data-point, not a winner.
- **GPU required** — `spgpu` partition, `--nv`. (SVRTK is CPU; NeSVoR is not.)

---

## 6. Files & status

- `run_nesvor.sh` — **OLD** single-phase runner on NiftyMIC's exported `t0` stacks (`scratch/niftymic/data`),
  clean + `resp_stack` variants (`docs/32`). Pre-dates the `scratch/eval` harness.
- `score.py` — **OLD** scorer with the `k·gt+b` calibration (superseded by the uniform percentile rule, §4).
- `nesvor.pdf` — the IEEE TMI 2023 paper.
- `scratch/nesvor/` — the `.sif`, `recon/` (clean), `recon_resp_stack/` outputs from the first run.

**BUILT (2026-07-12):** `scratch/eval/engine/run_nesvor.sh` — GPU sibling of `run_svrtk3d.sh`, same
env/positional contract + output layout + idempotency/atomic-move. Consumes the SAME frozen bundle
(`clean`/`breath` stacks + `mask_heart`), writes `<subject>/nesvor/recon_<var>/vol_t{NN}.nii.gz` +
`model_t*.pt`, reuses `assemble_and_gif.py` / `aggregate.py` with `prep_recon`'s per-method norm (§4).
Flags: `--registration none --thicknesses 8 --sample-mask <heart ROI> --output-resolution 1.4 --device 0`,
defaults else. **Bind gotcha:** bind the subject dir to a SIMPLE container path (`--bind "$SD:/data"`)
and use `/data/...` paths — NOT the host absolute path, whose `scratch`→`/gpfs` symlink doesn't resolve
inside the container (NeSVoR's `makedirs` walks the output path up to the missing symlink and dies).
Status: faithfulness + concurrency validated on Train_P053; single-subject visual check → then cohorts
(inline per user, gated on the visual). See `scratch/eval/README.md` for layout + breathing determinism.

Companion docs: `docs/32` (first run + bugs), `docs/31` (baseline roster + single-orientation limit),
`docs/29` (NiftyMIC + the calibration ⚠️), `docs/36` (SVRTK), `scratch/eval/README.md` (the harness).
