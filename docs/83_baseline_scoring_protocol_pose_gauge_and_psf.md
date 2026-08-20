# 83 — Baseline scoring protocol: the pose gauge, the PSF operator, and what a fair head-to-head requires

> **TL;DR & takeaway**
> Scoring VGGT against the classical SVR baselines has **two real protocol defects**, both now
> **measured on real data** (Test_P012, breath, t00, heart&FOV ROI) rather than argued:
> **(1) a free global pose gauge** — SVRTK's recon sits **+4.0 mm** off GT in z, NeSVoR **−4.0 mm**
> (opposite directions!), while **both VGGT arms sit at 0.0 mm** — worth **+0.62 / +0.35 dB** to the
> baselines and **+0.00 dB** to us; and **(2) a wrong resampling operator** — `assemble_and_gif.py`
> point-samples a thin plane where the physics is an 8 mm slab integral, worth **+0.55 dB** to SVRTK
> and **−0.65 dB** to NeSVoR. **Both are ~0.5 dB, NOT the 20-28 dB two independent debate agents
> predicted** — their estimates assumed genuine through-plane detail that a *single-stack* recon
> cannot produce (measured: SVRTK is only 13% sharper than GT in z). This matters because the real
> margin is small: on `breath` we lead SVRTK by **2.32 dB anchored → ~1.7 dB pose-corrected →
> ~1.2 dB after both fixes**. Also found: **on `clean` SVRTK BEATS us by 3.2 dB** (28.08 vs 24.91),
> our advantage is **breathing robustness** (SVRTK loses 8 dB to breathing, we lose 2.5), the
> baselines currently receive **more target-phase information than we do**, and **we are much
> blurrier than GT in z** (0.093 vs 0.146) so part of our PSNR lead is smoothness, not fidelity.
> **A fair comparison IS achievable** — the recipe is in §6. **No literature consensus exists**
> for this exact situation (Ferrante & Paragios 2017 §7); we must define and defend the protocol.

---

## 1. Why this doc exists

The question that started it: *are the SVRTK / NeSVoR / NiftyMIC / FC-SVR outputs comparable to
ours at all, given they reconstruct in their own reference frame and at their own resolution?*

Two distinct worries turned out to be real, and a third (bigger) one surfaced along the way:

| # | worry | verdict |
|---|---|---|
| A | classical SVR has **no absolute anchor** — its recon can float relative to GT | **REAL, measured** (§3) |
| B | isotropic recons are **resampled wrongly** onto the thick GT grid | **REAL but small** (§4) |
| C | the arms **do not receive the same input information** | **REAL, undocumented** (§5) |

Everything below is measured unless explicitly labelled as unverified.

**Provenance of the measurements.** All numbers come from
`_archive_prenativez_20260712/cmrxrecon/out/Test_P012`, phase `t00`, `breath` variant unless
stated, scored inside `mask_heart & mask` (heart&FOV ROI), SVRTK `-1` sentinel clipped to 0,
NeSVoR divided by its own in-ROI p99.9 (the `PURE_SCALE_METHODS` rule). Probe scripts were written
to the session scratchpad; **they are not yet in `tools/`** (see §7 TODO). ⚠️ **n=1 subject,
1 phase, translation-only, 1 mm granularity, no rotation, and the data is pre-native-z stale.**
Treat every number as **directional evidence, not a result**.

---

## 2. The current protocol, and where it is wrong

`evaluation/engine/assemble_and_gif.py`:

```python
SPACING_XYZ = (1.4, 1.4, 12.0)
SHAPE_XYZ   = (256, 256, 12)

def load_canon(path):
    img = nib.load(path)
    if tuple(img.shape[:3]) == SHAPE_XYZ and np.allclose(img.affine, canon_affine()):
        return np.asarray(img.dataobj, dtype=np.float32)          # ← identity short-circuit
    out = nibproc.resample_from_to(img, (SHAPE_XYZ, canon_affine()), order=1, cval=0.0)
```

Two consequences:

1. **`order=1` is trilinear = a thin-plane point sample.** But a GT voxel is the MRI signal
   **integrated over the 8 mm slice profile** (12 mm pitch = 8 mm thickness + 4 mm gap, docs/27).
   SVRTK and NeSVoR are explicitly *told* the thickness is 8 mm (`-thickness 8`,
   `--thicknesses 8`) — **the pipeline models the slab going in and discards it coming out.**
2. **The short-circuit fires only for VGGT and FC-SVR** (already on the canonical grid), so the
   three isotropic arms eat a resample that the two anisotropic arms never pay.

There is also a **latent native-z inconsistency**: `SHAPE_XYZ`/`SPACING_XYZ` are hardcoded to
12 planes at 12.0 mm, while `build_inputs/cmrxrecon.py:91` writes GT at **per-subject `dz`, native
`D`**. For CMRx2024 (uniform 12 mm pitch) a short subject is merely zero-padded — benign. For OOD
cohorts with `dz≠12` this would resample **GT itself**. ⚠️ **Incidence unverified** — the code is
aware `D` varies (`canonical_disp = len(disp_mag) == SHAPE_XYZ[2]`, line 270), so it may be
deliberate. **Check before touching.**

---

## 3. Defect A — the free pose gauge (REAL, ~0.5 dB)

### 3.1 Mechanism

The respiratory simulation is **one-sided**: `respiratory.py:44`, `d(r) = A·sin(πr)^(2n)`, n=3, so
`sin⁶ ≥ 0` **for all r**. With `amplitude_mm: 18.8` and `E[sin⁶] = 5/16`, the expected
displacement is **≈5.9 mm** — a systematic DC offset, not zero-mean jitter. (Test_P012's realized
mean is `d_Z=+4.62, d_Y=+2.44, d_X=−0.27 mm`.)

Classical SVR enforces only **slice-to-slice consistency**; nothing references GT. Shift every
slice by a common vector and it converges to a perfectly consistent volume sitting that far off
GT, and reports no error. **VGGT is different**: it predicts Δ off absolute `scanner_coords` and is
trained against GT, so it has an absolute anchor and **no free gauge**.

### 3.2 Measured (coordinate-descent translation search, ±8 mm, 1 mm steps)

| arm | anchored | after shift | gain | fitted shift (x,y,z) mm |
|---|---|---|---|---|
| `svrtk3d` | 20.07 | 20.68 | **+0.62** | (0, 0, **+4.0**) |
| `nesvor` | 17.72 | 18.07 | **+0.35** | (0, 0, **−4.0**) |
| `vggt_..._gather05_ep99` | 22.39 | 22.39 | **+0.00** | (0, 0, **0.0**) |
| `vggt_..._aug_moderate_ep99` | 22.26 | 22.27 | +0.01 | (0, −1.0, 0) |

**Three findings:**

1. **The gauge is real and matches theory.** SVRTK floats **+4.0 mm** against an applied mean
   `d_Z` of **+4.62 mm**.
2. **It must be MEASURED per arm, not derived.** NeSVoR floats **−4.0 mm — the opposite
   direction.** ⚠️ An earlier idea in this investigation ("just subtract the mean displacement from
   `manifest["breath"]["disp_dhw_mm"]`") **would have corrected NeSVoR backwards.** Each method's
   gauge is set by its own initialization/regularizer, not by the applied field. **Do not derive
   the correction analytically.**
3. **VGGT is empirically anchored** (0.0 mm, +0.00 dB). This is what makes a symmetric protocol
   defensible: we run the *same* search on every arm, and ours returns zero. The asymmetry is in
   the **result**, not the **treatment**.

### 3.3 ⚠️ These gains are UPPER BOUNDS

The search **maximized the PSNR that is then reported** — fitting directly on the reported metric.
The rigorous version fits the transform on a criterion that is **not** the score (mask centroid +
principal axes, or NMI on a held-out phase), freezes it, then applies it to all phases. Expect the
honest gain to be **smaller** than +0.62 / +0.35.

### 3.4 Where to apply the correction (interpolation safety)

**Never translate a volume on the 12 mm grid.** A 5.9 mm shift is ~0.5 voxel there, so each output
voxel becomes a blend of two voxels **12 mm apart** — a ~6 mm smoothing kernel. A debate agent
measured a **perfect, exactly-cancelling** ±5.9 mm round trip at 12 mm still degrading GT to
**~25 dB**. At 1.4 mm the same shift is 4.2 voxels (4 whole + a 0.2 blend across 1.4 mm) —
negligible.

Split by **what each method outputs**, not by grid resolution:

| method | outputs | apply correction by |
|---|---|---|
| VGGT | continuous 3D positions (`world_points`) | shift the coordinates, **re-splat** |
| FC-SVR | a displacement field | `compensate_motion` → `reconstruct_native` (**its native path**) |
| SVRTK / NeSVoR / NiftyMIC | only a volume | shift the **1.4 mm** volume, *then* PSF-downsample |

Every arm then pays **exactly one** rendering/resampling step. ⚠️ Correcting-before-rendering is
**one interpolation instead of two — not zero**: the splat still quantizes onto the 12 mm grid.

**FC-SVR already does this correctly** (`baselines/fcsvr_cardiac/pipeline.py:168-175`):
`compensate_motion` (orthogonal Procrustes — rigid, **rotation + translation** — fitting predicted
displacements to GT displacements, `cardiac.py:287-305`) runs **before** `reconstruct_native`
splats. That is why its readme labels the arm **"GT-pose-normalized"** and states it
"must not be described as deployable" (`readme.md:394-399`). Its `raw` column is the uncorrected
one; `pipeline.py:167` emits both.

**Gauge dimensionality:** a global rotation is as invisible to a slice-consistency objective as a
global translation, so the true gauge is **6-DOF**. The defence against a correction absorbing real
error is **off-metric fitting** (§3.3), *not* restricting DOF. (Our corruption is pure translation,
so translation-only is a defensible simplification; state it if used.)

---

## 4. Defect B — the resampling operator (REAL but SMALL, ~0.5 dB, sign varies)

### 4.1 What two debate agents predicted, and why they were wrong

Two independent agents each estimated the point-sample-vs-slab error at **20-28 dB**, i.e. that a
*perfect* isotropic reconstruction would ceiling at ~20-28 dB. **Both were refuted by measurement.**
Their error: they assumed genuine high-frequency through-plane structure. One used an **in-plane**
axis as a proxy (in-plane is far sharper); the other used a synthetic volume with 25% fine-scale
amplitude.

**The single-stack insight (from the user, then confirmed):** SVRTK/NeSVoR output on a 1.4 mm
isotropic **grid**, but with **one stack** there is no orthogonal through-plane information. The
grid is oversampled; the content is not.

### 4.2 Measured — through-plane sharpness (RMS finite difference, eroded heart ROI)

| volume | z-grad | x-grad |
|---|---|---|
| `svrtk3d` trilinear | 0.1644 | 0.0519 |
| **GT** | **0.1456** | **0.0456** |
| `svrtk3d` psf8mm | 0.1362 | 0.0448 |
| `nesvor` trilinear | 0.1550 | 0.0544 |
| `nesvor` psf8mm | 0.1384 | 0.0516 |
| **VGGT** (gather05) | **0.0928** | — |

**SVRTK is only 13% sharper than GT in z.** There is essentially **no through-plane
super-resolution** to preserve or destroy — which is exactly why the operator change is worth so
little.

### 4.3 Measured — operator effect on PSNR

| arm | trilinear | psf8mm | Δ |
|---|---|---|---|
| `svrtk3d` | 20.07 | 20.62 | **+0.55** |
| `nesvor` | 17.72 | 17.07 | **−0.65** |

**The sign differs per method.** NeSVoR's output is already smooth enough in z that an 8 mm PSF
**over-blurs** it.

### 4.4 Do we apply the PSF to VGGT? NO — and here is the evidence

The PSF converts an **anatomy estimate** into a **measurement**. SVRTK/NeSVoR are posed as
deconvolution (find `x` s.t. `PSF(x)` matches the slices), so their output is an anatomy estimate
and the PSF is correct by their own formulation. **VGGT does not estimate anatomy** — it
repositions acquired slice content, and each input slice is *already* an 8 mm-averaged measurement.

Measured confirmation: **VGGT z-grad = 0.0928 vs GT 0.1456** — we are already **much blurrier than
GT**. Applying an 8 mm PSF would double-blur something already smoother than the reference.
**Skipping it is the correct operator, not a favour.** This is the one genuinely asymmetric step in
the protocol; justify it with this measurement, not with an argument.

⚠️ **Uncomfortable corollary:** blur is MSE-optimal under uncertainty, so **part of our PSNR lead is
smoothness, not fidelity** (classic L1-trained-model regression-to-the-mean). SVRTK being *sharper*
than GT means it is doing something qualitatively different. **This is a real argument against a
PSNR-led headline** — see §6.5.

---

## 5. Defect C — the arms do not receive the same information (REAL, previously undocumented)

The frozen bundle guarantees **byte-identical input data**. It does **not** guarantee equal
**information**. Verified in `evaluation/engine/run_vggt.py:167-176` (`build_slots`, `onef`):

```python
slots = [(ref_k, 0)]                       # slot 0 = reference plane at the QUERIED phase
n = 1 if regime == "onef" else ...
for k in range(n_entries):
    if k == ref_k: continue
    s0 = int(rng.integers(T))              # ← each other plane: ONE frame at a RANDOM phase
    slots += [(k, (s0 + j) % T) for j in range(n)]
```

| arm | receives, for target phase `t` |
|---|---|
| SVRTK / NeSVoR / NiftyMIC | `breath/stack_t{t}.nii.gz` = **all D slices at phase `t`** (fully gated) |
| VGGT (`onef`) | **1 slice at phase `t`** (the reference) + **D−1 slices at random phases** |

**The baselines get strictly more target-phase information than we do.** Our 2.32 dB `breath` win
is achieved *despite* that handicap. This must be **stated explicitly** — it is in our favour to
disclose, and a reader who discovers it unaided will assume it was hidden.

Two defensible framings:
- **(a) each method gets its natural input** (current). Generous to the baselines; conservative
  headline. **Recommended.**
- **(b) matched scattered input** — feed SVRTK the same scattered stack. It would collapse (it
  assumes one cardiac phase per stack). Arguably unfair to it, but demonstrates *why* classical SVR
  cannot do the real task. Worth **one clearly-labelled row**, not the headline.

---

## 6. What must be done to score properly

### 6.1 Prerequisite — the baselines must be re-run

`evaluation/volumes/cmrx2024/out/` contains **only** the frozen bundle + `vggt_augaggr224hw2_ep300`
across its 29 subjects. **No `svrtk3d/`, `nesvor/`, or `niftymic/` dirs exist.** The extant
`results/cmrxrecon/{svrtk3d,nesvor}.json` are dated **Jul 23**, predating both the native-z
refactor (`_archive_prenativez_20260712`) and the 2026-07-31 slice-order fix — **stale** per
CLAUDE.md. **Therefore fixing the protocol first costs nothing.**

### 6.2 The scoring pipeline

1. **Estimate a pose transform for EVERY arm** with the *same* procedure, fitted **off-metric**
   (§3.3). Ours should return ≈0; if it does not, that is a **finding to chase**, not something to
   silently correct.
2. **Apply it per §3.4** — coordinates/field for VGGT and FC-SVR, native 1.4 mm volume for the
   classical arms. Always resample from the **original**, never from a previous iterate.
3. **Downsample the isotropic arms with an 8 mm PSF** (Gaussian, FWHM = slice thickness
   through-plane, 1.2× voxel in-plane — Kuklisova-Murgasova 2012, verified verbatim). **Do not
   apply it to VGGT/FC-SVR** (§4.4).
4. **Score in the heart&FOV ROI.** Report `psnr_3d_bbox`-style ROI metrics, plus `ncc()` which is
   invariant to the `SELF_NORM_METHODS`/`PURE_SCALE_METHODS` intensity gauge.
5. **Report BOTH columns** — anchored and pose-corrected — for every arm, plus the fitted transform
   per arm (it is itself a result: it says whether a method has an absolute anchor).
6. **Report both operators** (trilinear and PSF) while the sign is method-dependent (§4.3).

### 6.3 Drop `clean` from the headline

`build_inputs/cmrxrecon.py:125-127` writes the clean stack as **the GT planes themselves**
("clean stack == GT planes (the SVR upper-bound: nothing to correct)"). **A method that copies its
input scores perfectly.** `clean` is a sanity check; `breath` is the benchmark.

### 6.4 The margin, after corrections

| | VGGT vs SVRTK, `breath` |
|---|---|
| anchored, trilinear | **2.32 dB** |
| + pose correction | **~1.71 dB** |
| + PSF operator | **~1.2 dB** |

Real, but thin. **This is why ~0.5 dB terms matter** — they are 20-45% of the margin.

### 6.5 The headline should not be PSNR

The full picture on `breath`/`clean` (Test_P012, t00, heart&FOV ROI):

| arm | clean | breath | Δ (breathing cost) |
|---|---|---|---|
| `svrtk3d` | **28.08** | 20.07 | **−8.01** |
| `nesvor` | 20.09 | 17.72 | −2.37 |
| `vggt_gather05` | 24.91 | **22.39** | **−2.52** |
| `vggt_aug_moderate` | 24.60 | 22.26 | −2.34 |

**SVRTK beats us by 3.2 dB on clean input.** It is a genuine deconvolution method; we are a
repositioner. (Note we score only 24.91 on `clean` even though the clean input **is** GT — our
model moves content away from GT even when nothing needs correcting. **Open question worth
chasing.**)

**Our contribution is not "we reconstruct better" — it is "we are robust to breathing corruption
that classical single-stack SVR cannot handle."** SVRTK loses 8 dB; we lose 2.5. That is the
defensible claim and the numbers support it.

Combined with §4.4 (we win partly by being smooth), the recommendation is to **lead with
EF / volumes / Dice**, not PSNR. This matches cardiac practice: the closest published work to our
application (Wrobel/Steeden/Muthurangu 2025, arXiv:2506.22532 — real-time 2D stacks → isotropic 3D
cine) reports **no voxel metric at all**, only LV/RV volumes, EF, and Bland-Altman. CMRxRecon2024's
own organisers argue in print that SSIM/PSNR/NMSE "often fail to capture clinically relevant image
quality"; CMRxRecon2025 ranks **SSIM restricted to the heart region** (independent support for
bbox-over-full). It also matches this project's own prior finding
(`project_oracle_transport_probe_result`) that the per-voxel appearance wall is information-limited.

### 6.6 The unfairness no protocol can fix

**We trained on this exact respiratory simulation** (`training/data/respiratory.py`); SVRTK and
NeSVoR have never seen it and have no learned prior. A reviewer will correctly note that a learned
method beating classical methods on the corruption it was trained on is near-circular.

**No scoring choice fixes this.** The only answer is **OOD evaluation on real free-breathing data**
(OCMR real-time, MIITT RT), where nobody has a training advantage on the specific corruption.
**That is the load-bearing experiment**, not the simulated one.

---

## 7. Literature position

**There is no consensus for this exact situation.** Ferrante & Paragios, *Medical Image Analysis*
39:101-123 (2017), §7: *"the lack of open/public benchmarks with gold standard annotations
specially conceived to validate slice-to-volume registration methods."* No SVR reconstruction
challenge exists in fetal or cardiac imaging. **Methods that output isotropic and methods that
output on the acquired grid are never compared head-to-head in one space anywhere in the
literature.** We must define and defend the protocol ourselves.

**Verified directly (WebFetch, primary source):**
- **Kuklisova-Murgasova 2012** (MedIA 16(8), PMC4067058) — PSF = *"3D Gaussian with FWHM equal to
  the slice thickness in the through-plane direction and 1.2 × voxel size in-plane"*;
  leave-one-stack-out forward-projects the recon into the held-out slices; *"The reconstructed
  images were first rigidly aligned with the original volume and resampled"* before scoring.
  Metrics NRMSE / PSNR / TRE.
- **Ebner 2020 / NiftyMIC** (NeuroImage 206:116324, PMC7103783) — `Sim(y_ki, A_ki x)` in **slice
  space**, *"In absence of a ground-truth of the high-resolution volume…"*. ⚠️ **This is
  self-consistency against its OWN INPUT** — they have no GT. **Our situation differs: we have an
  independent GT.** Cite Ebner **for the forward-model operator only**, never as a matching
  protocol.

**Useful, NOT independently verified** (from a survey agent; spot-check before citing): Kainz 2015
(registration to GT *"necessary to compensate for potentially small offsets"*), NeSVoR/Xu 2023,
Uus 2020 (leave-one-stack-out, PSF-simulated), DMCVR MICCAI 2023 (held-out LAX-plane Dice — the
most transferable cardiac no-GT idea), FetMRQC_SR.

**Two positions we can claim:** (i) rigid registration to the reference before scoring is
**universal and never questioned as oracle-assistance** — so our compensated column is on
well-trodden ground; (ii) **nobody reports a method both gauge-free and gauge-corrected.** FC-SVR
removes the gauge algebraically in every metric but never shows the uncorrected number. Reporting
both is a small, real, cheap contribution.

⚠️ **PSF-in-the-metric is RARE** (only Kuklisova-Murgasova 2012 and Uus 2020) and **absent from
cardiac entirely.** Our §6.2 step 3 is **novel, not conventional** — justify it on physics, do not
claim it as standard practice.

---

## 8. TODO for the next agent

1. **Move the probes into `tools/`** (they are currently only in a session scratchpad and will be
   lost): the operator A/B, the through-plane sharpness metric, and the coordinate-descent gauge
   search. All are ~40 lines and read-only.
2. **Re-run SVRTK / NeSVoR / NiftyMIC** on the current native-z cohort (§6.1).
3. **Re-measure everything across the cohort** — every number here is **n=1 subject, 1 phase**.
4. **Replace the metric-maximizing fit with an off-metric fit** (§3.3) and re-report the gains.
5. **Implement the PSF path** in `assemble_and_gif.py` as an *option*, keeping trilinear available,
   until the sign question (§4.3) is settled on more than one subject.
6. **Resolve the hardcoded `SHAPE_XYZ` vs native-`dz`** question (§2) — read `docs/58` first.
7. **Chase the `clean` anomaly** (§6.5): why do we score only 24.91 when the input *is* GT?
8. **Run the OOD real-free-breathing comparison** (§6.6) — the experiment that actually decides it.

**Do not:** derive the pose correction from `manifest["breath"]["disp_dhw_mm"]` (§3.2), translate
anything on the 12 mm grid (§3.4), apply the PSF to VGGT/FC-SVR (§4.4), or headline `clean` (§6.3).
