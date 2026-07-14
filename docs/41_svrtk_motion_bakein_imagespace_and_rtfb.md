> **TL;DR & takeaway**
> **Single-stack SVRTK does not *correct* the simulated breathing — it *bakes it into* the reconstruction, on
> both motion axes, across all 43 eval subjects.** This is the image-space companion to `docs/40`'s `.dof` proof:
> instead of scoring SVR's internal transforms, we measure *where the reconstructed anatomy actually ended up*.
> Breathing is **dominated by through-plane (Z) motion** (dZ mean 7.2 mm ≈ 0.6 slice-pitch, 2.07× the in-plane
> dY 3.5 mm). Depth-registering each recon slice against GT shows its content sits at the breathing-**shifted**
> depth: **through-plane slope 0.77 vs injected dZ** (CMRx, control 0.17 planes) and **in-plane retains ~77%**
> (control 0.19 vox). Per-plane, **85 % of moving planes are baked in**, the median plane corrects **5 %**, and
> binning-out measurement noise gives a displacement ratio ≈ **1.0 at every shift magnitude** — i.e. **~0 %
> systematic correction**; the apparent 15 % "corrections" are measurement noise (10 % point the *wrong* way).
> This reconciles the seemingly-contradictory fact that SVR's `.dof` files *do* contain large tz (up to 28 mm):
> those transforms are **spurious** (nonzero on motion-free clean stacks, uncorrelated with the true motion),
> so they jitter slices randomly around the acquired position without ever pulling them back to truth.
> **Cause = no redundancy** (a single non-overlapping SAX stack cannot triangulate a slice's true position).
> **Would real-world RTFB (many frames/slice) fix it?** A 2-agent debate concluded (and both agents converged):
> **through-plane stays structurally limited** — single-orientation SAX with 8 mm slices + 4 mm gaps has **zero
> through-plane overlap**, so no amount of same-plane temporal redundancy manufactures a depth reference; the
> only handle is respiratory **gating/selection** (`docs/35`), which is pitch-bounded and SNR-costly, not
> correction. **In-plane improves**, but mostly as *deblur* that converges to the **respiratory-mean** position,
> **not** the end-expiration reference our GT uses — so it plateaus below the clean-control floor. This is the
> classical ceiling VGGT must beat, and *why* a learned anatomical prior (substituting for the missing
> redundancy) is the research bet. **Status: proof done + verified; the discriminating RTFB experiment is
> specified but not run.**

---

## 1. Why this doc exists (relation to docs/40)

`docs/40` proved non-recovery by scoring SVR's estimated per-slice transforms (`.dof`): inject a known
displacement, regress SVR's estimate against it → slope ≈ 0. That proof is rigorous but **indirect** (it reasons
about SVR's internal registration parameters, which have a gauge freedom and needed demeaning) and it scoped the
through-plane claim *down* to "consistent-with, not proven" because the `.dof` through-plane differential was
sub-resolution and lacked a matched positive control.

This doc closes that gap by measuring the thing that actually matters: **where the reconstructed anatomy landed**,
directly in image space, on the **dominant through-plane axis**, over the **whole cohort (43 subjects)**, with a
**validating control**. Same verdict, stronger evidence, and the through-plane axis is now measured rather than
argued a-fortiori.

Prompted by the user's correct challenge: an early version of this analysis measured only the *in-plane* axis,
which is the *smaller* component — see §3.

## 2. Method

All measurements on the **canonical grid** (256×256×12 @ 1.4/1.4/12 mm) where GT and the SVR recon co-live for
both datasets, windowed by the heart region (CMRx: `heart_seg`; MIITT: the SVR recon's own support, since the
native seg cannot be affine-resampled into the canonical frame — the grids share no world coordinates).

The frozen breathing sim (`training/data/respiratory.py`, replayed by `scratch/eval/*/build_inputs.py`) shifts
**each z-plane** by a known per-plane 3-D vector `disp_dhw_mm = (dZ, dY, dX)`, logged in each subject's
`manifest.json`. "Recovered" = recon lands at the true (clean-GT) anatomical position; "baked in" = recon stays
at the breathing-shifted position. Two axes, each with a no-motion control:

- **In-plane (Y)** — `scratch/eval/engine/prove_motion_imagespace.py`. Per plane, phase cross-correlation gives
  the in-plane Y-shift for: `s_out` = SVR(clean) vs SVR(breath) (breathing surviving in the output);
  `s_ctrl` = GT vs SVR(clean) (**control**, must be ≈0); `s_in` = clean-stack vs breath-stack (CMRx only, to
  normalise out measurement damping → `preservation = slope(s_out)/slope(s_in)`). Regress each vs injected dY.
- **Through-plane (Z)** — `scratch/eval/engine/prove_motion_throughplane.py`. Per plane, depth-register the recon
  slice against GT interpolated along z (fractional depths), **allowing in-plane translation at each candidate
  depth** to remove the dY confound, and take the depth offset δ* that maximises NCC. `δ*_breath` vs injected dZ
  (baked-in if ~dZ, recovered if ~0); `δ*_clean` vs dZ is the **control** (must be ≈0, validates that the depth
  registration itself introduces no offset).

Outputs: `scratch/eval/motion_imagespace.json`, `motion_throughplane.json` (+ `*_raw.npz`). Figures via
`fig_motion_imagespace.py` → `result/svrtk_motion/`. Self-contained report:
`_html/svrtk_motion_imagespace.html`.

## 3. Breathing is dominated by through-plane motion

Cohort per-plane displacement magnitudes (CMRx, moving planes, n=271):

| axis | mean \| max (mm) | in grid units |
|---|---|---|
| **through-plane dZ** | **7.23 \| 24.85** | **0.60 slice-pitch** (12 mm) |
| in-plane dY | 3.50 \| 24.18 | 2.50 vox (1.4 mm) |
| in-plane dX | 2.18 \| 12.61 | 1.56 vox |

**dZ/dY ≈ 2.07** — Z is the dominant axis. Note the tension that explains why the in-plane axis is *easier to
measure* despite being smaller: in **grid units** in-plane is a well-resolved 2.5-vox rigid translation, whereas
through-plane is sub-pitch (0.6 planes) on the coarse 12 mm grid. `docs/40` (and this analysis's first pass)
measured in-plane for that convenience; §4.2 measures the dominant through-plane axis directly.

## 4. Results

### 4.1 In-plane (Y) — corroboration

| metric | CMRxRecon (30) | MIITT (13) |
|---|---|---|
| output shift slope vs dY | 0.69 (r 0.72) | 0.56 (r 0.50) |
| mean \|shift\| output | 1.64 vox | 1.65 vox |
| **control** slope (GT vs SVR-clean) | 0.005 | −0.058 |
| **control** mean \|shift\| | **0.19 vox** | **0.40 vox** |
| input slope / **preservation** | 0.89 / **77 %** | — |

The control being flat (0.19 vox, slope ≈0) validates the method: with no motion, SVR lands on truth. With
breathing, the recon is displaced ~1.6 vox/plane tracking the injected shift; the input-normalised **preservation
is ~77 %** — SVR retained most of the in-plane shift.

### 4.2 Through-plane (Z) — the dominant axis, measured directly

| metric | CMRxRecon (30) | MIITT (13) |
|---|---|---|
| breath depth-offset slope vs dZ (\|dZ\|<1.5) | **0.77 (r 0.63)** | 0.60 (r 0.09) |
| **control** slope (GT vs SVR-clean) | −0.088 | 0.37 |
| **control** mean \|offset\| | **0.17 planes** | 0.44 planes ⚠ |
| n planes (reliable) | 920 | 472 |

**CMRx is clean and decisive:** recon depth tracks the injected dZ at slope 0.77, control flat at 0.17 planes.
The dominant through-plane motion is baked in, and 0.77 matches the in-plane 77 % preservation — **SVR bakes in
both axes at the same fraction.** MIITT through-plane is **under-controlled** (control offset 0.44 planes ≈ its
signal, from native→canonical depth ambiguity) → treat MIITT through-plane as suggestive only; the clean
through-plane proof rests on CMRx, corroborated by MIITT's (clean) in-plane.

Reliability is limited to |dZ| < 1.5 planes; larger shifts alias on the 12 mm grid (content substitution > one
plane — the hard regime also flagged in `docs/40`).

### 4.3 Per-plane distribution — does it *ever* correct?

On CMRx planes with meaningful injected motion (|dZ| > 0.3 planes, n=388):

- **85 % baked in** (recon closer to the shifted depth than to true); 15 % "corrected"; **10 % move *opposite*
  to the injected direction** (physically impossible as correction → pure noise).
- **Median recovery fraction (1 − measured/injected) = 0.05** — the typical plane corrects ~nothing.
- **Binned by shift size, the ratio is ≈ 1.0 at every magnitude:** dZ 0.46→0.43 (0.95), 0.80→0.83 (1.04),
  1.26→1.20 (0.95). No systematic under-shoot at any size ⇒ **no systematic correction.** The aggregate slope
  0.77 is a **noise-attenuated underestimate** of the baking-in; binning out the noise → ~100 %.
- Control noise floor (GT vs SVR-clean): mean 0.175 planes, std 0.407. The residual (injected − measured) has
  std **0.593 ≈ control noise** → the deviations from "fully baked in" are measurement noise, not correction.

**Conclusion: SVR does ~0 % systematic correction; the scatter around baked-in is measurement noise.**

## 5. Reconciling with the `.dof` transforms (the "but it has large tz" objection)

SVRTK *does* emit nonzero, sometimes large per-slice tz — this is **not** "placing slices with zero transform".
From `docs/40`'s `.dof` analysis (`motion_nonrecovery.json`):

- On **motion-free clean** stacks, estimated |tz| = 2.49 mm mean, 9.17 mm p95, **28.29 mm max** — nonzero where
  there is nothing to correct ⇒ **spurious by definition**.
- Under a **known injected** shift, estimated tz is **uncorrelated** with it (cohort slope −0.05, corr −0.13),
  random sign across subjects.
- **Positive control:** the *same* shift given as **two overlapping stacks** recovers **96–103 %** → the
  machinery works; the single-stack failure is the **redundancy limitation, not a bug**.

So the large tz are **spurious registration** (a single stack registers each slice against a blurry reconstruction
≈ itself, and wanders). They are exactly the **scatter** in §4.3 (max spurious tz 28 mm ≈ 2.3 planes ≈ the
residual spread). Net: slices are jittered randomly around the acquired position, never systematically pulled to
truth. The honest claim is **"no systematic recovery,"** not "zero transform."

## 6. Would real-world RTFB redundancy fix this? (2-agent debate)

Real-time free-breathing (RTFB) SAX is not single-frame-per-slice: continuous acquisition over many
heartbeats/breaths gives multiple frames per slice location, retrospectively bin-able by cardiac (and
respiratory) phase. User's prediction: *"better in-plane correction, but through-plane still limited."* Two agents
debated it (opening + rebuttal each); they **converged**:

**Agreed / robust:**
- **Through-plane stays structurally limited.** Single-orientation SAX (8 mm thick + 4 mm gap) has **zero
  through-plane overlap**; same-plane frames all sit at the same nominal z → they add **SNR, not depth
  triangulation**. No temporal redundancy manufactures the missing axis. (Contrast fetal SVR, which recovers
  through-plane *only* via multiple **orthogonal/overlapping** stacks — `docs/34`.) The only through-plane handle
  is respiratory **gating/selection** (`docs/35`): it *rejects* SI spread by picking a consistent breathing
  state — **pitch-bounded, SNR-costly, and it doesn't recover the sub-pitch residual or the deep-breath tail**.
  Corroborated by `docs/35`'s single-orientation recon attenuating EF to 20.6 % vs ~60 %.
- Temporal same-plane redundancy genuinely removes **relative** in-plane jitter (deblur + SNR).

**Refined (where the user's "better in-plane" needs two caveats):**
1. **The real respiratory lever is gating, not SVR registration.** RTFB helps mainly by letting you *select* a
   motion-consistent subset — avoiding the corruption, not correcting it. SVRTK is a passive consumer of the
   pre-cleaned stack. Framing the win as "SVR corrects in-plane better" mis-attributes the mechanism.
2. **In-plane "correction" converges to the wrong target.** Registering respiration-varied frames to a consensus
   lands each slice at the **respiratory-mean** in-plane position — but our GT is defined at **end-expiration**
   (the unshifted breath-hold anchor; `respiratory.py` leaves target/scanner_coords/GT at end-expiration). The
   Lujan sin²ⁿ model is not zero-mean about exhale, so the consensus fixed point is offset from GT by a
   **mean-minus-end-expiration bias that does not vanish with more frames**. So in-plane improves but **plateaus
   below the clean-control floor**, and more frames tighten variance around the *wrong* center. Also: SVRTK's
   `reconstructCardiac` applies a temporal PSF that may **collapse intra-bin frames before registration**, in
   which case the per-frame registration never happens and it lands at the bin mean by default.

**One-line conclusion:** *In RTFB, extra frames per slice let SVRTK sharpen the in-plane image (deblur toward the
average breathing position, not the true one) but cannot fix the dominant through-plane motion — SAX slices don't
overlap, so there's no depth information to correct against; the only handle is discarding off-phase frames
(gating). This is the ceiling the learned model aims to beat.*

**Implication for the project.** The SVRTK baseline ceiling holds up under RTFB: redundancy won't rescue the
dominant through-plane axis. This is exactly the gap VGGT bets on — a **learned anatomical prior** substitutes for
the geometric redundancy that isn't there, letting it infer plausible through-plane structure and map
respiratory-mean → reference, which SVR structurally cannot. (Research bet, not yet proven.)

## 7. Discriminating experiment (specified, NOT run)

Both agents proposed the same cheap test on the existing harness to convert prediction → measurement:

> Build a **cardiac-only, respiration-varied** multi-frame set per slice (K frames sampled across the respiratory
> distribution), feed to SVRTK via the multi-stack `-thickness 8 8 …` path (already working, `docs/40` §2).
> Measure the recon's per-axis content position against **two** references: (i) end-expiration GT, (ii) the input
> frames' respiratory-mean.
> - **In-plane → GT and through-plane ratio stays ~1.0** ⇒ axes **split**, "redundancy corrects in-plane".
> - **In-plane → respiratory-mean (residual = mean−exhale bias, independent of frame count) and both axes baked**
>   ⇒ **consensus-averaging + gating is the lever**, not registration.

Bin-width doesn't discriminate (both invoke it); the **per-axis split vs. move-together** does, as does the
GT-vs-respiratory-mean reference comparison.

## 8. Files

- Scripts (`scratch/eval/engine/`): `prove_motion_imagespace.py` (in-plane), `prove_motion_throughplane.py`
  (through-plane), `fig_motion_imagespace.py` (figures), `make_motion_html.py` (report).
- Data: `scratch/eval/motion_imagespace.json`, `motion_throughplane.json` (+ `*_raw.npz`).
- Figures: `result/svrtk_motion/{scatter_throughplane,scatter_shift,dipole_*,sagittal_*}.png`.
- Report: `_html/svrtk_motion_imagespace.html`. Companion `.dof` proof: `docs/40` +
  `_html/svrtk_motion_nonrecovery.html`.
- Reproduce (quiet node, ~few min each): `micromamba run -n svr python -u
  scratch/eval/engine/prove_motion_throughplane.py` (and `prove_motion_imagespace.py`), then
  `fig_motion_imagespace.py` and `make_motion_html.py`.
