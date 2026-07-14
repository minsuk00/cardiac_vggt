> **TL;DR & takeaway**
> **Classical single-stack SVR (SVRTK `reconstruct`) cannot recover valid per-slice motion in our
> scattered / one-frame-per-slice regime** — it does not *correct* the simulated breathing, it *bakes it
> in* (the −10 dB breath-PSNR drop, docs/29 / eval README, is the primary mismatch-free evidence). The
> `.dof`-based analysis here *corroborates why*: given a single stack, SVR's per-slice registration has no
> inter-slice redundancy to lock onto, so it estimates spurious noise instead of the true motion. The
> **clean proof** is the **in-plane controlled experiment** — inject a *known, fully-resolvable* rigid
> in-plane shift into a single stack and SVR recovers **~0** (slope/corr ≈ 0, n=12), while a **matched
> positive control** (the *same* in-plane shift given as *two overlapping stacks*) recovers **96–103 %**.
> That isolates **redundancy** as the cause. **Through-plane non-recovery then follows *a fortiori*** (it is
> strictly harder than in-plane — less redundancy, coarser sampling); the direct through-plane `.dof`
> measurement also reads ≈ 0 but is **under-controlled** (see Caveats: the demeaned through-plane
> differential is sub-resolution ~0.15 vox, injected `dz` is content-substitution that a rigid `tz` can't
> represent, and there is no matched through-plane positive control) — so treat it as *consistent with*,
> not *proof of*, through-plane non-recovery. Also: on motion-free clean stacks SVR still hallucinates
> through-plane |tz| = 2.5 mm mean (max 28). **Why it matters:** for the SVRTK-vs-VGGT head-to-head the
> honest motion row is **SVR ≈ 0** (it can't even recover resolvable in-plane motion single-stack). This
> methodology was **adversarially debated (2 subagents)** — demeaning was confirmed *required and fair*
> (gauge removal), and the in-plane claim *clean*; the through-plane `.dof` claim was *downgraded* to
> corroborating. Repro: `scratch/eval/engine/prove_motion_nonrecovery.py` → `motion_nonrecovery.json`.

## Context

The eval harness (`scratch/eval/`, see its README) simulates respiratory motion on gated CMR cine and
reconstructs with classical SVR (SVRTK 3D-per-phase, single gated stack K=1) as the baseline VGGT must
beat. Breathing costs ~10 dB PSNR (CMRx clean 28.2 → breath 17.6 dB) — the **image-domain** evidence that
SVR does not *correct* the motion. This doc asks the finer, `.dof`-domain question — does SVR even
*estimate* the motion? — and answers no, with the important nuance that the clean part of the answer is
in-plane, and through-plane is corroborated a fortiori rather than measured cleanly.

## The gauge problem (why raw .dof isn't directly comparable) — and why demeaning is fair

SVR reconstructs a *consensus* volume whose frame is a free gauge — shift every slice by +Δ and the recon
just shifts, leaving every `.dof` unchanged. So **absolute** `.dof` translations are unidentifiable; only
the **demeaned** (relative slice-to-slice) transform is, and SVR estimates the *correction* (≈ −injected).
All scoring below demeans per stack and compares `demean(T_i)` vs `demean(d_i)` (slope/corr). **Demeaning
is required, not a thumb on the scale** — it removes exactly the one unidentifiable DC degree of freedom
and preserves any real recovered slope (adversarially confirmed: subtracting a column mean cannot erase a
genuine correlation). If SVR recovered the motion, demeaned `.dof` would track −injected: slope ≈ −1,
|corr| ≈ 1.

## The experiments (on the frozen CMRx bundles, DEBUG=1 `.dof`)

**(0) Scorer sanity (synthetic).** Feed the metric `.dof = −injected` → slope −1.0, corr −1.0; `.dof = 0`
→ 0, 0. Confirms the *arithmetic* of the metric (not, by itself, end-to-end pipeline sensitivity).

**(1) Controlled single-stack calibration (n=12 subjects).** Inject a KNOWN per-plane displacement into
one clean stack via deform-then-reslice (`reslice_volume_vec(V, disp[z])[z]`, same mechanism as the
breathing sim), three patterns (linear on z, centered-quadratic on y, sinusoid on x), each ±8 mm;
reconstruct; regress recovered `.dof` (slice-order) vs injected per axis:

  | injected axis | recovered slope (mean±sd) | corr (mean±sd) | verdict |
  |---|---|---|---|
  | **x (in-plane)** | +0.04 ± 0.13 | +0.10 ± 0.33 | **clean null — 8 mm = 5.7 vox, fully resolvable, rigid** |
  | **y (in-plane)** | +0.08 ± 0.22 | +0.08 ± 0.35 | **clean null** |
  | z (through-plane) | −0.05 ± 0.56 | −0.13 ± 0.41 | ≈0 but under-controlled (see Caveats) |

All centered on zero with **inconsistent sign** across subjects (= noise, not weak recovery). **The
in-plane rows are the load-bearing result:** the injected motion there is a genuine, fully-resolvable rigid
translation a `.dof` *can* represent, yet SVR recovers none of it.

**(2) Positive control — matched in-plane, multi-stack (n=3).** Two overlapping parallel stacks, stack B
shifted the *same kind* of known +6 mm in-plane shift; reconstruct K=2 (SVR now HAS inter-slice
redundancy). Recovered relative tx: 5.73 / 6.15 / 5.89 mm → **recovery fraction 0.96 / 1.03 / 0.98**.
Matched to (1)'s in-plane axis, changing *only* redundancy → **the contrast isolates redundancy as the
cause of the in-plane null**, and proves reconstruct's registration + the `.dof` readout work. (Gotcha:
multi-stack `reconstruct` needs one `-thickness` value *per stack* — `-thickness 8 8` — else it errors out
with no `.dof`.)

**(3) Spurious motion on clean (n=423 slices, all 43 subjects, t00).** On clean stacks (zero motion to
correct) SVR still invents motion: in-plane |tx|,|ty| ≈ 0.9–1.0 mm mean; through-plane |tz| = 2.5 mm mean,
p95 9.2 mm, max 28.3 mm. Registration wanders into spurious local minima, worst through-plane.

## Caveats — what is and isn't cleanly established (from the adversarial debate)

- **In-plane non-recovery: cleanly established.** Resolvable rigid motion, correct axis mapping,
  demeaning legitimate, matched positive control. High confidence.
- **Through-plane non-recovery: corroborated a fortiori, NOT independently proven by the `.dof` slope.**
  Three reasons the direct through-plane `.dof` measurement is under-controlled: **(i) representational
  mismatch** — injecting `dz` via reslice fills plane z with *blurred content from a neighbouring depth*
  (content substitution), which a rigid `tz` cannot invert; injected-`dz`-vs-recovered-`tz` are not the
  same operation. **(ii) sub-resolution** — after demeaning, the plane-to-plane through-plane differential
  is ~1.78 mm ≈ **0.15 voxel** (12 mm pitch) ≈ 0.22× the 8 mm thickness — below the through-plane
  detection floor of *any* SVR, so even an oracle would read ≈ 0 there. **(iii) no matched positive
  control** — parallel stacks can't supply through-plane redundancy, so metric sensitivity on `tz` is
  never demonstrated end-to-end. The through-plane conclusion instead rests on: the in-plane failure +
  the fact that through-plane is strictly *harder* (less redundancy, coarser sampling) + the image-domain
  PSNR drop.
- **Pattern non-orthogonality.** The three injected patterns are NOT mutually orthogonal (corr(z,x) ≈
  −0.76 at k≈10; only z–y is orthogonal), so the univariate per-axis slopes are cross-contaminated — read
  them as rough per-axis probes, not clean axis-isolated responses. (More likely to manufacture a spurious
  *nonzero* than a false null, so it doesn't rescue recovery.) Fix if revisited: truly orthogonal patterns
  or a multivariate regression.
- **Small n / outliers.** k≈9–10 slices/subject with heavy-tailed `.dof` (±28 mm edge outliers); the
  "mean≈0, inconsistent sign" null is robust in *direction* but a Theil–Sen/Spearman + CI would harden it.

## Why (mechanism)

SVR recovers motion by exploiting *redundancy* — overlapping slices sampling the same location let it
triangulate each slice's true position (why fetal SVRTK uses multiple orthogonal stacks). Our regime is
**one slice per z-plane, no overlap** → zero redundancy → each slice trivially "matches" the volume it
itself defines → registration has nothing to lock onto. So the breathing is never estimated; it goes
straight into the volume as blur / misalignment → the −10 dB. Structural for single-stack input — not
fixable by tuning SVRTK.

## Implications

- **SVRTK-vs-VGGT motion-recovery row = SVR ≈ 0.** The strongest honest statement: SVR cannot recover even
  a fully-resolvable *in-plane* rigid shift from a single stack (let alone through-plane). If VGGT's resp
  metrics (`resp_slope`, `epe_dz_mm`; docs/38) show any recovery, it wins on a real basis. Report the
  **image-domain breath PSNR (28.2→17.6)** as the primary "doesn't correct" number; the `.dof` analysis as
  the "and here's why (no redundancy)".
- The "motion scorer" TODO is therefore *answered*: demonstrate structural non-recovery, don't score
  accuracy. No per-subject cohort `.dof` regression on the real breathing is needed (alignment-fragile;
  would just reproduce corr≈0).

## Repro / artifacts

- `scratch/eval/engine/prove_motion_nonrecovery.py` — scorer sanity + controlled calib + positive control
  + spurious-clean. Writes `scratch/eval/motion_nonrecovery.json`.
- `scratch/eval/engine/calibrate_motion_dof.py` — original per-axis single-subject calibration.
- Report: `_html/svrtk_motion_nonrecovery.html`.
- Methodology adversarially debated (2 subagents, 2026-07-13): demeaning confirmed required/fair; in-plane
  claim clean; through-plane `.dof` claim downgraded to corroborating (this doc reflects that outcome).
