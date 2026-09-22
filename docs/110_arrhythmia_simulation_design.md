# 110 — Arrhythmia / rhythm-irregularity simulation: design, sources, and the GT protocol it forces

> ⛔ **DEPRECATED — do not read this copy.** This is the untracked pre-implementation draft
> (design only, 3 arms, middle rung named `mild`, "not yet in `evaluation/`"). The **live, committed**
> version is on branch **`exp/af-sim`** at `docs/110_arrhythmia_simulation_design.md`
> (commit `bb45672`), which describes **4 arms** (`regular_frozen`/`regular`/`hrv`/`af`) and adds
> **§11** (how the shipped code departs from this design) and **§12** (fairness, claim scoping,
> pre-registered outcomes). Kept only as a record of the design before implementation.

> **TL;DR & takeaway**
> Design (**not yet run, not yet in `evaluation/`**) for a 3-rung robustness ladder — `regular` →
> `mild` (beat length ~N(1, 0.15)) → `af` (~N(1, 0.25), physiological cut-off/hold) — testing whether
> Fetal CMR 4D's EF advantage (docs/105: MAE 6.6 vs VGGT's 10.2–13.5) survives non-periodic input.
> Its self-gating assumes a **constant heart rate per slice with uniformly spaced phases** — verified
> in the authors' own MATLAB and stated in their paper's Discussion as a known limitation. Three
> findings shaped the design: (1) the authors' x-f FFT **cannot** estimate R-R from our 12-frame
> window (measured on real GT cine: 0 % of slices yield a peak as written; 3–8 % error with DC
> removal, vs 0.2 % at 17 beats/slice) — the `RR_NOMINAL_S = 1.0` placeholder is therefore a
> permanent, disclosed adaptation, not a bug; (2) **±4 % physiological HRV is sub-frame** at our
> 12-frames-per-beat rate and would be a null experiment, so the middle rung uses the precedent
> paper's ±15 %; (3) scoring against the clean ED-first `gt_t{k}` cine is **wrong** under AF — the
> GT must be the simulated heart at each reference frame's own true cardiac position, which costs
> VGGT its current "reference at a chosen phase" privilege and makes these arms non-comparable with
> the reported `rolled` numbers. Prototype: `temp/arrhythmia_sim/sim_arrhythmia.py` (reproduces
> `rolled/` bit-exactly in `regular` + frozen-breathing mode). **Every physiological premise is
> citable; the volume-matched resume, the 0.45 R-R floor and the full-ED hold are our own
> constructions and must be disclosed.** Most sources below were read at abstract/summary level only
> — flagged per row; read them in full before citing.

---

## 1. Why this experiment

docs/105 §8 reports Fetal CMR 4D at **EF MAE 6.6** on the 180-subject test split vs VGGT's
10.2–13.5 — the best EF of the whole roster. docs/105 itself attributes this to the disclosed
**12-frames-per-slice privilege** (the same-input 1-frame classical arms collapse to EF MAE 37–42).

The open question is whether the `rolled/` benchmark is *too easy* in a second way: it hands the
method **exactly one perfectly periodic cardiac cycle per slice**. Real free-breathing acquisition
has no such guarantee. This doc designs the input that removes that guarantee.

**Scope note.** Once adopted, this simulation becomes the input for **all** arms (VGGT included),
not a Fetal-4D-only ablation.

## 2. What the baseline actually assumes (verified, not inferred)

Read in the authors' MATLAB (`scratch/fetal_cmr_4d/repo/cardsync/`) and in the paper PDF
(`baselines/fetal_cmr_4d/fetal_cmr_4d_paper.pdf`, §2.4 + Discussion):

- `estimate_heartrate_xf.m` — one **scalar R-R per slice** from the x-f FFT peak in the heart ROI.
  No time-domain beat detection anywhere.
- `estimate_heartrate_xf.m:205` — `triggerTime = rrInterval * (0:nTrigger)`, i.e. **synthetic,
  perfectly uniform** triggers.
- `calc_cardiac_timing.m` → `cardiacPhase = (t_frame − t_prevTrigger)/rrInterval`;
  `cardsync_intraslice.m:124` → `theta = 2π · cardiacPhase`.
- Paper §2.4: *"A **constant heart rate** was estimated for each slice…"*
- Paper Discussion: *"The use of a constant heart rate for each slice **does not accommodate
  beat-to-beat variation** resulting in small timing errors."* ← the exact assumption under test.
- Paper acquisition: 72 ms/frame, **96 frames per slice ≈ 17 beats** — why their FFT works.

Our gate (`evaluation/src/engine/fetal4d_gate.py`) is the same formula with `RR = T·dt` forced:
`theta[z] = 2π((f − f_ED) mod T)/T`, with the ED anchor from nnU-Net LV area (their inter-slice
sync needs multi-orientation overlap we do not have — docs/34/35).

## 3. Measured: the x-f estimator cannot work on a 12-frame window

Python port of `estimate_heartrate_xf.m` (zero-padded spectrum, spatial ROI mean, smoothing,
`findpeaks` in the band), run on **real GT cine** (P012, 11 heart-ROI slices × 4 random start
phases; fixed-`dt` series simulated at a modified R-R, then R-R estimated back):

| frames/slice | as written (no DC removal) | with temporal mean removed (our modification) |
|---|---|---|
| **12 (~1 beat)** | **peak found in 0 % of slices** | mean err 3–8 %, max 10–17 % |
| 24 (~2 beats) | mean err 11–30 % | 0.4–1.7 % |
| 48 (~4 beats) | 0.5–3 % | 0.1–0.6 % |
| 204 (~17 beats) | 0.2 %, max 0.7 % (3 subjects) | not run |

**Reading.** With ~1 cycle in the window the DC spike smears across the heart-rate band and no peak
exists; after demeaning (which the authors' code does **not** do — grep found no detrend, and they
do not need it at 17 beats) the estimate is 3–8 % off, i.e. **as large as the ±4 % HRV it would be
measuring**. Usable accuracy needs ≳2–4 beats/slice.

⇒ **`RR_NOMINAL_S = 1.0` is a permanent disclosed adaptation.** It is also inert in the current
`rolled` arm: the engine uses R-R only via `dtrad = 2π·dt/rr` (`ReconstructionCardiac4D.cc:574`) and
the gate writes `dt = RR_NOMINAL_S/T`, so the ratio is `2π/T` whatever nominal value is passed.
Caveat: the 204-frame row loops one identical cycle (constant R-R, no breathing, no fresh noise) —
an idealised ceiling, not a prediction for real multi-beat data. The 12-frame failure is robust
(it fails under *cleaner*-than-real conditions).

## 4. The three arms

One generator, three settings. Beat lengths are in units of the nominal R-R; frames are sampled at
fixed `dt = 1/12` (a **real-time camera**, not a gated cine).

| arm | beat lengths | rendering | basis |
|---|---|---|---|
| `regular` | all 1.0 | — | reproduces today's `rolled/` |
| `mild` | iid `N(1, 0.15)` | compress/stretch + blend | Fürnrohr & Heckel 2026, their exact recipe |
| `af` | iid `N(1, 0.25)` | cut-off / hold / volume-matched resume + blend | §5 |

**Why not physiological HRV as the middle rung.** Jansz 2010's fetal walk
(`RR_n = RR_{n-1} + 0.1(1−RR_{n-1}) + N(0, 7ms/400ms)`) settles at ~±4 % — **less than one frame**
at 12 frames/beat, so it is a null experiment by construction. It remains in the prototype as
`HRV_MODEL = "jansz"`. The shipped middle rung is the precedent's ±15 %, labelled **"mild
variation"**, NOT "healthy HRV" (15 % is larger than true resting HRV).

**Why no `af_compress` arm.** It is `mild`'s mechanism at sd 0.25 — one mechanism, one parameter,
nothing new. Dropped; trivially re-addable as a sensitivity check against the published recipe.

## 5. The AF rendering model, claim by claim

Three steps per slice: (1) draw beat lengths; (2) map each of the 12 frame times to a cardiac
position in [0, 12]; (3) render that position to an image and apply breathing.

| claim | basis | **how much of the source I actually read** |
|---|---|---|
| iid Gaussian beat-length factor per cycle | Fürnrohr & Heckel 2026 (arXiv 2607.03299), App. A.6 | **PDF read** |
| AF spread sd = 0.25 | Markl et al., JCMR 2015: AF R-R 643 ± 161 ms (161/643) | abstract-level WebFetch summary only |
| phases advance at **normal speed**; the cycle-length change lands in diastole, not in contraction | Chung/Karamanoglu/Kovács, AJP Heart 2004 (PMID 15217800): *"E- and A-wave durations… very nearly independent of HR: 100 % increase in HR generated only an 18 % decrease…"*, *"Diastasis duration closely tracked MDD"*; Weissler `QS2 = −0.0021·HR + 0.546` | Chung **abstract read verbatim** (Europe PMC); Weissler **secondary summary only** |
| ⚠ …applied **beat-to-beat**, not just across steady rates | — | **inference.** Both sources measure steady-state HR (exercise/conditions), NOT beat-to-beat. Indirect support: a QT-variability study reports RR variability explains only 6–11 % of beat-to-beat QT variability (summary only). A directly-relevant paper exists ("Variability of the systolic and diastolic electromechanical periods in healthy subjects") but the host 403'd and it is not in Europe PMC. |
| a short preceding beat ⇒ weaker next beat | Tabata et al., AJP Heart 2001: EF vs RR1/RR2 r = 0.74, SV r = 0.65; AF stroke-volume literature (Hardman 1998) | summary only; **Tabata's SV ≈ 12 ml suggests an animal model** — needs a human citation |
| **implementing that by resuming at the systolic frame whose whole-LV volume matches the volume reached at cut-off** | — | **OURS. No precedent.** Assumes a half-filled heart in diastole ≈ the same-volume heart in systole. Must be disclosed. |
| R-R floor 0.45 (a beat cannot be shorter than systole; refractory period) | — | **OURS**, physiological reasoning, unsourced |
| long beats **hold the full-ED frame** (keeping the GT's atrial-kick frames) | — | **OURS**, a robustness choice. Physiologically AF has **no atrial kick**; holding at diastasis instead was tried in design but abandoned because P020's volume curve has no plateau (100 97 74 62 52 45 42 47 49 56 67 74 → +26 % jump to ED), which would have made every one of its beats start weak. Disclose. |
| **Markl's [0.57, 1.55] range is NOT used as a clip** | measured: clipping there shrinks the realised sd 0.25 → 0.216 | our decision; only the 0.45 floor is applied, by **redraw** (1.4 % of draws), realised sd 0.240 |

**Precedent's own recipe, for the record** (arXiv 2607.03299, App. A.6, quoted): *"we alter the
duration of the cardiac cycles by **phase-aware interpolation between video frames**"*; mild =
`N(1.0, 0.15)`; their "arrhythmia" = one cycle compressed to ~1/3 (`N(3.0, 0.5)` speed factor)
followed by *"repeating the last frame"*, every 2–3 beats — i.e. a **PVC-like event, never AF**, on
the same binned CMRxRecon 12-frame cine we use, to test periodicity assumptions in reconstruction.
Closest published precedent; cites no physiology for any parameter.

**PVC was considered and dropped** (it was the original request, `hrv_af` in the handoffs). Sourced
parameters exist (coupling 40–80 % of baseline R-R; compensatory pause s.t. coupling + pause ≈ 2×R-R;
Circ EP 2016 / US8457728 / Zhong PubMed 31801920 / post-extrasystolic potentiation Circulation 1977)
but AF subsumes it: AF's random lengths already produce premature beats and pauses, on **every**
slice rather than in a chosen subset, with **one** parameter instead of four. Also: "HRV + AF
together" (an explicit instruction in the earlier handoff) is **wrong for AF** — in AF the sinus
pacemaker is not driving the ventricles, Markl's measured spread already contains all variability,
and ±4 % under ±25 % is invisible. That instruction was written for PVC, where the surrounding
normal beats do still carry HRV.

## 6. Frame-wise breathing (new default for all three arms)

Current bundles freeze breathing per plane (`pooled.py` ~L278–312: one `disp0[z]` reused for all 12
frames), justified in `add_rolled`'s docstring as *"a 12-frame real-time burst is ~1 s, well inside
a breathing cycle"*. That holds only near end-expiration: the Lujan waveform
(`lujan_displacement`, `A·sin⁶(πr)`, mean A = 16–19 mm > one 12 mm slice pitch) moves fast through
inspiration, so a slice whose burst lands mid-inspiration drifts by ~a slice pitch across its own
12 frames.

Frame-wise: keep the bundle's per-plane start phase `r0[z]`, advance `r_f = r0[z] + f·dt/T_BREATH`
with `T_BREATH = 5` nominal beats (≈ the precedent's ~1 breath / 5 cycles ⇒ ~12 breaths/min), and
recompute the displacement per frame along the subject's fixed direction (recovered from the
manifest's stored per-plane vectors). Same number of `reslice_volume_vec` calls as today.
Consequence: the **regular reference arm must be rebuilt** with frame-wise breathing too, or the
comparison is not like-for-like. The reported `rolled` arm is untouched.

## 7. The GT protocol change this forces

**Scoring against the clean ED-first `gt_t{k}` cine is wrong here.** Under `af` a slice's 12 frames
may never reach a fully filled ED (cut-off beats), so part of the EF error would be information the
input never contained.

**Correct protocol.** The simulator knows every frame's true cardiac position `p_f`.

- **GT for target f** = the **clean, unbreathed, whole-volume** heart at position `p_f` of the
  **reference (mid) slice** — every plane at the same instant (blend of two clean GT phase volumes).
- **Input** = that same heart state, sliced and breathing-shifted, per slice with its own window.
- **VGGT**: sweep the reference over the mid slice's **12 real frames in acquisition order**
  ("reconstruct the heart as it was in picture f"), not over a chosen phase index.
- **Fetal CMR 4D**: its phase-indexed output cine is read out at those same 12 positions,
  interpolating between bins.
- **EF** (`ef_dice.py`): unchanged formula, `(max − min)/max` of the nnU-Net LV curve, computed over
  the 12 GT volumes and the 12 predicted volumes. Order-independent; ED/ES still `gt.argmax/argmin`.

**What this costs.** Today the scatter bundle gives VGGT's reference slot the **true phase**
(`ref_plane from breath/stack_t{k}`, runner sets `timesteps[0,0] = t`, scored against `gt_t{t}`) —
effectively a gated reference. Under `af` that privilege disappears: only the frames the camera
caught are available. ⇒ **these arms are NOT comparable with the reported `rolled` numbers**; the
rebuilt `regular` arm is their reference point.

**What it buys.** No method is scored on information absent from its input — the objection that
motivated the change. A window of weak beats lowers EF in GT *and* prediction together.

**Residual ambiguity to state in the paper.** In real AF clinical EF is itself an average over
beats of varying strength; our definition is *recovery of the heart state at each observed instant*.

## 8. Prototype status

`temp/arrhythmia_sim/sim_arrhythmia.py` (~300 lines, nothing in `evaluation/` touched):

- `rr_sequence` (§4), `pos_compress` / `pos_physio` (§5 step 2), `simulate` (step 3 + breathing).
- **Verified:** `regular` + `--frozen-breath` reproduces `rolled/` with **max |diff| = 0.00e+00**
  (so roll, axis order and reslice all match the real builder); the 12-frame run equals the first 12
  frames of a 36-frame run (per-`(subject, plane)` RNG streams, `np.random.default_rng([seed, z])`).
- **`BURN = 3`** beats are played before the window opens — without it every window's first beat was
  a fresh full-strength beat starting from a full heart (bug found and fixed during design).
- Blending is used for **every** arm (fractional positions), so arms differ only in their timing
  model, not in rendering. (Snapping to the nearest real frame is the alternative; last session's
  render showed adjacent-phase blends are visually indistinguishable from real frames.)
- Explainer GIFs: `temp/arrhythmia_sim/figs/<subject>_simple.gif` — one mid slice, tight crop, 36
  frames (3 beats), 3 arms side by side, with an LV-filling trace and beat-boundary markers under
  each. Built for P012, P020, MIITT ARVC, ACDC_patient020, OCMR_fs_0012_3T, and short-beat examples
  P016 / P062 / P064. (18 of the first 40 CMRx24 subjects scanned have a cut-off inside their mid
  slice's own 12-frame window.)

**Not built:** bundle writer (stacks + per-target GT volumes + truth manifest), VGGT input draw,
Fetal 4D runner pointed at the new stacks, oracle-timing variant, scoring against per-target GT.

## 9. Open items

1. **Read the summarised sources in full** before citing: Weissler 1968, Markl 2015, Tabata 2001
   (and find a human replacement if it is an animal study), Circulation 1996. Only Chung 2004
   (abstract) and Fürnrohr & Heckel (PDF) have been read directly.
2. **No beat-to-beat source** for "systole fixed / diastole absorbs" (§5 ⚠ row). Either find one or
   state the inference explicitly in the paper.
3. **Weak beats can still end in a hold.** A beat resuming mid-systole reaches "full" sooner than a
   normal beat, so some `RR < 1` beats end held. Unverified physiologically — check or disclose.
4. **Cohort LV-volume-curve check.** P020 shows no diastasis plateau and a +26 % jump into ED;
   ACDC_patient020's curve barely moves (100 → 88 %). The volume-matched resume depends on these
   curves being sane. Needs a cohort-wide pass.
5. **`evaluation/` edits** (gate/runner input-dir parameterisation, per-target GT scoring) require
   explicit user approval; a `git worktree` was agreed as the working arrangement, but `main`
   currently carries **uncommitted `evaluation/` changes + untracked docs 104–109**, which a
   worktree branched from HEAD would not see.
6. **VGGT DVF/segmentation retention**: `run_vggt.py` currently saves only the ED-target DVF
   (`ed_dvf.npz`); the other 11 targets' `world_points` are discarded. Saving all 12 (+ per-subject
   pred/GT segs) was requested — needs a storage budget decision (scratch/GPFS, never `$HOME`).
7. **Is "EF MAE 6.6" still current?** It predates the padded-mask rescoring (docs/107/109) — flagged
   unresolved in two prior handoffs and still unchecked. Everything here is measured against it.

## 10. Sources

- **Fürnrohr & Heckel**, *Piecewise Dynamic Diffusion Regularization for Reconstruction of Cardiac
  Cine MRI*, arXiv **2607.03299** (2026) — the closest precedent (same binned CMRxRecon 12-frame
  cine; cycle-duration change by phase-aware frame interpolation). PDF read.
- **van Amerom et al.**, *MRM* 82:1055 (2019) — Fetal CMR 4D; §2.4 + Discussion; local PDF
  `baselines/fetal_cmr_4d/fetal_cmr_4d_paper.pdf`; MATLAB `scratch/fetal_cmr_4d/repo/cardsync/`.
- **Markl et al.**, *JCMR* 2015, PMC4328928 — AF R-R 643 ± 161 ms (range 364–1000). Abstract only.
- **Chung, Karamanoglu & Kovács**, *AJP Heart* 2004, PMID 15217800 — diastolic sub-phase durations
  vs HR. Abstract read verbatim.
- **Weissler et al.** 1968 — systolic time intervals, `QS2 = −0.0021·HR + 0.546`. Summary only.
- **Tabata et al.**, *AJP Heart* 281:H573 (2001) — LV systolic function vs RR1/RR2 in AF. Summary only.
- **Jansz et al.**, *MRM* 2010 — fetal HRV bounded random walk (σ = 7 ms, Δ = 0.1, RR₀ = 400 ms).
  Primary text pasted by the user in an earlier session.
- **Corino et al.**, *BioMed Eng OnLine* 6:9 (2007) / IEEE TBME 58:3386 (2011) — AV-node AF R-R
  generator; the rigorous alternative to our iid Gaussian. Not read.
- PVC parameters (unused, kept for provenance): Circ EP 2016 coupling-interval variability;
  US8457728; Zhong PubMed 31801920; Circulation 1977 / Europace 2020 / JACC EP 2017 (potentiation).
