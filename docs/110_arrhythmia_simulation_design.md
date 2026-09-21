# 110 — Arrhythmia / rhythm-irregularity simulation: design, sources, and the GT protocol it forces

> **TL;DR & takeaway**
> A **4-arm** rhythm-robustness ladder — `regular_frozen` → `regular` → `hrv` (beat length ~N(1, 0.15))
> → `af` (~N(1, 0.25), physiological cut-off/hold) — testing whether Fetal CMR 4D's EF advantage
> (docs/105: MAE 6.62 vs VGGT's 10.2–13.5) survives non-periodic input. Its self-gating assumes a
> **constant heart rate per slice with uniformly spaced phases** — verified in the authors' own MATLAB
> and stated in their paper's Discussion as a known limitation. Three findings shaped the design:
> (1) the authors' x-f FFT **cannot** estimate R-R from our 12-frame window (measured on real GT cine:
> 0 % of slices yield a peak as written; 3–8 % error with DC removal, vs 0.2 % at 17 beats/slice) — the
> `RR_NOMINAL_S = 1.0` placeholder is therefore a permanent, disclosed adaptation, not a bug;
> (2) **±4 % physiological HRV is sub-frame** at our 12-frames-per-beat rate and would be a null
> experiment, so the middle rung uses the precedent paper's ±15 % (named `hrv`, **which is ~4× real
> resting HRV — always state the spread**); (3) scoring against the clean ED-first `gt_t{k}` cine is
> **wrong** under AF — the GT must be the simulated heart at each reference frame's own true cardiac
> position, which costs VGGT its current "reference at a chosen phase" privilege and makes these arms
> non-comparable with the reported `rolled` numbers. **Every physiological premise is citable; the
> volume-matched resume, the 0.45 R-R floor and the full-ED hold are our own constructions and must be
> disclosed.** Most sources below were read at abstract/summary level only — flagged per row; read them
> in full before citing.
>
> **Status (2026-09-19):** generator + cohorts + harness are **built and bug-hunted**; §11 records every
> way the implementation departs from the design above. **No pilot has been run** — the Fetal engine has
> never executed on a rhythm cohort. §12 scopes what the result will and will not license.

---

## 1. Why this experiment

docs/105 §8 reports Fetal CMR 4D at **EF MAE 6.62** on the 180-subject test split vs VGGT's
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

**Consequence, stated plainly for the paper:** their gate is **structurally incapable** of
representing beat-to-beat variation. One scalar R-R per slice plus uniform synthetic triggers means
θ is always an evenly spaced integer ramp; there is no fractional or repeated phase it can express.
This is an architectural property of the published method, not a tuning gap and not a handicap we
imposed. Verified on a real live-campaign gate (`cmrx2024/CMRx24_Test_P016`): every slice's bins are
integers, each slice being a pure rotation of `0..11` by its own ED index.

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

## 4. The four arms

One generator, four settings. Beat lengths are in units of the nominal R-R; frames are sampled at
fixed `dt = 1/12` (a **real-time camera**, not a gated cine). `ARMS` at `tools/build_af_bundle.py:69`
is the single source of truth.

| arm | beat lengths | breathing | rendering | isolates |
|---|---|---|---|---|
| `regular_frozen` | all 1.0 | frozen per plane | — | **replication** — reproduces today's `rolled/` bit-exactly |
| `regular` | all 1.0 | frame-wise | — | cost of frame-wise breathing (§6) |
| `hrv` | iid `N(1, 0.15)` | frame-wise | compress/stretch + blend | Fürnrohr & Heckel 2026, their exact recipe |
| `af` | iid `N(1, 0.25)` | frame-wise | cut-off / hold / volume-matched resume + blend | §5 |

`regular_frozen` was **added during implementation** (it is not in the original 3-rung design): it is
the anchor that proves the new cohort pipeline re-derives a published number, which matters because
§7's protocol change makes these arms non-comparable to the reported 6.62.

**Why the middle rung is named `hrv` and not "healthy HRV".** Jansz 2010's fetal walk
(`RR_n = RR_{n-1} + 0.1(1−RR_{n-1}) + N(0, 7ms/400ms)`) settles at ~±4 % — **less than one frame**
at 12 frames/beat, so it is a null experiment by construction. It remains selectable as
`HRV_MODEL = "jansz"`. The shipped rung is the precedent's ±15 %. **±15 % is ~4× real resting HRV**,
so docs and paper text must state the spread wherever the arm is named.

**Why no `af_compress` arm.** It is `hrv`'s mechanism at sd 0.25 — one mechanism, one parameter,
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
| ↳ **hold is capped at 0.55 nominal R-R (550 ms), by shortening the beat** (2026-09-21, **docs/114**) | the cap VALUE has no source — no paper bounds diastasis in a long beat. 0.55 = the rest a full beat gets at Markl's longest observed R-R (1.55 − 1.0) | **OURS, a modelling bound — present it as one.** Clips 6.0 % of in-window beats. The interim "stretch expansion across the long beat" rule (no physiological support; froze late-ES subjects) and the clamp-to-`es` teleport are both removed — docs/114. |
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

**What the simulation does NOT do — load-bearing for interpretation.** Every frame is rendered by
`render_phase(gt, pos)`, which samples the subject's **own clean 12-phase cine** at a re-timed
position. AF therefore changes **which** cardiac phases appear and **how often**, never what a given
phase *looks like*. A "weak beat" is a beat that skips the filled phases, not a beat with a
different-looking myocardium. Consequence: a single periodic 12-bin cine is in principle expressive
enough to represent every GT target, so a degradation in the Fetal 4D arm is attributable to
**mis-binning**, not to a lack of model capacity. See §12.

## 6. Frame-wise breathing (new default for all arms except `regular_frozen`)

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
comparison is not like-for-like — that rebuilt arm is `regular`. `regular_frozen` keeps the frozen
model precisely so one arm remains bit-comparable to the published `rolled` pipeline.

## 7. The GT protocol change this forces

**Scoring against the clean ED-first `gt_t{k}` cine is wrong here.** Under `af` a slice's 12 frames
may never reach a fully filled ED (cut-off beats), so part of the EF error would be information the
input never contained.

**Correct protocol.** The simulator knows every frame's true cardiac position `p_f`.

- **GT for target f** = the **clean, unbreathed, whole-volume** heart at position `p_f` of the
  **reference (mid) slice** — every plane at the same instant (blend of two clean GT phase volumes).
  It therefore carries the reference slice's own `t0` start offset, but has **no per-slice scatter**
  and **no breathing**.
- **Input** = that same heart state, sliced and breathing-shifted, per slice with its own window.
- **VGGT**: sweep the reference over the mid slice's **12 real frames in acquisition order**
  ("reconstruct the heart as it was in picture f"), not over a chosen phase index. Needs **no**
  readout correction — `run_vggt.py`'s existing sweep sets `timesteps[0,0] = t`, so `pred[f]` pairs
  with `gt_t{f}` directly.
- **Fetal CMR 4D**: its phase-indexed output cine is read out at those same 12 positions,
  interpolating between bins. This **REPLACES** the `gt_ed` roll — see §11.2.
- **EF** (`ef_dice.py`): unchanged formula, `(max − min)/max` of the nnU-Net LV curve, computed over
  the 12 GT volumes and the 12 predicted volumes. Order-independent; ED/ES still `gt.argmax/argmin`.

**What this costs.** Today the scatter bundle gives VGGT's reference slot the **true phase**
(`ref_plane from breath/stack_t{k}`, runner sets `timesteps[0,0] = t`, scored against `gt_t{t}`) —
effectively a gated reference. Under `af` that privilege disappears: only the frames the camera
caught are available. ⇒ **these arms are NOT comparable with the reported `rolled` numbers**;
`regular_frozen` is their reference point, not 6.62.

**What it buys.** No method is scored on information absent from its input — the objection that
motivated the change. A window of weak beats lowers EF in GT *and* prediction together.

**Residual ambiguity to state in the paper.** In real AF clinical EF is itself an average over
beats of varying strength; our definition is *recovery of the heart state at each observed instant*.

## 8. Generator status

**Shipped:** `tools/build_af_bundle.py` (555 lines) — generator + bundle writer. Key symbols:
`rr_sequence` :90, `pos_compress` :107, `pos_physio` :114 (**read its docstring before touching**),
`render_phase` :152, `breathing_model` :181, `simulate` :208, `build` :266, `check` :421, `main` :521.
`ARMS` at :69 maps arm → (rhythm, frozen_breath).

- **Verified:** `regular` + frozen breathing reproduces `rolled/` with **max |diff| = 0.00e+00**
  (so roll, axis order and reslice all match the real builder); the 12-frame run equals the first 12
  frames of a 36-frame run (per-`(subject, plane)` RNG streams, `np.random.default_rng([seed, z])`).
- **`BURN = 3`** beats are played before the window opens — without it every window's first beat was
  a fresh full-strength beat starting from a full heart (bug found and fixed during design).
- Blending is used for **every** non-regular arm (fractional positions), so arms differ only in their
  timing model, not in rendering.
- **32 writer checks pass** across 4 subjects × 4 arms; **6 fault injections** each fire on exactly
  the intended check (perturbed input frame, swapped GT files, GT replaced by source copies,
  perturbed ref plane, `af` pixels degenerated to `regular`, `rr` forced to 1.0).

`temp/arrhythmia_sim/sim_arrhythmia.py` is the **superseded** prototype. It still holds the explainer
GIF code (`<subject>_simple.gif`: one mid slice, tight crop, 36 frames, arms side by side with an
LV-filling trace and beat-boundary markers) and **will drift** — point it at the tracked module
before regenerating figures.

## 9. Open items

1. **Read the summarised sources in full** before citing: Weissler 1968, Markl 2015, Tabata 2001
   (and find a human replacement if it is an animal study), Circulation 1996. Only Chung 2004
   (abstract) and Fürnrohr & Heckel (PDF) have been read directly. **STILL OPEN.**
2. **No beat-to-beat source** for "systole fixed / diastole absorbs" (§5 ⚠ row). Either find one or
   state the inference explicitly in the paper. **STILL OPEN.**
3. ~~Weak beats can still end in a hold.~~ **CLOSED by measurement.** The volume-matched resume is
   the right model and needs no guard: dropping the `max(…, es)` clamp changes 1/38 subjects (a
   no-op), and adding an explicit refractory period on top changes none. The alternative
   "refractory" resume model appeared to improve the low-EF subjects only by *suppressing the
   weak-beat mechanism itself*, which is the physiology being simulated. **No code change.**
4. ~~Cohort LV-volume-curve check.~~ **CLOSED.** Cohort-wide pass over the 38 cmrx2024 test subjects:
   ES falls at frames 5–8 on all 38; 2/38 have one non-monotone systolic step (absorbed by the
   `np.minimum.accumulate` in `pos_physio`); excursion 48–80 %; the jump into ED has median 10.8 %
   and max 36 %. The volume-matched resume is well-posed on this cohort.
5. ~~`evaluation/` edits require explicit user approval.~~ **DONE** — approved and implemented on the
   `exp/af-sim` worktree; see §11.
6. ~~VGGT DVF/segmentation retention.~~ **DROPPED by user decision.** Per-arm segmentations are kept
   (cheap, ~2.9 KB/phase); retaining all 12 targets' `world_points` was ~97 GB and is not needed for
   any metric. `run_vggt.py` therefore needed **no** DVF change.
7. ~~Is "EF MAE 6.6" still current?~~ **CLOSED: yes, 6.62, not stale.**
   `metric_results/test/_ef/fetal_cmr_4d.json` was last written by commit `0c7f87c` (the padded-mask
   rescoring). Per-cohort MAE 5.65 / 6.23 / 6.48 / 7.07 / 7.86 over n = 26/38/49/46/21 = 180,
   subject-weighted mean **6.62**.

## 10. Sources

- **Fürnrohr & Heckel**, *Piecewise Dynamic Diffusion Regularization for Reconstruction of Cardiac
  Cine MRI*, arXiv **2607.03299** (2026) — the closest precedent (same binned CMRxRecon 12-frame
  cine; cycle-duration change by phase-aware frame interpolation). PDF read.
- **van Amerom et al.**, *MRM* 82:1055 (2019) — Fetal CMR 4D; §2.4 + Discussion; local PDF
  `baselines/fetal_cmr_4d/fetal_cmr_4d_paper.pdf`; MATLAB `scratch/fetal_cmr_4d/repo/cardsync/`.
- **Markl et al.**, *JCMR* 2015, PMC4328928 — AF R-R 643 ± 161 ms (range 364–1000). **Full text
  read 2026-09-21: it is a conference abstract** — real-time TEE in **5 AF patients, 2–4
  consecutive cycles each**; mean/sd/range are over those ~10–20 beats pooled across patients, so
  0.25 and [0.57, 1.55] mix within- and between-patient spread. Beat-level, but a thin source
  (docs/114 §2).
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

---

## 11. Implementation record — how the shipped code departs from §1–§10

Everything above §11 is the design. This section is the delta, so a reader never has to diff the
doc against the code to find out which is true.

### 11.1 Architecture: one cohort directory per arm

Each arm is a **new cohort** in the existing bundle layout, `scratch/eval/cmrx2024_<arm>/out/<subject>/`
(`evaluation/README.md:272` documents this as the sanctioned extension point). Every consumer
resolves paths through `paths.<fn>(dataset, subject, …)`, so the scoring chain
(`score/run.py` → `image_metrics` → `pose_psf` → `aggregate`) needed **zero** changes.

Per cohort: `gt/gt_t{00..11}` (per-target GT), `breath/stack_t{00..11}` (simulated input),
a `rolled -> breath` symlink (so `fetal4d_gate.py` needs no change), copied geometry siblings, and a
manifest with a `rhythm` block. `manifest["source"]` deliberately keeps the **base** name `cmrx2024`.

`paths.py` gains `RHYTHM_ARMS` / `EXTRA_DATASETS` / `ALL_DATASETS` (35 names) but these are
**deliberately NOT added to `DATASETS`**: six call sites do `default=list(paths.DATASETS)`, so widening
it would silently pull the rhythm cohorts into every bare sweep in the repo, including the live
campaign's. `ALL_DATASETS` is for argparse `choices=` only.

### 11.2 `ref_phase` REPLACES the `gt_ed` roll (it does not compose with it)

The index semantics change: `t` stops meaning "cardiac phase" and means "the reference slice's
frame f". Under `regular*` the positions are integers, so GT is a pure cyclic permutation of the
source `gt_t*` files (byte-copied) and every metric is invariant. Under `hrv`/`af` positions are
fractional and GT is a genuine blend.

An integer `np.roll` cannot represent a fractional target and cannot repeat one — and under AF the
targets do both (P004 has 11 identical targets). `run_fetal4d.sh` therefore selects the readout from
the **bundle manifest**, never the invocation: a bundle with a `rhythm` key is frame-indexed and uses
`ref_phase`; a legacy bundle has no such key and takes the original `gt_ed` roll verbatim. Both reads
**fail closed** (the script has no `set -e`, so a failed `$( )` would otherwise leave an empty string,
and an empty `READOUT` would silently select the new path and skip the `seg_gt` precondition).

`ref_phase` uses the **method's own gate θ** and opens **no GT file** — which is why the `seg_gt`
precondition now applies only to the legacy roll. Composing the two would double-correct by `gt_ed`:
under a periodic rhythm the gater's ED is `(gt_ed + roll_ref) mod T`, so `ref_phase` reduces exactly
to `np.roll(arr, gt_ed)`.

**This is stricter than the published benchmark.** `gt_ed_roll` aligned to GT's ED, silently
correcting any error in the method's own ED anchor. `ref_phase` does not. Measured previously: the
two rules agree on 31/38 cmrx2024 subjects and differ on 7. Together with frame-wise breathing (§6)
this is the second independent reason these arms are not comparable to 6.62.

### 11.3 The `fetal_cmr_4d_oracle` arm (not in the original design)

Same engine, same parameters, only `cardphase.txt` differs: the **true** per-(slice, frame) θ from the
simulation truth manifest (`theta = 2π(pos % T)/T`), instead of the self-gated estimate. It isolates
**gating error** from **reconstruction error**. `fetal4d_gate.py --oracle` needs no GPU/nnU-Net pass.

`pose_psf.base_method` had to gain `_oracle` in its strip tuple. **Load-bearing**: without it the arm
is not in `PSF_METHODS`, so it would be scored WITHOUT the PSF operator the self-gated arm receives —
a silent handicap having nothing to do with timing.

### 11.4 The 710 count token: verified, and why the oracle needs a constant θ offset

The engine's CLI is `-cardphase N θ₁ θ₂ …`, and because of the documented off-by-one (docs/105 §5c)
the **count** `N` is itself read as frame 0 / slice 0's phase. We pass `710 = 113·2π`.

**Verified against the engine source** (a question three handoffs carried): `cardPhase.push_back(atof(...))`
(`reconstructCardiac.cc:400`) stores the value raw, and its sole consumer `CalculateAngularDifference`
(`ReconstructionCardiac4D.cc:555-561`) computes `(cp−cp0) − 2π·floor((cp−cp0)/2π)` — a true modulo at
any magnitude. `angdiff(710)` is bit-identical to `angdiff(710 mod 2π)`. **Published Fetal 4D numbers
are unaffected.** (710 mod 2π = 6e-05 rad; the in-code comment saying 0.0004 is off by ~7×.)

For the self-gated arm `assemble()` puts an ED frame first, so θ[0][0] = 0 exactly and the literal
works. For the **oracle** arm θ[0][0] is fractional. A search for an integer whose residue mod 2π
matches it was written and **deleted — it cannot work**: 710 is a convergent of 2π
(6, 19, 25, 44, 333, 710, 103993), so reachable residues have ~9e-3 rad gaps; widening 710 → 20131
buys nothing and the next improvement needs n ≈ 103,993. Measured: 58 % of random θ₀ fail a 1e-3
assert, 11/16 real built cases fail. **Replaced** by shifting every θ by a constant so θ[0][0] ≡ 710
mod 2π (`fetal4d_gate.py:193`). A constant offset only relabels which output bin is "phase 0", and the
`ref_phase` readout samples the SAME shifted θ, so it cancels exactly. Residual 2.89e-07; argv stays
710 tokens. Verified on the built P016 oracle gate: every frame is offset by exactly −2.81105 in
cine-frame units.

### 11.5 `pos_physio` is solved in STORED-phase units — the ED-anchored version was reverted

> ⚠️ **CORRECTION (docs/111 §5, measured):** the invariant asserted below and in `pos_physio`'s
> docstring — *"all four arms open their window at the same stored phase and differ ONLY in
> rhythm"* — is **FALSE for `af`**. `simulate`'s `t0` is a **time** offset `frac·rr[BURN]` into
> beat 3. Under `pos_compress` (uniform stretch) that maps to phase `frac·T` exactly, so
> `regular*`/`hrv` open at `(-roll[z]) % T` with deviation 0.00. Under `pos_physio` phase advances
> at a fixed rate, so the open phase is `min(p_start[BURN] + frac·rr[BURN]·T, T)`, equal to
> `(-roll[z]) % T` only when `rr[BURN] == 1` **and** `p_start[BURN] == 0`. Measured deviation up to
> **5.52 phases**, with **16.4 % of `af` frames parked at the full-ED hold** (0 % in every other
> arm). The *behaviour* is defensible — a free-running camera opens at a uniformly random time
> inside a beat — but the **control claim is not**, and cross-cohort `af`-vs-`regular` deltas are
> therefore not rhythm-only. Within-cohort comparisons (e.g. the oracle gap) are unaffected.
> Fix is either to correct this text or to solve `t0` for a target phase under `pos_physio`; note
> `p_start[BURN] > frac·T` makes the latter unsolvable for some draws.

`pos_physio` takes `vol[0]` as "full". The stored cine is ED-first by convention but not always: the
seg-derived ED is frame 10/11 on 4 of the 38 cmrx2024 test subjects (measured: P046 ED = frame 11,
`vol[0]/vol[ED] = 0.959`) and ~14 % of CMRx25 (docs/105 §5d).

Solving in an ED-rolled frame was implemented to fix this, then **reverted**: it shifted the entire
`af` timeline by `ed` relative to every other arm (measured on P046: `pos_compress = [4,5,6…]` vs
`pos_physio = [3,4,5…]`), breaking the "arms differ only in rhythm" control for exactly the subjects
it meant to help. Patching `p_start[0]` does not work either — the resume recurrence resets every
later beat to ED, so the beat *grid* is ED-anchored by construction. The residual limitation
(hold parks at ~96–99 % of true ED, ~1-frame effect on ~10 % of subjects) is **disclosed in
`pos_physio`'s docstring** rather than fixed. **Regression invariant: with `rr ≡ 1`, `pos_compress`
must equal `pos_physio`.** Revisit only with a scheme that keeps the stored-phase time origin.

### 11.6 VGGT slot seeding is keyed on the base source

`run_vggt.py:443` now uses `seq = name_seed(man.get("source", ds_name), subject)`. A rhythm cohort
holds the same subject re-simulated; seeding on its **directory** name would give a different
`random.Random(seq_index)` and therefore a different slot **order** than the base cohort's run. The
slot *set* is identical either way (one frame per slice, slot 0 = `z_mid`) and `pin_scatter`
overwrites every companion phase, but VGGT is not permutation-equivariant, so a reordered draw would
stop `regular_frozen` being an exact control.

**Verified inert for every existing bundle:** all 291 live bundles across the 7 base cohorts have
`manifest["source"] == <directory name>`, so the hash is unchanged and no published number moves.

### 11.7 Smaller edits, each with a failure it prevents

- `run_baselines.py` — `thickness_mm(m["source"], …)` (thickness is a scanner property, not a rhythm
  property; `thickness_mm()` raises on an unknown name, so the dir name would hard-fail these
  cohorts); `GATE_ARM` support with a **refusal** when `GATE_ARM != METHOD` (mis-pairing them is
  otherwise silent and mis-attributes the recon); `--input scatter` now guards on the **directory**,
  because a rhythm cohort carries `manifest["scatter"]` (needed by `run_vggt.pin_scatter`) but writes
  no scatter stacks, so a key-only check passed and then failed inside the container.
- `fetal4d_gate.py` — `--work-tag` (without it `dump` writes into the shared `_fetal4d_gate/test/`
  holding 2160 live campaign inputs, and it has no skip guard), `--oracle`, `gate_dir(ds, subj, arm)`,
  `CARDPHASE_COUNT_TOKEN`.
- `build_af_bundle.py` writes `seg_gt` / `heart_seg.nii.gz` only when **every** target is an existing
  integer phase (a byte-exact permuted copy); under blended GT they are omitted and recorded in
  `heart_siblings_missing`, so a consumer fails loudly rather than pairing a stale phase index.

### 11.8 Data-safety traps (read before running anything)

- `image_metrics.py:399` defaults `EVAL_DATASET` to `"cmrx2024"`, and the rhythm cohorts reuse the
  **same subject names AND arm names**. A bare `python image_metrics.py <subj> <arm>` would overwrite
  live campaign `metrics.json` / `cine_*`, and `_guarded_write_path` whitelists exactly those
  basenames. **Always drive scoring via `score/run.py --datasets <cohort>`.**
- `run_fetal4d.sh:19` hardcodes `VGGT=/home/minsukc/vggt` (same for `run_seg.sh:16`), so worktree
  edits to `baselines/fetal_cmr_4d/` have **no effect**. Left alone deliberately.
- `run_vggt` prints `!! N companion slot phase(s) re-pinned to the bundle's scatter draw` on every
  rhythm-cohort subject. That is the frame-index remap working. **Expected, not an error.**
- **P004 is kept deliberately.** Its window sees only 21 % of the LV excursion and holds at full ED
  for 11 of 12 frames, so 11 GT targets are byte-identical and its GT EF collapses to ~15 % vs a true
  71 %. ⚠️ **Cause CORRECTED (docs/111 §5):** not "a long effective beat with the camera window
  opening inside its diastasis" as originally written, but the §11.5 `frac·rr[BURN]` time→phase
  conversion — with `roll[4] = 8 ⇒ frac = 1/3`, every other arm opens at phase 4 while `af` opens at
  phase **12.0** (full ED) and stays there.
  This is the model working as specified — both alternative resume models give the identical 21 %.
  Cohort-wide it is rare (median window sees 99 %; 1/38 below 30 %). Its signature is
  `rhythm.duplicate_targets` in the manifest. **Do not average its EF in silently.**
  P092 (36 %) and P034 (47 %) are a *different* phenomenon and not a bug: short beats ⇒ chronic
  underfilling ⇒ the heart genuinely ejects less, which is the AF physiology being simulated.

### 11.9 Pre-existing issue found en route (NOT from this branch)

`fetal_cmr_4d_mb3`, `fetal_cmr_4d_norobust` and `fetal_cmr_4d_motion` all carry `"psf": "none"` in
their `metrics.json` while `fetal_cmr_4d` has `"gauss_thickness"` — `_mb3`/`_motion` are not in
`pose_psf`'s strip tuple, and `_norobust`'s metrics predate its addition. Since the PSF blurs the
recon toward the GT's resolution, omitting it generally **lowers** PSNR (measured: 20.45 / 19.44 /
18.2), so those ablation comparisons in existing results are not like-for-like. **User's call whether
to rescore.**

## 12. What a result will and will not license

**Fairness — the comparison is sound.** Their assumption is preserved exactly (§2); the front end we
substituted is *better* than their own, which provably cannot run on a 12-frame window (§3); the
oracle arm measures the ceiling their design cannot reach (§11.3); PSF treatment is equalised across
arms; `regular_frozen` proves the pipeline re-derives a published number; and every arm shares
identical inputs, masks, geometry and scoring code.

**Scope — the claim must be narrower than "AF patients".** The AF model is ours: the volume-matched
resume, the 0.45 floor and the full-ED hold have no precedent (§5), and the input is simulated from
healthy gated cine, not acquired from patients in AF. Supportable: *"the constant-heart-rate gating
assumption fails under non-periodic rhythm, and here is the cost."* Not supportable from this
experiment: *"self-gated 4D SVR cannot work under AF"* — that is a claim about a method class we have
not tested. Also state plainly that VGGT is **told** the target instant while Fetal 4D must infer it;
that asymmetry is inherent to what the two methods are, not a defect to fix, but a reader should not
have to discover it.

**Pre-registered outcomes** (decided before any number exists):

| outcome | reading |
|---|---|
| self-gated degrades, oracle holds | the bottleneck is the **constant-HR gating assumption** — the headline result this doc set out to test |
| both degrade | **unlikely by construction** (§5, last paragraph): the simulation only re-times one clean periodic cine, so a 12-bin periodic model is expressive enough. If it happens anyway, the cause is upstream of gating and must be found, not assumed |
| neither degrades | rhythm irregularity does not bite at a one-beat window. `hrv` vs `af` then locates the threshold |

**A caution on reading a VGGT win:** VGGT was trained on periodic gated cine, so AF input is
out-of-distribution for it too. `regular` vs `af` **within** VGGT is the number that says whether it
actually survives, independent of how the baseline does.

**Not yet built, and deliberately so:** a gate that reads the *shape* of the per-slice LV-area curve
rather than only its argmax. `fetal4d_gate.py:161` already computes `area` as a (Z, T) trace and uses
only `detect_ed`'s argmax; under AF a hold should in principle appear as a flat segment there. Whether
that is detectable above segmentation noise at 12 samples is **unmeasured and remains a hypothesis**.
Building it would mean inventing a better baseline on the authors' behalf — a different paper.
