# 114 — AF simulator: hold rule (capped 550 ms), teleport fix, build-time post-conditions

> **TL;DR & takeaway**
> The `af` position model (`tools/build_af_bundle.py`, `physio_beats` + `pos_physio`) changed on
> 2026-09-21. **Long beats now play at the recorded rate and then HOLD at full relaxation**, instead
> of stretching expansion across the beat; the hold is **capped at 0.55 nominal R-R (550 ms)** by
> shortening the beat. A beat **cut during contraction now resumes at the phase it reached**
> instead of being clamped forward to ES. Both defects shipped in the `af24` bundles (freeze 12/180,
> teleport 36/180), so **`af24` must be rebuilt and re-gated; `regular24`/`hrv24` are untouched
> (bit-identical, 180/180 each)**. The hold *mechanism* is citable (Chung 2004); **the cap *value*
> is a modelling bound — no paper bounds diastasis in a long beat — and must be presented as one.**
> New hard build-time checks run on every plane; both were fault-injected and fire.

## 1. What was wrong (measured on the shipped `af24` manifests)

| defect | where | extent | cause |
|---|---|---|---|
| **teleport** | old `pos_physio`, `p_end = max(min(p_end, T), es)` | 36/180 subjects, up to +5.08 phases | A beat cut *during contraction* (`rr <= t_c`) was clamped UP to `es`, so the next beat resumed fully contracted — more emptied than the heart ever got. The branch was commented "unreachable in practice"; it is reachable whenever `es >= 6` (`t_c = es·DT` vs the fixed `RR_MIN = 0.45`). |
| **freeze** | old long-beat rule: expansion stretched to end exactly at `rr` | 12/180 subjects, up to 16 near-static frames | With late ES, `T − es` is 2–4 phases; stretching them over ~1 s gives ~0.12 phase/frame. |

Neither was caught because `check()` only inspects the **reference plane's GT targets**; both
defects live on the **input planes**.

**Refuted hypothesis (measured, 180 subjects):** the long rests are *not* an ED-anchoring artifact.
174/180 subjects have ED (argmax LV volume) at frame 0/1/11, and they still reach uncapped holds
of 1335 ms. Long holds occur at every `es` (per-subject max hold, mean: 580–710 ms for `es` 4–6,
~1000 ms for `es` 8–10). They come from the model itself: a weak resumed beat has a short nominal
duration (`t_nom` down to ~167 ms) and is followed by a normal or long R-R.

## 2. The new rule

`physio_beats(rr, vol) -> (realised rr, p_start, es)`; `pos_physio(times, rr, vol)` requires the
realised rr (asserted).

1. Contraction **and** expansion advance at the stored cine's rate.
2. Beat longer than its own nominal (`t_nom = t_c + (T−es)·DT`) → finishes, then holds at
   `pos == float(T)` until the next trigger.
3. **Cap:** `rr[i] = min(rr[i], t_nom + HOLD_MAX)`, `HOLD_MAX = 0.55`. The realised rr drives
   `starts` and `t0`, so `simulate()` computes it *before* building the timeline. Idempotent.
4. Beat cut during contraction (`rr <= t_c`) → next beat starts at `p_start + rr/DT` (no clamp).
5. Beat cut during expansion → volume-matched resume, unchanged.

### Why a hold, and what is and is not citable

- **Citable — the mechanism.** Chung, Karamanoglu & Kovács, *AJP Heart* 2004 (PMID 15217800,
  abstract read verbatim): E- and A-wave durations are "very nearly independent of HR" (100 % HR
  increase → 18 % shorter E-wave) and "diastasis duration closely tracked MDD". Active phases keep
  their duration; the rest period absorbs the cycle length. Caveat: bicycle exercise in healthy
  volunteers, so R-R ≲ 1 s and steady-state — applying it beat-to-beat and to long R-R is
  inference (same caveat as docs/110 §5). Lin et al., *EHJ-CVI* 2023 (PMID 35718877): cine-MRI
  diastasis = 24.2 ± 4.2 % of diastole in 20 healthy controls, HR ≤ 80.
- **The stretch has no support** — nothing says relaxation velocity scales with cycle length.
- **NOT citable — any cap value.** No source bounds diastasis duration in a long beat. "≈160 ms
  healthy diastasis" (an earlier agent's conversion of the 24.2 %) is a steady-rate figure, not an
  upper bound. Also corrected: PMID 23812582 says diastasis time is "markedly shortened (<83 ms)"
  at HR > 75 — not "85 ms" (that is its IRT figure); echo in CT patients, weak support.
- **Why 0.55.** It is the rest a *full* beat gets at the longest R-R in our cited AF range
  (Markl 2015: 364–1000 ms around a 643 ms mean → [0.57, 1.55] nominal; 1.55 − 1.0 = 0.55). So
  the cap never touches a full beat inside the cited range; it only stops a *weak resumed* beat
  from resting longer than any full beat could. **Source strength, checked 2026-09-21 (PMC4328928
  full text read):** Markl 2015 is a *conference abstract* (JCMR 17 Suppl 1) — real-time TEE in
  **5 AF patients, 2–4 consecutive cardiac cycles each**; "the average R-R interval duration for
  all five patients was 643 ± 161 ms, with a range of 364–1000 ms". So the range IS over individual
  beats (good — not a spread of per-patient means), but it is ~10–20 beats pooled across 5
  patients and normalised by the POOLED mean, so 1.55 mixes within- and between-patient spread.
  It is the same source the locked sd = 0.25 rests on; it supports 0.55 as "the longest beat
  reported there", not as a population bound. Rough plausibility check (arithmetic, E-wave/IVRT
  durations from memory, NOT sourced): at R-R 1.55 s, Weissler systole ≈ 465 ms + IVRT ≈ 80 +
  E-wave ≈ 200 leaves ≈ 0.8 s of rest; our 0.55 hold + the recorded cine's own diastasis is the
  same order.
- **Paper wording:** "Long beats play at the recorded rate and then rest at end-diastole
  (diastasis absorbs cycle-length variation [Chung 2004]). As a modelling bound, no beat rests
  longer than a full beat would at the longest observed AF R-R (0.55 of the nominal cycle); this
  shortens 6 % of beats."

## 3. Measured effect (all 180 `af24` subjects, positions re-simulated, read-only)

| | stretch (shipped) | hold, uncapped | **hold, cap 0.55 (new)** |
|---|---|---|---|
| teleports > 1.5 phases | 15 subj (max 5.08) | — | **0** |
| longest hold | n/a (freeze ≤ 16 frames) | 1538 ms / 16 frames | **550 ms / ≤ 7 frames** |
| in-window beats clipped | 0 | 0 | **343 / 5712 = 6.0 %** |
| R-R sd, drawn → realised (all 16 beats/plane) | — | — | **0.2390 → 0.2302** |
| ref-plane LV excursion mean / median / min | 96.0 / 97.5 / 50.2 % | 98.3 / — / 71.1 % | **98.3 / 99.4 / 71.1 %** |
| subjects < 50 % excursion | 0 | 0 | **0** |
| distinct GT frames of 24, min / median | 24 / — | 12 / — | **14 / 22** |
| P004 (the 12-frame degenerate case) | — | — | **99.4 % excursion** |

Stretch/uncapped columns are from the 2026-09-20 handoff measurements. All 180 subjects' `af24`
positions change; `regular24` and `hrv24` re-simulate **bit-identical** to the shipped manifests
(180/180 each) — they use `pos_compress` and never touch this code.

Duplicate GT frames are physically honest (a resting heart photographed twice) and do not affect
EF (`(max−min)/max`), but a held target is multiply weighted in per-frame PSNR/SSIM means.

## 4. Build-time post-conditions (`physio_postconditions`, hard `SystemExit`, before any write)

On **every plane**: (a) longest run of frames at `pos == T` ≤ `HOLD_MAX_FRAMES` = 7 (exact — a
held frame sits at exactly `float(T)`); (b) no forward step > `JUMP_MAX` = 1.5 phases across a beat
boundary (a beat cut in systole legitimately continues ~1 phase). Manifest gains
`rhythm.physio = {max_hold_ms, beats_clipped, beats_in_window, rr_sd_drawn, rr_sd_realised}`,
`rhythm.rr_drawn_per_plane` (`rr_per_plane` is now the REALISED sequence) and
`params.HOLD_MAX`.

**Fault-injected:** a +3-phase jump across a boundary and a 9-frame hold each fire on 180/180.

## 5. Consequences / to do

- **`af24` REBUILT 2026-09-21** (array 61615769, 20 shards, all COMPLETED) into
  `scratch/eval/_rebuild_af24_hold_20260921/`, verified file-by-file (180/180, 0 problems: 24
  stacks + 24 GT, `T=24`/`n_cardphase=12`, seed/scatter/rolled/breath identical to the old bundle,
  hold runs ≤ 7, no jump > 1.5, no backwards step within a beat, last stack + GT load finite;
  max hold 550 ms, 343/5712 clipped — identical to the read-only prediction), then swapped in.
  The stretch-rule bundles **and their 180 stale gates** are kept at
  `scratch/eval/_archive_af24_stretch_20260921/`.
- **Re-gate: array 61616097**, work tag `af24hold_s*`. ⚠️ **Stale-seg trap:** `nnUNet_predict`
  skips outputs that already exist and a rebuilt cohort reuses the same file names, so re-gating
  into the old `af24_s*` work dirs would have fed the gate the PREVIOUS bundles' segmentations with
  no error. `sbatch/gate_af24.sh` now takes `TAG_PREFIX`; use a new one after every rebuild.
- **Also fixed the same day** (2026-09-20 prove-it findings): `run_fetal4d.sh` no longer honours an
  inherited `T` (it forced `-numcardphase 24` → 12 of 24 bins empty; executed on real manifests
  with `T` unset and exported: `NF=24 NCP=12`, and `12/12` on a 12-frame cohort); its `SEG_GT`
  follows `SEG_CFG` like `paths.seg_gt_dir`; `tools/check_gate_24.py` refuses to run over an
  existing gate and parks its synthetic gate outside the bundle (it used to overwrite the live
  gate, stamped `"gater": "self"`); a degenerate LV curve now skips the subject (`DegenerateLV`)
  instead of aborting the batch.
- The superseded 12-frame AF arms (`af`, `af_rvr`, `af_pause`; docs/111–112) share this code and
  would change if rebuilt. They are not being rebuilt.
- Legacy demo scripts that call `pos_physio` with the *drawn* rr (`render_stride_demo.py`,
  `render_af_rule_gif.py`, `render_rhythm_demo.py`, `render_af24_demo.py`,
  `af_degeneracy_cause.py`) now hit the realised-rr assert; pass `physio_beats(rr, vol)[0]`.
- Preview: 5 subjects (one per source) rebuilt under
  `scratch/eval/_preview_afhold_20260921/`; GIFs (regular24 | hrv24 | af24 + LV trace) in
  `temp/af24_holdfix_demo/`.
- Unquantified: whether ED-static stretches help or hurt the Fetal self-gate — check
  `bin_phase_sd` after the re-gate.
