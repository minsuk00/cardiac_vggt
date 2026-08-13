# 69 — exp66 campaign results: wave 1 is NOT a negative. `Z_HALF_MM` × respiratory is a real interaction

> **TL;DR & takeaway**
>
> **⚠️ THIS DOC WAS REWRITTEN 2026-08-11 AFTER ITS ORIGINAL CONCLUSION WAS FALSIFIED.** The first
> version said "wave 1 is a clean negative, no knob matters." **That was wrong, and it was wrong
> because of a scoring bug, not because of the models.**
>
> **The finding.** Arm **x1** (`Z_HALF_MM` 90→66 **AND** respiratory sim off) scores
> **`amp_ratio` = 0.526** at 56.4k steps — against a broken control of 0.142 and good anchors at
> 0.740/0.747. **This is the campaign's first positive result.** It is a genuine **two-factor
> interaction**: neither knob does anything alone (a2 old-z-only **0.140**, a3 resp-off-only
> **0.152**, c0 control **0.142**), but together they give 0.526. A one-factor-at-a-time campaign
> is structurally blind to this, which is exactly why wave 1 read as a negative.
>
> **Why it was missed.** x1's originally-recorded score was **0.115**. It is really **0.526**. Two
> defects in `tools/e0_dump_phase_sweep.py` made that possible and BOTH still exist: (1) dumps
> record **no provenance** — no tree, no ckpt, no overrides — so a mis-decoded dump is
> indistinguishable from a good one afterwards; (2) **`ef_val_sweep` defaults to `true` and
> silently overrides `t_target_fixed`**, so a "phase sweep" can decode the same ED/ES pairs 12
> times. x1 must be decoded through `vggt-arm-a2` (`Z_HALF_MM=66`); through the main tree at 90
> every z is off by 1.36× and `amp_ratio` collapses. **The user caught this from the wandb DVF /
> volume panels while the agent kept quoting the bad number.**
>
> **The data lead is DEMOTED.** x1 ran on **current v2 data**. So v2 data does not prevent
> bootstrapping ⇒ the v1/v2 data difference is **not necessary** for it, at most an alternative
> route. (v1↔v2 is still fully characterised here: apex flip + odd-Z roll + the docs/54 ESPIRiT
> B1− shading bug.)
>
> **Still true from the original:** P0 (June's complete code tree) fails at 0.119; p0dino
> (DINOv2 stem unfrozen) is refuted and actively harmful; `pearson_median` ≈ 0.98 in EVERY arm ⇒
> the failure is **amplitude**, with timing/shape intact.
>
> **Never done, still the cheapest high-value check:** score **4wok's own checkpoint** by this
> evaluator (§11.1). Every "the model regressed" claim rests on a premise nobody has verified.

**Date:** 2026-08-08 → rewritten 2026-08-11.
**Predecessor:** [`66_…_experiment_plan.md`](66_fresh_bootstrap_4wok_regression_audit_and_experiment_plan.md) (the plan).
Doc 66 §7's ranking (compile #1, old-z #2, respiratory #3) is **partially vindicated**: old-z and
respiratory ARE the live factors — but only *jointly*, which doc 66's one-factor design could not
have found. Doc 66 did preregister x1 as the deliberate two-knob arm; that decision is what saved
the campaign.

---

## 1. What is being debugged

**The capability.** Slot 0 of the input is a real target-phase reference slice at the
mid-ventricular plane, marked by VGGT's native two-token `camera_token`. The model must read
contraction state from slot-0's *image content* and propagate it through global attention to
displace the remaining slices, which are each frozen at a random, unknown cardiac phase. Per the
information contract (docs/04), input `t` and respiratory phase `r` are withheld.

**The regression.** Warm-starting from the June 2026 "4wok" checkpoint retains patient-specific
contraction. Fresh-from-base runs since a large refactor do not: the reconstruction barely
contracts, by roughly the same amount for every patient.

**Do not confuse with docs/64** — a separate, solved dying-ReLU collapse at lr=3e-4. Every run here
is at a healthy 5e-5. Any `cond_ratio` evidence on `214366719_*` is VOID (that is the dead model).

---

## 2. The evaluator, its anchors, and ITS TWO FOOTGUNS

### 2a. The tools

- **`tools/e0_dump_phase_sweep.py`** — decodes a checkpoint into a 12-phase val sweep by invoking
  **that checkpoint's own code tree** via `training/launch.py mode=val t_target_fixed=k`. ~2 min
  per phase warm (~20 min/arm). The point of using the arm's own tree is that A2/X1 decode with
  `Z_HALF_MM=66`, P0 with old-code geometry, A4 with its own respiratory sampler.
- **`tools/e0_score_volumes.py`** — CorSeg-segments pred+GT, computes EF vs `cardiac_phase.csv`
  with bootstrap CIs, plus per-subject phase-transfer slope and `amp_ratio`. Cache GT segs with
  `--gt-seg-dir result/e0_dumps/_gt_segs_cmrx24val` (makes re-scoring ~5 min).

**Verdict metric: `amp_ratio`** = (max−min predicted LV volume across the 12 phases) ÷ (same for
GT), cohort median. **≥0.4 bootstrapped, ≤0.2 failed.**

### 2b. ⚠️ FOOTGUN 1 — `ef_val_sweep` silently overrides the phase sweep

`ef_val_sweep: true` is a **top-level default** in `default.yaml:99`. It replaces the val sampler
with a targeted {ED, ES} sweep and **ignores `t_target_fixed` entirely**. `e0_dump_phase_sweep.py`
does NOT disable it — it passes `t_target_fixed=k` and trusts it.

Symptom (hit on 2026-08-11, ~25 min of GPU wasted): every `tNN/` dir holds 116 files instead of 58,
and `ls tNN | grep -oP '_t\d+_' | sort -u` returns a MIX of labels rather than one. The scorer then
reports "29 subjects x 6 phases" and returns `nan`.

**ALWAYS pass `ef_val_sweep=false`,** and always verify before scoring:
```bash
for t in t00 t06 t11; do
  echo "$t: $(ls DUMP/$t | wc -l) files, labels $(ls DUMP/$t | grep -oP '_t\d+_' | sort -u | tr '\n' ' ')"
done
# CORRECT: 58 files, exactly one label per dir, matching the dir name
```

### 2c. ⚠️ FOOTGUN 2 — no provenance is recorded, and the wrong tree silently ruins a score

A dump dir contains only `t00/…t11/` + seg caches. **No tree, no ckpt path, no overrides.** So a
mis-decoded dump cannot be distinguished from a good one after the fact — which is how x1's 0.115
survived into the handoff table and got quoted for a whole session.

**Only two arms need the alternate tree — and one of them was the one that was wrong:**

| arm | MUST decode through | why |
|---|---|---|
| **a2, x1** | `/home/minsukc/vggt-arm-a2` | trained at `Z_HALF_MM=66`; the main tree decodes at 90 ⇒ every z off by 1.36× |
| P0 / p0v1 / p0dino | `/home/minsukc/vggt-oldcode-p0` | old geometry; **also `--dz 12`** (identity affines) |
| everything else | `/home/minsukc/vggt` | — |

**Recommended fix (NOT YET DONE):** have `e0_dump_phase_sweep.py` write a `_manifest.json`
(tree, ckpt, resolved overrides, git rev) into the dump dir, and either force `ef_val_sweep=false`
or refuse to run when it is on.

### 2d. Anchors (n=29, truth known independently)

| anchor | dump | truth | **amp_ratio** | transfer sl | ef_csv sl | pearson |
|---|---|---|---:|---:|---:|---:|
| fresh-pooled `214141126_*` | `anchor_freshpooled_aug` | BAD | **0.136** | 0.129 | 0.625 | 0.984 |
| warm-start `213966986_*_4wok_*` | `anchor_warmstart` | GOOD | **0.747** | 0.754 | 0.849 | 0.978 |
| 8mm replay `213853800_*` | `anchor_replay8mm` | GOOD | **0.740** | 0.789 | 0.829 | 0.987 |

⚠️ **The 8mm replay is NOT a valid positive control** — pre-relabel code resampling 12→8mm makes the
task measurably easier: padding inverts to cropping (at 12mm ~96% of subjects carry 1–3 zero planes;
at 8mm they resample to ~14 slices and get *cropped*, all real), and interpolation low-passes the
stack (adjacent-slice correlation **0.857 vs 0.645**).

---

## 3. The hub recipe (shared by every wave-1 arm)

`sbatch/exp66_common.sh`; thin arm scripts set `TREE`/`ARM_OVERRIDES` and source it.

```
HUB_OVERRIDES   data.augmentation.enable=true data.augmentation.tier=moderate
                loss.volume.gather_weight=0.0
                split_file=training/splits/cmrx24only.txt dataset_name=cmrx24only
                limit_train_batches=235 limit_val_batches=58
                data.{train,val}.dataset.dataset_configs.0._target_=fixed12_dataset.MRIDatasetFixed12
RECIPE          max_epochs=300 (=70.5k steps) seed_value=42 checkpoint.save_freq=60
                optim.optimizer.lr=5e-5  (+ the two coupled scheduler endpoints — LR IS THREE KNOBS)
```

Config `default`; fresh-from-base VGGT-1B; `PYTHONPATH=temp:training:.` (`fixed12_dataset` lives in
`temp/`).

**The hub already pins four factors**, clearing them by construction: gather is **off**, slice count
is **fixed D=12**, cohort is **CMRx24-only**, split is the **deduped** `cmrx24only.txt`. The
fresh-pooled anchor had the opposite on all four and also failed ⇒ cohort, gather, native-z and
split are cleared not by dedicated arms but because *both* settings fail.

---

## 4. Wave 1 — every arm, provenance, and CORRECTED results

All ten run **current code**. Five override config on the pristine main tree; four run isolated
copies differing by **exactly one file** (`diff -rq` verified); x1 reuses a2's tree.

| arm | tree | the one change | what it tests | old score | **re-scored @ckpt240 (56.4k)** |
|---|---|---|---|---:|---:|
| **c0_hub_repeat** | `vggt` | *(none)* | **the control** | 0.142 | not re-scored |
| a1_nocompile | `vggt` | `cuda.compile_attention_blocks=false` | compile's first-update cosine 0.48 vs eager | *never scored* | – |
| **a2_oldz** | `vggt-arm-a2` | `preprocess.py:52` `Z_HALF_MM 90.0→66.0` | old index-z alone | 0.149 | **0.140** |
| **a3_noresp** | `vggt` | `data.augmentation.respiratory.enable=false` | breathing sim off alone | 0.157 | **0.152** ✅ reproduces |
| a4_legacyresp | `vggt-arm-a4` | `respiratory.py:266-268` tilt per-SLOT `(B,S)`; + `tilt_max_deg=null amplitude_mm=16.0 group_by_burst=false per_slot=true` | legacy respiratory simulator | 0.167 | not re-scored |
| a5_z8400init | `vggt` | `checkpoint.resume_checkpoint_path=…/vggt1b_base_a5_z8400.pt` | z-embedder init replayed from 4wok's effective seed 8400 (built by `tools/e0_make_a5_base.py`) | 0.145 | not re-scored |
| a6_withreplacement | `vggt-arm-a6` | `composed_dataset.py:141,155` legacy `inside_random`; + `+data.train.common_config.inside_random=true` | with-replacement subject sampling | 0.167 | not re-scored |
| a7_epochclock | `vggt-arm-a7` | `dynamic_dataloader.py:57,214-270` 1000-step epochs; `max_epochs=70 limit_train_batches=1000` | old epoch definition | 0.140 | not re-scored |
| a8_4wok_lrtraj | `vggt` | `max_epochs=851` (→200,085 steps) | **LR-vs-step matched to 4wok.** Scheduler is fraction-of-training (`lengths [0.05,0.95]`), so the hub's 70.5k compresses 4wok's 200k **2.8×**: warmup 3,525 vs 10,000; at 42k hub LR ≈1.9e-5 vs 4wok ≈4.7e-5. Doc 66 "cleared warmup" citing the pooled run's *14,025*-step warmup — that clears a LONGER warmup, not a SHORTER one. Promoted by red-team review. | *never scored* | – |
| **x1_oldz_noresp** | `vggt-arm-a2` | a2's tree **+** `respiratory.enable=false` | the deliberate TWO-knob arm | 0.115 ❌ | **0.526** ✅ |

### 4a. THE RESULT — a 2×2 interaction

| | respiratory **ON** | respiratory **OFF** |
|---|---|---|
| **`Z_HALF_MM` = 90** | c0 **0.142** | a3 **0.152** |
| **`Z_HALF_MM` = 66** | a2 **0.140** | **x1 0.526** |

**Neither factor moves the needle alone; together they take amp_ratio from ~0.14 to 0.53.**
x1's full numbers: transfer slope median **+0.525** (IQR 0.434–0.602), EF slope **+0.586**,
pearson 0.963, 29 subjects × 12 phases.

**Why the a3 re-score matters as much as x1's.** a3 reproduced its old number (0.152 vs 0.157)
under the *identical* new procedure. That rules out "the agent's dump method differs" and localises
the discrepancy to x1 specifically — consistent with the wrong-tree explanation, since a3 lives on
the main tree either way and a2/x1 are the only arms needing `vggt-arm-a2`.

### 4b. P0 — old code, still fails

| arm | tree | change | amp_ratio |
|---|---|---|---:|
| **P0_faithful4wok** | `vggt-oldcode-p0` (pristine `8f7b94c`) | *none*; `config=mri_volume_diffusion max_epochs=200 split_file=random_8_1_1_prefixed.txt` | **0.136 @25k → 0.119 @35k** |

**P0's provenance is settled.** 4wok's commit `bc2553c` is lost, but its `wandb-metadata.json`
records the launch args verbatim:
```json
"args": ["--config","mri_volume_diffusion","exp_name=217720691_…","max_epochs=200"]
"git": {"commit": "bc2553c2efef3159ae8baa2ee7acae0453f7e37e"}
```
No overrides ⇒ 4wok took the chain defaults. And its `output.log` prints
`target_spacing=(1.4, 1.4, 12.0) target_shape=(256, 256, 12)` ⇒ **4wok ran at 12.0mm**, matching P0
(`8f7b94c`). The 8mm replay (`13e37fb`) was the artifact.

**⚠️ P0's 0.119 has NOT been re-scored and shares the provenance gap.** It is the sole basis for
"June's complete code also fails, therefore the code is exonerated" — a load-bearing claim that is
currently unverified.

### 4c. Respiratory / augmentation settings, verified from source

| run | respiratory sim | affine/photometric aug |
|---|---|---|
| 4wok | **ON**, `amplitude_mm: 16.0` (chain default, no override) | **OFF** (`gpu_aug.py:70` in its log) |
| P0 / p0v1 / p0dino | **ON** (same chain default at `8f7b94c`, `mri_volume.yaml:137`) | **OFF** |
| every wave-1 hub arm | ON, except a3/x1 | **ON**, tier `moderate` |

The old tree logs **nothing** about respiratory (0 occurrences of "resp" in 4wok's, P0's and
p0v1's logs), so absence of resp metrics is NOT evidence it was off — this had to be settled from
launch args + the config chain.

---

## 5. Wave 2 — testing what lies outside the code

Both run the pristine `vggt-oldcode-p0` tree, differing from P0 by one launch override. ⚠️ Both on
**L40S (spgpu2)** while P0 is on **A40 (spgpu)** — a second variable, probably minor (old tree is
eager, non-fused AdamW), but noted.

| arm | override vs P0 | hypothesis |
|---|---|---|
| **p0v1_olddata** (56854632) | `data_root=data/CMRxRecon2024_recon_v1_espirit_imagedomain` | June's code **+ June's data** |
| **p0dino_newdata** (56854633) | `optim.frozen_module_names=[]` | unfreeze the DINOv2 stem |

### 5a. p0dino — REFUTED, and it actively hurts

| window | metric | 4wok (v1) | p0v1 (OLD) | P0 (new) | **p0dino** |
|---|---|---:|---:|---:|---:|
| 24–26k | psnr_bbox | 27.70 | 27.84 | 27.64 | **25.80** |
| 24–26k | loss_volume | 0.0113 | 0.0110 | 0.0127 | **0.0157** |
| 24–26k | mean_disp_norm | 0.057 | 0.063 | 0.045 | **0.020** |

`psnr_bbox` flat at ~25.8 while everyone else climbs to ~28; displacement pinned at 0.013–0.020.
Reads as the stem degrading under training. **The freeze regime is not the cause; unfreezing is
harmful.** (Kept running past the verdict at user request.)

### 5b. p0v1 — still running, now DEMOTED

Windowed means (matched global step, instantaneous — single logged iters swing 0.49→0.91 and are
noise-dominated):

| window | 4wok (v1) | **p0v1 (OLD)** | P0 (new) | p0dino (new) |
|---|---:|---:|---:|---:|
| **coverage_frac** 4–6k | 0.752 | **0.751** | 0.658 | 0.660 |
| 14–16k | 0.736 | **0.750** | 0.651 | 0.655 |
| 29–31k | 0.731 | **0.747** | 0.649 | – |
| **mean_disp_norm** 14–16k | 0.061 | **0.065** | 0.033 | 0.016 |
| 29–31k | 0.058 | **0.065** | 0.045 | – |
| **loss_volume** 29–31k | 0.0106 | **0.0108** | 0.0125 | – |

p0v1 tracks 4wok on every metric from 4k to 31k; both new-data runs sit below.

**But `amp_ratio` for p0v1 has NOT been measured** (`checkpoint_30.pt` exists and is the designed
read point), and **x1 bootstrapped on v2 data**, which demotes this line: v2 data does not prevent
learning, so the data difference is at most *sufficient*, never *necessary*.

Also: discount `coverage_frac` — old vs new data differ by the flip and roll, which change how
planes land in the padded stack, so coverage tracking 4wok is largely a *data-geometry* property.
`mean_disp_norm` is the real information (model behaviour, ~2×). Neither is the verdict: **P0
scored a failing 0.119 with `mean_disp_norm` (0.045) within ~25% of 4wok's (0.058).**

### 5c. What v1 vs v2 data ACTUALLY differ by — measured 2026-08-11

`sax/3d_recon/sax_frame_00.nii.gz`, 7 subjects, z-scored correlation, testing which op best aligns
v1 onto current:

| subject | D | parity | identity | zflip | roll−1 | **zflip+roll−1** | best |
|---|---:|---|---:|---:|---:|---:|---|
| Train_P140 | 11 | odd | 0.291 | 0.447 | 0.331 | **0.812** | zflip+roll−1 |
| Train_P010 | 9 | odd | 0.272 | 0.407 | 0.278 | **0.659** | zflip+roll−1 |
| Test_P022 | 11 | odd | 0.379 | 0.496 | 0.384 | **0.814** | zflip+roll−1 |
| Test_P041 | 12 | even | 0.284 | **0.765** | 0.336 | 0.496 | zflip |
| Val_P044 | 6 | even | 0.277 | **0.794** | 0.421 | 0.393 | zflip |
| Train_P061 | 12 | even | 0.284 | **0.773** | 0.352 | 0.444 | zflip |
| Train_P092 | 10 | even | 0.250 | **0.811** | 0.300 | 0.503 | zflip |

**Perfectly parity-gated.** Three differences, all now characterised:

1. **Apex-at-z0 flip — all subjects** (docs/58 §10a: 893/1343 were base-first, physically
   `np.flip(axis=2)`'d). Array flip, NOT an affine edit — `Orientationd(axcodes="LPS")` would
   silently undo an affine edit. Motivation: `respiratory.py`'s one-sided z shift pushed the heart
   in *opposite* anatomical directions for a 66/34 split of the cohort before standardisation.
2. **Odd-Z −1 roll — odd-Z only** (docs/56: 464/466 odd-Z rolled in the shipped k-space; 183/294 =
   62% of CMRx24).
3. **Receive-coil (B1−) shading — the docs/54 ESPIRiT input-domain bug.** `EspiritCalib` was fed an
   **image** where its API expects **k-space**; it does `sp.resize(ksp, calib_shape)` to grab the
   centre block, which is the ACS calibration region in k-space but just a 32×32 anatomy crop in an
   image. Maps degenerated to ~uniform magnitude, SENSE collapsed to a phase-aligned coil sum
   (anatomy sharp, shading never divided out). One-line fix
   (`batch_reconstruct_cmrxrecon2024.py:108` passes `ref_kspace_gpu`;
   `…_v1_ORIGINAL_espirit_image_domain.py:99` passed `sp.ifft(ref_kspace_gpu)`). All downstream
   recon code is byte-identical.

   **Verified here that the shading IS the entire residual.** Dividing out a purely low-frequency
   field (Gaussian σ=(12,12,1)) recovers it:

   | subject | geo-aligned | after ÷ smooth field | shading dyn. range |
   |---|---:|---:|---|
   | Train_P140 | 0.761 | **0.959** | 1.17–9.87 (8.5×) |
   | Train_P010 | 0.471 | **0.943** | 1.14–9.07 (7.9×) |
   | Test_P041 | 0.684 | **0.935** | 1.22–10.22 (8.4×) |
   | Train_P092 | 0.776 | **0.956** | 1.27–10.62 (8.4×) |

   **Radial profile (each volume normalised to its own in-body mean):**

   | subject | | r 0–0.2 (LV) | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.2 |
   |---|---|---:|---:|---:|---:|---:|
   | Train_P140 | v1 | **3.51** | 2.02 | 1.15 | 0.51 | **0.28** |
   | | v2 | 1.61 | 1.07 | 1.01 | 0.93 | 0.89 |
   | Test_P041 | v1 | **4.15** | 2.34 | 1.28 | 0.76 | **0.32** |
   | | v2 | 1.95 | 1.19 | 0.90 | 0.87 | 0.93 |

   **v1 peaks ~3.5–4× at the LV/centre and decays ~11–13× to the periphery; v2 is flat** (5/5
   subjects). This is the *opposite* of textbook surface-coil shading (bright rim, dark middle)
   because it is not a real B1− profile — it is the residual left where the bad map estimate was
   accurate (centre, corr 0.984) versus collapsed (periphery, corr 0.651).

   **v1 also HARD-ZEROED real tissue**: 0.26–3.00% of voxels are ≈0 in v1 and clearly present in
   v2; the reverse is **0.00% in all 5 subjects**. Matches docs/54's in-body map support
   0.937 (v1) → 0.992 (v2). Where the estimated maps are ~0, both numerator and denominator of the
   SENSE combine collapse and real tissue is annihilated.

**NOT a v1/v2 difference** (recurring confusion): the ±180° slice rotation is a *train-time*
augmentation (`data.augmentation`, `moderate` tier), not an on-disk property.

### 5d. Both P0-family arms use the OLD split, duplicates dropped by absence — symmetric

`random_8_1_1_prefixed.txt` is the **old 240/30/31 split with all 7 duplicate entries still
listed** (prefixed names), NOT the deduped split. The 7 dirs are absent from **both** roots
(`_archive/`), so they drop identically: 6 of 7 in `[train]` (240−6 = **234**), P193 in `[val]`
(30−1 = **29**). Both logs report exactly `234 subjects` / `29 subjects`. The docs/56-era
train/eval leak is **broken, not present** (each pair lost its train-side copy). Inert asymmetry:
v1 has an extra `CMRx24_Train_P002_2slice_excluded` dir no split names.

---

## 6. Cross-arm structural findings

### 6a. `pearson_median` ≈ 0.98 in EVERY arm — amplitude is broken, timing is not

Across all scored dumps, good and bad, the per-subject correlation between predicted and GT
LV-volume-vs-phase curves is **0.963–0.987**. Failing arms reproduce the *shape and timing* of the
cardiac cycle almost perfectly and get the *amplitude* wrong by ~7×.

**Any proposed mechanism must explain an amplitude failure that leaves timing intact.** The model
IS reading the reference slice and DOES know where in the cycle it is; it fails to scale
displacement to this patient's contraction. Matches the docs/24 flat-EF signature, not a
phase-confusion or conditioning-blindness signature.

### 6b. EF slope is not the verdict — two independent demonstrations

- The **BAD anchor scores ef_csv slope +0.625** while flat by `amp_ratio` (0.136).
- **x1 had the only positive in-training EF slope (+0.32, stable) and was recorded as the WORST
  `amp_ratio`** — the two metrics ranked arms in contradictory order. (Post-rescore x1 is the best
  arm, so this particular contradiction dissolves — but the anchor demonstration stands.)

### 6c. In-training EF slope went stably NEGATIVE — open, unexplained

From `trainer_viz.py` `[ef] epoch N: slope=…`, last 8 val epochs before cancellation:

| arm | last-8 in-training EF slope |
|---|---|
| a2 | −0.33 −0.31 −0.37 −0.35 −0.34 −0.37 −0.34 −0.36 |
| a4 | −0.35 −0.36 −0.35 −0.36 −0.36 −0.37 −0.36 −0.36 |
| a3 | −0.65 → −0.63 |
| a5 | −0.49 → −0.44 |
| a7 | −0.47 → −0.43 |
| c0 | −0.24 → −0.30 |
| a6 | −0.22 → −0.09 |
| a1 | −0.33 … −0.12 (erratic) |
| a8 | −0.53 … −0.01 (erratic) |
| **x1** | **+0.32 → +0.30 (only positive)** |

By 67k steps a2 and a4 vary by <0.02 across eight consecutive epochs — this is a *stable*
measurement of patients being ranked **backwards**, not noise. **But the offline `ef_csv` slopes
for the same arms are all POSITIVE (+0.205 … +0.506).** Different measurements (in-training
`trainer_viz` vs offline CorSeg-vs-`cardiac_phase.csv`); the sign disagreement is **unexplained**.
Do not build on either without reconciling them. Note x1 being the only positive in-training arm
now looks like a genuine early signal that was dismissed.

---

## 7. Mechanism analysis for the `Z_HALF_MM` × respiratory interaction

**All of this is HYPOTHESIS. The 2×2 numbers are measured; no causal claim here is.**

### 7a. What `Z_HALF_MM` does and does not do

`z_norm = z_mm / Z_HALF_MM`, one fixed ruler for every subject (`loss.py:478-484`: "one normalized
z-unit is always exactly `Z_HALF_MM` mm for every subject **by construction** — unlike the old"
per-subject `(D-1)/2*dz`, which docs/58 removed).

**It does NOT set the output FOV and it does NOT change geometry.** It cancels in the splat:
```
pz = pos_z·z_scale + (D−1)·0.5,   z_scale = Z_HALF_MM/dz
   = (z_mm/Z_HALF_MM)·(Z_HALF_MM/dz) + (D−1)/2 = z_mm/dz + (D−1)/2
```
It survives only as float32 rounding noise (`splat.py:61-63`, `-4.77e-07`), which is why the splat
carries an `EPS=1e-3` guard. The output volume is `(D,256,256)` on **the GT's own grid** — taken
from `V_gt.shape[1:]` with a hard shape check (`loss.py:163-181`). Points outside `[0,D−1]` are
**masked out** (`torch.where`, never multiply — `0.0*NaN = NaN`), not cropped from a bigger box.

Out-of-range z **raises**, it does not crop: *"z_norm … exceeds Z_HALF_MM half-span … raise
Z_HALF_MM, do not crop the stack."*

It changes exactly two things:
- **Input width**: fixed-12 spans ±0.733 at 90, ±1.0 at 66.
- **mm per unit of predicted Δz**: `Δz_mm = Δz_norm · Z_HALF_MM`.

### 7b. RETRACTED hypothesis — "66 gives better depth resolution"

Claimed +37% inter-slice separation in the depth embedding makes depths more discriminable.
**Refuted by reading `ZIndexEmbedder` (`aggregator.py:26-42`):** frequencies π, 2π, 4π **plus a raw
linear term**. Per-slice phase step on the highest frequency is 96° at 90 and 131° at 66 — both
well inside Nyquist, and the linear term alone separates all 12 slices monotonically at either
scale. Discriminability is not the difference.

### 7c. CURRENT hypothesis — z-vs-in-plane gradient balance

`Z_HALF_MM` enters the splat as `z_scale`, so it scales the **gradient reaching the z output
channel**, while in-plane is untouched:
```
∂L/∂Δz_norm = ∂L/∂pz · z_scale ,  z_scale = 7.5 (at 90) vs 5.5 (at 66)
∂L/∂Δx_norm = ∂L/∂px · (W−1)/2 = ×127.5   — independent of Z_HALF_MM
```
**At 90 the z channel receives 36% more gradient than at 66.** So 66 *damps z relative to in-plane*
by ~27%. In a SAX stack, LV volume change — what `amp_ratio` measures — is driven mostly by
**in-plane radial** contraction.

The breathing sim is a pure **z-displacement** task. That gives the interaction:

| | resp ON — z is needed | resp OFF — z has little to do |
|---|---|---|
| **90** | c0 0.142: z over-driven and needed | a3 0.152: z *still* over-driven, in-plane starved |
| **66** | a2 0.140: z damped but breathing needs it ⇒ hurts | **x1 0.526: z damped AND unneeded ⇒ capacity goes to in-plane** |

**Three independent corroborations:** the user's observation that x1's in-plane DVF is visibly
stronger; x1's *physical* z displacement is measurably **smaller** (0.158 vs 0.180 mm) while its
normalized `mean_disp_norm` is larger (+20%); and only the doubly-favourable cell fires.

**Sharp, cheap prediction (NOT YET RUN):** reproduce x1 at `Z_HALF_MM=90` by scaling the head's z
output (or its loss contribution) by ~0.73. **This matters because 66 CANNOT be used under
native-z** — pool max span ~170mm ⇒ `z_norm` = ±0.94 at 90, but ±1.29 at 66, which both aliases the
π/2π/4π sinusoids and trips the hard raise. The finding must be translated into a native-z-safe
form, and gradient rebalancing is that form.

### 7d. DVF panels are NOT autoscaled — the panel comparison is valid

`trainer_viz.py:804-808` uses hard-coded constants: `IN_PLANE_R = 15.0`, `THROUGH_R = 25.0` mm
half-ranges, with `IN_PLANE_MM = (256−1)/2·1.4 ≈ 178.5` and `THROUGH_MM = Z_HALF_MM`. Since
`Δz_mm = Δz_norm · Z_HALF_MM`, **the Δz row plots true physical mm in every arm** — so panels are
comparable across arms on all three axes. Panel titles also carry `|Δ|(norm) p50/p95/p99`, which
are raw normalized numbers and the cleanest thing to compare.

⚠️ In the **old** tree `THROUGH_MM` is hardcoded `(12-1)/2*12.0`, so it mislabels a native-z run.

---

## 8. Status of every enumerated factor

| factor | how covered | status |
|---|---|---|
| Pooled vs CMRx24 cohort | fresh-pooled anchor 0.136; CMRx24 control 0.142 | cleared — both fail |
| Gather loss | hub pins `gather_weight=0.0`; pooled anchor had it | cleared — fails with and without |
| Variable slice count / native-z | hub pins fixed12; pooled anchor was native-z | cleared — both fail |
| Dataset split | hub `cmrx24only.txt`; P0 `random_8_1_1_prefixed` | cleared — both fail |
| torch.compile | a1 | **NEVER SCORED** (ckpt_180 ≈42.3k on disk) |
| Epoch clock | a7 | cleared (COMPLETED 70k, exit 0:0) |
| Initialization / seed | a5 | **partial** — z-embedder tensors only |
| LR schedule / warmup | a8 | **NEVER SCORED** (ckpt_240 ≈56.4k on disk) |
| Training code as a whole | P0 | fails at 0.119 — **but not re-scored** |
| Freeze regime / DINOv2 stem | p0dino | **refuted, and harmful** |
| Data (flip + roll + shading) | p0v1 | **DEMOTED** — x1 bootstrapped on v2 data |
| **`Z_HALF_MM` × respiratory** | **a2 / a3 / x1 / c0** | **✅ THE LIVE FINDING — interaction, 0.14 → 0.526** |
| Env (torch 2.13 / monai 1.6) | docs/49 | largely cleared: preprocessing float32 **byte-identical** across monai 1.4→1.6, 31 subjects, all 28 (X,Y) spacings, all 7 Z-paths incl. the ones P0 uses. Residual: gradients never bit-compared across torch versions. |

### Still untested by any arm

- **The splat boundary rewrite.** Old `splat.py:50-62` used `z0f <= D-2`, so at init the entire top
  plane z=11 and the boundary rows were *dropped from the splat*; new `splat.py:48-117` keeps them
  with EPS/clamp. Changes what the loss sees.
- **Aug-off on the hub.** 4wok and P0 trained with affine/photometric aug **OFF**; every hub arm
  forces `tier=moderate`, and "moderate" was itself redefined (old: flip p=.5, ±12° rotate; new: no
  flip, ±180° rotate).

---

## 9. The heart-ROI L1 series (launched 2026-08-11)

**Motivation, independent of the x1 finding.** The objective is an unmasked full-volume L1 over
`(D,256,256)` including X/Y zero-padding. The heart is only **4.3–4.6%** of that volume — measured
directly from `heart_roi_canonical.nii.gz` (P140: 0.0434; P041: 0.0462), matching the
`metric_heartseg_frac` 0.045–0.065 logged by native-z runs. So ~95% of the gradient is spent on
tissue that does not move with the heart.

**Design.** `objective += heart_weight · mean_{roi}|V_canon − V_gt|`. **ADDITIVE, not a mask** —
the masked variant (`V_gt > 1e-3`, still commented out in `loss.py`) was removed because it let the
model over-predict wherever GT was zero (ghost blobs); keeping the full L1 primary preserves that
penalty.

**⚠️ SCALE — the two terms are means over DIFFERENT voxel counts.** Effective per-voxel weight is
`1 + heart_weight/heartseg_frac` with frac ≈ 0.05:

| w | heart voxel weight | heart's share of gradient |
|---|---|---|
| 0.05 | ~2× | ~50% |
| **0.10** | **~3×** | **~67%** |
| 0.25 | ~6× | ~83% |
| 0.50 | ~11× | ~91% (near-masked; watch for ghost blobs) |

**The ROI** is `heart_roi_canonical.nii.gz`: binary uint8, **dilated whole-heart** (union over 12
phases — includes RV/atria, so NOT an "LV loss"), loaded per subject and transposed `(X,Y,Z)→
(D,H,W)`. The separate `heart_seg_canonical.nii.gz` is `(256,256,D,12)` with labels 1/2/3 at ~2% and
is *not* what the loss uses. `get_data` has **no split gate**, so the ROI is present for train as
well as val (only the `psnr_3d_heartseg` metric is val-gated).

**NATIVE-Z ONLY, by construction.** `temp/fixed12_dataset.py:51-52` pads `phases` and `mask` to
D=12 but **not** `heart_roi`, so under the fixed12 hub `get_data` drops the ROI on ~28/29 subjects.
The loss raises rather than silently no-op'ing. Native-z runs log **0** such warnings.

**Implementation** — `/home/minsukc/vggt-arm-heart`, a copy of main differing by **exactly two
files** (`training/loss.py`, `training/config/default.yaml`); `vggt/` core byte-identical.
**All three arms use this same tree, including the w=0 control** — `heart_weight=0` was unit-tested
to give `loss_heart=0.0` and `objective == loss_volume`, so `heart_weight` is the only variable.

**Verification performed** (all fault-injected, not assumed):
- `loss_heart` alone produces gradient on `world_points`: |grad| sum 0.38, **756 nonzero elements**,
  all finite; adding the term **changes the total gradient** vs w=0 (max|diff| 5.8e-3).
- Empty-ROI edge case ⇒ **exactly 0.0**, no NaN from 0/0, grads still finite.
- The missing-ROI guard was **proven to fire**.
- ⚠️ **A defect was caught by this testing:** the first implementation used
  `if bool(n_roi > 0)` — a Python-level branch on a tensor, which forces a GPU sync + graph break
  **every step** under `cuda.compile_attention_blocks: True`. The codebase documents this hazard
  two functions above ("a Python-level decision here costs 4 graph breaks, measured"). Rewritten
  branchless (numerator is exactly 0 on an empty mask; `clamp(min=1)` guards the denominator) and
  proven numerically identical. Jobs were restarted after the fix.

**Runs** (`sbatch/heartl1_{common,w000,w010,w050}.sh`), 300 epochs × 235 = 70.5k steps, matched to
the exp66 hub total:

| job | arm | heart_weight |
|---|---|---|
| 56990551 | heartl1_w000 | 0.0 (control) |
| 56990552 | heartl1_w010 | 0.10 |
| 56990553 | heartl1_w050 | 0.5 |

Cohort `cmrx24only.txt` on **current v2 data** (`CMRxRecon2024/Cine_combined/…`, zero recon_v1
paths), `limit_train_batches=235 limit_val_batches=58`, everything else production `default.yaml`
(native-z, aug moderate + respiratory ON, aggft, gather 0.5, tv 0/diffusion 1000, lr 5e-5, seed 42).

⚠️ **NOT comparable to the exp66 arm table** (those are fixed12). Judge against its own w=0 control.
⚠️ Training on the ROI makes `metric_psnr_3d_heartseg` a **training target** — judge on `amp_ratio`
and motion PSNR.

---

## 10. Gotchas (each paid for in GPU time)

- **`PYTHONPATH` must be `temp:training:.`** — `training:.` killed all 10 fixed12 arms instantly
  with `ModuleNotFoundError: fixed12_dataset` (~3h lost); only P0 survived (no fixed12 override).
- **`ef_val_sweep=false` is MANDATORY for phase dumps** (§2b) — ~25 min lost.
- **Never score an arm through the wrong tree** (§2c) — this is what produced the false negative
  that nearly ended the campaign.
- **Match on global step, instantaneous vs instantaneous.** Comparing 4wok at step 165 (still in
  its transient) against P0's *running average* produced a false "P0 collapsed" alarm.
- **A refuted mechanism, recorded so it is not resurrected:** "breathing dominates the unmasked L1
  so cardiac motion (~5% of voxels) can never win gradient." **False as stated** — 4wok trained with
  breathing ON, unmasked L1, no gather, and learned contraction fine. (The *interaction* in §7c is
  a different and still-live claim.)
- **4wok HAS intermediate checkpoints** — `checkpoint_{50,100,150}.pt` (=50k/100k/150k) plus
  `checkpoint_last.pt` in `scratch/logs/217720691_*/ckpts/`.
- **A3/X1 (`respiratory.enable=false`) log `val/resp/slope_dz = None`** and have near-zero DVF — by
  design, but their resp metrics are non-comparable.
- **`heart_roi_canonical shape != expected` warnings are a FIXED12 ARTIFACT**, not stale ROIs. The
  warning text guesses "likely a stale pre-native-z ROI" and that guess is wrong for this cohort.
- **A8's seed side effect:** `set_seeds(seed_value, max_epochs, rank)` ⇒ `max_epochs=851` gives
  effective seed 42×851 = 35,742 vs hub 12,600. No integer `max_epochs` lands on the hub seed.
- **The P0-family old tree writes no `metrics.jsonl`** — parse `Train/Val Epoch` lines from the
  slurm log. In-training EF is `trainer_viz.py: [ef] epoch N: slope=…`.
- **`--signal=B:USR1@120` was removed from the exp66 arms** so a walltime hit kills rather than
  requeuing and burning credits. `--requeue` kept for node failures.

---

## 11. Open questions

1. **IS THE PREMISE TRUE? — STILL NEVER RUN, still highest value (~40 min).** Everything assumes
   4wok learned contraction *by this evaluator*.
   ```bash
   micromamba run -n svr python tools/e0_dump_phase_sweep.py \
     --tree /home/minsukc/vggt-oldcode-p0 --config mri_volume_diffusion \
     --ckpt scratch/logs/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_50.pt \
     --out result/e0_dumps/anchor_4wok_50k --limit-val-batches 29 --master-port 29680 \
     --override split_file=training/splits/random_8_1_1_prefixed.txt --override ef_val_sweep=false
   micromamba run -n svr python tools/e0_score_volumes.py \
     --dump result/e0_dumps/anchor_4wok_50k --dz 12 \
     --gt-seg-dir result/e0_dumps/_gt_segs_cmrx24val
   ```
   ~0.74 ⇒ premise real. ~0.14 ⇒ the docs/33 "EF slope 0.77" came from a different measurement and
   part of this campaign chased a phantom. ⚠️ 4wok trained on **v1** data but this decodes against
   **current**; also try `data_root=data/CMRxRecon2024_recon_v1_espirit_imagedomain`.
2. **Re-score c0, P0, a8, a1** with `ef_val_sweep=false` and recorded provenance. a3 reproducing
   cleanly suggests main-tree arms are fine, but P0's 0.119 is load-bearing and unverified.
3. **Does the §7c gradient-balance hypothesis hold?** Test: `Z_HALF_MM=90` + z-output scaled ~0.73,
   respiratory off. If it reproduces x1, the finding becomes native-z-portable.
4. **Does x1 replicate on a second seed?** One arm, one seed, one checkpoint so far.
5. **Does x1 survive respiratory being ON?** Breathing-off is not shippable — the research goal is
   free-breathing. If the interaction requires resp off, the finding is diagnostic, not a fix.
6. **Does p0v1 bootstrap?** `checkpoint_30.pt` ready. Demoted but still informative.
7. **Reconcile the EF-slope sign disagreement** (§6c): in-training stably negative, offline
   `ef_csv` positive, same arms.
8. **A40 vs L40S** for both wave-2 arms.

## 12. Next steps

1. **Premise check** (§11.1) — 40 min, independent, could recontextualise everything.
2. **Re-score P0 and c0** — the two load-bearing numbers still on unverified provenance.
3. **Fix `e0_dump_phase_sweep.py`** — write a `_manifest.json` (tree, ckpt, overrides, git rev) and
   force/guard `ef_val_sweep=false`. Without this the next agent repeats today's failure.
4. **x1 follow-up, in priority order:** (a) native-z-safe form via z-gradient scaling at 90;
   (b) second seed; (c) respiratory ON at 66 for longer, to see whether a2 is slow rather than dead.
5. **Read the heart-L1 series** at 15–30k (`amp_ratio`, and the `V_canon-V_gt` panel for ghost
   blobs at w=0.5).
6. **Whatever wins:** confirm on fresh **pooled1337 native-z gather=0.5**. Fixed12/CMRx24 is a
   diagnostic bridge, not the destination.

---

## 13. Job accounting

**Cancelled 2026-08-11** after the (then-believed) negative — all at epoch 286–295 of 300 except as
noted. **Checkpoints survive cancellation; every arm retains `checkpoint_240` ≈56.4k steps.**

| job | arm | at kill |
|---|---|---|
| 56779008 | c0_hub_repeat | ep 286, ckpt_240 |
| 56779010 | a1_nocompile | ep 232 (slowest — no compile), ckpt_180 |
| 56779011 | a2_oldz | ep 294, ckpt_240 |
| 56779013 | a4_legacyresp | ep 295, ckpt_240 |
| 56779014 | a5_z8400init | ep 290, ckpt_240 |
| 56779015 | a6_withreplacement | ep 291, ckpt_240 |
| 56779017 | a8_4wok_lrtraj | ep 284, ckpt_240 |

**Completed naturally:** 56779016 a7_epochclock (70k, exit 0:0); 56779012 a3_noresp and 56779018
x1_oldz_noresp both ran to the 300-epoch cap.

**Running at time of writing:** 56779009 P0_faithful4wok · 56854632 p0v1_olddata · 56854633
p0dino_newdata · 56990551/2/3 heartl1_w000/w010/w050 · 56607435 pooled1337 warm-start deliverable.

**Killed earlier in the campaign:** 56722507, 56723520, 56724084 (superseded debug arms);
56616177 / wandb `jjfuofy9` (pooled nogather).

**Dumps on disk:** `result/e0_dumps/rescore_x1_240b` (0.526), `rescore_a3_240b` (0.152),
`rescore_a2_240c` (0.140) — all verified 58 files/phase with single correct t-labels.
`rescore_x1_240` is the INVALID `ef_val_sweep` dump; kept as the worked example of the failure mode.
