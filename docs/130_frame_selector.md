# 130 — Per-slice frame selector in front of frozen diff1000

> **TL;DR & takeaway** (2026-10-02)
> - **Idea.** diff1000 receives one frame per slice, drawn at a random cardiac phase. A small learned **frame selector** picks, for every slice, the frame whose phase best matches the target phase. It reads only diff1000's own frozen DINO tokens. diff1000 is not retrained.
> - **Ceiling.** A perfect (oracle) selector is worth +1.7 / +1.8 dB PSNR and cuts EF MAE from 7.5 / 7.9 to 4.2 pp (af12 / hrv12 test, n=180 each).
> - **What made it learn.** The plain selector stayed at chance: DINO tokens mostly encode static anatomy, and the heart is a small part of the slice. Four fixes got it learning:
>   - per-plane centering;
>   - per-plane RMS scaling;
>   - attention pooling;
>   - a heart-ROI attention loss (97% of the attention mass on the heart).
> - **Longer training.** 20k steps instead of 2k took val phase error from 1.81 to 0.91 (random = 3.0).
> - **Test, final selector** (B+ROI, 20k run, best step 18k; af12 / hrv12):
>   - PSNR +1.05 / +1.28 dB (25.55 / 25.87), i.e. 63% / 70% of the oracle gain;
>   - EF MAE 4.38 / 4.05 pp, statistically indistinguishable from the oracle (4.11 / 4.19 on the same subjects, p = 0.16 / 0.70);
>   - phase error 1.54 / 1.26;
>   - every selector arm vs diff1000: p ≤ 6e-8 (paired Wilcoxon).
> - **Cost.** About +0.5 s per subject (measured: 7.9 vs 7.4 s) when the DINO tokens are shared with the reconstruction.
> - **Shipped config** (branch `exp/frame-selector`, package `frame_selector/`): variant B + centering + slice-RMS norm + attention pooling, plus the optional `--roi-weight`. Variant A and the failed knobs were removed (§4 records them).
> - **Main open problem.** The test phase error is 0.35–0.63 worse than val. It is worst on CMRx25, ACDC and M&Ms.

## 1. Problem and information contract

- **The regime.** The eval regime (docs/110 rhythm arms `af12`/`hrv12`) gives F=12 frames per slice plane. Today the companion slot of plane z is the bundle's frozen **scatter** draw, one frame at an unknown phase. Slot 0 is the target-phase **reference** slice at the mid-ventricular plane (docs/25), and it sets the phase to reconstruct.
- **What the selector does.** For every queried frame f (reference = reference plane's frame f) and every other plane z, it returns `sel[f][z]`, the frame of plane z it thinks is at the reference's phase. `run_vggt.py --input sel` then puts that frame in plane z's slot. Nothing else changes: same frozen model, slot order, breathing and z.
- **Information contract (docs/04, docs/65).** At inference the selector sees only image content (the reference slice and the candidate frames) and z. It never sees t or r.
  - True cardiac positions are used only as **training labels**: gated cine phases.
  - The oracle tables and the offline phase error read the manifests' true positions. That is evaluation only.
- **Not used:** within-plane acquisition order. Candidate frames are scored independently, so order carries no information. Whether order may ever be used as a self-gating cue is the open question in docs/129 §5.

## 2. Ceiling: oracle selection

`python -m frame_selector.oracle` picks, per (f, z), the frame whose true position (`manifest.rhythm.pos_per_plane`) is circularly closest to the reference's true position (`rhythm.ref_pos`). This is the best any selector can do with these frames. The companion phase error is not 0, because a plane's 12 frames do not hit every phase exactly: 0.43 (af12) / 0.27 (hrv12).

| test (n=180) | PSNR (dB) | SSIM | EF MAE (pp) | companion phase err |
|---|---|---|---|---|
| af12 diff1000 (scatter) | 24.50 | 0.715 | 7.49 | ~3.0 (random) |
| af12 oracle | **26.17** (+1.67) | 0.795 | **4.20** | 0.43 |
| hrv12 diff1000 (scatter) | 24.60 | 0.719 | 7.88 | ~3.0 |
| hrv12 oracle | **26.42** (+1.82) | 0.806 | **4.19** | 0.27 |

- **Metric definitions:**
  - PSNR = `breath_psnr_unit_peak`, the subject mean pooled over the 5 sources.
  - SSIM = `breath_ssim`.
  - EF MAE = |EF_breath − EF_gt| from nnU-Net 3d_fullres segmentations (`evaluation/src/score/ef_dice.py`).
- **Significance:** oracle vs diff1000 on every metric, p ≤ 1e-10 (paired Wilcoxon).
- **Sources:** `evaluation/metric_results/test/<cohort>/{,ef/}vggt_final518_diff1000_ep300{,_sel_oracle}.json`. Tables were recomputed for this doc from per-subject rows.

## 3. Method (the shipped selector)

### 3.1 Features
- **Token source.** `FrozenPatchEmbed` reuses diff1000's own `aggregator.patch_embed` (DINOv2 ViT-L/14 with registers). It is frozen in aggft, so these are exactly the tokens the reconstruction computes.
- **Tokens per frame.** Each frame becomes a 518×518 model image (`_resize_to_model_res`), then 37×37 = 1369 tokens × 1024 channels.
- **Sharing with the reconstruction.** At eval, `run_vggt --selector-run` embeds every (plane, frame) once (`embed_all`, D×F images). The selector picks from those tokens. The reconstruction gathers the chosen slots' tokens from the same cache through the new `patch_tokens` input of `Aggregator.forward` (`batch["patch_tokens"]`), so DINO runs once per frame, not twice. With `patch_tokens=None` (every other caller) the aggregator is unchanged.

### 3.2 Network (`frame_selector/model.py::FrameSelector`, 1.96 M params)
For one plane's F candidate frames vs one reference:
1. **Center.** Subtract the plane's mean token over its own frames. Candidates use their own plane; the reference uses the reference plane's F frames. Only the phase-varying part is left.
2. **Scale.** Divide by one scalar per plane: the RMS over that plane's frames × tokens × channels. Heart tokens, which change a lot between frames, stay large relative to static background. A per-token LayerNorm would equalise them.
3. **Project and pool.** Linear 1024→256, then 2×2 average pooling of the token grid (37→19, 361 tokens).
4. **Add z.** A sinusoidal embedding of [candidate z_norm, candidate − reference z_norm] (physical z, docs/58).
5. **Cross-attend.** 2 cross-attention blocks: the candidate tokens attend to the reference tokens.
6. **Pool and score.** Learned attention pooling over tokens, then an MLP gives one score per frame. Softmax over the plane's frames.
- **Order-free.** Every frame is scored independently, so frame order and F don't matter.
- **Inference cost.** Candidates are projected once and reused for all F references (`select_from_tokens`).

### 3.3 Training data (`frame_selector/data.py`)
- **Episodes.** Generated on the GPU from pooled_curated_v2 **train** (628 subjects). Model selection uses **val** (90), with fixed seeded episodes.
- **One episode = one subject:**
  - the reference plane (the dataset's slot-0 plane, z_mid of the anatomy bbox) plus 3 random other planes (4 at val);
  - each plane shows its 12 gated cine phases as 12 frames, in cyclic order from a random start;
  - the reference is one random frame of the reference plane.
- **Breathing matches the eval bundles (`build_af_bundle.simulate`):**
  - one depth and one tilted direction per subject, sampled from diff1000's own respiratory config;
  - a per-plane start phase;
  - breathing phase advancing DT/T_BREATH per frame;
  - frames resliced by the trainer's own `extract_slices_with_respiratory_vec`.
- **Matches the eval pixels.** `tools/frame_selector_check_data.py` re-renders test bundles from their manifests and compares against the model images `run_vggt` builds from the stored `breath/stack_t*` frames.
- **Gap to eval.** Training uses gated cine at integer phases. The eval rhythm arms (af12/hrv12) present frames at irregular, fractional positions (AF / HRV rhythms). That is one candidate cause of the val→test gap (§6). It is untested.

### 3.4 Loss
- **Selection loss.** Soft circular-Gaussian labels on phase distance: q_j ∝ exp(−d_j² / 2σ²), σ = 0.5 phases. Cross-entropy against the softmax over the plane's frames.
- **Heart-ROI attention loss** (`--roi-weight`, B+; training only):
  - the heart ROI (`heart_roi_canonical`) is resliced with each frame's breathing;
  - it is pooled to the token grid, and a token counts as heart if > 30% of it is inside the ROI;
  - the loss is −log(attention-pooling mass on heart tokens).
  - It pushes the pooling onto the heart.
  - Logged val `attn_heart_mass` reaches 0.97, vs 0.90–0.95 in the 2k runs.
- **Optimizer.** AdamW, lr 3e-4, weight decay 0.05, OneCycle, grad clip 1.0, 4 episodes per step, one GPU.
- **Wall time** (from `log.jsonl` `sec`): 2k steps ≈ 70 min; 20k steps ≈ 11.8 h. The GPU model isn't logged; the training sbatch targets spgpu2 (L40S).

## 4. Everything tried (chronological, 2026-10-01 → 10-02)

Val phase error = mean circular distance (in phases) of the argmax frame on the 90 fixed val episodes. A random frame scores 3.0.

### 4.1 First design: stuck at chance
- **The first selector** (commit `9f49210`, then `tools/framesel_*`) had two variants:
  - **A** = mean-pool candidate tokens + mean-pool reference tokens + z-embedding → concat → MLP;
  - **B** = candidate tokens cross-attend to reference tokens → mean-pool → MLP.
  - Both used a **per-token LayerNorm** before the projection, with no centering.
- **Result: chance level.** Short probes stayed at ~3.0 val error and the loss stayed at the uniform value (~2.48 = ln 12).
  - `scratch/framesel/learncheck.log`: 3.05 at step 300.
  - `chk_plain.log`: 3.23 at step 300.
- **Diagnosis** (from the run notes, not re-measured here):
  - DINO tokens mostly describe static anatomy, which is the same in every frame of a plane;
  - the heart covers only a small part of the slice (~6% of tokens);
  - mean pooling plus per-token LayerNorm buried the phase signal under background.

### 4.2 Fixes (300-step probes, then 2k-step runs)
- **Probes** (logs `scratch/framesel/chk_*.log`; the logs don't record their flags, so the mapping comes from the file names):

| probe | step 300 val err |
|---|---|
| `chk_plain` (original design) | 3.23 |
| `chk_center` (+ per-plane centering) | 2.81 |
| `chk_new` (+ centering, slice-RMS norm, attention pooling) | 2.98 |
| `chk_onesub` (single-subject learnability test: train and val on one subject) | 2.43 |

- **The probes alone were inconclusive.** 300 steps was too short. The single-subject run learned, which showed the signal is learnable.
- **The 2k-step runs settled it** (commit `d94570d`; all use centering + slice-RMS + attention pooling):

| run | val err @ 500 / 1000 / 1500 / 2000 | best | test err af12 / hrv12 |
|---|---|---|---|
| A (attn-pool → concat → MLP) | 2.96 / 2.53 / 2.23 / 2.11 | 2.11 | 2.27 / 2.17 |
| **B** (cross-attn → attn-pool → MLP) | 2.67 / 2.24 / 2.12 / 2.02 | 2.02 | 2.19 / 2.12 |
| **B + heart ROI** (`--roi-weight 1`) | 2.11 / 2.11 / 1.81 / 1.87 | 1.81 | 2.14 / 2.03 |

- **A was dropped.** It was worse than B on val and on all 10 test cohorts. Its only advantage is a ~2.5× cheaper head (78 vs 200 ms per subject), which doesn't matter next to the ~1.2 s token pass.
- **The ROI loss learns faster:** 2.11 vs 2.67 at step 500. Its final test error is only slightly better at 2k steps.

### 4.3 Longer training: B + heart ROI, 20k steps (`scratch/framesel/runs/B_roi_20k`)

| step | 1k | 2k | 3k | 4k | 5k | 7k | 10k | 14k | 15k | **18k** | 20k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val err | 2.00 | 1.94 | 1.71 | 1.60 | 1.44 | 1.28 | 1.15 | 1.03 | 0.98 | **0.91** | 0.99 |
| ≤ 1 phase | 47% | 52% | 57% | 61% | 66% | 69% | 73% | 77% | 78% | **80%** | 78% |
| heart mass | 0.91 | 0.94 | 0.94 | 0.96 | 0.95 | 0.95 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 |

- `best.pt` = step 18k. Every 1k step is saved as `stepNNNNNN.pt` (~8 MB each).
- The curve was still improving until ~15k. The 2k runs were badly under-trained.

### 4.4 Ideas recorded but not built
See docs/129:
- a set-aware selector (ISAB over a plane's frames, shuffled order, appearance-distance labels);
- multi-frame priority compositing;
- a low-res read cascade for F in the hundreds.

## 5. Test results (af12 / hrv12, n=180 each, pooled over 5 sources)

| arm | test phase err | PSNR (dB) | Δ vs diff1000 | SSIM | EF MAE (pp) |
|---|---|---|---|---|---|
| diff1000 (scatter) | ~3.0 / ~3.0 | 24.50 / 24.60 | — | 0.715 / 0.719 | 7.49 / 7.88 |
| B (2k) | 2.19 / 2.12 | 25.13 / 25.21 | +0.63 / +0.62 | 0.744 / 0.750 | 4.77 / 4.61 |
| B + ROI (2k) | 2.14 / 2.03 | 25.17 / 25.32 | +0.67 / +0.72 | 0.746 / 0.755 | 4.94 / 4.55 |
| B + ROI, 20k run @ step 5k | 1.88 / 1.64 | 25.33 / 25.58 | +0.83 / +0.99 | 0.753 / 0.766 | 4.79 / 4.52 |
| **B + ROI, 20k run @ step 18k (final)** | **1.54 / 1.26** | **25.55 / 25.87** | **+1.05 / +1.28** | **0.765 / 0.779** | **4.38 / 4.05** |
| oracle | 0.43 / 0.27 | 26.17 / 26.42 | +1.67 / +1.82 | 0.795 / 0.806 | 4.20 / 4.19 |

- **Significance.** Every selector arm beats diff1000 on PSNR, SSIM and EF MAE: paired Wilcoxon p ≤ 6e-8 on every row. For PSNR/SSIM p ≤ 3e-21.
- **EF n.** On af12, the learned-selector EF rows have n=179: one subject's EF is missing in those arms.
- **EF.**
  - The 2k and step-5k selectors cut EF MAE by 34–43%. The final one cuts it by 42% (af12) and 49% (hrv12).
  - Final vs oracle, paired on the same subjects: af12 4.38 vs 4.11 (n=179, p=0.16); hrv12 4.05 vs 4.19 (n=180, p=0.70). The two are not distinguishable, so EF is at the selection ceiling.
  - On hrv12 every selector is below the classical baselines (docs/123: Fetal 5.98, CiNeVol 5.58).
- **PSNR** tracks the selector's phase error (2k → 5k → 18k: phase error 2.1 → 1.8 → 1.4 averaged over arms; ΔPSNR +0.66 → +0.91 → +1.16). The final checkpoint recovers 63% / 70% of the oracle gain. PSNR, not EF, is where headroom is left (+0.62 / +0.55 dB).
- **Per-source test phase error, best checkpoint** (`scratch/framesel/sel_tables/B_roi_20k/summary.json`):

| | CMRx23 | CMRx24 | CMRx25 | ACDC | M&Ms |
|---|---|---|---|---|---|
| af12 | 1.19 | 1.13 | 1.68 | 1.72 | 1.83 |
| hrv12 | 0.91 | 0.80 | 1.49 | 1.52 | 1.48 |

- **Timing:**
  - Standalone `frame_selector.predict` per subject: token pass ~1.2 s (all D×12 frames) + head ~0.2 s.
  - Inside `run_vggt --selector-run`, the token pass replaces the reconstruction's own DINO pass. The measured net extra cost is 0.4–0.5 s per subject (§7.1 item 8; RTX 6000 Ada). The cluster run notes said ~0.2 s, which was not re-measured here.
- **Result sources:**
  - `evaluation/metric_results/test/<cohort>/{,ef/}vggt_final518_diff1000_ep300_sel_{oracle,B,B_roi,B_roi_s5k,B_roi_20k}.json` (B_roi_20k from `origin/exp/frame-select` `a0f30d0`);
  - the selection tables (`sel_tables/<run>/summary.json`).
  - The selector evals ran from tables (`--sel-dir`), produced by `frame_selector.predict` (then `tools/framesel_predict.py`).

## 6. Open points
1. **Val→test gap.** Best checkpoint: val 0.91 vs test 1.54 / 1.26. It is worst on CMRx25, ACDC and M&Ms (multi-vendor, pathology), which is also where further gains would come from. Candidate causes, none tested:
   - training on integer gated phases vs the arms' fractional AF/HRV positions (§3.3);
   - val episodes use 4 planes from the same cohort mix as train, while test subjects are new.
2. **Next steps.** Run the ladder in docs/129 §6: the training-free K-pass composite probe, then the set-aware selector.

## 7. Port and verification (2026-10-02)

Ported from `origin/exp/frame-select` (`5a1274e`) onto the refactored main as branch `exp/frame-selector`.

| file | from | change |
|---|---|---|
| `frame_selector/model.py` | `tools/framesel_model.py` | B only. Centering, slice-RMS and attention pooling are hardcoded. A, `norm=ln` and mean-pool are removed. Parameter names are unchanged, so old checkpoints load with `strict=True`. |
| `frame_selector/data.py` | `tools/framesel_data.py` | absolute imports; `build_af_bundle` constants read from `evaluation/src/engine` |
| `frame_selector/train.py` | `tools/framesel_train.py` | `--variant/--center/--norm/--attn-pool/--one-subject` removed. Defaults: 20k steps, val every 1k. |
| `frame_selector/predict.py` | `tools/framesel_predict.py` | `load_selector` refuses a checkpoint whose recorded args name a removed variant |
| `frame_selector/oracle.py` | `tools/framesel_oracle_tables.py` | imports only |
| `tools/frame_selector_{compare,check_data}.py` | `tools/framesel_*` | one-off analysis |
| `sbatch/frame_selector_{train,eval}.sh` | `sbatch/framesel_{train,ceiling}.sh` | no per-user paths; train defaults to B+ROI 20k |
| `vggt/models/{aggregator,vggt}.py` | same | optional `patch_tokens` (None = unchanged) |
| `evaluation/src/engine/run_vggt.py` | same | `--input sel`, `--sel-dir`, `--selector-run`. The sel-only metadata/timing keys are written only for `--input sel`, so default output is unchanged. |

**Repro:**
```bash
python -m frame_selector.train --out scratch/framesel/runs/<name> --roi-weight 1.0          # ~12 h, 1 GPU
python -m frame_selector.predict --run scratch/framesel/runs/<name> --cohorts cmrx2024_af12 ... \
    --out scratch/framesel/sel_tables/<name>
python -m frame_selector.oracle --cohorts cmrx2024_af12 ... --out scratch/framesel/sel_tables/oracle
SEL=<name> sbatch sbatch/frame_selector_eval.sh           # recon + image metrics + EF, 10 tasks
python evaluation/src/engine/run_vggt.py --dataset <cohort> --ckpt <diff1000> --split test \
    --model-name final518_diff1000_ep300_sel_<name> --input sel --selector-run scratch/framesel/runs/<name>
```

**Locations (GPFS, via `scratch/`):**
- selector runs: `scratch/framesel/runs/{A,B,B_roi,B_roi_20k}/`, holding `best.pt`, `last.pt`, `args.json`, `log.jsonl`, plus `step*.pt` for the 20k run. Checkpoints are `{model, args, step}`, head only, ~8 MB.
- frozen model: `scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt`.
- tables: `scratch/framesel/sel_tables/<run>/`.
- recons: `scratch/eval/<cohort>/out/<subject>/vggt_final518_diff1000_ep300_sel_<run>/`.

### 7.1 Verification

All on caesar (RTX 6000 Ada), one subject or cohort, `CMRx24_Test_P001` / `cmrx2024_af12`.

1. **Tests:** `python -m pytest tests/` gives 335 passed. That includes `tests/test_frame_selector.py`, which covers shapes, `sel[f][ref]=f`, the oracle on a toy manifest, and the checkpoint guard.
2. **Old checkpoints:** `load_selector` on `runs/{B,B_roi,B_roi_20k}/best.pt` loads with `strict=True`. `runs/A` is refused with a clear error.
3. **Oracle:** `frame_selector.oracle` gives 59/59 tables (cmrx2024_af12, acdc_hrv12) byte-equal in `sel` and `err` to `sel_tables/oracle`.
4. **Predict:**
   - New `frame_selector.predict` vs the **original** `tools/framesel_predict.py` (`5a1274e`), on the same GPU, `runs/B_roi`, 38 subjects: **0 / 4,668 picks differ**.
   - Against the cluster's L40S tables, 267 / 4,668 (5.7%) differ, with the identical cohort phase error (1.96). These are near-ties flipping across GPUs, not the port.
5. **`run_vggt` default unchanged:**
   - Main (`f892c86`) vs this branch, `--input scatter`, run under `torch.use_deterministic_algorithms`: all 12 phase volumes bitwise identical.
   - Without deterministic mode they differ by ≤ 3.3e-7, the documented splat `scatter_add_` order noise.
   - The output JSONs have identical key sets; only arm name and timing differ.
6. **`--selector-run` (shared tokens) vs `--sel-dir` (model's own DINO) with the identical selection:**
   - Not bitwise: PSNR between the two recons 56.4 dB, mean |d| 6e-5, |d| > 0.01 in 0.02% of voxels.
   - Cause not measured. Likely the selector's chunked bf16 patch-embed pass vs the model's own pass.
   - This is the original branch's behaviour. Every reported result used tables (`--sel-dir`).
7. **Selection is near-tie sensitive:** the same live selector picked 6 / 120 frames differently under deterministic vs default kernels.
8. **Timing (live, deterministic mode):** total 13.4 s vs 13.0 s with a table; the selector pass (embed all + select) is 1.75 s inside the total. In default mode: 7.9 s live vs 7.4 s for plain scatter.
