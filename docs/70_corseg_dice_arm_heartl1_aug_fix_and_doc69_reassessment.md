# 70 — CorSeg-Dice arm, heart-L1 ROI aug-misalignment fix, and the doc-69 reassessment

> **TL;DR & takeaway**
>
> Three things happened 2026-08-11 (same session, after doc 69's rewrite):
>
> 1. **Doc 69's `Z_HALF_MM`×respiratory interaction is DEMOTED as the primary cause; the
>    v1→v2 DATA change is promoted** (user-verified from wandb: P0 does not contract, p0v1 —
>    identical code, v1 data — does). Old code is index-z (always fills ±1 ⇒ 66-equivalent,
>    z-grad gain 5.5 regardless of dz) with resp ON, so the interaction story *predicts 4wok
>    should have failed too* — it didn't. A 2-agent debate converged on: contraction AMPLITUDE
>    is a marginal second-order signal with several interchangeable "easing" levers {v1
>    shading, 8mm smoothing, resp-off@66}; v1's ESPIRiT-shading bug acted as an implicit
>    heart-weighted loss (~1.2–2× after normalization, measured on 4 subjects). Also found:
>    **scoring-time confound** — x1 was flat for 50k steps then phase-transitioned at ~51–54k
>    (0/42 positive EF-slope epochs before, 73/75 after), while P0 was amp-scored only at
>    25k/35k and the good anchors' budgets range 33k–192k.
> 2. **The running heart-L1 arms were training against a MISALIGNED ROI** — gpu_aug warped
>    only `phases`+`content_mask`, never `heart_roi_canonical`, while tier=moderate rotates
>    ±180°. Fixed in `vggt-arm-heart` (fault-injection verified); w010/w050 restarted as jobs
>    **57022432/57022433**; w000 (56990551) unaffected and kept as the shared control.
> 3. **A CorSeg soft-Dice arm was built and launched** (job **57023101**, tree
>    `vggt-arm-corseg`, `corseg_weight=0.002` ≈ 2× measured grad-parity): a frozen
>    CorSeg-CineSAX (MedNeXt-L, 2D) segments every slice of V_canon differentiably; soft Dice
>    vs `heart_seg_canonical[..., t_target]` rewards endocardial BOUNDARY placement — the
>    thing `amp_ratio` measures — where heart-L1 rewards ROI intensity. ⚠️ amp_ratio is
>    CorSeg-derived ⇒ decisive checkpoints of this arm MUST be re-scored with nnU-Net.

**Date:** 2026-08-11. **Predecessor:** doc 69 (kept as the campaign record; this doc supersedes
its causal ranking, not its measurements).

---

## 1. Doc-69 reassessment: data promoted, interaction demoted

### 1a. The facts that forced it

- **User-verified (wandb DVF/volume panels, most recent viz; treat as ground truth):** P0
  (old code, v2 data, 12 mm, resp ON, ~55–60k steps) does **not** learn contraction; p0v1
  (identical tree/launch, v1 data) **does**, visibly by ≤40k. Only controlled single-variable
  GOOD↔BAD flip in the campaign.
- **Old code is index-z** (`z_val=(z_i/(D−1))·2−1`, old `mri_dataset.py:380`): the stack always
  fills ±1 and the splat z-grad gain is `(D−1)/2 = 5.5` for every dz ⇒ 4wok, P0, p0v1 and the
  8mm replay are ALL "66-equivalent" with resp ON. The doc-69 2×2 assigns that cell 0.140 (a2)
  ⇒ it predicts 4wok/p0v1 fail. They don't. The interaction cannot be the primary mechanism.
- **The 8mm replay (213853800/xn9gkxr0) ran on CURRENT v2 data** (verified from its log:
  `scratch/data/CMRxRecon2024/Cine_combined`, spacing 8.0) — so v2+resp-ON is learnable under
  the 8mm-easier geometry (33k steps!). Data is sufficient-not-necessary, like every other lever.

### 1b. The debate synthesis (2 adversarial agents, both verified claims on disk)

**Converged mechanism:** per-patient contraction amplitude is a *residual on a residual* —
second-order, fighting the 95%-static unmasked-L1 background (which is why `pearson≈0.98`
everywhere while amp fails). Multiple levers add "easing dose": v1 shading, 8mm through-plane
smoothing, resp-off (resp-ON arms spend nearly all displacement on z: `mean_disp_norm`
0.04–0.05 vs 0.001–0.003 resp-off), and the z-grad balance (66 vs 90 — real but the smallest
lever). The v1↔v2 flip identifies **data as the largest single dose**, not a distinct mechanism.

**How v1 data helps — the contrast-upweighting mechanism (measured, medium confidence).**
The L1's voxel gradient is `sign(V−G)` (intensity-free), but the gradient reaching *position*
goes through the splat: `∂V/∂pos_i ∝ (I_i − V_local)·∂w_i/∂pos` — it scales with the moved
pixel's **contrast against its surroundings**. v1's ESPIRiT B1− bug multiplies the heart
~3.5–4× and dims the periphery ~3× ⇒ the endocardial boundary's placement gradient (the ONLY
channel that teaches amplitude) grows while competitors shrink. Measured on 4 subjects after
percentile normalization: heart-vs-bg contrast ~2.2–2.4× (v1/v2 ~2×), per-voxel |ED−ES|
1.2–1.7×, heart share of total temporal L1 ~1.2–1.4×. The bridge from ~2× to the 6× amp_ratio
flip is the open link — exactly what the heart-L1/corseg arms test in shippable form.

**Scoring-time confound (new, from x1's `metrics.jsonl`):** x1's in-training EF slope was
statistically identical to the dead arms through 50k (0/42 positive epochs, mean −0.54), then
transitioned sharply at 50.5→54k (−0.44→−0.03→+0.30) and stayed positive (73/75). Scored at
its 42k checkpoint x1 would have been called BAD. P0's failing 0.119 was scored at 25k/35k;
budgets across the matrix: 8mm replay 33k, hub arms 70.5k, 4wok 192k. ⇒ "P0 never escapes" is
unproven; "v1 escapes ≥20–30k steps earlier than v2 in identical code" is what's established.

**Standing cheap TODOs (unchanged from doc 69 §11, re-prioritized):** score p0v1
`checkpoint_30/_40`, 4wok `checkpoint_50` (premise), P0 `checkpoint_55/60` — each ~40 min,
commands in doc 69 §11.1. The "new code + v1 data" arm does NOT discriminate the hypotheses
(both predict rescue) — run only if the picture stays ambiguous.

### 1c. RESULTS of the cheap scorings (2026-08-11 evening — this section supersedes the TODO)

Run via `tools/e0_dump_phase_sweep.py` through `/home/minsukc/vggt-oldcode-p0`, `--dz 12`,
`split_file=training/splits/random_8_1_1_prefixed.txt`; p0v1 decoded against its own
training data (`data_root=/home/minsukc/vggt/data/CMRxRecon2024_recon_v1_espirit_imagedomain`,
ABSOLUTE path — the p0 tree has no `data/` symlink for v1, a relative path silently yields
0 subjects → ZeroDivisionError). NO `ef_val_sweep=false` override — the key doesn't exist in
the old tree (doc 69 §11.1's command is wrong on that detail); single-t-label verified per dir.
Dumps: `result/e0_dumps/arm_{p0v1_30,p0v1_40,p0_55,p0_60}` + `_score.json` siblings.

| arm (identical old code) | data | ckpt | **amp_ratio** | transfer slope | pearson |
|---|---|---|---:|---:|---:|
| p0v1 | v1 | 30 | **0.646** | 0.695 | 0.962 |
| p0v1 | v1 | 40 | **0.756** | 0.792 | 0.973 |
| P0 | v2 | 25/35k (doc 69) | 0.119 | — | — |
| P0 | v2 | 55 | **0.149** | 0.140 | 0.982 |
| P0 | v2 | 60 | **0.148** | 0.145 | 0.984 |

**Read:** the cleanest controlled pair in the campaign is now fully quantified. Same tree, same
launch, only the data differs: v1 is already bootstrapped at ckpt 30 (0.646) and reaches
anchor-level amplitude by 40 (0.756 ≈ warm-start 0.747 / 8mm replay 0.740); v2 stays failed
through ckpt 60 (0.148, plateaued 55→60, a small drift up from 0.119). The scoring-time
confound does NOT rescue P0 at these budgets — no x1-style late transition through 60. This
converts §1a's user-wandb read into evaluator numbers and hardens the data-primary conclusion.
(4wok `checkpoint_50` premise check remains impossible: only `checkpoint_last.pt` survives in
`217720691_*`; the intermediate ckpt no longer exists on disk.)

### 1d. MECHANISM CONFIRMED — heart-L1 w=0.5 bootstraps amplitude on v2 data (2026-08-12 early AM)

First arm verdict, ahead of the planned 15–30k read. w050's in-training `val/ef/slope` showed an
x1-style phase transition at step ~8.9k (15 consecutive positive val epochs, rising to +1.03,
while w000/w010/corseg-arms oscillated around 0). Offline e0 verification (main-tree recipe,
`--tree /home/minsukc/vggt-arm-heart --config default`, cmrx24only, sbatch job 57123809; dumps
`result/e0_dumps/arm_heartl1_{w050_12k,w000_ctrl}`):

| arm (today's code, v2 data, resp ON, native-z) | ckpt | **amp_ratio** | transfer slope | pearson |
|---|---|---:|---:|---:|
| heartl1 **w050** | 50 (~11.8k steps) | **0.640** | 0.639 | 0.968 |
| heartl1 w000 control | last (~17.2k steps) | 0.114 | 0.110 | 0.981 |

PSNR cost: none (w050 val bbox 26.0 / motion 21.5 dB vs w000 26.2 / 21.4). Dose–response:
w=0.1 still flat at 12.2k (in-training slope mixed-negative), w=0.5 transitioned — consistent
with §1b's contrast-upweighting story needing a sufficient dose. The §1b open bridge (~2×
gradient advantage → 6× amp flip) is now closed empirically: an explicit heart-weighted L1
reproduces on v2 data what v1's shading bug provided implicitly, in a SHIPPABLE configuration.
Corseg arms were only at ~8.8k steps at read time (the age w050 transitioned) — their flat
slopes are not yet a verdict. Next: full 5-arm read at 15–30k, then promote per §6.4 (pooled
native-z gather=0.5 confirm run).

### 1e. REFRAME (2026-08-12 ~19:00) — the CONTROL transitions too; weight sets transition TIME

In-training `val/ef/slope` streaks (all three heart-series arms, same recipe, v2 data, resp ON):

| arm | transition step | status @ ~48–53k |
|---|---:|---|
| w050 | ~8.9k | 10/10 positive, mean +0.97, motion PSNR 22.30 (best) |
| w010 | ~32.7k | 10/10 positive, mean +0.83, motion 22.15 |
| **w000 (control, production loss)** | **~43.9k** | 10/10 positive (39-epoch streak), mean +0.84, motion 21.66 |

**The unmodified production loss ALSO transitions on v2 data — just late** (~44k, echoing x1's
~51k). So v2 data does not *prevent* amplitude learning in this recipe; it *delays* the phase
transition, and heart-L1 weight is an ACCELERATOR with a clean monotone dose–response on
transition time (0.5→8.9k, 0.1→32.7k, 0→43.9k ≈ 5× spread). This softens §1d's framing:
heart-L1 is a confirmed accelerator (and PSNR-positive), but the v1↔v2 flip (§1c) is now best
read as a large transition-time shift, not an on/off switch — P0's "failed 0.148 @ ckpt 60"
may also be a too-early read (its own transition, if any, would come later on old-code budgets).
Corseg arms at ~41k remain flat BUT have not yet reached the control's ~44k transition point —
their verdict is deferred to ~45–50k, where the comparison vs w000 becomes fair. Offline
amp_ratio confirmation of the w010/w000 transitions still pending (parked scoring script).

## 2. Heart-L1 ROI augmentation misalignment (bug, fixed, arms restarted)

`gpu_aug.py` warped only `keys=["phases","content_mask"]`; `heart_roi_canonical` passed
through unwarped while `V_gt` was re-derived from the warped phases — under tier=moderate's
±180° rotation the train-time ROI loss upweighted a region misaligned with the actual heart
(partially mitigated by center-proximity + dilation; dilution factor unknown, plausibly ~2× —
the same order as the effect under test ⇒ a null result would have been uninterpretable).

**Fix** (in `/home/minsukc/vggt-arm-heart/training/data/gpu_aug.py`): ROI added to the affine
keys (nearest) + `aug_dict` plumbing + uint8 write-back. batchaug tolerates missing keys.
**Verified by fault injection** (synthetic blob, moderate tier): warped-ROI-vs-warped-anatomy
IoU 0.996 while the PRE-FIX behavior scores 0.000 (test provably catches the bug); 20-draw
exact co-warp (ROI == identically-warped mask, IoU 1.000); missing-ROI batch passes through.

**Restart:** w010/w050 killed (~3h in) → resubmitted as **57022432/57022433**. w000 keeps
running (loss_heart=0 never reads the ROI; the val heartseg metrics read it but val never
affine-augments). The heart tree now differs from main by THREE files (loss.py, default.yaml,
gpu_aug.py) — `sbatch/heartl1_common.sh` header updated.

## 3. The CorSeg-Dice arm (job 57023101)

**Idea (user's):** CorSeg is a differentiable segmenter — use it as a training loss. Dice
rewards putting the endocardial *boundary* in the right place, which is literally what
contraction amplitude is; heart-L1 rewards *intensity* fidelity inside a static ROI.

**Tree:** `/home/minsukc/vggt-arm-corseg` = copy of the FIXED vggt-arm-heart + 6 files:
`training/corseg_dice.py` (new), `loss.py`, `config/default.yaml` (heart_weight→0.0,
corseg_weight added), `data/gpu_aug.py` (`heart_seg_t` co-warp, nearest, round-not-threshold
write-back), `data/datasets/mri_dataset.py` (loads `heart_seg_canonical[..., t_target]` →
key `heart_seg_t`, warn-and-skip shape policy), `data/composed_dataset.py` (**GOTCHA:** sample
keys are explicitly whitelisted there — a new batch key silently vanishes without an entry;
found because the loss's raise-guard fired in the first smoke, exactly as designed).

**Per-step mechanics:** V_canon (1,D,256,256) → per slice: bilinear 1.4→1.25 mm (256→287) →
center-crop 224 → branchless per-slice z-score over nonzero (replicates
`corseg_infer.segment_stack(mode="paper")`) → frozen MedNeXt-L → fp32 softmax →
**volume-level** soft Dice (sums pooled over all D slices, one Dice per foreground class) vs
the GT one-hot; `loss_corseg = corseg_weight · (1 − mean Dice{myo, LV_cav, RV})`, additive on
the unchanged production objective. GT side: no upfront resample (`heart_seg_canonical` is
already canonical-grid); phase-sliced in get_data, co-warped by the train affine, mapped into
CorSeg's 224 frame per step with nearest. Absent classes get Dice=1 (constant, no gradient —
the MRI2CT `mask_absent` pattern at volume level); GT-empty slices contribute no Dice of their
own but false positives there still inflate the union (deliberate: hallucinated heart costs).

**Label spaces differ and are remapped** — GT (nnU-Net-style) 1=LV_cav/2=myo/3=RV vs CorSeg
1=myo/2=LV_cav/3=RV; determined EMPIRICALLY (3 val subjects, matched-pair IoU 0.902/0.706/
0.890, off-diagonal ≤0.04).

**Verification (all fault-injected, real data):** Dice-on-GT through the reimplemented path
0.918 (validates preprocessing+mapping end-to-end); 20px-rolled volume 0.747 loss and
wrong-LUT 0.643 loss (both injections fire); grads finite, 417k nonzero; empty seg → exactly
0.0; multi-label co-warp exact over 20 draws; 1-epoch training smoke on real data green
(`train_loss_corseg` 0.0018 at step 0, backward through it, exit 0).

**Weight calibration — do NOT reason from loss magnitudes.** Dice is O(1) vs loss_volume
~0.01, and its V-gradient is boundary-concentrated: measured ‖∂Dice/∂V‖/‖∂L1/∂V‖ ≈ 1400× ⇒
grad-parity at corseg_weight ≈ 0.001. Shipped arm: **0.002 (~2× parity)**, single arm per
user decision; judged against heartl1_w000 (same lineage, corseg_weight=0 ⇒ production loss).

**⚠️ Hygiene:** the campaign's `amp_ratio` verdict (tools/e0_*) is CorSeg-derived and this arm
*trains on CorSeg* — Goodhart risk. The user confirms nnU-Net is the real evaluator (CorSeg =
fast smoke only): decisive checkpoints of this arm must be re-scored with nnU-Net (Task114,
`-tr nnUNetTrainerV2_MMS`) before any conclusion.

## 3b. POST-MORTEM: the first w002 launch froze at step 33 — a NaN in the CorSeg branch,
## NOT a dead ReLU (two wrong theories were held briefly; the record below keeps all three)

Job 57023101 tripped the docs/64 alert (grad_aggregator ≤1e-6 for 200 steps). Log forensics:
Grad/aggregator AND Grad/point healthy through step 30, then **exactly 0.0000 from step 35**,
`loss_corseg` frozen at 0.0017, and — the giveaway missed at first — **every logged metric
bit-identical from step 35 to 230 across different subjects** (frozen meters, not a frozen
model).

**PROVEN root cause (direct evidence, after three falsified theories):** the frozen
checkpoint's state dict contains **930 tensors with NaN** — the entire trainable set
(aggregator + point head). Sequence: a **finite loss whose backward produced non-finite
gradients** passed the trainer's loss-side NaN guard (`_log_if_finite` checks only the
objective); under bf16 the GradScaler — which skips inf-grad steps for fp16 — is disabled,
so `optimizer.step()` wrote NaN into the weights once, and every forward was NaN thereafter
(`Non-finite objective (nan)` from step 33 in the job / step 17 in the smokes; frozen
bit-identical meters because the NaN-guard skips all updates). Confirmations: control smoke
with `corseg_weight=0` logs **zero** NaNs (Dice-training causal); val-mode on the frozen
ckpt reproduces NaN on every batch; a component-naming diagnostic (added to the trainer's
guard) showed `loss_corseg` and `loss_volume` FINITE while `loss_pos_tv`/`loss_diffusion`/
`mean_disp_norm` — raw `world_points` consumers — were NaN (the splat's range-mask silently
absorbs NaN points, which is why `loss_volume` looked healthy); amp-off and compile-off both
still NaN'd (killing the precision/compile theories); the weight scan settled it.

**Fix stack (all in the corseg tree, each verified):**
1. **trainer.py skip-step guard on non-finite grad norms** — the load-bearing fix; exactly
   GradScaler's fp16 behavior, reusing the norms the clipper already computes (no new sync).
   Logs `SKIPPING optimizer step`; a few per epoch are benign, a streak is not.
2. corseg branch as an **fp32 island** + z-score **sd floor 1e-3** (defense in depth).
3. **per-voxel grad clamp** ±5e-6 on the Dice branch (insurance for its measured ~100×
   spike tail — per-voxel max 16,000× the L1's unweighted; also bounds high weights: 39% of
   voxels saturate at the cap at w=0.1 vs 0.7% at 0.002).
4. **Chunked + checkpointed MedNeXt forward** (4 slices/chunk, `use_reentrant=False`):
   peak 6.85 GB @ D=21 at 386 ms/step worst case (was ~20 GB and OOM'd the L40S).

**Verification:** 100-step real-training smokes at **w=0.002 AND w=0.1** — zero NaN
objectives, 2 and 1 skipped steps respectively, gradients healthy to the end, and
`loss_corseg` DECREASING in both (0.0018→0.0007; Dice-loss 0.90→0.2–0.5 at w=0.1).
Relaunched as **57037405 (w002)** and **57037406 (w100)** — the high-weight arm was added at
the user's request to make the effect maximally visible; the user's "there is no painter"
argument (the head outputs positions, V_canon can only rearrange real input pixels) bounds
the classic segmenter-pleasing failure mode.

**Three falsified theories, kept so they are not resurrected:**
1. *"Dead ReLU from Dice gradient spikes"* (docs/64 pattern-match; motivated the clamp) —
   the freeze was never gradient-magnitude-driven; the probe "finite grads" reading that
   seemed to support partial health was an artifact of the probe exercising only the base
   loss. Clamp kept as insurance, not as the fix.
2. *"Coherent-push / gradient-desert (points driven out of the volume)"* — suggested by
   `mean|dvf|=0.27` and ~29 pre-clip Grad/point spikes at steps 10–15; those numbers were
   symptoms of already-NaN/degenerate weights, not the mechanism.
3. *"z-score sd~1e-8 → bf16 overflow in MedNeXt"* — plausible, fixed anyway (sd floor +
   fp32 island), but the NaN onset was unchanged by that fix and amp-off still NaN'd ⇒ not
   the trigger. Kept as hardening.

## 4. Job state after this session

| job | arm | note |
|---|---|---|
| 56990551 | heartl1_w000 | control, running since 13:39, unaffected by the ROI bug |
| 57022432 / 57022433 | heartl1_w010 / w050 | RESTARTED with the ROI aug fix |
| ~~57023101~~ | corsegdice_w002 (first attempt) | DEAD at step 33 — NaN-poisoned weights, see §3b |
| 57037405 / 57037406 | corsegdice_w002 / w100 | relaunched with the §3b fix stack (weights 0.002 / 0.1) |
| 56779009 / 56854632 / 56854633 | P0 / p0v1 / p0dino | still running (wave 2) |

Read heart-L1 + corseg arms at ~15–30k steps against w000; first checkpoints at
`checkpoint.save_freq` cadence in `scratch/logs/*heartl1_*` / `*corsegdice_*`.
