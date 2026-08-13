# 71 — Amplitude bootstrapping is a TRANSITION-TIME phenomenon; heart-L1 is the confirmed accelerator; CorSeg-Dice gives no acceleration

> **TL;DR & takeaway** — The doc-69/70 "amplitude regression" is not an on/off failure: **every
> heart-series arm on v2 data eventually transitions**, including the completely unmodified
> production loss (w000 control, ~43.9k steps). What the levers change is *when*. Transition
> onset tracks the loss's effective heart-region weighting monotonically across all six
> measured configurations: **heart-L1 w=0.5 → ~8.9k**, v1-data shading bug (implicit ~2×
> heart contrast) → ~23–30k, w=0.1 → ~32.7k, plain loss on v2 → ~43.9k, x1 (old-z+resp-off)
> → ~51k. Explicit heart-weighted L1 at w=0.5 is the **fastest amplitude bootstrap measured
> anywhere in the campaign** (~2.5–3× faster than old-code+v1-data, ~5× vs its own identical
> control), is fully shippable (v2 data, respiratory ON, native-z, today's code), costs
> nothing on PSNR (it has the BEST motion PSNR of all arms), and is offline-verified
> (amp_ratio **0.640** vs control **0.114** at ~12k, distributions non-overlapping, 28/29
> subjects bootstrapped). **Promote heart-L1 w=0.5.** CorSeg-Dice gives no acceleration at
> either weight (flat at ~41k, past both v1's and w010's onsets, on a metric biased in its
> favor); whether it is neutral or harmful resolves when its arms pass the control's ~44k
> onset, but it has already lost to heart-L1.

**Date:** 2026-08-12. **Predecessors:** docs/69 (campaign + evaluator), docs/70 (§1c/§1d/§1e
are the same-day incremental findings this doc consolidates; §2–§5 there hold the heart-ROI
aug fix and the CorSeg-Dice implementation/post-mortem this doc's arms run on).

---

## 1. The arms and their provenance

All five arms share one recipe (identical to each other, hub-like but **native-z**, NOT
fixed12): cmrx24only.txt (235/29/30), CURRENT v2 data, respiratory ON, aug moderate,
seed 42, lr 5e-5 (3-knob), 235 steps/epoch, batch 1. Trees are sibling COPIES, not branches:

| arm | job | exp dir (`scratch/logs/`) | tree | loss delta vs production |
|---|---|---|---|---|
| w000 | 56990551 | `213530039_*_heartl1_w000_*` | vggt-arm-heart | none (control; heart_weight=0) |
| w010 | 57022432 | `213520194_*_heartl1_w010_*` | vggt-arm-heart | + 0.1·L1(heart ROI) |
| w050 | 57022433 | `213520194_*_heartl1_w050_*` | vggt-arm-heart | + 0.5·L1(heart ROI) |
| corseg w002 | 57037405 | `213515736_*_corsegdice_w002_*` | vggt-arm-corseg | + 0.002·(1−softDice) via frozen CorSeg |
| corseg w100 | 57037406 | `213515736_*_corsegdice_w100_*` | vggt-arm-corseg | + 0.1·(1−softDice) |

Heart ROI = `heart_roi_canonical` (nnU-Net-derived, ~4.3–4.6% of the volume ⇒ w=0.5 is ~11×
effective upweighting, doc 69 §9). Both trees carry the docs/70 §2 aug co-warp fix (the ROI/seg
ride the same affine as `phases`; fault-injected soft-Dice 0.830 aligned vs 0.048 unwarped —
figure `figs/corseg_dice_verification.png`). The corseg tree also carries the docs/70 §3 NaN
skip-step guard (steady ~0.4% skipped steps, rate stable and data-driven — identical skip step
numbers in both seeded arms).

## 2. The transition signature and how it was measured

Two independent readouts:

1. **In-training `val/ef/slope`** (every val epoch, in each run's `metrics.jsonl`): regression
   slope of predicted-vs-true EF over the 29 val subjects (ED/ES via CorSeg). Flat model ⇒
   slope wobbles around 0 with coin-flip signs; transitioned model ⇒ unbroken positive streak
   with rising magnitude (the x1 signature from doc 70 §1b). "Transitioned at step S" below =
   first epoch of the still-unbroken positive streak.
2. **Offline 12-phase amp_ratio** (`tools/e0_dump_phase_sweep.py` + `tools/e0_score_volumes.py`,
   doc 69 §2): per-subject (max−min predicted LV ml over t=0..11) ÷ (same for GT), cohort
   median over 29 val subjects. ≥0.4 bootstrapped, ≤0.2 failed. Used to CONFIRM streaks
   (doc 69 §6c documented in-training-vs-offline sign disagreements; never trust the streak
   alone for a ship decision).

## 3. RESULTS — transition onset vs heart-weighting dose

| configuration | code | data | explicit/implicit heart weighting | transition onset (steps) | source |
|---|---|---|---|---:|---|
| **heartl1 w050** | new | v2 | explicit w=0.5 (~11×) | **~8.9k** | metrics.jsonl streak; **offline-confirmed** |
| p0v1 | old | v1 | implicit (~2× contrast, ESPIRiT shading bug) | ~23k | **user wandb read** (old tree logs no metrics.jsonl) |
| original 4wok | old | v1 | implicit (same) | ~30k | **user wandb read** |
| heartl1 w010 | new | v2 | explicit w=0.1 | ~32.7k | metrics.jsonl streak (offline pending) |
| **heartl1 w000 (control)** | new | v2 | none | **~43.9k** | metrics.jsonl streak, 39 epochs unbroken (offline pending) |
| x1 (doc 69) | new | v2 | none (old-z + resp-off easing) | ~51k | doc 69 §5 metrics.jsonl |
| corseg w002 / w100 | new | v2 | Dice via frozen segmenter | none by ~41k | metrics.jsonl (verdict deferred, see §5) |

State at reading time (~48–53k steps): w050 slope mean +0.97 (10/10 positive), w010 +0.83
(10/10), w000 +0.84 (10/10, transitioned latest so magnitude still catching up). Motion PSNR:
w050 **22.30** > w010 22.15 > w000 21.66 > corseg 21.15–21.20 — the heart weighting HELPS
intensity fidelity in the moving region, it does not trade against it.

### 3a. Offline confirmation (the load-bearing numbers)

sbatch job 57123809, dumps `result/e0_dumps/arm_heartl1_{w050_12k,w000_ctrl}` + `_score.json`:

| arm | ckpt | amp_ratio | transfer slope | pearson | per-subject |
|---|---|---:|---:|---:|---|
| w050 | 50 (~11.8k, mid-transition) | **0.640** | 0.639 | 0.968 | 28/29 ≥0.4, min 0.374, max 0.802 |
| w000 | last (~17.2k, pre-transition) | 0.114 | 0.110 | 0.981 | 0/29 ≥0.4, 27/29 ≤0.2, max 0.241 |

Distributions fully separated (w050's worst 0.374 > w000's best 0.241). Also re-confirmed the
doc-70 §1c old-code pair: p0v1 ckpt30/40 = 0.646/**0.756** (anchor level) vs P0 ckpt55/60 =
0.149/0.148 (dumps `arm_p0v1_{30,40}`, `arm_p0_{55,60}`). NOTE: given §3's reframe, P0's
0.148 is best read as PRE-transition, not terminal — do not quote it as "old code fails on v2".

## 4. Conclusions

1. **Mechanism (now closed):** the amplitude signal is a second-order residual whose escape
   from the 95%-static unmasked-L1 background is a phase transition; the effective heart
   weighting of the loss sets the transition time. v1 data's advantage (doc 70 §1b: contrast-
   scaled POSITIONAL gradient through the splat; the L1 itself is sign-only per voxel) was an
   implicit heart-weighting; an explicit heart-ROI L1 reproduces and BEATS it on v2 data.
2. **Ship lever: heart-L1 w=0.5.** Fastest measured onset, shippable config, PSNR-positive,
   offline-verified cohort-wide. Higher weights untested (watch the doc-69 ghost-blob concern
   in `V_canon−V_gt` panels before raising it); w=0.1 also works, just 3.7× slower.
3. **The "regression" was substantially a budget artifact.** Fresh v2 runs needed ~44k+ steps
   to transition under the plain loss; several campaign verdicts (P0 0.119@25–35k, fresh-pooled
   0.136@~70k pooled recipe — the one still-unexplained failure at a large budget) were read
   before or near that horizon. Doc 69's "scoring-time confound" is thereby CONFIRMED as a
   first-order effect, not a caveat.
4. **CorSeg-Dice: no acceleration at w∈{0.002, 0.1}.** Flat at ~41k on a metric biased toward
   it (these arms train on CorSeg, so CorSeg-derived EF should flatter them). Slightly WORSE
   motion PSNR than control. Already dominated by heart-L1 regardless of its final verdict.

## 5. Open items / how to finish this

1. **Corseg neutral-vs-harmful:** read both corseg arms after they pass ~44–50k. If they
   transition on the control's schedule → neutral (drop the loss, keep the infra); if flat
   past ~55k → actively harmful (document and archive). If anything looks positive, re-score
   with nnU-Net Task114 (`-tr nnUNetTrainerV2_MMS`) before believing it — CorSeg contamination.
2. **Offline-confirm w010 and w000 post-transition** + w050 at ~35k: parked script
   `sbatch/e0_score_followup_oneoff.sh` (already pointed at ckpt_150 for the heart arms;
   NOT submitted — user gated it). Expect w000 to climb toward anchor ~0.74; if it stalls low,
   weight also affects the CEILING, not just onset — that would strengthen the w=0.5 case.
3. **Promotion (doc 70 §6.4):** pooled1337 native-z gather=0.5 + heart-L1 w=0.5 fresh run.
   The fresh-pooled anchor's 0.136@70k is the one datapoint arguing pooled needs MORE than
   time — w=0.5's ~5× acceleration is exactly the insurance.
4. **Evaluator hygiene (still not done, docs 69/70):** `_manifest.json` provenance +
   `ef_val_sweep` guard in `e0_dump_phase_sweep.py`.

### Scoring recipe gotchas (verified this session, will bite again)

- Old-code P0-family dumps: tree `/home/minsukc/vggt-oldcode-p0`, config `mri_volume_diffusion`,
  `--dz 12`, split `random_8_1_1_prefixed.txt`, and **NO `ef_val_sweep=false` override** — the
  key does not exist in that tree (doc 69 §11.1's command includes it and is WRONG).
- p0v1 decodes need `data_root=/home/minsukc/vggt/data/CMRxRecon2024_recon_v1_espirit_imagedomain`
  as an **ABSOLUTE path** — the p0 tree has no `data/` symlink for v1; a relative path yields
  0 subjects → `ZeroDivisionError: ... seq_index % len(self.subjects)` in the dataloader.
- Heart/corseg arm dumps: each arm's OWN tree (`vggt-arm-heart` / `vggt-arm-corseg`), config
  `default`, `--limit-val-batches 29`, overrides `split_file=training/splits/cmrx24only.txt
  dataset_name=cmrx24only ef_val_sweep=false logging.ef_eval_enable=false`.
- Always verify per dump dir: 58 files and exactly one `_tNN_` label (doc 69 §2b).
- The trainer writes `checkpoint_last.pt.tmp` then renames atomically — staging a live run's
  `checkpoint_last.pt` is safe.
