# 43 — One-frame ablation: what to evaluate, compared against what

> **TL;DR & takeaway** (2026-07-13). We are training a **5-run controlled ablation** of the reference-slot,
> **1-frame-per-slice** VGGT-MRI model, all warm-started from the same 4wok seed, all 100 epochs, each
> differing from the others by **exactly one config** (see the 5 scripts under `sbatch/oneframe_*.sh`). The
> point of this doc is to state, up front, **what each run tests and which run it is compared against**, so the
> evaluation is decided by design, not improvised afterward. The design is a **hub-and-spoke**: the
> **`gather05` (gather=0.5)** run is the hub; each treatment (aug / physical-z / dino) is compared to it to
> isolate that one factor, and the **no-gather** run is compared to `gather05` to isolate the gather loss
> itself. Five A/B questions: **(1) does the gather aux loss help?** (no-gather vs gather05) → judged on the
> **breathing** metrics; **(2) does data aug help?** (gather05 vs +aug) → judged on **OOD transfer**
> (MIITT/OCMR), NOT in-distribution (aug is known to cost in-dist, docs/05); **(3) does continuous physical-z
> help?** (gather05 vs +contz) → judged on **off-grid OOD** (MIITT native 10 mm) AND a guardrail that it does
> not kill cardiac motion (the failure mode that broke multiframe `s20contz`); **(4) does unfreezing DINO
> help?** (gather05 vs +dino) → judged on **overall PSNR** with a **forgetting** guardrail; **(5) does less
> diffusion reg help EF?** (gather05 vs +lowdiff100, `diffusion_weight` 1000→100) → judged on the **EF sweep**,
> though prior evidence predicts null (docs/33: swapping the regularizer left EF unchanged). Metrics come from
> three existing layers: **in-distribution wandb val** (the GT-referenced ship-decision metrics of docs/38 +
> EF sweep), the **offline OOD head-to-head harness** (docs/42, `scratch/eval/engine/run_vggt.py` vs
> SVRTK/NeSVoR on the frozen breathing bundles), and **nnU-Net EF/Dice** (docs/39). **Decision rule (docs/38):
> a change wins iff `recov_frac_heart`↑ AND `psnr_motion`↑ WITHOUT `hole_frac_heart`↑**, plus the
> factor-specific metric below. **This is a fresh series** (new per-subject breathing + true one-frame regime)
> — do NOT compare its numbers to the old gather05 / docs/42 table; re-evaluate all 5 under this one common
> protocol.

Companion to: docs/37 (the gather aux loss), docs/38 (the val ship-decision metrics), docs/42 (the OOD
head-to-head harness + SVRTK/NeSVoR baselines), docs/39 (nnU-Net EF/Dice), docs/28 (continuous-z), docs/05
(aug hurts in-distribution), docs/24 + docs/33 (EF recovery). Checkpoints + naming: `scratch/checkpoints/README.md`.

---

## 1. The 5 runs (the ablation)

All: warm-start (weights-only) from `scratch/checkpoints/4wok_weights_only.pt` → epoch 0 + fresh optimizer +
fresh warmup→cosine; config `mri_volume_diffusion` (reference-slot, z-only, DPT head, L2 diffusion reg
`diffusion_weight=1000`); **`one_frame_per_slice=true`** (each in-FOV plane observed exactly once — the
sparse extreme, matches the `--regime onef` eval); 100 epochs; `save_freq=50`; peak LR 5e-5; new per-subject
breathing (config default); wandb tag group `1frame_series`.

| # | script | wandb tag | one delta vs the `gather05` hub |
|---|---|---|---|
| 1 | `oneframe_control0_nogather.sh` | `no_gather` | `gather_weight=0.0` (this run *is* the "no gather" baseline) |
| 2 | `oneframe_baseline_gather05.sh` | `baseline,gather05` | **the hub** — `gather_weight=0.5`, nothing else |
| 3 | `oneframe_aug_moderate.sh` | `aug_moderate` | + `data.augmentation.enable=true tier=moderate` |
| 4 | `oneframe_contz.sh` | `contz` | + `continuous_z=true` (physical-z) |
| 5 | `oneframe_dino_ft.sh` | `dino_ft,lr2e5` | + `optim.frozen_module_names=[]` (unfreeze DINO patch_embed) + peak LR 2e-5 |
| 6 | `oneframe_lowdiff100.sh` | `lowdiff100` | + `loss.volume.diffusion_weight=100` (10× lower L2 DVF smoothness) |

("Baseline" is ambiguous: run 1 is the *no-gather* baseline; run 2 is the *reference recipe* the treatments
build on. In this doc "the hub" = run 2 = gather05.)

---

## 2. The five comparisons — what vs what, to test what

Every arrow is a **single-factor A/B** (the two runs differ by exactly one knob).

| # | Question | Compare | Isolates | **Primary metric** | Win criterion |
|---|---|---|---|---|---|
| C1 | Does the **gather aux loss** help? | `no_gather` → `gather05` | the coverage-free gather-placement loss (docs/37) | **breathing**: `resp_slope_dz`↑, `resp_epe_dz_mm`↓, `resp_frac_deep_ignored`↓ | breathing recovery ↑ AND docs/38 rule holds |
| C2 | Does **data aug** help? | `gather05` → `aug_moderate` | moderate train-time GPU aug | **OOD transfer** (MIITT/OCMR head-to-head PSNR/SSIM/NCC) | OOD ↑ without catastrophic in-dist loss |
| C3 | Does **continuous physical-z** help? | `gather05` → `contz` | off-grid z sampling + interp (docs/28) | **off-grid OOD** (MIITT native 10 mm, eval `--continuous-z`) | off-grid OOD ↑ AND motion NOT killed (guardrail) |
| C4 | Does **unfreezing DINO** help? | `gather05` → `dino_ft` | patch_embed trainable + LR 2e-5 (a package) | **overall PSNR** (in-dist `psnr_bbox`/`psnr_motion` + OOD) | PSNR ↑ without a forgetting collapse |
| C5 | Does **less diffusion reg** help EF? | `gather05` → `lowdiff100` | L2 DVF-smoothness weight 1000→100 | **EF** (`ef_val_sweep` slope/Spearman) | EF slope↑ toward 1 without motion/hole regression |

**The universal decision rule (docs/38), applied to every comparison:** a change ships iff
`recov_frac_heart`↑ **and** `psnr_3d_motion`↑ **without** `hole_frac_heart`↑. The factor-specific metric above
is *why* it might win; this rule is the guardrail that it did not win by tearing coverage holes or by helping
the flat static region while hurting the heart.

### Factor-specific reasoning / expectations
- **C1 gather:** the gather loss is a per-pixel depth signal that specifically targets through-plane
  (breathing-z) placement, which the coverage-division splat damps (docs/37 toy: DVF slope 0.40→0.90). So the
  discriminating metric is the **breathing** block, and the discriminating *dataset regime* is **under
  breathing** in the OOD head-to-head (docs/42) — that is where gather should pay off. If gather helps nowhere,
  drop it (→ the no-gather run becomes the recipe).
- **C2 aug:** docs/05 found aug **hurts in-distribution**. Its entire justification is **OOD generalization**
  to real free-breathing cine (MIITT gated, OCMR/MIITT-RT). So judge C2 on the **OOD** head-to-head, treat a
  small in-dist val dip as expected, and only fail it if in-dist *collapses*.
- **C3 contz:** continuous-z matches training to physical z, so it should help on **≠12 mm-pitch OOD data**
  (MIITT native 10 mm, evaluated with `--continuous-z`). **Guardrail:** multiframe `s20contz` learned a
  "snap-to-nearest-plane" through-plane Δz that **froze cardiac motion** (whole-field relocation, not flicker
  — memory `project_gather_contz_relocation_not_flicker`). That failure was in the *multiframe* regime; this
  is 1-frame, so it may not recur — but you must **check it explicitly**: on in-dist val watch `psnr_motion`,
  `recov_frac_heart`, `resp_slope_dz`, and qualitatively that the recon still beats across the cardiac cycle
  (the `tools/miitt_viz/s20_sibling_decider*` aliveness/|Δz| harness is the direct probe). contz **wins only
  if** off-grid MIITT improves AND motion/EF are intact.
- **C4 dino:** unfreezing the pretrained DINOv2 patch_embed adds capacity but risks **catastrophic forgetting**
  of its features. Judge on overall PSNR (in-dist + OOD); **guardrail:** watch the *early-epoch* in-dist val
  PSNR for a cliff (forgetting shows up fast). dino is a 2-knob package (unfreeze + LR 2e-5), so a win is
  "DINO-finetune done properly," not attributable to the unfreeze alone.
- **C5 lowdiff100:** hypothesis is that the L2 DVF-smoothness prior (`diffusion_weight=1000`) blunts the sharp
  inward LV contraction → under-contracted EF. **Prior evidence predicts NULL:** docs/33 measured that swapping
  the regularizer entirely (diffusion→L1-TV) left EF **unchanged** (slope ~0.79 vs 0.77), the diffusion term is
  already only **~2% of the recon loss** at weight 1000 (its own config comment worries it's too *low*), and
  docs/24 proved the under-contraction is an **information/observation limit**, not an over-smoothing artifact.
  So this run is a **rule-out** under the new one-frame/breathing regime (docs/33's test was the old series).
  Judge on the **EF sweep** slope/Spearman vs `gather05`; **read EF at a mature epoch (~40–50), not early** —
  EF recovers with training (docs/33's earlier "flat EF" was undertrained). Guardrail: don't let the looser
  prior tear coverage (`hole_frac_heart`) or add DVF noise.

---

## 3. Metric layers (where each number comes from)

**Layer A — in-distribution wandb val (automatic, per epoch).** All 5 runs log the same val metrics (same
CMRxRecon val, same breathing, same one-frame regime → **directly comparable across the 5**, unlike across the
old series). Lead with the docs/38 GT-referenced metrics:
- `Val_Loss/metric_recov_frac_heart` — oracle-normalized cardiac-ROI recovery (=1 ceiling, <0 below floor).
- `Val_Loss/metric_psnr_3d_motion` (heart) vs `metric_psnr_3d_static` (flat control).
- `Val_Loss/metric_hole_frac_heart` — coverage tripwire (must not rise).
- breathing: `metric_resp_slope_dz`, `metric_resp_corr_dz`, `metric_resp_epe_dz_mm`, `metric_resp_frac_deep_ignored`.
- appearance: `val_metric_psnr_3d_bbox` (honest number), `psnr_3d_full`, `ssim_3d_full`.
- **EF** (clinical, in-dist): the `ef_val_sweep` predicted-vs-GT EF slope / Spearman (docs/24, docs/33).

**Layer B — offline OOD head-to-head (docs/42, `scratch/eval/engine/run_vggt.py`).** Runs a checkpoint on the
**frozen breathing bundles** and is scored by the *same* `assemble_and_gif.py` + `aggregate.py` as SVRTK/NeSVoR
→ PSNR/SSIM/NCC under **clean** and **breathing**, per-subject win rates, plus `resp_diag.json` (predicted Δz
vs applied breathing) and `ed_dvf.npz`. Datasets: **CMRx (30)** and **MIITT (13, real gated OOD)**; contz uses
`--continuous-z`. This is the real generalization test and the SVR-baseline comparison.

**Layer C — nnU-Net EF/Dice (docs/39, Task114).** Segment the saved `recon_*/vol_t*.nii.gz` (both datasets,
canonical, seg-ready) → LV/MYO/RV Dice + EF vs GT. The clinical endpoint; still the open item from docs/42.

---

## 4. Evaluation procedure

**Phase A — while training (wandb, project `vggt-mri`, filter tag `1frame_series`).** Overlay the 5 runs'
val curves. Early read per comparison: C1 = `resp_slope_dz` / `recov_frac_heart` (no_gather vs gather05);
C2/C4 = `psnr_bbox` + `psnr_motion` (aug/dino vs gather05, watch dino's early cliff); C3 = `psnr_motion` +
`resp_slope_dz` + `recov_frac_heart` (contz vs gather05, the motion guardrail). EF sweep across all 5. This
tells you which variants are trending, before spending eval compute.

**Phase B — after training (final `checkpoint_last.pt` of each run).**
1. **Stage each final ckpt** → `scratch/checkpoints/<date>_<name>.pt` and add a row + a "What each model is"
   entry to `scratch/checkpoints/README.md` (its maintenance rule: direct wandb link + prose entry).
2. **OOD head-to-head (Layer B)** — run `run_vggt.py --regime onef` on CMRx + MIITT for each of the 5
   (contz also with `--continuous-z`), score, aggregate → one docs/42-style table **per run**. Then build the
   4 pairwise diffs (C1–C4), each on its primary metric + the docs/38 guardrail.
3. **EF/Dice (Layer C)** — nnU-Net Task114 on each run's recons; compare EF slope/Dice pairwise.
4. **Write up** the 4 verdicts (ship / no-ship each factor) as a results doc (docs/44+) + a `_html/` report,
   with per-subject figures for anything that ships.

**One common protocol.** Evaluate all 5 with the *identical* harness/regime (they already share the val config
and the frozen bundles), so differences are the factor, not the protocol. Match the **eval regime to training**
(1-frame → `--regime onef`; memory `project_1frame_vs_multiframe_eval_regime`) and keep **LPS orientation**
everywhere.

---

## 5. Caveats (read before interpreting)
- **Fresh series.** New per-subject breathing + true one-frame regime ⇒ absolute PSNR will differ from the old
  gather05 / docs/42 numbers (the old run had extra frames + old breathing). Compare only **within these 5**.
- **aug on OOD, not in-dist** (docs/05). **contz on off-grid + motion guardrail** (the s20contz snap). **dino
  watch forgetting.** These are in §2 but bear repeating — the wrong metric will give the wrong verdict.
- **1-frame is harder** than the old extras regime (less redundancy) → lower absolute PSNR is expected and is
  the correct, contract-faithful operating point (docs/04), not a regression.
- **The reference for C2/C3/C4 is `gather05`**, not the no-gather run. Only C1 uses the no-gather run.

## 6. Reproduce (after ckpts exist)
```bash
# stage a final ckpt (weights kept full; naming per checkpoints/README.md)
cp scratch/logs/<exp>/ckpts/checkpoint_last.pt scratch/checkpoints/<date>_<name>.pt

# OOD head-to-head (Layer B), per run — e.g. the gather05 hub on CMRx then MIITT:
EVAL_DATASET=cmrxrecon PYTHONPATH=training:. micromamba run -n svr python \
  scratch/eval/engine/run_vggt.py --dataset cmrxrecon --regime onef \
  --ckpt scratch/checkpoints/<date>_<name>.pt --model-name <name> --stage-tmp
EVAL_DATASET=miitt PYTHONPATH=training:. micromamba run -n svr python \
  scratch/eval/engine/run_vggt.py --dataset miitt --regime onef \
  --ckpt scratch/checkpoints/<date>_<name>.pt --model-name <name> --stage-tmp   # add --continuous-z for the contz run
# then assemble_and_gif.py + aggregate.py (see docs/42 §9), and nnU-Net EF/Dice (docs/39) on recon_*/vol_t*.nii.gz
```
