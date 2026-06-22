# 18 — Joint refiner training degrades OOD generalization

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Visual comparison** (n=5 per mode, V_canon/V_refined at ED, native 256×256, percentile-99.5
> windowing — *no upscale*) of **three checkpoints** across val ON / val OFF / OCMR / Göttingen
> reveals that **jointly training the 3D UNet refiner with the backbone is the wrong move for OOD
> generalization**. Three findings:
>
> 1. **The backbone alone (OLD ckpt, no refiner) generalizes well to OOD** (OCMR + Göttingen),
>    even though it fails in-distribution when the breathing sim is severe (16±8 mm, val ON).
> 2. **Joint training delegates motion-correction responsibility to the refiner**: the NEW joint-
>    refiner ckpt's *V_canon* (the pre-refiner output) is visibly worse than the OLD ckpt's V_canon
>    on every mode. The backbone has been *trained to produce messier output that the refiner is
>    expected to clean* — fine in-distribution, broken OOD where the refiner doesn't generalize.
> 3. **When the motion is large (val ON), the reconstruction is input-draw-dependent**: across
>    6 different `(t, z)` input shuffles for Train_P053 on the OLD ckpt, V_canon visibly varies —
>    the canonical val draw is one realization of a noisy distribution, not a fixed property of
>    the subject. (Multi-draw ensemble could mitigate this — `docs/13`.)
>
> **Caveat caught by the user (2026-06-22)**: bpblrai2's frozen backbone is **not** literally the
> OLD `checkpoint_last.pt` state. It's the **same run** (`t59w6nqy` = `218747856`), warm-started
> from a weights-only export (`scratch/base_weights/t59w6nqy_resp_no_t_weights_only.pt`, Jun 16
> 19:01) — an **earlier snapshot** than OLD `checkpoint_last.pt` (Jun 17 22:17, ~27 h more
> training). Numerical: bpblrai2 backbone vs seed file diff = 0.0; seed vs OLD/last diff = 1.51e-2
> per tensor. So `bpblrai2 V_canon` ≠ `OLD V_canon` exactly because **bpblrai2's frozen backbone
> is less converged**, not because anything is architecturally different. Row 2 → Row 3 still
> isolates the pure SSIM-refiner effect on that frozen (under-converged) backbone, but Row 1 vs
> Row 2 is **not** an equality check — it reflects ~27 h of continued training.
>
> **Practical recommendation:** for OOD-facing inference, prefer the **OLD backbone + separately-
> trained refiner** (bpblrai2-style: backbone frozen at the OLD ckpt's state, refiner trained on
> top) over the jointly-trained model. The joint pipeline can win in-dist (which `docs/11`
> already measured at +1.0 dB motion / +1.5 dB bbox on val) but at the cost of degrading the
> backbone's standalone behavior — which is what OOD inference exposes.
>
> **Status (2026-06-22):** Qualitative, visual comparison only. No quantitative PSNR/SSIM scoring
> for these three checkpoints at ED; OCMR + Göttingen have no V_gt, so PSNR is impossible there.
> Sample sizes are small (n=5 val subjects, n=5 OCMR subjects, n=5 Göttingen subjects). Scripts:
> `tools/three_ckpt_compare.py`, `tools/bpblrai2_compare.py`, `tools/random_draws_multi_subj.py`,
> `tools/diagnose_refiner_question.py`. Artifacts: `result/4way_refiner/` (~30 PNGs).

**Date:** 2026-06-22
**Related:** [[11_unet_refiner_results]] (in-dist refiner gains), [[08_ood_clean_paradox]] (pre-refiner OOD-vs-in-dist),
[[13_limitations_and_improvements]] (multi-draw ensemble as input-variability fix), [[10_breathing_failure_mode]] (splat blur decomposition).

---

## 1. The question

After `docs/11` showed the jointly-trained refiner ckpt (`218349151_mri_refiner_joint`) adds +1.0 dB
motion / +1.5 dB bbox on val, the natural assumption was that the new ckpt is strictly better than
the OLD pre-refiner ckpt (`218747856_mri_volume_resp_allphases_aggft_z_no_t`) — including on OOD
data (OCMR `docs/06`, Göttingen `docs/16`). Visual inspection of the new ckpt's OOD outputs vs the
old ckpt's `_html/13`-style outputs raised a red flag: the **new ckpt looks worse** on OCMR and
Göttingen than the old ckpt's reports suggested. This doc settles whether that's real or a
rendering/perception artifact.

## 2. Setup

Three ckpts, same backbone architecture, differ only in refiner + training mode:

| ckpt | wandb | refiner | training |
|---|---|---|---|
| **OLD** `218747856_mri_volume_resp_allphases_aggft_z_no_t` | (see logs) | none | backbone + point_head + z_embedder + target_t_embedder; loss on V_canon |
| **bpblrai2** `218246076_mri_refiner_frozen_ssim` | `bpblrai2` | 3D UNet, SSIM loss | **backbone frozen** at OLD's state; only refiner trains (42 trainable tensors) |
| **NEW joint** `218349151_mri_refiner_joint` | (see logs) | 3D UNet, L1 loss | backbone + refiner trained **jointly** end-to-end; loss on V_refined |

Eval modes (each n=5, ED only, `target_t = -1.0`):

- **val ON** (in-dist): 5 val subjects (`seq 0..4`), breathing sim 16±8 mm applied to input slices.
- **val OFF** (in-dist): same 5 subjects, no breathing sim.
- **OCMR** (OOD): 5 real-time free-breathing SAX cines from `docs/06` (`us_0084_1_5T` volunteer + 4 patients).
- **Göttingen** (OOD): 5 radial RT free-breathing volumes from `docs/16` (`vol0001_vis1` etc.).

Render convention: **native 256×256** (no upscale — upscaling is cosmetic and inflates apparent
quality), **percentile-99.5 per panel** (top 0.5% of voxels saturate to white; standard medical-
imaging windowing — `max()` windowing lets a single bright voxel kill contrast on V_canon).

Inputs across the OLD-vs-NEW comparison are **bit-identical**: val uses `random.Random(seq_index)`
(deterministic per `seq_index`), OCMR uses `np.random.default_rng(0)` — both fixed-seed PRNGs.

## 3. Finding 1 — OLD backbone alone generalizes well to OOD; fails on val ON

See: `result/4way_refiner/outputs_oldckpt_{val_ON,val_OFF,OCMR,Goettingen}.png` (5 examples each).

- val OFF: cleanest, sharpest. Anatomy crisp.
- val ON: visibly grainier, speckle + dark periphery — the under-corrected breathing sim
  (`docs/08` §3.1, the `−2.31 dB motion` penalty). Heart still legible.
- OCMR: clean-but-soft (the resolution effect from `docs/08` §3.2 + `docs/10`).
- Göttingen: heart anatomy + LV/RV legible across all 5.

The backbone-only model produces qualitatively coherent reconstructions on data it has never seen
(OOD vendor, contrast, resolution, motion regime). Its failure mode is **in-distribution with the
aggressive breathing sim** (val ON) — the well-known motion-correction ceiling from `docs/05`/`07`,
where the model under-corrects deep SI shifts (predicted `‖Δz‖ 1.96 mm` vs sim `16±8 mm`).

## 4. Finding 2 — Joint training degrades the *backbone*, not just the refiner

See: `result/4way_refiner/Q2_OCMR_refiner_isolation.png` (3 rows × 5 OCMR subjects):
- Row 1: OLD ckpt V_canon.
- Row 2: NEW ckpt **V_canon** (the *raw splat output of the new backbone, before the refiner runs*).
- Row 3: NEW ckpt **V_refined** (post-refiner).

Critical observation: **Row 2 is visibly worse than Row 1** on every OCMR subject. That is, the
NEW model's backbone+point_head — *before the refiner does anything* — produces messier V_canon
than the OLD backbone. The architecture is identical (both build `VGGT(enable_camera=False,
enable_depth=False, enable_point=True, enable_track=False, use_z_pose_embedding=True,
use_t_pose_embedding=False, use_target_t_pose_embedding=True, train_on_residual_dvf=True)`); the
warm-start at step 0 produces the same `train_loss_volume = 0.0478`. The only thing that changed
during training is **the gradient signal**:

- OLD: loss = L1(V_canon, V_gt). Backbone pushed to produce a clean V_canon directly.
- NEW: loss = L1(V_refined, V_gt). Backbone is gradient-coupled to the refiner's needs. With a
  trainable downstream cleaner, the optimizer is **free to off-load polish to the refiner** if
  the combined loss decreases. That works in-distribution (the refiner cleans up). On OOD, where
  the refiner can't generalize, the messy V_canon survives unchanged → V_refined is worse than
  what the OLD backbone would have produced standalone.

This is the textbook **post-processor delegation failure**: jointly training a cleaner with its
producer trains the producer to *expect* a cleaner. The cleaner is a learned distribution-
specific operator; the producer's degradation is general.

The bpblrai2 ckpt **partially** illustrates the inverse: when the backbone is frozen at *some*
state (here: an earlier snapshot of the same `t59w6nqy`/`218747856` training run) and only the
refiner trains, the refiner adds in-dist improvement without catastrophically breaking OOD.
See `result/4way_refiner/bpblrai2_{val_ON,val_OFF,OCMR,Goettingen}.png`:

- Row 1 (OLD `checkpoint_last` V_canon) ≠ Row 2 (bpblrai2 V_canon) — small but visible diff,
  especially on Göttingen. The reason is the snapshot mismatch documented in the TL;DR caveat
  (bpblrai2's backbone is `t59w6nqy` at an earlier epoch than `checkpoint_last`, max-abs
  per-tensor diff 1.51e-2 compounding through 48 attention blocks). It is *not* a different
  architecture or a different training mode.
- Row 2 → Row 3: pure SSIM-refiner contribution on top of that frozen (under-converged)
  backbone. The refiner doesn't catastrophically destroy OOD generalization the way joint
  training does.

To get an apples-to-apples "frozen backbone + SSIM refiner vs the same backbone alone" isolation,
the proper Row 1 would be `t59w6nqy` evaluated at the seed-export epoch — not `checkpoint_last`.
Not done in this round.

## 5. Finding 3 — Output is input-draw-dependent under large motion

See: `result/4way_refiner/draws_seq{0..4}_*_{RESULTS,INPUTS}.png` (5 val subjects × 6 draws each
on the OLD ckpt, breathing ON).

Each subject is reconstructed 6 ways: the **canonical val draw** (`random.Random(seq_index)` —
bit-identical to what every `Val_Visuals_subj0` / cardiac filmstrip wandb panel sees) plus 5
draws under different global-RNG seeds. Same `(t, z)` sample-space (anatomy-bbox-clamped z without
replacement when `S ≤ bbox_z_size`; t with replacement from `{0..T_total-1}`); only the *shuffle
order and per-slot t* vary.

Result: V_canon at ED varies visibly across the 6 draws for **every** subject, with the
magnitude of variation tracking the breathing sim's severity. The model is not draw-invariant
under large motion — the canonical val draw shown in wandb is *one realization of a noisy
distribution*, not a fixed property of the subject.

This already motivated `docs/13`'s **multi-draw ensemble** as a "+1.76 dB motion, free" lever.
This finding sharpens the case: the noise is **subject-driven** (different subjects show
different draw sensitivity, partly tracked by `S` = anatomy bbox extent — `Test_P028` with `S=12`
covers the full canonical Z every draw and varies less than `Train_P053` with `S=9`) and large
enough to be visually obvious, not just a small ensemble gain.

## 6. Caveats

- **No quantitative scoring.** This entire analysis is visual. PSNR/SSIM weren't computed for the
  3-ckpt OCMR/Göttingen comparison (and can't be on OCMR/Göttingen — no V_gt). val PSNR for OLD
  vs NEW at ED across `seq 0..4` would settle Finding 2 quantitatively in-dist; not done here.
- **bpblrai2 also confounds loss type AND backbone epoch.** bpblrai2 was trained with **SSIM**
  loss; NEW joint with **L1**. So bpblrai2 vs NEW isolates not just "frozen vs joint" but also
  "SSIM vs L1." On top of that, bpblrai2's frozen backbone is `t59w6nqy` at an earlier epoch than
  OLD `checkpoint_last.pt` (see TL;DR caveat) — so Row 1 ≠ Row 2 in the bpblrai2 panels is the
  expected snapshot mismatch, not a refiner artifact. The cleaner experiment would be a
  **frozen-backbone L1-refiner** ckpt that uses OLD `checkpoint_last.pt` as the frozen seed,
  matching both NEW's loss and OLD's exact backbone state.
- **Small n.** 5 val + 5 OCMR + 5 Göttingen subjects per ckpt is enough for *qualitative* claims
  but not for confident quantitative comparison.
- **ED only.** All renders are at `target_t = -1.0` (ED). Other cardiac phases not tested in
  this round. `docs/09` showed a ~1.7 dB mid-systole dip on the OLD ckpt; the joint-vs-frozen
  comparison at ES isn't done here.
- **Rendering convention sensitivity.** Q1 (`result/4way_refiner/Q1_rendering_demo.png`) showed
  that `max()` windowing vs `pct99.5` windowing makes a perception difference when V has bright
  outlier voxels (notably V_canon). All cross-mode comparisons here use `pct99.5` per panel for
  consistency. Wandb's `Val_Visuals_cardiac_cycle_gif` uses `max()` windowing on V_canon — that's
  why it reads as "blocky", not because the model's output is broken. (`refiner_viz/cardiac_cycle_gif`
  shows V_refined with the same `max()` windowing.)

## 7. Recommendation

For OOD-facing inference (OCMR, Göttingen, future real-time data):
1. **Prefer the OLD backbone (`218747856`) for OOD work**, optionally with a separately-trained
   refiner on top (`bpblrai2`-style — frozen backbone + refiner trained alone). The joint-refiner
   ckpt (`218349151`) is the better in-dist scorer but degrades OOD.
2. **Don't joint-train new refiners** without verifying OOD doesn't regress. The +1.0 dB motion
   in-dist (`docs/11`) is real but doesn't survive distribution shift, and Finding 2 shows the
   loss is in the *backbone*, not the refiner — so the regression isn't recoverable just by
   swapping the refiner.
3. **Multi-draw ensemble at inference** (`docs/13` lever) — given Finding 3, single-draw outputs
   are not subject-deterministic under large motion. Cheap and free.
4. **Quantitative follow-up**: re-eval val PSNR (full / motion / bbox) for OLD vs bpblrai2 vs NEW
   at the same 5 val subjects, ED + a mid-systole phase, to put a dB number on the in-dist vs
   OOD trade-off. Optional: a frozen-bb **L1**-refiner ckpt to deconfound bpblrai2's SSIM choice.

## 8. Provenance & reproduce

- `tools/three_ckpt_compare.py` — 3-row layout (OLD V_canon | bpblrai2 V_refined | NEW V_refined)
  × 4 modes. Outputs `result/4way_refiner/three_ckpt_{val_ON,val_OFF,OCMR,Goettingen}.png`.
- `tools/bpblrai2_compare.py` — focused bpblrai2 isolation (OLD V_canon | bpblrai2 V_canon |
  bpblrai2 V_refined) × 4 modes. Row 1 ≈ Row 2 confirms backbone frozen. Outputs
  `result/4way_refiner/bpblrai2_{val_ON,val_OFF,OCMR,Goettingen}.png`.
- `tools/diagnose_refiner_question.py` — Q1 rendering-choice demo (max vs pct99.5, native vs
  upscale), Q2 OCMR refiner-isolation (OLD V_canon | NEW V_canon | NEW V_refined), Q3 per-val-
  subject all-z filmstrips (V_gt | V_canon | V_refined).
- `tools/render_oldckpt_4way.py` — OLD ckpt alone on all 4 modes (the standalone "does OLD
  generalize OOD" check; Finding 1). Outputs `outputs_oldckpt_*.png`.
- `tools/random_draws_multi_subj.py` — 5 val subjects × 6 draws on OLD ckpt, breathing ON,
  separate RESULTS + INPUTS PNGs per subject (Finding 3). Outputs `draws_seq{0..4}_*.png`.
- All run **directly on the interactive GPU node** (per [[feedback_run_directly_on_gpu_node]]) —
  no sbatch needed. Each script: `PYTHONPATH=training:. micromamba run -n svr python tools/<script>.py`.

Inputs across all OLD-vs-NEW comparisons are bit-identical via fixed-seed PRNGs (val:
`random.Random(seq_index)`; OCMR: `np.random.default_rng(0)`). Outputs match across reruns up to
bf16 cuBLAS/cuDNN reduction-order noise (`~1e-3` relative; visually invisible).
