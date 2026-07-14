# 42 — VGGT-model eval on the frozen breathing bundles: gather05 head-to-head vs SVRTK / NeSVoR

> **TL;DR & takeaway** (2026-07-13). Built `scratch/eval/engine/run_vggt.py` — the GPU counterpart of the
> classical `run_svrtk3d.sh`/`run_nesvor.sh` — that runs a VGGT-MRI checkpoint on the **frozen breathing
> bundles** (docs 40/41, `eval/README.md`) and is scored by the **identical** `assemble_and_gif.py` +
> `aggregate.py` as the classical baselines, so the comparison is provably fair (same corruption, GT, ROI,
> metric). Ran it on **gather05** (`216539845_…ftgather05_1frame`, reference-slot z-only, 1-frame, wandb
> `81li618p`). **Result (the headline): VGGT loses on *clean* input but WINS under *breathing* on both
> datasets, and is far more breathing-robust — confirmed by PSNR *and* SSIM *and* NCC (so it's not a
> normalization artifact).** CMRx (30): breath **20.6 dB vs SVRTK 17.6**, breathing cost **−1.5 vs −10.6**.
> MIITT (13): breath **16.2 vs NeSVoR 15.1 vs SVRTK 14.8**, cost **−1.4 vs −2.9 vs −6.4**; wins 29/30 (CMRx)
> and 10/13 (MIITT) subjects. **Why:** the model predicts breathing-corrective through-plane motion (slope
> **0.73 in-dist**, 0.45 OOD; in-plane far better, EPE 0.4–0.6 mm; deep-breath undershoot ~61% is the
> residual) — classical single-stack SVR corrects **none** of it (slope ~0, doc 40). SVRTK wins *clean*
> only because its input is an **oracle-gated coherent stack** while VGGT's "clean" is still a hard
> **scattered-phase-blind** reconstruction. Also: **continuous-z ≈ snapped for gather05** (integer-z-trained
> → snapped is the headline; contz only pays off if you *train* with continuous-z), and **fixed a
> reference-plane bug** (was canonical-cube center, now content-bbox center to match training — affected
> 22/30 CMRx subjects). Data efficiency: VGGT uses ~21 frames (CMRx) / ~39 (MIITT) for the whole cycle vs
> ~122 / ~305 classical (~6–8×) — but the deeper win is the **gating-free scattered acquisition** classical
> SVR structurally cannot use. Figures `result/vggt_eval_analysis/`, GIFs `result/*_cmp_*.gif`.

Companion to: `eval/README.md` (the frozen-bundle harness), doc 40/41 (SVRTK bakes breathing in → the −dB),
doc 33 (gather05/4wok per-axis analysis), doc 32 (NeSVoR), doc 38 (the trainer's resp val metrics this
mirrors), doc 04 (blind-input-phase contract), `docs/01`/`docs/05` (respiratory sim).

---

## 1. What & why

The `scratch/eval/` harness (doc 40/41) freezes one breathing realization per subject and reconstructs it
with the classical SVR baselines. The open critical-path item was **running our own model on the *same*
frozen breathing** so the numbers are directly comparable. The blocker: the pre-existing model runners
(`inference/run_cmrxrecon.py`, `run_gated_ood.py`) **re-apply breathing via `gpu_augment_batch(rcfg,
seq_index)`**, which re-samples from the trainer's *positional* seq_index — a **different** realization than
the harness's name-hash seed — so they can't be used for the head-to-head. This doc's runner consumes the
frozen breathing directly.

**Model under test — gather05:** `scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt`.
Architecturally the reference-slot z-only model (config `mri_volume_diffusion`, warm-started from
`4wok_weights_only.pt`, `loss.volume.gather_weight=0.5` — a training-only aux loss, doc 37; wandb `81li618p`).
**Trained 1-frame-per-slice** (`max_img_per_gpu=12`) and **snapped-z** (no `continuous_z`). Loaded via
`inference.inference.load_rtfb_model_reference(ckpt, refiner=False)`.

---

## 2. The runner — `scratch/eval/engine/run_vggt.py`

Loads the model **once**, loops all built subjects (`<dataset>/out/*/manifest.json`), writes into a dated
method dir `<dataset>/out/<subject>/vggt_<YYYYMMDD>_<model>[_contz]/`. Key design points:

- **Consumes the FROZEN breathing — never re-samples.** GT = frozen `gt/gt_t*.nii.gz`; input = frozen
  `breath/stack_t*.nii.gz`. CMRx: both already canonical → read direct. MIITT: `breath/` is NATIVE → placed
  to canonical with `assign_canonical_z` + `to_canonical_inplane` (the same placement `build_canonical_bundle`
  uses; verified byte-identical to the GT bundle — placed clean slice `corr 1.0, max|diff|=0`). No
  `gpu_augment`, no `rcfg`, no seed. So model and baselines provably share one corruption.
- **1-frame regime** (gather05): `--regime onef` → one slot per in-FOV plane, reference plane = swept slot 0
  ONLY (no reference-plane companions). This matters: piling the multiframe reference burst on a 1-frame
  model averages under the splat coverage-mean → the "frozen"/washed-out artifact (memory
  `1frame_vs_multiframe_eval_regime`). `--regime multiframe` (reference plane contributes all T phases +
  `frames_per_slice` bursts) is for future s20-style models. `build_slots` mirrors
  `run_cmrxrecon._build_multiframe_batch`.
- **Reference plane = content-bbox center** `z_mid=(z0+z1)//2` over in-FOV planes, then the entry nearest
  `z_mid`. This **matches the trainer + run_cmrxrecon** (slot 0's camera-token anchor was trained at
  bbox-center). **FIX:** the first version used `argmin|z_val|` (canonical-cube center), which diverges by one
  12 mm plane on off-center FOVs — verified **22/30 CMRx subjects** got plane 6 instead of 5 (see §4). Not a
  fairness issue (SVRTK has no reference) but it fed an out-of-convention anchor; fixed → 0/30 divergent.
- **Records everything** (README §6b analog) so any later metric recomputes offline with no re-run:
  `recon_{clean,breath}/vol_t*.nii.gz` (canonical, EF/Dice-ready), `cine_{clean,breath}.nii.gz` (written by
  the scorer), `metrics.json`, `resp_diag.json` (predicted Δz vs applied breathing, §6), `timing.json`
  (model-load + per-phase + per-subject wall), `provenance.txt` (cmd, ckpt, wandb id, git, GPU, torch/cuda,
  regime, z-mode), `metadata.json` (model card), `ed_dvf.npz` (ED Δ field + per-slot z/phase + applied disp —
  the VGGT analog of SVR's `.dof`).
- **`--stage-tmp` (GPFS load fix, §9):** copies the ckpt to node-local `/tmp` + strips to weights-only, loads
  from there (**~266 s → ~14 s**). Original file untouched; idempotent (staged once per node). Metadata still
  records the original GPFS path.
- **MIITT z placement:** `--continuous-z` keeps 10 mm native slices at true fractional canonical z (no 12 mm
  snap); default snaps. See §7.

**Scoring — reused verbatim.** The model recon is canonical `[0,1]`, so `assemble_and_gif.py`'s `prep_recon`
scores it **AS-IS like SVRTK** over `mask_heart & FOV` (PSNR/SSIM/NCC), and `aggregate.py` rolls it up —
identical to the classical methods. **One tiny backward-compatible tweak:** `assemble_and_gif.py:171` now
reads `os.environ.get("FOV_MASK","mask.nii.gz")` so MIITT can pass `FOV_MASK=mask_fov.nii.gz` (CMRx default is
byte-identical to the old literal → SVRTK/NeSVoR numbers unchanged). **MIITT foot-gun:** score the model recon
with `engine/assemble_and_gif.py` (canonical, identity resample), NOT `miitt/assemble.py` (assumes a
native-frame recon → would misplace the canonical model recon).

---

## 3. prove-it review (4 reviewers + adversarial verify)

Clean bill on the scored recon path (axis order, 1-frame slots, frozen-breathing byte-identity, splat/model
contract all independently cleared + CPU-probed). Confirmed + fixed (all in `run_vggt.py` unless noted):

1. **[medium] Reference plane** ≠ training convention — canonical-center vs bbox-center; **22/30 CMRx** off by
   one 12 mm plane. Fixed (§2), re-ran CMRx.
2. **[low] `resp_diag` "clean" used the breath disp** as "applied" → misleading clean slope/corr. Fixed:
   applied = 0 on the clean pass (negative control).
3. **[low, latent] `--refiner` output discarded** — always splatted `world_points`, never read `V_refined`.
   Fixed to prefer `V_refined` (not exercised by gather05).
4. **[low] `resp_diag` slot/threshold** didn't match the trainer's `metric_resp_slope_dz`. Aligned — then per
   user decision **slot 0 (reference) is now INCLUDED everywhere**: changed BOTH `run_vggt.resp_diag` AND
   `training/loss.py:metric_resp_slope_dz` (`range(1,…)`→`range(…)`) so eval and wandb numbers are comparable.
   The loss block is **val-only** (`not pos_pred.requires_grad`, line 570) → training numerics bit-identical.
   (Safe to change since no training run depended on it yet.)
5. Added an empty-FOV subject guard (skip instead of `argmin([])` crash).

---

## 4. Results — head-to-head (all scored identically over `mask_heart & FOV`)

### CMRx (30 subjects, VGGT gather05 vs SVRTK; NeSVoR only on Train_P053)
| method | clean PSNR | breath PSNR | cost | clean SSIM | breath SSIM | breath NCC |
|---|---|---|---|---|---|---|
| **VGGT gather05** | 22.09 | **20.57** | **−1.52** | 0.907 | **0.860** | **0.867** |
| SVRTK (oracle-gated) | **28.23** | 17.63 | −10.60 | **0.979** | 0.754 | 0.754 |

Per-subject: **VGGT wins breath on 29/30**. NCC (affine-intensity-invariant) ≈ SSIM → the win is real
structure, not a PSNR normalization effect.

### MIITT gated (13 subjects: 10 volunteers + 3 patients; all three methods)
| method | clean PSNR | breath PSNR | cost | breath SSIM | breath NCC |
|---|---|---|---|---|---|
| **VGGT gather05 (snap)** | 17.51 | **16.16** | **−1.35** | **0.704** | **0.711** |
| NeSVoR | 17.98 | 15.12 | −2.86 | 0.672 | 0.680 |
| SVRTK (oracle-gated) | 21.23 | 14.84 | −6.40 | 0.681 | 0.686 |

Per-subject: **VGGT wins breath 10/13** vs SVRTK; wins in both the volunteer and patient subgroups. Same
story as CMRx against *two* classical baselines.

**Interpretation.** SVRTK wins *clean* because its clean input is an **oracle-gated coherent stack** (all
slices at the known target phase → trivial super-resolution). VGGT's "clean" input is still **scattered across
random unknown cardiac phases** (only the breathing is removed) — it must infer each slice's phase and warp it
to the target, the core hard task. Under breathing, SVRTK bakes the per-slice shifts in (doc 40/41) → crashes;
VGGT corrects them → barely moves. **The meaningful comparison is under breathing.** ("Perfect motion" still
wouldn't give a perfect splat: scattered 2D slices lack through-plane target-phase content, and the splat is
coverage-limited.)

---

## 5. Breathing-prediction analysis (simulated vs predicted displacement)

`ed_dvf.npz` (applied per-slice disp) + `resp_diag.json` let us compare the model's predicted Δ per input
slice against the *exact* applied breathing. Per input slice, ED phase:

| axis | CMRx slope | CMRx EPE | MIITT slope | MIITT EPE |
|---|---|---|---|---|
| through-plane (SI, Δz) | **0.73** (corr 0.91) | **2.0 mm** | 0.45 (corr 0.49) | 4.5 mm |
| in-plane (AP) | 0.78 | 0.6 mm | 0.56 | 1.5 mm |
| in-plane (x) | 0.86 | 0.4 mm | 0.10 | 1.7 mm |

- **In-plane correction is ~3–4× more accurate than through-plane** — in-plane motion is *observed* in the
  slice; through-plane depth must be *inferred*. So the residual error concentrates through-plane (the
  physically hard, acquisition-limited direction).
- **Deep-breath undershoot:** recovery ≈ full for shallow breaths, drops to **~61%** for deep breaths
  (≥12 mm SI) — the main residual (matches doc 33's ~54–72% / deep-breath-ignored finding).
- **In-dist good (0.73), OOD weaker (0.45)** — real gated MIITT is genuinely harder; but even the weaker
  correction beats classical (slope ~0), which is why VGGT still wins breath OOD.

*(Numbers: through-plane slope/EPE from `resp_diag.json` — true input-FOV mask, per-subject, matches the
trainer; per-axis/deep-breath from `ed_dvf.npz` with a `|Δ|>0` FOV proxy, so its pooled through-plane slope is
a touch lower ~0.59 — same message.)*

---

## 6. Continuous-z vs snapped-z (MIITT)
| variant | clean | breath | cost |
|---|---|---|---|
| snapped-z (headline) | 17.51 | 16.16 | −1.35 |
| continuous-z | 16.92 | 16.18 | −0.74 |

**Breath essentially identical (+0.02 dB); clean slightly *worse* (−0.59).** gather05 is integer-z-trained, so
fractional z is mild OOD → no benefit, slightly hurts clean. **Use snapped for gather05.** The
physiological-correctness benefit of continuous-z only materializes if the model is *trained* with it
(`s20contz`) — a retrain, not an eval switch.

---

## 7. Data efficiency / frame counts (verified from `ed_dvf.npz` slot counts)
| | CMRx (T=12) | MIITT (T=30) |
|---|---|---|
| VGGT in-FOV planes S | ~10 | ~10 |
| VGGT, one target phase | ~10 frames (1 ref + ~9 non-ref) | ~10 |
| VGGT, whole cycle | **~21** (9 shared non-ref + 12 ref) | **~39** (9 + 30) |
| classical, whole cycle | ~122 (12×~10) | ~305 (30×~10) |
| ratio | **~6×** | **~8×** |

Nuances (state when reporting): (1) it's fewer *frames per slice*, not fewer *slices* (same z-coverage); (2)
savings are **whole-cycle** — per single phase both use ~10 frames; VGGT reuses the non-ref frames across all
phases; (3) the reference plane is *fully swept* (12/30 frames), so it's "1 frame per non-reference slice + a
reference-plane cine," not literally 1-frame everywhere (the 1-frame extreme is the aspirational contract,
doc 04). The **non-reference frames are at random, model-unknown cardiac phases** (e.g. CMRx `[10,7,6,3,…]`)
AND **per-slice respiratory phases** (per-subject amplitude, per-slice breath moment) — the model is blind to
both. The deepest differentiator is not the count but the **gating-free scattered acquisition** classical SVR
structurally cannot use.

---

## 8. Timing (measured, per run)
| run | model load | per phase | per subject | full run |
|---|---|---|---|---|
| CMRx (30, T=12) | 230 s (direct GPFS) | ~1.03 s | ~25 s | ~16.3 min |
| MIITT snap (13, T=30) | **14 s** (`--stage-tmp`) + 20 s stage | ~1.02 s | ~62 s | ~13.6 min |
| MIITT contz (13, T=30) | 15 s (reused stage) | ~1.34 s | ~82 s | ~18.0 min |

Per-phase (one full 3D volume) ~1 s everywhere; MIITT longer only due to T=30 vs 12; contz ~30% slower/phase
(keeps all ~13 native slices, no snap-dedup → more input slots). **vs classical: SVRTK ~2.4 min CPU (CMRx) /
NeSVoR ~39 min GPU** → VGGT is ~10²–10³× faster.

**GPFS load lesson:** direct `torch.load` from GPFS = **266 s**; from `/tmp` =
**4.7 s** — the cause is `torch.load`'s storage-by-storage *small/seeky* reads, which GPFS handles terribly
even for one large file (a *sequential* read is ~35 s; a full copy to /tmp then load is fast). NOT
deserialization, NOT the optimizer, NOT `weights_only` (the full 8.3 GB ckpt deserializes in 4.7 s off /tmp;
model weights are 3.77 GB of it). Fix = stage on /tmp (`--stage-tmp`).

---

## 9. Reproduce
```bash
# CMRx (30) — recon, score, aggregate
EVAL_DATASET=cmrxrecon PYTHONPATH=training:. micromamba run -n svr python \
    scratch/eval/engine/run_vggt.py --dataset cmrxrecon --regime onef [--stage-tmp]
for s in $(ls scratch/eval/cmrxrecon/out); do \
  EVAL_DATASET=cmrxrecon micromamba run -n svr python scratch/eval/engine/assemble_and_gif.py $s vggt_20260713_gather05; done
micromamba run -n svr python scratch/eval/engine/aggregate.py cmrxrecon vggt_20260713_gather05

# MIITT (13) — snapped + continuous-z ; MUST use FOV_MASK=mask_fov + the engine scorer
EVAL_DATASET=miitt PYTHONPATH=training:. micromamba run -n svr python \
    scratch/eval/engine/run_vggt.py --dataset miitt --regime onef --stage-tmp [--continuous-z]
EVAL_DATASET=miitt FOV_MASK=mask_fov.nii.gz micromamba run -n svr python \
    scratch/eval/engine/assemble_and_gif.py <subj> vggt_20260713_gather05[_contz]
```
Other model: `--ckpt <path> --model-name <name> --config <cfg> --regime {onef,multiframe}`.

## 10. Figures & GIFs
- `result/vggt_eval_analysis/1_robustness_dumbbell.png` — clean→breath PSNR per method (line = breathing cost).
- `…/1b_robustness_structural.png` — same in SSIM + NCC (normalization-robust).
- `…/2_per_subject_breath.png` — per-subject VGGT vs SVRTK under breathing (win rates).
- `…/3_breathing_cost.png` — per-subject breathing-cost strip.
- `…/4_resp_mechanism.png` + `…/5_breathing_prediction.png` — predicted Δz vs applied breathing (through-plane
  scatter, per-axis EPE, deep-breath undershoot).
- `result/{Train_P053,Volunteer1}_cmp_{breath,clean}.gif` — GT/SVRTK/NeSVoR/VGGT rows × z-plane columns
  (heart-ROI masked+cropped), over the cardiac cycle; breath GIFs annotate the applied `|disp|` per plane.
- Data JSONs: `/tmp/analysis_data.json`, `/tmp/struct_final.json`, `/tmp/breath_pred.npy` (regenerable from the
  per-subject `metrics.json`/`resp_diag.json`/`ed_dvf.npz`).

## 11. Files touched
- **New:** `scratch/eval/engine/run_vggt.py`.
- **Edited:** `scratch/eval/engine/assemble_and_gif.py` (FOV_MASK env, 1 line);
  `training/loss.py` (resp metric now includes slot 0 — val-only, training bit-identical). The GPFS→/tmp
  ckpt-load lesson (§8) was also codified in the user's global engineering guidelines.

## Open / next
- **EF / Dice** via nnU-Net Task114 on the saved `recon_*/vol_t*.nii.gz` (both datasets, canonical, seg-ready)
  — the clinical metric, not yet computed here (doc 39 has the seg pipeline).
- **CMRx NeSVoR on all 30** for a full 3-way there (currently only Train_P053).
- **Continuous-z-trained model** (`s20contz`) evaluated with `--continuous-z` — to realize the physiological
  z-fidelity benefit.
- **Real RTFB transfer** (OCMR/MIITT-RT, no GT) — the aspirational validation beyond simulated breathing.
- `--refiner` path is read-verified only (no refiner ckpt exists yet).
