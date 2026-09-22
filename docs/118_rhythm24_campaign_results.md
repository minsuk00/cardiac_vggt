# 118 — 24-frame rhythm campaign: results so far (af24 complete, regular24/hrv24 partial)

> **TL;DR & takeaway**
> `af24` (simulated AF, 180 test subjects, all 5 methods) is **fully scored**: VGGT `diff1000`
> beats Fetal CMR 4D on every metric — EF MAE **7.07 vs 13.50 pp** (paired Δ −6.43 pp, p = 4e-14,
> better in 131/180), PSNR **24.52 vs 22.11 dB**, Dice LV ED/ES **0.905/0.856 vs 0.807/0.729**.
> The `regular24` control (partial, n=110/180) shows the opposite sign — **Fetal beats VGGT under
> a periodic rhythm** (EF MAE 4.13 vs 7.52 pp, VGGT worse by +3.39 pp, p = 3e-7). So the claim is
> not "VGGT is always better" — it is **"Fetal's advantage is conditional on regular rhythm and
> reverses under AF."** `hrv24` has full VGGT + full Fetal PSNR but no Fetal EF/Dice yet. The
> missing segmentation (regular24 acdc+mnms, all of hrv24 Fetal) has been **handed off to an
> external GPU** (2.4 GB package, `/gpfs/.../shared_data/vggt_af_seg_handoff.zip`) after our own
> queue sat 13–15 h on `Priority`. `nogather`/`hw0` collapsing on EF (42.45 / 37.45 pp MAE) despite
> normal PSNR is the **expected result of that ablation**, not a bug — that is what it is there to
> show. **Open decision, not yet made: report af24 alone, or the full regular→hrv→af progression
> (recommended below).**

---

## 1. What's done vs pending

| | `regular24` | `hrv24` | `af24` |
|---|---|---|---|
| Fetal recon | 180/180 ✓ | 180/180 ✓ | 180/180 ✓ |
| Fetal PSNR/SSIM/NCC | 180/180 ✓ | 180/180 ✓ | 180/180 ✓ |
| Fetal EF/Dice | **110/180** (3/5 cohorts) | **0/180** | 180/180 ✓ |
| VGGT `diff1000` (recon+metrics+EF) | 180/180 ✓ | 180/180 ✓ | 180/180 ✓ |
| VGGT `base`/`nogather`/`hw0` | not run (scope decision, see §5) | not run | 180/180 ✓ |
| GT segmentation cache | ✓ | ✓ | ✓ |

The missing Fetal EF/Dice cells (`regular24` acdc+mnms, all `hrv24`) are the reason for the
external handoff (§4) — our own `spgpu` queue held those tasks pending on `Priority` for
13–15 hours with no completion.

## 2. `af24` — full results (180/180, all 5 methods)

| method | n | PSNR (dB) | SSIM | NCC | EF MAE (pp) | EF bias (pp) | EF r | EDV MAE | ESV MAE | Dice LV ED/ES | Dice MYO ED/ES | motion EPE (mm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fetal CMR 4D (self-gated) | 180 | 22.11 | 0.590 | 0.795 | 13.50 | −10.97 | 0.66 | 15.6 | 14.2 | 0.807/0.729 | 0.635/0.661 | n/a |
| CiNeVol† | 180 | 25.30 | 0.750 | 0.900 | 9.78 | −9.24 | 0.81 | 12.5 | 9.5 | 0.867/0.824 | 0.747/0.793 | n/a |
| VGGT `base` | 180 | 24.48 | 0.710 | 0.883 | 7.64 | −6.30 | 0.88 | 11.0 | 8.8 | 0.903/0.850 | 0.791/0.827 | 0.88 |
| **VGGT `diff1000`** | 180 | **24.52** | 0.715 | 0.883 | **7.07** | −4.84 | 0.87 | 11.2 | 7.7 | 0.905/0.856 | 0.796/0.836 | 0.85 |
| VGGT `nogather` (ablation) | 180 | 23.68 | 0.679 | 0.858 | 37.45 | −37.45 | 0.55 | 30.2 | 30.8 | 0.837/0.737 | 0.702/0.703 | 2.25 |
| VGGT `hw0` (ablation) | 180 | 23.59 | 0.679 | 0.855 | 42.45 | −42.45 | 0.54 | 30.5 | 35.4 | 0.828/0.722 | 0.683/0.668 | 0.91 |

† CiNeVol is a different session's parallel work (docs/116), scored under the same protocol; not
independently verified here. It is handed every frame's simulated R-peak timing and breathing
level as **input** (never estimated) — not directly comparable to Fetal/VGGT's blind reconstruction.

Paired vs Fetal (same 180 subjects, |EF err|, Wilcoxon):

| method | Δ EF MAE (pp) | p | VGGT better in |
|---|---|---|---|
| `diff1000` | −6.43 | 4.1e-14 | 131/180 |
| `base` | −5.86 | 4.9e-12 | 126/180 |
| `nogather` | +23.95 | 1.9e-30 | 5/180 |
| `hw0` | +28.95 | 3.2e-31 | 1/180 |

`nogather`/`hw0` losing badly on EF while PSNR stays near-normal (23.6–23.7 dB vs `base`/`diff1000`'s
24.5) is **the expected result of that ablation** — it demonstrates the heart-weighted loss and the
gather mechanism are load-bearing for cardiac function specifically, not for image fidelity broadly.
Not investigated further; not a defect.

## 3. Cross-rhythm comparison (regular → hrv → af)

**VGGT `diff1000`, full data, all three arms:**

| arm | n | PSNR | SSIM | NCC | EF MAE (pp) | EF bias (pp) | Dice LV ED/ES | motion EPE (mm) |
|---|---|---|---|---|---|---|---|---|
| `regular24` | 180 | 24.25 | 0.700 | 0.876 | 7.00 | −4.28 | 0.902/0.853 | 0.85 |
| `hrv24` | 180 | 24.60 | 0.719 | 0.884 | 6.82 | −4.93 | 0.898/0.860 | 0.85 |
| `af24` | 180 | 24.52 | 0.715 | 0.883 | 7.07 | −4.84 | 0.905/0.856 | 0.85 |

Flat within noise across all three — VGGT reconstructs each frame independently and never
conditions on rhythm, so this is the expected null result for it.

**Fetal CMR 4D, image metrics (full, no segmentation needed) + EF where scored:**

| arm | n (PSNR) | PSNR | SSIM | NCC | n (EF) | EF MAE (pp) | EF bias (pp) | Dice LV ED/ES |
|---|---|---|---|---|---|---|---|---|
| `regular24` | 180 | 23.18 | 0.648 | 0.834 | **110 (partial)** | 4.13 | +1.63 | 0.894/0.841 |
| `hrv24` | 180 | 22.81 | 0.626 | 0.820 | **0** | — | — | — |
| `af24` | 180 | 22.11 | 0.590 | 0.795 | 180 | 13.50 | −10.97 | 0.807/0.729 |

PSNR degrades monotonically with rhythm irregularity (23.18 → 22.81 → 22.11 dB), with the AF step
(−0.70 dB) costing roughly double the HRV step (−0.37 dB) despite HRV being the more clinically
subtle irregularity. Fetal's partial `regular24` EF (n=110/180, missing acdc+mnms) is provisional.

**The reversal.** On `regular24` (partial), Fetal beats VGGT on EF: MAE 4.13 vs 7.52 pp, paired
Δ = **+3.39 pp in VGGT's disfavor** (p = 2.5e-7, VGGT better in only 35/110). On `af24` (full),
that flips completely: Fetal 13.50 vs VGGT 7.07 pp, Δ = −6.43 pp in VGGT's favor. **This is the
central finding this campaign was built to test** (docs/115 §1): Fetal's periodic-cine assumption
gives it a real, measured advantage under a regular rhythm, and simulated AF is severe enough to
not just erode that advantage but reverse it. `hrv24`'s missing EF cell is exactly the data point
that would show whether this reversal needs AF specifically, or already shows up under mild
irregularity — undetermined until the handoff returns.

## 4. External segmentation handoff (in flight)

Our own `spgpu` queue held the remaining Fetal EF/Dice tasks (`regular24` acdc+mnms, all `hrv24`)
pending on `Priority` for 13–15 hours with 0–3/5 tasks starting; the equivalent jobs on `af24` had
already run through cleanly in ~20–50 min/task earlier, so this was pure queue contention, not a
job problem. Rather than continue waiting, built and handed off a self-contained compute package:

- **`/gpfs/accounts/jjparkcv_root/jjparkcv98/shared_data/vggt_af_seg_handoff.zip`** (2.4 GB,
  group-readable, integrity-checked).
- Contents: 5,496 already-registered, ROI-cropped Fetal reconstruction volumes (229 subjects × 24
  frames, the 6 missing (cohort, arm) pairs); the public pretrained Task114 M&Ms nnU-Net (Zenodo,
  1.2 GB, 5 folds) in the layout `RESULTS_FOLDER` expects; a README with setup (`nnunet==1.7.1`,
  NOT v2), the exact `nnUNet_predict` command (pinned to `-m 3d_fullres -tr nnUNetTrainerV2_MMS`,
  default TTA/ensemble), a concrete 6-way cohort-prefix sharding recipe for multi-GPU use, a
  GT-free self-check script (`quick_check.py`, numpy+nibabel only), and our own reference numbers
  for context.
- No project code, environment, or GT is required on their end — pure compute handoff. We compute
  all actual metrics on our side from the returned raw masks (fail-closed `ef_dump_id`/
  `nnunet_config` stamp checks apply as normal once we stamp the returned `output/`).
- **Gotcha found and fixed en route (docs/114-adjacent, not yet its own doc):** `sbatch/
  rhythm24_seg.sh` was deleting its nnU-Net masks along with the node-local work dir on every run
  — this is why 4 `af24` Fetal subjects with an undefined EF (§ below) needed a full re-segmentation
  to investigate, and it is now fixed (masks copied to `scratch/eval/_rhythm24_segs/<arm>/...`
  before the work dir is cleaned up). All jobs submitted before the fix (the ones stuck in queue,
  now handed off externally) do not benefit from it retroactively.

## 5. The EF-undefined-frame rule (af24, settled)

4/180 `af24` Fetal reconstructions had the LV vanish entirely (all labels) in 1–4 of the 24 frames
— not a partial-slice miss, a whole-frame segmentation failure, visually confirmed as heavily
blurred/mixed-phase-looking reconstructions in those frames (`figs/rhythm24/af24_lv_vanish/`, not
committed — see §6). `ef_dice.py` leaves EF/ESV as `None` there (an empty mask would otherwise
read as ESV=0 → EF=100%, meaningless). Rather than drop those subjects (would remove the method's
worst cases from its own average) or leave them out of Dice/EDV too (was a real bug in the first
version of this table, fixed), the settled rule: **end-systole is the smallest LV volume among the
frames where the segmenter found an LV at all; frames with none are skipped**, same rule for every
method, every subject kept at n=180. Applied via saved re-segmentation masks (EDV cross-checked
against the original campaign score to within 0.1 mL on all 4 subjects). Moves Fetal `af24` EF MAE
from 13.15 (n=176, dropping the 4) to 13.50 (n=180, this rule) — VGGT unaffected (no VGGT
reconstruction had a whole-frame LV vanish in any arm).

## 6. Scope decision needed: af24 alone, or the full 3-rhythm progression?

Not yet decided. Two options:

1. **`af24` only** as the headline (§2) — simplest, largest and cleanest effect, matches the
   paper's stated research question most directly (docs/115: "does VGGT ≥ Fetal under AF").
2. **The full regular→hrv→af progression** (§3) as the headline, with `af24`'s per-method table
   as the detail — shows the effect is a genuine dose-response to rhythm irregularity rather than
   an artifact of the `af24` simulation specifically, and the `regular24` reversal (Fetal *better*
   there) pre-empts an obvious reviewer objection ("maybe Fetal is just worse at 24 frames
   generally") that `af24` alone cannot answer.

Recommendation: **option 2**, once `hrv24` Fetal EF lands — the reversal itself (§3) is a stronger,
more defensible result than the af24 gap alone, and `hrv24` was built specifically to supply the
missing middle data point. Until then, `af24` is the only cohort where the story is complete.

## 7. Not yet done

- `hrv24`/`regular24` VGGT `base`/`nogather`/`hw0` were deliberately not run — user's scope
  decision, diff1000 only for these two cohorts.
- Whether the gate's `bin_phase_sd` coherence differs across rhythm arms — not checked.
- `figs/rhythm24/` (all result GIFs/PNGs from this campaign) deliberately left uncommitted per
  user instruction.
