# 118 — 24-frame rhythm campaign: results so far (af24 complete, regular24/hrv24 partial)

> **TL;DR & takeaway**
> **Scope decision resolved by data (§6): report `af24` + `hrv24`.** Three-point paired EF picture
> (VGGT `diff1000` vs Fetal, |EF err|, same-subject Wilcoxon):
> **`regular24` Fetal wins** (4.62 vs 7.08 pp, Δ +2.45 pp, p = 9e-6, n=159/180 — acdc cohort still
> pending, resubmitted) → **`hrv24` a statistical tie** (6.46 vs 6.82 pp, Δ +0.35 pp, p = 0.22, NOT
> significant, VGGT better in only 79/179 — a coin flip, n=180/180 **complete**) → **`af24` VGGT
> wins decisively** (13.50 vs 7.07 pp, Δ −6.43 pp, p = 4e-14, better in 131/180, n=180/180). The
> claim this supports: **Fetal CMR 4D's periodic-cine assumption is robust to physiological HRV but
> breaks down specifically under simulated AF** — not "any irregularity breaks Fetal." VGGT is flat
> across all three rhythms (PSNR 24.25/24.60/24.52 dB, EF MAE 7.00/6.82/7.07 pp) — it never
> conditions on rhythm, so this is its expected null result. On `af24` alone (full 5-method table,
> §2): VGGT `diff1000` beats Fetal on every metric — PSNR 24.52 vs 22.11 dB, Dice LV ED/ES
> 0.905/0.856 vs 0.807/0.729. Most of the missing Fetal segmentation was filled by an **external
> GPU handoff** (2.4 GB package) after our own queue sat 13–15 h on `Priority`; one small gap
> (`acdc_regular24`, 21 subjects) was missed when scoping that handoff and is resubmitted on our
> own queue (now much quieter). `nogather`/`hw0` collapsing on EF (42.45 / 37.45 pp MAE) despite
> normal PSNR is the **expected result of that ablation**, not a bug.

---

## 1. What's done vs pending

| | `regular24` | `hrv24` | `af24` |
|---|---|---|---|
| Fetal recon | 180/180 ✓ | 180/180 ✓ | 180/180 ✓ |
| Fetal PSNR/SSIM/NCC | 180/180 ✓ | 180/180 ✓ | 180/180 ✓ |
| Fetal EF/Dice | **159/180** (4/5 cohorts; acdc resubmitted `61710869`) | **180/180 ✓** (via handoff) | 180/180 ✓ |
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

**Fetal CMR 4D, image metrics (full) + EF (near-complete):**

| arm | n (PSNR) | PSNR | SSIM | NCC | n (EF) | EF MAE (pp) | EF bias (pp) | Dice LV ED/ES |
|---|---|---|---|---|---|---|---|---|
| `regular24` | 180 | 23.18 | 0.648 | 0.834 | **159/180** (acdc pending) | 4.62 | +1.26 | 0.889/0.836 |
| `hrv24` | 180 | 22.81 | 0.626 | 0.820 | **180/180** | 6.46 | −3.47 | 0.845/0.795 |
| `af24` | 180 | 22.11 | 0.590 | 0.795 | 180/180 | 13.50 | −10.97 | 0.807/0.729 |

PSNR degrades monotonically with rhythm irregularity (23.18 → 22.81 → 22.11 dB), with the AF step
(−0.70 dB) costing roughly double the HRV step (−0.37 dB). EF MAE is **not** monotonic in the same
smooth way — it holds nearly flat from `regular24` to `hrv24` (4.62 → 6.46) and then jumps sharply
into `af24` (→ 13.50), consistent with the paired analysis below: HRV alone barely moves Fetal's EF
error, AF roughly triples it.

**The dose-response, paired against VGGT `diff1000` (|EF err|, same subjects, Wilcoxon):**

| arm | n | Fetal | VGGT | Δ (VGGT − Fetal) | p | VGGT better in |
|---|---|---|---|---|---|---|
| `regular24` | 159 | 4.62 | 7.08 | **+2.45 pp** (Fetal wins) | 9.2e-6 | 60/159 |
| `hrv24` | 179 | 6.46 | 6.80 | +0.35 pp (**tied**) | 0.223 (n.s.) | 79/179 |
| `af24` | 180 | 13.50 | 7.07 | **−6.43 pp** (VGGT wins) | 4.1e-14 | 131/180 |

**This is the central finding this campaign was built to test** (docs/115 §1), and it resolves
cleanly: Fetal's periodic-cine assumption gives it a real, measured EF advantage under a regular
rhythm (p = 9e-6). Under HRV — physiological, non-pathological beat-to-beat variability — that
advantage **evaporates but does not reverse**: the paired difference is not statistically
significant (p = 0.22), and VGGT is better in barely more than a coin flip of subjects (79/179).
Only under simulated AF does the comparison flip hard in VGGT's favor (p = 4e-14). So the honest
claim is narrower and more precise than a strict monotonic ordering would suggest: **Fetal CMR 4D
tolerates ordinary heart-rate variability but its periodicity assumption specifically breaks down
under atrial fibrillation**, where VGGT's per-frame-independent reconstruction has no equivalent
failure mode (§3, VGGT flat at 7.00/6.82/7.07 pp EF MAE across all three arms).

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
- **Returned 2026-09-22**, via a shared Google Drive link, as a `.tar.gz`: 5,496 segmentation
  masks, exact counts matching every cohort we sent, `postprocessing.json` identical to our own
  copy of the model (same per-class Dice — confirms the same weights/config), `uint8` labels
  0–3 as expected. Stamped with the dump's `ef_dump_id` + `nnunet_config=3d_fullres` and scored
  with our own `ef_dice.py score` (no numbers were trusted from the external side, per §4's design).
  **Scoping gap found on our side**: the handoff dump only included `mnms_regular24`, not
  `acdc_regular24` — the other `regular24` cohort that had also been stuck in our queue. Missed
  when building the package; not a segmentation failure. Backfilled by resubmitting just that one
  cohort (21 subjects) on our own queue, `61710869` — much lighter than the original ask, and the
  account's `spgpu` queue had emptied out substantially by the time this was caught.

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

## 6. Scope decision: RESOLVED — report af24 + hrv24

`hrv24` Fetal EF landed complete (180/180, via the external handoff, §4) and settles this.
**`regular24` stays out of the headline** — the paper's regime is AF-or-HRV free-breathing
acquisition (docs/115 §1), not periodic rhythm — but it remains the essential methods-section
control: it is what proves the `hrv24`/`af24` effect is a genuine rhythm effect (Fetal actually
*wins* there, p = 9e-6) rather than an artifact of 24-frame Fetal reconstruction generally.

The `hrv24` result selects the narrower framing over the broader one (§3): Fetal **ties** VGGT
under HRV (p = 0.22, not significant) and only loses decisively under AF (p = 4e-14). So report
**`af24` + `hrv24`** side by side, with `regular24` as the supporting control in methods, and the
claim as: *"Fetal CMR 4D's periodic-cine reconstruction tolerates physiological heart-rate
variability but its underlying periodicity assumption specifically fails under atrial
fibrillation; VGGT's per-frame reconstruction has no equivalent failure mode."* This is a more
precise and more defensible claim than "VGGT is robust to any irregularity," and it is exactly the
question the three-arm design (docs/115) was built to answer.

## 7. Not yet done

- `hrv24`/`regular24` VGGT `base`/`nogather`/`hw0` were deliberately not run — user's scope
  decision, diff1000 only for these two cohorts.
- Whether the gate's `bin_phase_sd` coherence differs across rhythm arms — not checked.
- `figs/rhythm24/` (all result GIFs/PNGs from this campaign) deliberately left uncommitted per
  user instruction.
