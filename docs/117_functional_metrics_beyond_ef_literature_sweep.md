# 117 — Functional metrics beyond EF MAE: literature sweep + what two CiNeVol trial subjects showed

> **TL;DR & takeaway**
> EF MAE stays (reviewers expect it) but cannot be the only functional number on the rhythm arms: it
> reads 2 of 24 frames, is blind to timing, cancels a proportional size bias, and under AF its
> max/min over a 2 s window can pair the EDV of one beat with the ESV of another. The literature
> scores arrhythmia **per beat** (Contijoch 2016, Olausson 2026; ≥5-beat average in AF, Lang 2015).
> No established *numeric* volume-curve-error metric was found by two independent searches — it is a
> technical metric and must be labelled as one. **Recommended set:** EDV/ESV/SV/EF + LV mass reported
> ACDC-style (bias ± SD, MAE, correlation); Dice/HD95 over **all 24 frames**; LV volume-time-curve
> error (% EDV) + correlation; per-beat EF on `af24` (feasible on ~90 % of subjects, one beat each);
> optionally x-t spatiotemporal SSIM (CineVN). Skip strain, PFR/PER, index beat. **Nothing here is
> implemented.** Citations were gathered by three search agents; only the SCMR 2020 wording and the
> CiNeVol paper's own metrics were re-checked first-hand — verify the rest before citing.

## 1. Why this came up (measured, 2026-09-21, CiNeVol trial fits, arm `cinevol_debug`)
| subject (`cmrx2023_af24`) | label misplacement | CiNeVol EF err | LV Dice ED/ES | NCC | LV curve error | curve corr |
|---|---|---|---|---|---|---|
| `CMRx23_Test_P005` (17th pct, easy) | 0.92 ph | **+0.8** | 0.87 / 0.89 | 0.846 | 10.7 ml = 7 % EDV | 0.92 |
| `CMRx23_Train_P006` (hardest with Fetal+VGGT EF) | 1.80 ph | **−24.7** | 0.70 / 0.80 | 0.889 | 26.4 ml = 19 % EDV | 0.40 |

Same subjects, other arms — P005: Fetal −6.9, VGGT base −6.0 (Dice 0.92/0.94); P006: Fetal −8.8,
VGGT base +5.7 (Dice 0.92/0.94). Lessons: (i) on the easy subject CiNeVol replays one averaged beat
(errors up to 38 ml mid-beat) yet EF is near-perfect because ED/ES are plateaus and its ~5 % oversize
cancels in the ratio; (ii) on the hard subject everything collapses **except NCC/SSIM** — image
metrics alone would have missed it; (iii) n = 2, no claim about cohort ranking.

## 2. What the sweep found
**Guideline-standard LV measures** (SCMR 2020, Schulz-Menger et al., JCMR 22:19 — wording re-checked
first-hand): "LV end-diastolic volume, LV end-systolic volume, LV stroke volume, LV ejection fraction,
cardiac output, LV mass, and body-surface area indexed values". No mention of volume-time curves,
peak filling/ejection rate. On strain the Task Force "chooses to refrain from making a dedicated
statement".

**What error size matters** (agent-opened, not re-checked): human inter-observer mean absolute
difference LVEDV 6.1–8.8 ml, LVESV 4.1–7.1 ml, LV mass 4.2–6.6 g (Bai et al., JCMR 2018);
inter-study CoV EF 2.4–7.3 %, EDV 2.9–4.9 %, ESV 4.4–9.2 %, mass 2.8–4.8 % (Grothues 2002).

**Arrhythmia** (agent-opened unless noted): ≥5 beats averaged in AF (Lang 2015, ASE/EACVI);
index-beat alternative (RR1≈RR2); real-time CMR reports per-beat EDV/ESV/SV/EF grouped by preceding
R-R (Contijoch 2016 — abstract only; Olausson 2026 preprint); SV/EF depend on preceding R-R (Iwase
1988; Muntinga 1999). SCMR/ESC statements on CMR in AF: not found.

**Comparable papers**: ACDC (Bernard 2018) reports clinical indices as correlation, bias ± SD, MAE,
% patients with EF error < 5 %, with inter-observer MAE as the human ceiling; Dice + Hausdorff at
ED/ES only. CineVN (Vornehm 2025) uses a spatiotemporal SSIM on x-t profiles + Bland-Altman function
+ peak strain (agent-opened; my re-check fetch failed). Qin 2020 reports Dice/contour distance and
myocardial volume preservation per apical/mid/basal. **CiNeVol itself (read first-hand from the
PDF): SSIM/PSNR/NMSE, stroke-volume difference (3D nnU-Net; ED = frame 0, ES = curve minimum),
sharpness index, blinded expert 5-point score; LV volume curves and x-t profiles only as figures.**

## 3. Feasibility on our bundles (measured, reference plane, 180 test subjects)
| arm | 0 complete beats in the 2 s window | exactly 1 | 2+ | has an index beat (RR1/RR2 0.96–1.04) |
|---|---|---|---|---|
| `regular24` | 0 % | 100 % | 0 % | 100 % |
| `af24` | 10 % | 77 % | 13 % | 13 % |
⇒ a 5-beat per-subject average is impossible; per-beat EF works on ~90 % (median 11 frames/beat,
min 5); index beat covers too few; pooling one beat/subject gives ~190 beats for an
error-vs-preceding-R-R plot, which is how the arrhythmia papers present results.

## 4. Recommendation
| metric | status | reveals what EF MAE cannot | cost |
|---|---|---|---|
| EDV, ESV, SV, EF: bias ± SD, MAE, corr, % with EF err < 5 % | guideline-standard + ACDC format | size bias (proportional → SV sees it; constant → EF sees it) | EDV/ESV already in the EF json |
| LV mass | guideline-standard, tightest reproducibility | wall blurring/thinning | already computed (`lvm_*`) |
| Dice + HD95 over all 24 frames (mean, worst frame) | standard seg metric, extended | mid-beat shape/timing errors | segs already produced, currently discarded |
| LV volume-time-curve error (% EDV) + correlation | **technical, no precedent as a number** | the averaged-beat failure (P005: 7 %, P006: 19 %) | same segs |
| per-beat EF / SV error on `af24`, vs preceding R-R | literature-standard for arrhythmia | removes the cross-beat max/min pairing | same segs + manifest beat boundaries |
| x-t spatiotemporal SSIM through the LV centre | precedent CineVN (not re-checked) | image-domain motion error without any segmentation | new small script on the scored cines |
| myocardial volume constancy across frames | plausibility check (Qin-style) | implausible wall volume change | same segs |

Skip: strain (SCMR declines to endorse; global Ecc CoV ~20 %, radial ~33 %, Morton 2012), PFR/PER
(not guideline-standard; 83 ms frames too coarse), index beat, cardiac output (= SV × HR).

**Blocker:** `ef_dice.py score` keeps only ED/ES summaries and the sbatch deletes the seg work dir, so
every all-frame metric needs either a small change there (evaluation/ is human-curated — user's call)
or a `tools/` scorer run on kept segmentations.
