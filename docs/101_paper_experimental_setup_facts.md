# 101 — Paper §4.1 "Experimental Setup": code-verified fact sheet for the writing agent

> **TL;DR & takeaway** (2026-09-14, HEAD `bfcfb80`, three independent fact-gathering passes over
> `training/splits/`, `evaluation/`, `baselines/`, `sbatch/train_final_518.sh`, and the live
> `scratch/logs/210823094_final518_base_curated898` run.)
>
> This doc is everything a writing agent needs for the three Setup paragraphs of `_paper/sections/experiments.tex`
> — **Data and evaluation splits**, **Baselines and evaluation metrics**, **Implementation details** —
> with every number traced to a file, a doc, or a command. It is *not* a results doc: no result numbers
> here are final (§5 lists what is still PENDING).
>
> **The five facts the paragraphs must carry:**
> 1. Cohort = **898 misalignment-curated gated cine subjects** from 5 public sources (CMRxRecon 2023/24/25,
>    ACDC, M&Ms-1), split **628 / 90 / 180** stratified by (source, vendor, centre, pathology); **Canon** and
>    **CMRx25 Center012 (Philips)** are unseen-vendor holdouts (val/test only). **MIITT (13, paired
>    gated + real-time)** and **OCMR (8)** are eval-only, never trained on.
> 2. Every method scores on the **same frozen per-subject bundle**; the `scatter/` variant hands the
>    classical baselines the **byte-identical** one-frame-per-slice, breathing-corrupted stack VGGT sees
>    (verified 6e-8). Baselines get the heart ROI, nominal slice poses, per-subject slice thickness — no
>    GT poses, no reference frame (Dangi gets the reference-plane *index* as its anchor).
> 3. **NCC is the primary image metric** (intensity-gauge invariant); PSNR/SSIM are secondary; all are
>    computed after a per-phase **6-DOF NCC registration** to GT, with deconvolving baselines additionally
>    **PSF-blurred** at the subject's slice thickness. Cardiac function (EDV/ESV/EF/LVM) and Dice/HD95 come
>    from **one nnU-Net (Task114, M&Ms) applied to every arm *and* GT on the same heart crop**.
> 4. Final recipe: VGGT-1B warm start; **only the z-embedder is new** (8k params); DINOv2 ViT-L encoder
>    frozen (304M), 637M trainable; AdamW 5e-5, wd 0.05, 5 % warm-up → cosine; **300 epochs × 628 steps**,
>    batch 1, bf16, 518² input and 518² splat; loss L1 + 0.5·heart-L1 + 0.5·observation-consistency, no
>    smoothness term; aggressive aug + breathing sim; best-val `psnr_3d_heartseg` selection; trained on
>    1× L40S, 29 GB peak, ~20 min/epoch. **Evaluation hardware:** GPU methods (ours, NeSVoR, nnU-Net)
>    on an **NVIDIA A40**; SVRTK on **16 cores of a Xeon Gold 6154**; **Dangi inference ran on CPU by
>    accident** (all 582 outputs) though it is a learned method — re-run on A40 before timing it (§2.4a).
> 5. **Not yet true on disk (do not write as fact):** the headline 518-px arms are still training
>    (base at ep 196/300); NeSVoR v2 scoring is mid-run; NiftyMIC / FC-SVR / Fetal-CMR-4D / CiNeVol /
>    Stolt-Ansó are pending or cite-only; the "~0.1 s per volume" inference number is unmeasured;
>    baseline breathing-EPE, v2 identity floors, and any GT-free real-time number do not exist yet.

Companion docs: **docs/99** (project explanation + Method-section guidance), **docs/97** (split), **docs/96**
(curation), **docs/98** (eval pipeline), **docs/88** (metric hierarchy), **docs/83** (registration + PSF),
**docs/100** (Dangi), **docs/87** (real-time inference). This doc deliberately repeats the few numbers from
those that the paper needs, so the writer does not have to open them.

---

## 0. How to use this doc

- Each of §1–§3 maps to one Setup paragraph. Each opens with **"What the paragraph must say"**, then the
  facts with sources, then **"Do not write"** traps.
- §4 is a drop-in LaTeX draft of the three paragraphs with `\todo{}` markers where a number is PENDING.
- §5 is the PENDING list. **Anything in §5 must appear in the paper only as a placeholder.**
- Source notation: `file:line` (repo HEAD `bfcfb80`), `docs/NN §x`, or `[pandas over manifest.csv]` (the
  script that produced the split tables is reproducible: `micromamba run -n svr python -c "..."` over
  `training/splits/manifest.csv`, column `split_curated_v2`, and agrees with parsing
  `pooled_curated_v2.txt` for all 898 ids).

---

## 1. Data and evaluation splits

### 1.1 What the paragraph must say
Five public gated breath-hold bSSFP SAX cine sources; 898 subjects after misalignment curation; 628/90/180
stratified; two unseen-vendor holdouts; two fully held-out datasets (MIITT, OCMR); native slice geometry
preserved (D 6–20, pitch 5–12 mm); T = 12 phases, ED = phase 0; all data LPS, apex at z0. Curation, the
stratification rule, and preprocessing go to the appendix with one sentence each in the main text.

### 1.2 The live split: `training/splits/pooled_curated_v2.txt`

**Totals:** 898 = **628 train / 90 val / 180 test** (69.9 / 10.0 / 20.0 %). [`pooled_curated_v2.txt` header
lines 1–9; docs/97 TL;DR; pandas]

**Per source × split** [pandas]:

| source | train | val | test | total | pathology (all splits) |
|---|---|---|---|---|---|
| CMRxRecon2023 | 102 | 13 | 26 | 141 | 141 healthy |
| CMRxRecon2024 | 154 | 19 | 38 | 211 | 211 healthy |
| CMRxRecon2025 | 172 | 23 | 46 | 241 | 29 healthy / 206 diseased / 6 unknown |
| ACDC | 84 | 11 | 21 | 116 | 28 healthy / 88 diseased |
| M&Ms-1 | 116 | 24 | 49 | 189 | 52 healthy / 137 diseased |
| **all** | **628** | **90** | **180** | **898** | 461 healthy / 431 diseased / 6 unknown |

Diseased share per split: train 47.0 %, val 51.1 %, test 50.0 % [docs/97 §3].

**Per vendor × split** [pandas]:

| vendor | train | val | test |
|---|---|---|---|
| Siemens | 459 | 59 | 112 |
| UIH | 83 | 8 | 20 |
| Philips | 53 | 10 | 20 |
| GE | 33 | 4 | 9 |
| **Canon** | **0** | 9 | 19 |

**Unseen-vendor holdouts** (never in train, val:test = 1:2): **Canon** (M&Ms centre 5, 28 subjects →
0/9/19) and **CMRx2025 Center012, Philips Ingenia CX** (9 subjects → 0/3/6). [`pooled_curated_v2.txt`
lines 4–5; docs/97 §2 rule 1] ⚠️ Philips as a *vendor* is **not** unseen (M&Ms centres 2/3 Philips are in
train, 53 subjects); only the Center012 **scanner/centre** is a holdout. Write "unseen scanner" or
"held-out centre", not "unseen vendor", for Center012. Canon is a genuinely unseen vendor.

**Centres:** ACDC (Dijon, Siemens); CMRx2023/2024 (Fudan, Siemens Vida 3T); CMRx2025 (7 centres:
Siemens Aera/Sola/CIMAX/Prisma/Vida, UIH umr670/680/780/790/880, Philips Ingenia CX; 1.5 T 77 subjects,
3 T 164); M&Ms centres 1–5 (Siemens/Philips/Philips/GE/Canon). [manifest `scanner_model`, `centre`,
`field_strength`; pandas]

**Field strength:** CMRx2023/24 3 T; CMRx2025 1.5 T (59/5/13) + 3 T (113/18/33); ACDC 1.5 T Aera + 3 T Trio
(per-subject unknown, `scratch/data/ACDC/README.md:24`); M&Ms **not recorded in repo** (cite Campello
et al. 2021). [pandas; UNVERIFIED for M&Ms]

**Pathology detail** (train/val/test) — ACDC: NOR 50/11/19, DCM 56/5/19, HCM 45/8/10, MINF 13/3/5, RV
10/2/5. M&Ms: HHD 9/2/1, ARV 5/1/2, LVNC 0/1/1, IHD 2/0/0, AHS 1/1/0, Other 9/1/8 (+ NOR/DCM/HCM). CMRx2025
(free text, multi-label): HCM 18/4/1, CAD 8/2/2, PAH 10/0/4, DCM 9/1/0, MI 7/0/2, myocarditis 4/0/2,
congenital 3/2/2. [pandas `pathology_detail`]

**Demographics** (only CMRx2025 + M&Ms carry them, n = 273/45/93): mean age 49.7/47.5/47.9 y; M/F
160/113, 29/16, 51/42. [pandas]

**Slice count D per split:** train 6–20 (median 10), val 6–18 (median 11), test 6–20 (median 10). [pandas `n_z`]

**How the split was built** [docs/97 §2; `tools/build_curated_split_v2.py`, seed 42]:
1. Holdouts (Canon, Center012) never train; split val:test 1:2 by largest remainder.
2. Remaining 861 stratified by `(source, vendor, centre, pathology_label)`; section totals fixed at
   70/10/20 = 628/90/180; per-stratum quotas by largest remainder; strata with < 3 subjects go train-only.
3. v1 membership kept where quota allows (134 subjects changed section vs v1).

**Why re-split** [docs/97 §1]: the earlier `pooled.txt` honoured the ACDC/M&Ms *official* test sets, so
test was 60 % pathology and held all Canon and most GE while val looked like train (Canon 0/3.3/13.6 %,
GE 2.3/4.3/15.2 % train/val/test) — val could not predict test. No published SVR baseline uses those
official test sets, so nothing was lost. Result: per source/centre, val and test agree within ~1 pt.

### 1.3 Per-source acquisition facts

| source | sequence / gating | field | native in-plane | thickness / gap / **pitch** | D (v2) | T native → 12 | source |
|---|---|---|---|---|---|---|---|
| CMRxRecon2023 | TrueFISP bSSFP, retro-ECG-gated segmented breath-hold, ~1 BH/slice | 3 T | 1.5 mm acquired | 8 / 4 / **12 mm** (139), 10 (2) | 6–13 | 12 on disk (challenge k-space) | `scratch/data/CMRxRecon2023/README.md:37,53–72` |
| CMRxRecon2024 | same; healthy volunteers | 3 T | 1.34–1.64 mm | 8 / 4 / **12 mm** | 8–14 | organizers standardised 19–28 → 12 | `CMRxRecon2024/README.md:46–55` |
| CMRxRecon2025 | cine k-space; multi-centre/vendor/disease | 1.5 T / 3 T | UNVERIFIED per centre | thickness 6–10 by centre, +4 mm gap ⇒ **12 mm** (178) / **10 mm** (63, Prisma relabeled); Center004 Aera 12-vs-14 unresolved | 6–18 | 12 on disk | docs/54 §10c; manifest |
| ACDC | SSFP cine, breath-hold | 1.5 T / 3 T | 1.37–1.68 mm | 5 mm + 5 gap ⇒ **10 mm** (103), 5 (11), 6.5/7 (2) | 6–20 | 13–35 → 12 ED-anchored | `scratch/data/ACDC/README.md:24–43` |
| M&Ms-1 | gated BH SAX cine; 4 vendors, 5 centres | UNVERIFIED | UNVERIFIED | **10 mm** (146), 9.6 (19), 8.0 (15), other (9); measured centre-to-centre | 8–20 | 20–34 → 12 | `scratch/data/MNMs/README.md:15`; docs/58 §2 |

Pooled: pitch 5–12 mm, D 6–20. **All sources are ECG-gated breath-hold cine; the asynchronous acquisition
is simulated from them** (CLAUDE.md, docs/99 §1.5). `ACDC_patient124` (D = 21) is excluded everywhere (slot
cap 20).

### 1.4 Held-out, evaluation-only datasets

**MIITT** (`training/splits/miitt_eval.txt`, 13 subjects, all `[val]`; docs/78 §2, docs/87 §1): U-Michigan
paired ECG-gated + real-time free-breathing SAX cine. 10 volunteers + 3 patients (ARVC, HCM,
cardiomyopathy + AFib). **Gated arm:** 1.5 × 1.5 mm, dz = 10 mm (8 + 2 gap), 30 native phases → 12,
D = 12–15. **Real-time arm:** golden-angle spiral, R = 12, **2.3 × 2.3 mm, 25 ms/frame**, ~180 frames ×
D slices, dz = 10 mm; **no GT** (retrospective subsampling impossible) → qualitative only. Never in v2
training. ⚠️ pre-v2 `pooled_miitt.txt`-series checkpoints trained on Volunteer1–5; the final-518 arms
did not. Vendor/field strength: UNVERIFIED in repo.

**OCMR** (`ocmr_eval.txt`, 8 gated SAX stacks, all `[val]`; docs/79; `tools/convert_ocmr_to_12phase.py:23–33`):
fully-sampled ECG-gated breath-hold SSFP, RSS recon; 2 × 3 T + 6 × 1.5 T (from series ids); in-plane
1.98–2.25 mm; pitch 7.8–10 mm (measured from slice positions); D 10–14; 182 native frames → 12. Never
trained on.

**Other real-time data:** Göttingen radial RT bSSFP (68 volumes, 1.6 mm, 6 mm slices, ~33 ms/frame,
RT-NLINV via BART; docs/16) exists but recent eval docs (79/87) use only MIITT RT. **PENDING** whether it
appears in the paper.

### 1.5 Curation (docs/95, docs/96)

- Measured: inter-slice in-plane breath-hold misalignment on all 1337 pooled subjects × 12 phases, two
  agreeing estimators (nnU-Net epicardial-centroid offset; NCC registration to neighbouring slices);
  severity = max of spike/step. Median 1.2 mm; **38.3 % of subjects have a > 5 mm discontinuity**, 21.3 % > 8 mm.
- Rule: keep ≤ 5 mm (clean + mild), drop 5–8 (moderate) and > 8 (severe); 18 unmeasurable short stacks
  reviewed by eye; **114 by-eye overrides** (24 excluded, 90 included).
- Result: **1337 → 898 (67 % kept)**; per source kept ACDC 116/148, CMRx23 141/195, CMRx24 211/294,
  CMRx25 241/359, M&Ms 189/341. Applied to train, val and test alike.
- **Uncurated ablation split** `pooled_uncurated_ablation.txt`: 1066/90/180 = v2 + 438 curated-out
  subjects returned to train; val/test byte-identical to v2. (Trained at 224 px, see §3.8.)

### 1.6 Preprocessing (one sentence each; `training/data/preprocess.py`)

| step | value | line |
|---|---|---|
| Orientation | `Orientationd(axcodes="LPS")`; ACDC/M&Ms brought to LPS losslessly at conversion (`tools/convert_to_sax_layout.py:25–29,124–151`) | 266 |
| In-plane | resample to **1.4 mm** (`Spacingd(pixdim=(1.4,1.4,0.0))`), crop/zero-pad to **256²** | 39–40, 44, 267–270 |
| Through-plane | **never resampled or padded**: native D and pitch kept (`TARGET_SHAPE=(256,256,-1)`) | 44 |
| Intensity | clip-and-scale to phase-0's **0.5 / 99.9 percentiles** over in-FOV voxels, applied to all 12 phases | 12, 70–88, 242–243 |
| Phases | `NUM_PHASES = 12`; CMRx already 12 on disk; ACDC/M&Ms/MIITT/OCMR pick 12 of T native, ED-anchored nearest frame, no interpolation (`convert_to_sax_layout.py:172–174`) | 46 |
| ED | phase 0 for all 898 (manifest `ed == 0`; docs/58 §13) | — |
| Slice order | apex at z0 for every subject (standardised on disk 2026-07-31, docs/58 §10a) | — |
| Physical z | `z_norm = z_mm / 90` (`Z_HALF_MM = 90.0`) | 52 |

Why native z (one clause for the appendix): forcing a shared z grid caps even a Δ = 0 identity splat at
25.4 dB vs exact reconstruction (docs/58).

### 1.7 Data-engineering caveats (appendix sentences)

1. **In-house ESPIRiT/SENSE reconstruction** of all CMRxRecon k-space (2023: 196, 2024: 294, 2025: 360
   subjects) after fixing an ESPIRiT-calibration input-domain bug (docs/54).
2. **True slice pitch = thickness + gap**: CMRx NIfTI headers carried thickness (8 mm); relabeled to 12 mm
   (docs/27); CMRx2025 measured per centre (docs/54 §10c).
3. **Odd-Z cyclic roll** in every CMRx challenge release (464/466 odd-Z stacks rolled by −1) fixed on
   disk (docs/56).
4. **Slice order** standardised apex-at-z0 by flipping arrays (never affines) on 893/1343 subjects
   (docs/58 §10). "LPS" is a stamped convention; SAX stacks are double-oblique.
5. 6 cross-source duplicates excluded; ACDC/M&Ms official splits not honoured (docs/97 §1).

### 1.8 Do not write
- "Unseen vendor Philips" — only Center012 is held out; Philips is in train via M&Ms.
- Any per-subject field strength for ACDC or M&Ms; any M&Ms in-plane resolution (not in repo).
- "Real-time free-breathing training data" — there is none; all training is simulated from gated cine.
- "Official ACDC / M&Ms test sets" — v2 does not use them.
- Pre-2026-07-31 numbers of any kind (pre-flip, pre-native-z; docs/58 §10b).

---

## 2. Baselines and evaluation metrics

### 2.1 What the paragraph must say
(a) Every method receives the **same frozen, breathing-corrupted, one-frame-per-slice stack** (the
`scatter/` bundle), the heart ROI, nominal slice poses and slice thickness — nothing else. (b) Which
baselines: SVRTK, NeSVoR (single-stack SVR), Dangi (learned slice alignment), plus †-flagged dense-input
methods if run. (c) Metrics: NCC primary (why), PSNR/SSIM secondary, all after per-phase rigid
registration + PSF for deconvolvers; EDV/ESV/EF/LVM and Dice/HD95 from one nnU-Net on the heart crop
for every arm including GT; demeaned breathing EPE (ours only); runtime.

### 2.2 The frozen evaluation bundle (`evaluation/src/engine/build_inputs/pooled.py`)

Per subject (`pooled.py:12–20, 244–292`): `gt/gt_t{00..11}`, `clean/stack_t*` (= GT planes), `breath/stack_t*`
(each plane resliced by its own frozen breath displacement), **`scatter/stack_t*`**, `mask.nii.gz` (FOV),
`mask_heart.nii.gz`, `heart_seg.nii.gz`, `manifest.json` (T, D, dz, seed, full `disp_dhw_mm`, the scatter draw).

- **Breathing** is drawn by the *same* `sample_resp_disp` and config the model trained on (18.8 ± 7.35 mm
  SI, AP 0.35, tilt U(0°,45°), one phase per plane), seeded `sha256("<source>/<subject>")`
  (`pooled.py:106–108, 235–241`; `default.yaml:379–396`). Written once, read by every arm
  (`evaluation/README.md:90–92`).
- **`scatter/` = same input for everyone**: `draw_scatter` builds a one-subject `MRIDataset` exactly as
  `run_vggt.make_dataset` (reference slot, one frame per slice, integer z, `seq_index = seed`) and stores
  `scatter/stack_t{k}[z] = breath/stack_t{p_z}[z]` for z ≠ ref, `= breath/stack_t{k}[ref]` at the reference
  plane. VGGT's `images_splat` equals the scatter plane to **5.96e-8** on D = 6..20; 291/291 bundles carry it
  (`pooled.py:141–202`; docs/98 §7).
- Two baseline input regimes: `INPUT=gated` (all D planes at the target phase — strictly *more*
  target-phase information than VGGT gets; docs/83 §5) and **`INPUT=scatter`** (byte-identical to VGGT).
  The paper's headline is the scatter regime; the gated regime is a favourable-to-baselines sanity row.
- **Cohort:** val **111** (ACDC 11, CMRx23 13, CMRx24 19, CMRx25 23, M&Ms 24, MIITT 13, OCMR 8); test
  **180** (ACDC 21, CMRx23 26, CMRx24 38, CMRx25 46, M&Ms 49; MIITT/OCMR are val-only). [split files +
  bundle manifests; docs/98 TL;DR]
- **Heart ROI:** nnU-Net whole-heart segmentation, union over 12 phases, hole-fill/close, dilate 6 mm
  in-plane and ±1 slice, clamped to FOV (`tools/nnunet_mnms_eval/build_heart_roi.py:35–49`); copied as
  `mask_heart` (`pooled.py:102`).

### 2.3 What each baseline receives
Heart ROI (`-mask` / `--stack-masks --sample-mask`), nominal slice poses from the NIfTI affine, per-subject
slice **thickness** (`THICK`: cmrx2023/24 = 8; cmrx2025 = dz − 4; acdc = 5 if dz = 10 else dz; mnms = dz;
miitt = 8; ocmr = ISMRMRD header) — `evaluation/src/engine/run_baselines.py:8–18`. **No GT poses, no
reference frame.** Dangi receives the reference-plane *index* as its alignment anchor (`run_dangi.py:58,72–74`).
Baselines run **12 independent 3D reconstructions, one per phase** (`run_svrtk3d.sh:29,53`;
`run_nesvor.sh:36,70`). VGGT sweeps the reference slot over t = 0..11 with the companions fixed at the frozen
scatter phases: one forward + splat per phase (`run_vggt.py:215–240`).

### 2.4 Baseline table

† = requires dense temporal input (every frame of every slice); evaluated on its native full-cine
reconstruction, so it sees more per subject than any single-stack method (`experiments.tex` footnote).

| Baseline | bib key | Implementation | Command / key flags | Our modifications | Runtime | **v2 status** |
|---|---|---|---|---|---|---|
| **SVRTK** (3D, per phase) | `kuklisovamurgasova2012reconstruction` | `mirtk reconstruct`, Singularity `scratch/fetal_cmr_4d/sif/svrtk.sif` (upstream commit UNVERIFIED) | `-thickness $THICK -mask mask_heart -resolution 1.4 -iterations 4 -no_robust_statistics` | `-no_robust_statistics` (robust EM flags every slice of a single stack as outlier: 14–18 → 26.9 dB clean) | 122–145 s / subject (12 phases), Xeon Gold 6154 ×16 | **RUN** gated + scatter, val + test, scored incl. EF/Dice |
| **NeSVoR** | `xu2023nesvor` | native `nesvor-t2` env, upstream v0.5.0 (730ddaa) + 10-line torch-2 CUDA port (docs/90) | `--registration none --stack-masks/--sample-mask mask_heart --thicknesses $THICK --output-resolution 1.4`; n-iter 6000 | `--registration none` (single stack); mandatory `--sample-mask` (docs/32); output gauge ~700 → stack-anchored self-norm at scoring | ~158–174 s / phase, ~31 min / subject, A40 | **RUNNING** (array 61109592; `nesvor` 250/291, `nesvor_scatter` 111/291 on disk); **no metric JSON yet** |
| **NiftyMIC** | `ebner2020automated` | `renbem/niftymic` v0.9 container | `niftymic_reconstruct_volume` defaults | SimpleITK 2 patch; +1.0 pedestal → self-norm | 18–25 min / subject CPU (n = 2 pilot, docs/29) | **NOT BUILT for v2** (no engine runner); no paper row |
| **FC-SVR** | `young2024fcsvr` | `baselines/fcsvr_cardiac` fork, stage 1 only | trained on old `cmrx24only` 235/29/30 | single SAX orientation; translation-only; 8/12 mm slabs as 4 × 3 mm planes; rejects dz ≠ 12; result GT-pose-normalised (must be disclosed) | UNVERIFIED | **PENDING** v2 retrain + runner (docs/98 §11) |
| **Dangi et al. 2018** | `dangi2018cine` | 3rd-party PyTorch reimpl (`sin021004/dangi-cmr-alignment` d42d338), audited faithful (docs/100 §3); **retrained by us** on v2 train (628, 100 ep, LV-centroid targets) → `checkpoints/dangi_pool_v2/best.pt` | `run_dangi.py --input scatter\|gated --anchor reference` | anchor = reference plane centre (paper: image centre / mean) | 4.5–10 s / subject CPU | **RUN** gated + scatter, val + test, image metrics scored; `_ef/dangi*.json` PENDING (jobs 61125839/40) |
| **Stolt-Ansó 2024** | `stoltanso2024intensity` | none | — | — | — | **cite-only**; not built |
| **Fetal CMR 4D†** (van Amerom) | `vanamerom2019fetal` + `akesson2025retrospective` (sync step) | SVRTK `mirtk reconstructCardiac` (same `svrtk.sif` as SVRTK), `evaluation/src/engine/run_fetal4d.sh`; self-gating `fetal4d_gate.py` = LV-area ED anchor (Åkesson 2025 rule, nnU-Net Task114 on the input frames) — docs/105 | `-iterations 4 -rec_iterations 10 -rec_iterations_last 20 -resolution 1.4 -numcardphase 12`, robust stats ON (decided, docs/105 §5a); resolution 1.4 not author's 1.25, to match svrtk3d/nesvor/niftymic (docs/105 §5b); `-rrinterval 1.0` nominal | **input = `rolled/`**: the 12-phase breath stack with each plane's phases circularly rolled by a frozen unknown offset (12 frames/slice, unknown phase — a privileged-input row vs VGGT's 1 frame/slice); paper's overlap-based inter-slice sync replaced by the per-slice LV-area ED anchor (SAX slices don't overlap, docs/34); 12 output phases (= GT) not 25; nominal R-R; `mask_heart_pad10` recon ROI; **three input defects found + fixed 2026-09-16 (docs/105 §5c):** frame-duration header (1.0 s default, then a `sec` unit tag MIRTK reads ×1000 — both gave a whole-cycle temporal window), the container binary's `-cardphase` off-by-one (worked around with a 710-token list, slice-0 ED first), and output phase 0 rolled to the GT's seg-derived ED (docs/105 §5d) | median 492 s / subject on 16 cores, 48 GB, `standard` (n = 180, provenance.txt per subject); no speed-up beyond ~4 threads, 16 allocated to match SVRTK/NiftyMIC | **TEST DONE + SCORED** (jobs 61280727/61285878, 2026-09-16; `metric_results/test/{<ds>,_ef}/fetal_cmr_4d.json`): 22.64 dB unit-peak / SSIM 0.620 / NCC 0.810, **EF MAE 6.6 (bias −1.3, r 0.78)**, LV Dice ED/ES 0.848/0.782 — best classical arm on images, best of ALL arms on EF; every `final518_*` arm wins images (+1.6–2.0 dB) and Dice but under-contracts (EF MAE 10.2–13.5, bias −8…−12) — docs/105 §8. Val not run |
| **CiNeVol†** | `vogt2026cinevol` | none | — | — | — | **cite-vs-run UNDECIDED** (docs/98 §11) |
| **Identity / no-correction floor** | — | `evaluation/src/score/dice_floors.py` (label arithmetic; `cardiac` = phase scatter only, `full` = + breathing) | — | — | — | pre-v2 only (docs/86: LV Dice 0.732 ED / 0.664 ES, n = 144); **NOT recomputed for v2** |

### 2.4a Evaluation hardware (state this explicitly in the paper)

All evaluation runs on the U-Michigan Great Lakes cluster. Recorded per run in each output's
`provenance.txt` / `timing.json`, verified by grepping every v2 output on disk (2026-09-14):

| task | hardware | allocation | evidence |
|---|---|---|---|
| **GPU methods: VGGT inference, NeSVoR, nnU-Net segmentation** | **NVIDIA A40 (48 GB)**, `spgpu` partition | 1 GPU, 5 CPUs, 48 GB RAM | `sbatch/eval_pooled_val.sh:3–8`, `eval_baseline_nesvor_v2.sh:3–8`; NeSVoR `provenance.txt` "gpu: NVIDIA A40" on 387/389 v2 outputs (2 pilot outputs on V100, pre-v2 job 58259807 — exclude); `evaluation/README.md:191` |
| **CPU method: SVRTK** | **Intel Xeon Gold 6154 @ 3.00 GHz** (36-core nodes), `standard` partition | 16 CPUs, 48 GB, J = 8 phases × OMP = 2 | `eval_baseline_svrtk_v2.sh:3–7`; SVRTK `provenance.txt` "cpu model" on 583/584 outputs (1 on Xeon Platinum 8358 — flag or re-run) |
| **Dangi (learned, but inference ran on CPU)** | trained on L40S (`sbatch/train_dangi_baseline.sh:3–4,45`); **all 582 v2 inference outputs stamp `device: "cpu"`** because the runner defaults to CUDA only when available (`run_dangi.py:103`) and the eval jobs ran on CPU nodes | — | `dangi*/timing.json` |

⚠️ **Dangi timing is a CPU number (4.5–10 s/subject).** On an A40 a per-slice CNN would be well under 1 s,
so the CPU figure understates it relative to ours. Decide: re-run Dangi inference on `spgpu` (~1 h for
291 subjects, `--device cuda`) so every learned method is timed on the same GPU, or keep CPU and state it
in the table footnote. Recommended: re-run.
| Training (for §3) | NVIDIA L40S 48 GB, `spgpu2` | 1 GPU | §3.7 |

⚠️ `run_vggt.py`'s `timing.json` records `model_load_sec` and per-phase ms but **not the GPU name**;
the A40 attribution for VGGT rests on the sbatch partition, not on a per-run stamp. Worth adding
`torch.cuda.get_device_name()` to the timing record before the final eval pass.

Timing fairness: GPU methods (NeSVoR, VGGT) timed on the same A40/spgpu class (docs/98 §8). VGGT per-phase
wall on CMRx24 val (curation-ablation **224-px** arm): **~175 ms per volume** steady-state (first phase
~570 ms, warm-up), **3.6 s per 12-phase cine**, model load 17.7 s excluded (`run_vggt.py:233–238,389–391`;
`…/vggt_curation_curated_ep300/timing.json`).

**518-px arm — MEASURED 2026-09-18** (from the final518 five-arm sweep's `timing.json` files, `evaluation/
src/engine/run_vggt.py`'s per-phase `torch.cuda.synchronize()`+`perf_counter()` timer; A40, bf16 autocast,
eager PyTorch — no `torch.compile`/TensorRT/ONNX at inference). Per-volume time (forward pass + splat)
scales almost perfectly linearly with the subject's own slice count `D` (`one_frame_per_slice: true` ⇒
`S == D`, so attention cost tracks slot count): across all 291 (subject × final518-arm) `timing.json`s
spanning 7 eval datasets, steady-state (first-phase warm-up excluded) `ms/volume ≈ -210 + 122.6·D`
(r = 0.996, D range 6–20): **~530 ms at D=6, ~1.0 s at D=10 (the modal subject), ~2.2 s at D=20**. All five
final518 arms (base/diff1000/hw2/motion10/motion10_hw2) are indistinguishable (same 518×518 architecture,
loss weights only differ). First-phase warm-up adds ~200–250 ms; model load ~14–15 s (excluded, one-time).
⚠️ **The oft-repeated "~100 ms per volume on an A40" (docs/87:29, docs/99:260) is WRONG for the 518-px
arm** — it undershoots even the smallest (D=6) subject by ~5×; it may have been extrapolated from the
224-px number without accounting for image-size or D-scaling. SVRTK example
(CMRx24_Test_P001, scatter): 141 s end-to-end for 12 phases at J = 8 (mean 78 s per phase single-threaded
equivalent); NeSVoR ~158–174 s per phase single-fit on A40.

### 2.5 Metrics exactly as implemented (`evaluation/src/score/`)

| Metric | Definition | Source |
|---|---|---|
| **Scoring ROI** | `mask_heart ∧ mask` (heart ROI ∩ native FOV); FOV fallback if no heart mask | `image_metrics.py:294–302` |
| **Registration** (every arm, incl. VGGT) | Per (subject, arm, phase): 6-DOF rigid (mm translation, Euler deg about mask centroid), objective = masked NCC to that phase's GT; coarse translation grid ±8 mm / 2 mm (729) → rotation ±8° / 4° (125) → Adam 200 steps, cosine lr 0.3, box-bounded. Recon is *sampled* at transformed GT voxel centres; the coarse z grid is never resampled. For VGGT the fit is expected ≈ 0 | `pose_psf.py:269–329`; docs/83 §3; docs/98 §3 |
| **PSF** (deconvolvers only) | `PSF_METHODS = {svrtk3d, nesvor, niftymic, fetal_cmr_4d, cinevol}` (+`_scatter`): normalised Gaussian on the recon's grid, FWHM = per-subject slice **thickness** through-plane, 1.2 × acquired in-plane voxel in-plane; then sampled. Not applied to VGGT (already blurrier than GT in z) | `pose_psf.py:63–66,125–189,332–336`; docs/83 §4.4 |
| **Raw column** | unregistered, unblurred trilinear read also reported (`{var}_*_raw_mean`) | `image_metrics.py:333–365` |
| **Intensity gauge** | VGGT/SVRTK scored as-is (SVRTK −1 sentinel → 0). NeSVoR/NiftyMIC (`SELF_NORM_METHODS`): two-point map (recon p0.5, p99.9) → (input-stack p0.5, p99.9) over coverage ∩ FOV, computed once from the raw read, clamp [0,1]; anchored to `scatter/` stacks for `_scatter` arms | `image_metrics.py:91–145,347–351`; docs/88 §1–2 |
| **NCC — PRIMARY** | Pearson correlation over ROI voxels per phase, mean over 12 phases. Primary because invariant to each baseline's arbitrary affine intensity gauge; standard in the SVR literature | `image_metrics.py:195–205`; docs/88 §3 |
| **PSNR** (secondary) | `10·log10(peak²/MSE)` over ROI, **peak = GT max inside ROI** (`psnr_unit_peak`, peak = 1, also emitted); gauge-dependent and blur-favouring | `image_metrics.py:148–168` |
| **SSIM** (secondary) | skimage `structural_similarity`, Gaussian 11 × 11, σ = 1.5, `data_range = 1`, `use_sample_covariance = False`, computed **per 2-D SAX slice**, SSIM map averaged over ROI voxels (a 3-D window would span ~80 mm in z). Supersedes every pre-docs/98 "SSIM" | `image_metrics.py:171–192`; docs/98 §4 |
| Cohort statistic | mean ± std over valid subjects | `aggregate.py:44–51,217–255` |
| **Heart crop before segmentation** | every cine (GT, VGGT, baselines) × `mask_heart`, because the segmenter is measurably worse on crops (ACDC expert labels, n = 40: LV/MYO/RV ED Dice 0.952/0.871/0.918 → 0.936/0.818/0.868) and SVRTK/NeSVoR only reconstruct inside the ROI by protocol — uniform handicap, no bias | `ef_dice.py:73–93`; docs/98 §5 |
| **Segmenter** | nnU-Net v1 (1.7.1) **Task114 (M&Ms) 2d, trainer `nnUNetTrainerV2_MMS`**, isolated `nnunet` env; applied to the scored (registered / PSF'd / gauged) cines; GT segmented once per subject (`seg_gt/` cache). **Weights:** the authors' official pretrained release, Zenodo record 4288464 (`https://zenodo.org/record/4288464`, `Task114_heart_MNMs.zip`, md5 verified, docs/15), 2d config only, five folds, no fine-tuning; labels 1 = LV, 2 = MYO, 3 = RV. **Cite:** Isensee et al. 2021 Nature Methods (framework, doi 10.1038/s41592-020-01008-z) + Full et al. STACOM 2020 (M&Ms winner, the model) | `run_seg.sh:43`; `ef_dice.py:10–13,108–115`; nnunet `download_pretrained_model.py` registry |
| **ED / ES** | from the GT LV voxel-count curve: ED = argmax, ES = argmin; predictions use their own curve extremes for volumes | `ef_dice.py:336,212–226` |
| **EDV / ESV / EF / LVM** | voxel counting: EDV = max LV count × voxel mm³ / 1000 (mL); ESV = min; EF = (max − min)/max × 100; LVM = MYO voxels at GT-ED × voxel × 1.05 g/mL. RV EDV/ESV/EF also. Cohort: paired MAE vs same-subject GT; EF also slope (polyfit) and Spearman ρ | `ef_dice.py:212–235,364–384` |
| **Dice / HD95** | LV, MYO, RV at GT-ED and GT-ES (pred at the same phase index); HD95 = 95th pct symmetric surface distance (mm, EDT with header spacing); cohort mean per structure × phase. ⚠️ "Mean Dice" in the draft table is **not computed by code** — define it (mean of the 6 cells, or of LV/MYO/RV at ED) before filling | `ef_dice.py:238–260,385–392` |
| **Breathing EPE / slope / Pearson** (VGGT only) | at ED: per slot (all S incl. reference; FOV gate `img > 0.05`): pred Δz = mean predicted through-plane Δ × 90 mm; applied = frozen `disp_dhw_mm[z,0]`. EPE = mean\|pred − applied\|; **demeaned EPE** = mean\|err − mean(err)\| (removes the pose gauge); slope = polyfit(applied, pred); Pearson r | `run_vggt.py:253–278`; `aggregate.py:54–75` |
| Baseline EPE | **not built** (would need `.dof` / NeSVoR poses → resp_diag); baseline JSONs carry `resp_* = null` | docs/98 §11 |
| **Runtime** | VGGT: `timing.json[breath].total_sec` (12-phase sweep, forward + splat, cuda-synced, excl. model load); Dangi: same format, CPU; SVRTK/NeSVoR: end-to-end wall for all phases (J parallel). Surfaced as `recon_wall_sec` | `run_vggt.py:233–238,449`; `run_svrtk3d.sh:117–120`; `aggregate.py:78–91` |

### 2.6 Where per-source / OOD numbers come from
One JSON per (split, source, arm): `evaluation/metric_results/{val,test}/<src>/<arm>.json`, with `all`
(cohort mean/std) and `per_subject` rows (image metrics, Dice/HD95/EF/EDV/ESV/LVM/RV, `resp_*`,
`recon_wall_sec`, pose, psf). Source is the directory key, so a per-domain table is a file walk
(`aggregate.py:270–284`). Cross-cohort EF: `metric_results/<split>/_ef/<arm>.json`. Ranking helper:
`evaluation/src/analysis/compare_table.py <ds> --metric breath_ncc`.

### 2.7 Real free-breathing real-time evaluation (docs/87; `evaluation/src/engine/run_vggt_rt.py`)
One 3-D volume per **each of ~180 real RT frames** (MIITT, spiral, 2.3 mm, 25 ms/frame) by sweeping the
reference slot; companions = one fixed real frame per z-plane. **Qualitative only; no GT; nothing scored;
no GT-free self-consistency number defined.** Pre-v2 runs exist for Volunteer1/2/3 + AFib on 224-px
checkpoints; the volumes directory is currently archived. **PENDING** on the v2 518-px checkpoints.

### 2.8 Do not write
- "Baselines receive the same input" without saying *which* regime (scatter) — the gated regime gives
  them more.
- "NeSVoR / NiftyMIC / FC-SVR / Fetal-4D / CiNeVol results" — none exist on v2 yet.
- Any pre-docs/98 SSIM (was a global statistic, not SSIM) or pre-docs/88 NeSVoR PSNR (gauge bug).
- "Identity floor" numbers from docs/86 (pre-v2, n = 144, different split).
- A real-time EF or any real-time number.
- "Mean Dice" until its definition is fixed.

---

## 3. Implementation details

Final recipe = `sbatch/train_final_518.sh` **ARM=base** over `training/config/default.yaml`. Live run
`scratch/logs/210823094_final518_base_curated898` (SLURM 61000966, wandb `zm6w1ft6`), config verified from
its `run_meta.jsonl` (git `77f968d`). **Status 2026-09-14: epoch 196/300 — every "final" number is interim.**

### 3.1 What the paragraph must say
VGGT-1B initialisation, what is frozen/trained, input and splat resolution, optimiser + schedule +
epochs, loss weights, augmentation + breathing sim on, checkpoint selection, hardware + training time,
inference time. ≈ 4 sentences in the main text; the rest to the appendix.

### 3.2 Model

| item | value | source |
|---|---|---|
| Init | `./scratch/base_weights/vggt1b_base.pt` (HF `facebook/VGGT-1B`), `strict: false` | `default.yaml:244–245`; `trainer.py:430` |
| New vs VGGT-1B | **only `z_embedder`** (`ZIndexEmbedder`: `Linear(7→1024)` on `[z, sin/cos(2^iπz)]_{i=0..2}`, **8,192 params**). Reference marker **reuses** the pretrained two-row `camera_token`; the two-row `register_token` is standard VGGT | `aggregator.py:27–45,104–111,168–169` |
| Backbone | DINOv2 `vitl14_reg`, embed 1024, patch 14, 518² → 37² = 1369 patch tokens; 4 register tokens; aggregator depth 24 alternating frame/global (48 attention blocks) | `aggregator.py:76–88,121–151` |
| Head | `DPTHead(dim_in=2048, output_dim=4, activation="linear")` → 3-ch residual Δ + 1 conf (unused) | `vggt.py:27,34`; `head_act.py:49–50` |
| Params | **total 941,775,140**; **frozen 304,372,736** (`*patch_embed*` = entire DINOv2 ViT-L); **trainable 637,402,404** (aggregator 604.7M + point head 32.65M + z-embedder 8k) | `frozen.txt`, `trainable.txt` in log_dir; `default.yaml:511` |
| Gradient checkpointing | on | `default.yaml:408` |

### 3.3 Optimiser and schedule

| item | value | source |
|---|---|---|
| Optimiser | AdamW (`fused=True`), lr **5e-5**, weight decay **0.05**; betas not set ⇒ torch default (0.9, 0.999) [inferred from absence] | `default.yaml:512–517,555–558` |
| Schedule | linear warm-up 1e-8 → 5e-5 over the first **5 %** of steps, then cosine 5e-5 → 1e-8 | `default.yaml:544–553` |
| Epochs / steps | **300 epochs × 628 steps** (one exact pass over the 628 train subjects per epoch) = 188,400 steps | `sbatch:57,61`; `run_meta n_train_subjects: 628` |
| Batch | 1 subject; S = that subject's D slices (cap 20) | `default.yaml:271–274` |
| Grad clip | L2 max-norm 1.0, separately on aggregator and head param groups | `default.yaml:533–541` |
| Precision | bf16 autocast, no GradScaler; TF32 on; `torch.compile` (`dynamic=True`) on the 72 attention blocks | `default.yaml:518–520,568–571`; `log.txt:14` |
| Seed | 42 (`seed + 100·epoch` per epoch) | `default.yaml:43`; `trainer.py:157,252,709` |
| EMA | none | `trainer.py:617` |
| Checkpoint selection | best val **`metric_psnr_3d_heartseg`** (weights-only `checkpoint_best.pt`), evaluated every epoch; `checkpoint_last.pt` every epoch, archive every 50 | `default.yaml:243,250–260`; `trainer.py:538–603` |
| Interim best | 22.77 dB at epoch 194 — **INTERIM, do not quote** | `metrics.jsonl` |

### 3.4 Loss (base arm)

| term | weight / value | source |
|---|---|---|
| full-volume L1 | 1.0 | `default.yaml:421` |
| heart-ROI L1 (mean over ROI voxels) | **0.5** | `sbatch:65`; `loss.py:303–322` |
| observation-consistency (gather), gate `I > 1e-3`, zeros padding | **0.5** | `sbatch:64`; `loss.py:353–362` |
| TV / diffusion / motion-weighted L1 | 0 / 0 / 0 | `default.yaml:451`; `sbatch:63,66` |
| `splat_res` | 518 (field + native slice resampled to 518² before scatter) | `sbatch:67`; `loss.py:151–166` |
| splat gate / coverage ε / in-bounds tol | `I > 1e-3` / `1e-6` / `1e-3` voxel | `loss.py:188–189`; `splat.py:9,144` |
| `Z_HALF_MM` | 90 mm | `preprocess.py:52` |

### 3.5 Augmentation (tier `aggressive`; affine/photometric **train-only**; `training/data/gpu_aug.py`)
One draw per subject shared across all 12 phases, the content mask and the heart ROI (nearest).

| op | p | range | line |
|---|---|---|---|
| RandFlipd (W) | 0.5 | — | 190 |
| RandAffined | 0.9 | rotate ±180° in-plane, translate ±32 px | 191–197 |
| RandAdjustContrastd (γ) | 0.75 | 0.6–1.5 | 202 |
| RandBiasFieldd | 0.7 | degree 3, ±0.4 | 203 |
| RandZoomd (isotropic) | 0.5 | 0.8–1.2 | 272 |
| RandSimulateLowResolutiond | 0.4 | zoom 0.5–0.85 | 274–275 |
| RandGibbsNoised | 0.3 | α 0.5–0.75 | 277 |
| phase-encode ghosting (custom) | 0.3 | 2–5 ghosts, 15–40 % line attenuation, random axis | 286–292 |

Respiratory simulation (train **and** val; val deterministic per `seq_index`): 18.8 ± 7.35 mm SI, sin⁶,
AP 0.35, tilt U(0°,45°) per subject, one breath phase per plane (`default.yaml:378–397`;
`trainer.py:847–851`). Provenance of the constants: docs/01 (18.8 ± 7.35 = Burger & Meintjes 2013
*diaphragm* mean ± SD; sin⁶ = Lujan `cos^{2n}`, n = 3; 0.35 from Shechter 2004 / Faranesh 2013 component
ratios; tilt range = design choice for SAX obliquity).

### 3.6 Validation during training
Every epoch; 180 entries = 90 val subjects × {ED, ES} (`ef_val_sweep`), deterministic slot sampling
`random.Random(seq_index)`; breathing on, affine off; per-subject pred + GT NIfTIs dumped to
`${log_dir}/val_volumes/`; EF metric via CorSeg (`default.yaml:94–150`; `trainer.py:725,847–865`).

### 3.7 Hardware, cost, software

| item | value | source |
|---|---|---|
| GPU | **1× NVIDIA L40S 48 GB** (`spgpu2`, `jjparkcv_owned1`), no DDP | `sbatch:2–12`; `run_meta gpu` |
| Peak GPU memory | **29.0 GB** | `log.txt` |
| Wall per epoch | ~20 min (≈16.5 min train + 3.5 min val) | `metrics.jsonl` timestamps |
| Total | 68.0 h for 197 epochs ⇒ **~102 h (~4.3 d) projected for 300** — PENDING | same |
| Inference | 224-px curation arm: **~175 ms/volume** steady-state, 3.6 s/12-phase cine, model load 17.7 s. **518-px final518 arms (MEASURED 2026-09-18): ~530 ms–2.2 s/volume, scaling linearly with subject slice count D** (`ms ≈ -210 + 122.6·D`, r=0.996; ~1.0 s at the modal D=10), model load ~14–15 s. The "~100 ms on A40" figure in docs/87/99 is WRONG for 518-px (§2.4a) | docs/87; docs/99; `timing.json` |
| Model load | GPFS `torch.load` ~266 s vs ~5 s from `/tmp` (docs/50) — engineering, not a paper number | docs/50 |
| Stack | Python 3.10.14, torch 2.13.0+cu130, torchvision 0.28.0, triton 3.7.1, monai 1.6.0, numpy 2.2.6, CUDA 13.0, hydra 1.3.2, batchaug 0.1.0, fused-ssim 1.0.0 (val SSIM metric only); eval segmenter nnU-Net 1.7.1 | `micromamba run -n svr`; `run_meta` |

### 3.8 Ablation arms on disk (for §4.5 planning; **statuses as of 2026-09-14**)

| arm | override vs base | status | log dir |
|---|---|---|---|
| final518 **base** | — | RUNNING ep 196/300 | `210823094_final518_base_curated898` |
| final518 diff1000 | `diffusion_weight=1000` | RUNNING ep 191 | `…_diff1000_…` |
| final518 hw2 | `heart_weight=2.0` | RUNNING ep 192 | `…_hw2_…` |
| final518 motion10 | `motion_l1_weight=10` | RUNNING ep 190 | `…_motion10_…` |
| final518 motion10_hw2 | both | RUNNING ep 187 | `…_motion10_hw2_…` |
| final518 **nogather** | `gather_weight=0` (commit bfcfb80) | **PENDING** (job 61109456) | — |
| Curation curated vs uncurated | `train_curation_ablation.sh`: **224 px, native splat, constant LR — NOT the final recipe**; 628 vs 1066 train | both done ep 299 | `210823531_curation_curated`, `210822937_curation_uncurated` |
| Resolution 224 / 336 | pooled1337 series (diffusion 1000, native splat — older recipe) | done ep 299 | `213340611_…augaggr224…`, `213321784_augaggr336_…` |
| Resolution 518 | `…518native.sh` / `…518splat518.sh` | log dirs not located by glob — UNVERIFIED | — |
| Frames-per-slice (burst_k 1 / 12) | 224 px, `pooled_curated_v2_dmax15.txt`; `burst_k` key absent from `default.yaml` and `sbatch/` at HEAD → **lives on another branch/worktree** | burst1 done; burst12 ep 65, resume PENDING | `210823345_burst1noreg224dmax15_curatedv2`, `…burst12…` |

⚠️ No ablation arm other than the final518 family shares the final recipe. A single-factor ablation
table needs every row at 518/518 with the base loss; today only the final518 family qualifies, and the
reference-off / breathing-sim-off / frozen-backbone arms **do not exist**.

### 3.9 Do not write
- "~0.1 s inference" — MEASURED WRONG (§2.4a/§3.7): the 518-px arms run ~530 ms–2.2 s/volume, D-dependent.
- "Trained on A40" — the final arms run on L40S (`spgpu2`).
- 200 epochs (the `default.yaml` value); the sbatch overrides to 300.
- "Frozen patch-embedding layer" — the frozen module is the whole DINOv2 encoder.
- Any epoch-300 val number before the run finishes.

---

## 4. Drop-in LaTeX for the three Setup paragraphs

`\todo{}` marks a PENDING value. Bib keys that are **missing** from `_paper/reference.bib` as of today:
nnU-Net, M&Ms, ACDC, CMRxRecon 2023/24/25, OCMR, MIITT, DINOv2, SSIM (Wang 2004) — add before compiling.

```latex
\subsection{Experimental Setup}
\label{sec:setup}

\paragraph{Data and evaluation splits.}
We pool five public ECG-gated breath-hold short-axis cine cohorts: CMRxRecon 2023, 2024 and
2025~\citep{cmrx2023,cmrx2024,cmrx2025}, ACDC~\citep{bernard2018acdc}, and
M\&Ms-1~\citep{campello2021mnms}, spanning five vendors, twelve centres, 1.5\,T and 3\,T, and healthy
and pathological hearts. Because inter-slice breath-hold misalignment in the gated volumes would corrupt
the target, we retain the 898 subjects whose measured inter-slice discontinuity is below 5\,mm
(Appendix~\ref{app:data}). Subjects are split 628/90/180 into train/val/test, stratified by source,
vendor, centre and pathology; all Canon scans and one Philips centre are held out from training as
unseen-scanner test cases. Each subject keeps its native slice count ($D{=}6$--$20$) and pitch
(5--12\,mm); in-plane, volumes are resampled to 1.4\,mm on a $256\times256$ grid, with $T{=}12$ cardiac
phases and end-diastole at phase~0. Two further datasets are never used for training: MIITT
(13 subjects with paired gated and real-time free-breathing acquisitions~\citep{miitt}) and OCMR
(8 gated stacks~\citep{chen2020ocmr}).

\paragraph{Baselines and evaluation metrics.}
For every test subject we freeze one simulated asynchronous stack (one breathing-corrupted frame per
plane at a random cardiac phase, Sec.~\ref{sec:learning}) and give the \emph{identical} stack, the
heart region of interest and the nominal slice geometry to every method; no method receives
ground-truth poses or cardiac phases. We compare against single-stack slice-to-volume reconstruction
(SVRTK~\citep{kuklisovamurgasova2012reconstruction}, NeSVoR~\citep{xu2023nesvor}), learned slice
alignment (Dangi et al.~\citep{dangi2018cine}, retrained on our training split)\todo{, and FC-SVR /
dense-input methods if run}. Baselines reconstruct each of the 12 phases independently; ours sweeps the
reference slot. All reconstructions are rigidly registered to the ground truth per phase and scored
inside the heart ROI. We report normalized cross-correlation (NCC) as the primary image metric, since it
is invariant to each method's intensity scale, with PSNR and per-slice SSIM as secondary metrics.
Clinical accuracy is assessed by segmenting every reconstruction and the ground truth with the same
nnU-Net~\citep{isensee2021nnunet} on the same heart crop, yielding Dice and HD95 for LV, myocardium and
RV, and end-diastolic/end-systolic volume, ejection fraction and LV mass errors. Respiratory correction
is measured as the demeaned through-plane endpoint error between the predicted and applied per-slice
displacement. Runtime is measured end-to-end per subject: learned and GPU methods (ours, NeSVoR,
\todo{Dangi et al. — currently a CPU number, re-run on GPU}) on one NVIDIA A40, and SVRTK on 16 cores
of an Intel Xeon Gold 6154.

\paragraph{Implementation details.}
We initialise from VGGT-1B~\citep{wang2025vggt}, freeze the DINOv2 encoder (304M parameters) and
fine-tune the remaining 637M parameters; the through-plane embedding (8k parameters) is the only new
module. Inputs are $518\times518$ and the splat uses a $518\times518$ point grid per slice. We train
for 300 epochs of 628 steps (one subject per step) with AdamW (learning rate $5\times10^{-5}$, weight
decay 0.05), 5\% linear warm-up followed by cosine decay, bf16 precision, and gradient clipping at 1.0,
selecting the checkpoint with the best validation heart-region PSNR. Loss weights are
$\lambda_h=\lambda_o=0.5$ with no explicit smoothness term. During training, each gated cine is
augmented once per sample, consistently across all phases, with in-plane geometric transforms (flip,
rotation, translation, isotropic zoom), intensity transforms (gamma, bias field), and MR
acquisition-artifact proxies (simulated low resolution, Gibbs ringing, phase-encode ghosting), before
the target volume and asynchronous observations are extracted; the respiratory simulation of
Sec.~\ref{sec:learning} is applied to the inputs in both training and validation. Training takes
\todo{$\approx$4 days} on one NVIDIA L40S (29\,GB peak memory); reconstructing one volume takes
$\approx$1\,s on an NVIDIA A40 (530\,ms--2.2\,s, scaling linearly with the subject's slice count).
Exact augmentation probabilities and ranges are listed in Appendix~\ref{app:impl}.
```

---

## 5. PENDING before the paragraphs can be finalised

| item | blocks | owner / job |
|---|---|---|
| final518 base (and sibling arms) reach epoch 300 | every headline number; ckpt selection | SLURM 61000966 etc. |
| nogather arm | ablation row | job 61109456 |
| NeSVoR v2 scoring (gated + scatter) | Table 1/2 rows | array 61109592 |
| Dangi `_ef` JSONs | Table 2 row | jobs 61125839/40 |
| FC-SVR v2 retrain + runner | Table row or drop | docs/98 §11 |
| CiNeVol run-vs-cite | † row or drop | undecided |
| Fetal CMR 4D full test run + EF/Dice | † row | **DONE 2026-09-16** (docs/105 §8): 180/180 test, image + EF/Dice scored + aggregated; val not run; paper name still open |
| NiftyMIC engine runner | row or drop (currently no row) | not built |
| v2 identity / no-correction floor | floor row | `dice_floors.py` on v2 |
| 518-px inference timing | ~~"xx ms"~~ | **DONE 2026-09-18** (§2.4a/§3.7): `timing.json` across final518 sweep |
| Dangi inference re-run on A40 (`--device cuda`) + GPU name stamped in every `timing.json` | runtime column fairness | `run_dangi.py`, `run_vggt.py` |
| "Mean Dice" definition | Table 2 column | writer decision |
| Reference-off / breathing-sim-off / frozen-backbone arms at the final recipe | ablation rows | not started |
| Real-time MIITT on v2 ckpts (+ any GT-free number) | §4.6 | `run_vggt_rt.py` |
| Missing bib entries (nnU-Net, M&Ms, ACDC, CMRx ×3, OCMR, MIITT, DINOv2, SSIM) | compile | writer |
| M&Ms field strength / resolution; MIITT & OCMR vendor | appendix data table | cite source papers |
