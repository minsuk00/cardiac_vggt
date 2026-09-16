# 100 — Dangi et al. (2018) slice-alignment baseline: audit of the reimplementation + adaptation to our harness

> **TL;DR & takeaway.** The third-party PyTorch reimplementation of Dangi, Linte & Yaniv (SPIE 2018,
> "Cine Cardiac MRI Slice Misalignment Correction Towards Full 3D LV Segmentation") Stage A — a per-slice
> CNN that regresses the LV-cavity centre, then translates every slice so the centres are collinear — is
> **correct and faithful** (two independent audits, every geometric convention verified numerically, its
> tests pass). It is vendored at `baselines/dangi-cmr-alignment/` (upstream `.git` detached, kept in
> `temp/`). Three integration changes, none touching the method: **(1)** train on our `pooled_curated_v2`
> train section, all 12 phases, nnU-Net LV centroids (closer to the paper's 4D STACOM training than ACDC
> ED/ES, and it removes an ACDC test-leak); **(2)** apply the predicted mm shifts on our native 256²/1.4 mm
> grid (`evaluation/src/engine/run_dangi.py`); **(3)** anchor to the **reference plane's predicted centre**
> instead of the paper's image centre — same per-slice shifts up to a whole-stack constant, and the same
> information VGGT gets. **Structural limit (not fixable without changing the method):** a real SAX stack's
> LV centres are NOT collinear even without motion — measured on 32 motion-free GT subjects, median
> apex-to-base drift 10.4 mm (max 21 mm), per-slice offset from mid 4.8 mm, same order as our 8.5 mm
> simulated breathing — and Dangi straightens that anatomy along with the motion. It also cannot represent
> phase mismatch or through-plane motion. Report it as the honest lower bound it is.
> Training job 61089656 (spgpu2) → `scratch/baselines/dangi/pool_v2/`. Outputs stay in `temp/dangi/`
> until verified; nothing in `evaluation/` or the bundles yet.

## 1. Sources

- Paper: PDF in `baselines/dangi-cmr-alignment/`; DOI 10.1117/12.2294936 (PMC author manuscript).
- Implementation: https://github.com/sin021004/dangi-cmr-alignment (another agent's work; two commits,
  d42d338 head), cloned 2026-09-14. Its README + HANDOFF.md document every assumption.

## 2. What the paper does (Stage A only — the part that is a motion-correction baseline)

1. Preprocess: resample SAX in-plane to 1.5625 mm, centre-crop/zero-pad to 192².
2. Augment: 9 translations (±10 % x/y) × 7 rotations (0, ±10°, ±20°, ±30°) × 3 scales (0.9/1/1.1) = 189.
3. CNN (Fig. 1): 5 levels of two 3×3 conv+ReLU (4/8/12/16/20 ch), 4 max-pools; at levels 1–4 a
   1-channel 3×3 side conv flattened (36864+9216+2304+576) + full flatten of level 5 (2880) = 51840 →
   FC 256 → 2 = (x, y) LV centre. SSD loss, Xavier-uniform, 100 epochs, best-val checkpoint.
4. Correction: translate each slice so its predicted centre sits at the image centre → collinear.
5. Eval (Sec. 4): initial misalignment = |true − stack-mean(true)|, residual = |pred − true|; median
   3.13 → 2.07 px on 9 STACOM test cases.

Stage B (atlas + graph-cut LV segmentation) is downstream and not a motion-correction baseline; not
implemented, correctly.

## 3. Audit result (my review + one independent subagent; all claims run, not read)

| Component | Verdict | Evidence |
|---|---|---|
| Architecture | FAITHFUL | features per level `[36864, 9216, 2304, 576, 2880]`, 13.29 M params (FC1 13.27 M); same-padding is forced by 36864 = 192²; side-branch ReLU per Fig. 1 legend; FC2 ReLU = documented interpretation |
| Preprocess | FAITHFUL | oblique 30°, anisotropic, odd-size landmark → physical error 1e-3 mm; returned affine exact to 1e-14; (X,Y,Z)→(Z,Y,X) at one site |
| Augmentation | FAITHFUL | all 189 combos: blob centroid == returned target (atol 0.04 px); rotation and scale about (96,96) |
| Targets | FAITHFUL + documented deviation | blood-pool centroid; empty slices propagated from nearest mid-slice; partial-myocardium slices kept if non-empty (paper propagated) |
| Alignment | FAITHFUL | +x right, +y down; constant predictor → content lands exactly on (96,96); output intensities untouched (normalization is CNN-input only) |
| Training | FAITHFUL + documented deviations | SSD, Xavier, 100 epochs, best-val, val unaugmented, resume sane. Adam 1e-3 and batch 32 are assumptions (paper silent). Repo epoch = every slice × all 189 (paper: one random augmentation per volume per epoch) |
| Evaluation | FAITHFUL | matches Sec. 4 incl. KS test |
| Repo tests | PASS | 6 passed, 1 skipped (ACDC path) |
| Silent-corruption sweep | none | axis order, ½-pixel conventions, normalization order all checked |

## 4. Adaptation decisions (user-confirmed 2026-09-14)

| Decision | Choice | Why |
|---|---|---|
| Training data | `pooled_curated_v2` `[train]` (628 subj) / `[val]` (90), **all 12 phases**, target = centroid of nnU-Net `heart_seg.nii.gz` label 1 (LV cavity) on the native grid | Paper trained on 4D all-phase data; ACDC ED/ES would leave mid-systole OOD for our ungated inference. Same cohort as VGGT. The repo's own ACDC 80/10/10 split would have put our eval ACDC subjects (patient020/029/035/041/050/060/061/071/075/092/093 …) into Dangi's train set. nnU-Net LV Dice ≈ 0.94; a centroid is robust to boundary error. Expert-vs-nnU-Net ED centroids on ACDC_patient001 agree to ~1–3 px (different native orientation handling between the raw ACDC and `ACDC_sax` copies accounts for part of that; both are internally consistent). |
| Epoch definition | one random 189-grid transform per slice per epoch (paper's scheme); ~75 k slices/epoch | With 12 phases × 628 subjects there is enough data; exhausting the grid (repo default) was a small-data workaround for ACDC |
| Anchor at inference | **reference plane's predicted centre** (`--anchor reference`); `image_center` (paper literal) and `mean` (paper Sec. 4) available | Inference-only; training never uses an anchor. Removes one whole-stack offset the paper's (96,96) would charge us for; VGGT conditions on the same reference slice, so no oracle |
| Output grid | predict on Dangi's 1.5625 mm/192² grid, apply shifts in mm on the native 256²/1.4 mm stack | scorer reads native-grid recons unchanged |
| Input | `scatter/` (VGGT's frozen one-frame-per-slice + breathing draw) as headline; `breath/` gated as secondary | same input as VGGT and the classical baselines |
| Arm/layout | `<subj>/dangi_scatter/recon_breath/vol_tNN.nii.gz` + `stamp.json` + `timing.json`, mirroring SVRTK | scorer needs no change once copied in |
| Where it lives | `dangi/pool.py`, `--pool-split` in `dangi/train.py`, `anchor_point` in `dangi/align.py`, `sbatch/train_dangi_baseline.sh` (run dir `scratch/baselines/dangi/pool_v2/`); after verification (user go, 2026-09-14) the runner moved to **`evaluation/src/engine/run_dangi.py`** and `best.pt` (+ config/history/split) to **`evaluation/checkpoints/dangi_pool_v2/`**; arms `dangi_scatter` / `dangi` written into the bundles. Diagnostics stay in `tools/` (`dangi_centre_diagnostic.py`, `render_dangi_examples.py`, figures `temp/dangi/figs/`) | standing eval code only in `evaluation/` |

Orientation check: the eval-bundle stacks carry a nominal positive-diagonal (RAS-labelled) affine while
the native `sax_frame_*.nii.gz` are LPS, but the **array index order is identical** (rendered side by side,
`temp/dangi/orientation_check.png`), so the 2D net sees the same view at train and test.

## 5. Verification done

- Repo tests + new `tests/test_pool_anchor.py`: 10 pass, 1 skip.
- Pool loader on real subjects: (120, 192, 192) per subject, centres in-range, 90 % slices with a measured
  LV, 1.1 s/subject; float16 cache.
- 2-epoch CPU training smoke on 2 subjects (`temp/dangi/smoke`): val SSD 10593 → 1359.
- Runner on CMRx24_Test_P001 scatter with the smoke ckpt: 12 volumes, reference-plane shift exactly 0,
  native-grid point round trip (2.8, −1.4) mm → (+2, −1) voxels exact; 2.1 s/subject on CPU.

## 6. Structural limitation (native to the method — report, don't fix)

Dangi assumes a motion-free SAX stack has the LV centre at the same in-plane position on every slice.
Measured on our motion-free GT (nnU-Net label 1, 32 subjects, 4 sources): median per-slice offset from the
mid-slice centre 4.8 mm; apex-to-base drift median 10.4 mm, max 21.3 mm. Our simulated breathing moves
slices 8.5 mm on average, so the anatomy Dangi erases is the same order as the motion it corrects. A
per-slice 2D regressor cannot separate the two. Together with its inability to model cardiac-phase
mismatch and through-plane motion, this makes it an informative lower bound, not a strawman — say so.

## 7. Training result (job 61089668, spgpu2, 59 min wall, ~25 s/epoch)

78,348 train slices/epoch, 11,928 val slices. Best epoch 80: val SSD **19.95 px²** (4.47 px RMS ≈ 7.0 mm
at 1.5625 mm); train 12.6; last epoch 23.3 — a modest gap, handled by best-val checkpointing (as the paper).
`history.jsonl` copied beside the checkpoint.

**Faithfulness check on ACDC EXPERT labels** (paper's Table 1 metric; 32 ACDC subjects in v2 val+test,
64 ED/ES frames, 660 slices, label 3 = LV cavity, mid-slice propagation as the paper):

| px | ours (ACDC, held-out) | paper (STACOM test) |
|---|---|---|
| initial misalignment, mean ± sd / median | 1.62 ± 1.07 / 1.40 | 3.30 ± 1.71 / 3.13 |
| residual after correction, mean ± sd / median | 4.12 ± 3.75 / 2.70 | 2.40 ± 1.54 / 2.07 |

The residual is in the paper's range, so the trained model behaves like theirs. ACDC is gated + breath-held,
so its "initial misalignment" (LV-axis obliquity only) is already below the regressor's error and the
correction makes it slightly worse — the same pattern as on our bundles (§8).

## 8. What Dangi does on our scatter input (`tools/dangi_centre_diagnostic.py --out-root scratch/eval`, all 291 val+test subjects, phases 0+6, 5096 planes, mm)

| in-plane centre error vs GT LV centre (nnU-Net label 1) | mean | median | p90 |
|---|---|---|---|
| regressor: \|pred − true input centre\| | 6.00 | 4.95 | 10.56 |
| input, no correction (= in-plane breathing offset) | 2.80 | 0.84 | 8.49 |
| after Dangi, **reference** anchor (shipped) | 6.99 | 5.77 | 13.98 |
| after Dangi, mean anchor (paper §4) | 5.57 | 4.79 | 9.88 |
| after Dangi, image-centre anchor (paper §2.4 literal) | 20.66 | 19.40 | 37.34 |
| after Dangi, oracle anchor (true ref-plane centre) | 6.82 | 5.71 | 12.44 |

Reading: our breathing is mostly THROUGH-plane for SAX (SI axis ≈ slice normal), so the in-plane offset it
leaves is small (median 0.7 mm) while the regressor's own error is ~5 mm — so Dangi slightly *worsens*
in-plane alignment under every anchor, and by construction leaves through-plane and phase mismatch
untouched. This is the method as published, consistent with its own marginal 3.13 → 2.07 px gain and with
§7; report it as the floor it is. The simulator sign convention (d > 0 samples at +d, content appears at
−d) was checked against Dangi's shift signs on CMRx23_Test_P002.

Figures: `tools/render_dangi_examples.py` → `temp/dangi/figs/` (all z-planes input / Dangi / GT with
predicted + and GT o centres, XZ/YZ cuts): predicted centres sit on GT for mid-ventricular planes and
drift on apex/base planes where the LV is small — that is where the ~5 mm comes from.

## 9. Integration into `evaluation/` (user go, 2026-09-14)

Runner → `evaluation/src/engine/run_dangi.py` (default ckpt `evaluation/checkpoints/dangi_pool_v2/best.pt`,
default `--out-root scratch/eval`); arms `dangi_scatter` (headline, same frozen scatter input as VGGT /
SVRTK-scatter / NeSVoR-scatter) and `dangi` (gated `breath/`), val + test, all 7 sources; scored with the
standing `score/run.py` unchanged (same `recon_breath/vol_tNN.nii.gz` + `stamp.json` + `timing.json`
layout as SVRTK). The `temp/dangi/eval` dry pass was NOT copied; the bundles were regenerated by the
engine script (val 111 + test 180 subjects × 2 inputs, 0 errors). Optional rows still
available via `--anchor image_center|mean`.

**10a. Timing (paper's compute-cost column).** Generation was first run on CPU without flagging the
device choice to the user — caught and corrected same-day. Re-ran on an A40 (`sbatch/time_dangi_baseline_gpu.sh`,
job 61141965, spgpu/jjparkcv98, `run_dangi.py --overwrite --device cuda`) for all 291 val+test subjects ×
2 inputs, overwriting only `vol_tNN.nii.gz`/`timing.json`; `metrics.json`/EF NOT rescored (verified
identical: cmrx2024 val dangi_scatter breath PSNR 18.44±0.90 before and after, spot-checked). **A40 wall-clock:
dangi_scatter 3.98 s/subject mean (3.78 median), dangi 3.83 s/subject mean (3.70 median), ~0.33 s/phase**
— essentially the same as CPU, because the cost is NIfTI I/O + `scipy.ndimage.shift`, not the 13M-param
CNN forward pass. This is the number for the paper's speed comparison (vs SVRTK ~130 s/subject, VGGT ~3–4 s/subject).

## 10. Scored result (standing scorer, registered 6-DOF breath PSNR mean / SSIM; `metric_results/<split>/<ds>/dangi_scatter.json`)

**val, scatter input** (same frozen input for every column):

| source | Dangi (dangi_scatter) | SVRTK-scatter | VGGT curated ep300 |
|---|---|---|---|
| cmrx2023 | 18.35 / 0.58 | 18.35 / 0.58 | 20.50 / 0.70 |
| cmrx2024 | 18.44 / 0.56 | 18.51 / 0.55 | 20.63 / 0.68 |
| cmrx2025 | 19.18 / 0.55 | 19.63 / 0.60 | 21.86 / 0.71 |
| acdc | 16.39 / 0.42 | 16.64 / 0.46 | 18.55 / 0.57 |
| mnms | 15.73 / 0.38 | 14.68 / 0.37 | 18.15 / 0.54 |
| miitt | 15.45 / 0.34 | 15.14 / 0.37 | 18.10 / 0.54 |
| ocmr | 17.46 / 0.55 | 17.47 / 0.57 | 20.11 / 0.71 |

Dangi sits on top of SVRTK-scatter on every source and ~2 dB below VGGT — consistent with §8: after the
scorer's own rigid registration, a per-slice in-plane shift with ~5 mm error adds nothing to the input
stack. `resp_*` breathing-recovery metrics are null for Dangi by construction (no through-plane estimate).
Test split + gated (`dangi`) arm: same files under `metric_results/test/` and `<ds>/dangi.json`.
Figures (final bundle volumes): `temp/dangi/figs/index_dangi_scatter.html`, `index_dangi.html`.
