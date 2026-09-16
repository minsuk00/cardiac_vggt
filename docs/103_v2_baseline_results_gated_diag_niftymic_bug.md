# 103 — v2 baseline results (SVRTK / NeSVoR / Dangi / VGGT), the gated-input diagnostic, the docs/86 EF discrepancy, and the NiftyMIC +1 intensity bug

> **TL;DR & takeaway** (2026-09-15)
>
> **Same-input ("scatter") comparison, pooled 7 sources, test split (n=180):** VGGT 20.25 dB PSNR /
> 0.648 SSIM / 22.5 pp EF MAE vs SVRTK 17.93 / 0.522 / 37.0, NeSVoR 17.76 / 0.532 / 41.8, Dangi
> 17.91 / 0.502 / 41.9. Val (n=111) tells the same story. NeSVoR and SVRTK are tied on everything
> (≤0.5 dB, ≤0.01 SSIM/NCC, EF within 1 pp gated). All numbers are from the 224 px
> `vggt_curation_curated_ep300` stand-in; the 518 px `final518_*` arms are still training.
>
> **Gated-input diagnostic (new `run_vggt.py --input gated`):** feeding VGGT the same gated stack the
> classical methods get, its EF MAE drops to 4.6–5.3 pp (slope 0.88–0.94) and it beats SVRTK/NeSVoR
> gated on every metric (+2.7–3.3 dB PSNR). So the scatter-arm EF gap (22 pp) is an **information
> limit of the scattered input, not a model limit** — a same-checkpoint, same-pipeline control for
> the flat-EF-amplitude finding (docs/24/25/33).
>
> **docs/86's 13.4 pp EF MAE vs today's 22 pp:** re-running docs/86's own checkpoint through today's
> pipeline gives 16.8 pp. Roughly ~3 pp is pipeline (the docs/98 bundle: registration, crop-everyone
> segmentation, native-z rebuild — NOT isolated to a component) and ~5 pp is checkpoint/split. The
> ground-truth EF itself shifts ~3.6 pp/subject between the two pipelines, so the GT is not fixed
> either. Which component causes the pipeline share is still open.
>
> **NiftyMIC was scoring 11–13 dB because of a NiftyMIC bug, not our scorer.** Under SimpleITK 2.x,
> `slice_sitk += 1` in its Shepard initialiser mutates the input slice in place, so every input
> slice (and slice mask) is permanently shifted by +1 and the solver reconstructs `input + 1`
> (intercept 0.82 ± 0.04 across 24 subjects). A 2-line copy fix (`sitk.Image(slice.sitk)`) brings
> it to `output ≈ input` (intercept 0.02) and heart PSNR to SVRTK's level on 4 subjects × 3 sources.
> All shipped NiftyMIC volumes are being regenerated with the patch (another agent).

## 1. What was run this session

| step | arms | jobs |
|---|---|---|
| NeSVoR image metrics | `nesvor`, `nesvor_scatter` × val/test | `sbatch/eval_score_baseline.sh` (new; generic METHODS/SPLITS scorer, jjparkcv98/spgpu, `--gpu_cmode=shared`) |
| NeSVoR EF/Dice | same | 4× `sbatch/eval_ef_dice.sh`, jjparkcv98/spgpu (first attempt on jjparkcv_owned1 sat `PENDING (AssocGrpGRES)` — lab-wide 40-GPU cap, unrelated jobs) |
| VGGT gated diagnostic | `vggt_curation_curated_ep300_gated_diag` × val/test | `INPUT=gated sbatch/eval_pooled_val.sh` → score → EF/Dice, spgpu2 |
| docs/86 repro | `vggt_docs86_repro_224` (val, cmrx2024 only) | old ckpt `213340611_…augaggr224…/checkpoint_last.pt` through today's `run_vggt.py` + scorer + `ef_dice.py` |
| NiftyMIC | full cohort generation (`61194028_0-15`, standard) + scoring — **invalid, see §5** | by the NiftyMIC agent (docs/102) |

`--input {scatter,gated}` in `evaluation/src/engine/run_vggt.py`: `gated` sets every companion
slot's phase to the queried target `t` (what SVRTK/NeSVoR-gated see); default `scatter` is
byte-identical to before. Guard: `--input gated` requires `gated` in `--model-name` so it can never
overwrite the real scatter arm. `INPUT` env var threaded through `sbatch/eval_pooled_val.sh`.

## 2. Results — pooled over all 7 sources

Image metrics: `breath_*` mean over subjects (registered read, heart∩FOV ROI). EF: LV EF MAE in
pp, slope of pred-vs-GT EF. Dice at ED. VGGT = `vggt_curation_curated_ep300` (224 px, curated_v2
split). Reproduce with the snippet in §7.

### Gated input (each method gets the full gated stack at the queried phase)

| split | method | PSNR | SSIM | NCC | n_ef | EF MAE | slope | Dice LV / MYO / RV |
|---|---|---|---|---|---|---|---|---|
| val (111) | SVRTK | 19.34 | 0.642 | 0.802 | 111 | 6.1 | 0.86 | 0.86 / 0.69 / 0.83 |
| | NeSVoR | 19.22 | 0.646 | 0.817 | 111 | 6.5 | 0.88 | 0.88 / 0.72 / 0.84 |
| | Dangi | 18.62 | 0.557 | 0.770 | 98 | 5.8 | 0.87 | 0.85 / 0.66 / 0.83 |
| | **VGGT (gated diag)** | **22.34** | **0.774** | **0.902** | 111 | **5.3** | 0.88 | **0.91 / 0.80 / 0.89** |
| test (180) | SVRTK | 20.07 | 0.670 | 0.816 | 180 | 5.7 | 0.85 | 0.88 / 0.72 / 0.85 |
| | NeSVoR | 19.54 | 0.666 | 0.823 | 180 | 6.0 | 0.92 | 0.88 / 0.72 / 0.84 |
| | Dangi | 19.17 | 0.580 | 0.780 | 180 | 5.7 | 0.90 | 0.85 / 0.65 / 0.83 |
| | **VGGT (gated diag)** | **22.78** | **0.787** | **0.905** | 180 | **4.6** | **0.94** | **0.92 / 0.80 / 0.89** |

### Scatter input (same-input bundle: one random phase per slice, what VGGT is built for)

| split | method | PSNR | SSIM | NCC | n_ef | EF MAE | slope | Dice LV / MYO / RV |
|---|---|---|---|---|---|---|---|---|
| val (111) | SVRTK | 17.24 | 0.496 | 0.705 | 111 | 37.6 | 0.31 | 0.76 / 0.53 / 0.71 |
| | NeSVoR | 17.43 | 0.516 | 0.725 | 111 | 44.7 | 0.22 | 0.80 / 0.59 / 0.78 |
| | Dangi | 17.37 | 0.480 | 0.700 | 111 | 43.5 | 0.23 | 0.77 / 0.53 / 0.74 |
| | **VGGT** | **19.80** | **0.635** | **0.830** | 111 | **21.6** | **0.63** | **0.87 / 0.71 / 0.83** |
| test (180) | SVRTK | 17.93 | 0.522 | 0.724 | 180 | 37.0 | 0.33 | 0.77 / 0.56 / 0.73 |
| | NeSVoR | 17.76 | 0.532 | 0.733 | 180 | 41.8 | 0.31 | 0.80 / 0.59 / 0.78 |
| | Dangi | 17.91 | 0.502 | 0.710 | 180 | 41.9 | 0.27 | 0.76 / 0.54 / 0.74 |
| | **VGGT** | **20.25** | **0.648** | **0.833** | 180 | **22.5** | **0.64** | **0.87 / 0.71 / 0.83** |

Observations:
- **NeSVoR ≈ SVRTK** on every axis. Dangi ≈ them on PSNR, worse on SSIM (0.50–0.58).
- **Scatter input costs the classical methods ~2 dB and ~31–36 pp EF**; they average over
  phases (slope 0.2–0.3 → nearly flat EF). VGGT loses 2.5 dB and ~17 pp EF vs its own gated run
  and keeps slope 0.63.
- Val predicts test (curated_v2 split, docs/97): ranking and gaps identical on both.
- Dangi val gated EF has n=98/111 (13 subjects failed segmentation); all other cells are complete.

## 3. Gated diagnostic: information limit, not model limit

The only difference between `vggt_curation_curated_ep300` (scatter) and `…_gated_diag` is one line
(`batch["timesteps"][0, :] = t`); same checkpoint, same scorer, same subjects. Gated → EF MAE
4.6–5.3 pp, slope 0.88–0.94, Dice 0.88/0.81/0.86 (LV/MYO/RV, averaged ED); scatter → 22 pp, slope
0.63. The model can recover EF when the input contains the target phase throughout the stack; it
cannot invent contraction amplitude from one reference slice plus phase-scattered companions.
This is the direct control for docs/24/25/33's flat-EF finding and for the "no extra reference
frames" contract.

## 4. The docs/86 EF discrepancy (13.4 pp then, 22 pp now)

Ruled out: (a) different input regime — `run_vggt.py` at docs/86's commit (`3c84427`) already used
`ds.get_data()`'s one-frame-per-slice sampler with independent random phases; the frozen
`pin_scatter` bundle only pins the *draw*, not the difficulty. (b) pooled vs cmrx2024-only — today's
checkpoint on cmrx2024 val alone is 22.1 pp.

Measured: docs/86's own checkpoint (`augaggr224_ep300`) through today's pipeline on cmrx2024 val:

| pipeline | ckpt | n | EF MAE | slope |
|---|---|---|---|---|
| pre-v2 (archived `evaluation/_archive/metric_results_prev2_20260913/_ef/`) | augaggr224_ep300 | 28 (pooled-1337 val) | 13.4 | 0.97 |
| today's v2 | augaggr224_ep300 | 19 (curated_v2 val) | 16.8 (16.2 excl. P091) | 0.41 |
| today's v2 | curation_curated_ep300 | 19 | 22.1 | — |

So ≈3 pp is attributable to the pipeline bundle (docs/98: 6-DOF registration, PSF, gauge,
crop-everyone-before-segmenting, native-z rebuild) and ≈5 pp to checkpoint/split. Note the two
rows also differ in val membership (28 vs 19 subjects), so the pipeline share is an estimate.
Against the archived snapshot, on shared subjects, **GT EF itself moved** (mean |Δ| 3.57 pp, signed
−2.63) and VGGT's predicted EF moved more (|Δ| 7.48, signed −5.68): the crop-everyone policy
hits GT and pred symmetrically but not equally. `CMRx24_Train_P091` flips +7.8 → −27.0 pp but
moves the aggregate only 0.6 pp. **Which docs/98 component drives the 3 pp is not isolated**;
doing so means single-component scorer toggles on the already-reconstructed
`vggt_docs86_repro_224` volumes (an `evaluation/` change → needs a go-ahead). Two wrong mechanisms
were asserted before measuring this session (input regime; crop-everyone alone) — measure first.

## 5. NiftyMIC +1 intensity bug (root-caused and fixed)

**Symptom.** Full-cohort NiftyMIC (docs/102 runner) scored 11.5 dB val / 12.8 dB test PSNR vs
19–20 for SVRTK/NeSVoR; 66/235 subjects < 10 dB; NCC normal (0.72–0.85). Per-subject PSNR
correlated −0.66 with the in-heart DC residual and −0.65 with the recon's background level.

**Not the scorer.** The raw engine output (`vol_t*.nii.gz`, untouched by `image_metrics.py`) obeys
`output = a·input + b` with a = 1.19 ± 0.19 but **b = 0.82 ± 0.04** (24 subjects, all sources):
a constant additive floor, air ≈ 0.8–0.9 instead of 0. The scorer's two-point self-norm gauge
(`norm_map`, p0.5→p0.5 / p99.9→p99.9) cannot remove it because the recon's p0.5 is 0 (mask fringe),
so the map degenerates to pure scale. SVRTK (a≈1, b≈0) and NeSVoR (pure scale, b≈0.5% of signal)
are clean — checked the same way.

**Root cause (verified in the container, SimpleITK 2.1.1.2).**
`niftymic/reconstruction/scattered_data_approximation.py::_run_discrete_shepard_reconstruction`
does `slice_sitk += 1` as a "mark observed" trick and subtracts it from its own sum. But
`_get_image_slice` returns `slice.sitk` itself (no copy) and SimpleITK 2.x `+=` mutates in
place (tested: `img2 = img; img2 += 1` changes `img`). Every input slice is therefore permanently
+1 before the TK1L2 solve, which faithfully reconstructs `input + 1`; `_get_mask_slice` does the
same to the slice masks, which is why the whole box gets reconstructed instead of the heart mask.
Same SimpleITK-2.x incompatibility family as the existing `joint_image_mask_builder.py` patch.

**Refuted alternatives (one phase of `CMRx24_Test_P028`):** `--iter-max 100` → b = 0.92 (not the
un-converged LSMR); `--alpha 1e-4` → b = 0.84 (not regularisation); `--sda` alone → air = 3.0,
max = 143 (the Shepard initialiser is itself broken at 12 mm pitch / 1.4 mm sigma, but LSMR
starts from zero so this does not feed the solve).

**Fix (2 lines):**
```python
def _get_image_slice(slice):  return sitk.Image(slice.sitk)        # was: slice.sitk
def _get_mask_slice(slice):   return sitk.Image(slice.sitk_mask)   # was: slice.sitk_mask
```
Bind-mounted over the container file (second `--bind` in `run_niftymic_v2.sh`, patch under
`scratch/niftymic/patch/`). Verified on 4 subjects × 3 sources × 2 phases:

| subject (phase) | shipped: b / air / heart (input) | patched: b / air / heart | heart PSNR unregistered: shipped / patched / SVRTK |
|---|---|---|---|
| CMRx24_Test_P028 (t0) | 0.87 / 0.98 / 1.13 (0.14) | 0.02 / 0.11 / 0.14 | 2.3 / 26.6 / 24.4 |
| CMRx24_Test_P020 (t5) | 0.85 / 0.82 / 1.01 (0.09) | 0.02 / 0.10 / 0.08 | 1.9 / 26.4 / 25.5 |
| OCMR_fs_0045_3T (t5) | 0.87 / 0.96 / 0.99 (0.07) | 0.03 / 0.10 / 0.06 | 2.0 / 29.6 / 31.0 |
| MNMs_D1H6U2 (t5) | 0.88 / 0.94 / 1.07 (0.06) | 0.01 / 0.08 / 0.06 | 0.9 / 35.6 / 32.6 |

In-heart slope stays 1.04–1.09 either way: only the offset changes. Figures:
`temp/niftymic_check/{root_cause_before_after,copyfix_3more,raw_input_vs_output}.png`.
**All shipped NiftyMIC volumes and their `metric_results/*/*/niftymic*.json` are invalid**; the
regeneration + rescore is being done by another agent (docs/102 runner + patch). NiftyMIC rows will
be added here when it lands.

## 6. Status / open

- **Dangi is final** (recon timestamps postdate scoring only because recon *time* was re-measured on GPU).
- **final518 sweep** (5 headline arms at epoch ~270/300; `nogather` + `diff1000_nogather` at ~10):
  run scatter arm → score → EF/Dice for each when it reaches 300; they replace the 224 px stand-in.
- **Not evaluating** FC-SVR or CiNeVol (decision 2026-09-15). Fetal-CMR-4D self-gating: in progress by the user.
- **Cleanup**: delete the diagnostic `vggt_docs86_repro_224` arm from `evaluation/metric_results/val/` once §4 is settled; commit the v2 eval diff once NiftyMIC and final518 land.

## 7. Repro

```bash
# 4-way table (§2)
python3 - <<'EOF'
import json, numpy as np, os
R="evaluation/metric_results"; SRC=["cmrx2023","cmrx2024","cmrx2025","acdc","mnms","miitt","ocmr"]
def img(s,a):
    v=[[r["breath_psnr"],r["breath_ssim"],r["breath_ncc"]] for d in SRC if os.path.exists(f"{R}/{s}/{d}/{a}.json")
       for r in json.load(open(f"{R}/{s}/{d}/{a}.json"))["per_subject"] if r.get("breath_psnr") is not None]
    return len(v), np.mean(v,0)
def ef(s,a):
    d=json.load(open(f"{R}/{s}/_ef/{a}.json")); ok=[r for r in d["per_subject"] if r.get("ef_gt") is not None and r.get("ef_breath") is not None]
    g=np.array([r["ef_gt"] for r in ok]); p=np.array([r["ef_breath"] for r in ok]); return len(ok), np.mean(abs(p-g)), np.polyfit(g,p,1)[0]
for s in ["val","test"]:
    for a in ["svrtk3d","nesvor","dangi","vggt_curation_curated_ep300_gated_diag","svrtk3d_scatter","nesvor_scatter","dangi_scatter","vggt_curation_curated_ep300"]:
        print(s, a, img(s,a), ef(s,a))
EOF
# NiftyMIC offset fit (§5): output = a*input + b on the raw engine grid
#   rec = vol_tNN.nii.gz; st = stack_tNN resampled onto it; m = rec>0.05; np.polyfit(st[m], rec[m], 1)
```
