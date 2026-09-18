# 106 — Three findings from the EF-MAE investigation that are not about EF: PSNR peak convention, `splat_res=518` at inference, and `final518_hw2`'s late breathing blow-up

> **TL;DR & takeaway** (2026-09-16; measured in the docs/104 session on the `final518_hw2` ep250
> harness run, jobs `61216621`/`61216622`). **(1) The harness headline PSNR uses a nonstandard
> peak** — `peak = GT.max()` *inside the heart ROI* (`image_metrics.py:psnr`) instead of the
> field-standard full-image max (≈ 1.0 for our [0,1] data; fastMRI and CMRxRecon's own scorers
> both use the whole-volume/slice GT max). It costs ~3–4 dB pooled on cmrx2024 (21.85 → 25.62 dB
> for hw2 ep250), less on ACDC/M&Ms (~0.8–1.8 dB), and the offset is subject-dependent, not a
> constant. `psnr_unit_peak` is already computed and stored for every arm and subject; **decision:
> make it the paper headline (aggregation-key change only, no recompute).** **(2) The 518 arms
> train with `loss.volume.splat_res=518`; rendering the same checkpoint at native 256 instead makes
> everything worse** (−0.5 dB, EF MAE 11.0 → 11.9–12.0 pp on 19 subjects, blur traded for speckle
> in Δ on non-reference planes). Keep `splat_res=518` at inference — it is what the model trained
> on, not a rendering trick, and the 518-vs-224 EF gain survives without it. **(3) `final518_hw2`
> destabilises after epoch ~200**: val breathing EPE goes 1.03 mm (ep200) → 2.46 → 5.95 → 8.36
> (ep275) → 5.91 mm (ep299), driven by a few basal slots per subject predicted hundreds of mm
> through-plane (27/111 val subjects at ep250, max 1187 mm). `base`, `diff1000` stay at ~0.95 mm;
> `motion10_hw2` has only a mild bump (0.97 → 1.47 → 1.26). **hw2 ep300 is disqualified as the
> reporting arm**; candidates are `motion10_hw2` ep300 or hw2 ep200.

Companion to docs/104 (EF-MAE crop bias). The three threads below were found while checking that the
docs/104 conclusions hold on the real 518 px checkpoint; none of them is about EF, so they get their
own doc. Full recon/score numbers for hw2 ep250 are in `evaluation/metric_results/val/*/vggt_final518_hw2_ep250.json`.

## 1. PSNR peak convention

`evaluation/src/score/image_metrics.py::psnr` computes `peak = max(gt[roi])`, where `roi` = whole-heart
seg ∩ FOV. The heart is often not the brightest part of the FOV, so this peak is typically 0.3–0.9,
and the reported PSNR is lower than the trainer's `metric_psnr_3d_*` (peak 1.0) by
`20·log10(peak)` per subject. `psnr_unit_peak` (same ROI, peak 1.0) is emitted alongside and already
aggregated (`aggregate.py` keys `breath_psnr_unit_peak`, `clean_psnr_unit_peak`).

Pooled effect, `vggt_final518_hw2_ep250`, val:

| cohort | `breath_psnr` (ROI peak) | `breath_psnr_unit_peak` |
|---|---|---|
| cmrx2024 | 21.85 | 25.62 |
| acdc | 19.68 | 20.45 |
| mnms | 19.20 | 20.98 |

The offset is not a constant: an early single-subject spot check (P012, dark small heart) suggested
~8 dB; the pooled cmrx2024 number is ~3.8 dB and ACDC is under 1 dB. Only pooled numbers are
meaningful.

**Field standard, verified from source:** fastMRI `fastmri/evaluate.py` uses `maxval = gt.max()`
over the whole volume; CMRxRecon's `Evaluation/Evaluation.py` uses `data_range = gt.max()` per
slice after per-slice normalisation. Both are full-image max, never an ROI-restricted max. NeSVoR's
own scorer could not be fetched (auth/404) and is not part of this claim.

**Decision (user, 2026-09-16):** paper headline = `psnr_unit_peak`. Implementation is a headline-key
swap in `aggregate.py` (and `docs/103` tables) — no re-scoring. Rankings are unchanged because every
arm shares the same ROI and GT; only the constant per subject moves.

## 2. `splat_res=518` must stay on at inference for the 518 arms

The 518 arms set `loss.volume.splat_res: 518` (upsample the native 256² slice grid to 518² points
before splatting, a coverage-averaging low-pass; `default.yaml` comment block). `run_vggt.py` reads
the run's own value from `run_meta` and renders with it. Test: render the same `final518_hw2`
ep250 checkpoint with `splat_res=None` (native 256, what the 224 arms do) on the 19 shared cmrx2024
val subjects.

| render | breath PSNR | EF MAE (n=19) | visual |
|---|---|---|---|
| `splat_res=518` (as trained) | ref | 11.0 pp | smooth |
| `splat_res=None` | −0.5 dB | 11.9–12.0 pp | pixel-level speckle in Δ on non-reference planes |

Worse on every axis, and EF is basically insensitive (within n=19 noise). Conclusion: the 518 render
is not an artefact to strip; the model was trained expecting it. The 518-vs-224 EF improvement
(docs/104: 21.6 → 11.2 pp tight crop, 18.4 → 9.8 pp padded) is a model gain, not a rendering trick.
Scratch script: reused `run_vggt.py`'s `load_model_from_run`/`make_dataset`/`load_bundle`/`reconstruct`
via `sys.path.insert` with `PYTHONPATH=training:.` (not kept; recipe only).

## 3. `final518_hw2` breathing-correction instability after epoch ~200

`val/resp/epe_dz_mm` (mean |predicted − applied| through-plane shift over input slots) from each
run's `metrics.jsonl`, `scratch/logs/210823094_final518_*_curated898/`:

| epoch | base | diff1000 | hw2 | motion10_hw2 |
|---|---|---|---|---|
| 150 | 1.50 | 1.56 | 1.09 | 1.11 |
| 175 | 1.42 | 1.19 | 1.05 | 0.98 |
| 200 | 1.23 | 1.10 | **1.03** | 0.97 |
| 225 | 1.11 | 1.01 | **2.46** | 1.10 |
| 250 | 1.04 | 0.97 | **5.95** | 1.02 |
| 275 | 0.97 | 0.97 | **8.36** | 1.47 |
| 299 | 0.96 | 0.95 | **5.91** | 1.26 |

Per-slot inspection of the harness's `resp_diag.json` for `vggt_final518_hw2_ep250` (val, all
cohorts): 27 of 111 subjects have at least one slot with |pred − applied| > 10 mm; the worst is
1187 mm. These are a few basal slices per subject thrown entirely out of the reconstructed volume,
not diffuse noise. hw2's EF/PSNR still looked best (EF 10.9 pp, motion PSNR 21.49 in the training
log) because the volume L1 barely sees a slice that lands outside the grid.

**Consequence:** `hw2` ep300 / `checkpoint_last.pt` must not be the reporting arm, and warm-restarting
from it is off the table (it would extend the unstable run). Candidates: `motion10_hw2` ep300
(EF 11.1 pp / motion PSNR 21.60 in-log, indistinguishable from hw2 on those; never run through the
harness yet) or `hw2` ep200 (EF 12.2 pp, EPE 1.03 mm, keeps the hw2 recipe). Decision pending the
ep300 harness evaluation of every 518 arm (plan: evaluate all arms at ep300 at the end of the
padded-mask campaign, docs/104 §6).

## 4. Repro

```bash
# 1. PSNR keys already stored
python3 -c "import json;d=json.load(open('evaluation/metric_results/val/cmrx2024/vggt_final518_hw2_ep250.json'))['all'];print(d['breath_psnr'],d['breath_psnr_unit_peak'])"
# 3. EPE trajectories
for a in base diff1000 hw2 motion10_hw2; do python3 -c "
import json;r=[(x['epoch'],round(x['value'],2)) for x in map(json.loads,open('scratch/logs/210823094_final518_${a}_curated898/metrics.jsonl')) if x['name']=='val/resp/epe_dz_mm'];print('$a',r[::25])"; done
# 3. blown slots
python3 -c "
import json,glob,numpy as np
for f in glob.glob('scratch/eval/*/out/*/vggt_final518_hw2_ep250/resp_diag.json'):
    d=json.load(open(f))['breath'];e=np.abs(np.array(d['pred_dz_mm'])-np.array(d['applied_dz_mm']))
    if e.max()>10: print(f,round(e.max(),1))"
```
