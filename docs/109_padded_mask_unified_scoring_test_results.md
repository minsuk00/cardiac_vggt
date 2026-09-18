# 109 — Padded-mask, unified-scoring campaign: final test-split results (same-input regime)

> **TL;DR & takeaway** (2026-09-17). Completed the docs/107 campaign: regenerated SVRTK/NeSVoR/
> NiftyMIC on test/scatter under the padded mask, unified the image-metric scoring ROI to the
> same padded mask (docs/107 update), switched the headline PSNR to unit-peak (docs/106), and
> re-scored every affected arm (Dangi ×2, VGGT curation ×4, all three regenerated baselines).
> **Same-input ("scatter") test results, n=180, one consistent protocol for every method:**

| Method | PSNR (unit-peak) | SSIM | NCC | EF MAE (pp) | slope | Dice LV/MYO/RV |
|---|---|---|---|---|---|---|
| SVRTK | 21.52 | 0.561 | 0.769 | 36.9 | 0.37 | 0.72/0.55/0.68 |
| NeSVoR | 20.82 | 0.555 | 0.779 | 42.3 | 0.31 | 0.73/0.56/0.69 |
| NiftyMIC | 19.95 | 0.545 | 0.784 | 42.2 | 0.34 | 0.72/0.54/0.68 |
| Dangi | 21.08 | 0.520 | 0.738 | 41.8 | 0.23 | 0.71/0.52/0.67 |
| **VGGT** (`vggt_curation_curated_ep300`) | **23.60** | **0.682** | **0.856** | **19.2** | **0.74** | **0.82/0.72/0.77** |

> VGGT beats every baseline on every metric (+2.08 dB PSNR over the next-best, SVRTK; EF MAE
> roughly half the best baseline's; slope nearly double). Ranking is unchanged from docs/103's
> pre-padded-mask numbers — this campaign fixed the protocol, not the conclusion.

## 1. What this campaign did

Per docs/107 §4's "Next" list:
1. Regenerated `svrtk3d_scatter`, `nesvor_scatter`, `niftymic_scatter` on test split under
   `mask_heart_pad10` (their reconstruction ROI). All three ran clean end-to-end (`sacct` all
   `COMPLETED`, 180/180 subjects each).
2. Switched `image_metrics.py`'s scoring ROI from the tight `mask_heart` to the same padded mask
   — a scope addition beyond docs/107's original text, made explicit this session (image metrics
   were the one place still reading the tight mask; every arm now scores under one ROI end to
   end, not two). Also switched the headline PSNR key from `breath_psnr` (ROI-peak) to
   `breath_psnr_unit_peak` (docs/106) in `aggregate.py`'s print + `compare_table.py`'s default.
3. Re-scored (image metrics) and re-`ef_dice`'d every test arm touched by either change: `dangi`,
   `dangi_scatter`, `vggt_curation_curated_ep300`, `vggt_curation_curated_ep300_gated_diag`,
   `vggt_curation_uncurated_ep177`, `vggt_curation_uncurated_ep300`, plus the three regenerated
   baselines above. All completed clean, no failed subjects.
4. Fixed one real bug found in a 3-reviewer correctness pass on this work: `aggregate.py`'s
   `summarize()` computed `breath_psnr_unit_peak` but never added the matching
   `clean_psnr_unit_peak` key — silently broke the `compare_table.py` column for that metric
   (never consumed by anything else, no impact on the results above). Fixed, verified against a
   live JSON, `tests/` re-passed (372).

## 2. What is NOT in this campaign (known gaps)

- **Gated-input (non-scatter) SVRTK/NeSVoR/NiftyMIC were not regenerated or rescored.** Their
  live `metric_results/test/*/{svrtk3d,nesvor,niftymic}.json` files don't exist right now — only
  the pre-padded-mask versions in `evaluation/_archive/metric_results_prepad_20260916/` do. The
  docs/103 "gated" comparison table cannot be reproduced with today's protocol without a further
  regeneration pass (scope was explicitly test/scatter only, per docs/107 §4 and the user's
  direction this session).
- **Dangi (gated)** and **`vggt_curation_curated_ep300_gated_diag`** *were* rescored this session
  (they needed no regeneration — Dangi has no mask, VGGT reconstructs full-FOV) so their gated
  numbers under the new protocol exist, but there is no gated SVRTK/NeSVoR/NiftyMIC counterpart to
  compare them against right now.
- **Val split** was not touched this session — only the archived (tight-mask) val numbers exist.
- **518px `final518_*` arms at ep300** are a separate track (docs/108), not part of this campaign.

## 3. Repro

```bash
python3 - <<'PY'
import json, numpy as np
cohorts = ["cmrx2023","cmrx2024","cmrx2025","acdc","mnms"]
def pool(arm):
    rows=[]
    for c in cohorts:
        try: rows += json.load(open(f"evaluation/metric_results/test/{c}/{arm}.json"))["per_subject"]
        except FileNotFoundError: pass
    psnr=[r["breath_psnr_unit_peak"] for r in rows if r.get("breath_psnr_unit_peak") is not None]
    ef_gt=np.array([r["ef_gt"] for r in rows if r.get("ef_breath") is not None])
    ef_pr=np.array([r["ef_breath"] for r in rows if r.get("ef_breath") is not None])
    print(arm, len(rows), round(np.mean(psnr),2), round(np.abs(ef_gt-ef_pr).mean(),1))
for a in ["svrtk3d_scatter","nesvor_scatter","niftymic_scatter","dangi_scatter","vggt_curation_curated_ep300"]:
    pool(a)
PY
```

## 4. Next

1. Decide whether to regenerate gated (non-scatter) SVRTK/NeSVoR/NiftyMIC under the padded mask
   to complete the docs/103-style gated table, or accept scatter-only as the paper's headline
   (the same-input regime is already documented as the paper's primary comparison — docs/98 §7).
2. Fold this table into docs/103 or the paper draft directly; docs/103 itself is left as the
   historical record of the pre-padded-mask investigation, not edited in place.
3. Once docs/108's 518-arm reporting checkpoint is decided, this scatter table is the baseline
   comparison it should be checked against.
