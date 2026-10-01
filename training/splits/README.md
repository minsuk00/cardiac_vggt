# training/splits/ — the single home for every split file

Format: `[train]` / `[val]` / `[test]` sections, one `<source_dir>/<subject>` rel path per line
under `data_root = scratch/data`, `#` comments. Read by `MRIDataset._find_subjects`.

| file | status | what |
|---|---|---|
| **`pooled_curated_v2.txt`** | **LIVE** | THE split. 898 misalignment-curated subjects, re-split 628/90/180 so train/val/test share one (source, vendor, centre, pathology) composition; Canon + CMRx25 Center012 (Philips) are unseen-vendor holdouts (val/test only). Built by `tools/build_curated_split_v2.py`. docs/97. |
| `miitt_eval.txt` | eval-only | All 13 MIITT gated SAX subjects under `[val]` — a HELD-OUT dataset for the v2 series (paired real-time scans). Never add to a pool. |
| `ocmr_eval.txt` | eval-only | 8 OCMR gated SAX stacks, all under `[val]`. Never trained on; never add to a pool. docs/79. |
| `overfit_1.txt`, `tiny_10_2.txt` | utility | 1-subject overfit (train = val) and 10/2 quick-iteration splits drawn from v2 train/val in-section (no eval-subject leak). Live-format successors of the archived `overfit_p001` / `tiny_10_2`. |
| `manifest.csv` | metadata | One row per subject (vendor, centre, pathology, geometry, `split` = pooled.txt section, `split_curated_v2` = v2 section). |
| `_archive/` | record | Superseded splits: `pooled.txt` (pre-curation 1337-subject pool, every pre-2026-09-11 run), `pooled_curated_v1.txt` + `_manual_overrides.csv` (curation before the re-split), `pooled_uncurated_ablation.txt` (uncurated ablation arm, val/test identical to v2), `pooled_miitt.txt`; and pre-pooling files (`random_8_1_1`, `cmrx24only`, `default`, `overfit_p001`, `tiny_10_2`) in the old bare-id format. |
