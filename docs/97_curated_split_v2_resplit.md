# 97 — `pooled_curated_v2.txt`: re-split of the curated cohort so val predicts test; one split directory

> **TL;DR & takeaway** — The curated 898-subject cohort (docs/96) was **re-split** into
> **628 / 90 / 180 train/val/test (69.9/10.0/20.0)** so that train, val and test share the same
> (source, vendor, centre, pathology) composition. Before, val looked like train while test was 60%
> pathology and held all Canon + most GE (inherited from the ACDC/M&Ms *official* test sets), so val
> could not predict test. Nobody publishes slice-to-volume numbers on those official test sets, so
> honouring them bought nothing; the only thing worth keeping was the **unseen-vendor holdout**:
> Canon (M&Ms centre 5) and CMRx25 Center012 (Philips) are **never in train** and split val:test = 1:2
> so val also carries an unseen-vendor signal. 134 subjects (15%) changed section vs v1 — the eval
> bundles for those must be rebuilt/relabelled before they are scored under v2. **`training/splits/`
> is now the single home for every split file** (`evaluation/splits/` deleted, 5 dead pre-pooling
> files archived, fresh `overfit_1.txt` / `tiny_10_2.txt` for sanity runs). **`default.yaml` now
> points at v2** (`split_file`, `dataset_name: curated898`, `limit_train_batches` 935→628,
> `limit_val_batches` 266→180); the curated-vs-uncurated ablation is still owed (§5).

## 1. Why re-split

Per-section composition of `pooled_curated_v1.txt` (share of each section):

| | train | val | test |
|---|---|---|---|
| Siemens | 76.5% | 68.5% | 49.5% |
| UIH | 15.0 | 10.9 | 4.3 |
| Philips | 6.3 | 13.0 | 17.4 |
| GE (MNMs c4) | 2.3 | 4.3 | 15.2 |
| Canon (MNMs c5) | 0 | 3.3 | 13.6 |
| diseased | 44% | 52% | 59% |

The skew is a side effect of one rule in `tools/build_pooled_split.py`: keep ACDC's official
`testing/` (50) and M&Ms' official Training/Validation/Testing verbatim. M&Ms Testing is 136 subjects
(40% of M&Ms) and holds all Canon and most GE by the challenge's own design, so test ended up much
harder than val. That comparability argument is void for this task (no published SVR numbers on
those sets; docs/46's ACDC-test numbers are our own, already non-comparable post-refactor).

Also: per-centre val cells were 0–4 subjects for most CMRx25 centres and five CMRx25 centres had 0 test
subjects, so per-centre reporting was noise.

## 2. Rule (`tools/build_curated_split_v2.py`, seed 42)

1. **Holdouts** — `vendor == Canon` (28 subjects) or CMRx25 `Center012` (9, the only Philips in
   CMRx25): never train; each group split val:test = 1:2 (largest remainder) → 12 val / 25 test.
2. **Everyone else** (861) — stratified by `(source, vendor, centre, pathology_label)` from
   `manifest.csv`. Section totals are fixed first at 70/10/20 of 898 = 628/90/180; val/test quotas
   net of the holdouts (78 / 155) are distributed over strata proportionally to stratum size by
   largest remainder, so the totals are exact. Strata with < 3 subjects are train-only.
3. **Minimal churn** — within a stratum, a section quota is filled first with subjects already in that
   section in v1, then the surplus is drawn at random. This is a tie-break only; proportions are set
   by step 2.
4. Line order within a section is `pooled.txt`'s (so diffs against v1/pooled stay readable).

The script writes the split file and a `split_curated_v2` column into `manifest.csv`
(`--dry-run` prints the tables only).

## 3. Result

628 / 90 / 180. Holdouts: Canon 0/9/19, Philips-C012 0/3/6. 134 subjects changed section vs v1
(test→train 52, train→test 52, val→train 12, test→val 8, train→val 6, val→test 4).

Share of each section (train / val / test):

| group | train | val | test |
|---|---|---|---|
| ACDC Dijon | 13.4 | 12.2 | 11.7 |
| CMRx23 Fudan | 16.2 | 14.4 | 14.4 |
| CMRx24 | 24.5 | 21.1 | 21.1 |
| CMRx25 (all centres) | 27.4 | 25.6 | 25.6 |
| MNMs (all centres) | 18.5 | 26.7 | 27.2 |
| Canon | 0 | 10.0 | 10.6 |
| Philips | 8.4 | 11.1 | 11.1 |
| GE | 5.3 | 4.4 | 5.0 |
| Siemens | 73.1 | 65.6 | 62.2 |
| UIH | 13.2 | 8.9 | 11.1 |
| diseased | 47.0 | 51.1 | 50.0 |

Per source and per centre, val and test agree within ~1 pt; per vendor the largest gap is Siemens
(65.6 vs 62.2) because the holdouts are 13.3% of val vs 13.9% of test. Over the non-holdout pool
alone, every centre's share is within 1 pt across all three sections (independently verified
2026-09-11 by a subagent working only from the raw files). The train-vs-eval gap is exactly the
holdout (Canon/Philips-C012 are ~13% of val and test and 0% of train). Every CMRx25 centre except
UIH Center002 (4 subjects) now has ≥1 val and ≥1 test subject. Full per-centre table: run the
script with `--dry-run`.

## 4. Directory consolidation

- `evaluation/splits/` **deleted**. It held `ocmr_eval.txt` plus two symlinks to `training/splits/`
  that nothing read. The "can't be pulled into a pool if it lives outside training/splits" argument
  was empty: a subject trains only if `default.yaml`'s `split_file` names its file.
  `ocmr_eval.txt` moved to `training/splits/`; `sbatch/eval_pooled_val.sh` and `evaluation/README.md`
  updated.
- `training/splits/_archive/` ← `random_8_1_1`, `cmrx24only`, `default`, `overfit_p001`,
  `tiny_10_2`. All use the pre-pooling bare-id format (`CMRx24_Train_P001`), and **none of their
  subjects resolve on disk**. Still named by legacy `baselines/` scripts and by
  `build_pooled_split.py`'s 2024 continuity rule; those paths were updated.
- New `overfit_1.txt` (1 subject, train = val) and `tiny_10_2.txt`, both drawn from v2 in-section
  (no eval-subject leak), for pipeline sanity runs. Same purpose as the archived
  `overfit_p001` / `tiny_10_2`, in the live path format.
- `training/splits/README.md` lists every file's status. `pooled.txt` got a "superseded" header.

## 5. Consequences / still owed

- **`default.yaml` switched to v2** (2026-09-11): `split_file`, `dataset_name: curated898` (so
  `exp_name`/log dirs/wandb runs start a visibly new series), `limit_train_batches` 935→628,
  `limit_val_batches` 266→180, `log_visual_frequency.train` 935→628.
- **Curated-vs-uncurated ablation (docs/38 rule)**: two runs, identical config, same seed and epoch
  budget, differing only in the training set:
  - curated arm: `default.yaml` as is (train = v2's 628);
  - uncurated arm: `split_file=training/splits/pooled_uncurated_ablation.txt dataset_name=uncurated1066
    limit_train_batches=1066` — v2 **plus 438 of the 439 curated-out subjects back in `[train]`**
    (`ACDC_patient124` has D=21 slices, over the `img_nums=[20,20]` slot cap; it sat in `pooled.txt`'s
    test section so no run ever trained on it, and the first uncurated submission, job 61000433, crashed
    on it at epoch 0), `[val]`/`[test]` byte-identical to v2. **Not plain `pooled.txt`**: its train section holds 6 of v2's val and 52 of
    v2's test subjects (moved by the re-split), so that arm would be scored on subjects it trained on.
  Score both with the eval harness on v2 val (90); compare `recov_frac`↑, `psnr_motion`↑, `hole_frac`
  not ↑, plus the EF chain. In-training val curves are comparable too (same 90 subjects), but the
  uncurated arm sees 1.7× more subjects per epoch, so compare at equal epochs AND equal steps.
- **Eval bundles**: `evaluation/` bundles stamp `manifest["split"]` at build time and
  `paths.filter_by_split` drops off-split bundles. `sbatch/eval_pooled_val.sh`'s default split file is
  now v2, and so is `build_inputs/pooled.py:DEFAULT_SPLIT_FILE` (user-approved evaluation/ edit).
  Bundles for the 14 subjects that are val in v2
  but were not val in v1/pooled need building; the 16 that left val need nothing (they are dropped by the
  split gate).
- **Comparability**: every existing wandb run / checkpoint was trained on `pooled.txt`'s train set and
  validated on its val set. v2 val shares 76 of its 90 subjects with v1 val; of the 14 new ones, 6 were
  *train* subjects for every existing run. Treat v2 runs as a fresh series; don't compare val numbers across.
- **MIITT is now a held-out eval dataset**: `miitt_eval.txt` puts all 13 subjects under `[val]`
  (same pattern as `ocmr_eval.txt`); `sbatch/eval_pooled_val.sh` routes `miitt` to it. v2-trained
  models never see MIITT, so its paired real-time scans are a true OOD test. Checkpoints from the
  docs/78 `pooled_miitt` series trained on Volunteer1–5 and are clean only on the other 8.
  `pooled_miitt.txt` is kept as that series' record. MIITT eval bundles were stamped with
  `pooled_miitt.txt`'s sections (5 train / 5 test), so 10 of the 13 need rebuilding as `val` before
  the split gate will score them.
