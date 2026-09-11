# 96 — Misalignment curation, final: automatic score + by-eye review → `training/splits/pooled_curated_v1.txt` (898 of 1337 subjects)

> **TL;DR & takeaway** — The pooled cohort was curated for inter-slice breath-hold misalignment in
> two stages: (1) the automatic two-estimator severity score of docs/95 (v3.1 rules), keeping
> **clean + mild (≤5 mm)** and dropping moderate/severe; (2) the user's **by-eye review of every
> subject** in the per-band galleries, recorded as 114 overrides (24 exclude, 90 include) in
> `training/splits/pooled_curated_v1_manual_overrides.csv`. Result: **`training/splits/pooled_curated_v1.txt`
> = 898 subjects, 622 train / 92 val / 184 test** (67% of 1337), **all three sections filtered**
> (val/test GT carries the same corruption). Every centre/vendor group survives (lowest MNMs centre 1
> Siemens 43%, all others ≥50%). Verified by two independent agents that re-derived the file from the raw
> inputs and the user's verbatim ID lists (byte-identical). **Not yet wired into `default.yaml`** — the
> split-file swap and a possible centre-stratified re-split are a separate, pending decision (see §6).
> Regenerate with `tools/misalign_v2/analyze.py` (§5); the score itself is docs/95.

## 1. Why

docs/94 → docs/95 established that stacked-slice GT volumes contain real in-plane breath-hold
misalignment (38% of subjects have a >5 mm discontinuity, 21% >8 mm). Training against a misaligned
GT teaches the model to reproduce the artefact or adds noise it cannot resolve; scoring against one
penalises a correct reconstruction. The user's decision (2026-09-11): **curate at the subject level,
train/val/test alike, keep clean+mild, and check every bucket by eye** — "our curation pipeline is
automation + manual inspection". Slice-level masking and re-registration were rejected in docs/95.

## 2. Pipeline

```
metric_v2.py  (GPU sweep, 1337 subj × 12 phases, ±40 mm NCC + seg centroids)   -> temp/misalign_v2/full_v3.csv
analyze.py    (v3.1 consensus rules -> severity, band; + manual overrides)         -> subjects_v3.csv, curated_sev{5,8,10}mm.txt
render_slab.py + build_band_gallery.py  (every subject, one gallery per band)      -> _html/95d_gallery_{clean,mild,moderate,severe,unmeasurable}.html
USER by-eye review of each gallery  -> manual_overrides.csv (exclude / include)
analyze.py again -> curated_sev5mm.txt == training/splits/pooled_curated_v1.txt
2 independent verifier agents re-derive the file from pooled.txt + subjects_v3.csv + the ID lists
```

**Automatic rule (docs/95 v3.1, summarised).** Two independent estimators per slice — epicardial
(LV+myo) centroid from the nnU-Net segmentation, and NCC translation registration of the image to its
two neighbours (±40 mm window) — both reduced to the median over the 12 cardiac phases. Two measures:
**spike** (slice vs the mean of its two neighbours; catches one slipped slice) and **step** (adjacent
pair jump minus the subject's own median drift; catches a shifted block). Value = min(seg, img) when
both answer; a registration pinned at its search bound counts as "≥40 mm" and the seg value is used; a
featureless registration leaves the slice unscored; a step whose two vectors disagree in direction keeps
min(seg, img) if both ≥8 mm and is zeroed otherwise. Subject severity = max over scored slices/pairs.
Bands: clean ≤3, mild 3–5, moderate 5–8, severe >8 mm. **Unmeasurable** = fewer than 3 gated slices
and fewer than 3 gated pairs (short stack, small heart, empty segmentation); 18 subjects, kept by the
automatic rule, reviewed by eye separately.

**Manual stage.** The user reviewed the galleries and gave verbatim ID lists:
- clean → exclude 3, mild → exclude 16 ("these are also bad");
- moderate → include 66, severe → include 24 ("fine, keep"; everything else in those bands is dropped);
- unmeasurable → exclude 5 (after all 18 were rendered on request: `temp/misalign_v2/audit/unmeasurable_18.png`).
Each override row records `rel, action, auto_band, auto_severity_mm, split, reviewer, date, note`.
`exclude` always drops, `include` always keeps, regardless of the automatic score.

## 3. Result

| section | pooled | curated | kept |
|---|---|---|---|
| train | 935 | 622 | 67% |
| val | 133 | 92 | 69% |
| test | 269 | 184 | 68% |
| **all** | **1337** | **898** | **67%** |

By automatic band (kept / total): clean 486/489, mild 309/325, moderate 66/224, severe 24/281,
unmeasurable 13/18. By source: ACDC 116/148, CMRx23 141/195, CMRx24 211/294, CMRx25 241/359,
MNMs 189/341.

Per centre/vendor (whole dataset; figure `temp/misalign_v2/audit/curation_per_centre_kept_dropped.png`,
table `temp/misalign_v2/curation_per_centre.csv`):

| centre_key | total | kept | kept % |
|---|---|---|---|
| CMRxRecon2024/Fudan/Siemens-Vida | 294 | 211 | 72 |
| CMRxRecon2023/Fudan/Siemens-Vida | 195 | 141 | 72 |
| ACDC/Dijon/Siemens | 148 | 116 | 78 |
| MNMs/centre1/Siemens | 95 | 41 | 43 |
| MNMs/centre2/Philips | 73 | 38 | 52 |
| MNMs/centre4/GE | 72 | 46 | 64 |
| CMRx25/Center001/UIH-umr780-30T | 61 | 38 | 62 |
| CMRx25/Center006/Siemens-Prisma-30T | 60 | 38 | 63 |
| MNMs/centre3/Philips | 51 | 36 | 71 |
| MNMs/centre5/Canon | 50 | 28 | 56 |
| CMRx25/Center004/Siemens-Aera-15T | 44 | 36 | 82 |
| CMRx25/Center004/UIH-umr680-15T | 34 | 19 | 56 |
| CMRx25/Center003/UIH-umr880-30T | 28 | 19 | 68 |
| CMRx25/Center005/Siemens-Vida-30T | 25 | 19 | 76 |
| CMRx25/Center006/UIH-umr790-30T | 23 | 15 | 65 |
| CMRx25/Center002/Siemens-CIMAX-30T | 21 | 11 | 52 |
| CMRx25/Center005/UIH-umr670-15T | 20 | 14 | 70 |
| CMRx25/Center012/Philips-IngeniaCX-30T | 12 | 9 | 75 |
| CMRx25/Center001/Siemens-Vida-30T | 10 | 9 | 90 |
| CMRx25/Center002/UIH-umr880-30T | 8 | 4 | 50 |
| CMRx25/Center007/Siemens-Sola-15T | 7 | 6 | 86 |
| CMRx25/Center003/UIH-umr670-15T | 4 | 2 | 50 |
| CMRx25/Center007/Siemens-Prisma-30T | 2 | 2 | 100 |

No centre is wiped out; the groups that shrink most are the ones worst at misalignment (MNMs centre 1,
CMRx25 CIMAX/UIH-umr680), which is the intended effect, not a sampling artefact.

## 4. Verification

- Two independent subagents, given only `pooled.txt`, `subjects_v3.csv` and the user's verbatim ID
  lists (not `analyze.py`), each re-derived the expected file — one by set algebra, one by a
  line-by-line forward walk plus a reverse justification of every kept line. Both: byte-identical
  sequence, every ID resolves to exactly one subject, all excludes actually removed something, all
  includes actually added something, no section moves, no duplicates, overrides CSV exact. (Run before
  the 5 unmeasurable exclusions were added; those were applied by the same code path and checked by
  grep.)
- Random by-eye audit of 48 clean + 24 mild by the agent found no missed jump; the user's own full
  pass found 19, which is the reason the manual stage exists.
- Automatic-score correctness: `tools/misalign_v2/test_fault_injection.py` (16 checks) — docs/95.

## 5. Files and how to regenerate

**Permanent (git-tracked):**
- `training/splits/pooled_curated_v1.txt` — the final split file (pooled.txt format; header names the
  rule, the override counts and the unmeasurable subjects kept). Point `data.split_file` at it to use.
- `training/splits/pooled_curated_v1_manual_overrides.csv` — the 114 by-eye decisions.
- `tools/misalign_v2/` — `metric_v2.py`, `analyze.py`, `test_fault_injection.py`, `render_slab.py`,
  `build_band_gallery.py`, `build_categorized_examples.py`, `build_html.py`, `manual_overrides.csv`
  (the working copy `analyze.py` reads; keep identical to the `training/splits/` copy).
- `docs/95` (measurement + rule history), this doc.

**Regenerable (`temp/`, gitignored):** `temp/misalign_v2/{full_v3.csv, subjects_v3.csv, slices_v3.csv,
steps_v3.csv, summary.json, curated_sev{5,8,10}mm.txt, band_changes_vs_v3.csv, curation_per_centre.csv,
slabs/ (1337 panels), audit/}`, review bundle `temp/misalign_review_v31/`, previous sweep
`temp/misalign_v2/_v3_superseded/`. HTML galleries in `_html/95*.html`.

```bash
PY=/home/minsukc/micromamba/envs/svr/bin/python          # `micromamba run` hits a lock when parallel
$PY tools/misalign_v2/metric_v2.py temp/misalign_v2/full_v3.csv        # ~35 min on one A40 (only if the metric changes)
$PY tools/misalign_v2/test_fault_injection.py                          # must print ALL PASS
$PY tools/misalign_v2/analyze.py temp/misalign_v2/full_v3.csv          # seconds; applies manual_overrides.csv
diff temp/misalign_v2/curated_sev5mm.txt training/splits/pooled_curated_v1.txt   # header may differ, body must not
$PY tools/misalign_v2/render_slab.py <subjects.csv> <slices.csv> <steps.csv> temp/misalign_v2/slabs [rel ...]  # ~8 s/subject
$PY tools/misalign_v2/build_band_gallery.py {clean|mild|moderate|severe|unmeasurable}
```
To add a by-eye decision: append a row to `manual_overrides.csv` (both copies), rerun `analyze.py`,
copy `curated_sev5mm.txt` over `pooled_curated_v1.txt`.

## 6. Open / not done

- **`default.yaml` still points at `pooled.txt`.** Switching to `pooled_curated_v1.txt` and the docs/38
  ablation (curated vs uncurated) are pending the user's go.
- **Re-split.** Two facts of the *existing* split surfaced during the per-centre check: MNMs centre 5
  (Canon, 50 subj) and CMRx25 Center012 (Philips, 12) have **zero train subjects** (val/test only), and
  most CMRx25 centres have 1–7 val/test subjects, so per-centre evaluation on them is not meaningful.
  A centre-stratified re-split would fix this at the cost of comparability with every existing run.
  Deferred by the user until the curation was signed off; the curation is now signed off.
- 38 subjects have `weak_step_evidence` (every scored pair zeroed as a seg/img contradiction); their
  score rests on spike alone. Flagged in `subjects_v3.csv`, not acted on.
- The `include` decisions keep 90 moderate/severe-scored subjects; if the docs/38 ablation shows the
  curated arm is not cleaner, those are the first place to look.
