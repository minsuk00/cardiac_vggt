# 79 — `evaluation/` + `inference/` rebuilt on the native-z contract

> **TL;DR & takeaway**
> The eval harness was two years of drift behind training: it crashed on four API changes and,
> worse, still hand-wrote the **retired** z convention `z/(D-1)*2-1` in three places — a silent
> 1.8× through-plane stretch, not a crash. Fixed by deleting the second implementation rather than
> patching it: **OCMR was converted to the standard 12-phase layout** (`tools/convert_ocmr_to_12phase.py`
> → `OCMR_sax/`), which was the last thing keeping the adapter stack alive, so every source now
> flows through `MRIDataset.get_data` and the harness *is* `trainer.val_epoch` with breathing
> supplied by frozen pixels. **≈2,400 lines archived, ≈700 written.** Verified against the trainer's
> own `val_per_subject.csv`: same subject, same ROI, **29.62 dB vs its recorded 29.49** — and
> fault-injected (a wrong `z_scale` collapses it to 21.9). Adding val subjects is now incremental:
> both random draws are keyed on the subject **name**, proven by inserting a subject at the head of
> `[val]` and getting byte-identical output. **Two prior claims were measured and corrected** — the
> old builders were NOT mis-breathing, and the ~9 dB harness/trainer gap was a PSNR normalization
> difference, not an error.

## 1. What was broken

Four hard crashes: `splat_predictions` gained a required `z_scale`; `compute_volume_intensity_loss`
requires `batch["z_scale"]`; `gpu_augment_batch`'s respiratory path requires `batch["dz_mm"]`;
`MRIDataset.gt_grid_shape` is gone.

The silent failure mattered more. Three hand-written copies of the geometry contract
(`inference/adapters/base.py`, `inference/run_cmrxrecon.py`, `evaluation/engine/run_vggt.py`) still
used the retired index-normalized z. On `ACDC_patient006` (D=11, dz=10 mm) the top plane should sit
at `z_norm = 0.5556`; the old formula puts it at `1.0`. That is a 1.8× stretch fed to both
`z_embedder` and `scanner_coords`, producing plausible numbers rather than an error. A fourth copy
lived in the **scorer**: `assemble_and_gif.load_canon` force-resampled every volume onto a hardcoded
`(256,256,12)` @ `(1.4,1.4,12.0)` grid, re-snapping each subject's real stack *after* the model had
reconstructed it correctly.

Root cause worth naming: **one contract, four implementations.** They drifted together and would
drift again.

## 2. The fix: delete the second implementation

`MRIDataset.get_data` already emits everything correctly. The harness now asks it for a batch and
swaps in the frozen pixels:

```python
batch = ds.get_data(seq_index=name_seed(source, subject))
batch["phases"] = frozen_bundle          # clean/ or breath/ pixels, (T, D, 256, 256)
batch.pop("images")                      # ← load-bearing, see below
batch = gpu_augment_batch(batch, None, device, respiratory_cfg=None, train=False)
V, _ = _splat_preds_native(preds, batch, (D, 256, 256), z_scale)
```

**`batch.pop("images")` is load-bearing.** `gpu_augment_batch` rebuilds `images` only when absent,
but always rebuilds `images_splat`. Leaving the dataset's clean `images` in place would feed the
model CLEAN slices while the splat rendered BREATHED content — invisible, with nothing to catch it.

`_splat_preds_native` (not `splat_predictions`) because it renders from `images_splat` at native
resolution, which is what the trainer's loss does; the model-resolution fallback exists precisely
for hand-built batches like the ones now archived.

### 2.1 OCMR conversion unlocked the deletion

OCMR was the only source without an `MRIDataset` entry, and the sole reason the adapter stack,
`build_canonical_bundle` and `geom.place_to_canonical` had to survive.
`tools/convert_ocmr_to_12phase.py` writes `scratch/data/OCMR_sax/OCMR_<series>/sax/` in the standard
layout (source tree read-only; verified by mtime). 8 subjects, 182 native frames → 96.

Two differences from the MIITT converter, both forced by the data:

| | |
|---|---|
| source is one 4-D cine, not per-frame files | frames are sliced out and re-saved with the SOURCE affine — no resample, no re-framing (OCMR already ships axis-aligned LPS) |
| spacing genuinely varies (in-plane 1.98–2.25, pitch 7.8–10.0 mm) | MIITT's fixed `EXPECTED_ZOOMS` equality becomes a range check |

**Slice order needed a per-subject flip.** OCMR is mixed — measured **3 apex-first, 5 base-first** —
so it needs the standardization the rest of the cohort got on 2026-07-31. Done *inside the
converter* rather than via `tools/fix_slice_order.py`, because that tool holds a single cohort-wide
provenance sidecar for 893 subjects and refuses to re-run without `--force`, which would overwrite
their revert record. This tree is regenerable from read-only source in ~2 min, so it needs no
sidecar; `convert_meta.json` records the flip in `reframe.flips[2]` plus a `slice_order` block.
An undetermined subject RAISES rather than being guessed.

ES quantization onto the 12-grid costs ≤5.0 EF points (worst `fs_0074`: 63.3 → 58.3), in line with
the ~4.1 pt worst case already measured on ACDC. EDV is unchanged for all 8 — only ESV moves, which
is the expected signature.

## 3. What was archived (moved, never deleted)

| | lines / size |
|---|---|
| `inference/{adapters,inference.py,run_cmrxrecon.py,run_gated_ood.py,run_rtfb.py,render.py}` | ≈1,700 |
| `evaluation/engine/build_inputs/{acdc,miitt,ocmr,geom}.py` | 653 |
| `evaluation/{results,MODELS.md,models.json}` (19 pre-refactor arms) | — |
| `scratch/eval/{cmrxrecon,acdc,miitt,ocmr}` → `_archive_prenativez_20260712/` | **264 GB** |

Each archive carries a README explaining why. The bundles are stale three ways over:
fixed-12-plane geometry, a cohort re-flipped/renamed/re-laid-out on 2026-07-31, and pre-refactor
checkpoints.

`percentile_scale` + `to_canonical_inplane` are the only part of `adapters/base.py` that survived
native-z (in-plane axes only), so they moved to `inference/canonical_inplane.py` rather than into
the archive — `baselines/fetal_cmr_4d/` still imports them.

## 4. Protocol comes from the checkpoint, not `default.yaml`

Every previous eval script read its protocol via `compose(config_name="default")` — i.e. from
whatever `training/config/` looks like *today*. For the checkpoint under test that resolves to
`img_size 518` / tier `moderate`, while the run actually trained at **224 / aggressive**. Nothing
crashes; you just score the model at the wrong input resolution under a protocol it never saw.

`inference/load_run.py::load_model_from_run` reads the run's own `run_meta.jsonl`. Precedence
`config.model.<k>` → `config.<k>` → default handles pre-`40b652a` runs where `img_size` lived at the
top level and `backbone`/`patch_size` were absent; `patch_size` finally derives from the backbone
(docs/77). The six retired VGGT kwargs are dropped explicitly rather than left to `**kwargs` — which
is also why `--refiner` is gone from every CLI: it was a silent no-op.

`--regime` and `--continuous-z` are deleted for the same reason. They are properties of the run, so
the eval regime can no longer disagree with training.

## 5. Adding val subjects is incremental

Both draws are keyed on the subject **name**, never its position:

| | seeded by |
|---|---|
| breathing realization | `sha256("<source>/<subject>")` |
| input slot draw | the same hash, via a **one-subject** `MRIDataset` |

The one-subject dataset is needed because `get_data` uses `seq_index` for BOTH the subject index
(`seq_index % len(subjects)`) and the val RNG seed (`random.Random(seq_index)`). With a single
subject the index term is always 0, freeing `seq_index` to be the name hash. No `training/` change.

**Verified, not argued:** inserting a subject at the head of `[val]` — the worst case, shifting
every `seq_index` — leaves an existing subject's seed and every NIfTI byte identical. Existing
bundles are skipped, so a re-run does only the new work.

## 6. Verification

| check | result |
|---|---|
| OCMR pixels vs source after the recorded flip | bit-identical, 8/8 |
| OCMR apex-at-z0 on disk; canonical/native z agree | 8/8; `preflight_zdir` ok |
| ED lands on frame 0 for all 8 | yes (`cardiac_phase.csv` regenerated: +8 rows, **0 changed**) |
| `MRIDataset` loads OCMR, native-z contract | 16/16 sweep entries, `z_norm` = `(D-1)/2·dz/90` exactly |
| bundle rebuild determinism | byte-identical |
| position independence | byte-identical under head-insertion |
| **harness vs trainer, bbox ROI** | **30.78 vs 30.39 dB** |
| **harness vs trainer, heart ROI** | **29.62 vs 29.49 dB** |
| panels' Δz vs `resp_diag` | max abs diff **5.15e-05 mm** over 11 slots |
| `pytest tests/` | 363 passed (−4 = the archived adapter-equivalence file) |

**Fault-injected** (a green check that has never been shown to fire is worthless): a wrong
`z_scale` collapses PSNR 30.78 → 21.93 / 20.18; a reversed seg flips the apex-first detector; a
reversed native seg makes `preflight_zdir` report REVERSED (corr −0.807); the missing-sibling
detector fires when the ROI file is hidden.

## 7. Two claims measured and CORRECTED

Both were plausible mechanism stories that the measurement refuted. Recorded because the
plausible version is what a reader will otherwise re-derive.

**7a. The old builders were NOT mis-breathing.** The plan asserted they silently took the per-slot
iid branch instead of the configured `group_by_burst` one, since `n_planes` is inert without
`group_ids`. Measured: with `S == D == n_planes`, the burst branch's
`gather(rand((B,P)), arange(D))` consumes the generator in exactly the same order as the per-slot
branch's `rand((B,S))`, and with `per_slot=False` the amplitude draw matches too — **bit-identical
displacement**. The match is real. It rests on *both* coincidences though, and breaks if either
goes (`per_slot` at its config default diverges; `S != D` diverges), so `pooled.py` passes the
grouping explicitly and leaves `per_slot` alone — structural agreement instead of accidental.
Ironically the old builders' own `rcfg.per_slot = False` line was the one thing that *would* have
diverged from training had `group_by_burst` ever been turned off.

**7b. The ~9 dB harness/trainer gap is a normalization difference, not an error.** The harness's
heart-ROI PSNR read 22.4 dB against the trainer's 29.5 dB for the same subject and the *same*
`heart_roi_canonical`. Cause: the harness normalizes by the GT's max **inside the ROI**, the trainer
by peak = 1.0. Measured `gt[roi].max() = 0.353` → exactly **−9.04 dB**. `psnr_unit_peak` is now
emitted alongside so the two suites are reconcilable without re-deriving this.

## 8. Open

- **Baselines** (SVRTK / NeSVoR / NiftyMIC) have not been re-run on the new bundles — model-only
  first pass. The bundles are built and frozen, so they can be run at any time against identical
  inputs.
- **`scratch/eval/_ef_ood/`** (7 GB) derives from the archived bundles and is left in place, flagged
  stale in that README.
- **`baselines/fetal_cmr_4d/selfgate_lvarea_extract.py`** still imports the archived `MIITTAdapter`.
  It was already stale (placeholder MIITT spacing, superseded by docs/78) and feeds a baseline that
  was never built; flagged in place rather than silently fixed.
- **`evaluation/analysis/{compare_methods,compare_bars,compare_table,breathing_pred_vs_applied}.py`**
  had only their dataset-name defaults and OOD wording updated; they have not been exercised
  end-to-end on the new bundles.
