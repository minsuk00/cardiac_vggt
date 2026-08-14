# W&B visualization coverage and intensity-display controls

> **TL;DR & takeaway** — Validation visualization now covers the five pooled sources and the
> requested scanner vendors, logs ordinary reconstruction panels at ES only, preserves both ED
> and ES in dedicated panels, and renders one 12-phase GIF per selected subject every five
> validation epochs under `media_filmstrip/`. All grayscale MRI media use **display-only gamma
> 0.7**; error, motion, DVF, mask, loss, metric, and saved-volume values remain linear/raw. Input
> normalization is independently configurable through `intensity_percentiles` and still defaults
> to the checkpoint-compatible **phase-0 nonzero-FOV p0.5–p99.9**. The explored p2–p98 setting was
> a visualization experiment only and is **not** the shipped default.

## 1. Why this changed

The old fixed validation indices were not representative under the sorted pooled split: several
indices landed in ACDC, leaving later sources and held-out vendors invisible. The ordinary
ED/ES-sweep panels also reused one W&B key for ED and ES, so the later ES image was effectively the
visible one without making that policy explicit. Finally, linearly displayed normalized MR images
were too dark in W&B, especially for CMRx24.

This pass made the intended diagnostic policy explicit without changing model computation.

## 2. Selected validation subjects

`training/train_utils/val_logging.py::pick_visual_indices` and
`TrainerVizMixin._ED_ES_SUBJECTS` deterministically select:

| Source | Requested coverage | Pooled-val index used in the verified run |
|---|---|---:|
| ACDC | Siemens | 0 |
| CMRx23 | Siemens | 15 |
| CMRx24 | Siemens | 34 |
| CMRx25 | Siemens, UIH, Philips | 63, 64, 70 |
| M&Ms-1 | GE, Siemens, Canon | 101, 102, 104 |

The real subjects resolved to `ACDC_patient006`, `CMRx23_Test_P002`,
`CMRx24_Test_P012`, three CMRx25 vendor examples, and `MNMs_A9F3T5`, `MNMs_B0H7V0`,
`MNMs_C5L0R0`. Canon is intentionally represented because it is absent from M&Ms-1 training.
M&Ms Philips was deliberately not requested.

Vendor metadata comes from the manifest adjacent to the split file. The active pooled manifest is
complete, as are all current repository splits. A partially populated future/custom manifest could
omit a targeted source instead of falling back to one source-level example; this diagnostic-only
edge case was reviewed and deliberately left unfixed because it cannot occur with the current
cohort artifacts.

## 3. Phase and panel policy

With `ef_val_sweep: true`, `MRIDataset.val_targets` is exactly:

```text
[all subjects at their own ED] + [all subjects at their own ES]
```

The phase choices come from `cardiac_phase.csv`; they are not fixed global frame numbers.

- Ordinary per-subject `Volume`, `DVF`, and `Lookup` panels log **ES only**. The dispatcher maps
  the sweep entry back to its subject and explicitly gates on the second half of `val_targets`.
- `media_val_ED_ES/` logs both the subject-specific ED and ES reconstructions together, one panel
  per selected subject.
- The startup motion-mask background is correctly labelled **`t=0`**, not ED. It does not consult
  `cardiac_phase.csv`, so calling it ED would be false for subjects whose ED is not frame zero.
- The train augmentation before/after panel remains separate under `media_others/`.

## 4. Cardiac-cycle GIFs

Every selected subject gets one animated GIF:

```text
media_filmstrip/Val_Visuals_subj{index}_{subject}_cardiac_cycle_gif
```

Each GIF contains all **12 cardiac phases**. Every frame has two rows: `V_gt` above and `V_canon`
below, with five z-planes evenly spanning that subject's native stack. Subjects can have different
native `D`, but the cardiac phase count remains 12. Unique keys prevent one subject or phase from
replacing another.

The default cadence is `logging.filmstrip_every_n_val_epochs: 5`. The cheaper ED/ES and
augmentation panels use the independent `visual_panels_every_n_val_epochs: 3` cadence. The GIF
cadence was chosen after measuring the exact renderer on an A40: the nine GIFs took **173.824 s
(2m54s) total**, with individual subjects taking **14.39–25.81 s** depending primarily on native
stack depth and model work.

## 5. Display-only gamma correction

All grayscale MRI intensity media use the shared helper
`training/trainer_viz.py::_display_gamma` with gamma 0.7:

```text
I_display = clip(I / vmax, 0, 1) ** 0.7
```

For values in `(0,1)`, a power below one raises dark and mid-range values while preserving black,
white, and intensity ordering. The helper is applied only after tensors have been detached and
converted to NumPy. It is used for:

- input slices;
- `V_gt` and `V_canon` volume rows;
- ED/ES intensity rows;
- lookup input, reconstructed, and GT intensity columns;
- the grayscale background of the motion-mask panel;
- augmentation before/after MRI images;
- GIF byte generation.

Comparative GT/prediction panels share the same pre-gamma `vmax`. The following remain linear and
unchanged: signed differences, lookup errors, motion magnitude, DVF components, masks/overlays,
losses, metrics, model inputs, cached tensors, and saved NIfTI volumes.

A non-finite future filmstrip input could poison the shared display scale and produce a black GIF
instead of a clear failure warning. This did not occur in the verified run and was deliberately
left as optional diagnostic hardening; it cannot affect training or numerical evaluation.

## 6. Configurable input normalization

Input normalization and display gamma are separate operations. The former changes the actual model
problem; the latter changes only pixels sent to W&B.

The root config now exposes:

```yaml
intensity_percentiles: [0.5, 99.9]
```

Both train and validation `MRIDataset` instances receive this value. The dataset validates
`0 <= lower < upper <= 100`, converts both values to floats, and passes them to
`get_canonical_transforms`. The transform computes the two percentiles once from phase 0's exact
nonzero FOV voxels after spatial preprocessing, then uses that same `(vmin, vmax)` to clip and scale
all 12 phases into `[0,1]`. This preserves cross-phase intensity relationships.

Example override:

```bash
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config default intensity_percentiles='[2,98]'
```

Changing this setting changes model inputs, targets, loss scale, PSNR scale, cache contents, and
checkpoint comparability. `cache_signature(lower, upper)` therefore gives each percentile pair a
different `PersistentDataset` cache directory, and the resolved config plus cache signature are
written to `run_meta.jsonl`.

The default `[0.5,99.9]` is numerically and cache-signature compatible with the prior fixed
implementation. The p2–p98 comparison made images much brighter but clipped more highlights in an
already-bright M&Ms subject. It remains an explicit experimental override, not a visualization
window and not the default.

## 7. Verification record

The final post-change verification was:

- **352/352 tests passed** (`micromamba run -n svr python -m pytest tests/`).
- Focused visualization/normalization suite: **111/111 passed**.
- Hydra composition probe: default and `[2,98]` overrides reached both train and val configs.
- Real-data preprocessing probe on the same subject:
  - p0.5–p99.9 cache signature `f937056bb3`, normalized mean `0.1495`;
  - p2–p98 cache signature `038ce5a12f`, normalized mean `0.2117`.
- Full A40 pooled validation loaded the checkpoint strictly with **0 missing / 0 unexpected** keys
  and completed all **266/266** subject-specific ED/ES targets.
- The offline W&B run produced **9 ED/ES panels, 27 ordinary Volume/DVF/Lookup panels, one
  motion-mask panel, and 9 twelve-phase GIFs**, with no CUDA, rendering, or logging exception.
- Three independent prove-it reviewers traced gamma isolation, normalization/config/cache flow,
  visual selection, phase gating, cadence, and compatibility; the two intentionally ignored
  diagnostic edge cases above survived adversarial review but do not occur in the current data/run.

Validation artifacts are under `temp/gamma07_full_validation/` and the offline W&B media under its
`wandb/wandb/offline-run-*` directory.
