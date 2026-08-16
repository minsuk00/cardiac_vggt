# `inference/_archive/` — the retired RTFB adapter stack

Moved here 2026-08-16, **not deleted**: these files are the provenance behind docs/06, docs/16,
docs/23, docs/28–35 and are cited by ~50 `tools/` experiment scripts. Nothing in the current
harness imports them, and they are **not expected to run** — several were already broken by the
native-z refactor before being moved.

## Why each one went

| file | reason |
|---|---|
| `adapters/` (`base`, `acdc`, `miitt`, `ocmr`, `__init__`) + `goettingen_adapter.py` | Every source now flows through `MRIDataset.get_data`. OCMR was the last holdout; `tools/convert_ocmr_to_12phase.py` closed it. |
| `inference.py` | Model loading now reads the run's own `run_meta.jsonl` instead of a hardcoded config. |
| `run_cmrxrecon.py`, `run_gated_ood.py` | Duplicated `evaluation/engine/run_vggt.py`. Their reason to exist was re-applying breathing per run; the frozen bundle replaces that. |
| `run_rtfb.py`, `render.py` | RTFB is out of scope for the gated + breathing-simulated harness. |
| `tests/test_eval_ocmr_equivalence.py`, `tests/_legacy_ocmr.py` | Guarded the OCMR adapter against a frozen snapshot; both sides are archived. Kept out of `tests/` so pytest does not collect them. |

## The deeper reason

`adapters/base.py` hand-wrote the geometry contract — `GRID_SHAPE = (12, 256, 256)`,
`D_CANON = 12`, `CANON_Z_SPACING_MM = 12.0`, `MM_PER_NORM = (178.5, 178.5, 66.0)` — and so did
`run_cmrxrecon.py` and `evaluation/engine/run_vggt.py`. Three copies of one contract is why they
drifted apart when training moved to native-z (docs/58), where every subject keeps its own `D` and
`dz` and there is no fixed cube at all. The retired z convention `z/(D-1)*2-1` produces
plausible-looking numbers rather than a crash: on ACDC_patient006 (D=11, dz=10 mm) it puts the top
plane at `z_norm = 1.0` where the physical convention `(z-(D-1)/2)*dz/90` gives `0.5556` — a 1.8x
through-plane stretch fed silently into both `z_embedder` and `scanner_coords`.

**Do not port anything out of here.** If you need a batch, call `MRIDataset.get_data`.

The two functions that DID survive — `percentile_scale` and `to_canonical_inplane`, which touch
only the in-plane axes — were moved to `inference/canonical_inplane.py` and are still importable.
