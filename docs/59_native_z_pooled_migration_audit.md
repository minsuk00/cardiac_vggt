# 59 — Audit of the native-z + pooled-cohort migration (`/prove-it`, 2026-07-31)

> **TL;DR & takeaway**
>
> A 7-reviewer `/prove-it` audit of everything docs/58 changed in the **training path** (native-z grid,
> physical `z_norm`, pooled 1343-subject cohort, refiner removal). **The migration's core is correct**:
> the `z_norm → pz` inverse round-trips to `2e-15`, the x/y 518↔256 map is exact, breathing is applied
> in true millimetres at every pitch, all 1343 subjects load with all 12 frames, no subject exceeds
> `Z_HALF_MM=90`, `pytest` is green (215), and a real `torchrun` on the pooled config runs end-to-end
> with zero swallowed exceptions.
>
> **But three things need fixing before a long run.**
>
> 1. **A float32 cancellation drops the ENTIRE apex plane for 56 of 1343 subjects.** `pz` for plane 0
>    lands at `−4.77e-07`, and `splat_to_volume`'s hard `pz >= 0` gate discards the whole slice — not a
>    fractional weight. Measured cost: **−9.6 dB and −5.0 dB** identity PSNR on the two affected *val*
>    subjects. It deterministically corrupts the Δ=0 identity and oracle splats behind
>    **`metric_recov_frac_heart`**, the primary ship-decision metric. This is a **native-z regression** —
>    the old grid-relative formula gave `pz = 0` exactly. **~4 lines to fix.**
> 2. **`tools/gate_native_z_identity.py` cannot detect it.** It masks PSNR on `coverage > 0.5`, so a
>    fully-dropped plane is excluded from the metric by construction. It reports **120.00 dB on a
>    subject whose plane 0 has coverage 0.0**. A gate that passes the broken case is not evidence.
> 3. **Three pairs of byte-identical duplicate subjects straddle split boundaries** — train↔val,
>    train↔test, val↔test. **Inherited from ACDC and M&Ms themselves** (raw voxel arrays are bit-identical;
>    our converter is innocent), and their metadata self-contradicts (same pixels labelled Philips/centre-2
>    *and* GE/centre-4), which undercuts the "Canon only in val/test ⇒ free unseen-vendor test" premise.
>
> Everything else found is medium-or-lower and listed in the triage table in §2. Nothing here invalidates
> docs/58's design decisions — native-z is sound; these are implementation and data-hygiene defects.
>
> **UPDATE 2026-07-31 — F1, F2, F3, F5, F6, F7 are FIXED and verified.** All three blockers are
> closed: the apex-plane drop (fault-injection proven, 9.96 → 119.92 dB, bit-identical on unaffected
> subjects and `fullgraph` torch.compile clean), the blind gate (now reports 31.23 dB FAIL on the
> subject it used to pass at a fake 120.00 dB), and the duplicate leakage (6 subjects excluded,
> cohort 1343 → 1337 at 935/133/269). Plus the three one-liners F5/F6/F7. **A long run is no longer
> blocked.** Per-finding fix records are inline below (struck-through original text + a FIXED note);
> what remains is listed in §9.

---

## 1. Scope and method

**Target:** the training path as changed by docs/58 (`0981b1b..HEAD`):
`vggt/utils/splat.py`, `vggt/models/vggt.py`, `training/data/{preprocess,datasets/mri_dataset,composed_dataset,gpu_aug,respiratory}.py`,
`training/{loss,trainer,trainer_viz,ef_eval}.py`, `training/config/*.yaml`, `training/splits/{pooled.txt,manifest.csv}`.

Deliberately **out of scope** (per the standing scoping rules, and unchanged by this audit):
`inference/`, `evaluation/`, `baselines/`, `tools/`. These fail **loudly** (`TypeError`/`AttributeError`)
rather than silently, because `z_scale` and `spacing` were made required arguments — that design worked.

**Method:** 7 independent reviewers, each reading the *whole* target with a different assigned lens
(coordinate/scale math · variable-`D`/`S` shape propagation · batch-key plumbing · respiratory & aug
physics · pooled-cohort data integrity · config coherence & refiner residue · numerical/degenerate edges),
then adversarial verification of each suspected bug, then **runtime confirmation by the orchestrator on an
A40 GPU node**. Every finding below is labelled with how it was established.

**Confidence labels used throughout:**
`MEASURED` = reproduced by running code in this session · `READ` = established by reading source ·
`DISPUTED` = reviewers disagreed, unresolved · `UNVERIFIED` = plausible, not confirmed.

---

## 2. Triage: priority × effort

Ordered by **fix-first value**. "Effort" is implementation effort only, not the cost of re-running training.

| # | Finding | Priority | Effort | Silent? | Regression from docs/58? | **Status** |
|---|---|---|---|---|---|---|
| **F1** | fp32 drops the entire apex plane, 56/1343 subjects | 🔴 **Critical** | **~4 lines** | **Yes** | **Yes** | ✅ **FIXED** |
| **F2** | Acceptance gate structurally blind to F1 | 🔴 **High** | **~3 lines** | Yes | Yes (gate is new) | ✅ **FIXED** |
| **F3** | 3 duplicate pairs straddle train/val/test | 🟠 **High** | **~3 lines** (+ a decision) | Yes | Yes (pooling is new) | ✅ **FIXED** |
| F4 | No test can distinguish the respiratory dz fix from its reversal | 🟠 Medium | ~15 lines | Yes | Yes | ✅ **FIXED** |
| F5 | monai cache is 66% dead weight (~54 GB /tmp, 3× I/O) | 🟠 Medium | **1 line** | No (perf) | No (pre-existing, scaled up) | ✅ **FIXED** |
| F6 | `len_train=1000` vs 940 → ACDC oversampled 1.60× every epoch | 🟠 Medium | **1 line** | Yes | Yes | ✅ **FIXED** |
| F7 | `z_scale`/`dz` read `[0]`; `B>1` would silently mix scales | 🟠 Medium | **1 line** (assert) | Yes (if triggered) | Yes | ✅ **FIXED** |
| F8 | Val has **zero** 8 mm subjects; 93% of val is 10/12 mm | 🟠 Medium | Re-split (hard) | Yes | Yes | ⬜ **ACCEPTED** |
| F9 | `one_frame_per_slice` makes `num_slices`/`img_nums`/`max_img_per_gpu` stale | 🟡 Medium | Comments + 1 assert | Partly | Yes | ✅ **FIXED** |
| F10 | `ef_val_sweep` ED/ES are seg-derived, disagree with source labels | 🟡 Low-Med | Medium | Yes | Partly | ⬜ WONTFIX |
| F11 | `metric_ssim_3d_full` is now `D`-dependent | 🟡 Low-Med | Document, or drop metric | Yes | Yes | ✅ **FIXED** |
| F12 | Refiner-era ckpt loads, then dies on optimizer state | 🟡 Low | ~2 lines (log key names) | No (loud) | Yes | ⬜ WONTFIX |
| F13 | `val_volumes/` now ~925 MB/epoch (was ~207 MB) | 🟡 Low | 1 config line | No | Yes | ⬜ accepted |
| F14 | `ef_eval.py` hardcodes 12 mm — benign, but now fixed | 🟢 Low | ~10 lines | Cosmetic | Yes | ✅ **FIXED 2026-08-01** |
| F15 | `splat.py`'s `0.0 * NaN = NaN` poisons the whole volume | 🟡 Low-Med | ~2 lines | Yes→loud | **No** (pre-existing) | ✅ **FIXED** |
| F16 | Respiratory zero-padding blanks/dims basal slices | 🟡 Low-Med | Design decision | Yes | **No** (pre-existing) | ⬜ **ACCEPTED** |
| F17 | `_find_subjects` warn-and-skip has no post-condition | 🟢 Low | ~3 lines | Yes (if triggered) | Partly | ✅ **FIXED** |
| F18 | `Z_HALF_MM=90` has only 5.9% headroom; bare `assert` | 🟢 Low | ~2 lines | No (crash) | Yes | ✅ **FIXED** |
| F19 | `S=D` could exceed `max_img_per_gpu` on a re-seeded split | 🟢 Low | ~2 lines | No (OOM) | Yes | ✅ **FIXED** |
| F20 | Comment/docstring drift (`D=12`, "786K voxels", freeze rationale) | 🟢 Cosmetic | Trivial | No | Yes | ✅ **FIXED** |
| **F21** | **`ef_val_sweep`'s entire ES half unreachable — EF cannot be computed** | 🔴 **Critical** | **1 line** | **Yes** | **No — introduced BY the F6 fix** | ✅ **FIXED** |

~~**Recommended fix order:** F1 → F2 (F2 is what stops F1-class bugs recurring) → F3 → F5, F6, F7 (all
one-liners) → F4 → the rest as convenient.~~

**DONE 2026-07-31, in exactly that order: F1 → F2 → F3 → F5, F6, F7**, then a second pass closing
**F4, F9, F11, F15, F17, F18, F19, F20**. All are verified (see the per-finding FIXED notes below).
**F10, F12, F13 were reviewed and deliberately NOT fixed** — reasons in §9. **F8 and F16 are
ACCEPTED** limitations (§9).

⚠️ **A re-verification pass on 2026-08-01 found that the F6 fix had introduced a NEW critical
regression, F21** — see §10. The lesson generalises: *a one-line fix to a sampler length is not
local.* `len_train` is read by `__len__`, and `__len__` bounds the val epoch, so shrinking it to
`len(subjects)` silently truncated a sweep that is `2 × len(subjects)` long. Neither `pytest` (251
green with the bug present) nor the identity gate could see it; it took an end-to-end run plus
counting the volumes actually written.

---

## 3. The critical bug (F1), in full

### What happens

`training/data/datasets/mri_dataset.py:477` emits, in float32:

```python
z_val = (z_i - (D - 1) / 2.0) * dz / Z_HALF_MM      #  -> np.float32 in scanner_coords
z_scale = Z_HALF_MM / dz                            #  -> np.float32 in the batch
```

and `vggt/utils/splat.py:43` inverts it:

```python
pz = pos[..., 2] * z_scale + (D - 1) * 0.5
```

Algebraically `pz == z_i` exactly. **But `dz/90` and `90/dz` round independently in float32**, and at
`z_i = 0` the two ends of the product cancel catastrophically. The residual leaves the domain:

```
D=11, dz=9.600000381469727  ->  pz[plane 0] = -4.768e-07
D= 8, dz=8.0                ->  pz[plane 0] = -2.384e-07
```

`splat.py:66-70` then tests `pz >= 0` as a **hard binary gate**. Since every pixel of an input slot shares
one `z_val`, a `−5e-07` undershoot discards **the entire slice**, not a fractional weight.

### Scope — MEASURED over the real manifest

**56 of 1343 subjects: 45 train / 2 val / 9 test.** All M&Ms. Distribution:

| `dz` (mm) | `D` values | n |
|---|---|---|
| 9.600000381469727 | 11, 12, 13, 14 | 52 |
| 8.0 | 8, 13 | 4 |

Affected **val** subjects (these are the ones that corrupt logged metrics): `MNMs_B0H7V0`, `MNMs_C6E0F9`.

Only the **bottom** end is violated — `0/1343` overflow at `pz > D-1`, because the trailing `+ (D-1)*0.5`
absorbs the error on that side. (Note: the top plane sits at `pz == D-1` *exactly*, so it is knife-edge too;
any future change that perturbs `pz` upward would start dropping the basal plane instead.)

### Cost — MEASURED through the production loss

`compute_volume_intensity_loss` with `world_points = scanner_coords` (Δ=0), on real subjects:

| subject | `D` | `dz` | plane-0 coverage | identity PSNR (bbox) now | with fix | Δ |
|---|---|---|---|---|---|---|
| `MNMs_C6E0F9` (val) | 12 | 9.60 | **0.0** | 21.87 dB | 31.42 dB | **+9.56** |
| `MNMs_B0H7V0` (val) | 13 | 9.60 | **0.1** | 26.85 dB | 31.86 dB | **+5.01** |
| `MNMs_A9C5P4` (control) | 10 | 10.00 | 170377 | 26.24 dB | 26.24 dB | +0.00 |
| `CMRx24_Test_P001` (control) | 10 | 12.00 | 174993 | 31.41 dB | 31.41 dB | +0.00 |
| `ACDC_patient140` (control) | 20 | 5.00 | 181593 | 24.85 dB | 24.85 dB | +0.00 |

The fix is **bit-identical on unaffected subjects** — that is the property that makes it safe to apply.

### Why it matters more than 56/1343 suggests

The Δ=0 code paths are **deterministically** wrong for those subjects:

- `training/loss.py:352` — `V_id`, the identity splat → `metric_mse_heart_identity`
- `training/loss.py:359` — `V_or`, the oracle splat → `metric_mse_heart_oracle` →
  **`metric_recov_frac_heart`**, the docs/38 primary ship-decision metric
- `training/trainer_viz.py:105` — the startup identity baseline baked into every
  `val_psnr_{full,bbox}/…base{b}` metric name and into `baseline_identity.json`
- `metric_hole_frac_heart` — the coverage tripwire itself is computed on the corrupted coverage

During training the model predicts `Δ ≠ 0`, so the gate is a knife-edge rather than a guaranteed drop —
roughly the half of apex pixels predicting `Δz ≤ 0` vanish, with **exactly zero gradient** (`in_bounds` is a
comparison result, so the dropped points contribute no gradient path back to `pos`). Expected symptom:
a systematic apex-plane coverage deficit correlated with dataset source. *(This training-time consequence
is `READ`/inferred, not measured on a converged checkpoint — see §7.)*

### This is a native-z regression

Under the pre-migration grid-relative convention (`z_norm = k/(D−1)*2−1`, `pz = (z+1)*0.5*(D−1)`), plane 0
had `z_norm = −1` exactly and `pz = 0` exactly. `READ` from `git show 0981b1b:vggt/utils/splat.py`.

### Suggested fix (MEASURED working)

Widen the in-bounds test by a sub-voxel epsilon **and clamp the continuous position before flooring**, so
the 8 corner weights stay a valid partition of unity:

```python
EPS = 1e-3   # sub-voxel; absorbs the fp32 round-trip residual (~5e-7 voxel)
in_bounds = (
    (px >= -EPS) & (px <= W - 1 + EPS)
    & (py >= -EPS) & (py <= H - 1 + EPS)
    & (pz >= -EPS) & (pz <= D - 1 + EPS)
).to(dtype)
if weight is not None:
    in_bounds = in_bounds * weight.to(dtype)
px = px.clamp(0, W - 1); py = py.clamp(0, H - 1); pz = pz.clamp(0, D - 1)
# ... floor / weights / corner clamp as before
```

⚠️ **A naive `+eps` nudge on `z_val` is NOT a fix** — measured: it trades the apex plane for the basal one
(control subject lost **4.16 dB** because `pz = D-1` was pushed out the top). The clamp is what makes it
symmetric.

### ✅ FIXED 2026-07-31

Applied in `vggt/utils/splat.py` as suggested: a module-level `EPS = 1e-3` (voxel units) widens the
in-bounds test on **all three axes**, and the **continuous** `px/py/pz` are clamped into the domain
before flooring, with the interpolation weights recomputed from the clamped values so the 8 corner
weights stay a partition of unity.

**Fault-injection proof** — identity splat of the real GT stack, `EPS = 0.0` vs `EPS = 1e-3`:

| `D`, `dz` | plane-0 coverage before → after | PSNR before → after |
|---|---|---|
| 11, 9.6 | **0.0 → 64.0** | 9.96 → **119.92 dB** |
| 8, 8.0 | **0.0 → 64.0** | 8.61 → **120.10 dB** |
| 13, 9.6 | **0.0 → 64.0** | 13.89 → **120.10 dB** |
| 10, 12.0 (control) | 64.0 → 64.0 | 120.00 → 119.95 dB |
| 20, 5.0 (control) | 64.0 → 64.0 | 120.03 → 119.94 dB |

Unaffected pitches are untouched, which is the property that makes it safe to drop in.

**torch.compile checked (this was an explicit concern).** `torch._dynamo.explain(splat_to_volume)`
reports **`graph_count 1, break_count 0, break_reasons []`**; `torch.compile(..., fullgraph=True)`
succeeds on both `splat_to_volume` and `splat_predictions` with output **bit-identical to eager**
(`max|Δv| = 0.0`), and gradients still flow (finite, non-zero). Expected: `EPS` is a module-level
Python float folded in as a constant and `.clamp()`/`torch.floor()` are plain tensor ops — no
data-dependent control flow, no host-device sync. *(The inductor **C++ CPU** backend fails on this
node with `unrecognized command line option '-std=c++20'` — a pre-existing system-g++ issue
unrelated to this change; the check used the `aot_eager` backend, and the GPU/triton path is
untouched.)*

`pytest tests/` — 215 passed.

---

## 4. The gate is blind to it (F2)

`tools/gate_native_z_identity.py:96`:

```python
m = (cov[0] > 0.5) & (gt > 1e-3)
```

The PSNR is masked **on coverage**, so a plane with coverage 0 is excluded from the metric by construction.

**MEASURED:** on `MNMs_G7S6V0` (D=11, dz=9.6 — an affected subject that is *in the gate's own subject list*),
plane-0 coverage is `0.0` and the gate reports **120.00 dB, PASS**.

The gate's fault injection (scaling `z_scale` by 0.4) cannot catch this either, because it displaces *all*
planes, which the coverage mask still sees.

**Fix:** assert full coverage rather than masking on it — e.g. require
`(coverage.sum(dim=(-2,-1)) > 0).all()` per z-plane, or compute PSNR over `gt > 1e-3` alone and let
uncovered voxels count as error. Either makes the gate fire on F1.

**Generalizable lesson (worth remembering):** *a gate that masks on the same quantity a bug destroys can
never see that bug.* The standing fault-injection rule was followed here — the gate was proven to fail on a
corrupted `z_scale` — but the injected fault was chosen from the same family the mask tolerates.

### ✅ FIXED 2026-07-31

Two changes in `tools/gate_native_z_identity.py`:
1. The PSNR mask is now `gt > 1e-3` **alone** — coverage is no longer part of it, so an uncovered
   voxel counts as full error instead of vanishing from the metric.
2. An **independent structural check**: every z-plane that holds anatomy must receive non-zero
   coverage. `_identity_psnr` now returns `(psnr, empty_planes)` and the runner FAILs (and prints
   the dropped plane indices) if `empty_planes` is non-empty, regardless of PSNR.

**Proof it now catches F1** — same subject, F1 deliberately re-injected (`EPS = 0.0`):

```
MNMs_G7S6V0  dz=9.60mm D=11  identity= 31.23 dB  [FAIL]  DROPPED PLANES [0]
CMRx24_Test_P001  dz=12.00mm D=10  identity=120.00 dB  [PASS]     <- control unaffected
```

That is the exact subject that previously reported **120.00 dB PASS**. With F1 fixed the full gate
is **35/35 PASS at 120.00 dB with no dropped planes**, and the original `z_scale`-corruption fault
injection still correctly fails **0/35** (14–22 dB), so the gate remains proven-to-fail.

---

## 5. Data-integrity findings

### F3 — Duplicate subjects across split boundaries

**MEASURED** (md5 over all 12 canonical frames):

| pair | splits | frames identical |
|---|---|---|
| `ACDC_patient055` ≡ `ACDC_patient118` | **train ↔ test** | 12/12 |
| `MNMs_A7G0P5` ≡ `MNMs_K3R0Y7` | **train ↔ val** | 12/12 |
| `MNMs_C8J7L5` ≡ `MNMs_C8O0P2` | **val ↔ test** | 12/12 |

(Plus 3 within-train pairs: `ACDC_patient074`≡`076`, `MNMs_A8C9H8`≡`Q0Q1Y4`, `MNMs_C5Q2Y5`≡`E9L4N2`.)

**Origin — MEASURED, and this correction matters:** the raw source files have **different md5s**, but their
**voxel arrays and affines are bit-identical** (`np.array_equal → True`, maxdiff 0; only the gzip containers
differ). So the duplication is **shipped by ACDC and M&Ms across their own official splits**, and
`tools/convert_to_sax_layout.py` reproduced it faithfully. *An earlier reviewer claim that the raw files were
byte-identical is wrong on the letter but right on the substance.*

**Consequences:**
- 1 of 134 val subjects and 1 of 50 ACDC-test subjects are memorized train data.
- **The metadata self-contradicts.** `MNMs_A7G0P5` is labelled Philips/centre-2 and `MNMs_K3R0Y7`
  GE/centre-4 — for identical pixels. `C8J7L5` is F, `C8O0P2` is M, same age, same pixels. So at least one
  M&Ms vendor label is demonstrably wrong, which weakens the §2.1/§13 argument that respecting M&Ms'
  official split buys a clean unseen-vendor (Canon) test.
- Neither `tools/build_manifest.py` nor `tools/build_pooled_split.py` does any duplicate detection.

**Detection method (reusable):** full-cohort content fingerprint — 16×16×8 thumbnail of `frame_00`, cosine
similarity, all pairs. Exactly these 6 pairs at 1.0000; next-highest pair 0.6724. The clean gap means this is
the **complete** duplicate set, and there are no cross-year CMRx or cross-source near-duplicates.

### ✅ FIXED 2026-07-31

**Origin re-confirmed at the RAW source** (not the converted `*_sax/` copies), on the **full native
4D** — all 25/30 native frames, not just our 12 resampled phases. `np.array_equal` on voxels **True**
and `max|diff| = 0.0` for all 6 pairs, with **identical affines**; only the gzip containers differ
(074/076 are byte-identical files). Visual: `result/duplicate_pairs/duplicate_pairs.png`
(`tools/render_duplicate_pairs.py`).

**6 subjects excluded → cohort 1343 → 1337, split 935 / 133 / 269 (69.9 : 9.9 : 20.1).** Rule: never
delete from an eval split when a train member exists; for val↔test keep test; for train↔train drop
the lexicographically later id. Dropped: `ACDC_patient055`, `ACDC_patient076`, `MNMs_A7G0P5`,
`MNMs_C8J7L5` (the one val drop), `MNMs_Q0Q1Y4`, `MNMs_E9L4N2`.

**Critically, the exclusion is applied AFTER assignment**, not by filtering the manifest up front:
`build_pooled_split.py`'s RNG draws depend on list lengths, so pre-filtering would have reshuffled
the entire pool. Verified: the regenerated `pooled.txt` differs from the previous one by *exactly*
the 6 deleted lines (`comm` shows zero lines added, zero subjects moved between splits), and
`manifest.csv` still has 1343 rows with exactly 6 changed to `split=excluded_duplicate`. Because the
ratio stays inside 7:1:2, **no re-split was needed**. All three splits load through the real
`MRIDataset` at 935/133/269.

**Recurrence guard added:** `tools/build_pooled_split.py --check-duplicates` re-derives the duplicate
set from pixel content and **hard-fails** on any pair not in the `DUPLICATE_PAIRS` list.
**Run over the full 1343-subject cohort it recovered exactly these 6 pairs and no others**, which
independently confirms the audit's completeness claim. ⚠️ One correction to §5's numbers: with the
*implemented* fingerprint the next-closest unrelated pair is **0.9655**, not 0.6724 — a much tighter
margin than the audit reported for its own differently-built thumbnail (mean-pooled cardiac MR is
globally self-similar). The 0.999 threshold is still safely in the gap, but it should not be loosened, so a new duplicate must be triaged by a
human rather than silently absorbed. Full record: docs/58 §13a.

### F6 — ACDC is oversampled 1.60× every epoch

`mri_dataset.py:166` `len_train = max(1000, len(subjects))` = **1000**, with `:280` `subj_idx = seq_index % 940`.

**MEASURED** with the real sampler: 1000 draws/epoch, all 940 subjects covered, `Counter({1: 880, 2: 60})`.
The double-sampled set is `subj_idx ∈ 0..59` and is **invariant to seed and epoch** — it does not average out.
Because `build_pooled_split.py` writes `sorted()` paths and `ACDC_sax/…` sorts first, **indices 0–59 are all
ACDC**:

| source | subjects | share | samples/epoch | share |
|---|---|---|---|---|
| ACDC | 85 | 9.0% | 145 | **14.5%** |
| CMRxRecon2023 | 156 | 16.6% | 156 | 15.6% |
| CMRxRecon2024 | 235 | 25.0% | 235 | 23.5% |
| CMRxRecon2025 | 289 | 30.7% | 289 | 28.9% |
| M&Ms | 175 | 18.6% | 175 | 17.5% |

ACDC is the pathology-labelled, finest-pitch source — precisely the axis pooling was meant to balance.

**Note the docs/58 §4 fix is inert here:** at n=940 the old hardcoded `1000` produces the *identical* sample
multiset. The hazard the doc anticipated (subjects past index 999 never sampled) does not exist at 940; the
live hazard is the `1000 % 940 = 60` residual meeting an alphabetically-sorted split file.

**Fix:** `len_train = len(self.subjects)` (and match/relax `limit_train_batches`), or round `len_train` up to
a multiple of `len(subjects)`.

#### ✅ FIXED 2026-07-31

Took the first option: `mri_dataset.py` now sets `self.len_train = len(self.subjects)` — one exact
pass per epoch. Coupled config edits: `limit_train_batches: 1000 → 935` and
`logging.log_visual_frequency.train: 1000 → 935` (so the train visual still fires once per epoch).
Both are 935, not 940, because F3's duplicate exclusion landed at the same time.

**Verified on the real pooled loader** — was `Counter({1: 880, 2: 60})`, now:

```
subjects 935   len(ds) 935
draws/epoch 935   coverage 935   multiplicity Counter({1: 935})
```

Every subject exactly once; the ACDC 1.60× oversample is gone.

⚠️ **Side effect to be aware of:** the epoch is now 935 steps instead of 1000, so at the unchanged
`max_epochs: 200` a run does **187k steps instead of 200k** (−6.5%). The LR schedule is epoch-driven
so its *shape* is preserved. Raise `max_epochs` to ~214 if total step count must be held constant —
**not done**, left as a deliberate user decision.

### F8 — Val does not cover the pitch range native-z exists to handle

**MEASURED** from `manifest.csv` (which I verified matches on-disk headers, 1343/1343):

```
pitch    train  val  test
 5.000     10    4     9
 6.000      0    0     1
 6.500      1    0     0
 7.000      0    1     0
 8.000     18    0     5   <-- val = 0, and 18 sit in train
 8.050      1    0     0
 8.800      6    2     1
 9.520      1    0     1
 9.600     45    2     9
 9.960      1    0     0
10.000    249   53   175
12.000    608   72    68
```

Val covers **6 of 12 distinct pitches**, and **93% of val is 10 or 12 mm**. No split rule stratifies by pitch
(ACDC/M&Ms take official splits; CMRx23/24 are plain random; CMRx25 stratifies by vendor×pathology only).
So the *generalization-across-pitch* claim motivating docs/58 is largely unmeasurable on the current val set.

**ANALYSIS 2026-07-31 — where the pitch diversity actually lives (this constrains every possible fix).**
Recomputed on the post-exclusion 1337 cohort, joined with `source`:

| pitch | train | val | test | sources |
|---|---|---|---|---|
| 5.00 | 10 | 4 | 9 | ACDC 21, M&Ms 2 |
| 6.00 | 0 | 0 | 1 | M&Ms 1 |
| 6.50 | 1 | 0 | 0 | ACDC 1 |
| 7.00 | 0 | 1 | 0 | ACDC 1 |
| **8.00** | **18** | **0** | **5** | **M&Ms 23** |
| 8.05 | 1 | 0 | 0 | M&Ms 1 |
| 8.80 | 6 | 2 | 1 | M&Ms 9 |
| 9.52 | 1 | 0 | 1 | M&Ms 2 |
| 9.60 | 45 | 2 | 9 | M&Ms 56 |
| 9.96 | 1 | 0 | 0 | M&Ms 1 |
| 10.00 | 244 | 52 | 175 | M&Ms 246, ACDC 125, CMRx25 97, CMRx23 3 |
| 12.00 | 608 | 72 | 68 | CMRx24 294, CMRx25 262, CMRx23 192 |

**The decisive fact: every non-10/12 mm pitch comes from M&Ms or ACDC.** The CMRx years are
*exclusively* 10 or 12 mm (CMRx24 is 100% 12.0; CMRx23 is 192×12.0 + 3×10.0; CMRx25 is 262×12.0 +
97×10.0). This kills the cheapest-looking fix: **re-drawing the CMRx random splits cannot help at
all**, because there is no pitch diversity in them to redistribute. Any fix must move **M&Ms/ACDC**
subjects, which are exactly the two sources held on their *official* splits.

**Why moving M&Ms train→val is nonetheless safe.** The official-split rule was adopted (docs/58
§2.1) because *"Canon appears ONLY in Validation/Testing — respecting the official split buys a free
unseen-vendor test."* Moving subjects out of M&Ms **official Training** into our val cannot touch
that: Canon is not in official Training, so the ones we would move are non-Canon, and the move only
*removes* them from train. The unseen-vendor property is a statement about what is absent from
train, and this makes train strictly smaller. The 9 M&Ms 8.8 mm and 2 M&Ms 5.0 mm subjects already
sit in val under the current rule, which shows the official split was never the binding constraint
on val's pitch coverage — 8.00 mm just happened to draw zero.

**Concrete minimal option, CONSIDERED AND REJECTED:** move ~2 M&Ms 8.0 mm and ~4 M&Ms 9.6 mm
subjects from train to val (8.00 mm → 16/2/5, 9.6 mm → 41/6/9, cost 6 train subjects). Although this
would not leak Canon, **the user rejected it 2026-07-31: deviating from the M&Ms official split is
not worth it.** The official split is a published, citable boundary; quietly moving subjects across
it costs more in reproducibility/defensibility than the extra stratification buys.

### ✅ RESOLUTION — ACCEPTED AS-IS (2026-07-31), and the audit's framing was too strong

The audit's "largely unmeasurable" claim overstates the problem. **Val already spans four sub-10 mm
pitches**, which is enough to test cross-pitch generalization — just at small n:

| pitch | n in val | subjects |
|---|---|---|
| 5.00 | 4 | `ACDC_patient035/075/092/094` |
| 7.00 | 1 | `ACDC_patient093` |
| 8.80 | 2 | `MNMs_C4E9I1`, `MNMs_D1H6U2` |
| 9.60 | 2 | `MNMs_B0H7V0`, `MNMs_C6E0F9` |
| 10.00 | 52 | — |
| 12.00 | 72 | — |

**9 of 133 val subjects (6.8%) are sub-10 mm, spanning 5.0 / 7.0 / 8.8 / 9.6 mm** — including the
5 mm extreme, which is the hardest case (2.4× finer than the 12 mm majority and the one native-z
most changes). The honest statement is therefore *not* "unmeasurable" but: **cross-pitch behaviour is
measurable on n≈9, under-powered for a per-pitch breakdown, and 8.00 mm specifically is absent.**
Report sub-10 mm val performance as a pooled group rather than per-pitch.

Note also that `MNMs_B0H7V0` and `MNMs_C6E0F9` — the two 9.6 mm val subjects — are exactly the two
val subjects F1 was corrupting (−5.01 dB and −9.56 dB identity PSNR). So before the F1 fix, the fine-
pitch end of val was not merely thin, it was **actively broken**. That is now repaired, which
materially improves what this small group can tell us.

### F10 — `ef_val_sweep`'s ED/ES are segmentation-derived, not ground truth

`mri_dataset.py:229-254` reads `scratch/data/whs/cardiac_phase.csv`, produced by nnU-Net whole-heart segs,
but the docstring and `mri_finetune.yaml:64` call these "GT ED/ES".

**MEASURED** against the independent source labels in `manifest.csv` (ACDC `Info.cfg`, M&Ms CSV), on the 49
val subjects where both exist:

- **20/49 disagree** on ES (cyclic distance histogram `{1: 18, 3: 1, 5: 1}`); worst `ACDC_patient050`:
  manifest `es=4` vs csv `ES=9` — half a cycle.
- ED is non-zero for **19/134** val subjects despite the ED-anchored convention — though in fairness
  16 of those are just ±1 (`ED ∈ {11, 1}`); only 3 are genuinely far (`ED = 7, 8, 10`).

No crash (all 268 entries parse, all in `[0,12)`). But `val/ef/{slope,spearman,mae_pct}` is regressed against
a noisy label, and a few sweep entries reconstruct a phase that isn't really ED or ES.

#### ⬜ WONTFIX — keep the segmentation-derived labels (user decision, 2026-07-31)

Rationale is **preprocessing consistency**: `cardiac_phase.csv` covers all 1337 subjects under ONE
convention, whereas source labels exist only for ACDC and M&Ms and come from a different (per-dataset,
human) definition. Mixing the two would make EF incomparable **between sources**, which is worse than a
uniform label noise. Accepted cost: 20/49 val subjects disagree with their source ES label (worst case
half a cycle), so `val/ef/*` carries known label noise — report it as such. The "GT ED/ES" wording in
`mri_dataset.py`'s docstring and `mri_finetune.yaml:64` remains inaccurate and is worth a comment fix if
anyone touches that code.

---

## 6. Remaining findings, briefly

- **F4 — the respiratory native-z fix is untested.** `tests/conftest.py:20` sets `SYN_SPACING=(1.4,1.4,12.0)`
  and `tests/test_gpu_aug.py:46` pins `dz_mm=12.0`. Since `SPACING_MM[0]` is *also* 12.0,
  `spacing=(dz,1.4,1.4)` is **bit-identical** to the old hardcoded default in every fixture — reverting the
  fix cannot fail a test. Likewise the `group_by_burst` tests never pass `n_planes` and use plane ids ≤ 6,
  where `clamp(0,11)` and `clamp(0,D-1)` are both no-ops. docs/58 §8.1a explicitly asked for this test.
  **Fix:** add a case at `dz=5.0, D=21` asserting the resliced plane index equals `z_i + d/dz`, plus a burst
  case with plane ids > 12; fault-inject by restoring 12.0 and confirm it fails.

  **✅ FIXED 2026-07-31 — `tests/test_respiratory_native_z.py` (36 tests).** Built as prescribed,
  then **proven to have teeth by fault injection**. Coverage:
  - *Core invariant*, parametrized over all 7 real cohort pitches (5.0/6.0/8.0/8.8/9.6/10.0/12.0) ×
    D ∈ {5, 12, 21}: a volume whose plane `z` holds value `z/100` is resliced by `d` mm, and the
    decoded output must equal plane `z_i + d/dz` to 2e-3 — directly measuring the quantity the
    pitch conversion decides.
  - *Fault-injection tests* (`test_pitch_fault_injection_is_detectable`,
    `test_n_planes_fault_injection_is_detectable`) assert the LEGACY behaviour is measurably
    different — >0.5 planes apart, and planes ≥12 collapsing onto plane 11 — so a silent revert
    cannot pass. Non-12 pitches only, since dz=12 is exactly the blind spot that let F4 exist.
  - Sign convention, the `d=0` identity, and a sub-voxel shift (2 mm at dz=5 → 0.4 planes; the
    legacy 12 mm gives 0.167, a 2.4× understatement no integer-plane test could see).
  - In-plane axes must NOT be scaled by dz (guards the plausible bad fix of scaling all 3 axes).
  - `group_by_burst` at `n_planes=D=21`: planes 12–20 independent; same-plane slots still share one
    breath; thin stacks (D=5) clamp to D−1.
  - *Integration*: `gpu_augment_batch` is spied on to confirm it hands the reslicer
    `spacing=(dz,1.4,1.4)` and `n_planes=D`. The unit tests prove the reslicer is right GIVEN the
    right spacing; this proves the trainer path supplies it.
  - Plus the F7 mixed-pitch guard exercised through the real aug entry point.

  **Fault-injection result — this is the F4 claim, now measured.** With the docs/58 fix reverted in
  `gpu_aug.py` (`spacing=SPACING_MM`, `n_planes=None`):

  | suite | result with the fix REVERTED |
  |---|---|
  | pre-existing 215 tests | **215 passed** ← exactly the blind spot F4 described |
  | new `test_respiratory_native_z.py` | **3 failed**, 33 passed |

  Restored: **251 passed**.

- **F5 — monai cache is 66% dead weight.** `preprocess.py:275` `ConcatItemsd` does not drop its source keys,
  so all 12 float32 `phase_NN` tensors are pickled alongside the float16 `phases` built from them.
  **MEASURED** on a live cache entry: 47.8 MB/file, of which **31.4 MB is never read** (useful fraction 0.34;
  `get_data` reads only `phases`, `content_mask`, `dz_mm`). At 1074 train+val subjects that is ~54 GB of
  `/tmp` instead of ~18 GB, and ~3× the per-epoch read volume. **Fix:** `DeleteItemsd(keys=phase_keys)` after
  `ConcatItemsd` — backward-compatible (stale fat entries still load).

  **✅ FIXED 2026-07-31** — exactly that: `DeleteItemsd(keys=phase_keys)` inserted after `ConcatItemsd`
  in `training/data/preprocess.py`. **Verified on a fresh cache dir**: entry keys are now
  `['content_mask', 'dz_mm', 'phases', 'sax_dir', 'subj_id']` (the 12 `phase_NN` keys gone) and the
  file is **18.02 MB** for a `D=11` subject (the old fat layout was 47.8 MB at `D=10`). Crucially,
  the surviving payload is **bit-identical**: re-running the pipeline on a subject that has an old
  fat cache entry gives `torch.equal == True` for both `phases` and `content_mask`, and the same
  `dz_mm`. No behavioural change, ~2.9× less I/O per sample per epoch.

  *Scope note for the record:* this is a throughput/disk issue only — `PersistentDataset` writes to
  node-local `/tmp`, which is wiped on every new node, so there was never a stale-cache correctness
  risk. The costs were a slower first-epoch build, a permanently inflated per-sample read on **every**
  epoch, and ~51 GB of `/tmp` at 1074 subjects.

- **F7 — `B>1` would silently mix z scales.** `loss.py:134` and `gpu_aug.py:356` read
  `batch["z_scale"/"dz_mm"].reshape(-1)[0]`. `batch_size == 1` holds only because
  `floor(max_img_per_gpu / img_nums) = floor(20/20) = 1`. **MEASURED:** *every* train `D` value carries
  multiple pitches (e.g. `D=10` spans 5.0–12.0 mm), so same-`D`-different-`dz` pairs **collate successfully**
  and row 1 would be splatted and breathed at row 0's scale — a silent 20%+ through-plane geometry error.
  Different-`D` pairs crash loudly on collate instead. ~~**Fix:** `assert batch["z_scale"].numel() == 1`.~~

  **✅ FIXED 2026-07-31, but NOT with the suggested assert.** `numel() == 1` was implemented first and
  **broke 14 tests**: several fixtures (`test_loss_bbox.py`, `test_loss_val_metrics.py`) legitimately
  build `B=2` batches with *identical* `z_scale`, which is harmless. The suggested assert was
  therefore too strict — it banned a safe case while targeting the wrong invariant. The guard now
  tests **uniformity**, which is the actual hazard: `loss.py` raises if
  `not (batch["z_scale"].reshape(-1) == ...[0]).all()`, and `gpu_aug.py` does the same for `dz_mm`.
  Both messages name the offending values and cite this finding. `pytest tests/` — 215 passed.

- **F9 — `one_frame_per_slice: true` made three config knobs stale.** `mri_dataset.py:369-372` overrides
  `S` with the subject's in-FOV plane count, and since native-z never zero-pads z, `anatomy_bbox` is always
  `[0, D)` ⇒ **`S == D` exactly**. So `num_slices: 20` and `img_nums: [20,20]` no longer describe the slot
  budget, and — operationally the important part — **"reduce `max_img_per_gpu` on OOM" now does nothing**
  (`floor(12/20)` and `floor(20/20)` are both 1). `img_nums` is *not* fully dead: it still sets batch size,
  which is the F7 landmine.

  **✅ FIXED 2026-07-31 (F9 + F19 together).** A comment at the `one_frame_per_slice` override in
  `mri_dataset.py` records the consequence, and a real guard was added. One **correction to the
  finding**: `max_img_per_gpu` is not merely stale — verified at `dynamic_dataloader.py:176`,
  `batch_size = floor(max_img_per_gpu / img_nums)`, so **lowering** it is a no-op (both
  `floor(12/20)` and `floor(20/20)` are 1) but **raising** it past `2*img_nums` gives
  `batch_size >= 2`, which mixes subjects and trips the F7 uniformity guards.

  **Then resolved properly (user call): batch size is now PINNED TO 1** in
  `dynamic_dataloader.py`, and `max_img_per_gpu` is documented as **inert** in both the dataloader
  docstring and `mri_finetune.yaml`. Rationale, measured:

  | pairing | behaviour |
  |---|---|
  | different `D` (10 vs 12) | **loud** `RuntimeError` in `default_collate` |
  | same `D`, different pitch (dz 7.5 vs 18.0) | **collates cleanly** — the silent F7 hazard |

  So under native-z the only configuration safe *by construction* is B = 1. The upstream adaptive
  formula (`floor(max_img_per_gpu / random_image_num)`) is simultaneously inert (it already
  evaluated to 1, and `one_frame_per_slice` sets S = D regardless) and a footgun (raising the knob
  silently enables the unsafe case). `max_img_per_gpu` is kept **only** so the existing Hydra
  configs resolve. To trade memory, change `D` or the model.

  **Then removed entirely (user call, 2026-07-31).** Keeping a dead knob that *looks* like an OOM
  control is worse than deleting it, so `max_img_per_gpu` is gone from `default.yaml`,
  `mri_finetune.yaml` (all 5 occurrences) and both `dynamic_dataloader.py` signatures. Verified: all
  5 configs still Hydra-`compose(resolve=True)` cleanly with `max_img_per_gpu` absent from the
  resolved tree. **`img_nums` was KEPT** — it is *not* dead: it is the S cap the F19 guard enforces
  (and the slot budget whenever `one_frame_per_slice` is false), so deleting it too would have
  silently disabled that guard.

  **F19 guard:** `S` is compared against the budget it just overrode and raises if `S > budget` — no
  new constructor argument needed, since the incoming `S` *is* the configured budget. ⚠️ **Gated on
  `img_per_seq is not None`**, i.e. only when the REAL dataloader supplied the budget. The first
  version was not gated and immediately broke `tools/gate_native_z_identity.py` (35 → 29 passing):
  standalone construction falls back to `self.num_slices`, whose default **12** is not the training
  budget **20**, so every legitimate D>12 subject was rejected. Caught by re-running the gate after
  the edit — a good argument for re-running the acceptance gate after *any* dataset change. **Measured headroom:** max `D` is **18 in train and val**, but the pool holds
  `D=19` (`ACDC_patient127`), `D=20` (`ACDC_patient140`, `MNMs_L8N7Z0`) and `D=21`
  (`ACDC_patient124`) — **all four currently in test**, so a re-seeded split is exactly the scenario
  that would have silently exceeded the budget.

- **F11 — `metric_ssim_3d_full` is now `D`-dependent.** `fused_ssim3d` uses an 11-tap (radius-5) window in
  all three dims with zero padding; the fraction of edge-contaminated z-planes is 100% at D=5 vs 48% at D=21.
  **MEASURED** on identically-constructed random pairs: `0.9727 (D=5)`, `0.9659 (D=8)`, `0.9616 (D=12)`,
  `0.9581 (D=21)` — thin stacks read high. Under the old fixed D=12 this was a constant bias; now the val
  mean mixes D=5…21, so SSIM is **not comparable across subjects** and not comparable to pre-native-z runs.

  **✅ FIXED 2026-07-31 — `metric_ssim_3d_full` REPLACED by per-slice `metric_ssim_2d_full`.**
  Reshaping `(B,D,H,W) → (B*D,1,H,W)` treats z as a batch dim, removing the z-padding entirely: the
  window is only ever in-plane and `D` just sets how many slices are averaged. SSIM is
  **metric-only** — verified it appears nowhere in the objective (single call site, `train=False`)
  — so this cannot affect training.

  The 3D metric was **dropped, not kept alongside** (user call). An intermediate version logged
  both, justified as "continuity with old runs" — but that does not survive contact with the fact
  that pre-native-z runs are **already incomparable** (V_gt frame, normalization and grid all
  changed). Logging both would have meant two numbers where one is knowingly wrong. Side effect:
  `vggt/utils/fused_ssim_compat.py` is now unimported — left on disk (standalone utility).

  ⚠️ **Two corrections, from re-measuring.** The audit's numbers did not reproduce. On pure random
  noise the D-dependence is invisible (0.9910→0.9908) because SSIM saturates. Isolating it properly
  — ONE structured volume with ONE error field, cropped to different depths so only D changes:

  | D | `ssim3d` | per-slice `ssim2d` |
  |---|---|---|
  | 5 | 0.9929 | 0.9947 |
  | 12 | 0.9933 | 0.9947 |
  | 21 | 0.9938 | 0.9946 |
  | 32 | 0.9939 | 0.9946 |

  So (1) the **direction is opposite** to the audit's report — thin stacks read **LOW**, not high;
  and (2) the **magnitude is ~15× smaller** (0.0010 spread, not 0.0146). The per-slice form is ~10×
  flatter (0.0001). The finding is real and the fix is right, but it is a **minor** comparability
  issue, not the significant bias the original table implied.

- **F12 — a refiner-era checkpoint loads, then dies on optimizer state.** `trainer.py:325` loads with
  `strict=False` (refiner keys land in "unexpected") and `:326` logs only a *count*, so nothing reveals what
  the ckpt was; `:330-336` then raises `ValueError: loaded state dict contains a parameter group that doesn't
  match…`. 8 refiner-era checkpoints are still on disk under `scratch/logs/*refiner*/`. The seed ckpt the
  sbatch scripts actually use (`scratch/checkpoints/4wok_weights_only.pt`) has no refiner keys — safe.
  ~~**Fix:** log the *names* of unexpected keys.~~

  **⬜ WONTFIX 2026-07-31 (user decision): no refiner checkpoint will be loaded.** The seed the sbatch
  scripts actually use is clean, and the failure is **loud** (`ValueError`), not silent. Accepted cost:
  if someone does reach for one of the 8 refiner-era checkpoints still on disk, the error names neither
  the cause nor the checkpoint.

- **F13 — val artifact volume scaled 4.5×.** Under the sweep the `_save_val_volumes` dedup key is
  `(subject, t_val)` → 268 keys → **536 gzipped NIfTIs per val epoch** (`val_epoch_freq: 1`), ~925 MB/epoch to
  GPFS (was ~207 MB). `_save_ef_volume` likewise segments 268 volumes per EF epoch instead of 60.

  **⬜ ACCEPTED 2026-07-31 after checking the mechanism.** Read `trainer_viz._save_val_volumes`:
  filenames are deterministic (`subj{idx:02d}_t{t:02d}_{subject}`) and val runs `shuffle=False`, so the
  **same files are overwritten in place every epoch** — the ~925 MB is a **constant disk footprint, not
  cumulative**. What remains is per-epoch *write time* (532 gzipped NIfTIs to GPFS), not a disk leak.
  One config line (`save_val_volumes: false`, or an every-N-epochs gate) if it ever shows up in
  profiling. Note the sweep is now 266 entries, not 268 (val is 133 subjects post-F3).

- **F14 — `ef_eval.py:23` `CANON_SPACING = (1.4,1.4,12.0)` is stale but BENIGN.** Verified two ways:
  EF is a ratio `(v_ed − v_es)/v_ed`, so the shared `VOX_ML` cancels exactly; and nnU-Net v1's
  `PreprocessorFor2D.resample_and_normalize` sets `target_spacing[0] = original_spacing_transposed[0]`
  (`READ` from the installed source), i.e. **2D models never resample z**, so the wrong stamp does not change
  the segmentation. Residual harm is only that the written NIfTI headers are geometrically wrong on disk and
  `_lv_ml` absolute volumes are off by `dz/12`. *An earlier concern that this corrupts the EF metric is
  refuted.*

- **F15 — `splat.py`'s in-bounds gate is a multiply, so `0.0 * NaN = NaN`.** **MEASURED:** one NaN or +Inf
  coordinate out of ~5.4M produces 8 NaN voxels and `V.mean() = NaN`, poisoning `loss_volume` and every
  full-volume metric instead of that pixel simply contributing zero weight. **Pre-existing** — the
  pre-migration code had the identical form (`READ` from `git show 0981b1b`). Compounding it: `trainer.py`'s
  non-finite guard (`:982`) runs *after* `_update_and_log_scalars` (`:1022`), so poisoned scalars are already
  in the `AverageMeter` running sums (which then stay NaN for the epoch) and already pushed to wandb; and
  **val has no finiteness guard at all**, so a NaN walks straight through every `try/except` (a NaN is a
  *value*, not an exception). **Fix:** `torch.where(in_bounds_bool, w, 0)` — the same form `loss.py:258-262`
  already uses for exactly this reason. Also applies to the (currently inert) gather aux at `loss.py:192`.

  **✅ FIXED 2026-07-31.** Applied in `splat.py`, and it needed **two** changes, not one — the
  suggested `torch.where` on the gate alone is NOT sufficient:
  1. the gate itself: `in_bounds = torch.where(in_bounds_bool, weight, 0)` instead of a multiply;
  2. **the positions feeding the interpolation weights**:
     `px = torch.where(in_bounds_bool, px, 0).clamp(...)` (same for py/pz). Without this, `clamp`
     **propagates NaN** (torch returns NaN for `clamp(NaN)`), so `wx1 = px - floor(px)` is NaN and
     `in_bounds * NaN` re-poisons the scatter even though `in_bounds == 0` there.

  ⚠️ **Precision correction to the finding:** only **NaN** poisons, not `+Inf`. Measured — a `±Inf`
  coordinate fails the comparisons (so `in_bounds_bool` is False) *and* `clamp` finitizes it, so the
  old code already handled it. The audit's "one NaN or +Inf coordinate" overstated the trigger.

  **Fault injection (old multiply form vs new), 2 poisoned points out of 4000:**

  | coordinate | old form | new form |
  |---|---|---|
  | `NaN` | **16 non-finite voxels, `V.mean() = nan`**, gradients corrupted | 0 non-finite, `mean` unchanged |
  | `+Inf` / `-Inf` | already clean | 0 non-finite, `mean` unchanged |

  With the fix, the result is **exactly equal** to dropping those 2 points via `weight=0`
  (`torch.allclose` True), gradients on all other points stay finite, and the function still
  compiles `fullgraph` with **0 graph breaks**, bit-identical to eager on clean input.

  Not applied to the inert gather aux at `loss.py:192` — it is `gather_weight: 0.0` in every live
  config, so it is dead code today; noted rather than touched.

- **F16 — respiratory zero-padding fabricates dimmed/blank input slices.** `respiratory.py:363` uses
  `padding_mode="zeros"`, and since `d ≥ 0` always, the shift always runs off the high-z (basal) end, which
  `grid_sample` returns **linearly attenuated** until `d ≥ dz`, then zero. The amplitude error is *unlearnable*
  (the head predicts position, not intensity) and `splat_weight = intensity > 1e-3` does not gate a
  0.4×-dimmed slice. ⚠️ **DISPUTED:** two reviewers measured the cohort-wide fully-blanked rate at **2.74%**
  and **12.4%** respectively. Both agree it is **pre-existing and barely changed by native-z** (their
  respective "old grid" baselines were 1.80% and 11.9%). ~~*The magnitude is unresolved — measure before acting.*~~

  **✅ DISPUTE RESOLVED 2026-07-31 by measuring it — the two reviewers were counting DIFFERENT
  THINGS, and both were right.** Over all 935 train subjects at the live `mri_volume` respiratory
  config (`amplitude_mm=18.8`, `jitter=7.35`, `cos2n=3`, `group_by_burst`, tilt 0-45 deg), one draw
  per subject, computing each slot's landing plane `z_i + d_D/dz` against its own `D`:

  | outcome | slots | share of 9739 |
  |---|---|---|
  | **fully blank** (lands >1 plane outside the stack ⇒ `grid_sample` returns 0) | 212 | **2.18%** |
  | **partially dimmed** (lands within 1 plane of the edge ⇒ linearly attenuated) | 923 | **9.48%** |
  | **either** | 1135 | **11.66%** |

  Reviewer A's **2.74%** was the *fully blank* rate; reviewer B's **12.4%** was *blank + dimmed*.
  Same measurement at two thresholds, not a contradiction. Blank slots are spread across all five
  sources (CMRx23 25, CMRx24 39, CMRx25 60, ACDC 31, M&Ms 57) — a property of breathing amplitude vs
  stack height, not of any one dataset.

  **Mechanism, demonstrated directly** (plane-coded volume, D=10, dz=10 mm; the value read back
  *is* the plane index sampled):

  ```
  slot z=4:  d=0 -> reads plane 4.00 | d=5mm -> 4.50 | d=10mm -> 5.00 | d=20mm -> 6.00   (= z + d/dz)
  slot z=9:  d=0 -> value  9.00      | d=5mm -> 4.50 | d=10mm -> 0.00 | d=25mm -> 0.00
  slot z=0:  d=0 -> reads plane 0.00 | d=10mm -> 1.00 | d=20mm -> 2.00                   (always INTO the stack)
  ```

  Read the middle row carefully: at `z=9, d=5mm` the output is **4.50, not 9.5**. There is no plane
  9.5 to read, so `grid_sample` interpolates plane 9 (value 9) against the zero padding
  (value 0) → `0.5*9 + 0.5*0`. The slice is not *displaced*, it is **half-faded to black**. That is
  the whole bug: an intensity corruption masquerading as a geometric shift.

  **The per-subject picture is much worse than the 2.18% headline (MEASURED 2026-07-31).**
  Aggregate percentages hide that the damage is concentrated on ONE slice of every subject:

  | quantity | value |
  |---|---|
  | through-plane displacement `d_D` range | **[0.000, 27.289] mm — always ≥ 0** |
  | slots landing below plane 0 (apex end) | **0 / 9739** |
  | slots landing above plane D−1 (basal end) | 1135 / 9739 = 11.65% |
  | **subjects whose MOST BASAL slot (z = D−1) lands off-slab** | **882 / 935 = 94.3%** |

  So it is not "2% of slices somewhere" — **the basal-most input slice is fabricated (fully blank or
  partially faded) in ~94% of training samples**, because `d ≥ 0` guarantees the top slot always
  reaches for anatomy above the stack. That is a systematic corruption of the base, which is where
  the LV cross-section is largest and therefore what EF/volume estimates depend on most. This raises
  F16's priority above the audit's 🟡 Low-Med.

  **Which end is affected: the BASE, not the apex.** With apex-at-z0 on disk (docs/58 §10a) and
  `d > 0` sampling at `z_i + d/dz`, inspiration reads from higher z, i.e. anatomy moves toward lower
  z = inferior = apex (physiological). The slots that run out of volume are therefore the **most
  basal** ones (bottom row above: z=0 always reads *into* the stack and is never affected). Do not
  confuse this with **F1**, which dropped the **apex** plane in the *splat* for a completely
  unrelated float32 reason.

  **Where the content BELONGS, and why coverage holes are not caused by any of this.** A slot
  acquired at plane `z_i` contains anatomy from reference plane `z_i + d/dz`, so a perfectly
  breathing-corrected model places it there. Since `d >= 0` (one-sided sim), landings only ever move
  to HIGHER z. Measured over all 935 train subjects under perfect correction:

  | quantity | value |
  |---|---|
  | slots landing off-slab | 1135 / 9739 = **11.65%** |
  | reference planes with **no** in-bounds source | 2959 / 9739 = **30.4%** |

  Uncovered planes cluster at **both ends** — 695 at the basal extreme (rel-z 1.0) and 341 at the
  apex extreme (rel-z 0.0), vs ~200 per bin in the middle. The apex spike has a clean cause: plane 0
  can only be covered by slot 0 with `d ≈ 0`, and `d > 0` almost always.

  ⚠️ **This corrects an earlier claim in this session.** Gating off-slab slots was described as
  "creating more holes"; that is wrong in emphasis. **The holes already exist (30.4%) and are a
  property of the one-sided breathing simulation**, not of any proposed fix — the splat's bounds
  check already discards a correctly-placed off-slab point. What a validity gate actually changes is
  only where the *dimmed* content goes: today it is placed wherever the (under-correcting) model
  predicts, i.e. somewhere in-bounds, polluting V_canon; gated, it contributes nothing.
  *(Caveat: the 30.4% assumes perfect correction. The real model under-corrects — measured slope
  0.844 — so actual landings sit closer to `z_i` and real coverage is better; `metric_hole_frac_heart`
  is the number that matters in practice.)*

  **Does coverage-division rescue the dimming? NO — measured.** `V = Σ w·I / Σ w` where
  `w = trilinear × splat_weight`, and `splat_weight = (intensity > 1e-3)` is a **binary gate, not the
  intensity**. So the denominator counts *contributions*, not brightness, and a dim contribution stays
  dim: a single 0.5-intensity slot alone on a plane gives `coverage=1.000, V=0.5000` where the truth
  is 1.0; mixed with one correct slot it gives `coverage=2.000, V=0.7500`. Dimming is **not**
  self-correcting. (Also: a *fully blank* slot contributes to neither numerator nor denominator — the
  intensity gate already drops it — so blank slots do not corrupt the loss, they just waste a slot.
  It is the **partially dimmed** ones that pollute `V_canon`.)

  **Severity distribution among off-slab slots (MEASURED) — this refutes a "blank them all" fix:**

  | overshoot past D−1 | slots | share | signal retained |
  |---|---|---|---|
  | 0.00–0.10 | 404 | **35.6%** | 90–100% |
  | 0.10–0.25 | 128 | 11.3% | 75–90% |
  | 0.25–0.50 | 151 | 13.3% | 50–75% |
  | 0.50–0.75 | 132 | 11.6% | 25–50% |
  | 0.75–1.00 | 108 | 9.5% | 0–25% |
  | ≥ 1.00 (fully blank) | 212 | **18.7%** | 0% |

  Median retained signal is **0.708**, and **35.6% are ≥90% intact**. An earlier recommendation in
  this session — *"any off-slab slot → blank it and gate it out"* — is therefore **withdrawn**: it
  would discard 404 essentially-intact slices to remove 212 bad ones.

  **REVISED RECOMMENDATION: `padding_mode="border"` + record the EFFECTIVE displacement.**
  - `border` is a **no-op for every in-slab slot** (verified: identical to `zeros` for all target
    planes ≤ D−1) and off-slab returns plane D−1 exactly. It is mathematically the same as
    rescaling the dimmed slice by `1/(1−a)`, which *exactly* recovers the real content, so for the
    35.6% at `a < 0.1` the correction is near-exact.
  - The catch, and why the second half matters: with `border` the slice content is plane D−1, i.e.
    the **effective** shift is truncated to `(D−1−z_i)·dz`, not the drawn `d`. If `resp_disp_mm`
    keeps recording the full `d`, then `metric_resp_*` scores the model against a shift that was
    never actually applied, which would read as spurious under-correction. Recording the effective
    (clamped) displacement makes the bookkeeping self-consistent.

  ⚠️ **`border` has a flaw that may disqualify it (user challenge, 2026-07-31).** For the
  basal-most slot, ANY `d > 0` lands off-slab, so `border` returns plane D−1 — the *unshifted*
  content — every time. Combined with the 94.3% figure above, that means **the basal slice would be
  effectively never breathing-simulated**, and the model would be trained on a consistent
  (image, shift≈0) pairing there. That is not noise, it is a **confidently wrong learned prior**
  ("basal slices don't move") which would transfer to real inference, where they certainly do.
  Arguably worse than the current fabricated darkness, which at least teaches nothing definite.

  ⚠️ **This also affects VALIDATION and the frozen baseline bundle.** Respiratory sim runs in
  **both train and val**, and `evaluation/engine/build_inputs/*.py` imports the trainer's OWN
  `training/data/respiratory.py` to build the frozen `gt/ clean/ breath/` bundle. `evaluation/README.md`
  states: *"Never regenerate the bundle under a subject without re-running every arm on it."* So
  changing `padding_mode` changes the bundle and **invalidates every SVR baseline comparison
  (SVRTK / NeSVoR / NiftyMIC) until all arms are re-run.**

  **Scope of the actual harm, put in proportion.** Fully-blank slots (2.18%) are already gated out by
  `intensity > 1e-3` and contribute to neither numerator nor denominator — they cost a wasted slot,
  not a corrupted loss. Slots retaining ≥90% (4.1% of all slots) are near-harmless. The genuinely
  polluting band is the middle: **391 slots = 4.0% of all slots** carry 0–75% intensity into
  `V_canon`.

  ### 3-agent debate outcome (2026-07-31): the exclude/keep dichotomy is FALSE

  Three independent agents argued exclude / keep / break-the-framing. Adjudicated result:

  **The winning option is neither: renormalize by the exactly-known attenuation AND splat with a
  soft validity weight.** The attenuation is one exact scalar per slot (`z_coord` is constant over
  H,W, `respiratory.py:359`), so with `v = clamp(1 − f, 0, 1)`: divide the slice by `v` (recovering
  true intensity) and pass `v` as the slot's splat weight. `splat_to_volume` already accepts an
  arbitrary `weight ∈ [0,1]` — the capability exists and is unused.

  **Verified by me, not taken from the agent** (truth = 1.0, one contributor):

  | attenuation `v` | KEEP (`I=v, w=1`) | RENORM+SOFT (`I=1, w=v`) |
  |---|---|---|
  | 1.00 | 1.0000 | 1.0000 |
  | 0.90 | 0.9000 | **1.0000** |
  | 0.50 | 0.5000 | **1.0000** |
  | 0.25 | 0.2500 | **1.0000** |

  Mixed (one dim `v=0.5` slot + one clean slot on the same plane): KEEP → 0.7500, RENORM+SOFT →
  **1.0000**. Unbiased at every overshoot, and it *contains both debate positions as endpoints*
  (`v=1` ⇒ keep, `v=0` ⇒ exclude), so it needs no arbitrary threshold — which is the concession the
  pro-exclude agent itself had to make (it retreated to a tunable `τ≈0.75`).

  **Also established: the "any fix invalidates the frozen bundle" objection is WRONG** for fixes
  placed in `gpu_aug.py` + `splat.py`. `evaluation/engine/build_inputs/cmrxrecon.py` calls
  `reslice_volume_vec` from `respiratory.py` directly; leaving that file untouched keeps the frozen
  `gt/ clean/ breath/` inputs byte-identical, so SVRTK/NeSVoR/NiftyMIC need no re-run. *(Open
  nuance: if the soft weight is applied at eval too, only VGGT's own arm changes — legitimate, but
  it should be disclosed, since baselines receive the same dim slices without the correction.)*

  **⚠️ Correction to this doc's own "94.3% fabricated" framing.** MEASURED: among the 882 subjects
  whose basal-most slot lands off-slab, the **median `v` is 0.776** and only **29.9% have `v ≤ 0.25`**.
  The Lujan waveform's exhale dwell means the typical breath is shallow, so the typical basal slot is
  only ~22% attenuated, not blank. "94.3% fabricated" is true but overstates severity; the severe
  subset is ~30% of that.

  **Surviving flaw (all keep-the-content options share it, incl. `border`):** for the basal-most slot
  `z_i = D−1`, renormalized content is plane D−1 exactly, so its **effective shift is 0 regardless of
  `f`**. The soft weight mitigates this exactly where it matters — the position lie is `f` planes and
  `v = 1−f`, so large lies get small weight — but it does not remove it. It also requires recording
  the **effective** displacement in `resp_disp_mm` (and weighting `metric_resp_*` by `v`), or the
  breathing metric scores the model against a shift that was never applied.

  ### Agreed plan (post-debate)

  All three agents independently agreed on two things: the fade is real and structurally
  uncorrectable by a position-only head, and **the expected headline-PSNR gain is inside the
  appearance-wall noise** (docs/46: ~88% of error is shared appearance synthesis; a measured 2×
  placement improvement bought +0.04 dB). So this is a **training-signal correctness / hygiene**
  issue, to be judged on **basal- and EF-localized** metrics, never on headline PSNR.

  ### ✅ FINAL DECISION (user, 2026-07-31): DOCUMENT ONLY — NO CODE CHANGE

  On proportionality, and it is the right call. Restating the magnitude plainly:
  - a slice that goes **fully black is already free** — the `intensity > 1e-3` gate excludes it from
    both numerator and denominator, so it contributes nothing to the loss. Nothing to fix.
  - a slice that **dims a bit contributes a bit of error** — ~4% of slots, bounded, uncorrectable by
    a position-only head but not a gradient pathology.

  Against docs/46 (~88% of error is shared appearance synthesis; a real 2× placement improvement
  bought **+0.04 dB**), that is inside the noise — while the "fix" needs FOUR coupled changes
  (renormalize, soft weight, effective-displacement bookkeeping, `v`-weighted metrics), each with its
  own risk. **Not worth the surgery.** Recorded here so it is waiting if basal/EF metrics ever look
  suspicious.

  ⚠️ **One cost argument used earlier in this debate is void, and it did NOT change the conclusion.**
  "Don't touch it, it would invalidate the frozen eval bundle and force baseline re-runs" is moot:
  the bundle is **already stale** — CLAUDE.md records that everything derived from the cohort before
  2026-07-31 12:19 is pre-flip stale, and on top of that the cohort is now 1337 (duplicates excluded)
  and the F1 splat fix changed reconstruction. **All arms need re-running regardless**, so
  `respiratory.py` was fair game the whole time. The decision rests on magnitude alone, not on cost.

  **If it is ever revisited**, the option to implement is the debate winner — renormalize + soft
  weight in `gpu_aug.py` + `splat.py`, with a `v < 0.25` floor so near-blank slots are dropped rather
  than noise-amplified by `1/v`, plus effective-displacement bookkeeping in `resp_disp_mm` and
  `v`-weighting of `metric_resp_*`. Measure first (val-only: log per-slot `v`, split
  `recov_frac_heart`/`psnr_3d_motion` by z-band); if the basal deficit is flat in `v` it is the
  appearance wall and the fix would be treating a symptom.

  **Rejected:** threshold exclusion (arbitrary `τ`; the continuous form dominates it), plain
  `border` (same content as renorm but at FULL weight, asserting an off-slab sample is as trustworthy
  as an in-slab one), and clamping `d` (suppresses basal breathing explicitly).

  **Falsifier for step 2:** split `metric_resp_slope_dz` by validity. If the off-slab (`v<1`) slope
  collapses toward 0 while in-slab holds ~0.84, the scheme has taught "the base doesn't breathe" and
  must be rejected in favour of exclusion.

  **The honest limitation, which no option removes:** we never acquired anatomy above the base, so
  deep-breath correction *at the basal-most slice* cannot be taught from this data. Every option
  either fabricates (current `zeros` fabricates darkness; `border` beyond ~half a slice fabricates a
  duplicate) or truncates. The real choice is **fabricate darkness that the position-only head can
  never explain** (today) versus **truncate honestly while keeping real anatomy at correct
  brightness** (`border` + effective bookkeeping). Still worth an A/B against the current behaviour
  before adopting. Related: `loss.py:436` skips slots whose `img_int` is all-zero.

  ⚠️ **The audit called this "censoring the deepest breaths, biasing `metric_resp_*` optimistic".
  That is WRONG, and it is retracted (user challenge, 2026-07-31).** A fully-blanked slot carries
  **no information**: the input is black, so there is nothing in it from which any model could infer
  the applied shift. There is no correct answer to score against, so including it would penalize the
  model for an impossible task, not reveal a weakness. Excluding it is the **correct** behaviour, and
  the metric's meaning is precisely the right one: *"on the slots where breathing correction is
  well-posed, how well does the model do?"* Deep breaths are still represented in the metric via
  (a) deep shifts on apical slices, which never run off the stack, and (b) partially-dimmed slots,
  which are retained. The exclusion is a property of the data and the seed, not of the model, so it
  is also stable when comparing two runs.

  The only legitimate residual is **transparency, not bias**: nothing logs how many slots were
  excluded, so the metric's denominator is invisible. A `metric_resp_frac_offslab` counter would be a
  nice-to-have. Not a defect.

- **F17 — `_find_subjects` has no post-condition.** `mri_dataset.py:222-226` warns and skips a missing
  subject dir. Today **0/1343 skip (MEASURED)**, but nothing compares the loaded count to the split-file line
  count, so a rename or a mount hiccup would silently shrink the cohort with only a startup warning.

  **✅ FIXED 2026-07-31.** `_find_subjects` now collects missing paths and raises `FileNotFoundError`
  naming the count, the split file, `data_root` and the first few offenders. The split file is the
  contract for how many subjects a run trains on; a silently smaller cohort changes epoch length and
  every val mean. Still 0/1337 missing, so it never fires today.

- **F18 — `Z_HALF_MM = 90` has 5.9% headroom, and the guard is a bare `assert`.** **MEASURED:** max
  half-span over all 1343 subjects is **85.00 mm** (`D=18, dz=10`, CMRx25 Philips/Prisma), so `|z_norm|max =
  0.944` and `0/1343` trip. But `assert` is stripped under `python -O`, in which case `z_norm > 1` propagates
  silently into `ZIndexEmbedder`'s period-2 sinusoids (two planes of one subject aliasing) with **no crash**.
  One protocol step away from tripping: `D=20 @10 mm` (190 mm), `D=17 @12 mm` (192 mm).

  **✅ FIXED 2026-07-31.** Replaced with a real `if ...: raise ValueError(...)`, same message. The
  `assert` was the whole problem: stripped under `python -O`, after which `|z_norm| > 1` flows
  silently into `ZIndexEmbedder`'s period-2 sinusoids and aliases two planes of one subject
  together, with no crash.

- **F19 — `S == D` ignores `max_img_per_gpu`.** **MEASURED:** train and val max `D` is **18** (≤ 20, safe
  today), but the pool contains `D=21` and two `D=20` subjects, all currently in **test**. A re-seeded split
  that moves one into train/val gives `S=21 > 20` with no guard. Per-sample cost also varies 5–18× at fixed
  batch size 1 (train min `D=5`).

  **✅ FIXED 2026-07-31 — implemented together with F9; the guard and its measurements are written up in
  the F9 entry above.** In short: `mri_dataset.py` now raises when the data-derived `S` exceeds the
  budget it just overrode, gated on `img_per_seq is not None` so standalone callers (tools, tests, the
  identity gate) are unaffected. MEASURED: max `D` is 18 in train and val, while `D=19` (`ACDC_patient127`),
  `D=20` (`ACDC_patient140`, `MNMs_L8N7Z0`) and `D=21` (`ACDC_patient124`) all sit in **test** — exactly the
  re-split scenario that would otherwise have overflowed silently.

- **F20 — comment/docstring drift.** `respiratory.py:14-16` and `gpu_aug.py:3` still declare `D=12` /
  `spacing (D=Z=12.0…)`; `mri_volume.yaml:45` says "786K voxels" (= 12·256·256); `preprocess.py:253` says the
  mask is `(1,256,256,12)`; `mri_finetune.yaml:226-244`'s freeze rationale reasons about `t_embedder` /
  `target_t_embedder`, which **no longer exist** in `aggregator.py`; `trainer.py:721-727`'s "n is constant
  (3 for t=0..5, 2 for t=6..11) … that drift is the smoke alarm" no longer holds under `ef_val_sweep`;
  `trainer_viz.py:411`'s GIF caption hardcodes "(planes 4-8)".

  **✅ FIXED 2026-07-31** — all corrected: `respiratory.py` / `gpu_aug.py` headers now say `D` / `dz`
  with a native-z pointer; `preprocess.py`'s mask shape is `(1,256,256,D)` and its `torch.quantile`
  comment gives the real range (256·256·D, ~1.5M worst case, 11× under the `2**24` limit);
  `mri_volume.yaml` drops "786K voxels"; `trainer.py` drops the "3 for t=0..5, 2 for t=6..11" that
  `ef_val_sweep` broke; `trainer_viz.py`'s GIF caption drops "(planes 4-8)". The
  `mri_finetune.yaml` freeze rationale — the worst one, reasoning about deleted modules — now opens
  with a ⚠️ noting `t_embedder`/`target_t_embedder` **no longer exist** and that `mri_volume.yaml`
  overrides the whole block to aggft anyway.

  ⚠️ **One drift deliberately NOT changed:** `dataset_name: "pooled1343"` is now numerically wrong
  (cohort is 1337). It feeds `exp_name` → log dir → wandb run name, so changing it renames every
  future run and breaks continuity with the existing series. Left as a user decision.

---

## 7. Verified CLEAN — do not re-audit these

Recorded so a future agent doesn't spend the tokens again. All `MEASURED` unless noted.

**Coordinate / scale math**
- `pz = pos_z·z_scale + (D−1)/2` round-trips to **`|pz − k| < 2e-15`** over every real `(D, dz)` combination.
- The 518↔256 x/y mapping is exact to `3e-15`: `x_norm = px/517·2−1` inverts against
  `px_vox = (x_norm+1)·0.5·255` and matches `F.interpolate(align_corners=True)`. Endpoints are fp-exact
  (0→0, 517→255) — which is precisely why x/y never suffer F1.
- `sample_volume`'s two-step chain (physical → voxel → `grid_sample`'s own `[-1,1]`) is consistent with the
  push side under `align_corners=True`; `max(D-1,1)` is a correct D=1 guard (unreachable, min pool D=5).
- Corner clamping never double-counts: at an exact boundary `x0 == x1` but `wx1 = px − floor(px) = 0`, so all
  weight lands on the single true plane; the 8 corner weights sum to exactly 1 for every in-bounds point.
- `trainer_viz.py` `IN_PLANE_MM = 178.5` and `THROUGH_MM = Z_HALF_MM = 90` are both correct per axis;
  `loss.py:405`'s `through_mm = Z_HALF_MM` is correct, and the breathing metric compares the same axis, units
  and sign (expected slope **+1** for a perfect model, verified at D=7/dz=10, D=12/dz=12, D=21/dz=5).

**Respiratory / augmentation**
- `_norm_delta` is **exact at every pitch**: a delta plane shifted by `d` mm lands at exactly `k − d/dz` with
  full peak amplitude (no interpolation spread), verified at `dz ∈ {5, 8, 9.6, 10, 12}`. Fault injection
  (forcing 12 mm) shifts the landing and/or drops the peak from 64.00 to 32–53 — the fix is real.
- `gpu_aug.py` passes `spacing=(dz,1.4,1.4)` and `n_planes=D` from the right keys/dims; no training-reachable
  caller falls back to `SPACING_MM` or `N_CANON_PLANES=12`.
- Sign convention is physiological post-flip: `d>0` moves anatomy toward lower z = inferior.
- Val breathing is deterministic per `seq_index`; the global RNG stream is bit-untouched by both paths.
- Affine slots verified empirically: `rotate_range` slot 0 **is** the in-plane H-W rotation; `translate/scale`
  slot 0 **is** D and `(0.0, …)` freezes it exactly; `RandFlipd(spatial_axis=[2])` hits **W for both keys**
  (`phases` (B,T,D,H,W) and the unsqueezed `content_mask` (B,1,D,H,W)).

**Data**
- **All 1343 `pooled.txt` paths resolve** (940/134/269) and **all have all 12 `sax_frame_*.nii.gz`** — zero
  silent drops.
- `manifest.csv`: 1343 rows, 1343 unique ids, split column agrees with `pooled.txt` in both directions,
  per-source counts match docs/58 §13 exactly (CMRx23 156/19/20, CMRx24 235/29/30, CMRx25 289/37/33,
  ACDC 85/15/50, M&Ms 175/34/136). `n_z`/`pitch_mm` match on-disk headers **1343/1343**.
- `Z_HALF_MM = 90`: **0/1343 exceed** (max half-span 85.00 mm).
- `ef_val_sweep`: 134 subjects → 268 targets, **each visited exactly once** at `limit_val_batches=268`; all
  ED/ES in `[0,12)`; no degenerate ED==ES; `cardiac_phase.csv` covers all 134 (so neither `KeyError`,
  `ValueError` nor `FileNotFoundError` can fire).
- `heart_roi_canonical.nii.gz` is present for all 1343 and on each subject's own `(256,256,D)` grid — the
  warn-and-skip path never fires and **`metric_psnr_3d_heartseg` is live** (confirmed firing in a real run).

**Plumbing / config**
- `dz_mm`/`z_scale` traverse the full chain intact: `get_data` → the hand-written `ComposedDataset` allowlist
  → collate → `copy_data_to_device` → loss/gpu_aug/viz, including all three hand-built viz batches.
- **No val metric can silently vanish** for a missing `z_scale`: `loss.py:129` raises *outside* any
  `try/except`.
- Every in-scope `splat_to_volume` / `splat_predictions` / `sample_volume` call site passes the correct
  `z_scale`. Signature changes fail **loudly** everywhere (the required-arg design worked as intended).
- Refiner deletion is complete in the live path: zero hits for any refiner symbol in `training/`, `vggt/`,
  `tests/`, or any yaml/sh outside prose.
- All five configs Hydra-`compose(resolve=True)` cleanly; `MRIDataset.__init__`'s signature matches both
  dataset blocks; the gradient clipper handles the removed `["refiner"]` group.
- `D` and `T` are never conflated: `trainer_viz.py:254`'s `T_total = NUM_PHASES` is the phase count and is
  correct; every `phases[...]` index uses dim 1 for T and dim 2 for D.
- `world_points` is fp32 (the point head is wrapped in `autocast(enabled=False)`); `grid_sample` stays fp32
  under autocast.
- `torch.quantile` in `ScaleIntensityByT0PercentilesD` is safe: the hard limit is `2**24 = 16777216` elements
  and the worst case here (256·256·23) is **1507328**, 11× under.
- Coverage division `V/(cov+1e-6)` is correct (additive epsilon, no amplification). Empty-mask bbox fallbacks
  are correct and effectively unreachable under the conservative aug tier. `S` is never 0 or 1.

**Runtime**
- `pytest tests/` — **215 passed**.
- `tools/gate_native_z_identity.py` — fault-injected **0/35 pass** (14–22 dB); real **35/35 at exactly
  120.00 dB**. *(Valid as far as it goes — but see F2 for what it cannot see.)*
- **Real `torchrun --config default`** on the pooled cohort: 940/134 subjects loaded, base weights staged,
  identity baseline, 3 train steps with real gradients, checkpoint saved, full val epoch with the complete
  metric set (including `heartseg` and all `resp_*`), per-phase panels, val volumes written at per-subject
  native `D` (11/9/8). **0 tracebacks, 0 swallowed `try/except` warnings.** GPU aug (conservative) and
  respiratory both fired.

**Not verified (honest gaps):**
- Whether a *converged* model actually develops apex-plane coverage holes on the 56 F1 subjects (§3's
  training-time consequence is inferred, not measured on a checkpoint).
- The true magnitude of F16 (reviewers disagree 2.74% vs 12.4%).
- `heart_roi_canonical` *alignment* (only shape and a weak intensity-ratio proxy were checked; a Dice against
  `heart_seg_canonical` on the canonical grid would settle it).
- First-epoch monai cache build wall-clock at 1074 subjects (F5 triples the write volume).

---

## 8. Reproduction

- F1 scope + the splat drop: iterate `manifest.csv`, compute
  `pz = float32((k−(D−1)/2)·dz/90) · float32(90/dz) + (D−1)·0.5` for `k ∈ {0, D−1}`, flag `pz ∉ [0, D−1]`.
- F1 cost: `compute_volume_intensity_loss` with `world_points = scanner_coords` on `MNMs_C6E0F9` /
  `MNMs_B0H7V0`, against a monkeypatched `splat_to_volume` carrying the EPS+clamp.
- F2: run the gate's `_identity_psnr` on `MNMs_G7S6V0` and print `cov[0][0].sum()` alongside the PSNR.
- F3: md5 the 12 canonical frames pairwise; then `np.array_equal` the **raw source** `*_4d.nii.gz` /
  `*_sa.nii.gz` arrays to establish origin.
- F6: `torch.randperm(1000)` under the real `DynamicDistributedSampler`, `Counter(i % 940)`.

Related: docs/58 (the migration being audited), docs/38 (the val ship-decision metrics F1 corrupts),
docs/56 (slice roll), docs/27/54/55 (pitch and recon provenance).

---

## 9. What is left to do (updated 2026-08-01, re-verification pass)

**All 21 findings are closed.** **15 fixed in code** (F1–F7, F9, F11, F15, F17, F18, F19, F20, and
**F21**); F14 verified benign at audit time; **F8 and F16 accepted as-is** after measurement plus
explicit user decisions; F10, F12, F13 reviewed and deliberately not fixed. Everything below is
either an accepted limitation worth citing in a paper, a deliberate non-change, or an honest gap in
what was verified.

⚠️ **This section previously read "ALL 20 findings are closed / Nothing is left open" — and that was
wrong**, because the F6 fix had silently introduced **F21** (§10), which killed the EF metric. It was
found only by a re-verification pass that re-ran the measurements instead of trusting the status
column. **Do not read a ✅ in §2 as evidence; re-measure.** The re-verification also confirmed every
other ✅ by measurement (apex-plane coverage restored `0.0 → 203086` with **+7.6 / +6.6 dB** identity
PSNR on the two affected val subjects; the repaired gate FAILS under fault injection at 31.03 dB;
cache entries 47.8 MB → 16.4 MB; 2D SSIM D-spread 1.5e-2 → 1.4e-4; NaN/±Inf produce zero NaN voxels;
duplicates 1343 → 1337 with one member of each pair excluded).

### Accepted limitations (closed — worth citing as limitations, not bugs)

~~- **F8 — val does not cover the pitch range native-z exists to handle.**~~ **CLOSED as ACCEPTED
  2026-07-31 (user decision).** Deviating from the M&Ms official split is not worth the
  reproducibility cost, and the audit's "largely unmeasurable" framing was too strong: **val already
  spans 5.0 / 7.0 / 8.8 / 9.6 mm (9 of 133 subjects, 6.8%)**, including the 5 mm extreme. Cross-pitch
  behaviour is therefore measurable, just **under-powered for a per-pitch breakdown** — report
  sub-10 mm val as a pooled group. Only 8.00 mm is genuinely absent. See §6 F8 for the full table and
  why re-drawing the CMRx splits could never have helped (they are exclusively 10/12 mm).
~~- **F16 — respiratory zero-padding fabricates dimmed/blank input slices.**~~ **CLOSED as ACCEPTED
  2026-07-31 (user decision, on proportionality).** Magnitude dispute resolved by measurement (2.18%
  fully blank / 9.48% dimmed / 11.66% either — the two reviewers were counting blank-only vs
  blank+dimmed and **both were right**). Fully-black slices are **already free** (the
  `intensity > 1e-3` gate excludes them from numerator AND denominator); dimmed ones are a bounded
  ~4%-of-slots noise floor, inside docs/46's appearance-wall noise, against a four-part fix.
  **Documented, not fixed.** A 3-agent debate identified the option to use if it is ever revisited
  (renormalize by the exactly-known attenuation + soft splat weight — unbiased at every overshoot,
  with exclude/keep as its endpoints); see §6 F16. Also retracted there: the audit's claim that
  skipping all-zero slots biases `metric_resp_*` optimistic — a blank slot has no signal to correct,
  so excluding it is correct.

### Reviewed and deliberately NOT fixed

- **F10 — `ef_val_sweep` ED/ES are segmentation-derived.** *Keep them.* Rationale: **preprocessing
  consistency** — the nnU-Net-derived `cardiac_phase.csv` covers all 1337 subjects uniformly, while
  source labels exist only for ACDC/M&Ms under a different convention. Mixing would make EF
  incomparable *between sources*, which is worse than uniform label noise. Accepted cost: 20/49 val
  subjects disagree with their source ES label (worst case half a cycle). The "GT ED/ES" wording in
  the docstring/config is still inaccurate.
- **F12 — refiner-era checkpoint dies confusingly.** *We will not load one.* The seed the sbatch
  scripts use has no refiner keys and the failure is loud, not silent.
- **F13 — `val_volumes/` write volume.** *Accepted.* Verified from `trainer_viz._save_val_volumes`:
  filenames are deterministic and val is `shuffle=False`, so the **same files are overwritten every
  epoch** — ~925 MB is a **constant footprint, not cumulative**. Only per-epoch write time remains.

### Deliberate non-changes (naming / schedule)

- **`max_epochs` left at 200.** With the epoch now 935 steps (F6), a full run is **187k steps,
  −6.5%**. Raise to ~214 to hold the step count.
- **`dataset_name` → `pooled1337` (DONE)**, and all 26 `EXP_NAME=` lines in `sbatch/` renamed.
  The 8 remaining `pooled1343` strings are `RESUME_FROM` / `TREAT_DIR` / `CTRL_DIR` paths pointing at
  **existing run directories on disk** and must NOT be renamed.
- **`img_nums` deliberately KEPT** when `max_img_per_gpu` was deleted (F9). It is not dead: it is the
  S cap the F19 guard enforces, and the slot budget whenever `one_frame_per_slice` is false. Deleting
  it too would have silently disabled that guard.
- **`vggt/utils/fused_ssim_compat.py` is now unimported** (F11 dropped the 3D SSIM, its only caller).
  Left on disk — it is a standalone utility, not something this work authored. Delete only on purpose.
- **The gather aux at `loss.py:192` was NOT given the F15 `torch.where` treatment.** It is
  `gather_weight: 0.0` in every live config, i.e. dead code today. Noted rather than touched; fix it
  if the gather aux is ever re-enabled.

### Standing action item (NOT part of this audit, but blocked on it)

**The frozen `evaluation/` bundle and every SVR baseline arm need rebuilding + re-running.** Three
independent reasons, none of them F16: the cohort is now **1337** (duplicates excluded, F3), the
slice-order flip landed 2026-07-31 (docs/58 §10a — CLAUDE.md records that anything derived from the
cohort before 12:19 that day is pre-flip stale), and the **F1 splat fix changes reconstruction** on
56 subjects. So `gt/ clean/ breath/` plus SVRTK / NeSVoR / NiftyMIC outputs are all stale.
*Consequence worth remembering: this also voids the "don't touch respiratory.py, it would invalidate
the bundle" cost argument used earlier in the F16 debate — the bundle was already invalid.*

### Not verified (honest gaps)

- Whether a *converged* model develops apex-plane coverage holes on the 56 F1 subjects. Moot going
  forward, but **every checkpoint from before 2026-07-31 carries it**, so old-vs-new comparisons are
  not apples-to-apples on those subjects.
- `heart_roi_canonical` *alignment* (shape + a weak intensity proxy only; a Dice against
  `heart_seg_canonical` would settle it).
- First-epoch monai cache build wall-clock at ~1068 subjects — ~2.9× cheaper after F5, unmeasured.
- The F16 measurement is one displacement draw per subject at a fixed seed; the ±spread across
  seeds was not characterized.

### Also worth noting

The M&Ms **metadata self-contradiction** surfaced by F3 (identical pixels labelled Philips/centre-2
*and* GE/centre-4; F *and* M) is unresolved and weakens the unseen-vendor-test premise in docs/58
§2.1/§13. A data-quality fact about M&Ms, not something we can fix.

---

## 10. F21 — the fix pass introduced a new critical regression (found 2026-08-01)

> **Read this before "fixing" any sampler-length knob.** F6 asked for one line. That line was
> correct for its own purpose and broke something two files away.

### What broke

`training/data/datasets/mri_dataset.py` — F6 replaced `len_train = max(1000, len(subjects))` with
`len_train = len(self.subjects)`, assigned at line 173, **before** `val_targets` is built at 175+.

`__len__` returns `len_train`, and the dataloader cannot yield more samples than the dataset
declares. Under `ef_val_sweep: true` the sweep list is
`[(i, ED) for every subject] + [(i, ES) for every subject]` — **length 2N** — indexed by
`seq_index % len(val_targets)`. With `__len__ = N = 133`, `seq_index` only ever reached `0..132`:

```
val subjects        133
val_targets (ED+ES) 266
len(dataset)        133      <-- the sampler's ceiling
ED half reached     133/133
ES half reached       0/133  <-- EVERY ES entry unreachable
```

`trainer.py:627`'s `limit_val_batches = len(val_targets) = 266` **cannot** rescue it — that raises
the iteration cap, not the dataset length.

### Consequence

`EF = (EDV − ESV)/EDV`. With no ES volume there is nothing to compute, so the predicted-EF metric —
the reason `ef_val_sweep` exists (docs/24, 25, 33) — is silently dead. Per-phase val panels also
covered ED only.

**Under-appreciated detail:** F6 traded a *bounded, quantified* 1.60× ACDC oversampling for a
*total* loss of the EF metric. The pre-F6 `max(1000, 133) = 1000 ≥ 266` had made all 266 reachable
by accident, so this is a strict regression.

### How it was caught (and why the existing checks could not)

- `pytest tests/` was **251 green with the bug present** — no test constructs a val dataset with
  `ef_val_sweep=True` and asserts `len(ds) == len(ds.val_targets)`.
- `tools/gate_native_z_identity.py` builds its own single-subject splits; it never touches the sweep.
- The identity-baseline log line even printed `ef_val_sweep ON — 266 (subject, t_target) pairs`,
  which looks correct — the truncation happens later, in the loader.
- It took **an end-to-end `torchrun` on the real default config**, then two independent tells:
  the per-phase panel printed only `t0,t1,t7,t8,t10,t11` (exactly the measured ED distribution
  `{0:114, 1:4, 7:1, 8:1, 10:1, 11:12}`), and `${log_dir}/ef_tmp/pred/` held **133 volumes, not 266**.

### The fix

Set `len_train` **after** the sweep is built, from the thing an epoch actually enumerates:

```python
self.val_targets = self._build_val_targets(csv_path)
...
self.len_train = len(self.val_targets)      # an epoch enumerates SWEEP ENTRIES, not subjects
```

Verified (all measured):

| check | result |
|---|---|
| val + sweep: `len(dataset) == len(val_targets)` | 266 == 266 |
| every sweep entry reachable exactly once | 266/266 |
| ES half reachable | 133/133 |
| `t_target` coverage | `{0:115, 1:4, 3:7, 4:16, 5:36, 6:43, 7:19, 8:6, 9:3, 10:4, 11:13}` — spans ES |
| val **without** sweep: `len == n_subjects` (F6 intact) | 133 == 133 |
| train: `len == n_subjects` (F6 intact) | 935 == 935, `val_targets is None` |
| train: every subject exactly once per epoch | distinct 935, max repeats 1 |

### Regression test — ADDED

`tests/test_ef_val_sweep_length.py` (5 tests, synthetic fixtures, no real data). It asserts
`len(dataset) == len(val_targets)`, that every sweep index — the ES half specifically — is reachable
from the `seq_index` range an epoch issues, and that both ED *and* ES `t_target` values are actually
visited (the tell the real run's per-phase panel would have shown). It uses **two** val subjects, not
one, so the failure is the same factor-of-2 truncation as the real 133-vs-266 case rather than an
off-by-one. A sixth guard covers the opposite direction (F6): with the sweep off, `len == n_subjects`.

**Proven to have teeth, both ways.** In-test: `test_fault_injection_the_assertions_have_teeth`
re-injects `len_train = len(subjects)` on the object and asserts the ES half goes unreachable.
Against the source: reverting the one-line fix in `mri_dataset.py` makes exactly the 3 core tests
fail (`test_sweep_length_is_targets_not_subjects`,
`test_every_sweep_entry_including_ES_is_reachable`, `test_sweep_visits_both_ED_and_ES_phases`);
restoring it returns the suite to green. Full suite: **256 passed**.

### The generalizable lesson

`len_train` is not a private field: it is read by `__len__`, which bounds the dataloader, which
bounds the val epoch. **Any change to a dataset's declared length must be checked against every
per-epoch enumeration that length gates** — here, a sweep whose length is a *multiple* of the
subject count. Both F6 and F21 are the same underlying hazard (docs/58 §4 item 4) pointing in
opposite directions: F6 was `len_train > N_entries` causing duplicates; F21 was
`len_train < N_entries` causing silent truncation. The invariant to assert is
`len(dataset) == number_of_things_one_epoch_should_visit`, whatever that happens to be.

---

## 11. F14 fixed too (2026-08-01), and the final end-to-end verification

### F14 — `ef_eval.py` now uses each subject's own pitch

Fixed even though it was verified **benign for every reported metric** (EF is a ratio, so the voxel
volume cancels at `ef_eval.py`'s `(v_ed - v_es)/v_ed`; and nnU-Net `-m 2d` reproduces input geometry
verbatim). It was fixed because the dumped NIfTIs were geometrically false — up to 2.4× wrong in z at
5 mm — every absolute mL was off by `dz/12`, and the wrong header **would** have started changing the
segmentation the moment anyone switched to `3d_fullres`, which does resample z.

Three edits: `save_pred_volume(..., dz_mm)` takes the subject's pitch and is **required, no default**
(same rationale as `splat.py`'s `z_scale` — a silent fallback writes a plausible-looking volume with
false geometry and nothing errors); `_lv_ml` reads the voxel volume from the **seg's own header**
instead of a module constant, so writer and reader cannot drift; the `CANON_SPACING`/`VOX_ML`
constants are deleted. `trainer_viz.py:487` passes `batch["dz_mm"]`, which was already in the batch.

The `_lv_ml` half is only sound because of a measurement, not an assumption: over the 133 real ED/ES
pairs from a prior run, nnU-Net's output geometry is **identical to its input** — same zooms, same
shape, with `D = 8..13` passing through unresampled. That is what guarantees the seg header equals
what `save_pred_volume` wrote.

Verified: written headers carry the right pitch at `dz ∈ {5.0, 9.6, 10.0, 12.0}`, `_lv_ml` returns
the exact analytic volume in each case, and omitting `dz_mm` raises `TypeError` rather than silently
falling back to 12 mm.

### Final end-to-end run (real default config, `ef_val_sweep` + `ef_eval` both on)

`torchrun --config default max_epochs=1 limit_train_batches=3`, epoch 0 (an EF epoch, since
`epoch % ef_eval_every_n_val_epochs == 0`):

| check | before the F21/F14 fixes | after |
|---|---|---|
| tracebacks | — | **0** |
| val steps reached | 133 (truncated) | **265** (0-indexed ⇒ all 266) |
| per-phase panel | `t0,t1,t7,t8,t10,t11` (ED only) | **`t0,t1,t3,t4,t5,t6,t7,t8,t9,t10,t11`** — ES range present |
| volumes written to `ef_tmp/pred/` | 133 | **266** |
| distinct z spacings written | `{12.0: 133}` (all wrong) | **`{5.0:8, 7.0:2, 8.8:4, 9.6:4, 10.0:104, 12.0:144}`, 0 mismatches vs manifest** |
| EF metric | never computed | **`slope=0.262 spearman=0.019 mae=57.80% n=132`** |
| `pytest tests/` | 251 | **256** |

(The EF *values* are meaningless here — 3 training steps from base VGGT-1B weights. What matters is
that the metric is computed at all, over 132 subjects, instead of being silently absent.)

### What this run did NOT exercise

Stated so nobody reads "end-to-end verified" as broader than it is:

- **Multi-epoch behaviour** — 1 epoch only. Checkpoint resume / SLURM requeue, the `val_epoch_freq`
  and `filmstrip_every_n_val_epochs=5` cadences, and LR-schedule progression are unexercised.
- **A full train epoch** — 3 batches of 935. Long-run memory stability and the cold-cache build for
  the 935 train subjects (only val is warm) are unmeasured.
- **`WANDB_MODE=online`** — the run was offline.
- **F1 coverage in the gate is thin**: only **1 of the 56** affected subjects (`MNMs_G7S6V0`) is in
  `gate_native_z_identity.py`'s subject list. It does fire on it (verified by fault injection), but
  adding a couple more `dz=9.6, D∈{12,13,14}` subjects would make that gate less lucky.
