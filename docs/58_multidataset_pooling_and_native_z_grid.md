# 58 — Multi-dataset pooling and the native-z canonical grid

> **TL;DR & takeaway**
>
> We are growing the training cohort from **294 subjects (CMRxRecon2024 only) to ~1343** by pooling
> CMRxRecon **2023 + 2024 + 2025** (848) with **ACDC** (150) and **M&Ms-1** (345). The blocker is
> geometric, not logistical: the canonical grid is hardcoded to `(256, 256, 12)` at
> `(1.4, 1.4, 12.0)` mm, and that only ever worked because **CMRxRecon2024 is uniformly 12 mm**.
> About **45% of the new pool is not 12 mm** (5–12 mm), so the existing `Spacingd` would start doing
> real through-plane interpolation — destroying exactly the slice-spacing diversity that motivated
> pooling, and mismatching the real-time free-breathing (RTFB) deployment case, where you *cannot*
> resample in z because neighbouring slices are different respiratory/cardiac states.
>
> The obvious middle road — keep the fixed 12 mm target, feed native slices — is a **trap**, and this
> is measured: an *identity* splat (Δ=0, a perfect model) of a 10 mm ACDC stack into a 12 mm grid
> tops out at **25.4 dB**, versus **numerically exact (120 dB)** when input slices coincide with grid
> planes. Cause: `V_gt` is built by *pull* interpolation and the splat *pushes*; the two agree only
> on-grid. The same measurement condemns `continuous_z`, which puts inputs off-grid deliberately —
> **28.3 dB** at the default `z_jitter=0.5`.
>
> **Decision: never resample z.** In-plane still goes to 1.4 mm / 256×256; z is left at whatever the
> scanner acquired, and the canonical volume becomes `(D, 256, 256)` where `D` = that subject's slice
> count. `V_gt` is then literally the acquired slice stack, `V_canon` is allocated to match per sample,
> and the identity splat is exact for every pitch (verified). Because `D` and the stack height now
> vary (~30–168 mm across the pool), the depth coordinate must be made **physical**:
> `z_norm = z_mm / Z_HALF_MM` with `Z_HALF_MM = 90`, and the normalized→voxel converters must be told
> the scale (`z_scale = Z_HALF_MM / dz`) instead of assuming the volume spans ±1. `Z_HALF_MM = 90`
> rather than 72 because `ZIndexEmbedder`'s sinusoids have period 2 and the tallest stack would
> otherwise alias *within a single subject*. `continuous_z` goes off.
>
> **Status: IMPLEMENTED AND VERIFIED (2026-07-31).** All of §8.1's training-path changes are done.
> The fault-injected identity gate (`tools/gate_native_z_identity.py`, not the originally-planned
> `probe_zgrid_alignment.py::e3_native_grid` — see §12) passes exactly-120dB on 35 real subjects
> spanning 5–12mm pitch and correctly fails at 14–22dB when `z_scale` is deliberately corrupted;
> `pytest tests/` is green (203/203); a real `torchrun` smoke run completes cleanly; a 6-reviewer
> `/prove-it` pass found and fixed one real live bug (§12). **§8.2 (manifest CSV + pooled split
> file) is now DONE too — see §13.** `training/splits/manifest.csv` (1343 subjects, real on-disk
> geometry + joined demographics) and `training/splits/pooled.txt` (928→940 train / 134 val / 269
> test depending on iteration, final: 940/134/269 ≈ 70:10:20 — **later 935/133/269 = 1337 once 6
> source-shipped duplicate subjects were excluded, see §13a**) are built and verified through the
> real `MRIDataset` loader. **The config is now switched over (§14)** — `mri_finetune.yaml` (and
> everything that inherits it) points at `pooled.txt`/`data_root=scratch/data`, and
> `heart_roi_canonical`/`heart_seg_canonical` (§8.4) and `cardiac_phase.csv` (§8.4) are both
> regenerated/built, so a real pooled `torchrun` with `D≠12` and `ef_val_sweep: true` is unblocked.
> §8.3 (`inference/`/`evaluation/`/`tools/`) is deliberately still deferred, per the original plan.
> The `enable_refiner` open item in §10 is moot — **the refiner was removed entirely**, see §12.

---

## 1. Why we are transitioning

Training has used **CMRxRecon2024 `Cine_combined` only** — 301 directories, 294 unique subjects after
the duplicate archival, split 240/30/31 by `training/splits/random_8_1_1.txt` (now deprecated; see the
header of that file). Since then three things landed that make a much larger cohort available:

- **CMRxRecon2023 and 2025 were reconstructed** to the identical on-disk layout (docs/54), pitch-
  relabeled (docs/27, docs/55) and slice-roll-fixed (docs/56).
- **ACDC** (150) and **M&Ms-1** (345) are on disk as magnitude NIfTI — no reconstruction needed.
- Within-year duplicates are resolved in all three CMRx years.

The motivation is not only subject count. The current cohort is **one vendor family, one protocol, one
slice pitch, healthy volunteers**. The pooled cohort adds four vendors (incl. Canon and GE), multiple
centres and field strengths, a large pathology fraction (ACDC 5 groups; M&Ms ~74% diseased), and — the
point of this document — **a range of slice spacings**.

## 2. What is available

| source | subjects | on-disk layout | T native | z pitch | z extent |
|---|---|---|---|---|---|
| CMRxRecon2023 | 195 | `<ID>/sax/3d_recon/sax_frame_00..11.nii.gz` | **12** | 12 mm ×138, 10 mm ×2 | 90–144 mm |
| CMRxRecon2024 | 294 | same | **12** | 12 mm | 72–168 mm |
| CMRxRecon2025 | 359 | same | **12** | 12 mm ×101, 10 mm ×39 | 60–180 mm |
| ACDC | 150 | single `patientNNN_4d.nii.gz` | 12–35 | **10 mm ×127, 5 mm ×21, 8 mm ×2** (all 150) | 50–110 mm |
| M&Ms-1 | 345 | single `<CODE>_sa.nii.gz` | 18–36 | **10 ×250, 9.6 ×56, 8.0 ×23, 8.8 ×9, 9.52 ×2, 5.0 ×2, 9.96/8.05/6.0 ×1** (all 345) | 32–150 mm |
| **total** | **1343** | | | **5–12 mm** | **32–180 mm** |

(CMRx rows are from a 140-subject random sample per source. **ACDC and M&Ms rows are exact, full-cohort**
— ACDC from the published protocol, M&Ms from a full 345-file header scan; both are now recorded in the
respective dataset `README.md`. Slice counts: CMRx 6–14, M&Ms 5–20, ACDC up to ~23 at 5 mm — so `D`, and
hence `S` under `one_frame_per_slice`, spans roughly **5–23**. Extents are `(D−1)·dz`.)

**Pitch is pitch, not thickness, for both new sources** — the question that burned CMRx (docs/27, header
carried `SliceThickness`). ACDC's `pixdim[3]` includes the interslice gap for 100% of subjects (documented
protocol). M&Ms is *empirically indicated* rather than documented: the non-round values `8.05`, `9.52` ×2 and
`9.96` are the signature of a spacing computed from `ImagePositionPatient` deltas, i.e. measured
centre-to-centre — a thickness-valued header is all round numbers. **Not verified against DICOM**; the cheap
confirmation is an LV long:short axis-ratio comparison against ACDC once the segs exist.

Pooled pitch distribution: **~55% at 12 mm, ~36% at 10 mm, ~7% at 8–9.6 mm, ~2% at 5–7 mm.**

### 2.1 Proposed splits

**Superseded by §13 — kept as the original plan record.** The actual implemented split treats
each CMRx year separately (not "CMRx ×3" as one stratified pool) and uses a residual mechanism
for 2025 to hit an exact grand-total ratio; see §13 for the real numbers and rule.

| source | train | val | test | rule |
|---|---|---|---|---|
| CMRx ×3 | ~746 | ~51 | ~51 | random, stratified by year / centre / pitch; keep 2024's current val/test members where stratification allows |
| ACDC | 85 | 15 | 50 | carve val from `training/` (3 per pathology group); keep official `testing/` **intact** — it is the published benchmark and docs/46 reports on it |
| M&Ms-1 | 175 | 34 | 136 | official three-way split, as shipped |
| **total** | **~1006** | **~100** | **~237** | |

M&Ms should be taken as-shipped because of its vendor composition (verified from
`211230_M&Ms_Dataset_information_diagnosis_opendataset.csv`):

```
Training/Labeled     150   Philips 75, Siemens 75
Training/Unlabeled    25   GE 25
Validation            34   Canon 10, GE 10, Philips 10, Siemens 4
Testing              136   GE 40, Canon 40, Philips 40, Siemens 16
```

**Canon appears only in Validation and Testing.** Respecting the official split therefore buys a
genuine unseen-vendor test for free. The cost is 50 Canon subjects absent from training; if that turns
out to matter, folding half of `Testing` into train is a one-line manifest edit.

~100 val subjects × 2 for the `ef_val_sweep` ED/ES pass = ~200 entries, which lands on the existing
`limit_val_batches=200`. **Stale — the real val set is 134 (§13), so 268 entries; `limit_val_batches`
was raised to 268 when the config was switched over (§14). At 200 the last ~68 ES entries would
never have been evaluated.**

**Split file mechanics.** `MRIDataset._find_subjects` does `os.path.join(self.data_root, line, "sax")`,
so a split-file line containing `/` already works. Set `data_root = /home/minsukc/vggt/scratch/data`
and write lines as relative paths (`CMRxRecon2024/Cine_combined/CMRx24_Train_P001`,
`ACDC_sax/ACDC_patient001`, …). **No loader change.** Generate the `.txt` from a manifest CSV
(`id, rel_path, source, n_z, pitch_mm, z_extent_mm, T_native, vendor, centre, pathology, ed, es, split`)
so stratification is auditable and regenerable.

## 3. The data design we are leaving

`training/data/preprocess.py`:

```python
TARGET_SPACING = (1.4, 1.4, 12.0)
TARGET_SHAPE   = (256, 256, 12)          # (X, Y, Z), monai order
```

Every subject is resampled on **all three axes** and cropped/zero-padded to one cube of
358.4 × 358.4 × 144 mm.

- **`V_gt`** = the target-phase volume on that cube, `(12, 256, 256)` in splat order.
- **`V_canon`** = splat into `(12, 256, 256)`; `grid_shape` comes from the config constant
  `gt_grid_shape: [12, 256, 256]`.
- **input slices** = read off *integer planes of the same cube* (`phases_splat[t, z]`,
  `mri_dataset.py:455`).
- **`z_norm`** = `k/(D−1)*2−1` (`mri_dataset.py:472`) — grid-relative, i.e. ±1 is *defined* as the
  cube's outermost planes.
- **`continuous_z`** (default off) optionally jitters input slot z by ±`z_jitter` planes and extracts
  by 2-plane blending.

**Why it worked:** CMRxRecon2024 is uniformly 12 mm, so `Spacingd`'s z step was an identity. `V_gt`
planes *were* the acquired slices, and input slices landed exactly on grid planes. The whole scheme was
aligned by the accident of a single homogeneous cohort.

## 4. What breaks with the pooled cohort

1. **The z resample stops being an identity for ~45% of subjects.** For those, `Spacingd` performs real
   through-plane interpolation: `V_gt` planes become blends rather than acquired images, and the
   spacing diversity that motivated pooling is erased before the model ever sees the data.
2. **The naive fix is a trap** (§5). Feeding native slices into the fixed 12 mm target measures a
   **25.4 dB** ceiling.
3. **~5% of subjects are taller than 144 mm** (CMRx2025 to 180 mm, M&Ms to 160 mm) and would be
   silently center-cropped.
4. **`MRIDataset.len_train = 1000`** with `seq_index % len(self.subjects)` (`mri_dataset.py:167, 281`)
   — the sampler draws indices `0..999`, so at ~1006 train subjects **everything past index 999 is
   never sampled.** Silent.
5. **ACDC and M&Ms need converting**: single 4D files, T = 12–35 and 20–34. M&Ms additionally stores
   **true-oblique affines** — sampled slice-normal `|S|` component averages **0.437** (dominant axis R
   or A), versus **exactly 1.0** for CMRx and ACDC. `Orientationd(axcodes="LPS")` assigns array axes by
   dominant patient axis, so for M&Ms it puts the 10 mm slice axis into an in-plane slot and `Spacingd`
   then resamples the wrong axes.
6. **`trainer_viz.py:250`** does `T_total = mri_ds.gt_grid_shape[0]` — that is the **plane** count, used
   as the **phase** count. Correct today only because both are 12.

## 5. Options considered, and the measurements

All numbers from `tools/probe_zgrid_alignment.py`. "Identity splat" = splat the input slices back with
**Δ = 0** on a single static frame; a perfect model can do no better, so it is a **ceiling**, not a
baseline. PSNR is over `(coverage > 0.5) & (V_gt > 0)`; 120 dB is the script's clamp for
numerically-exact.

### Option 1 — resample everything to 12 mm (status quo)

`V_gt` and `V_canon` always `(12, 256, 256)`. Zero code change; inputs stay on-grid so the identity
splat stays exact.

**Rejected.** It gives the model **exactly one** slice spacing, forever. At inference on RTFB the stack
arrives at whatever pitch the scanner used and **cannot be resampled in z** — slice *k* and *k+1* are
acquired at different times, respiratory positions and cardiac phases, so interpolating between them
blends two different anatomical states rather than producing intermediate tissue. So you would be
forced to feed off-grid slices to a model that has never seen one, into a grid that penalises it
(§5.3). Option 1 would suffice for simulated + gated validation and would fail exactly where the
research goal lives.

### Option 2 — native z spacing (chosen)

Only in-plane is resampled. `V_gt` and `V_canon` become `(D, 256, 256)` where `D` = the subject's
slice count:

```
12 mm × 12 slices  =>  (12, 256, 256)
10 mm × 14 slices  =>  (14, 256, 256)
```

### 5.3 The trap: native slices into a fixed 12 mm target

```
E1  fixed 12 mm target grid: inputs ON grid vs native/OFF grid
  CMRx24 P001    native 12.0 mm x 10   on-grid 120.00 dB   native/off-grid 120.00 dB
  ACDC p001      native 10.0 mm x 10   on-grid 120.00 dB   native/off-grid  25.41 dB
```

CMRx is unaffected because 12 mm native *is* the grid. ACDC at 10 mm is capped at **25.41 dB** — below
what the model already scores.

**Mechanism.** `V_gt` is produced by `Spacingd`, i.e. *pull* interpolation (sample the volume at each
output plane). The splat *pushes* (scatter each input pixel into the two bracketing planes). These are
different operators. They agree exactly when input positions coincide with output planes and disagree
otherwise. 10 mm slices into a 12 mm grid never line up.

### 5.4 The same measurement condemns `continuous_z`

```
E2  continuous_z jitter on the fixed 12 mm grid (CMRx24 P001)
  jitter 0.0 (continuous_z OFF)        120.00 dB
  jitter +-0.25 plane (= +-3 mm)       31.79 dB
  jitter +-0.5 plane (= +-6 mm)        28.31 dB
```

`continuous_z` deliberately puts inputs off-grid, so it triggers the same mismatch. At the default
`z_jitter=0.5` the identity ceiling sits at **28.3 dB**, around the model's operating point.

Two caveats on reading this. It is the **identity** floor, not the model's ceiling — the model is given
each slot's `z_norm`, so it *could* learn a Δz that re-centres content onto planes and recover much of
it. But that is capacity spent undoing an augmentation, entangled with the motion it is meant to
predict, and it would manifest as a large near-constant whole-field Δz — which is what was previously
measured on the `s20contz` models (memory: "gather-aux & continuous-z look worse OOD = relocation").
**That connection is a hypothesis, not verified here**; it is cheap to test by comparing `mean_t(Δz)`
against the jitter actually applied. Under Option 2 the jitter is no longer needed as a stand-in — real
off-grid variation comes from the data — so `continuous_z` goes **off**.

### 5.5 Option 2 verified

```
E3  proposed design: z NEVER resampled, D = native slice count
  CMRx24 P001    dz=12.000 mm (native)  D=10  span= 108.0 mm   identity 120.00 dB
  ACDC p001      dz=10.000 mm (native)  D=10  span=  90.0 mm   identity 120.00 dB
  ACDC p002      dz=10.000 mm (native)  D=10  span=  90.0 mm   identity 120.00 dB
```

Exact for every pitch, because the grid planes **are** the acquired slices.

`Spacingd(pixdim=(1.4, 1.4, 0.0))` is the mechanism — MONAI documents that a non-positive `pixdim`
component keeps that axis's original spacing, and the observed `dz` values (10.000, 12.000) confirm it.
`ResizeWithPadOrCropd(spatial_size=(256, 256, -1))` leaves the z length alone.

## 6. The chosen design

```
in-plane : resample to 1.4 mm, crop/pad to 256x256          (unchanged)
z        : NEVER resampled.  D = subject's slice count
V_gt     : the acquired slices, stacked  ->  (D, 256, 256)
V_canon  : splat into the same (D, 256, 256); grid_shape read from the batch
inputs   : integer planes of that subject's own stack  -> always aligned
z_norm   : z_mm / Z_HALF_MM,  Z_HALF_MM = 90.0
-> grid  : z_grid = z_phys * (Z_HALF_MM / dz) / ((D - 1) * 0.5)
continuous_z : false
```

At inference the rule is **identical**: `dz` = the acquired pitch, `D` = the number of slices, feed each
slice at its integer plane. No resampling of RTFB inputs, ever, at any pitch.

### 6.1 Why the depth coordinate must become physical

Under Option 2 there is no common cube, and stack heights across the pool span roughly **30–168 mm**.
The existing grid-relative `z_norm = k/(D−1)*2−1` makes ±1 mean "this subject's outermost planes", so
the millimetres-per-unit ruler varies by **5.6×**:

| subject | stack height | 1.0 of `z_norm` | `Δz = 0.1` means |
|---|---|---|---|
| 12 slices @ 12 mm | 132 mm | 66 mm | 6.6 mm |
| 13 slices @ 10 mm | 120 mm | 60 mm | 6.0 mm |

The model cannot observe the span, so it would have to emit different outputs for identical physics.
With `z_norm = z_mm / 90`, both need `Δz = 6/90 = 0.0667`. The nuisance variable disappears, and the
*gap between consecutive input `z_norm` values becomes the pitch on a common scale* — real information
the model currently never receives.

### 6.2 Why the converters must be told the scale (this is the subtle one)

`splat_to_volume` (`splat.py:33-35`) and `sample_volume` (`splat.py:143`, via
`F.grid_sample(align_corners=True)`) both **assume the volume spans exactly ±1**:

```python
pz = (pos[..., 2] + 1) * 0.5 * (D - 1)
```

That is correct today because `z_norm` is *defined* as grid-relative. With a physical `z_norm` it is
wrong for **every** subject, not just tall ones. A 10-slice 10 mm stack spans `z_norm ∈ [−0.5, +0.5]`,
and the formula stretches ±1 across the grid, so the subject is compressed into the middle half of its
own volume:

```
plane 0 -> pz = 2.25   (should be 0)
plane 9 -> pz = 6.75   (should be 9)

V_canon planes:  0    1    2    3    4    5    6    7    8    9
content:        ---  ---  [######## all 10 slices ########]  ---  ---
V_gt:           [######## real anatomy in all 10 planes ########]
```

Planes 0, 1, 8, 9 of `V_canon` would be empty while `V_gt` has real anatomy there — a large loss the
model cannot fix, because the error is in the coordinate mapping. The error scales with how far the
subject's span is from `2 × Z_HALF_MM`: a 168 mm stack is off by ~7%, a 90 mm stack by 50%.

Fix:

```python
pz = pos[..., 2] * z_scale + (D - 1) * 0.5        # z_scale = Z_HALF_MM / dz
```

Make `z_scale` a **required** argument on both functions, not an optional one. A silent default means a
missed call site produces a plausible-looking but compressed volume. Note in particular that
`sample_volume` is used at `loss.py:398` to build the **oracle** behind
`metric_mse_heart_oracle` → `metric_recov_frac_heart`, the primary ship-decision metric (docs/38) — so a
missed call site there would quietly corrupt the number we make decisions on. (It is also used by the
gather aux loss at `loss.py:196`, currently off: `gather_weight: 0.0`.)

### 6.3 Why `Z_HALF_MM = 90`, not 72

`ZIndexEmbedder` (`aggregator.py:33-42`) feeds `[z, sin(2^i·π·z), cos(2^i·π·z)]` for `i = 0,1,2`. The
sinusoids have **period 2**, so any two depths 2.0 apart in `z_norm` collide on 6 of the 7 channels:

```
E4  ZIndexEmbedder input features, num_freqs=3
  z=+1.167 vs z=-0.833   max|d sinusoidal| = 8.88e-16   |d linear| = 2.000
  z=+1.050 vs z=-0.950   max|d sinusoidal| = 7.77e-16   |d linear| = 2.000
  z=+0.933 vs z=-0.933   max|d sinusoidal| = 1.49e+00   |d linear| = 1.866
```

Within-subject `z_norm` range = `span / Z_HALF_MM`. The tallest stack in the pool has span 168 mm:

- `Z_HALF_MM = 72` → range **2.33 > 2** → the most basal plane and a mid-apical plane of the **same
  subject** get bit-identical sinusoidal features, distinguished only by the raw linear channel.
- `Z_HALF_MM = 90` → range **1.87 < 2** → aliasing is impossible.

Costs of 90 over 72, honestly: (a) `z_norm` values are compressed, so the sinusoids vary slightly less
across a stack — a typical 120 mm stack still gets ~2.7 cycles of the top frequency (`sin(4πz)`,
period 0.5), so this is not a problem; (b) `loss_pos_tv` (`loss.py:165-168`) takes `.abs().mean()` over
all three coordinate channels equally, so scaling z by 66/90 = 0.73 reduces through-plane smoothness
penalty ~27% relative to in-plane — real but mild, and `tv_weight` is a knob; (c) Δz values are ~27%
smaller numerically, which is irrelevant (unbounded linear head, `z_scale` scales the gradient back,
Adam is per-parameter scale-invariant).

Anything in **90–120** works. 90 is tight to the measured 84 mm max half-span and preserves the most
positional resolution. Raising it later invalidates whatever the model learned about scale, so if much
taller stacks are expected, choose higher now. Add a **fail-loud** guard:

```python
Z_HALF_MM = 90.0   # > max measured half-span (84 mm). Must satisfy span/Z_HALF_MM < 2 or
                   # ZIndexEmbedder's period-2 sinusoids alias within one subject.
assert abs(z_norm).max() <= 1.0, f"half-span exceeds Z_HALF_MM={Z_HALF_MM}"
```

If it fires, raise `Z_HALF_MM` and retrain — **do not** crop the stack.

### 6.4 What does NOT need to change

- **`splat_to_volume`'s bounds test** is on voxel indices (`splat.py:48-53`), not on ±1, so `|z_norm| > 1`
  was never discarded there. (`sample_volume` *does* clip via `padding_mode="zeros"`, but with
  `Z_HALF_MM = 90` nothing reaches ±1 anyway.)
- **The network never sees `D`.** The only model-side use of `grid_shape` is `vggt.py:113`, inside
  `if self.refiner is not None` — the refiner is off by default. patch_embed, the 24×24 attention
  blocks, `z_embedder`, `camera_token` and the DPT head all operate on `S × 37²` tokens and emit a
  per-pixel Δ at 518×518. Nothing has a `D`-dependent dimension.
- **The head is unbounded.** `point_activation = "linear"` when `train_on_residual_dvf`
  (`vggt.py:29` → `head_act.py:49-50`), and nothing clamps `world_points`.
- **GPU aug is `D`-safe** — `translate_range=(0.0, X, X)` freezes the D axis in all three tiers and
  rotation is in-plane only, so no parameter is in through-plane voxel units.
- **Cardiac phase is not a model input.** `t_indices` / `target_t_indices` are accepted by
  `Aggregator.forward` and dropped (`aggregator.py:220`, gated on `use_z` / `use_reference_token`).
  This is why the docs/17 and docs/39 hazard — CMRx's `target_t = k/12` not being the same cardiac
  state as ACDC's — **does not apply to training**. Temporal anchoring only affects filmstrips,
  per-phase logging buckets, and `ef_val_sweep`.

## 7. Temporal handling

Keep **T = 12**, ED-anchored, **nearest native frame** (no temporal interpolation — `V_gt` is the
supervision target and blended targets teach blur). ACDC gives ED/ES in `Info.cfg` (1-based); M&Ms in
its CSV (0-based).

T could in principle be native, since `t_target` is already derived from the data
(`phases_splat.shape[0]`, `mri_dataset.py:301`) and no phase index reaches the model. The constraints
are mechanical, not semantic: (a) the `phases` `(T, D, H, W)` tensor rides in every batch for GPU aug
and `compute_motion_mask` — at native T = 35 that is ~3× the per-iteration transfer; (b)
`get_canonical_transforms` builds a **fixed 12-key list** and `ConcatItemsd`s it, shared across all
subjects by `PersistentDataset`.

If native T is ever wanted, the clean route is to **load the 4D file with T as the channel dim**
(`LoadImaged` on one key + `EnsureChannelFirstd(channel_dim=3)` → `(T, X, Y, Z)`; `Orientationd` /
`Spacingd` / `ResizeWithPadOrCropd` then treat T as channels). That is *simpler* than the current
12-key `ConcatItemsd` and makes T variable for free. CMRx already ships `4d_recon.nii.gz`. Door left
open, not taken now.

## 8. Implementation checklist

### 8.1 Training path

**DONE — all rows below implemented and verified (§12).** Table kept as the original plan record;
see §12 for what actually happened, including two bugs this table didn't anticipate.

| file | change |
|---|---|
| `preprocess.py:34,39,228-234` | `Spacingd(pixdim=(1.4, 1.4, 0.0))`; `ResizeWithPadOrCropd(spatial_size=(256,256,-1))`; store `dz` and `D` in the cached dict; bump `cache_signature()` |
| `mri_dataset.py:148,302` | drop the `gt_grid_shape` assert |
| `mri_dataset.py:472` | `z_norm = z_mm / Z_HALF_MM` for `scanner_coords` **and** `z_indices`; add the `|z_norm| ≤ 1` assert |
| `mri_dataset.py:167` | `len_train = max(1000, len(self.subjects))` |
| `loss.py:110,142,391-398` | `grid_shape = tuple(batch["gt_target_volume"].shape[-3:])`; thread `z_scale` into both splat calls and `sample_volume` |
| `splat.py:35,143` | `pz = pos_z * z_scale + (D-1)*0.5`; `z_scale` **required** on `splat_to_volume` and `sample_volume` |
| `trainer_viz.py:45,250,252` | `grid_shape` from the batch; **`:250` uses the plane count as the phase count** — read T from the batch |
| `respiratory.py:37,132` | `SPACING_MM` / `N_CANON_PLANES` become per-sample (`dz`, `D` from the batch) — see §8.1a. **Only place a wrong value is silent.** Write a test |
| `mri_volume.yaml` | `continuous_z: false`; `gt_grid_shape` becomes advisory/unused |

### 8.1a `respiratory.py` in detail — the only silent-failure path

The mm→grid conversion is `respiratory.py:287-289`:

```python
def _norm_delta(d_mm, spacing_mm, size):
    return (d_mm / spacing_mm) * (2.0 / (size - 1))
#          '-- mm -> voxels --'   '- voxels -> normalized -'
```

`size` is **already** read from the tensor (`B, T, D, H, W = phases.shape`), so that half is correct
under variable `D`. The broken part is `spacing[0]` — hardcoded `SPACING_MM = (12.0, 1.4, 1.4)` at
line 37 — and `gpu_aug.py:357` calls `extract_slices_with_respiratory_vec(...)` **without** passing
`spacing`, so the module default is what runs. In-plane is unaffected (always 1.4 mm).

Error magnitude: a 10 mm subject gets `d/12` voxels instead of `d/10` (**17% too small**); an 8 mm
subject `d/12` vs `d/8` (**33% too small**).

**MEASURED (2026-07-31)**, by shifting a delta through the real `reslice_volume_vec` and reading
off where it lands. Requested displacement 20 mm for every subject:

| subject `dz` | OLD voxels | OLD actual mm | NEW voxels | NEW actual mm |
|---|---|---|---|---|
| 12.0 mm | 1.67 | 20.0 (0%) | 1.67 | 20.0 (0%) |
| 10.0 mm | 1.67 | **16.7 (−17%)** | 2.00 | 20.0 (0%) |
| 9.6 mm | 1.67 | **16.0 (−20%)** | 2.08 | 20.0 (0%) |
| 8.0 mm | 1.67 | **13.3 (−33%)** | 2.50 | 20.0 (0%) |
| 5.0 mm | 1.67 | **8.3 (−58%)** | 4.00 | 20.0 (0%) |

The framing that matters: **the OLD code was the inconsistent one.** It held the *voxel* shift
constant across subjects, so the *physical* motion silently scaled with pitch. The fix makes the
**physical mm** constant and lets the voxel count vary — the correct invariant, since a voxel is a
different distance in each subject. CMRxRecon2024 is uniformly 12 mm, which is exactly why this
was invisible until the pool added ACDC (5–10 mm), M&Ms (8–10 mm) and CMRx2025 (10 mm).

**Why this one matters more than its size suggests.** `gpu_aug` writes the *requested* mm displacement
into the batch as `resp_disp_mm`, and `metric_resp_epe_dz_mm` / `metric_resp_slope_dz` (docs/38) grade
the model's predicted Δz against it. If the applied shift is 17–33% smaller than the recorded label,
the model correctly learns to undo the *applied* shift while the metric scores it against a label that
was never applied — so **the breathing slope reads ~0.67–0.83 instead of ~1.0 and looks like model
under-correction.** Silent, systematic, and keyed to slice pitch, hence correlated with dataset, hence
easily misread as a cohort effect. Same failure shape as the "breathing at oracle ceiling" reading that
had to be retracted (memory: `project_breathing_z_at_oracle_ceiling`).

**Second bug in the same file.** `N_CANON_PLANES = 12` (line 132) is the burst-grouping key:

```python
P = N_CANON_PLANES
gid = group_ids.clamp(0, P - 1).long()      # (B,S) z-plane per slot
r = torch.gather(rand((B, P)), 1, gid)      # breath phase shared within a plane
```

With `D` up to 18, every plane ≥ 12 is **clamped to 11** and inherits plane 11's breath phase instead
of drawing its own — the tall stacks lose independent-breath-per-slice realism exactly where they have
the most slices. Fix: `P = D` from the tensor. (`CANON_D = 12` at line 36 is dead — defined, never
read.)

**Fix**: thread the subject's `dz` from the batch into `gpu_aug.py:357` as `spacing=(dz, 1.4, 1.4)`,
set `P = D`, and **drop the `spacing=SPACING_MM` default** on `extract_slices_with_respiratory_vec` /
`reslice_volume_vec` — same argument as `z_scale`: a caller that forgets should raise, not silently run
at 12 mm.

**Test**: apply a known mm displacement at two different `dz`, assert the resulting voxel shift equals
`d_mm/dz`. Fault-inject (hardcode 12.0 back) and confirm it fails.

### 8.2 Data work

- **CMRx 2023 / 2024 / 2025** — nothing. Layout already correct.
- **ACDC + M&Ms converter** → `<PREFIXED_ID>/sax/3d_recon/sax_frame_00..11.nii.gz`. Three jobs:
  split the 4D file; pick 12 ED-anchored nearest native frames; **re-frame the affine into acquisition
  space** (compute the slice axis, flip so index increases toward +S, choose/flip the in-plane axes to
  best match L and P, stamp a synthetic axis-aligned LPS affine). Run the same rule on **all** sources
  and assert `|slice_dir·S| > 0.9` afterwards — the assert is what catches the next M&Ms. Render ~20
  M&Ms subjects and confirm LV donuts + consistent RV position before trusting all 345.
- ~~**Pooled split file** + manifest CSV (§2.1).~~ **DONE, see §13.**

### 8.3 Surface outside the training loop (must not be forgotten)

These all construct `z_indices` / `scanner_coords` or hardcode the `(256,256,12)` / `(1.4,1.4,12.0)`
grid, and will silently produce wrong geometry under the new convention:

- `inference/adapters/base.py`, `inference/run_cmrxrecon.py` (`CANON_SPACING` at line 35),
  `inference/run_gated_ood.py`
- `evaluation/engine/run_vggt.py`, `evaluation/engine/assemble_and_gif.py`,
  `evaluation/engine/build_inputs/{geom,cmrxrecon,acdc,ocmr,miitt}.py`
  — ⚠️ per the standing rule, **do not add anything to `evaluation/` unprompted**; these are edits to
  existing files and should be confirmed with the user
- `tests/{conftest,test_preprocess,test_mri_dataset,test_canonical_invariants,test_gpu_aug,
  test_respiratory_e2e,test_trainer_diagnostics,test_refiner}.py`
- ~20 `tools/*.py` one-off scripts — will break; fix on demand, not preemptively

### 8.4 Also affected, needs regeneration

- ~~**`heart_roi_canonical.nii.gz`** — currently on the old `(256,256,12)` grid, must be
  regenerated.~~ **DONE (verified 2026-07-31).** All **1343/1343** subjects now have both
  `heart_roi_canonical.nii.gz` `(256,256,D)` and `heart_seg_canonical.nii.gz` `(256,256,D,12)` on
  each subject's **own native-z grid** with that subject's own `dz` (spot-checked: ACDC 10.0,
  M&Ms 9.52/8.8, CMRx 12.0). The warn-and-skip guard described in §12 therefore no longer fires,
  and **`metric_psnr_3d_heartseg` / `metric_mae_3d_heartseg` are live, not inert.**
  Consequence for §10a: these files are **live data that must be flipped alongside the images**,
  not stale artifacts to be regenerated afterwards.
- ~~**`cardiac_phase.csv`** (`scratch/data/whs/`) — `_build_val_targets` raises `KeyError` for any val
  subject missing from it, and it is CMRx-only and v1-era. ACDC/M&Ms give ED/ES free from metadata;
  CMRx needs regeneration.~~ **DONE (verified 2026-07-31).** Concatenated the 1343 per-unit
  `scratch/data/whs/rows/*.csv` into a headered `whs_manifest.csv` (only needed for the `seg_flag`
  join column; `compute_cardiac_phase.py` reads `heart_seg.nii.gz` directly for ED/ES/EF, so it
  isn't a hard dependency) and ran `tools/nnunet_mnms_eval/compute_cardiac_phase.py` →
  `scratch/data/whs/cardiac_phase.csv`, 1514 gated subjects, all CMRx `ES` now populated (was blank
  for 848/848). Verified live: constructing the real val `MRIDataset` via
  `instantiate(cfg.data.val, ...)` off `mri_volume` (which per §14 now sets `ef_val_sweep: true`
  and points `cardiac_phase_csv` at this file) succeeds — it would previously raise
  `FileNotFoundError`. This clears §14's "blocker for an actual run."
- Monai cache — `cache_signature()` bump routes to a fresh subdir automatically. `/tmp` is cleared on
  GPU nodes, so no manual purge needed.

## 9. Verification gate

**IMPLEMENTED as `tools/gate_native_z_identity.py` (§12).** Before any long run: sweep ~30 pooled
subjects, splat with **Δ = 0**, assert PSNR vs `V_gt` is numerically exact for each. One pass
catches the §5.3 misalignment trap, a mis-threaded `z_scale`, and a botched M&Ms affine re-frame.

Per the fault-injection lesson (memory: `feedback_fault_inject_verifiers`), **prove the gate fires**:
run it once with `z_scale` deliberately wrong and confirm it fails. (Implemented exactly this way —
see §12 for why the gate does NOT route through the real 518×518 input pipeline, unlike the
`probe_zgrid_alignment.py::e3_native_grid` template originally proposed here.)

## 10. Open items and known risks

- **Unmeasured**: whether Option 2 actually trains better than Option 1. What *is* measured is that
  Option 2 has no built-in ceiling, and that the alternatives do. Treat the training benefit as a
  hypothesis.
- **`respiratory.py` is the silent-failure risk** (§8.1a). Everything else either crashes or is
  visible; a wrong `dz` there applies wrong-scale breathing shifts *and* mis-grades the breathing
  diagnostics against a label that was never applied.
- **Metric comparability**: averaging PSNR across subjects now mixes raster resolutions, and with
  `D` = slice count there is no z zero-padding, so `anatomy_bbox`'s z range becomes the whole volume
  and `metric_psnr_3d_full` is no longer inflated by padded z planes. `_full` and `_bbox` will differ
  from historical values for reasons unrelated to model quality. **Fresh series.** Bucket val metrics
  by source and by pitch.
- **`torch.compile`**: `cuda.compile_attention_blocks` already sees a varying `S` under
  `one_frame_per_slice`; `D` never reaches compiled code. No new exposure expected, but watch for
  recompilation churn.
- **Cross-year CMRx duplicates were never empirically verified** (`CMRxRecon2024/DUPLICATES.txt:65`);
  the basis is the organizers' non-overlap guarantee. User has judged this a non-issue. Note also that
  **CMRxRecon-300 is the same 300 volunteers as CMRxRecon2023** — do not add it.
- ~~**`enable_refiner`** is the one component genuinely `D`-dependent...~~ **RESOLVED by deletion,
  see §12.** The native-z refactor made the refiner's forward-pass branch hard-require
  `batch["gt_target_volume"]` to derive `grid_shape` (previously a fixed model attribute) — which
  would have broken ground-truth-free OOD inference (`inference/run_rtfb.py`) if the refiner were
  ever enabled. Since `enable_refiner` was `false` in every config and no checkpoint had ever
  trained it, the user chose to delete the whole subsystem rather than redesign its grid_shape
  sourcing. `vggt/models/refiner.py` no longer exists.
- **The `continuous_z` → whole-field-Δz-relocation connection is a hypothesis** (§5.4), not verified.

## 10a. Slice-order (base/apex) standardised to APEX-AT-z0 — **APPLIED ON DISK 2026-07-31**

> **Status: DONE.** 893 of 1343 subjects were base-first and have been flipped on disk
> (`np.flip(axis=2)`) by `tools/fix_slice_order.py`. Every subject now stores **apex at z0** with
> the index increasing toward the base (= superior), so `respiratory.py`'s one-sided breathing
> displacement moves the heart **inferiorly on inspiration** for the whole cohort — physiological,
> and uniform instead of the previous 66/34 split in opposite directions.
>
> Affines were **not** touched (they already declared `+z = Superior`; flipping the array is what
> makes them honest). `cache_signature()` was bumped `bab7860607` → `f937056bb3` so monai cannot
> serve pre-flip volumes from unchanged paths. Reversible via
> `scratch/data/_provenance/slice_order_fix.json`. Verification record in §10b.

The sections below are the original investigation, kept as the reasoning record.

**Status 2026-07-31 (superseded): measured but UNDER-POWERED. Nothing has been changed. Do not act
on this until the numbers below are confirmed at full cohort size.**

Measured by `tools/probe_slice_order.py` — fits labeled-area against z on the ED frame of
`heart_seg.nii.gz` (the LV tapers to a point at the apex and is widest near the base, so a positive
slope ⇒ area grows with z ⇒ apex stored first). `--csv` emits per-subject rows, which is what a fix
must be driven from:

| source | n | base-first (z0 = base) | apex-first (z0 = apex) |
|---|---|---|---|
| CMRx (all 3 years) | 60 | **59** | 1 |
| ACDC | 3 | 3 | 0 |
| M&Ms | 3 | 0 | **3** |

⚠️ **ACDC and M&Ms are n=3** — only the smoke-test subjects were segmented at the time. The seg job
reaches those blocks ~5–6 h in; re-run the measurement then. CMRx is **59/60, not unanimous**, so any
fix must be driven **per subject from the measured slope**, never by a blanket per-source rule.

#### Re-measured at FULL n (2026-07-31) — CMRx is genuinely mixed, and it is a 2025/scanner effect

Segmentation is now complete for all 1343 subjects, so the table above is superseded:

| source | n | base-first | apex-first | consistency |
|---|---|---|---|---|
| CMRx (all years) | 846 | 738 | **108** | 87.2% |
| ACDC | 150 | **150** | 0 | 100% |
| M&Ms | 345 | 3 † | **342** | 99.1% |

† Of these 3, **2 are single-feature detector error** (the 3-feature detector calls them apex-first,
agreeing with the metadata) and **1 (MNMs_T2Z1Z9) is probably metadata error** — see the validation
subsection. Treat M&Ms as **344 apex-first + 1 unresolved**, not as 3 real base-first subjects.

The n=60 sample badly under-represented the disagreement (it implied 98% CMRx consistency; the
truth is 87%). Broken out by year and scanner, the 108 are **not** scattered noise:

| | base-first | apex-first |
|---|---|---|
| CMRx2023 | 193 | 2 |
| CMRx2024 | 292 | 1 |
| CMRx2025 Siemens | 163 | 5 |
| CMRx2025 UIH | **91** | **87** |
| CMRx2025 Philips | 0 | **12** |

**CMRx2023/2024 are effectively unanimous base-first** (3 outliers in 488 ≈ the detector's noise
floor). All the real disagreement is **CMRxRecon2025**, where UIH is a near-perfect 50/50 split
*within one vendor* and all 12 Philips subjects are apex-first. So the ordering is a per-scan
property in 2025, not a per-source or even per-vendor constant — which settles the open question of
whether a blanket per-source flip would do: **it would not.** Same conclusion as docs/56's odd-Z
slice roll, arrived at from the opposite direction.

Provenance: `tools/probe_slice_order.py --per-source 0 --csv` (single feature, full cohort) and
`tools/render_slice_order_check.py --csv` (three features + agreement flag). The two agree to
within 1 subject per source. Per-subject rows:
`result/slice_order_check/slice_order_full.csv`, `.../features_all.csv`.

#### M&Ms doubles as a labelled validation set for the detector — error rate ~0.3–0.9%

All three columns above are from the **anatomy detector**, not from metadata. But for M&Ms the
metadata gives an independent label: `convert_to_sax_layout.py` flips axis 2 so the index increases
toward **+S**, and the base is superior, so the converter guarantees **apex-first for all 345**.
M&Ms is the only source where this cross-check exists (CMRx's affines are the SimpleITK default,
ACDC's are `sform=qform=0`) — which makes it the only available ground truth for the detector.

| detector | agrees with M&Ms metadata |
|---|---|
| `probe_slice_order.py` (f1 only) | 342/345 = **99.1%** |
| `render_slice_order_check.py` (f1+f2+f3) | 344/345 = **99.7%** |

**This is what makes the CMRx number interpretable: 108/846 = 12.8% is 15–40× the detector's
measured noise floor, so the CMRx mixture is real, not detector error.**

The 3 disagreements are instructive rather than alarming — inspect them with
`render_slice_order_check.py --subjects MNMs_E3I4V1,MNMs_L1Q9V8,MNMs_T2Z1Z9`
(→ `result/slice_order_check/mnms_outliers.png`):

| subject | \|S\| | f1 | f2 | f3 |
|---|---|---|---|---|
| MNMs_E3I4V1 | **0.088** | −0.008 | +0.014 | +0.038 |
| MNMs_L1Q9V8 | 0.584 | −0.007 | +0.067 | +0.041 |
| MNMs_T2Z1Z9 | **0.087** | −0.133 | −0.085 | +0.001 |

**E3I4V1 and L1Q9V8 are single-feature (f1) errors**: their features contradict each other at
|f| ≈ 0.01 (a clean subject reads 0.10–0.13) and both are **truncated stacks** — the cavity curve
rises then falls off a cliff instead of tapering, so there is no cone to measure. The 3-feature
detector calls both apex-first, agreeing with the metadata; only `probe_slice_order.py`'s f1 gets
them wrong. This is what the 99.1% → 99.7% improvement in the table above is made of.

**MNMs_T2Z1Z9 is the opposite case — the metadata is the suspect party.** The converter decides the
axis-2 flip from the **S component alone**, and here the recorded slice normal is
`(−0.859 L, +0.504 A, +0.087 S)`. Left + anterior point toward the **apex**, overruling the 0.087 S
by roughly 10:1 — the LV long axis of this scan runs base→apex in the inferior-anterior-**leftward**
direction, i.e. the stack is nearly perpendicular to SI. Detector, taper curves and the rendered
side view all read base-first. Compare the other two, whose R/A/S components all agree
(`+0.721 R, −0.687 P, +0.088 S` and `+0.395 R, −0.709 P, +0.584 S` — right + posterior + superior
all point toward the base). **Status: unresolved, and it is the anatomy that should win.**

Scope of the concern: **21 M&Ms subjects have |S| < 0.15**, where the converter's rule rests on a
tiny projection — but the detector agrees with the metadata on **20 of the 21**. So this is a
one-subject discrepancy, not a systemic conversion bug. It does mean the converter's `+S` rule is
the wrong *criterion* in principle for near-perpendicular stacks (the right one would project onto
the full apex→base direction, not just S); it is simply almost always harmless.

⚠️ **The M&Ms affine is real, but it does not pin the direction uniformly.** A SAX slice normal *is*
the LV long axis, tilted well off S: measured across all 345, |S| has median **0.402** and min
**0.069**, with **9 subjects below 0.1**. The converter used the *sign* of S, which is reliable at
healthy |S| but is deciding off an 0.08 projection for those 9. So "M&Ms is the only source whose
ordering came from real scanner geometry" is correct; "so its ordering is certain" is not.

#### Detector v2: the decision rule was SELECTED BY MEASUREMENT, and the obvious ideas lost

Before flipping anything on disk (§10a "the fix"), the aggregation rule was tuned against the M&Ms
ground truth. **Every intuitively-appealing improvement measured worse**, so this is recorded in
detail to stop someone re-introducing them:

| aggregation of (f1, f2, f3) | M&Ms errors |
|---|---|
| **f1+f2, must agree (v2, adopted)** | **1** (the contested T2Z1Z9) |
| 3-feature majority vote (v1) | 1 |
| robust-scaled mean | 6 |
| capped scaled mean (c=1, c=1.5) | 6 |
| median of scaled features | 1, but 498/1341 undetermined |

And on the 22 M&Ms subjects where features disagree — i.e. *exactly* where the tie-break matters:

| tie-break | correct |
|---|---|
| f1+f2 / majority | **21/22 (95%)** |
| f1 alone | 19/22 (86%) |
| max-\|z\| feature ("trust the strongest signal") | 13/22 (59%) |
| **f3 alone** | **3/22 (14%) — ANTI-informative** |

Two things fall out. **(a) `f3` (cavity fraction) is worse than a coin flip on hard cases** and is
therefore computed and reported but **never voted**. **(b) The "obvious" fix of letting a large-|f|
feature outvote two weak ones is wrong** — max-|z| tie-breaking is 59%, barely above chance. The
v1 rule's own tuned threshold independently re-derived *unanimity* (τ\*=0.35 on a ±1/±⅓ vote score,
i.e. reject the 2-1 splits), confirming that "features disagree ⇒ don't call it" was already the
right confidence signal.

Threshold selection used a **seeded half-split of M&Ms** (tune on 172, report on 173) so the quoted
accuracy is not fit on its own test set. Selecting *which* aggregator to adopt still used all of
M&Ms, so v2's 342/343 carries ~1 bit of selection — it is a tight bound, not an independent estimate.

**v2 rule:** `call = sign(f1) if sign(f1)==sign(f2) else UNDETERMINED`.

| source | determined | apex-first | base-first | undetermined |
|---|---|---|---|---|
| CMRx | 839 | 104 | 735 | 9 |
| ACDC | 149 | 0 | **149** | 1 |
| M&Ms | 343 | 342 | 1 † | 2 |
| **total** | **1331** | **446** | **885** | **12** |

(10 of the 12 undetermined are f1/f2 disagreements; 2 — `CMRx24_Test_P044`,
`CMRx25_R1val_Center006_Siemens_30T_Prisma_P010` — have fewer than 4 labeled planes and cannot
support a slope fit at all.) † still `MNMs_T2Z1Z9`.

Changes vs v1: **ACDC becomes unanimously base-first** (its single apex-first call was a v1 artifact),
CMRx apex-first tightens 107 → 104, and the adjudication set shrinks from 94 to **12**. Per-subject
rows: `result/slice_order_check/features_v2.csv`; the 11 renderable ones are in
`result/slice_order_check/adjudication_set.png`.

#### The M&Ms header, read properly, agrees with the detector 343/343 — and resolves T2Z1Z9

The converter's `+S` rule is **the wrong criterion**, and correcting it dissolves the only
detector-vs-metadata conflict. The apex→base axis is not "superior"; it is superior **and**
posterior **and** rightward. Averaging the converted `+z` direction over the 342 apex-first M&Ms
subjects recovers it empirically:

```
apex -> base  =  ( R +0.611,  A -0.677,  S +0.410 )      i.e. base is right, posterior, superior
```

which is textbook LV long-axis orientation, derived here from data. Projecting each subject's
converted `+z` onto that axis and calling apex-first iff the projection is positive:

| | |
|---|---|
| geometry vs detector, M&Ms | **343 agree, 0 disagree** |
| projection magnitudes | median 0.984, 1st pct 0.859, min 0.255 — strongly bimodal, 1 subject in (−0.3, 0.3) |

**`MNMs_T2Z1Z9` projects at −0.831 → base-first, agreeing with the detector.** Its `S` component is
+0.087 while its L and A components (which point toward the apex) dominate ~10:1, so the S-only rule
flipped it the wrong way. **Nothing is wrong with the subject** — it is a genuine base-first stack
and should be flipped with the rest. The earlier "hold it out" recommendation is withdrawn.

The same check resolves both undetermined M&Ms subjects: `MNMs_E3I4V1` (+0.942) and `MNMs_L1Q9V8`
(+0.961) are **apex-first**, i.e. already in the target convention, no flip.

⚠️ Circularity caveat: the axis `b` was averaged over subjects the detector called apex-first, so
343/343 is not a fully independent validation. But **T2Z1Z9 was excluded from that average** (it is
the base-first one), making it a genuine held-out test, and it lands at −0.831 — far from the
decision boundary. `b` also matches textbook anatomy independently.

#### STRONGEST validation: M&Ms ships ground-truth segmentations — detector agrees 320/320

Found during the `/prove-it` audit (2026-07-31) and **supersedes both the metadata check and the
geometry projection**, because it depends on neither the detector nor the derived axis:
**M&Ms ships its own GT masks** at `scratch/data/MNMs/MNMs1/*/<ID>/<ID>_sa_gt.nii.gz`, covering
**320 of 345** subjects. Nothing in this investigation had used them.

Method: derive a base/apex marker from the GT LV-cavity area profile, then **validate the marker**
on the 163 subjects with `|S| > 0.4` — where the converter's S-only rule is geometrically
unambiguous, so the answer is known independently. Result **163/163 correct**. Applying the
validated marker to all 320:

| | |
|---|---|
| detector/decision file vs GT-derived label | **320/320 agree** |
| …including the small-\|S\| (<0.15) subjects, where the converter's rule is unreliable | **19/19 agree** |
| converter output that was actually base-first | **exactly 1 subject: `MNMs_T2Z1Z9`** |

So the flip file flips **precisely the one subject the ground truth says needs it**. `T2Z1Z9` is
**CORRECTLY FLIPPED** — verified from `T2Z1Z9_sa_gt.nii.gz` directly: at ED the RV area collapses
monotonically toward high z (4752 → 149) and the LV cavity shrinks, i.e. the *source* stack is
unambiguously base-first, and `flips[2]=False` preserved that order.

⚠️ **Correction to the rule-selection ground truth.** The claim used to select the decision rule —
"the converter guarantees apex-first for all 345 M&Ms subjects" — is **unsound**. It fails for
`T2Z1Z9` because `plan_reframe()` sets `flips[2] = (slice_dir_ras · S) < 0`, the **sign of the S
component alone with no magnitude guard**, and that subject's S is +0.087. The practical damage is
bounded and in the *favourable* direction: the GT was wrong for 1 of 320, and the error direction
**penalised the adopted rule for its one correct call**, so the quoted selection scores
(f1+f2 21/22 etc.) are a slight **under**-estimate, not an over-estimate. If these rules are ever
re-scored, use the GT-mask labels (320/345 coverage) rather than the converter guarantee.

⚠️ **This check is available for M&Ms only.** ACDC's `convert_meta.json` records
`slice_dir_ras = (0, 0, 1)` and `|S| = 1.0` for **all 150** subjects — its `sform=qform=0` header
makes nibabel fall back to an axis-aligned default, so there is no real geometry to project. CMRx
has the SimpleITK default for the same reason. Anatomy remains the only evidence for those two.

#### Final adjudication and flip list

The 10 renderable undetermined subjects were adjudicated by eye from
`result/slice_order_check/adjudication_proposed.png` (side views + taper curves + the two extreme
labeled short-axis planes; proposals drawn in orange, detector calls in yellow). Outcome — user
confirmed all proposals except `..._umr790_P017`, which was corrected to apex-first:

| subject | call | decided by |
|---|---|---|
| CMRx23_Val_P005 | base-first | visual |
| CMRx25_R1test_Center006_UIH_30T_umr790_P017 | **apex-first** | visual (corrected my proposal) |
| CMRx25_R2test_Center006_Siemens_30T_Prisma_P013 | base-first | visual |
| CMRx25_R2val_Center004_Siemens_15T_Aera_P045 | base-first | visual |
| CMRx25_train_Center002_UIH_30T_umr880_P001 | base-first | visual |
| CMRx25_train_Center006_Siemens_30T_Prisma_P002 | base-first | visual |
| CMRx25_train_Center006_UIH_30T_umr790_P001 | apex-first | visual |
| ACDC_patient047 | base-first | visual |
| MNMs_E3I4V1 | apex-first | geometry, proj +0.942 |
| MNMs_L1Q9V8 | apex-first | geometry, proj +0.961 |

**Final per-subject decisions: `result/slice_order_check/slice_order_decisions.csv`** (1343 rows,
columns `subject, source, order, decided_by, flip, f1, f2, f3`) — this is the driver for the
on-disk fix, so the flip is reproducible and auditable rather than resting on a transcript.

The last two — `CMRx24_Test_P044` (heart segmented on only 3 of 6 planes) and
`CMRx25_R1val_Center006_Siemens_30T_Prisma_P010` (2 of 6) — are **segmentation failures, not
ordering puzzles**, and were adjudicated from the raw images across all `D` planes
(`--min-planes 2 --all-planes`, → `result/slice_order_check/cmrx_undetermined.png`). Both
confirmed **base-first**. Note for the record: the agent's own read of `P044` was apex-first (its
cavity grows toward `z=5`); the user's base-first call was taken after inspecting the images and
is what the decision file records.

**FINAL:**

| source | flip (base-first) | keep (apex-first) | undetermined |
|---|---|---|---|
| CMRx | 742 | 106 | 0 |
| ACDC | **150** | 0 | 0 |
| M&Ms | 1 (T2Z1Z9) | 344 | 0 |
| **total** | **893** | **450** | **0** |

ACDC is **150/150 base-first** — fully uniform. **Every one of the 1343 subjects has a decision;
nothing is left undetermined.** `MNMs_T2Z1Z9` is flipped along with the rest: it is not an
anomalous subject but a converter mistake (see the geometry subsection above), independently
confirmed by its all-planes strip running visibly opposite to a normal M&Ms subject's
(`result/slice_order_check/t2z1z9_vs_normal.png`).

⚠️ **Caveat on the 108**: 58 CMRx subjects have *disagreeing* features (the size-based f1/f2 vs the
shape-based f3), and truncated stacks that never reach the apex make the taper genuinely ambiguous.
The per-scanner clustering above is strong evidence the bulk of the 108 are real, but the
individual calls near |slope| ≈ 0 are not yet trustworthy enough to drive an on-disk flip. A
confidence threshold + eyeball pass on the low-|slope| tail is required first.
**[RESOLVED — this was done: detector v2 (`sign(f1)==sign(f2)`) plus human adjudication of the
12-subject ambiguous tail. See §10b. Do not read this paragraph as a live blocker.]**

#### The breathing direction is now MEASURED, not derived

`tools/render_respiratory_direction.py` drives the real `reslice_volume_vec` over a full Lujan
cycle and renders the long-axis side view → `result/respiratory_direction/*.gif`. Confirmed
visually: as `d` grows from 0 (end-expiration) to `A` (peak inspiration), **content moves toward
lower z**, with `padding_mode="zeros"` opening a black band at the high-z edge. This matches the
sign traced in the code (`z_coord = z_base + dz_norm`, so output plane `z_i` samples input `z_i+d`)
and the fact that `A·sin^{2n}(πr) ≥ 0` always. So the §10a derivation was correct; it is no longer
a derivation.

Consequence, per subject: **base-first ⇒ the simulated heart moves superiorly on inspiration
(backwards); apex-first ⇒ inferiorly (physiological).** At the measured full-n counts that means
**891 of 1341 subjects (66%) currently breathe the wrong way** *(pre-v2 detector figure; the
final applied count is **893 of 1343** — §10b)*, and — worse than a uniform error —
the cohort contains *both* directions with nothing telling the model which it is looking at.

**Mechanism (understood, not speculative).** `tools/convert_to_sax_layout.py` flips array axis 2 so
the index increases toward **+S (superior)**. The base is superior, so a stack honouring that lands
**apex at z0**. M&Ms is the only source with a real `sform` (ACDC has `sform=qform=0`, CMRxRecon's
recon writes the SimpleITK default), so M&Ms is the only one whose ordering was derived from actual
scanner geometry — and that is precisely what made it differ from the other two.

**So by the convention our own stamped LPS affine declares, M&Ms is correct and CMRx/ACDC are
reversed.** Their affines never encoded a real z-direction, so nobody had checked.

### Why this is worth fixing rather than tolerating

Like chirality, a reversed stack is **not** a corrupt training sample — inputs and `V_gt` come from
the same volume and are reversed together, so the task stays self-consistent and the identity splat
is still exact. It is distribution shift, not corruption. But there is a second-order consequence
that chirality does not have:

`respiratory.py`'s Lujan waveform is **one-sided** — `d(r) = A·sin^{2n}(πr)`, `d=0` at end-expiration
(rest) rising to `d=A` at peak inspiration — and is applied along the array's D axis. Tracing the
reslice: `z_coord = z_base + dz_norm` and `grid_sample` samples the input there, so output plane
`z_i` shows content from `z_i + dz`; with `dz > 0` the anatomy therefore appears to move toward
**lower z**. Physiologically the diaphragm descends on inspiration and the heart follows it
**inferiorly** (docs/01: "heart tracks the diaphragm at ~0.6×, end-of-exhale as the rest position").
Hence:

| convention | lower z is | apparent motion at d>0 | physiological? |
|---|---|---|---|
| apex at z0 | apex = inferior | heart moves inferiorly | ✅ |
| base at z0 | base = superior | heart moves superiorly | ❌ |

⚠️ ~~**This is a derivation from the sign convention, NOT a measurement.** Confirm with
`tools/render_respiratory_examples.py` before acting on it.~~ **SUPERSEDED — it WAS measured**,
with `tools/render_respiratory_direction.py` (not `render_respiratory_examples.py`): driving the
real `reslice_volume_vec` over a Lujan cycle shows content moving toward lower z. See
"The breathing direction is now MEASURED" above and §10b. If it holds, the existing simulation is
already correct under apex-at-z0 and backwards for most CMRx+ACDC subjects *(the "998" here
assumed whole sources flip together; the measured figure is **892** CMRx+ACDC, since 106 CMRx
subjects were already apex-first — §10b)* — and no change to
`respiratory.py` is needed, only a consistent slice order.

**Note `respiratory.py` has no anatomical anchor of any kind** (verified): it operates purely in
D-index space, with no mapping from D to superior/inferior and no per-source handling. The
direction has never been tied to anatomy for *any* source. That is a pre-existing gap this finding
exposes rather than creates.

### If confirmed, the fix

Standardise on **apex at z0** (index increasing toward the base/superior). ⚠️ **The original
phrasing here — "flip CMRx and ACDC, not M&Ms" — is WRONG and was not what was done**: the
decision is PER SUBJECT, and in the end 106 CMRx subjects were left alone while 1 M&Ms subject
(`T2Z1Z9`) *was* flipped. See §10b. The apparent cost, "this invalidates the existing training series", is largely already
paid: the native-z change alters `V_gt`'s frame, the normalisation, and `z_norm` itself
(grid-relative → physical), so every learned z-embedding is already stale and PSNR is already
non-comparable. ~~The real incremental cost is re-segmenting the flipped subjects
(~15 GPU-hours).~~ **WRONG — no re-segmentation is needed at all**: a z-flip is an exact array
reversal, so flipping `heart_seg` alongside the images preserves voxel correspondence bit-exactly.
See §10b.

Prefer the on-disk route, mirroring `tools/fix_slice_roll.py` (docs/56): atomic, `--revert`-able,
provenance sidecar, driven per subject from the measured slope. It keeps the loader free of hidden
transforms and makes training, `inference/` and `evaluation/` all see one convention.

### ⚠️ Trap: do NOT "fix" this by making the stamped affine honest

`Orientationd(axcodes="LPS")` in `preprocess.py` reorders array axes **by what the affine says**.
The converter stamps `+z = superior`, which is what makes `Orientationd` a no-op and leaves the
written order intact. If someone later "corrects" the affine to declare `+z = inferior` to match a
base-first array, `Orientationd` will silently flip every one of those subjects back — reversing the
cohort with no error. **The affine must keep declaring `+z = superior`; the array order is chosen
independently.** Whenever the two disagree, the affine's z-direction is nominal, and that has been
true of CMRx and ACDC all along.

## 10b. The flip, as applied (2026-07-31) — what was done and how it was verified

`tools/fix_slice_order.py --apply`, driven per subject by
`result/slice_order_check/slice_order_decisions.csv`. **893 subjects / 15030 NIfTI files /
151 `convert_meta.json`.** ~50 min wall-clock on GPFS.

### What was written

| item | action |
|---|---|
| `sax/3d_recon/sax_frame_00..11.nii.gz` | `np.flip(axis=2)` |
| `sax/4d_recon.nii.gz` | `np.flip(axis=2)` |
| `sax/heart_seg.nii.gz`, `sax/heart_roi.nii.gz` | `np.flip(axis=2)` |
| `sax/heart_seg_canonical.nii.gz`, `sax/heart_roi_canonical.nii.gz` | `np.flip(axis=2)` |
| **NIfTI affines** | **UNCHANGED** — see the trap above |
| `sax/convert_meta.json` `reframe.flips[2]` | **toggled** (ACDC/M&Ms only, 151 files) |
| `lax/`, `lvot/` | untouched (different orientations) |

All six file types share `z == array axis 2`, so a single flip axis covers `(X,Y,Z)`,
`(X,Y,Z,T)`, `(256,256,D)` and `(256,256,D,12)` alike.

**No re-segmentation was needed** — the earlier estimate of "~15 GPU-hours to re-segment the
flipped subjects" (§10a above, now stale) was wrong: a z-flip is an exact array reversal, so
flipping `heart_seg` alongside the images preserves voxel correspondence bit-exactly.

### Verification record

Every check below was run; the two marked ⓕ were **proven to fail first**, per the standing
fault-injection rule (a check never shown to fire is not evidence).

| check | result |
|---|---|
| Preflight: canonical z-direction matches native (`preflight_zdir`) | **893 ok, 0 REVERSED, 0 unusable** |
| Flip is an exact involution (real subject, 17 files) | 17/17 exact z-reversals; **two flips restore bit-identical data** |
| ⓕ `--verify` **before** the flip | **FAILS**, exit 1, 885 base-first ⇒ the check can fire |
| `--verify` **after** the flip | **PASS**, exit 0 — apex-first **1331**, base-first **0**, undetermined 12 |
| ⓕ `--fault-inject CMRx24_Train_P197` | flipping one subject back ⇒ detector reports `base-first`; restoring ⇒ `apex-first` |
| ⓕ `verify_sax_conversion.py` ACDC | **150/150 before AND 150/150 after**; own fault-injection fires both times |
| ⓕ `verify_sax_conversion.py` M&Ms | **345/345 before AND 345/345 after**; own fault-injection fires both times |
| `render_respiratory_direction.py` | **4/4 now INFERIOR → PHYSIOLOGICAL** (2 of the 4 were BACKWARDS before) |
| `gate_native_z_identity.py` | **35/35 at exactly 120.00 dB** — the flip does not disturb the native-z splat |
| `pytest tests/` | **203 passed** |
| `ef_val_sweep` on pooled val, post-flip | 134 subjects → **268 pairs** = `limit_val_batches` |

The losslessness rows are the load-bearing ones for the metadata change: the check compares on-disk
data against *the original source re-transformed by the recorded `reframe.flips`*. It passing
**after** the flip is what proves the `convert_meta.json` `flips[2]` toggle is correct — had the
data been flipped without the record, all 151 would have failed.

`1331 = 446 + 885` exactly: every previously-base-first subject became apex-first, and the 12
undetermined stayed undetermined (a flip cannot make disagreeing features agree). 8 of those 12
were flipped anyway on adjudicated calls; they remain detector-undetermined by construction.

### Reversibility

`scratch/data/_provenance/slice_order_fix.json` records the operation, the decision file, the
per-source counts, the full 893-subject list, and the file types touched. `--revert --apply`
re-flips them (the operation is its own inverse) and renames the sidecar to `.reverted`.

⚠️ **Revert ordering**: undo THIS fix **before** docs/56's `slice_roll_fix.json`. Reverting the
odd-Z roll underneath an applied flip would corrupt the stack. Recorded in the sidecar as
`revert_order`.

⚠️ `convert_meta.json`'s `post_conversion_fixes` log records the **resulting** `flips[2]` value
rather than "applied"/"reverted", because `flip_subject` is its own inverse and is used for both
directions — so the appended entries stay self-describing through a revert.

### ⚠️ Downstream data that is now PRE-FLIP STALE (eval only — training is unaffected)

The flip changed the on-disk cohort, so anything derived from it *before* 2026-07-31 12:19 is in
the old convention. **User's decision: the eval GT and derived bundles will all be regenerated
from the current (flipped) data.** That resolves the staleness — but regeneration alone is NOT
sufficient, because three of these are **code** bugs that would simply be baked into the new GT.
Read this before rebuilding:

| item | status | needs |
|---|---|---|
| `scratch/eval/` frozen bundle (`gt/`, `clean/`, `breath/`, `mask_heart`, `heart_seg`) | pre-flip | regenerate |
| SVR baseline outputs built against it (NiftyMIC / NeSVoR / SVRTK) | pre-flip orientation | re-run or re-score |
| ~~`cmrxrecon.py:85` `assert D == N_CANON_Z`~~ | **FIXED** — assert removed, `D` is per-subject |
| ~~`cmrxrecon.py:124` `reslice_volume_vec` with no `spacing`~~ | **FIXED** — passes `spacing=(dz,1.4,1.4)` |
| ~~`cmrxrecon.py:91` `sample_resp_disp` with no `n_planes`~~ | **FIXED** — passes `n_planes=D` |
| `cmrxrecon.py` saved affine + manifest hardcoded `z=12.0` | **FIXED** (4th issue, found while fixing the other 3 and flagged by nobody) — per-subject `dz` now |
| **`baselines/export_resp_stack.py:101`** `reslice_volume_vec(V, disp[0,z])` — **no `spacing`** | ⚠️ **STILL OPEN** — silently 12 mm; feeds the SVR baselines, so it would bake the same error into the new baseline comparison |
| `cmrxrecon.py:28-29` still `DATA_ROOT=CMRxRecon2024/Cine_combined` + `random_8_1_1.txt` | ⚠️ open — decide whether the eval bundle should move to the pooled split |
| ~~`scratch/data/CMRxRecon20*_recon_v1_espirit_imagedomain/` (851 subjects)~~ | **RESOLVED 2026-07-31 — DEPRECATED, no action.** User decision: this recon is deprecated and out of scope. It is not flipped and will not be; it is absent from `pooled.txt`/`manifest.csv` and nothing in the training path reads it. `tools/eval_cmrxrecon_2023_2025.py` still points at it — treat that script as deprecated too rather than fixing it. |

The `:124` one is the trap. `acdc.py:143` and `ocmr.py:144` both pass `spacing=native_spacing_dhw`;
`cmrxrecon.py` is the lone outlier, so for any non-12 mm subject the simulated breathing shift is
understated proportionally (10 mm → 17 % too small, 5 mm → 58 %) **while `resp_disp_mm` records the
requested value** — i.e. exactly the silent, pitch-keyed mis-grading described in §8.1a, which is
the failure mode that had to be retracted once before. Regenerating with this unfixed produces a
clean-looking bundle with the bug inside it.

⚠️ Per the standing rule, `evaluation/` was **not** modified by this work — these are flagged for
the user to action, not applied.

### The `/prove-it` audit (2026-07-31) — what it found AFTER the flip was applied

6 reviewers + 4 adversarial verifiers, on a GPU node. **Conclusion: the 15030 rewritten files are
not corrupted** (see the verification table above, plus round-trip fidelity across all 3 on-disk
dtypes, CMRx 3D-vs-4D bit-consistency 60/60 subjects / 720 frames, and an end-to-end `MRIDataset`
load showing corr(V_gt, native image) +0.93…+0.9999 and corr(canonical seg, native seg) **+1.0000**).
The defects it found were **operational**, not data corruption:

| finding | status |
|---|---|
| **`--apply` had no re-run guard** — a second run would silently flip all 893 back and overwrite the sidecar with a record falsely asserting the fix is applied. `preflight_zdir` is *provably* blind to this (Pearson r is invariant when both profiles are reversed). | **FIXED** — refuses if `PROV` exists; `--force` overrides |
| **`fix_slice_roll.py --revert` ordering** — flip and roll do NOT commute (`F∘R_k = R_{-k}∘F`); undoing the roll first leaves the stack **2 slices wrong, silently**, on 410 of 464 subjects | **FIXED** — refuses while the flip sidecar exists |
| `fix_slice_roll.py` called `main()` without propagating its return, so the new guard printed but exited **0** | **FIXED** — `raise SystemExit(main())` |
| **`fix_slice_roll.py::subject_files()` omits `heart_seg`/`heart_roi`/`*_canonical`** (they postdate it) — a revert would move images but not labels | ⚠️ **OPEN** — the guard above blocks the path for now; extend `subject_files()` before ever reverting |
| 14 sbatch launchers hardcoded `EXP_NAME=…_Cine_combined`, overriding the composed `exp_name` | **FIXED** — now `…_pooled1343` (they override only `exp_name`/`max_epochs`, never `data_root`/`split_file`, so runs always used the right cohort) |
| 7 latent-only issues (`--limit` sidecar desync, partial-revert record loss, non-atomic sidecar write, tmp-name/glob collision, preflight non-abort, `_profile(ed_only=False)`, `f1==0` sign class) | open, **all fire only on a re-run**, which the guard now blocks |

**The single most valuable thing it surfaced** is the M&Ms GT-mask validation (320/320) recorded
above — and the correction that the "converter guarantees apex-first" premise used to select the
detector rule is itself unsound.

### The cache trap

`cache_signature()` in `training/data/preprocess.py` was bumped by adding
`"slice_order_apex_at_z0"` to the signature tuple: **`bab7860607` → `f937056bb3`**. This is not
cosmetic — monai's `PersistentDataset` keys its cache on the input **paths**, not on file
contents, and the flip left every path unchanged, so without the bump a warm node would have
silently trained on pre-flip volumes. The old cache dir is left in place (orphaned on `/tmp`),
not deleted.

## 10c. Chirality (LV/RV left-right handedness) — MEASURED and CLOSED via flip aug (no on-disk fix)

> **Status 2026-07-31 (updated): 29% of CMRx is genuinely mirrored, and this is confirmed a
> *reflection*, not a rotation (§"Ruled out" below). Decision taken: NO on-disk fix. The training
> objective is exactly mirror-equivariant (measured), so mirroring is a training no-op, and
> `RandFlipd` has been RE-ENABLED in all three aug tiers (`training/data/gpu_aug.py`), and
> `data.augmentation.enable` is now **`true`** by default in `mri_volume.yaml`, making the model
> explicitly chirality-robust. Nothing on disk has been changed and `tools/fix_chirality.py` is
> deliberately not being built. The two follow-on worries (val/test source confound; nnU-Net
> Task114 label quality on mirrored subjects) were both raised and both **resolved** — see below.
> The only genuinely open item is the root cause, which is now optional curiosity, not a blocker.**

Same bug *class* as §10a (base/apex) but on the **in-plane** axes instead of Z: a mirror-flipped
in-plane axis puts the RV on the anatomically wrong side of the LV. In a normal (D-looped, situs
solitus, >99.9% of humans) heart the RV sits **anterior and to the patient's right** of the LV —
this is real fixed anatomy, not a viewing convention, so it makes a usable detector: for each
subject, take `heart_seg.nii.gz` (LV=1, RV=3, union over all T), compute the LV→RV centroid vector
in raw voxel space, rotate it into world RAS mm via the affine's linear part (translation cancels
in a vector difference), and check the sign of the R component. Built as
`tools/probe_chirality.py`, mirroring `tools/probe_slice_order.py`'s structure.

### Result across the full 1343-subject training pool

| source | n | R>0 (normal) | R<0 (flipped?) |
|---|---|---|---|
| ACDC | 150 | **150** | 0 |
| M&Ms | 345 | 340 | 5 |
| CMRx | 848 | 601 | **247** |

**ACDC is clean.** **M&Ms is clean once you look past the raw split**: M&Ms ships its own GT
segmentations (`MNMs1/*/<CODE>/<CODE>_sa_gt.nii.gz`, real oblique affine, independent of the
converter) — cross-checking each of the 5 "R<0" M&Ms subjects against their GT-derived R sign
shows all 5 are actually R>0 (27–34mm, confidently positive); the detector's read on the
*converted* axis-aligned affine only goes wrong because these 5 subjects' true LV→RV vector points
almost straight anterior (A=37–47mm) rather than right-anterior, a precision/borderline artifact
on the synthetic affine, not a real flip. So M&Ms is effectively **345/345 correct**.

**CMRx is a real, substantial anomaly: 247/848 (29%).** The R<0 group's mean R is **-14.0** vs the
R>0 group's **+22.1**.

> ⚠️ **Correction (2026-07-31): an earlier version of this paragraph called those "two clearly
> separated populations, not a cluster near zero". That was wrong** — it was inferred from group
> *means* without looking at the distribution. Recomputed from the same
> `result/chirality_check/chirality.csv`: **160/848 CMRx subjects (18.9%) have |R| < 5 mm** and
> 103/848 (12.1%) have |R| < 3 mm; 67 of the 247 flagged-as-flipped are inside 5 mm. The
> distribution is continuous through zero, not bimodal with a gap. The real contrast is
> *cross-dataset*: ACDC has **0/150** inside 5 mm (M&Ms 10/345 — and 5 of those were exactly the
> false positives this section already had to retract using GT masks). **Consequence: the
> single-feature R-sign rule is NOT trustworthy enough to drive a per-subject on-disk edit** —
> roughly one CMRx subject in five would be adjudicated on a few millimetres of centroid offset,
> with no CMRx ground truth and root cause unknown, and a wrong call silently converts a *correct*
> subject into true situs inversus. This is a major input to the "no on-disk fix" decision below.
> (Per-vendor breakdown recomputed and confirmed: CMRx2025-Philips 83% R<0 (n=12), CMRx2025-UIH
> 51% (n=178), CMRx2025-Siemens 16% (n=169), CMRx2024 21% (n=294), CMRx2023 29% (n=195) —
> suggesting *two* mechanisms, one vendor-level and one per-subject.)

Broken out by source/vendor, **CMRxRecon2025 UIH is 88 vs 90 — an almost exact 50/50 split
within one scanner**, and CMRx2023/2024 (both "single-vendor, documented-healthy") also show
20–29% minorities. Situs inversus/dextrocardia (the only real biological cause of reversed
chirality) is ~1 in 10,000 people — a near-coin-flip split within one vendor rules out biology and
points to a systematic, roughly-per-subject data/pipeline issue (leading hypothesis, **not yet
checked**: patient table positioning, i.e. head-first vs feet-first — a real DICOM field a recon
pipeline can silently ignore, which produces exactly this signature).

### Ruled out: this is not a rotation, and not a rendering artifact

**The decisive test (added 2026-07-31) is a direct one on the data, and it supersedes the affine
argument below as the primary evidence.** "Flipped" and "mirrored" are the same thing; the
meaningful contrast is **reflection (det −1) vs rotation (det +1)**, and no sequence of rotations
ever produces a reflection. The two hypotheses make *different, checkable* predictions about the
LV→RV vector's **A** (anterior) component, which the R-sign detector ignores entirely:

- a **180° in-plane rotation** maps `(R, A) → (−R, −A)` — **both** flip;
- a **mirror** maps `(R, A) → (−R, +A)` — only R flips.

Measured over `result/chirality_check/chirality.csv` (CMRx, n=848): the fraction with `A > 0` is
**0.401 in the R>0 group vs 0.437 in the R<0 group**. Rotation predicts ≈0.60. **It is a
reflection.** No affine reasoning required — this is read straight off the segmentation centroids.

Two further things had also been checked before trusting the R-sign split, both driven by the user
pushing back on the first pass:

1. **Rendering bug (fixed).** The first visual render (`tools/render_chirality_check.py`) used
   `imshow` with no `aspect` argument on **raw native images** (not yet resampled to the canonical
   isotropic grid — in-plane pixel spacing varies per subject and isn't always square), which
   visibly squashed/stretched anisotropic subjects. Fixed by computing
   `aspect = voxel_sizes[0]/voxel_sizes[1]` from the affine per subject. The chirality read was
   unchanged after the fix — not an artifact of the squashing.
2. **Rotation vs. mirror (settled by direct affine comparison, not just eyeballing).** A rotation
   preserves handedness (RV stays on the correct side, just at a different clock-position); only a
   reflection (determinant −1) actually flips it — and no sequence of rotations can produce a
   reflection, so distinguishing them matters. Compared the raw affines of an R>0 and an R<0 CMRx
   subject directly:
   ```
   R>0 (CMRx25_R2test_Center001_Siemens_30T_Vida_P013):  diag(-1.62, -2.56, +12.0)
   R<0 (CMRx24_Train_P005):                              diag(-1.58, -1.64, +12.0)
   ```
   Both are pure diagonal matrices with the **identical sign pattern** (-,-,+) — the magnitudes
   differ (different native pixel spacing, expected) but the rotation/reflection part of the
   coordinate mapping is the same for both subjects. CMRx stamps one fixed axis-aligned convention
   on every subject, not a per-subject measured rotation. Since the two subjects' affines encode
   the *same* coordinate transform, RV landing on opposite sides of LV between them cannot be
   explained by a rotation difference (there isn't one) — the only remaining explanation is that
   the raw pixel array itself is mirrored between the two groups. Confirmed visually on 5 example
   pairs per group (`result/chirality_check/render_check.png`): RV sits consistently *above* LV
   when R>0 and consistently *below* LV when R<0, with the affine-derived R/A reference arrows
   pointing in the identical screen direction in every panel.

### Decision (2026-07-31): no on-disk fix; re-enable flip augmentation instead

Unlike §10a (base/apex), where the array order silently set the *direction of simulated breathing*
and therefore had to be corrected on disk, **chirality is a no-op for the training objective.** This
was measured, not argued:

- `loss_volume = (V_canon - V_gt).abs().mean()` (`training/loss.py:153`) takes both operands from
  the *same* subject array (`mri_dataset.py:300`, the single axis-order site), and `scanner_coords`
  is a pure pixel-index→[−1,1] map (`mri_dataset.py:472-483`) — so a consistent W-mirror mirrors the
  intensities **and** each pixel's assigned x-coordinate together.
- Verified numerically: `max|splat(−x) − flipW(splat(x))| = 1.25e-06` over 20k random points (fp32
  scatter noise), and the L1 against a correspondingly-flipped `V_gt` is `allclose → True`.
- RV location is **observable in every input slice**, not prior knowledge. A learned prior only buys
  something for information *absent* from the input. The old `gpu_aug.py` rationale ("mirror hearts
  degrade the LPS-trained RV-location prior") asserted a benefit from a prior on an observable and
  is retracted.
- **Also retracted: the respiratory argument.** Two independent reviewers verified the mirror is on
  `W` (=X=L/R) while `_build_disp_dhw` puts the AP shift on `H` with the W component identically
  zero (`respiratory.py:198-206`, `ap_axis="H"`), and the tilt azimuth is `φ ~ U(0,2π)` — symmetric
  under a W mirror. **Respiratory simulation is chirality-neutral, exactly and in distribution.**

Given that, plus the detector ambiguity quantified in the correction box above (18.9% of CMRx
within 5 mm of zero, no CMRx ground truth, root cause unknown, 6 file types per subject to rewrite),
an on-disk fix is **higher-risk than the defect**. Chosen response instead: **`RandFlipd(prob=0.5,
spatial_axis=[2])` re-enabled in all three tiers** of `training/data/gpu_aug.py`, making the model
explicitly chirality-robust rather than betting on a millimetre-scale per-subject adjudication.

Known, unmeasured cost of that choice, recorded honestly: the point head outputs a **vector** field,
so mirror-invariance requires a coupled spatial flip **and** a `Δx` sign negation — a strictly harder
symmetry than nnU-Net's label-map mirroring (the standard precedent for mirror aug in this domain),
paid from a fixed capacity/step budget. Flip is otherwise the cleanest transform available: an exact
voxel permutation, with no interpolation blur and no zero-padding, unlike every other aug in the
tiers.

### Two follow-on worries, both raised and both RESOLVED (2026-07-31)

An earlier draft of this section listed the following as open. Both were wrong; recording them
with their resolutions so they don't get re-raised.

1. ~~"Chirality is a near-perfect source label (21.2% train / 18.7% val / 10.4% test under
   `pooled.txt`), and since aug is train-only (`trainer.py:209`) it can't clean the val/test/OOD
   confound."~~ **Resolved — this misunderstands what invariance buys.** You do not need to
   augment val. If the model is flip-invariant, its response to a mirrored val subject is the same
   as to a non-mirrored one, so the val/test chirality mix stops being a distinguishing feature at
   all; augmenting val would be the wrong lever. Separately, the chirality mix is identical across
   every method evaluated on this cohort, so method-vs-method baseline comparisons are unaffected
   regardless. *Residual (minor, not a blocker):* our invariance is learned, not architectural, so
   it is approximate — verifiable post-hoc by feeding a val subject and its W-mirror to the trained
   model and confirming the metrics match.
2. ~~"The headline EF/Dice path is chirality-consuming and external: `heart_seg`,
   `heart_roi_canonical` (consumed by `training/loss.py:316-330`), `cardiac_phase.csv` and the EF
   pseudo-truth for the 247 mirrored subjects were generated by M&Ms-trained nnU-Net Task114, and
   we cannot make that segmenter chirality-robust."~~ **Resolved — Task114 is already
   chirality-invariant by construction.** Verified in the installed nnU-Net v1:
   `nnunet/training/data_augmentation/default_data_augmentation.py:70-71` sets
   `"do_mirror": True` and `"mirror_axes": (0, 1, 2)`, and
   `nnUNetTrainerV2_MMS.setup_DA_params()` (the trainer this project uses, `-tr
   nnUNetTrainerV2_MMS`) calls `super().setup_DA_params()` and overrides rotation / elastic /
   scale / brightness / gamma but **never `do_mirror` or `mirror_axes`**. Inference mirror-TTAs by
   default too (`nnUNetTrainer.py:459`). So the segmenter was trained *and* is inferred with
   3-axis mirroring; the derived labels for the mirrored subjects are not suspect, and the
   originally-proposed "measure Task114 on mirrored vs non-mirrored subgroups" probe is
   unnecessary.

### Still genuinely open

- **Root cause remains unknown** and is still worth chasing: the Philips-83% / UIH-51% /
  Siemens-16–29% pattern suggests one vendor-level convention plus one per-subject DICOM field
  (patient table positioning, head-first vs feet-first, is the leading unverified hypothesis).
  Finding it would give a per-subject decision rule for free, instead of tuning a v2 detector
  against a boundary where a fifth of the cohort sits.
- **`tools/fix_chirality.py` deliberately does not exist.** Nothing on disk has been touched.

Provenance: `tools/probe_chirality.py` (detector, full-cohort CSV →
`result/chirality_check/chirality.csv`), `tools/render_chirality_check.py` (visual check →
`result/chirality_check/render_check.png`).

## 11. Provenance

- `tools/gate_native_z_identity.py` — the implemented acceptance gate (§9, §12). Splats each
  subject's own `gt_target_volume` planes back at their own exact position (Δ=0) at native 256×256
  resolution, using that subject's real `dz_mm`/`z_scale` pulled straight from `MRIDataset`.
- `tools/probe_zgrid_alignment.py` — E1–E4, reproduces every number quoted here.
- `tools/convert_to_sax_layout.py` — ACDC/M&Ms → CMRx layout (§8.2 A2); `sbatch/convert_sax_layout.sh`.
- `tools/verify_sax_conversion.py` — proves the conversion is lossless (bit-exact vs source after the
  recorded permute+flip), fault-injected on every run. ACDC 150/150, M&Ms 345/345.
- `tools/render_converted_sax.py` — visual gate, all z-planes, CMRx rows as the orientation reference
  → `result/converted_sax_check/`.
- `tools/fix_slice_order.py` — **the on-disk flip** (§10b). Per-subject from
  `result/slice_order_check/slice_order_decisions.csv`; atomic, `--revert`, `--verify`,
  `--fault-inject`, and a `preflight_zdir` that aborts if any subject's canonical files run the
  opposite z direction to its native files.
- `result/slice_order_check/slice_order_decisions.csv` — the per-subject decision list (1343 rows:
  `subject, source, order, decided_by, flip, f1, f2, f3`). **This file, not any transcript, is the
  authority for what was flipped and why.**
- `tools/render_slice_order_check.py` — three taper features + the LONG-AXIS SIDE VIEW that makes
  the base/apex call visible by eye (§10a). Every SAX plane is a donut, so the taper is invisible
  slice-by-slice; reformatting through the LV centroid along z shows the cone.
- `tools/render_respiratory_direction.py` — drives the real `reslice_volume_vec` over a Lujan
  breath cycle and renders the side view as a GIF, turning the §10a direction *derivation* into a
  measurement (§10a). Default is a **reduced** sim (pure SI, tilt 0, no AP) that isolates the
  direction question; **`--real`** instead reads the live `mri_volume.yaml` and applies the FULL
  trainer displacement (per-subject tilt 0–45°, AP = 0.35×SI, amplitude 18.8±7.35 mm) with a
  second short-axis panel so the in-plane component is visible too.
- `tools/probe_slice_order.py` — base-vs-apex ordering from the segmentations (§10a). Decides from
  **anatomy**, unlike `plan_reframe()` which decides from the affine — the only trustworthy route
  when 2 of 3 sources have a fabricated header.
- Related: docs/27 (slice pitch), docs/38 (GT-referenced val metrics), docs/54 (2023/2025 recon),
  docs/55 (Philips in-plane), docs/56 (odd-Z slice roll), docs/17 & docs/39 (cardiac-phase semantics
  across cohorts — dissolved for training by the reference-slice design, see §6.4).

## 12. Implementation (2026-07-31) — what actually happened

§8.1 (training path) is fully implemented and verified. This section records what the plan above
got right, where implementation deviated from it, and — most importantly — the bugs the plan did
not anticipate. Read this before assuming §8's checklist is a complete account of what changed;
it isn't, it's the *plan*, and reality had two surprises the plan didn't have line items for.

### What matched the plan exactly

The core formula change (`pz = pos_z*z_scale + (D-1)*0.5`, `z_scale = Z_HALF_MM/dz` required on
`splat_to_volume`/`sample_volume`), `Z_HALF_MM = 90.0`, `z_norm = (z_i-(D-1)/2)*dz/Z_HALF_MM` with
the `|z_norm|<=1` assert, `preprocess.py`'s `Spacingd(pixdim=(1.4,1.4,0.0))` /
`ResizeWithPadOrCropd(spatial_size=(256,256,-1))`, `continuous_z: false`, `len_train =
max(1000, len(self.subjects))`, `trainer_viz.py`'s `T_total` fix, and the `respiratory.py`
`SPACING_MM`/`N_CANON_PLANES` fixes (§8.1a) — all landed as designed. `dz` is stored via a new
`RecordSpacingD` monai transform (not explicitly speced in §8.1's table, but the natural
implementation of "store `dz` in the cached dict").

### Deviations from the plan

- **`extract_slices_with_respiratory{,_vec}`'s `spacing` became required (no default) — but
  `reslice_volume{,_vec}`'s did NOT**, contrary to §8.1a's "drop the `spacing=SPACING_MM` default
  on ... `reslice_volume_vec`". Reason: unlike `extract_slices_with_respiratory_vec` (zero
  `evaluation/` callers, confirmed by grep), `reslice_volume_vec` **is** called from
  `evaluation/engine/build_inputs/cmrxrecon.py` without an explicit `spacing` argument — and per
  the standing rule, `evaluation/` is not to be touched without being asked. Made `spacing`
  required only on the functions with zero `evaluation/` callers; the two `sample_displacements`/
  `sample_displacement_vectors`/`sample_resp_disp` functions instead gained a new *optional*
  `n_planes=None` kwarg (falls back to the old `N_CANON_PLANES` for un-migrated callers), and only
  the real training call site (`gpu_aug.py`) was updated to pass it. **Net effect: `evaluation/`'s
  respiratory-motion code is still silently on the old fixed-12mm/D=12 assumption** — this was a
  deliberate, scoped choice, not an oversight; flagging it here so nobody assumes `evaluation/` was
  covered by "the respiratory fix."
- **`heart_roi_canonical.nii.gz`'s stale-grid handling is warn-and-skip, not a hard assert.** §8.4
  said "must be regenerated... and a shape assert added." Implemented instead as: check the shape,
  and if it doesn't match `(D,256,256)`, log a warning and omit the key from the batch (metric
  simply skips that sample) rather than raising. Reason: this lets training run *now*, before the
  ROI regeneration (§8.4, still not done — blocked on the whole-heart-seg job) — the existing
  standing rule against `rm`-ing regenerable outputs to force a rebuild applies in spirit here too;
  a hard assert would have made every subject with a stale ROI file (i.e. all of them, right now)
  block training entirely.
- **The verification gate is a new file, not `probe_zgrid_alignment.py::e3_native_grid`.** See
  next subsection — the originally-proposed template's methodology turned out to measure the wrong
  thing.
- **The refiner was deleted, not fixed.** §10's open item said `enable_refiner` "needs handling if
  ever enabled" — implementation found the specific handling required (deriving `grid_shape`
  without requiring GT) and the user chose deletion over that fix, since the refiner was unused
  anyway. See "Refiner removal" below.

### Bug the plan did not anticipate: `composed_dataset.py`'s silent key allowlist

The single most important thing to know if you're touching this pipeline again:
**`training/data/datasets/mri_dataset.py::MRIDataset.get_data()`'s returned dict does NOT
automatically reach the training batch.** `training/data/composed_dataset.py::ComposedDataset.__getitem__`
is a **hand-written, per-key allowlist** (`if "t_target" in batch: sample["t_target"] = ...`, one
line per key) — not a generic dict copy or collate. When `get_data()` was extended to emit `dz_mm`/
`z_scale` (needed by `gpu_aug.py`'s respiratory path and by `loss.py`/`vggt.py`), those two keys
were silently dropped by this allowlist. Every unit test still passed (tests build batches by hand,
bypassing `ComposedDataset` entirely) — the bug was only caught by actually running `torchrun`,
where `training/trainer.py`'s `train_epoch` hit a `KeyError` deep in `gpu_augment_batch`. **Any
future key added to `get_data()`'s return dict needs an explicit line added to this allowlist, or
it will silently vanish before reaching the model.** (Confirmed via full audit this is the *only*
such chokepoint — `dynamic_dataloader.py`'s collate and `trainer.py`'s `copy_data_to_device` are
both generic/recursive.)

### Bonus fix found along the way: a pre-existing `splat_to_volume` boundary bug

Unrelated to native-z but touched in the same diff since it directly blocked the verification
gate: `splat_to_volume`'s in-bounds check (`vggt/utils/splat.py`) tested the *floored* voxel index
against `size-2` (`x0f <= W-2`), which excludes a point sitting **exactly** on the last valid voxel
index (it has no "next neighbor" to interpolate toward, so the old check discarded it). This
silently dropped the boundary plane/row/column of every splat. Invisible in x/y (the 518→256
resize oversamples, so neighboring points cover for the dropped exact-boundary one) and harmless
under the old fixed-12-plane grid (the top z-plane was usually zero-padding anyway). Under
native-z, `D` is each subject's *real* slice count with no padding, so the top z-plane is real
anatomy for every subject — this would have failed the verification gate outright. Fixed by
testing the *continuous* position against `size-1` and clamping both interpolation corners into
range (verified: never double-counts, since the corner landing exactly on the boundary always has
zero weight for its phantom neighbor; still correctly drops genuinely out-of-bounds points).

### Why the gate is a new file, not the planned template

The first attempt at `tools/gate_native_z_identity.py` followed §9's plan literally: splat the
model's *actual* 518×518 `images`/`scanner_coords` (post bilinear-upsample from the 256×256
canonical slice) back into the 256×256 output grid. Measured ~35–47dB, not the expected ~120dB.
Root cause, confirmed by isolating a single fixed z-plane (making z_scale/native-z irrelevant) and
still measuring ~39dB: the 256→518→(re-splat)→256 round trip has its **own, unrelated,
pre-existing** lossiness (needed for the DINOv2 518px input requirement — a different instance of
the same push/pull mismatch described in §5.3, just in x/y instead of z). Rewrote the gate to
splat `gt_target_volume`'s own planes directly at native 256×256 resolution (no image-upsample
step), isolating z-axis correctness specifically — that version cleanly hits 120dB. **Lesson: when
testing one axis's correctness, don't route through a pipeline stage with its own independent
lossiness on a different axis — it swamps the signal.**

### Refiner removal

The native-z refactor exposed a real design tension in the optional 3D-UNet "refiner" module
(`vggt/models/refiner.py`, now deleted): its forward-pass branch used to read a *fixed*
`self.grid_shape` (set once at model construction); under native-z that had to become
`tuple(batch["gt_target_volume"].shape[1:])`, derived per-forward — which made ground truth a hard
requirement to run the refiner **at all**, including at inference, breaking
`inference/run_rtfb.py`'s documented ground-truth-free real-time use case. A `/prove-it` adversarial
verifier confirmed this is a real, reachable code path (traced), but currently **dormant**: no
config anywhere sets `enable_refiner: true`, no checkpoint has ever trained refiner weights, and an
earlier missing-weights crash in the inference model loader would fire first regardless. Given all
of that, the user said to delete the subsystem outright rather than redesign its `grid_shape`
sourcing (e.g., accepting an optional override, falling back to GT-derivation only in training).
Removed: `vggt/models/refiner.py`, `tests/test_refiner.py`, all refiner wiring in
`vggt/models/vggt.py`/`training/loss.py`/`training/trainer_viz.py`/`training/trainer.py`, and every
`enable_refiner`/`refiner_*` config key. Verified via a live Hydra compose check and the full test
suite (203/203 passed post-removal). **Deliberately NOT touched**: the ~60 `tools/`/`inference/`/
`evaluation/` files that reference "refiner" in some form — almost all are one-off historical
analysis scripts already out of scope per the standing `tools/`-fix-on-demand /
`evaluation/`-off-limits rules, not newly broken by this removal.

### Verification performed

`tools/gate_native_z_identity.py` on 35 real subjects (CMRx2024, CMRx2025 incl. the tallest known
subject D=18, ACDC, M&Ms) spanning 5–12mm pitch: fault-injected run (deliberately wrong `z_scale`)
→ all 35 correctly FAIL at 14–22dB; real run → all 35 PASS at exactly 120.00dB. `pytest tests/`:
221/221 before the refiner removal, 203/203 after (the 18-test delta is exactly
`tests/test_refiner.py`, deleted). A real `torchrun --config mri_volume` smoke run (3 train steps,
2 val steps, `ef_val_sweep=false`) completed cleanly with real gradients and the full metric set. A
6-reviewer + 2-adversarial-verifier `/prove-it` pass covered the full diff; findings are the two
bugs described above (both fixed) plus the already-known-and-deferred `inference/`/`tools/`
breakage cataloged in §8.3.

## 13. Manifest CSV + pooled split (2026-07-31) — what actually happened

§2.1's split table was the *plan*; this section is what got built, verified, and is now on disk
at `training/splits/manifest.csv` (1343 rows) and `training/splits/pooled.txt`. Superseded §2.1's
"CMRx ×3 as one stratified pool" — the real rule treats each CMRx year separately, and 2025 uses a
residual mechanism instead of a fixed ratio. Built via `tools/build_manifest.py` (manifest) and
`tools/build_pooled_split.py` (split), both git-tracked.

### The manifest

One row per subject, columns: `id, source, rel_path, official_split, n_z, pitch_mm, z_extent_mm,
T_native, num_phases, vendor, scanner_model, field_strength_t, centre, pathology_label,
pathology_detail, age, sex, height_cm, weight_kg, ed, es, split, source_file` — a superset of
§2.1's originally-proposed columns. Two design choices worth recording:

- **Geometry (`n_z`/`pitch_mm`/`z_extent_mm`/`num_phases`) is read directly off each subject's own
  `sax/3d_recon/sax_frame_00.nii.gz` on disk via nibabel** — never trusted from a provenance
  JSON/CSV — so the manifest can't silently drift from what `MRIDataset` actually loads. Verified:
  the resulting pitch distribution (748@12mm/478@10mm/56@9.6mm/...) and M&Ms diseased fraction
  (256/345 = 74.2%) both independently reproduce numbers already on record (§2, CLAUDE.md), which
  cross-validates the read-from-disk approach.
- **CMRxRecon2025 is NOT single-vendor/healthy like 2023/2024** — this was undocumented before
  this session. Joined against `_provenance/CMRxRecon2025_TaskR1_TaskR2_Disease_Info.xlsx` (via
  `tools/scan_cmrx2025_disease_info.py`, imported as a sibling module): of the 359 on-disk SAX
  subjects, only ~34 are healthy (~10%), the rest diseased/unknown, across 3 vendors (Siemens/UIH/
  Philips — no GE despite it appearing in the xlsx sheets) and 8 centres. **Philips (12 subjects)
  appears ONLY in the challenge's own official R1test/R2test splits — zero in train or val** — the
  same shape as M&Ms's Canon-only-in-val/test finding in §2.1, just not previously known for 2025.
- `es` is blank for all 848 CMRx subjects (needs `cardiac_phase.csv` regeneration, §8.4, blocked on
  the whole-heart-seg job) — `ed` is always 0 by construction (ED-anchored convention, §7) for
  every source, CMRx included.

### The split rule

- **ACDC** — official `testing/` (50 subjects) kept **intact**, exactly as §2.1 planned. Within
  `training/` (100 subjects, 20 per pathology group × 5 groups): shuffle each group, first 3 → val,
  remaining 17 → train. (85/15/50, matches §2.1's plan exactly.)
- **M&Ms** — official split as-shipped: `Training/{Labeled,Unlabeled}` → train, `Validation` → val,
  `Testing` → test. (175/34/136, matches §2.1's plan exactly.) Canon confirmed 0 subjects in train
  by direct query of the built manifest.
- **CMRx2023 / CMRx2024** — plain random **8:1:1**, computed independently per year (not pooled
  across years as §2.1 originally proposed) — both are single-vendor, single/near-single-centre,
  documented-healthy cohorts, so there is no diversity axis worth stratifying by, and 8:1:1 matches
  the historical `random_8_1_1.txt` convention rather than an arbitrary new ratio. 2024 additionally
  soft-preserves val/test membership from the deprecated `training/splits/random_8_1_1.txt` (filtered
  to ids that still exist on disk, capped to the new smaller 8:1:1 targets) so continuity with the
  prior training series isn't gratuitously broken.
- **CMRx2025** — two-stage. (1) The 12 Philips subjects are pinned out of train entirely (split 6
  val / 6 test) — mirrors the M&Ms Canon treatment, for the reason found above. (2) Everyone else
  (Siemens/UIH, 347 subjects) is stratified by `(vendor, pathology_label)` via a round-robin
  interleave (shuffle each vendor×pathology bucket, then pop one subject from each non-empty bucket
  in turn to build one interleaved list, so a straight positional cut still gives every split a
  representative share of every bucket) — **but the cut sizes are not a fixed ratio.** They are the
  exact residual needed to make the WHOLE POOL's grand total land on 7:1:2 (70:10:20), given ACDC/
  M&Ms/2023/2024 are already fixed by the rules above:
  ```
  grand target        = round_targets(1343, [0.70, 0.10, 0.20])       = 940 / 134 / 269
  fixed contribution   = ACDC + M&Ms + CMRx2023 + CMRx2024             = 651 /  97 / 236
  CMRx2025 residual (incl. Philips)                                    = 289 /  37 /  33
  minus Philips (0 / 6 / 6)  ->  non-Philips 2025 target                = 289 /  31 /  27
  ```
  **Non-obvious consequence, flagged during design and worth restating:** ACDC (33% test) and
  M&Ms (39% test) already supply 236 of the pool's 269 test slots (88%) but only 97 of its 134 val
  slots (72%) — the test budget gets claimed faster than the val budget. So the residual left for
  2025 is val-heavier than test-heavy (37 val vs 33 test) — **2025's own per-source ratio is
  80.5%/10.3%/9.2%, MORE train-heavy than 2023/2024's plain 8:1:1, not less** — the opposite of
  "2025 should get more test scrutiny since it's the diverse source." This was a deliberate,
  discussed tradeoff (user chose the exact-70:10:20-grand-total mechanism over letting 2025 keep
  its own test-heavy ratio), not an oversight — recorded here so a future reader doesn't mistake it
  for a bug.
- **Tie-breaking**: `round_targets`'s largest-remainder rounding breaks ties toward the LATER
  split index (test over val) — a tied val/test target lands val≤test, never the reverse, per user
  preference.

### Final numbers (verified against the actual files, not recomputed from memory)

| source | train | val | test |
|---|---|---|---|
| CMRxRecon2023 | 156 | 19 | 20 |
| CMRxRecon2024 | 235 | 29 | 30 |
| CMRxRecon2025 | 289 | 37 | 33 |
| ACDC | 85 | 15 | 50 |
| M&Ms-1 | 175 | 34 | 136 |
| **total** | **940** | **134** | **269** |

`940 + 134 + 269 = 1343`. Ratio 69.99% / 9.98% / 20.03% ≈ **70:10:20**.

> ⚠️ **SUPERSEDED 2026-07-31 by the duplicate exclusion — see §13a. The live cohort is 1337
> (935 / 133 / 269).** The table above is the assignment *before* 6 source-shipped duplicate
> subjects were removed. The assignment itself is unchanged; only those 6 rows dropped out.

### Verification performed

Re-read directly from `training/splits/manifest.csv` (not from memory/recomputation): 1343 rows,
1343 unique ids, every row has a non-empty `split`, per-source counts match the table above exactly.
ACDC val is exactly 3-per-pathology-group (DCM/HCM/MINF/RV/NOR) and all 50 ACDC test rows have
`official_split == "testing"` (none leaked from `training/`). M&Ms Canon: 0 in train, 10 val / 40
test. CMRx2025 Philips: 0 in train, 6 val / 6 test. The generated `training/splits/pooled.txt`
loads through the real `MRIDataset(...)` constructor (not a hand-parse) for all three splits,
returning exactly 940/134/269 subjects. `tools/build_manifest.py` and `tools/build_pooled_split.py`
are both idempotent and reproducible (`random.Random(42)` throughout) — rerunning regenerates the
identical split.

### Not done yet

~~Nothing in the actual training config points at `pooled.txt`.~~ **DONE — see §14.** ~~The
remaining §8.3/§8.4 items (heart_roi_canonical regeneration, `cardiac_phase.csv` for CMRx `es`) are
still open.~~ **Both also DONE (2026-07-31) — see §8.4.** §8.3 (`inference/`/`evaluation/`/`tools/`)
remains deliberately deferred.

## 13a. Duplicate subjects excluded — cohort is 1337, not 1343 (2026-07-31)

The `/prove-it` audit (docs/59 F3) found that **ACDC and M&Ms each ship some subjects twice under
different ids**, including across their own official split boundaries. This was inherited, not
introduced: `tools/convert_to_sax_layout.py` faithfully converted both copies.

**Evidence — verified at the RAW source** (`scratch/data/ACDC/…/patientNNN_4d.nii.gz`,
`scratch/data/MNMs/MNMs1/…/XXXXXX_sa.nii.gz`), on the **full native 4D** (all 25/30 native frames,
not just our 12 resampled phases), using `np.array_equal` on the voxel arrays plus an exact affine
comparison:

| pair | native shape | voxels equal | max abs diff | affine identical | file md5 equal |
|---|---|---|---|---|---|
| `ACDC_patient055` (train) ↔ `ACDC_patient118` (test) | (256,216,9,25) | ✅ | 0.0 | ✅ | ✗ |
| `ACDC_patient074` ↔ `ACDC_patient076` (both train) | (256,256,8,30) | ✅ | 0.0 | ✅ | ✅ |
| `MNMs_A7G0P5` (train) ↔ `MNMs_K3R0Y7` (val) | (320,320,10,30) | ✅ | 0.0 | ✅ | ✗ |
| `MNMs_C8J7L5` (val) ↔ `MNMs_C8O0P2` (test) | (256,256,10,25) | ✅ | 0.0 | ✅ | ✗ |
| `MNMs_A8C9H8` ↔ `MNMs_Q0Q1Y4` (both train) | (256,256,10,25) | ✅ | 0.0 | ✅ | ✗ |
| `MNMs_C5Q2Y5` ↔ `MNMs_E9L4N2` (both train) | (256,256,10,25) | ✅ | 0.0 | ✅ | ✗ |

Same voxels *and* same affine — only the gzip containers differ (074/076 are byte-identical files).
`ACDC/training/patient055` and `ACDC/testing/patient118` are literally the same scan filed twice.
Visual confirmation: `result/duplicate_pairs/duplicate_pairs.png`
(`tools/render_duplicate_pairs.py`).

**Which member was dropped.** Rule: never delete from an evaluation split when a train member
exists (train has ~940 to spare; eval sets are the scarce resource); for a val↔test pair keep test;
for a train↔train pair the choice is arbitrary, so drop the lexicographically later id.

| dropped | duplicate of | was in | reason |
|---|---|---|---|
| `ACDC_patient055` | `ACDC_patient118` | train | train↔test leak; protect test |
| `MNMs_A7G0P5` | `MNMs_K3R0Y7` | train | train↔val leak; protect val |
| `MNMs_C8J7L5` | `MNMs_C8O0P2` | **val** | val↔test leak; test is the more precious |
| `ACDC_patient076` | `ACDC_patient074` | train | 2× weight only, no leak |
| `MNMs_Q0Q1Y4` | `MNMs_A8C9H8` | train | 2× weight only, no leak |
| `MNMs_E9L4N2` | `MNMs_C5Q2Y5` | train | 2× weight only, no leak |

### Live cohort (supersedes §13's table)

| source | train | val | test |
|---|---|---|---|
| CMRxRecon2023 | 156 | 19 | 20 |
| CMRxRecon2024 | 235 | 29 | 30 |
| CMRxRecon2025 | 289 | 37 | 33 |
| ACDC | **83** | 15 | 50 |
| M&Ms-1 | **172** | **33** | 136 |
| **total** | **935** | **133** | **269** |

`935 + 133 + 269 = 1337`. Ratio **69.9 / 9.9 / 20.1** — still within the 7:1:2 rule, so **no
re-split was needed**.

### Why the exclusion is applied AFTER assignment (important)

`tools/build_pooled_split.py`'s assigners draw from `random.Random(42)`, and the draws depend on
**list lengths**. Filtering the 6 rows out of the manifest *before* assignment would have changed
every shuffle and reshuffled the entire pool, invalidating comparisons with anything already run.
So the script assigns all 1343 rows exactly as before, then marks the 6 as
`split=excluded_duplicate` and omits them from `pooled.txt`. **Verified:** the regenerated
`pooled.txt` differs from the previous one by *exactly* those 6 deleted lines — zero subjects moved
between splits — and `manifest.csv` still has 1343 rows with exactly 6 changed.

The manifest rows are **kept** (marked, not deleted) so the provenance survives, and `pooled.txt`
carries a header naming each excluded subject and its twin. **Do not "restore" them.**

### Recurrence guard

`tools/build_pooled_split.py --check-duplicates` re-derives the duplicate set from pixel content
(16×16×8 mean-pooled `frame_00`, L2-normalized, all-pairs cosine). **Run over all 1343 subjects it
found exactly these 6 pairs and no others**, at cosine 1.0000, with the next-closest unrelated pair
at **0.9655** — so the 0.999 threshold sits in the gap. (0.9655 is far tighter than the 0.6724
docs/59 F3 quoted for its own differently-built thumbnail; mean-pooled cardiac MR is globally
self-similar. Clean, but do not loosen the threshold.)
It **hard-fails** on any pair not already listed in `DUPLICATE_PAIRS`, so a new duplicate must be
triaged by a human rather than silently absorbed.

### Knock-on config edits

`limit_train_batches: 940 → 935`, `limit_val_batches: 268 → 266` (= 2 × 133 for the ef sweep),
`logging.log_visual_frequency.train: → 935`. `tools/gate_native_z_identity.py` had `MNMs_E9L4N2`
hardcoded in its subject list; swapped for its surviving twin `MNMs_C5Q2Y5`.

### Caveat this exposes

The duplicated M&Ms subjects carry **contradictory metadata**: `MNMs_A7G0P5` is labelled
Philips/centre-2 while `MNMs_K3R0Y7` is GE/centre-4 — for identical pixels; `C8J7L5` is F and
`C8O0P2` is M, same age, same pixels. At least one vendor label per pair is demonstrably wrong,
which weakens §2.1/§13's "respecting M&Ms' official split buys a clean unseen-vendor (Canon) test"
argument. Not acted on — recorded so it isn't rediscovered.

## 14. Config switched to the pooled split (2026-07-31)

`training/config/mri_finetune.yaml` (inherited by `mri_volume.yaml` and every `mri_volume_*`
variant) now points at the pooled cohort. Four coupled edits, all in the parent config:

| key | was | now | why |
|---|---|---|---|
| `data_root` | `.../scratch/data/CMRxRecon2024/Cine_combined` | `.../scratch/data` | split-file lines are relative paths under the shared parent (`CMRxRecon2024/Cine_combined/…`, `ACDC_sax/…`); `_find_subjects` joins them directly — **no loader change**, as §2.1 predicted |
| `split_file` | `training/splits/random_8_1_1.txt` | `training/splits/pooled.txt` | the deprecated CMRx2024-only split → the 940/134/269 pooled split |
| `limit_val_batches` | 200 | **268** | 134 val subjects × {ED, ES} = one full deterministic `ef_val_sweep` pass; at 200 the last ~68 entries were unreachable |
| `dataset_name` | `${basename:${data_root}}` | `"pooled1343"` | the resolver would now yield the useless `"data"`, and it feeds `exp_name` → log dir → wandb run name. `exp_name` becomes `<rev_ts>_mri_volume_dynamic_axial_pooled1343` |

Plus one new key: **`cardiac_phase_csv`**, defaulted to
`scratch/data/whs/cardiac_phase.csv` and threaded into the val `MRIDataset`. This was previously
implicit — `MRIDataset` falls back to `${data_root}/../../whs/cardiac_phase.csv`, which was correct
*only* because `data_root` used to sit two levels deeper. With the new `data_root` that fallback
resolves to `scratch/whs/…` and would have failed, so the path is now explicit rather than derived.

**Verified**: Hydra `compose(config_name="mri_volume")` resolves all five keys as above, and a real
`MRIDataset(...)` constructed against the new `data_root`/`split_file` returns exactly **940 / 134 /
269** subjects for train/val/test. `pytest tests/` green.

~~**⚠️ Blocker for an actual run, not introduced here:** `mri_volume.yaml` sets `ef_val_sweep: true`,
and `scratch/data/whs/cardiac_phase.csv` **does not exist yet**...~~ **RESOLVED (2026-07-31), see
§8.4.** `cardiac_phase.csv` is built (1514 units); the real val `MRIDataset` now constructs cleanly
under this config with `ef_val_sweep: true`, yielding **134 subjects → 268 (subject, t_target)
pairs** — exactly `limit_val_batches: 268`, i.e. one full deterministic ED/ES pass with nothing
truncated. Re-verified **after** the §10b slice-order flip.

**The slice-order flip does not invalidate `cardiac_phase.csv`**: every column is either a *time*
index (`ED`, `ES`, `T`), a *volume* (`EDV_mL`, `ESV_mL`, `EF_pct` — voxel sums, invariant under a
z-axis reversal), or metadata. Nothing in it is slice-index dependent.
