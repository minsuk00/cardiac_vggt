# 56 — Every CMRxRecon challenge release rolls odd-Z SAX stacks by −1 slice

> **TL;DR & takeaway**
>
> **In all three CMRxRecon *challenge* releases (2023, 2024, 2025), any SAX stack with an
> ODD number of slices is cyclically rolled by −1: the most basal slice is stored LAST,
> after the apex.** Even-Z stacks are fine. The rule is essentially deterministic —
> **464/466 odd-Z subjects are rolled, 0/383 even-Z subjects are.**
>
> **Mechanism (inferred, §6):** an **unpaired `fftshift` on an axis that is not being
> transformed** — `fftshift`/`ifftshift` differ by a roll of −1 for odd N and are *identical*
> for even N. Parity-dependence is what makes this diagnostic: an index off-by-one would hit
> even Z too, a de-interleave would scramble, a flip would reverse. Verified prediction: of the
> axes the recon does not transform, `nslice` is the **only** one that is ever odd — and the
> LAX **view** axis (length 3, always odd) is rolled by −1 in **7/7** subjects.
>
> **This is in the shipped k-space, not our reconstruction** — three independent legs (§5): the
> recon code has no slice-axis operation at all; an independent RSS straight from
> `cine_sax.mat` reproduces it; and the organizers' own *different* recon of the *same* people
> (`CMRxRecon-300`) is clean (0/129 odd-Z). Controls: ACDC 0/54 odd-Z, M&Ms-2 1/122 odd-Z.
>
> **Impact: 464 of our 849 live subjects (54.7%) — including 183/294 (62%) of the
> CMRxRecon2024 set the model currently trains on — carry one slice placed ~(Z−1)×pitch
> ≈ 10–14 cm from where it belongs**, in every one of the 12 cardiac phases. The canonical
> preprocessing preserves slice order, so both `V_gt` and the input slices are affected.
>
> **The LAX view axis (length 3 = always odd) is rolled too — 7/7 subjects vs CMRx-300.** So the
> challenge LAX views are *not* in the documented `3ch, 2ch, 4ch` order, which matters because
> both slice-pitch measurements read LAX (§6a).
>
> **Fix APPLIED 2026-07-27** (§7): `np.roll(vol, +1, axis=2)` on the **464 detected** subjects,
> 6032 SAX NIfTIs / 21.8 GB rewritten in place, affine untouched; re-detection afterwards reports
> **0 rolled at both parities**. Reversible (`--revert`); full 849-row audit in each dataset's
> `_provenance/slice_roll_fix.json` and `result/slice_roll_check/slice_roll_audit_applied.csv`.
>
> Detector: `tools/probe_slice_roll.py`. Figures: `result/slice_roll_check/`.

---

## 1. The question

Raised 2026-07-27: *"for volume, the slices might be shifted circularly — top slice supposed
to be at bottom? Compare 0th slice vs top slice vs top-1 slice."*

The 2023 README already recorded a *relative* version of this (`docs`-adjacent: "🔴 TRAP — the
two releases differ by a CYCLIC SLICE ROLL of +1 for some subjects", roll=0 for 101 / roll=+1
for 95 subjects, measured against CMRxRecon-300). It was written off as a **packaging artifact
with "no impact on the geometry borrow"**, because only per-volume scalars were being
transferred. That framing missed the real question: **which release is right?** This doc
answers it — and the answer is that *ours* is the wrong one.

## 2. Method — two independent estimators, no reference release needed

The trick is that a correct SAX stack has exactly **one** discontinuity: the wrap-around
pair (apex → base). Everything else is a smoothly varying neighbour.

1. **Local (`k_adjacent`).** Compute Pearson r for all `Z` *cyclically* adjacent slice pairs
   `(z, (z+1) mod Z)`. Take the argmin. If it is at `Z−1` (the wrap), the order is correct;
   otherwise the implied roll is `k = Z−1−argmin`.
2. **Global (`k_global`).** Cannot be fooled by a single odd slice. Similarity must decay
   monotonically with `|z_i − z_j|`. For each candidate roll `k`, relabel positions
   `t_i = (i+k) mod Z` and score the ordering by Spearman corr between `|t_i − t_j|` and
   `C[i,j]` over **all** pairs; the best `k` is the most negative.

Both run on `sax/3d_recon/sax_frame_00.nii.gz`. `tools/probe_slice_roll.py`.

## 3. Result

**The two estimators agree on 849/849 subjects.** The implied roll is only ever 0 or 1 —
never anything else. Margins are large (local: 2nd-min − min, median 0.47; global: median
0.32), so no subject is a close call.

| | even Z | odd Z |
|---|---|---|
| CMRxRecon2023 (challenge) | 0 / 101 | **94 / 95** |
| CMRxRecon2024 (challenge) | 0 / 111 | **183 / 183** |
| CMRxRecon2025 (challenge) | 0 / 171 | **187 / 188** |
| **total** | **0 / 383** | **464 / 466 (99.6%)** |

Per-Z, it is exact — e.g. 2024: `Z=9: 65/65`, `Z=11: 118/118`, `Z=10: 0/89`, `Z=12: 0/10`.

**Vendor, centre and year have zero explanatory power once Z parity is accounted for.** The
2025 groups that looked immune (Philips 0/12, Sola 0/7, Center007 0/9) are simply all even-Z.

## 4. Controls — the metric is not biased toward flagging the last pair

| dataset | even Z | odd Z |
|---|---|---|
| ACDC | 0 / 95 | **0 / 54** |
| M&Ms-2 (SA_ED) | 0 / 238 | **1 / 122** |
| **CMRxRecon-300 (paper release, same 300 people as 2023)** | 0 / 140 | **0 / 129** |

The controls carry 305 odd-Z subjects between them and flag 1. The CMRxRecon-300 row is the
decisive one: **same volunteers, same scanner, same protocol, different release** — and it is
clean. So the ordering in the paper release is right and the **challenge repackaging is what
rolls the stack**.

(`CMRxRecon-300 TrainingSet/P073` and `CMRx23_Train_P073` are the *same* anomalous subject in
both releases — mostly *negative* adjacent correlations, no coherent ordering at all. That is
a separate per-subject data-quality problem, not a roll. Worth a look before it is used.)

## 5. It is in the shipped k-space, not our recon — three independent legs

**(a) The recon code cannot do it.** `recon_code/batch_reconstruct_cmrxrecon2024.py` (the same
`reconstruct_subject` used for 2023 and 2025):

```python
for slc in range(nslice):
    ...
    slice_image_gpu = sp.ifft(slice_kspace_gpu, axes=[-2, -1])   # in-plane ONLY
    ...
    all_recons[frame, slc] = img_cropped                          # slc written straight through
```

then `sitk.GetImageFromArray(all_recons[frame])`, whose first axis is z. There is no transform,
permutation, flip or roll on the slice axis anywhere in the path. NIfTI z index ≡ `.mat` slice
index, by construction.

**(b) An independent RSS from the `.mat` shows the roll.** No ESPIRiT, no SENSE, no NIfTI, no
SimpleITK — four lines, with `ifftshift`/`ifft2`/`fftshift` restricted to `axes=(-2,-1)`, so
nothing in it *can* permute the slice axis. It reproduces the **identical argmin** as the
shipped NIfTI on 6 subjects across all three years (e.g. `CMRx24_Test_P002` Z=11: k-space RSS
argmin 9, NIfTI argmin 9). If our recon were the culprit, this would have come out clean.

**(c) A different recon of the same people is clean.** `CMRxRecon-300` was reconstructed by the
organizers with a different algorithm (iterative CS-SENSE on R=3 data), and shows **0/129**
odd-Z rolls. That rules out ESPIRiT/SENSE, sigpy, cupy, and our geometry handling as sources.

Also **constant across all 12 cardiac phases** (checked frame-by-frame on 9 subjects) —
consistent with a slice-axis indexing artifact, not anything temporal.

⚠️ **An attempted FFT-free confirmation was inconclusive and is not evidence.** Correlating
`log|k-space|` between cyclically adjacent slices — no transform of any kind — matched the
image-domain argmin on only 3 of 6 subjects. The reason is benign: the k-space magnitude
envelope is nearly slice-invariant, so the spread between adjacent-pair correlations is ~0.05
versus ~0.47 in image space. The test simply has no power; it neither supports nor contradicts.
Leg (b) already carries the point, because its FFT is restricted to the in-plane axes.

## 6. Mechanism — an unpaired `fftshift` on a non-transformed axis

For a length-`N` axis, `fftshift` and `ifftshift` differ by a cyclic roll of:

```
N= 9 (odd)  -> 8  ==  -1 (mod 9)      N=10 (even) -> 0
N=11 (odd)  -> 10 ==  -1 (mod 11)     N=12 (even) -> 0
N=13 (odd)  -> 12 ==  -1 (mod 13)     N=14 (even) -> 0
```

**Odd ⇒ off by exactly −1; even ⇒ identical**, reproducing the observation including the
*direction*. Two variants give the identical signature and the data cannot separate them:
`fftshift` used where `ifftshift` was needed, or `fftshift` applied **twice** (e.g. the common
`fftshift(ifft2(fftshift(k)))` pattern, where `np.fft.fftshift(x)` with **no `axes=`
argument shifts every axis**, not just the two being transformed).

**Why the parity dependence is the informative part.** It rules out the obvious alternatives:

- an off-by-one **index** bug would roll even-Z stacks too — we see 0/383;
- a **de-interleave / sort** bug (Siemens acquires slices interleaved, and the interleave order
  *is* parity-dependent) would produce a scrambled permutation, not a clean roll of 1;
- a **flip** of the slice axis would reverse, not rotate.

A shift-type operation is essentially the only common one whose effect depends on axis-length
parity.

**A checkable prediction, verified.** The bug is only visible on axes that are *not* Fourier
transformed and happen to have odd length. Across all 840 readable `cine_sax.mat`:

| non-transformed axis | odd length in |
|---|---|
| `nframe` | 0 / 840 (always 12) |
| `ncoil` | 0 / 840 (always 10) |
| **`nslice`** | **465 / 840** |

So `nslice` is the *only* non-transformed axis that is ever odd — which is why SAX shows it and
nothing else does. (`ny` is odd in 60 of 840 and `nx` never, but those are the transformed axes,
where a shift is part of the intended transform and a 1-row offset is not separately detectable.)

### 6a. The prediction generalises — the LAX **view** axis is rolled too (7/7)

`cine_lax.mat` packs the 3 long-axis views (`3ch, 2ch, 4ch` per the organizers'
`Format_cine.txt`) in the slice dimension. **3 is always odd**, so the mechanism predicts every
subject's LAX should be rolled — no parity split. Tested against `CMRxRecon-300`'s
`cine_lax_ks.mat` (same subjects, different release), zero-filled RSS, 3×3 NCC:

| | mapping | matched NCC | off-diagonal |
|---|---|---|---|
| P001, P002, P004, P005, P006, P007, P008 | **roll −1 (1,2,0) — 7/7** | 0.575–0.686 | 0.12–0.36 |

Unambiguous and unanimous. A *different axis*, in a *different file*, rolled by the *same* −1,
with no parity split because the length is always odd. This is the strongest support for the
mechanism, and it is the reason to treat the story as more than a coincidence of signature.

⚠️ **Practical fallout: the challenge LAX views are not in the documented order.** If
`CMRxRecon-300` follows `Format_cine.txt` (`3ch, 2ch, 4ch`) then the challenge stores
`2ch, 4ch, 3ch`. **`tools/measure_cmrx2023_slice_pitch.py` and the 2025 pitch measurement both
read LAX to find the 4-chamber view** — if either selects a view by index, it read the wrong
one. **Not yet checked.** (What is *measured* here is that the two releases differ by −1 on
this axis; that CMRx-300 is the side matching the documentation is inferred from the SAX result,
not separately verified.)

### 6b. What is inference, not measurement

The organizers' preprocessing code is not public. The **observation** — odd-Z SAX stacks and all
LAX view stacks are rolled by −1 — is measured and stands on its own. The **mechanism** is an
inference from an exact signature match plus the verified odd-axis prediction; the specific line
of their code, and which of the two `fftshift` variants it was, is not recoverable from the data.

## 7. Impact and the fix — ✅ APPLIED 2026-07-27

**Impact.** 464 of the 849 live subjects have their most basal slice stored at the apical end
of the array — a displacement of `(Z−1) × pitch` ≈ **96–144 mm** for typical Z=9–13 at 12 mm
pitch. `training/data/preprocess.py` resamples along z but preserves slice order, so the
misplacement propagates into the canonical cube for **both** `V_gt` and the sampled input
slices, in every phase. For the currently-trained CMRxRecon2024 cohort that is **183/294
(62%) of subjects**. It also means the two ends of the stack are wrong for those subjects,
which is exactly where `anatomy_bbox` and z-coverage sampling operate.

### What was done

`tools/fix_slice_roll.py --apply` — **464 subjects / 6032 files / 21.8 GB rewritten in place**
(2023: 94, 2024: 183, 2025: 187), `np.roll(data, +1, axis=2)` on all 13 SAX NIfTIs per subject
(`sax_frame_00..11` + `4d_recon`), atomic (tmp + `os.replace`), under 10 minutes on 16 workers.
The other 385 subjects were never opened for writing and are byte-identical on disk.

**Verified after:** `--verify` and a fresh independent run of `probe_slice_roll.py` both report
**0 rolled across all three years at both parities** (2023 0/101 even + 0/95 odd; 2024 0/111 +
0/183; 2025 0/171 + 0/188), estimators still agreeing 849/849. Proven on copies first: rolled
subject → `equals np.roll(+1)`, pure z-permutation, affine/zooms unchanged, `3d frame00 ==
4d[...,0]`; clean subject byte-identical; `--revert` → byte-identical to original.

### The two decisions, and how they were settled

- **Detect per subject, never apply the parity rule blind.** Both estimators must agree on
  `k=1`. This is what protected `CMRx25_train_Center007_Siemens_30T_Prisma_P001` (odd Z=11 but
  genuinely correctly ordered — wrap is the minimum at 0.12, all other pairs 0.59–0.74;
  **visually confirmed fine by the user**) and `CMRx23_Train_P073`.
- **Affine left untouched.** Every subject's origin is `(0,0,0)` — CMRxRecon ships no slice
  position or orientation, and the recon writes the SimpleITK default. Rolling the voxels
  translates the stack by one pitch in a frame with no absolute meaning, and
  `Orientationd`/`Spacingd`/`ResizeWithPadOrCropd` never read the origin.
- **On disk, not at load time.** 66 scripts outside `training/` read this data directly; a
  load-time roll in `MRIDataset` would leave `baselines/`, `evaluation/`, `inference/` and every
  tool silently on the buggy volumes.

### Audit trail

- `<dataset>/_provenance/slice_roll_fix.json` per year — the rolled-subject list (`--revert`
  reads it), plus `odd_z_NOT_rolled`, `even_z_rolled_UNEXPECTED` (empty) and a per-subject row
  for **all 849**.
- `result/slice_roll_check/slice_roll_audit_applied.{json,csv}` — the same audit as one flat
  table: year, subject, Z, parity, action, both k estimates, `corr(last,first)`,
  `corr(last-1,last)`, margin.
- **Revert:** `python tools/fix_slice_roll.py --revert --apply`.

⚠️ **The monai cache must be cleared on any node that has a warm one** —
`PersistentDataset` keys on input *paths*, not contents, so it would silently serve pre-fix
volumes: `rm -rf /tmp/vggt-mri_$USER_monai_cache/`. (None existed on the node used here.)

⚠️ **Anything measured on the pre-fix tree is now a different series** — the v2
re-reconstruction verification JSONs, the `scratch/eval` harness, and `evaluation/results/`.

## 8. Loose ends

- **One genuine exception**: `CMRx25_train_Center007_Siemens_30T_Prisma_P001` (Z=11, odd, not
  rolled, margin 0.47). Unexplained. Detect per subject rather than applying the parity rule
  blind — the tool does exactly that.
- 🔴 **`CMRx23_Train_P073` — EXCLUDE from the cohort (decided 2026-07-27).** Diagnosed: the
  thorax is at a **different in-plane rotation in every one of its 13 slices**, so it is not a
  stack of parallel SAX planes and has no coherent through-plane geometry. Adjacent-slice
  correlations are mostly negative (`+0.74, -0.19, ... , +0.76`); no roll is definable (both
  estimators return `k=11`), so `fix_slice_roll.py` skipped rather than guessed; it is **not**
  interleaved (de-interleaving either way still leaves the min at −0.19); and the same defect is
  in `CMRxRecon-300 TrainingSet/P073` (`k=10`), so it is in the source data.
  **ARCHIVED 2026-07-27** (nothing deleted): `Cine_combined/CMRx23_Train_P073` →
  `_archive/CMRx23_Train_P073_slice_rotation_excluded/`, so no split or dataloader can pick it
  up. 2023 `Cine_combined/` now holds **195**, live 3-year cohort **848**. The 849-subject counts
  throughout this doc are the measurement as run and were deliberately not rewritten.
  Figure `result/slice_roll_check/P073_all_slices.png`.
- The 2023 README's "TRAP" section should be updated: the roll is not a harmless packaging
  difference, and the challenge release — the one we reconstructed from — is the wrong side
  of it.
- **LAX is rolled too (§6a) and the downstream check is still owed**: confirm which stored view
  index is the 4-chamber, then re-check `tools/measure_cmrx2023_slice_pitch.py` and the 2025
  pitch measurement (`docs/54` §10c, `tools/analyze_pitch_measurements.py`), which both read LAX
  to find it. If either selects a view by index it measured the wrong view — and the 2025 result
  relabeled 60 Prisma subjects 12 → 10 mm.

## 9. Repro

```bash
python tools/probe_slice_roll.py out.json          # all 3 years, ~3 min on 16 workers
python tools/probe_slice_roll.py out.json --glob '/path/to/*/sax_ED.nii.gz'   # controls
```

Figures: `result/slice_roll_check/slice_roll_summary.png` (worked example + all group rates),
`slice_roll_examples.png` (per-slice correlations for 6 stacks), `zoom_P002.png`.
