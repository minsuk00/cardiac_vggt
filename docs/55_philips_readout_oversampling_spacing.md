# 55 — The readout-oversampling spacing rule, and the Philips `pixel_x` fix

> **TL;DR & takeaway**
>
> `reconstruct_subject` derives in-plane spacing as **`FOVx / ReconMatrix_X`**. That is the pixel size
> only if `ReconMatrix_X` counts **acquired readout samples**. It does not — it is an **output grid
> size**. For every Siemens subject the two coincide (`nx = 2·rx`, `ReadOutOversample = 2`, so
> `base = nx/2 = rx`), which is why the formula was right for all 490 Siemens volumes across three
> years. **CMRxRecon2025 Philips acquires `nx = 304` and reconstructs onto `rx = 256`, so
> `base = 152 ≠ 256` and the stamped `1.168 mm` under-scales the readout axis by `256/152 = 1.684`.**
>
> **Fix applied 2026-07-27** (`tools/fix_philips_pixel_x.py --apply`): `pixel_x` 1.168 → **1.967 mm**
> on the 12 Philips subjects, 156 files, header-only, atomic, fully reversible via `--revert`
> (sidecar `_provenance/philips_pixel_x_fix.json`). **`pixel_y` is NOT touched** — it was already
> correct. Verified: 156/156 files at the new zooms, LPS preserved, every 3d frame still
> voxel-identical to its 4d slice, 0 stray temporaries.
>
> **This supersedes the "Philips is stretched along the phase axis; relabel `pixel_y` 1.168 → 0.689"
> diagnosis** in `CMRxRecon2025/README.md`. That got the **aspect ratio right (~1.69) and the axis and
> direction wrong**: nothing needed shrinking in y; x needed growing. Its fix would have produced a
> ~180 mm thorax.
>
> **The general rule, one formula for all three years:** `pixel_x = FOV_spanning_the_full_readout / nx`
> (x is *cropped*, so pixel size is preserved) and `pixel_y = FOVy / ReconMatrix_Y` (y is
> *zero-filled*, so FOV is preserved). It reproduces every currently-stamped value except Philips.
>
> **The driver is patched too** (`tools/reconstruct_cmrx2025.py`, Philips-only exception), so a
> re-reconstruction will not revert it: 348 of 360 subjects bit-identical, 12 Philips changed.
>
> ⚠️ **Quantitative confirmation is still OWED.** The evidence is arithmetic plus visual; three
> successive hand-rolled LV detectors failed their own controls during this work and none of their
> numbers should be quoted. **nnU-Net Task114** is the outstanding check.

---

## 1. What the dataset actually ships

`cine_sax_info.csv` gives `FOVx, FOVy, ReconMatrix_X/Y, ReadOutOversample, SliceThickness, SliceNum` —
and the k-space array `(nt, nz, nc, ny, nx)`. **It never ships a voxel size.** `pixel_x` is *derived*
by our recon, so this is a fix to a derivation, not an override of shipped data.

## 2. The defect

```python
recon_x = int(meta["ReconMatrix_X"])          # an OUTPUT grid size
spacing_x = float(meta["FOVx"]) / recon_x     # correct only if recon_x counts ACQUIRED samples
```

Readout oversampling (declared: `ReadOutOversample = 2` for **every** vendor and every year) means the
scanner digitises `nx = 2 × base` samples spanning twice the nominal FOV — free along the readout axis,
and it keeps out-of-FOV signal from wrapping in. The recon then **centre-crops the image** back down
(`start_x = (nx - recon_x)//2`, applied after the iFFT).

**Cropping preserves pixel size.** So `pixel_x` is fixed by the acquisition, `FOVx_full / nx`, and has
nothing to do with how many pixels you keep.

| | `nx/rx` | `base = nx/ROos` | vs `rx` | `FOVx/rx` correct? |
|---|---|---|---|---|
| 2023 Siemens (196) | 2.00 | 256 | **equal** | ✅ |
| 2024 Siemens (294) | 2.00 | 256 | **equal** | ✅ |
| 2025 Siemens (169) | ~2.00 | ≈ rx | **equal** | ✅ |
| 2025 UIH (179) | 1.33 | 224 vs 336 | ≠ | ✅ — the driver's UIH branch already computes `FOVx/nx`, which equals `FOVx_nominal/base` |
| **2025 Philips (12)** | **1.19** | **152 vs 256** | ≠ | ❌ **off by 1.684×** |

## 3. The y axis was never wrong

The two in-plane axes undergo **different** operations and must not be treated alike:

- **x is cropped** (304 → 256) ⇒ pixel size preserved ⇒ `pixel_x = FOVx_full / nx`.
- **y is zero-filled** (`ny` ≈ 108–120 → `ry` = 256) ⇒ **FOV** preserved, pixel gets *finer* ⇒
  `pixel_y = FOVy / ReconMatrix_Y` — which is exactly what the code already does.

So the corrected Philips grid is **anisotropic 1.967 × 1.168 mm**, whose *acquired* voxel is isotropic
at 1.967 mm (`FOVy / ky_full = 299/152`, since the full ky extent `ny/pf_frac` measures 147–157 ≈ 152 =
`base`). **This is precisely the pattern the UIH volumes already carry** — acquired 1.607 × 1.62
isotropic, stored 1.607 × 1.074 anisotropic. Philips is not a special case; it is the same case.

## 4. Which FOV convention — and the discriminator is measured, not invented

`ReadOutOversample` gives the *factor*, not the *convention*: it cannot say whether the quoted `FOVx`
spans the base matrix (nominal, post-crop) or the full oversampled grid. Measured over all 405 source
CSVs:

| scanner | n | `FOVx` | `FOVy` | `FOVx/FOVy` | `(FOVx/2)/FOVy` | convention |
|---|---|---|---|---|---|---|
| Philips_30T_IngeniaCX | 12 | 299 | 299 | **1.000** | 0.500 | nominal |
| Siemens_30T_CIMAX | 21 | 352 | 349 | **1.009** | 0.505 | nominal |
| Siemens_30T_Vida | 36 | 358 | 345 | **1.040** | 0.520 | nominal |
| Siemens_15T_Aera | 49 | 340 | 276 | **1.231** | 0.615 | nominal |
| Siemens_30T_Prisma | 78 | 400 | 322 | **1.243** | 0.622 | nominal |
| **Siemens_15T_Avanto** | 3 | **720** | 300 | 2.399 | **1.200** | **oversampled** |
| **Siemens_15T_Sola** | 14 | **760** | 289 | 2.627 | **1.314** | **oversampled** |
| UIH_30T_umr880 | 37 | 720 | 448 | 1.607 | **0.804** | oversampled |
| UIH_15T_umr670/680 | 61 | 720 | 368 | 1.957 | **0.978** | oversampled |
| UIH_30T_umr780/790 | 94 | 720 | 320 | 2.250 | **1.125** | oversampled |

🔴 **The convention is NOT a vendor property.** Siemens **Sola** and **Avanto** quote `FOVx`
oversampled, exactly like UIH. The driver's `scanner.startswith("UIH")` branch is therefore wrong as a
general rule — it is *accidentally harmless* for Sola/Avanto only because they have `nx = rx`, where
`FOVx/rx ≡ (FOVx/2)/(nx/2)` identically.

### 4a. 🔴 The proposed "nearer to square" decision rule FAILS per subject — do NOT implement it

The table above is per-scanner **medians**. Applied **per subject** across all 360 and compared against
what the driver stamped:

| outcome | n |
|---|---|
| reproduced within 2% | **333** |
| small deviation 2–5% | 1 |
| **factor-of-2 convention FLIP** | **14** — Prisma 11, CIMAX 1, Vida 1, umr880 1 |
| Philips (the intended change) | 12 |

The 14 failures are subjects with a genuinely **rectangular prescribed FOV** — `FOVx/FOVy` of
**1.44–1.73**, past the √2 = 1.414 boundary — e.g. `Prisma_P012` at 400 × 230.8 mm (1.733). A wider
readout FOV than phase FOV is an entirely ordinary prescription (it costs no scan time), so the
"roughly square" premise is simply false for them, and the rule mispicks **silently, by 2×**.

**The driver's current `scanner.startswith("UIH")` branch gets all 14 right**, because they are Siemens
and it treats every Siemens subject as nominal. So the proposed "improvement" would have *introduced*
14 new errors while fixing 12. **It must not be implemented as stated.**

### 4b. How much of the Philips fix is derived, and how much is inferred

Stated precisely, because an earlier draft of this doc over-claimed ("does not depend on the picker"):

1. **That the shipped `1.168 mm` is wrong — DERIVED.** `pixel = FOV_full / nx`. `ReconMatrix_X` is an
   *output grid size*, and `reconstruct_subject` **crops** the image rather than resampling it (read in
   the source), so the pixel size is fixed by the acquisition. `299/256` uses a denominator that is
   neither `nx` nor `base`; it is wrong under **either** FOV convention. No anatomical prior is used.
2. **That the replacement is `1.967` and not `0.983` — INFERRED.** `FOVx` nominal ⇒ `598/304 = 1.967`;
   `FOVx` oversampled ⇒ `299/304 = 0.983`. Both are arithmetically self-consistent. They are separated
   only by anatomy: `0.983` implies a **~150 mm thorax**. A coarse and safe prior, but a prior.

**No metadata-only discriminator has been found.** `nx/rx` does **not** work: Philips is *nominal* at
1.19 while UIH umr780 is *oversampled* at 1.33 — adjacent values on opposite sides. The driver's
`startswith("UIH")` branch is therefore an **empirical vendor lookup that happens to be right for all
348 non-Philips subjects we hold**, not a derivation. Settling it properly needs vendor documentation,
a DICOM export, or the organisers.

⚠️ An earlier draft of this rule used an invented absolute window ("a cardiac FOV is 250–500 mm").
That number had no source and must not be used either.

## 5. Evidence

**Arithmetic (does not depend on segmentation):** `nx = 304`, `ReadOutOversample = 2` ⇒ base = **152**;
full ky extent `ny/pf_frac` = **147–157**; so the acquisition is ~152 × 152 over a 299 × 299 mm FOV —
an ordinary isotropic ~1.97 mm cine prescription.

**Physical scale:** at the shipped 1.168 mm the whole image box is 299 mm wide and the body fills ~60%
of it ⇒ a **~180 mm thorax**, against 300–400 mm for an adult and ~313 mm for the CMRx24 reference.
Rescaling by 1.684 puts every one of the 12 in the normal range
(`result/cmrx2025_recon_check/philips_cohort_scale_all.png`).

**Aspect ratio:** the user's independent visual reading — that `FOVy = 176` (⇒ aspect 1.695) gives the
roundest LV — matches the derived 1.684 to within rounding. Both routes give the same *shape*; only the
scale route differs, and only raising `pixel_x` yields a plausible body
(`philips_three_hypotheses.png`).

**Figures:** `philips_before_after.png`, `philips_cohort_scale.png`, `philips_cohort_scale_all.png`,
`philips_three_hypotheses.png`, `physical_scale_montage.png`, `uih_fov_convention_check.png`.

## 6. 🔴 Three of my own measurement scripts failed their own controls

Recorded because the failures were caught only by controls, never by reading the code:

1. `render_uih_fov_check.py` v1 applied the UIH `×nx/rx` rescale to **every** vendor, so the Siemens
   "control" panels showed a stretch that cannot occur. Caught by the built-in
   identical-by-construction control.
2. The blood-pool detector reported **26 mm** for a known-good Siemens LV (true ~45 mm) — so none of its
   diameters (the "23–30 → 39–50 mm" figures) are usable.
3. `render_philips_before_after.py` re-detected the blob *separately per hypothesis* instead of
   measuring one fixed region at two scales, so its LV numbers were incomparable across columns.

Consequence: **every LV-diameter number produced during this investigation is withdrawn.** The
surviving evidence is arithmetic (§5) and visual comparison at a shared millimetre scale. Reinforces
memory `feedback-fault-inject-verifiers`.

## 7. What was changed

- **`tools/fix_philips_pixel_x.py`** — new. Header-only rescale of axis-0 spacing; keeps the volume
  centre fixed; atomic (`tmp + os.replace`, tmp retains `.nii.gz` so nibabel infers the format);
  records every original in `_provenance/philips_pixel_x_fix.json`; `--revert` restores.
- **12 Philips subjects, 156 files**: `pixel_x` 1.168 → 1.967 mm (1.1719 → 1.9737 for P003/P013).
  Verified 156/156 at the new zooms, LPS, 3d ≡ 4d, 0 strays.
- **The staged CSV and `recon_report.json` move with the header — three places, not one.** The first
  attempt changed only the NIfTIs and was therefore *incomplete*; it was reverted and redone. Why each
  matters:
  - `Cine_combined/<cid>/sax/cine_sax_info.csv` is **not** a copy of the source — `normalize()`
    pre-bakes the intended pixel size into it (`FOVx = pixel_x · rx`), which is why UIH carries
    `540` where its source says `720`. Philips' staged `FOVx` is now **503.578947** (= 1.9671 × 256),
    where the source says 299.
  - **`tools/verify_recon_v2.py` re-derives expected in-plane spacing as `FOV/ReconMatrix` from that
    staged CSV** (`:191-195`). Had the CSV been left alone, all 12 Philips subjects would have failed
    its `inplane` check — a self-inflicted regression in the standing verifier.
  - `recon_report.json` `pixel_mm` / `in_plane_aniso` updated, with the superseded values kept as
    `pixel_mm_assumed` and `pixel_x_source`. Precedent: `docs/54` §10c did the same for the Prisma
    pitch relabel. Backup at `recon_report.json.bak_prephilips`.
  - Post-fix consistency check: **12/12 subjects have header == staged CSV == report**.
- **Philips is no longer excluded** from the (not-yet-built) 2025 splits. Note this was never encoded
  anywhere: all four `training/splits/*.txt` are 2024-only and `--exclude-scanner` defaults to empty,
  so the exclusion existed solely as prose in `CMRxRecon2025/README.md`, now updated.

## 8. What is still owed

1. **nnU-Net Task114 LV segmentation** on the 12 Philips + a Siemens/UIH reference group, across **all
   18 slices** — confirms scale *and* aspect quantitatively, and separates genuine basal-slice
   ellipticity from geometry. Until then §7 rests on §5.
2. ✅ **DONE — the driver is patched** (`tools/reconstruct_cmrx2025.py`, "PHILIPS-ONLY EXCEPTION"), so a
   re-reconstruction no longer reverts the fix. Scoped to `scanner == "Philips_30T_IngeniaCX"`.
   **Verified two ways:** replaying the old and new expressions over all 360 subjects gives
   **348 identical / 12 changed (all Philips, to 1.9671 and 1.9737 — exactly the on-disk headers)**;
   and a real `normalize()` run per scanner gives Philips `FOVx = 503.578947` → 1.9671 while Sola
   (2.2500), UIH umr780 (1.6071) and Vida (1.4062) are unchanged.

   ⚠️ **Why it is scoped to one scanner instead of generalised — the "obvious" patch is UNSAFE and was
   measured to be so.** Swapping the
   non-UIH denominator to `base = nx / ReadOutOversample` while keeping the `startswith("UIH")` branch
   gives, over the 2025 non-UIH subjects:

   | group | `2·rx/nx` | effect |
   |---|---|---|
   | Siemens with `nx = 2·rx` exactly (majority) | 1.0000 | none |
   | Aera / CIMAX / Prisma / Vida, `nx` 2–6 samples off `2·rx` | 0.9897–1.0136 | 0.4–1.4%, both directions |
   | **Siemens_15T_Sola** (`nx = rx`) | **2.0000** | 🔴 **pixel size DOUBLED** |

   Sola belongs to the **oversampled** convention group (`FOVx` 720/760) but the vendor branch
   classifies it as nominal, so `base = nx/2` halves its denominator and doubles its spacing. **The
   driver patch is therefore blocked on the same unresolved question as §4b** — there is no
   metadata-only way to classify Sola correctly.

   The only safe minimal change is a **Philips-only exception** keyed on
   `scanner == "Philips_30T_IngeniaCX"` (precedented by the existing UIH exception), *not* a general
   rule change. The 0.4–1.4% Siemens ambiguity (does `FOVx` span the acquired base 193 or the recon
   matrix 192?) is genuinely unresolved but is ~0.014 mm on a 1.4 mm canonical grid — leave it.
3. **Philips pitch.** Left at 10 mm (6 mm thickness + the 4 mm gap rule that `docs/54` §10c confirmed
   holds everywhere). 18 × 10 = 180 mm is large but, per §10c's Prisma finding, plausibly genuine
   over-coverage rather than an error. Not changed here.
4. A handful of 2025 Siemens subjects have `nx/rx = 2.03` rather than exactly 2.00; a strict
   `base = nx/2` would shift them ~1.5%. Not acted on.

## Sources

- `scratch/data/CMRxRecon2024/recon_code/batch_reconstruct_cmrxrecon2024.py` — the `FOVx/ReconMatrix_X`
  derivation and the post-iFFT centre crop (read directly).
- `tools/reconstruct_cmrx2025.py` — the `startswith("UIH")` FOV branch and the ky fill.
- All 405 CMRxRecon2025 source `cine_sax_info.csv` (§4 table), and `recon_report.json`.
- `docs/54` §10c — the measured 2025 pitch, which refutes the contiguous-slice hypothesis floated
  during this investigation and establishes that the +4 mm gap rule holds.
