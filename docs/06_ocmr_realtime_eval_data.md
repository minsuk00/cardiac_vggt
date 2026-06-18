# 06 — OCMR real-time free-breathing data: reconstruction for transfer evaluation

> **TL;DR & takeaway** *(human-facing; the rest is the detailed record for agents)*
>
> **We turned OCMR's real real-time free-breathing short-axis k-space into usable image-domain cines
> so we can finally test the model *out of simulation*.** OCMR is the only verified-downloadable dataset
> with genuine real-time free-breathing SAX *stacks* (it's already on disk at `scratch/data/ocmr/`, 124 GB).
> From its 212 real-time (`us_*`) series we selected **47 SAX stacks → 23 with ≥12 slices → 17 non-FOV-aliased
> → 11 that initially passed recon QC → 10 confirmed clean** (1 volunteer + 9 patients). One stack
> (`us_0179`) passed the quick visual QC but was caught at the model-eval stage as a *milder* FOV-aliasing
> failure (ringing arcs, no clear LV) and moved to `_failed_fov_aliasing/`. Each kept stack is genuinely **R≈9 undersampled** real-time
> data, so a single frame is unreconstructable on its own; we reconstruct each slice's *whole cine* with a
> **k-t CS-SENSE** recon (ESPIRiT coil maps + PDHG temporal-TV, **λ=0.02**, GPU) → `recon/{subject}/sax_cine.nii.gz`
> (4-D `[W,H,slice,frame]`, ~292 MB total).
>
> **Three things future agents MUST understand:**
> 1. **`frame` is a real-time time index, NOT a gated cardiac phase, and frames are NOT aligned across slices.**
>    Each slice is its own ungated free-breathing clip; "frame 64 across slices" is a jumble of different
>    cardiac+respiratory states. Reconstructing a coherent volume from these scattered slices is the model's job.
> 2. **k-t recon = the *optimistic* eval.** It uses *many* frames/slice to clean each input — i.e. it leans on
>    the temporal info the one-frame-per-slice pitch is trying to *avoid*. So clean k-t inputs test
>    anatomy/contrast + cross-slice transfer, but are **cleaner than a true single-shot acquisition would give.**
>    A *faithful* one-frame test must feed **degraded single-shot frames** (no temporal recon) — not yet built.
> 3. **There is no clean 3-D reference** (the real-time data is *prospectively* undersampled → no ground truth).
>    Scoring is Tier-C (held-out-frame self-consistency) or Tier-B (reconstruct a reference from k-space) — never
>    a paired Tier-A answer key (see [[04_inference_information_contract]] and `_html/08`).
>
> **Status (2026-06-18):** **10 clean** subjects reconstructed + verified (us_0179 excluded as FOV-aliasing);
> recon code + outputs under `scratch/data/ocmr/`. **Model has now been run** on all 10 (z-only resp
> checkpoint) → qualitative beating-heart + predicted-DVF report `_html/13_ocmr_inference_results.html`
> (script `tools/eval_ocmr_inference.py`). **Not yet done:** the 6 long (197-frame) stacks (FOV-shift bug),
> the faithful single-shot degradation, and the FOV-shift fix that would also recover us_0179.

**Date:** 2026-06-18
**Status:** 10 OCMR real-time SAX cines reconstructed + model run (qualitative). See `_html/13`.
**Goal:** Provide real (not simulated) real-time free-breathing short-axis cine to evaluate the
gated→real-time transfer that the whole project hinges on and that is unmeasured anywhere
([[02_related_work_literature_review]], [[03_cardiac_svr_literature_review]]).

---

## 1. Why OCMR

The model trains *entirely on simulated* sparse sampling over gated breath-hold cine (CMRxRecon2024).
The headline risk is **gated→real-time-free-breathing transfer**, which no published work measures. To
test it we need *real* real-time free-breathing SAX data. The external-dataset sweep (`_html/08`,
[[project_eval_datasets_ocmr]]) found **OCMR** is the only verified-downloadable set with genuine
real-time free-breathing SAX **stacks** + raw k-space — and it was already fully downloaded on disk.

## 2. What OCMR is

- **Open-access multi-coil raw k-space cardiac cine** dataset (OSU; Chen & Ahmad, arXiv 2008.03410).
  ISMRMRD `.h5` (raw k-space, *not* images). bSSFP cine. 0.55T / 1.5T / 3T Siemens.
- **On disk:** `scratch/data/ocmr/` (= GPFS `.../minsukc/vggt/data/ocmr`), **124 GB, 377 `.h5`** =
  **165 fully-sampled gated (`fs_*`)** + **212 prospectively-undersampled real-time (`us_*`/`pse`)**.
  Per-file metadata: `ocmr_data_attributes.csv` (fetch from S3; columns smp/viw/sli/dur/sub/scn/slices).
- **Two regimes (do not conflate):**
  - `fs_*` = ECG-gated, breath-hold, fully sampled → reconstruct a clean image with a simple IFFT+SENSE
    *combine* (no inverse problem). This is the *opposite* of what we want (it's our training domain).
  - `us_*`/`pse` = **real-time, non-gated, free-breathing**, prospectively undersampled (R≈8–9). **No
    ground-truth exists** for these (the missing k-space was never acquired). This is the target domain.
- **Why a single real-time frame can't be reconstructed alone:** each frame acquires only ~13–18 of the
  phase-encode lines (R≈9) using **VISTA** variable-density sampling (center oversampled, periphery
  rotated). Frames sample *complementary* lines on purpose, so combining neighbors fills k-space — but a
  *lone* frame is hopelessly aliased (spatial-only parallel imaging / CS fails at R≈9). You must use the
  temporal neighborhood (k-t recon) or accept a degraded single shot.

## 3. Subject selection — the funnel to 10

Filters on `ocmr_data_attributes.csv` + geometry/recon checks:

| Step | Criterion | Count |
|---|---|---|
| Real-time SAX stacks | `smp=pse` & `viw=sax` & `sli=stk` | 47 |
| Enough z-coverage | `slices ≥ 12` | 23 |
| Not FOV-aliased | `fov ≠ ali` (OCMR's wrap-around flag) | 17 |
| Recon passes quick QC | k-t recon looks like a clean heart | 11 |
| **Confirmed clean at model-eval** | no residual FOV-aliasing on close look | **10** |

- **The 10 usable subjects** (1 volunteer + 9 patients, all 1.5T Sola/Avanto):
  `us_0084` (volunteer, 128 fr), `us_0169/0170/0171/0172` (~30 fr), `us_0173/0174` (63 fr),
  `us_0175` (31 fr), `us_0183` (61 fr), `us_0197` (29 fr).
- **`us_0179` (31 fr) — excluded.** Passed the quick visual QC but on close inspection (at the model-eval
  stage) shows the same **FOV-aliasing** signature as the long stacks below — ringing arcs at the bottom,
  a bright surface-coil blob, no clear LV, intensity p99 ~5× the others. Same root cause (FOV-shift), just
  milder. Moved to `_failed_fov_aliasing/`; recon kept for the record.
- **The 6 long stacks that FAIL** (`us_0198/0199/0200/0204/0207/0211`, all 197–200 fr): a **half-FOV wrap**
  in the time-averaged image corrupts the ESPIRiT coil maps → striped recon. Ruled out: λ, PDHG step-size,
  temporal window, k-space coverage (100%), header phase-encode center (=eNy//2). Root cause is an
  **FOV-shift / off-isocenter offset** these acquisitions carry that the recon doesn't correct — TODO
  (the same fix would also recover us_0179).
- **Curated symlink dir:** `scratch/data/ocmr/eval_rt_sax_stacks/` holds the 10 good symlinks + a
  `README.txt` + `_failed_long_fovshift/` (the 6) + `_failed_fov_aliasing/` (us_0179), kept for the record.
  Geometry of the 10: in-plane **2.16–2.84 mm**, slice **8 mm thick / ~10 mm spacing** (true center-to-center
  from `meta.json` positions, NOT the 8 mm thickness), **~110 mm extent (covers the heart)**, mostly
  patients. NOTE the geometry varies per subject — normalized later by the canonical preprocess.

## 4. The reconstruction — k-t CS-SENSE

Per slice (each slice is an independent real-time cine):
1. **Assemble** multi-coil k-space `y[T, C, ky, kx]` (readout fully sampled; ~13–18 ky lines/frame, R≈9).
2. **ESPIRiT coil maps** from the **time-averaged** k-space (full ky coverage) — sigpy
   `EspiritCalib(calib_width=24, thresh=0.02, crop=0.0)`. *(CPU: cupy is broken in this env; ESPIRiT is the
   only CPU step.)*
3. **k-t CS-SENSE** — reconstruct the *whole cine* jointly:
   `min_x  ½‖A x − y‖² + λ‖D_t x‖₁`, where `A x = mask · FFT2(mps · x)` (centered, `norm='ortho'`) and
   `D_t` = temporal finite difference. **λ = 0.02** (chosen by sweep on `us_0084`). Solved by **PDHG
   (Chambolle-Pock), 200 iters**, step `σ=τ=0.95/√(‖A‖²+4)` with `‖A‖²` estimated by **power iteration**
   (robust across coil counts). **GPU (torch)** — cupy/sigpy GPU unavailable, so the iterative solve is
   hand-written in torch.
4. **Magnitude** → crop 2× readout oversampling.

**Why these choices (lessons paid for):**
- **k-t (temporal-TV), not single-frame:** single-frame ESPIRiT+L1 *failed* at R≈9 (washed out); SoS /
  zero-filled is pure aliasing. Combining neighbors' complementary lines + a temporal-smoothness prior is
  what unfolds R≈9.
- **PDHG, not FISTA:** an early FISTA implementation used an *approximate* inner TV prox (few Chambolle
  iters) and was visibly worse at every λ. PDHG solves the L1(D·) term exactly → matches CPU quality.
- **Robust step size:** `σ=τ=0.45` is *marginally* over the PDHG stability bound (`στ‖K‖²>1`) and slowly
  diverged on some subjects; the power-iteration step (≈0.42) is always stable.
- **CMRxRecon contrast:** the fully-sampled CMRxRecon recon (`lixuan_simulation/`) is just an ESPIRiT
  *combine* — it works only because that data is complete. It is **not** transferable to OCMR's
  undersampled frames (a combine is not an un-aliaser).

**THE conceptual tension (most important caveat).** k-t recon needs *many frames per slice* — which is
exactly the slow many-frames acquisition the project exists to eliminate. So feeding the model
**k-t-reconstructed clean frames is an *optimistic* test**: it isolates the anatomy/contrast domain gap +
cross-slice reconstruction, but hands the model cleaner input than a true one-frame-per-slice acquisition
ever would. A **faithful** one-frame test must feed *degraded single-shot* frames (raw aliased, or
low-res "fast single shot", with **no temporal recon**) and rely on the model's cross-slice geometry to
compensate. Both tiers should be run; only the optimistic-input path exists today.

## 5. Outputs

```
scratch/data/ocmr/
├── ocmr_recon_ktcs.py            # THE recon script (exact algorithm + params; provenance)
├── eval_rt_sax_stacks/           # 10 good symlinks + README + _failed_long_fovshift/ (6) + _failed_fov_aliasing/ (us_0179)
└── recon/
    ├── recon_all.log
    ├── _failed_fov_aliasing/     # us_0179 (excluded; kept for the record)
    └── <subject>/                # 10 of these
        ├── sax_cine.nii.gz       # 4-D [W, H, slice, frame] float16; z header = 8 mm THICKNESS
        └── meta.json             # slice_positions_mm (→ true ~10 mm spacing), in-plane mm, thickness, TRes, n_frames, λ
```

**Model-inference outputs** (`tools/eval_ocmr_inference.py`, z-only resp checkpoint):
`result/ocmr_eval/<subject>/` — per subject: 3 random-draw beating-heart GIFs (mid-z V_canon over 12
target phases), an input contact-sheet, an ED V_canon volume panel, and a predicted-DVF figure
(`Δ = world_points − scanner_coords` in mm). Self-contained report: `_html/13_ocmr_inference_results.html`.

- **~292 MB total**, 10 clean subjects (+ us_0179 in `_failed_fov_aliasing/`).
- **NIfTI layout:** `[W=176, H=(117–180), slice=(12–14), frame=(29–128)]`; in-plane 2.16–2.84 mm; the z
  header is **8 mm THICKNESS** — the true slice **spacing is ~10 mm** (center-to-center from
  `meta.json:slice_positions_mm`); the adapter uses the latter. The 4th axis is **time**. Index a 2-D
  image as `v[:, :, slice, frame]`.
- **Save gotcha (fixed):** `sitk.GetImageFromArray` on a 4-D array silently makes a 3-D **vector** image
  (the readout axis becomes per-voxel components → a 5-D NIfTI with the time axis masquerading as space).
  The data is fine; the container is wrong. The script now builds per-frame 3-D images + `sitk.JoinSeries`
  (same as the CMRxRecon recon) to write a correct 4-D cine. Already-written files were re-wrapped in place.

## 6. How to use these for evaluation

- **Form model input** by sampling **one frame per slice** at scattered times (the scattered single-frame-
  per-slice regime), then resample each slice into the canonical cube (`1.4×1.4×8 mm`, `256×256×12`) via
  the existing `training/data/preprocess.py` path. **Done** in `tools/eval_ocmr_inference.py` (uses the true
  ~10 mm z-spacing, per-subject percentile norm, z-only model, `target_t` sweep → V_canon). Qualitative
  results: `_html/13`.
- **Scoring (no clean 3-D reference exists):**
  - **Tier-C (start here):** hold out input frames; re-slice `V_canon` at their `(z,t)` and compare —
    re-projection self-consistency + qualitative. Needs only this data.
  - **Tier-B (later):** reconstruct a per-scan motion-resolved reference from the k-space and score against it.
  - **Tier-A:** impossible here (prospectively undersampled → no ground truth; and gated `fs_*` scans are
    different subjects).
- **Domain shifts these data carry vs training:** real R≈9 recon texture, 1.5T (vs 3T CMRxRecon),
  patients (vs healthy volunteers), 2.16–2.84 mm in-plane, free-breathing motion — all *intended*.

## 7. Known issues / TODO

1. **6 long stacks + us_0179 fail FOV-aliasing** — FOV-shift correction (read the acquisition's FOV/position
   offset, apply the phase ramp before recon) → would recover `us_0198/0199/0200/0204/0207/0211` **and**
   `us_0179`.
2. **Faithful single-shot test not built** — need degraded single-frame inputs (raw aliased / low-res
   single-shot) to test the *true* one-frame regime, not just k-t-clean inputs.
3. **Canonical-preprocess wiring** — DONE (`tools/eval_ocmr_inference.py`).
4. **Run the model** — DONE (qualitative, z-only resp ckpt; `_html/13`). Quantitative scoring (Tier-B/C)
   still open — current eval is visual only (no reference).
5. **ESPIRiT on CPU** is the per-slice bottleneck (cupy broken); fine for 11 subjects (~3 min each).

## 8. Provenance & reproduce

- Recon: `scratch/data/ocmr/ocmr_recon_ktcs.py <subject.h5> recon/` (GPU; one subject ≈ 3 min).
- Selection/QC/sweep scripts were exploratory (under `/tmp` during the session); the *authoritative* recon
  is the script above. The dataset survey that motivated using OCMR: `_html/08_candidate_evaluation_datasets.html`.
- Related: [[02_related_work_literature_review]], [[03_cardiac_svr_literature_review]],
  [[04_inference_information_contract]] (blind-input / no-Tier-A-reference stance),
  [[05_respiratory_variants_results]] (the simulated-breathing results this aims to validate against reality).
