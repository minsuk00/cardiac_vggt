# 16 · Göttingen radial real-time free-breathing bSSFP — download + RT-NLINV reconstruction

> **TL;DR & takeaway.** We added a second real-time free-breathing short-axis (SAX) evaluation
> source — the **Göttingen radial real-time bSSFP cine** dataset (Blumenthal/Uecker, NLINV-Net,
> MRM 2024) — to complement OCMR (Cartesian) with a **radial** domain. Downloaded all **68 volumes**
> (209 GB raw k-space, 5 Zenodo parts) and reconstructed every one with **classical RT-NLINV** (the
> authors' own BART pipeline, `mrirecon/nlinv-net`; **no neural net**, deliberately, to avoid a
> learned-prior hallucination confound in a downstream model eval). Output: 68 magnitude SAX cines
> `(160,160,Z,127)`, Z=20–27, in `scratch/data/goettingen/recon/`. We measured the respiratory
> motion directly (the authors don't report it): the heart's in-plane (≈AP) drift is only **~1.45 mm**
> — real but gentle (resting volunteers, shallow breathing; the dominant SI motion is through-plane
> for SAX and unmeasurable from single slices). **Each slice is an independent acquisition at an
> uncorrelated cardiac+respiratory phase**, so the stack is genuinely misaligned — the real
> "scattered acquisition" the project simulates. Eval plan: **self-consistency / leave-one-slice-out**
> (no clean 3D GT exists). Full data-side detail lives in `scratch/data/goettingen/README.md`;
> tooling in `tools/goettingen_recon/`; BART build notes in memory `reference_bart_build_greatlakes`.

## Why this dataset
The project needs to test transfer of the VGGT slice-to-volume model to **real** real-time
free-breathing cine (our training data is gated breath-hold CMRxRecon; see the main README and
`docs/04`). OCMR (`docs/06`) gives Cartesian RT/FB SAX; this adds a **radial** trajectory domain.
It's the *real* version of the scattered acquisition we simulate: gated breath-hold has cardiac
motion only, but here each slice carries both the beating heart **and** genuine respiratory drift,
and the slices share no cardiac/respiratory clock.

## Dataset facts (verified from headers + the MRM paper)
- **68 volumes** (vol0001–0087 with gaps, all `vis1`), **209 GB** BART `.cfl/.hdr` radial k-space,
  5 Zenodo records (`10492333, 10912299, 10492343, 10492455, 10493095`), CC-BY-4.0. (The paper cites
  40 volunteers; the public raw release has 68 RT SAX acquisitions.)
- Siemens Magnetom Skyra **3T**; **radial bSSFP, real-time, free-breathing, ungated**.
  TR/TE/flip 2.58/1.29 ms/23°. **13 spokes/frame**, 5-turn interleaved (360°/65 ≈ 5.54°/spoke),
  ~**33 ms/frame**, ~19× undersampled. FOV 256×256 mm, matrix **160×160** (1.6 mm), **6 mm** slice.
- Per subject: **20–27 slices** (median 24), **127 frames/slice** (uniform) ≈ 4.26 s, 28–36 coils.
  Totals: **1,630 slices**, **207,010** 2D frames.
- ⚠️ Slice *gap* is undocumented (raw `.cfl` has no slice positions); we assume **6 mm contiguous**.

## Reconstruction — classical RT-NLINV, no neural net
Both of the authors' recon paths run on **BART**; we use the classical one
(`nlinv-net/01_scripts/10_reco_rt.sh -R -S13 -V`): `01_prep` (5-turn trajectory + gradient-delay
correction + **ROVir** coil compression to 10) → `bart nlinv --real-time -S -i6 --cgiter=30
--sens-os=2 -g` (**RT-NLINV**: iteratively-regularized Gauss–Newton, 6 steps; CG 30-iter inner
solver; jointly estimates image **and** coil sensitivities, calibrationless; `--real-time` = **temporal
ℓ2 regularization**, frame *t* toward *t−1*) → temporal median filter → `img_rt_fil` → magnitude,
center-crop 240→160 → NIfTI. ~31 min/volume on an A40.

**Decisions:**
- **No NLINV-Net** (the trained model, weights Zenodo `11469859`). A learned prior can hallucinate
  detail, which is a confound when these images feed *another* model's reconstruction eval. The
  classical recon is the honest "what the measured k-space supports".
- **`nlmeans` skipped** (`SKIP_NLMEANS=1`). It's a ~33 min CPU edge-preserving denoise (not part of
  the recon; GPU idle meanwhile) that changes the image by only **1.37%** of dynamic range and
  rescales intensities ~3.6×. Dropped for faithfulness + domain-consistency (our training cine isn't
  nlmeans-processed) + cost. Re-enable by unsetting the flag.
- **Each frame uses its own disjoint 13 spokes** — no view-sharing; neighbors enter only via the
  temporal ℓ2 prior. (NLINV-Net, by contrast, uses a non-causal 3-frame receptive field.)

## Respiratory motion — measured (the authors don't report it)
`tools/goettingen_recon/measure_respiratory_motion{,_heartroi}.py`; figures/JSON in
`scratch/data/goettingen/analysis/`. Method: per-slice subpixel phase-correlation translation
tracking → frequency-split into respiratory (0.1–0.5 Hz) and cardiac (0.7–2.0 Hz) bands; PCA → ≈AP
axis. Controls: cardiac band recovers ~85 bpm (method valid); frame-shuffle collapses the drift
(physiology, not noise).
- **Whole-FOV AP = 1.9 mm**; **heart-ROI AP = 1.45 mm** (heart localized by cardiac-band power).
- The heart-ROI is *smaller* than whole-FOV — **overturning** our initial "static background dilutes
  it" hypothesis; the **chest wall moves more in AP than the heart**, so whole-FOV was slightly
  inflated. The heart's in-plane AP respiratory motion is genuinely small.
- Real (2.7× above shuffle control) but **gentle** — shallow free-breathing. AP is intrinsically the
  small component; the dominant SI motion is **through-plane** for SAX (shows as slice-content change,
  not displacement) so it can't be given a direct mm number (implied ~4 mm). See `docs/01` for the
  respiratory-sim literature this validates.

## Evaluation plan (no clean 3D GT)
Because the slices are misaligned (independent acquisitions, no shared clock), there is **no clean
ground-truth volume**. So eval is **self-consistency / leave-one-slice-out**: reconstruct the volume
from S−1 scattered single-frame-per-slice inputs, query at the held-out slice's cardiac phase, sample
at its `z`, and compare the predicted slice to the actually-acquired one (the held-out real slice is
the GT for that `(z,t)`). Optional later: build a **gated reference** from the dataset's self-gating
indices for an absolute PSNR (Tier-B), noting that reference is itself an approximate derived recon.

## Layout & repro
- Data: `scratch/data/goettingen/{radial,recon,analysis}/` + `README.md` (full data-side record).
  Recon dir is ~108 GB after deleting the useless BART intermediates (`img_rt_col` 889 GB +
  `img_rt` 89 GB); kept the 18 GB of NIfTIs + the complex `img_rt_fil`.
- Tooling: `tools/goettingen_recon/` — `download_goettingen.sh`, `bart_env.sh`, `recon_vol.sh`,
  `recon_all.sbatch` (array, prove-it-reviewed), `cfl_to_nifti.py`, `measure_respiratory_motion*.py`.
- BART: built from source with CUDA at `scratch/bart/` (gcc/13.2.0 + cuda/12.8.1, conda `bart-build`
  env for fftw/openblas, `liblapacke.so`→`libopenblas.so` symlink) — see memory
  `reference_bart_build_greatlakes`.
