# fetal_cmr_4d — every difference from the author code & paper

Running van Amerom et al. (MRM 2019) `mriphysics/fetal_cmr_4d` on adult, single-orientation,
already-reconstructed MIITT data. This file records **every** way our run departs from the
authors' implementation + paper, with the reason. Keep it current.

The upstream repo (`scratch/fetal_cmr_4d/repo`) is **byte-for-byte untouched** (`git status`
clean). All our changes live in `baselines/fetal_cmr_4d/`. Where we shadow an author `.m`/`.bash`
file, it's a copy on an earlier search path; the original is unmodified.

## A. NECESSARY differences (data-driven; the pipeline cannot run on MIITT without them)

| # | Difference | Author | Ours | Why unavoidable |
|---|---|---|---|---|
| A1 | Heart-rate search band | fetal `[105,180]` bpm | adult `[45,110]` bpm | MIITT is adult; fetal band would miss the beat / lock a harmonic |
| A2 | Real-time data source | Philips ReconFrame `xtRcn.mat` | our NIfTI (`s01_rlt_ab`) | MIITT ships reconstructed images, not Philips raw k-space |
| A3 | ktrecon front-end (k-t SENSE) | run on Philips raw | **skipped** | no Philips raw; data already reconstructed |
| A4 | `cardsync_intraslice` fig guard | (none) | `+ && isvalid(hFig)` | fixes a crash on the `verbose=false` path under headless MATLAB (authors always run verbose=true) |
| A5 | dc_vol stack-stack registration | `-stack_registration` | **dropped** | one SAX stack → nothing to register to; the flag hits a degenerate "avg weight 0" segfault |
| A6 | Inter-slice cardiac sync | `cardsync_interslice` | **omitted; intra-slice phases used directly (offset = identity)** | interslice sync aligns phase via slice spatial *overlap*, which only exists across multiple orientations; parallel SAX slices don't overlap |

## B. CHOICE differences (reasonable, but could be reverted; call them out in any writeup)

| # | Difference | Author | Ours | Rationale |
|---|---|---|---|---|
| B1 | Heart / chest ROIs | manual, drawn in MITK | **auto**, from cardiac-band spectral power | reproducible + scales to 13 subjects without manual labor; spot-checked against images. Trade: a bad auto-mask would hurt gating/recon |

## C. Things that MUST stay at author defaults (revert my earlier mistakes)

These I had changed for speed/memory without cause. Faithful = author values:

| Param | Author (`recon_cine_vol.bash`) | Status |
|---|---|---|
| `-resolution` | **1.25 mm** | revert (I had run 1.5–2.0) |
| `-rec_iterations` | **10** | revert (I had run 5) |
| `-rec_iterations_last` | **20** | revert (I had run 10) |
| robust statistics | **ON** (no `-no_robust_statistics`) | revert (I had it OFF) |
| `-iterations` | 4 | already faithful |
| `-numcardphase` | 25 | already faithful |
| recon mask | `mask_cine_vol` (tight, heart-derived, built by the wrapper) | use it (I had used the oversized `mask_chest`) |

## D. Single-orientation reduction (consequence of A5+A6)

The paper's 5-stage flow assumes multi-orientation. On one SAX stack it reduces to:
`intra-slice gating → dc_vol static MC → recon_cine_vol (4D SR)`.
The dropped stages (stack-stack reg, slice_cine, interslice sync) are all multi-orientation
machinery. This reduction is itself a reportable finding, not a shortcut.

## E. LV-area self-gating + the `run_selfgate_recon.sh` reconstructCardiac call (doc 35, 2026-07-05)

**⚠️ UPDATE 2026-09-16, CONFIRMED (§F below):** `s01_rlt_ab.nii.gz` (from `export_miitt.py`) carried no frame-duration header, so nibabel defaulted it to 1.0 s against a true 25 ms — a ~40×-too-wide temporal-PSF window. **Re-run with the header fixed (author-faithful params, `temp/fetal4d_3way/miitt/`): EF rose from 20.6% to 43.2%.** **⚠️ SUPERSEDED the same day (docs/105 §5c): that re-run tagged the pixdim as `sec`, which MIRTK reads ×1000 (25 ms → "25" against an R-R in seconds) — its window was flat too — and both runs carry the container binary's `-cardphase` off-by-one (every frame gets its predecessor's phase). Neither EF figure characterises the method.** `export_miitt.py` now writes 0.025 s with the time-unit code UNSET (the authors' own convention); MIITT is not being re-run for the paper.

The self-gating run (`run_selfgate_recon.sh`, fed by `cardphases_lvanchor_cardsync.txt`). **No author
`.m`/`.bash` file is modified** — the LV-area gater is NEW code (`selfgate_lvarea_{extract,assemble}.py`)
operating on the authors' OWN outputs (`rrintervals.txt`, `cardphases_intraslice_cardsync.txt`); the
re-anchoring changes only the per-slice phase OFFSET (see A6). Its `reconstructCardiac` **parameter
values are 100 % author-faithful** (§C: resolution 1.25, rec 10/20, robust ON, iterations 4,
numcardphase 25 — all verified against `recon_cine_vol.bash` defaults RESOLUTION=1.25/NSR=10/
NSRLAST=20/NUMCARDPHASE=25 and `ITER=$((NMC+1))=4`). Deviations vs the authors' line-147 call, all
NECESSARY (single-orientation) or cosmetic:

| # | Author call | Ours | Why |
|---|---|---|---|
| E1 | `-cardphase cardphases_interslice_cardsync.txt` | `cardphases_lvanchor_cardsync.txt` | interslice sync impossible single-orientation (A6); LV-area ED anchor is the substitute (doc 35). **Algorithmic, necessary, the whole point.** |
| E2 | `-mask mask_cine_vol.nii.gz` | `-mask s01_mask_heart.nii.gz` | `mask_cine_vol` is built (recon_cine_vol.bash L106-142) by transforming per-slice heart masks with the **dc_vol slice dofs**, which don't exist single-orientation (dc_vol segfaults, doc 34 §5). The tight heart mask `s01_mask_heart` is the closest faithful substitute; the author's erode-2/close-9 refinement on the recon grid is skipped (not reconstructable without dc_vol). |
| E3 | `-dofin stack-transformation*.dof -slice_transformations <dc_vol>` | **dropped** | staged dc_vol moco init; dc_vol segfaults single-orientation (doc 34 §5). reconstructCardiac does its own register↔reconstruct moco from scratch. |
| E4 | `-remote` | **dropped** | TaskSpooler (`tsp`) not in the container (doc 34 §7); infra, not algorithmic. |
| E5 | `-force_exclude_stack/-force_exclude_sliceloc/-force_exclude` (+ files) | **dropped** | our `data/force_exclude_*.txt` are EMPTY (0 exclusions) → passing them = `... 0` = no-op = identical to omitting. Functionally identical. |
| E6 | `-debug` | **dropped** | saves intermediate volumes only; cosmetic. |

**Forced GATING deviations** (same as A1/A6, restated for this run): adult HR band `[45,110]` (A1);
LV-area ED anchor replacing interslice sync (A6). Both are the doc-34/35 findings, not tuning.

## F. v2 eval-bundle arm (`evaluation/src/engine/run_fetal4d.sh` + `fetal4d_gate.py`, docs/105, 2026-09-15)

The paper-comparison arm runs on the shared v2 bundle (`scratch/eval/<source>/out/<subject>/`), not
on MIITT real-time data. Engine call otherwise identical to §E (author params verbatim: iterations 4,
rec_iterations 10/20, robust statistics ON, temporal PSF) except resolution (F7 below). Differences vs §E:

| # | §E (MIITT RT pilot) | v2 arm | Why |
|---|---|---|---|
| F1 | 180 real-time frames/slice | **12 frames/slice = the stored cine, each plane circularly rolled by a frozen random offset** (`rolled/`) | the only bundle input on which self-gating has a job: on the stored ED-first cine the gater just reads the index back (measured, docs/105); real RT exists only for MIITT/OCMR |
| F2 | x-f Fourier R-R per slice (MATLAB) + `-rrintervals` measured | **period not estimated; R-R nominal 1.0 s** for every slice | one full cycle per slice ⇒ Fourier peak = window length by construction (352/380 slices); sources carry no timing. Only sets the temporal-PSF time axis |
| F3 | ED anchor: nnU-Net Task114 on the RT frames | same rule, nnU-Net Task114 on the **rolled input frames** (never the bundle's GT seg — it carries GT frame order, i.e. the hidden label) | same substitution as A6/E1, same segmenter the scorer uses |
| F4 | `-numcardphase 25` | **`-numcardphase 12`** | match the 12 GT phases; output length is a free choice in the authors' code (doc 34 §3.1) |
| F5 | mask = spectral auto heart ROI | **`mask_heart.nii.gz`** (nnU-Net LV+MYO+RV union, dilated) | the bundle ROI every other baseline (SVRTK, NeSVoR, NiftyMIC) reconstructs in — identical across arms |
| F6 | one breathing state per frame (real) | one breathing state per **plane** (the bundle's `breath/`) | a 12-frame RT burst is ~1 s ≪ a breathing cycle; keeps the corruption byte-identical to every other arm |
| F7 | `-resolution 1.25` (author literal default, `recon_cine_vol.bash`) | **`-resolution 1.4`** | match `svrtk3d`/`nesvor`/`niftymic`, which all deliberately override THEIR OWN tool defaults to 1.4 mm to match each other (docs/105 §5b) — `fetal_cmr_4d` was the only SVR arm still at a literal author/tool default. Measured trade-off on P012 (docs/105 §5b): coarser grid reduces the per-phase pose jitter that made the cine look unstable (mean centroid step 5.6→3.2 mm, GT is 2.1 mm) but also partial-volumes the ES cavity boundary more, blunting the contraction amplitude (LV-volume range 2888–5538 at 1.25 mm vs 3873–4933 at 1.4 mm vs GT 2504–5924) — kept at 1.4 mm anyway for cross-baseline consistency, per user decision. |

Unchanged: single stack (A5/D), stages dropped (E3), inter-slice sync replaced (A6/E1). Base/apex
slices with no segmentable LV keep offset 0 (docs/35 §6). Gating diagnostics vs the frozen roll are
written to `gate.json` for auditing only; the arm never reads the roll.

**⚠️ Frame-duration header (found 2026-09-16, docs/105 §5a).** `reconstructCardiac` takes each
frame's duration from the input 4D NIfTI's TIME pixdim (`ReconstructionCardiac4D.cc`:
`_slice_dt = attr._dt`; temporal-PSF width `dtrad = 2π·dt/rr`). The authors' k-t SENSE export writes
the true frame time; **our exports did not**: `export_miitt.py` (§A2, `s01_rlt_ab.nii.gz`) and the
first version of `fetal4d_gate.py` both left nibabel's default **1.0 s** = one whole R-R, so the sinc
window spanned the entire cycle and every output phase blended all frames. `fetal4d_gate.py` now
writes `dt = RR/T`. **Every §E MIITT result (doc 36 §7's "temporal PSF flattens contraction") was
produced with the 40×-too-wide window and must be re-run with `dt = 0.025 s` before being cited.**

**⚠️ Two more, found on a `-debug` run 2026-09-16 (docs/105 §5c), both now handled in
`fetal4d_gate.py` / `run_fetal4d.sh`:**
- **Time-unit tag.** A `sec`-tagged time pixdim is read ×1000 by MIRTK while `-rrinterval` is in
  seconds, so `dt = RR/T` tagged `sec` still gave a flat window (`2π·83.3/1.0`). The unit code must
  stay UNSET, as the authors' `mrecon_writenifti.m` leaves it (`xyzt_units = 0`, seconds).
- **`-cardphase` / `-rrintervals` off-by-one in the container binary.** The count token is stored
  as entry 0 (frame 0's phase / slice 0's R-R) and every value shifts by one. The authors' scripts
  pass the lists the same way (their logs: "R-R intervals are 0:13 …", "2341 images" for 2340), so
  their runs carry it too. Workaround: count = 710 (= 113·2π ≡ 0), slice 0 re-rolled so its ED frame
  is frame 0, θ₁… zero-padded to 710 tokens; `-rrintervals` not passed (engine fallback = `-rrinterval`).
- **Output phase index.** `run_fetal4d.sh` rolls the finished cine so phase 0 = the GT's
  seg-derived ED (`seg_gt/` LV argmax, the scorer's own index); GT frame 0 is not ED for ~14 % of
  CMRx25 subjects (docs/105 §5d). One scalar, no per-slice information, no-op when GT ED = 0.
