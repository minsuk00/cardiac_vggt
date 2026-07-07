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
