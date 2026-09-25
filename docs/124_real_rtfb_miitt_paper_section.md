# 124 — Real free-breathing real-time (RTFB) reconstruction: MIITT AF patient + volunteer, paper Sec. 4.4

> **TL;DR & takeaway**
> VGGT `final518_diff1000` runs zero-shot, with no code change, on REAL real-time free-breathing
> ungated MIITT scans: the cardiomyopathy + AF patient (D=12) and Volunteer1 (D=13). Neither is in
> `pooled_curated_v2` (0 MIITT lines). Two qualitative results go into the paper as **Sec. 4.4
> "Real Free-Breathing Ungated Acquisitions"** (Overleaf, figure `figures/real_rt.pdf`).
> (1) **Spatial:** in long-axis (LAX) cuts, the naive stack (frame f of every slice) has visible
> stair-steps; ours from the same input is a coherent ventricle.
> (2) **Temporal:** ours beats regularly (volunteer) or irregularly (AF), and the naive stack shows
> no coherent rhythm. The LV volume with the reference slice excluded still follows the rhythm
> (r=0.98 vs the real reference-slice LV area, both subjects), so the rhythm is propagated to slices
> whose input never changes.
> The paper text shows **no numbers** by design. The numbers below are reviewer-rebuttal backup
> only. The results do **not** support absolute LV volumes or EF (see §5). No baselines were run on
> RT data (decision, §6).

## 1. Data and model
- RT data: `scratch/data/MIITT/nifti/<subj>/realtime/sax/{4d_recon,heart_seg}.nii.gz`
  (golden-angle spiral R=12, 2.3 mm, 8 mm slices + 2 mm gap, 25 ms/frame, 180 frames = 4.5 s per
  slice, slices sequential, no ECG/resp). `heart_seg` is this project's own nnU-Net Task114
  **2D** segmentation of every RT frame (docs/39; labels LV=1/MYO=2/RV=3). It is NOT
  data-provided.
- Checkpoint: `scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt`.
  Reference = slot 0 = z6 (mid) for both subjects.
- Paper wording: "in-house" dataset (coauthor J. Hamilton's lab). The paper does not say "MIITT",
  "paired", or "cardiomyopathy".
- **Open:** "prospectively acquired" (3 places in the paper) assumes each RT frame is
  reconstructed independently, with no temporal regularisation or sliding window. This is
  unconfirmed; ask Dr. Hamilton, and drop the phrase if not.

## 2. Runs (all outputs under `scratch/temp/miitt_afib_rt/`, GPFS)
| what | how | output |
|---|---|---|
| Original RT recon (random companion draw) | `run_vggt_rt.py` via a scratchpad wrapper that only redirects `OUT_ROOT` | `vggt/` + `vggt/slot_frames.json` (draw recovered on CPU; 10/12 planes match by correlation, 1 has no heart, 1 is deformed by design) |
| **Frame-0 cine (LV curve)** | same wrapper, but every companion `timesteps` forced to 0; reference swept over 0..179 | `vggt_frame0/{AFib,Volunteer1}/` (142 s / 168 s, 1 GPU) |
| **Frame-f stacks (LAX panels)** | `tools/run_vggt_rt_stacked.py`: every slot = frame f of its plane, one forward each | `stacked_vs_ours_180/<subj>/frame_XXX/{input,ours}.nii.gz` (all 180 f, AFib + Volunteer1); also `stacked_vs_ours_all/` (f = 0,10,..,170, + Volunteer4) and `stacked_vs_ours/` (hand-picked) |
| LV curves (2D) | `tools/rt_lv_curve_2d.py`: nnU-Net Task114 **2d** per frame, LV px × voxel volume | `lvseg2d/lv_curves.json`, `lv_curves_noref.json` (reference slab z6 excluded) |
| Alignment score | nnU-Net 2d on all 720 frame-f volumes (input + ours) | `alignseg2d/seg/`, `lax_filtered/scores.json` |
| **Paper figure** | `_paper_figures/make_rt_figure.py --row MIITT_Volunteer1:20:volunteer1:"Healthy volunteer" MIITT_Patient_2024Jan04_Cardiomyopathy_AFib:170:afib:"AF patient"` (one `--row` with both values: `nargs="+"`, so a second `--row` would replace the first) | `figure/rt_figure_short.{png,pdf}` (final, compact layout) → Overleaf `figures/real_rt.pdf` |

Browsing/QC renderers: `render_rt_stacked_frames.py` (candidate naive stacks),
`render_rt_stacked_vs_ours.py`, `render_rt_lax_candidates.py` (all-frame LAX sheets),
`render_rt_orthoviews.py` (GIFs), `render_rt_mmode.py` (M-mode; rejected for the paper as hard to read).

## 3. Design decisions
- **The LAX panels use frame f of every slice. The LV curve uses frame-0 companions with the
  reference swept.** Both follow the paper's cine protocol (`method_lx.tex` `sec:inference`). They
  are different inputs, which is why the panels are not frames of the curve.
- **LV volume from 2D per-slice segmentation for both curves.** A 3D segmenter is
  out-of-distribution on a misaligned stack and would penalise the naive curve for the wrong
  reason. Per-slice disc summation is invariant to in-plane shifts, so the naive curve differs only
  through desynchronised cardiac/breathing states.
- **Naive curve = frame k of every slice.** A frozen frame-0 stack would be trivially flat apart
  from the reference slice.
- **The frame-0 companion draw differs from the original seeded draw** (for example z3=81,
  z8=179). The user chose frame 0 as "the first frame as acquired".
- **Figure frames** (volunteer f=20, AF f=170) are the user's picks after viewing all 180 frames.
  The alignment filter (§4) was used only for browsing.

## 4. Measured numbers (backup; NOT in the paper text)
| | AF patient | Volunteer1 |
|---|---|---|
| Alignment score (centroid RMS off a straight axis ⊕ LV-radius 2nd-difference RMS), input → ours, median [IQR] | 7.75 [6.8–9.2] → 5.42 [4.9–6.0] mm | 8.91 [7.6–10.2] → 6.62 [5.4–7.4] mm |
| Frames where ours is better aligned | 171/180 (95%) | 162/180 (90%) |
| corr(our LV volume, real ref-slice LV area) | 0.98 | 0.99 |
| corr(naive LV volume, same) | 0.24 | −0.22 |
| **corr(ours with the ref slice excluded, same)** | **0.98** | **0.98** |
| LV swing: ref slab vs other slices | 17 vs 99 mL (total 114) | 13 vs 63 mL (total 75) |
| End-systole times, real ref slice vs ours | within 0.03 s (5/5 beats) | within 0.03 s (5/5) |
| Our beat-interval CV | 0.15 (irregular) | 0.02 (regular) |

The timing agreement is **inherited from the reference by construction**. The non-trivial
evidence is the reference-excluded correlation and the ~85% of the swing that comes from the other
slices.

## 5. Caveats (measured)
- Absolute volumes and EF are unreliable. Ours peaks at ~195–200 mL vs gated EDV 184 (AF) and
  ~150 vs 138 mL (vol). Rough EF read off the curve is ~54 vs 43% gated (AF) and ~41 vs 55%
  (vol), with errors in opposite directions. This also compares 2D against 3D segmentation.
- Motion is attenuated away from the reference. With the random-draw run, recon/raw temporal std
  in the heart is 0.10–0.28 at apical planes and 0.4–0.5 next to the reference (the sim af12
  companions give 0.36–0.53 vs GT). The curves look squarish.
- n = 2, and there is no 3D ground truth.

## 6. Decisions and debate record
- Baselines on RT data: **not run**. Fetal CMR 4D / CiNeVol need a multi-beat self-gate and
  cardiac/resp labels (plan in `scratch/temp/miitt_afib_rt/PLAN.md`), and the scatter baselines
  produce static hearts. We rely on the "gated/binned averages beats" argument plus the simulated
  af12/hrv12 results.
- Two 2-subagent debates (claim scope; section wording) converged on the framing "zero-shot on
  real RTFB + coherence + reference-state propagation, not accuracy". The reviewer side wanted
  all 13 MIITT subjects and a rigid-centroid baseline. **Not done**: the user chose a
  qualitative-only section.
- Placement: Sec. 4.4, after Ablations, as a qualitative sim-to-real finale.
