# Reference-conditioning figure (paper Fig. 6) and the Fig. 7 ablation-motion restyle

> **TL;DR & takeaway** — Paper Fig. 6 (`fig:reference_conditioning`) shows that the reference frame
> alone sets the reconstructed cardiac state. For one AF and one HRV test subject (different
> subjects), the whole input stack is fixed and only the reference frame changes (ED vs ES). The same
> input slice deforms outward toward ED and inward toward ES.
> - **Shown:** AF = `CMRx24_Train_P007` slice 6, input frame 9 (ED t11 / ES t4, mean in-plane motion
>   3.78 / 2.94 mm). HRV = `MNMs_E3L8U8` slice 7, input frame 0 (ED t4 / ES t9, 3.61 / 3.30 mm).
>   Picked by eye from a gallery of the 8 best subjects per rhythm; the ranking came from a sweep of
>   80 candidate slices × 12 input frames.
> - **The displayed input frame of the fixed slice is a real simulated frame, but NOT the frozen
>   evaluation draw.** ED and ES pull in opposite directions, so the evaluation frame
>   (min(ED, ES) ≈ 0.55 mm for both subjects) cannot show motion toward both targets. All other slices
>   keep their evaluation frames. The paper does not describe the selection (user decision); it only
>   avoids calling the example "representative".
> - **Style (both Fig. 6 and Fig. 7 panel b):** standard cardiology SAX view (RV left, anterior up),
>   arrows only (`#F39078`, true length, no magnitude map, no colour bar), and a `#6D8EAE` 1.0 pt
>   reference inset (Fig. 6).
> - **Paper:** Overleaf commits `9b3c7ed` (figure + Results paragraph + Sec 4.4 back-reference),
>   `dbc1297` (Fig. 7 flip, inset colour), `1cfff5b` (arrows only, captions). Page count not yet
>   checked.

Everything below is the agent-facing record.

## 1. What is in the paper

- **Fig. 6** `figures/reference_conditioning.pdf`, label `fig:reference_conditioning`. It fills the
  placeholder that was already commented out in Results ("Reference-conditioned reconstruction with fixed
  non-reference observations and different designated reference frames"). Caption:
  > Reference-conditioned reconstruction under AF (left) and HRV (right), on different test subjects. For
  > each subject, the input stack is fixed and only the reference frame (blue inset) changes, to the ED or
  > the ES frame. Each target panel shows the in-plane deformation of the same input slice as arrows
  > within the heart ROI, after removing the slice's rigid shift due to breathing.
- **Results paragraph** after "Reconstruction efficiency" (ED/ES are spelled out here: the paper defined
  neither before this point):
  > \paragraph{Reference-conditioned cardiac state.} To isolate the role of the reference slice, we fix
  > the input stack of a test subject and change only the reference frame, to either the end-diastolic
  > (ED) or the end-systolic (ES) frame (Fig.~\ref{fig:reference_conditioning}). Under both rhythms, the
  > same input slice deforms outward toward the ED state and inward toward the ES state. Since the
  > reference frame is the only input that differs, it alone determines the cardiac state that
  > CardioStitch reconstructs.
- **Sec 4.4 (real RT data)**: "…even though the input of the remaining slices never changes, consistent
  with Fig.~\ref{fig:reference_conditioning}." That section makes the same claim on real data.
- **Fig. 7** `figures/ablation_motion.pdf` (`fig:ablation_motion`, was "Fig 6" before this figure was
  inserted): panel (b) is now flipped to anterior-up and drawn as arrows only; the caption drops
  "over its in-plane magnitude map".
- **Placement decisions (user):** a paragraph in the existing Results subsection, not a new section and
  not the appendix. It is not an ablation, it is not Method, and Sec 4.4 is for real data only. No
  appendix on how the example was chosen. No "Best viewed in color": it was added when the figures had the
  magma map, then removed once the map was dropped.

## 2. The experiment

- **Model:** the scored arm `vggt_final518_diff1000_ep300`, ckpt
  `scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt`.
- **Input:** the frozen evaluation bundle `evaluation/volumes/<src>_{af12,hrv12}/out/<subject>/`
  (breath inputs, `mask_heart.nii.gz`, `manifest.json`), batch built by
  `evaluation/src/engine/run_vggt.py` (`prepare_batch` with the bundle's scatter draw).
- **ED / ES target** = the reference frame with the largest / smallest GT LV volume in that rhythm
  (`lv_curve_gt` in `evaluation/metric_results/test/<src>_<rhythm>/ef/<arm>.json`).
- **What changes between panels:** only slot 0 (the reference) is swapped to the ED or ES frame. For each
  subject, every non-reference slot, including the displayed slice, has identical pixels in both runs
  (`render_motion_edes.py` asserts that the ED and ES panels share one input frame).
- **Motion shown:** the point head's in-plane Δ for that slot, converted to mm, inside the heart ROI.
  The slice's mean Δ over the ROI (its rigid breathing correction) is subtracted, as in Fig. 7.
- **Why the frame is overridden:** with the frozen evaluation frame, one target barely moves
  (min(ED, ES) = 0.56 mm for P007, 0.55 mm for E3L8U8). `sweep_input_frame.py` replaces the displayed
  slice's frame with each of its 12 real simulated frames and scores min(mean ED motion, mean ES motion).

## 3. Selection pipeline (what was run, 2026-09-25, caesar, GPUs 0–2)

1. **Candidates:** `tools/pick_motion_candidates.py --n 40 --planes 2` →
   `temp/motion_sweep/candidates.txt` (80 `source:subject:z`). It takes the 40 test subjects with the
   largest ED–ES GT LV change, using the smaller of the AF and HRV values, and for each the 2
   non-reference planes with the largest LV area (GT seg, frame 0).
2. **Sweep:** `tools/sweep_input_frame.py --out scratch/motion_edes/sweep80 --subjects …`, 3 shards (one
   per GPU). 80 planes × 2 rhythms × 12 frames × 2 targets = 1,920 frame evaluations in about 20 min.
   It writes scores only, to `scratch/motion_edes/sweep80/scores/<rhythm>/<subject>_z<z>.json`.
3. **Rank:** `tools/rank_input_frame_sweep.py scratch/motion_edes/sweep80 --k 8 --specs <file>` keeps the
   best plane of each of the 8 best subjects per rhythm:

   | # | AF (min mm) | HRV (min mm) |
   |---|---|---|
   | 1 | CMRx25_R2val_Center004_Siemens_15T_Aera_P041 z4 f0 (3.14) | MNMs_E3L8U8 z7 f0 (3.30) |
   | 2 | MNMs_H8K2K7 z5 f11 (3.08; = its frozen frame) | MNMs_A1K2P5 z6 f7 (3.23) |
   | 3 | **CMRx24_Train_P007 z6 f9 (2.94)** | MNMs_H8K2K7 z5 f10 (3.11) |
   | 4 | ACDC_patient130 z7 f10 (2.88) | CMRx24_Train_P007 z6 f6 (3.09) |
   | 5 | MNMs_A1K2P5 z6 f7 (2.76) | CMRx25_…P041 z4 f4 (2.66) |
   | 6 | MNMs_E3L8U8 z7 f9 (2.64) | CMRx23_Val_P017 z6 f3 (2.64) |
   | 7 | CMRx24_Test_P013 z6 f8 (2.61) | ACDC_patient130 z7 f11 (2.59) |
   | 8 | CMRx24_Val_P016 z6 f9 (2.44) | CMRx24_Val_P016 z6 f8 (2.51) |

4. **Save fields** for those frames: `sweep_input_frame.py --out scratch/motion_edes/sweep80_save
   --subjects <source>:<subject>:<z>:<frame> …`. The 4th field is a per-subject `--save-frame`, so one
   model load per GPU serves many subjects. Fields are ~32 MB per target, which is why step 2 saves none.
5. **Gallery:** `tools/render_motion_edes_gallery.py <specs> scratch/motion_edes/sweep80_save
   figs/motion_edes` → `figs/motion_edes/gallery_{AF,HRV}.png` (per-row colour scale).
6. **Pick (user, by eye):** AF #3 P007 (cleanest LV ring: outward at ED, converging at ES) and HRV #1
   E3L8U8. AF and HRV must be different subjects: with the same subject the two halves are near-duplicates
   (same breathing shift, about 1% pixel difference).

## 4. Rendering decisions

- **Standard SAX orientation** (`sax_standard` in `render_motion_edes.py`). The bundle affines are
  diagonal placeholders (origin 0), so orientation cannot come from metadata; it comes from anatomy.
  (1) Rotate so the RV centroid (seg label 3) is left of the LV centroid (label 1), with the RV→LV line
  horizontal. (2) Flip vertically if the body surface nearest the heart (the anterior chest wall) is
  below it. Images, masks, insets and field vectors all get the same transform (`orient`,
  `orient_field`). A synthetic radial field stays exactly radial (cos 1.0) through
  rotation + flip, so vector rotation is verified. Result: P007 rot 4° + flip; E3L8U8 rot 57°, no flip.
  Step (2) is a heuristic; check each new subject by eye (heart against the top wall, liver
  bottom-left).
- **Fig. 7 uses flip only** (`render_ablation_motion.py --flip-v`). For P093 slice 4 the heuristic gave
  rot −19° + flip. The user rejected the rotation: the standard view only needs RV left and anterior up,
  not an exact RV–LV angle. **Open inconsistency:** Fig. 6 still applies its rotation (4° / 57°). The
  57° for E3L8U8 does real work, because without it the RV sits above the LV. Nobody decided whether
  Fig. 6 should also be flip-only.
- **Arrows only** (`--no-map --arrow-color "#F39078"` in both scripts): no magma map and no colour bar.
  Arrows stay at true length with grid step 11, as in Fig. 7. A 2× length, sparser, thicker-arrow variant
  was tried and **rejected** because it broke the match with Fig. 7.
- **Inset** (Fig. 6): always bottom-right, 0.30 of the panel side (`INS`), reference image cropped tight
  on its heart ROI, `#6D8EAE` outline at 1.0 pt, no "Ref." label (the caption defines it). A widened
  off-centre crop that avoided inset/heart overlap made the heart and arrows too small and was reverted;
  the inset may overlap a low-motion ROI corner.
- **Caption text is not coloured.** The paper loads `xcolor`, but `#F39078` text on white has about 2.3:1
  contrast. The colours are named in words ("blue inset").

## 5. Reproduce the paper figures (verified pixel-identical to Overleaf `1cfff5b`)

```bash
# Fig. 6 (fields in scratch/motion_edes/sweep80_save, GPFS)
PYTHONPATH=tools micromamba run -n svr python tools/render_motion_edes.py --dump scratch/motion_edes/sweep80_save \
  --af cmrx2024/CMRx24_Train_P007,6,CMRx24_Train_P007_z6f9 --hrv mnms/MNMs_E3L8U8,7,MNMs_E3L8U8_z7f0 \
  --no-map --arrow-color "#F39078" --out reference_conditioning.pdf
# Fig. 7 (saved eval outputs only, no inference)
PYTHONPATH=tools micromamba run -n svr python tools/render_ablation_motion.py \
  --subject cmrx2023_af12/CMRx23_Train_P093 --tight --flip-v --no-map --arrow-color "#F39078" --out ablation_motion.pdf
```

Without `--flip-v --no-map --arrow-color`, `render_ablation_motion.py` reproduces the pre-change Fig. 7
exactly (0.0 pixel difference); the defaults of both scripts are unchanged. The Fig. 6 fields can be
regenerated with step 4 of §3 (`CMRx24_Train_P007:6:9`, `MNMs_E3L8U8:7:0`).

## 6. Verification record

- The motion dumps replay the scored run to within 7e-7, and the sweep at the frozen frame reproduces the
  dump exactly (earlier session).
- The save runs (step 4) reproduce the step-2 sweep scores exactly (max |Δ| 0.0 mm, all saved subjects).
- `rank_input_frame_sweep.py --specs` reproduces the hand-built gallery list exactly.
- Both paper PDFs regenerate pixel-identically from the committed scripts (§5).
- The Overleaf updates changed only what was intended (pixel diff): Fig. 7 rows (a) and (c) are
  untouched, and Fig. 6 changed only in the inset band.
- **Not done (user: not needed):** a quantitative outward/inward check over all 80 slices. The sweep
  scores magnitude only, so the "outward toward ED, inward toward ES" direction rests on the two shown
  examples and the gallery, read by eye.

## 7. Gotchas

- **zsh does not word-split `$var`.** `--subjects $subs` passed all 80 specs as one argument and
  crashed; use `${=subs}`.
- **`pgrep -f "<pattern>"` in a wait loop matches the loop's own command line**, so it never exits. Use
  `pgrep -f "[s]weep_input_frame"`.
- **Machines:** caesar and Great Lakes have separate home directories. A script that exists only on one
  is invisible to the other until it is pushed (`render_ablation_motion.py` was, in `1ca96be`). `scratch/`
  (GPFS) is shared.
- The sweep writes scores only; rendering needs the saved fields (step 4).
- Overleaf repo `~/vggt/_paper` (git remote git.overleaf.com): always pull first, and push only when the
  user explicitly says so.

## 8. Files

- Code: `tools/pick_motion_candidates.py`, `tools/sweep_input_frame.py`, `tools/rank_input_frame_sweep.py`,
  `tools/render_motion_edes_gallery.py`, `tools/render_motion_edes.py`, `tools/render_ablation_motion.py`
  (`tools/dump_vggt_motion.py` made the earliest drafts on the frozen evaluation input).
- Data (GPFS, not git): `scratch/motion_edes/sweep80/` (scores), `scratch/motion_edes/sweep80_save/`
  (fields + scores of the 15 saved picks).
- Figures (git): `figs/motion_edes/motion_edes_arrows.png`, `figs/motion_edes/ablation_motion_arrows.png`
  (the paper versions), `figs/motion_edes/gallery_{AF,HRV}.png` (selection).
- Open: page count after adding Fig. 6; Fig. 6 rotation vs Fig. 7 flip-only (§4).
