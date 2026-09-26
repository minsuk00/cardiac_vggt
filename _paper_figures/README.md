# _paper_figures — code that makes the paper's figures and table numbers

The paper itself is the Overleaf repo (`~/vggt/_paper`, remote git.overleaf.com, gitignored here). This
directory holds the **final renderers** for the figures we make and the builder for the table numbers. Selection
and sweep helpers stay in `tools/` (linked below). Run everything from the repo root in the `svr` env.

**Fonts:** every figure here uses Arial (math in STIX sans) via `paper_rc()` in `render_runtime_accuracy.py`
(`make_rt_figure.py` sets the same rcParams). Arial must be installed or matplotlib silently falls back to
DejaVu Sans; see docs/127 §9 for the install and a one-line check.

Figs. 1–4 (teaser, method, splatting, qualitative comparison) are not made here.

| Overleaf file (`figures/`) | Paper figure | Script | Exact command | Inputs | Doc |
|---|---|---|---|---|---|
| `runtime_vtc_af_hrv.pdf` | runtime–accuracy + LV volume–time curves, AF / HRV | `render_vtc_2x2.py` (helpers: `render_vtc_figure.py`, style: `render_runtime_accuracy.py`) | `python _paper_figures/render_vtc_2x2.py --out <dir>` → `<dir>/runtime_vtc_2x2.pdf` (defaults: `--source cmrx2023 --subject CMRx23_Train_P045 --table temp/allphase_seg/paper_tables_v2.json`) | EF jsons `evaluation/metric_results/test/<cohort>/ef/`; table JSON (below) | docs/120 (runtimes) |
| `reference_conditioning.pdf` | reference-conditioned reconstruction (fixed stack, ED vs ES reference) | `render_motion_edes.py` | `python _paper_figures/render_motion_edes.py --dump scratch/motion_edes/sweep80_save --af cmrx2024/CMRx24_Train_P007,6,CMRx24_Train_P007_z6f9 --hrv mnms/MNMs_E3L8U8,7,MNMs_E3L8U8_z7f0 --no-map --arrow-color "#F39078" --crop-window-af --gamma-af 0.9 --out reference_conditioning.pdf` (half text width, AF row above HRV; AF window from its crop + gamma 0.9 to brighten, reference insets included) | saved motion fields (GPFS) from `tools/sweep_input_frame.py`; eval bundles | docs/127 |
| `ablation_motion.pdf` | motion learned by each loss-ablation arm | `render_ablation_motion.py` | `python _paper_figures/render_ablation_motion.py --subject cmrx2023_af12/CMRx23_Train_P093 --tight --flip-v --no-map --arrow-color "#F39078" --crop-margin -12 --label-color black --label-box --out ablation_motion.pdf` | saved eval outputs (`ed_dvf.npz`, EF jsons, `_motion_epe/af12/`) | docs/127 |
| `real_rt.pdf` | real free-breathing RT (MIITT volunteer + AF patient) | `make_rt_figure.py` | `PYTHONPATH=training:. python _paper_figures/make_rt_figure.py --root scratch/temp/miitt_afib_rt --row MIITT_Volunteer1:20:volunteer1:"Healthy volunteer":0 MIITT_Patient_2024Jan04_Cardiomyopathy_AFib:170:afib:"AF patient":90 --gamma 1.3 --out <path>` → `<path>.pdf` (one `--row` with both values; last field = LAX cut angle, 0 = view 1, 90 = view 2; half text width, placed beside `reference_conditioning.pdf`) | RT recon outputs + LV curves under `scratch/temp/miitt_afib_rt/` | docs/124 |

**Table numbers** (typed into the Overleaf tables and read by the runtime/VTC figure):

```bash
PYTHONPATH=training:. python _paper_figures/seg_allphase_dice_hd95.py --procs 16 --out temp/allphase_seg/seg_allphase.json
PYTHONPATH=training:. python _paper_figures/paper_results_table.py --seg temp/allphase_seg/seg_allphase.json --json temp/allphase_seg/paper_tables_v2.json
```

`temp/` is gitignored, so rebuild the JSON on a new machine (the first step takes about 2 min on 16 CPUs). The second
prints Tables 2–5 (AF/HRV main, AF/HRV ablation with Disp.) with 4 digits, leading 0 counted (0.883, 24.50, 167.4), as
in the paper since Overleaf `4df5183` (2026-09-26); all 181 table cells were checked against this output.

**2026-09-25 revision** (Overleaf commits `8a83bbe`, `18457e7`, `52afcc5`): the four PDFs there are the outputs of exactly the commands above.
The earlier versions (Overleaf `65c253f`) had different layouts (full-width one-row `reference_conditioning`,
taller `runtime_vtc_af_hrv` without point labels, taller `ablation_motion --tight`, full-width `real_rt` with both
LAX views and a dashed no-reference curve); to regenerate them, use this directory's git history.

**Editable PowerPoint version:** `export_pptx.py` runs the four commands above unchanged and writes **one deck,
`_paper_figures/paper_figures.pptx` (committed), one slide per figure at 1:1 physical size** (paper order);
scratch renders go to `temp/pptx_work`.
- **Every text** the figure draws is a native PowerPoint text box: tick labels, axis labels, titles, legend entries,
  headers and numbers, in Arial at the same size, colour, weight, rotation and position. Math becomes real text with
  sub/superscript runs (ℒ in Cambria Math).
- **Each axes panel** (image, curves, arrows, markers, ticks, spines) is its own transparent PNG at 600 dpi, cropped
  to what it draws. Each inset is a separate picture. Figure-level graphics (legend samples, header rules) are one
  more PNG.
- **Verified 2026-09-25**, against the Overleaf PDFs:
  - **Every picture layer** aligns at shift (0, 0), by best-shift search at 400 dpi (0.18 pt).
  - **Every text box** matches the PDF's words by string and position.
  - **Rendered glyphs** (LibreOffice render vs PDF raster, per text box): median offset 0 in both directions and
    almost all within ±0.18 pt. The exceptions are labels containing math glyphs (↓, †, →, ℒ), off by up to about
    1.8 pt, because PowerPoint draws them with Arial or Cambria Math instead of matplotlib's STIX, and a few
    labels off by about 0.4 pt.
  - The deck passes the pptx skill's validator.
- **How it stays exact (each fix answered a measured failure):**
  - Layers and text positions both come from the PDF backend, like the paper figure: Agg repacked legends and moved
    the crop box by 0.7 pt.
  - Text positions are recorded in the same bbox-cropped save as the layers.
  - Text is hidden with alpha 0, not `set_visible(False)`, because hiding it repacks legends.
  - Text is placed by **baseline**: a top-anchored box with the baseline `BASE_FROM_TOP` = 1.00 font-size below its
    top (measured in LibreOffice; PowerPoint unverified). The baseline sits max(descent of the text, descent of
    "lp") above matplotlib's bbox bottom, as in `Text._get_layout`. Centring boxes left text 0.9–1.8 pt low.
  - Width slack goes on the side away from the alignment edge.
  - Tick locators are frozen to the PDF draw, and rcParams are reset between renderers.
- It needs python-pptx, which is not in `svr`: `pip install --target <dir> python-pptx`, then
  `PYTHONPATH=<dir>:training:.`.

**Helpers in `tools/`:**
- Reference-conditioning selection: `pick_motion_candidates.py`, `sweep_input_frame.py`,
  `rank_input_frame_sweep.py`, `render_motion_edes_gallery.py`.
- VTC subject selection: `vtc_af12_candidates.py`, `vtc_pair_candidates.py`.
- Real RT: `render_rt_orthoviews.py`, `render_rt_lax_candidates.py`, plus `evaluation/src/engine/run_vggt_rt.py`.

`render_runtime_accuracy.py`'s own `main()` writes only the older `d1`–`d8` runtime drafts; its role here is the
shared style (`paper_rc`, palette, `TEXTW`).
