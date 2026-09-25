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
| `reference_conditioning.pdf` | reference-conditioned reconstruction (fixed stack, ED vs ES reference) | `render_motion_edes.py` | `python _paper_figures/render_motion_edes.py --dump scratch/motion_edes/sweep80_save --af cmrx2024/CMRx24_Train_P007,6,CMRx24_Train_P007_z6f9 --hrv mnms/MNMs_E3L8U8,7,MNMs_E3L8U8_z7f0 --no-map --arrow-color "#F39078" --out reference_conditioning.pdf` | saved motion fields (GPFS) from `tools/sweep_input_frame.py`; eval bundles | docs/127 |
| `ablation_motion.pdf` | motion learned by each loss-ablation arm | `render_ablation_motion.py` | `python _paper_figures/render_ablation_motion.py --subject cmrx2023_af12/CMRx23_Train_P093 --tight --flip-v --no-map --arrow-color "#F39078" --out ablation_motion.pdf` | saved eval outputs (`ed_dvf.npz`, EF jsons, `_motion_epe/af12/`) | docs/127 |
| `real_rt.pdf` | real free-breathing RT (MIITT volunteer + AF patient) | `make_rt_figure.py` | `PYTHONPATH=training:. python _paper_figures/make_rt_figure.py --root scratch/temp/miitt_afib_rt --row MIITT_Volunteer1:20:volunteer1:"Healthy volunteer" MIITT_Patient_2024Jan04_Cardiomyopathy_AFib:170:afib:"AF patient" --out <path>` → `<path>.pdf` (one `--row` with both values) | RT recon outputs + LV curves under `scratch/temp/miitt_afib_rt/` | docs/124 |

**Table numbers** (typed into the Overleaf tables and read by the runtime/VTC figure):

```bash
PYTHONPATH=training:. python _paper_figures/seg_allphase_dice_hd95.py --procs 16 --out temp/allphase_seg/seg_allphase.json
python _paper_figures/paper_results_table.py --seg temp/allphase_seg/seg_allphase.json --json temp/allphase_seg/paper_tables_v2.json
```

`temp/` is gitignored, so rebuild the JSON on a new machine (the first step takes about 2 min on 16 CPUs). The rebuilt
HRV table matched the paper's appendix table exactly (2026-09-25).

**Verified 2026-09-25:** all four commands above regenerate the Overleaf PDFs (commit `65c253f`) pixel-identically.

**Helpers in `tools/`:**
- Reference-conditioning selection: `pick_motion_candidates.py`, `sweep_input_frame.py`,
  `rank_input_frame_sweep.py`, `render_motion_edes_gallery.py`.
- VTC subject selection: `vtc_af12_candidates.py`, `vtc_pair_candidates.py`.
- Real RT: `render_rt_orthoviews.py`, `render_rt_lax_candidates.py`, plus `evaluation/src/engine/run_vggt_rt.py`.

`render_runtime_accuracy.py`'s own `main()` writes only the older `d1`–`d8` runtime drafts; its role here is the
shared style (`paper_rc`, palette, `TEXTW`).
