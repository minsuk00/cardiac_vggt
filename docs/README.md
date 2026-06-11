# docs/ index

Research findings, design decisions, and experiment write-ups for the VGGT-MRI project. Numbered
`NN_*.md`, each opening with a `> **TL;DR & takeaway**` blockquote (see `CLAUDE.md → Docs` for the
convention). Separate from per-version implementation logs in `version_history/`.

**This index is the entry point** — skim it to find the right doc, then open only that one. Add a
one-line pointer here whenever you create a new `docs/NN_*.md`.

| Doc | Summary |
|---|---|
| [`01_respiratory_motion_simulation.md`](01_respiratory_motion_simulation.md) | Respiratory motion simulation: literature survey (motion models, magnitudes, phantoms, SVR code), current-data format recap, and design plan. **Simulation core implemented** (2026-06-11): `training/data/respiratory.py` + `tests/test_respiratory.py` + example report `_html/06_respiratory_motion_simulation_examples.html`; GPU-aug/config/trainer wiring deferred. |
| [`02_related_work_literature_review.md`](02_related_work_literature_review.md) | Ranked related-work sweep for the project goal (single/few-frame-per-slice → volume at a target phase): INR SVR (NeSVoR/SVoRT), phase-conditioned template-deformation (5D-MoCo/M-DIP/STINR), real-time cine + gated→real-time gap, classical 5D MR baselines, and the VGGT/DUSt3R geometry lineage. |
| [`03_cardiac_svr_literature_review.md`](03_cardiac_svr_literature_review.md) | Focused classical/DL cardiac SVR sweep (52% adversarial kill rate). Confirms the open gap. Closest classical: Jantsch 2013 (free-breathing 2D cine stacks → 3D+t, many frames/slice). Closest DL: Chen 2024 (motion-correction + through-plane SR → 3D volume, breath-hold). Confirms XD-GRASP, TetHeart, DMCVR are NOT cardiac SVR baselines. |
| [`04_inference_information_contract.md`](04_inference_information_contract.md) | **Design stance** (not yet implemented): at inference assume the model knows only `z` per input slice — input cardiac `t` and respiratory `r` are UNAVAILABLE (one-frame-per-slice ⇒ no self-gating; no ECG/respiratory device assumed). Target queries (`target_t`/`target_r`) stay free (chosen, sim GT). Pin `target_r` (4D, correct-not-resolve) since input `r` is unseeable; ECG/self-gating are the fallback via input-phase dropout. |
