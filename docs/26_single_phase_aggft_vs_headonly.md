# 26 — Single-phase aggregator-finetune vs head-only (ED), + single-phase `val_motion` logging

> **TL;DR & takeaway**
> Does finetuning the aggregator (aggft) beat training only the point-head, and does it depend on
> single- vs multi-phase targets? **Aggft wins in BOTH regimes on motion PSNR** (the honest
> dynamic-voxel metric; identity/do-nothing floor ≈ 20.1 dB at ED).
> - **Multi-phase:** head-only ≈ identity floor (does ~no motion correction); aggft **+3.4 dB**. Aggft
>   is effectively *required*.
> - **Single-phase (ED-only):** head-only is **NOT** at the floor — it recovers **+2.6 dB** (22.79 vs
>   20.15). This **corrects `docs/09`**, whose "head-only ≈ zero at ED" was a *multi-phase-specific*
>   result. Aggft still wins (**≥ +0.56 dB** over head-only, and rising).
> - **Why aggft helps even single-phase:** input slices are *always* scattered across cardiac phases
>   (`mri_data_mode: dynamic`), so fixing the *target query* phase does not remove the warp task.
> - **Methodology / honesty:** the first single-phase run was killed **undertrained** (ep62/200, LR
>   still ~82% of peak), so its number was an under-estimate. We re-ran a fair pair (single-phase,
>   z-only/no-t, no breathing, 100-epoch **fully-annealing** cosine, identical except the freeze).
>   Once aggft at **only 28% trained** already beat near-converged head-only, we **stopped both jobs
>   early** (binary settled, saved ~3 GPU-days). ⇒ aggft's *converged magnitude* in single-phase is
>   **not measured** (lower bound +0.56 dB; would be larger).
> - **Tooling fix:** single-phase (`t_target_fixed`) runs now log `val_motion` (previously gated to
>   multi-phase only — motion got dropped with the redundant per-phase panels).
> - **Status:** direction settled; converged single-phase aggft magnitude left unmeasured by choice.
> Scripts: `tools/eval_ed_motion_aggft_vs_headonly.py` (t-on), `tools/eval_ed_motion_fixedED_z_no_t.py`
> (t-off live); runs `sbatch/train_fixedED_z_no_t_{headonly,aggft}.sh`.

All numbers are **motion PSNR** unless stated — PSNR over only the dynamic heart voxels (~5–8% of the
bbox), the honest measure of motion correction (see `docs/09`). bbox/full are dominated by static
tissue and overstate everything. Eval protocol throughout: target phase = **ED (t=0)**, **n=30** val
subjects, one common pass, identity (Δ=0 splat) floor computed on the same subjects.

## The question and the mechanism

Original question: *for single-phase target reconstruction, does aggft add value over head-only, or is
it only needed for multi-phase?* The intuition was "multi-phase needs aggft, single-phase might be fine
with head-only."

**Key mechanism that reframes it:** `mri_data_mode: "dynamic"` is the default. Input-slice phases are
sampled **independently of the target** (`mri_dataset.py`: `t_sequence = [rng.randrange(T) …]`, seeded
by `seq_index` only). So "single-phase" (`t_target_fixed=0`) fixes only the *target query* to ED — the
*input slices still come from all 12 random phases* and must be warped to ED. The motion-correction
task is present in single-phase exactly as in multi-phase; it's driven by the scattered **inputs**, not
the number of target phases. So aggft should help in both — the only mode with no motion to correct is
`mri_data_mode: "static"` (all inputs at the target), which is not our pipeline.

## Evidence

### Family 1 — Multi-phase (re-eval at ED, t-on, this session)

| model | trains | motion | bbox | full |
|---|---|---|---|---|
| identity (Δ=0) | — | 20.13 | 29.72 | 31.35 |
| `mp_headonly` (219575690) | head only | 20.45 (**+0.3 ≈ floor**) | 29.74 | 29.52 |
| `mp_aggft` (219576158) | agg+head | 23.87 (**+3.4**) | 32.83 | 34.34 |

Multi-phase head-only does **statistically no motion correction** (≈ identity, even on bbox); aggft
adds +3.4 dB. Confirms `docs/09` Part A (mean-over-phases there: head-only 21.29 / aggft 23.44).
Mechanism: correcting motion needs the aggregator's cross-slice attention to fuse scattered-phase
slices and produce the warp; a frozen backbone + DPT head can only refine positions it's given.

### Family 2 — First single-phase attempt (t-on) was UNDERTRAINED → discarded

The pre-existing `fixedED` head-only checkpoint (`220164663`) gave motion **22.10** at ED — but the log
ends mid-epoch at **ep62 of a configured `max_epochs=200`** (killed by walltime, not a clean finish).
The schedule is 5% warmup → 95% cosine (5e-5→1e-8); at ep62 the cosine is only ~27% through ⇒ **LR ≈
4.1e-5, ~82% of peak**. The annealing phase never happened, so 22.10 is an **under-estimate**. A flat
val-bbox curve at *high* LR is not convergence. This motivated a fair re-run. (Also: motion is not
logged for single-phase runs — see the tooling fix below — so only one motion point was available.)

### Family 3 — Fair single-phase re-run (t-off, fully-annealing), DEFINITIVE for direction

Two fresh runs, **identical except the freeze pattern**: single-phase ED, z-only / **no input-t**, no
breathing, no aug, `max_epochs=100` so the cosine **fully anneals**, SLURM auto-requeue. head-only =
whole aggregator frozen (32.7M trainable); aggft = `frozen_module_names=[*patch_embed*]` +
`find_unused_parameters=true` (637.4M trainable). **Live read** (t-off, ED, n=30):

| model | state when read | motion | bbox | full |
|---|---|---|---|---|
| identity (Δ=0) | — | 20.15 | 29.74 | 31.42 |
| head-only | ep75/100 (near-converged) | **22.79 (+2.64)** | 31.72 | 30.67 |
| aggft | ep28/100 (**partial — lower bound**) | **23.35 (+3.20)** | 32.75 | 34.42 |

**head-only single-phase recovers +2.64 dB over the floor** — clearly *not* "≈ zero". And **aggft, at
only 28% trained, already beats near-converged head-only by +0.56 dB motion** (+1.0 bbox, +3.8 full)
and was still climbing (prior aggft kept rising to ~ep96). Since aggft's number is a lower bound, the
binary conclusion is locked: **aggft adds value over head-only in single-phase too.**

**Early-stop decision (deliberate):** once aggft (lower bound) beat near-converged head-only, both jobs
were cancelled — the *yes/no* was settled and continuing aggft ~3 more days only measures *how much*.
Consequence: aggft's converged single-phase magnitude is **unmeasured** (≥ +0.56 dB). head-only was
stopped at ep76 — near-converged (LR ~16% of peak, bbox plateaued) but not 100%.

## Conclusions

1. **Aggft beats head-only in both regimes.** Multi-phase: +3.4 dB (head-only dead at the floor) → aggft
   required. Single-phase: ≥ +0.56 dB and growing → aggft still worthwhile.
2. **Correction to `docs/09`.** "head-only ≈ zero at ED" holds only for *multi-phase* head-only. A
   single-phase-specialized head-only recovers ~+2.6 dB: a frozen backbone + head can memorize **one
   fixed-target warp** but cannot produce **12 target-conditioned warps**, so it collapses to the floor
   only in multi-phase. The real axis is *how much target-conditioned warping the frozen head must do*,
   not single- vs multi- per se.
3. **Don't cross-compare absolute dB across the t-on (Family 1/2) and t-off (Family 3) families** —
   different input-conditioning. Compare head-only-vs-aggft *within* a family.

### Caveats / open
- aggft single-phase **converged** magnitude not measured (stopped at 28%; lower bound +0.56 dB).
- head-only stopped at ep76 (near- not fully-converged); only one motion point (no per-epoch motion
  trajectory for the old-code runs).
- Family 3 is **z-only / no-t** (the deployment contract per `docs/04`), not the absolute-information
  ceiling (t-on would give head-only more to exploit).

## Tooling fix — single-phase `val_motion` logging

Motion PSNR was computed per-epoch only inside the per-phase accumulation block in
`trainer._update_and_log_scalars`, gated on `t_target_fixed is None`. So single-phase runs logged
bbox/full (via the standard meters) but **never motion** — the headline metric got dropped with the
*redundant per-phase panels* it was bundled with. Fix (purely additive, multi-phase byte-identical,
train path untouched): the per-phase block now runs in both modes (single-phase buckets under its one
fixed phase), and the logging loop emits **only `val_motion`** for single-phase (`val_motion/t{K}_…` +
`val_motion/mean_…` with the identity-floor baseline baked in); per-phase full/bbox stay suppressed
(already covered by `Loss/val_metric_*`). Verified: full suite 183 passed + a real fixed-phase smoke
train logged `val_motion`. **Future single-phase runs log motion live; the runs in this doc predate the
fix** (their motion came from the eval scripts). Code: `training/trainer.py`,
`tests/test_trainer_diagnostics.py`.

## Repro
- t-on multi-phase + old single-phase eval: `tools/eval_ed_motion_aggft_vs_headonly.py` →
  `scratch/ed_motion_aggft_vs_headonly.json`
- t-off live single-phase eval: `tools/eval_ed_motion_fixedED_z_no_t.py` →
  `scratch/live_eval/ed_motion_z_no_t.json`
- Runs: `sbatch/train_fixedED_z_no_t_headonly.sh`, `sbatch/train_fixedED_z_no_t_aggft.sh`
