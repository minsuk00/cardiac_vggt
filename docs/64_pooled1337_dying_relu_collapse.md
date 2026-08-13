# 64 — Why the pooled1337 aug/noaug runs died: a dying ReLU in the DPT output head

> **TL;DR & takeaway**
>
> **Both pooled1337 long runs (SLURM 55996915 aug / 55996916 noaug) were dead from epoch ~17 of
> 300 and spent ~70 epochs × 2 GPUs × 2 days producing nothing.** The cause is **proven, not
> inferred**: `point_head.scratch.output_conv2[1]` — the `ReLU(inplace=True)` between the last two
> convs of the DPT head — is **100% dead** in both checkpoints. Every one of its ~34 M
> pre-activations is ≤ 0 (aug max = **−1.45e-05**, noaug max = **−0.138**), in **all 32 channels**,
> for arbitrary inputs.
>
> Everything observed in the logs follows from that one fact, and each link was measured:
> dead ReLU ⇒ its output is exactly 0 ⇒ the final 1×1 conv emits **only its bias** ⇒ the predicted
> DVF is **bit-exactly spatially constant** (per-channel spatial std `[0,0,0]`) and **completely
> input-independent** (two different random batches differ by **2.06e-12** relative) ⇒
> `loss_diffusion = ‖∇u‖²` is **exactly 0** (measured 1.02e-18) ⇒ and because a dead ReLU has
> derivative 0, **the entire gradient path upstream is severed**: parameter grad norms fall seven
> orders of magnitude across it, `grad_aggregator` → **1.56e-10** (base-model control on the same
> batch: **1.81**). The `grad_point ≈ 0.5` that kept appearing in the logs is *only* the final
> conv's **bias** gradient — the sole trainable parameter downstream of the dead ReLU. It is
> **mathematically unrecoverable**, and AdamW `weight_decay=0.05` then ground every now-gradient-free
> aggregator LayerScale gamma down **~3×** over the following 64 k steps.
>
> **Trigger — SETTLED BY EXPERIMENT (§3a): the peak LR causes it, and `5e-5` is safe.** A
> constant-LR sweep on the real pipeline from real base weights (`tools/exp_relu_death_vs_lr.py`,
> 4 arms × 3000 steps) shows a **bifurcation, not a smooth 1/LR scaling**: at **5e-5 and 1e-4** the
> system reaches a **stable equilibrium** — dead-channel fraction plateaus (0.20 / 0.35), `pre_max`
> stops falling and *recovers*, and `grad_aggregator` stays healthy at ~1.5e-2 / 2.1e-2. At
> **2e-4 and 3e-4** it collapses — `pre_max` falls **monotonically** toward 0 (1.93 → 0.91) and
> `grad_aggregator` craters to **1.3e-5 / 3.7e-5**, ~1000× below the stable arms. The threshold
> lies between **1e-4 and 2e-4**; the production runs sat at 3e-4, well past it.
> ⇒ **`5e-5` does not merely delay the collapse — it does not collapse.** (An earlier draft of this
> doc worried it might; the sweep refutes that, and the native-z refactor is exonerated as a
> sufficient cause since native-z at 5e-5 is stable for the whole run.)
>
> **Applied:** `PEAK_LR` → `5e-5` and `RESUME_FROM` cleared in both sbatch scripts (the collapsed
> checkpoints must not be resumed); **gradient-collapse alarm** (`optim.grad_collapse_alarm`, emails
> via `train_utils/notify.py`) and **weights-only best-by-val checkpointing**
> (`checkpoint.save_best`, `metric_psnr_3d_heartseg`) added — the failure was silent for 70 epochs
> and the peak weights were never saved.
>
> Repro: `PYTHONPATH=training:. python tools/probe_aggft_collapse.py --ckpt <checkpoint_last.pt>`

---

## 1. What the on-disk logs showed

Both runs, read with `tools/load_run.py` (docs/60). Val is **deterministic** in this
configuration — `metric_mse_heart_identity` = 0.0171 and `..._oracle` = 0.0030 are constant across
all 86 epochs — so epoch-to-epoch differences are pure model, not the redraw noise that
`docs/experiment_evaluation_instruction.md` §4.3 warns about.

| quantity | epochs 0–16 (healthy) | epochs 17–86 (dead) |
|---|---|---|
| `train/optim/grad_aggregator` | median 0.01–0.08 | **< 1e-6 at 100% of steps, every epoch** |
| `train/optim/grad_point` | 0.3–1.0 | 0.3–1.0 (unchanged — see §3) |
| `train/loss/diffusion` | ~1.3e-4 | **exactly 0.0** |
| `train/metric/mean_disp_norm` | ~0.05 | **0.012, pinned** |
| `metric_psnr_3d_motion` | peak 19.53 aug / 19.70 noaug | 19.03–19.13 (floor 18.53) |
| `metric_recov_frac_heart` | peak 0.248 / 0.281 | 0.15–0.17 |
| `val/resp/slope_dz` | 0.4–0.74 | **exactly 0.00000** |
| `val/ef/slope` | up to 0.22 | 0.05 (`mae_pct` pinned 54.8) |

Two independent corroborations that this is a hard stop, not a plateau:

- **The two arms became numerically indistinguishable** from epoch 17 on — aug vs noaug agree to 4
  significant figures (ep 30: 19.1031 vs 19.1005; ep 43: 19.0396 vs 19.0364) and per subject
  (`ACDC_patient006` t0 MSE 0.009534 vs 0.009515). Different augmentation cannot produce identical
  numbers unless neither model is learning anything data-dependent.
- **It is not overfitting**: train loss ≈ val loss (aug 0.0505 vs 0.0514) and train loss is *worse*
  than its own epoch-11 best (0.0396 → 0.0510). The model fits neither set.

Exact death step: **aug step 16550 (epoch 17)**, **noaug step ~15345 (epoch 16)**.

## 2. The proven mechanism

`tools/probe_aggft_collapse.py` loads a checkpoint, runs a real forward/backward, and measures
where spatial variance dies (forward) and where the gradient dies (backward). **A base VGGT-1B
control is run through the identical probe** — without it, "the gradient is zero" is worthless,
since a broken probe reports the same thing.

The head's output stack is
`output_conv2 = Sequential(Conv2d(128→32,3×3), ReLU(inplace=True), Conv2d(32→4,1×1))`.

| measurement | aug (dead) | noaug (dead) | native-z @ 5e-5, ep 3 | base VGGT-1B |
|---|---|---|---|---|
| pre-ReLU **max** | **−1.45e-05** | **−0.1375** | +9.38 | +61.07 |
| pre-ReLU mean | −0.245 | −11.92 | +0.299 | +1.550 |
| fraction of pre-activations ≤ 0 | **1.000000** | **1.000000** | 0.554 | 0.611 |
| channels dead **everywhere** (of 32) | **32/32** | **32/32** | 0/32 | 0/32 |
| post-ReLU std | **0.0** | **0.0** | 1.201 | 7.772 |
| conv2 out per-channel spatial std | **[0,0,0,0]** | **[0,0,0,0]** | [.008,.012,.014,.71] | [.21,.21,.06,5.48] |

The consequences, all measured on the aug checkpoint:

- **DVF is bit-exactly constant.** Per-channel spatial std `[0.0, 0.0, 0.0]`; **100.0%** of
  in-plane neighbour differences are *exactly* 0.
- **DVF is input-independent.** Two different random batches: `mean|dvf₁ − dvf₂| = 1.07e-14`,
  relative difference **2.06e-12**. The head ignores its input entirely.
- **`loss_diffusion` = 1.02e-18.** `diffusion_loss_l2` is the mean *squared* in-plane gradient of
  `u`; a constant field is its exact global minimum. This reproduces the log's `0.000000` precisely.
- **The gradient is severed at the ReLU**, not before or after it:

  | site | aug grad norm | base control |
  |---|---|---|
  | `dL/d(dvf)` | 2.53e-02 | 2.39e-01 |
  | `output_conv2` params (**bias only**) | 4.08e-02 | 1.13e+01 |
  | ← *the dead ReLU sits here* | | |
  | `output_conv1` | **2.42e-09** | 1.63e-01 |
  | `refinenet1` | 4.42e-09 | 6.77e-01 |
  | `layer4_rn` | 2.80e-11 | 5.93e-02 |
  | `projects` | 4.55e-11 | 2.80e-01 |
  | `dL/d(aggregator tokens[23])` | 1.06e-14 (99.13% exact 0) | 1.15e-04 (0.36%) |
  | **`grad_aggregator` (trainer-equivalent)** | **1.56e-10** | **1.81** |

- **The aggregator itself is fine.** Its output tokens are alive and spatially varying
  (`tokens[23]` std-across-patches 5.44e-2). The damage is entirely inside the head; the aggregator
  is merely starved.
- **`grad_point` stayed "healthy" for a dead model** because the final 1×1 conv's **bias** is the
  only trainable parameter downstream of the ReLU, and `dL/db ≠ 0` always. This is why the standard
  gradient-norm panel never flagged the failure.
- **Unrecoverable, then actively degraded.** No gradient can revive a dead ReLU. Meanwhile AdamW
  `weight_decay=0.05` kept shrinking the starved aggregator: every trainable LayerScale gamma is
  **2.9–3.6× smaller than base** (`frame_blocks.6.ls2` mean 0.0145 vs 0.0522; `frame_blocks.23.ls2`
  0.985 vs 2.863), matching `(1 − lr·wd)^64400 ≈ 0.42` for the zero-gradient steps since death.
  No non-finite values anywhere; **nothing exploded** (max |w| over all trainable aggregator params
  = 4.0).

## 3. What triggered it — and the honest boundary

**Certain:** the mechanism above. **Not certain:** what pushed the pre-activations over.

Evidence for the `3e-4` peak LR:

- It is a **6× override** of `default.yaml:403`'s `5e-5`, set by `PEAK_LR` in the sbatch recipe and
  documented there as a deviation. It applies to **~636 M pretrained** transformer params
  (aggft: everything but `patch_embed`) at **batch size 1** (pinned, docs/59 F9).
- Warmup is 5% × 300 = **15 epochs**. Both arms survived the entire ramp and died at **epoch 16–17**,
  1–2 epochs after reaching peak — within one epoch of each other on different data streams.
  Deterministic timing across two arms fits a drift-rate story, not a random event.
- **The noaug arm died with no gradient spike at all** (`grad_point ≈ 2.3, ~median`). This refutes
  the "one outlier batch" reading that the aug arm alone suggests (aug did have a 34.5 spike one
  step before). Further, aug survived **39 steps with `grad_point > 20`** in epochs 0–1 and a 23.2
  spike at epoch 12 — so spike magnitude alone is not sufficient. `grad_point`'s median decays
  monotonically 2.79 → 0.64 over epochs 0–17, i.e. the network is drifting steadily.
- Direction of drift, from the final weights: `output_conv2.0.bias` mean **+0.0139 (base) →
  −0.0069 (aug) / −0.0086 (noaug)**, fraction of negative bias channels **0.312 → 0.719 / 0.750**.
  ⚠️ Measured post-collapse, so partly weight decay — this is corroborating, not clean.

Against attributing it *solely* to LR:

- The previous good runs differed in **both** LR (5e-5) **and** the native-z refactor (docs/58:
  physical-z input, per-subject `D`, variable-`D` `V_canon`/`V_gt`). These are confounded.
- The only native-z-at-5e-5 checkpoint available (`214436485`, no LR override ⇒ default 5e-5)
  is **healthy** — 0/32 dead channels, post-ReLU std 1.20 — but it only reached **epoch 3**, and the
  collapsed runs were also healthy at epoch 3. **This control does not exclude native-z.**
- Note the drift is already visible at 5e-5: mean pre-activation 1.550 (base) → 0.299 (ep 3).
  Whether that stabilises or continues to zero at 5e-5 is **not established**.

## 3a. The controlled experiment that settled it

`tools/exp_relu_death_vs_lr.py` — real native-z pipeline, real data + augmentation + respiratory,
real loss, aggft freeze, real base VGGT-1B weights, AdamW `wd=0.05`, **constant LR** (no
warmup/cosine) so LR is the only variable. 4 arms × 3000 steps. Constant LR is what makes this
cheap: the production schedule ramped 1e-8 → 3e-4 over 15 epochs (~14,025 steps), so the model
only experienced peak LR for ~2,500 steps before dying — reachable from step 0 here.

Mean over the last 500 steps of each arm:

| LR | dead channels /32 | `pre_max` | `pre_mean` | `grad_aggregator` | verdict |
|---|---|---|---|---|---|
| **5e-5** | **0.208** | **10.07** | −0.102 | **1.46e-02** | **stable** |
| **1e-4** | 0.351 | 4.06 | −0.189 | **2.13e-02** | **stable** |
| 2e-4 | 0.714 | 0.91 | −0.413 | **1.34e-05** | collapsing |
| 3e-4 | 0.629 | 0.92 | −0.181 | **3.67e-05** | collapsing |

`pre_max` trajectory (the survival indicator; ≤ 0 ⇒ 100% dead):

| steps | 5e-5 | 1e-4 | 2e-4 | 3e-4 |
|---|---|---|---|---|
| 0–300 | 16.64 | 15.28 | 10.66 | 9.48 |
| 300–800 | 6.16 | 3.89 | 1.93 | 1.49 |
| 800–1400 | **10.99** | 4.42 | 1.40 | 0.93 |
| 1400–2000 | **12.00** | 3.91 | 1.27 | — |
| 2000–2600 | 8.92 | 4.35 | 1.00 | — |
| 2600–3100 | **10.06** | 3.95 | **0.91** | — |

**This is a bifurcation, not a rate difference.** Every arm dips early (normal fine-tuning
adaptation away from the natural-image init). Then the low arms *turn around*: 5e-5 recovers
6.16 → 12.00 and its dead-channel fraction plateaus at ~0.20 — the healthy ReLU attrition level.
The high arms never turn around: 2e-4 declines monotonically 1.93 → 0.91 with dead channels still
climbing (0.44 → 0.71) at step 3000, and its `grad_aggregator` is already ~1000× below the stable
arms and heading for the <1e-6 production signature.

Two conclusions this licenses that the observational data could not:

1. **5e-5 is safe.** The concern that it "only delays" the collapse is refuted — the drift reverses
   and settles. No arm at ≤1e-4 was trending toward death at step 3000.
2. **native-z is not a sufficient cause.** It is present in every arm, including the two stable
   ones, so it cannot by itself drive the ReLU to death. It plausibly moved the operating point
   closer to the cliff (§3 §2), but the LR is what pushes it over.

⚠️ **`chan_dead_everywhere` is a PER-BATCH proxy, not a count of permanently dead units.** It means
"≤ 0 at every pixel of *this* batch"; a channel can be dead for one batch and alive for the next.
Measured: the count goes **down almost exactly as often as it goes up** (5e-5: 50 rises / 48 falls;
2e-4: 42 / 41). So during the sweep those channels are **marginal, and they do recover** — the
0.20 plateau at 5e-5 is a population *equilibrium*, not accumulated damage.

That distinction matters for how the failure actually works, and it rules out a tempting-but-wrong
story ("permanent deaths accumulate one by one until none are left"). There is **no incremental
ratchet.** What LR controls is how far down the whole pre-activation *distribution* is pushed;
individual channels dip below and come back the entire time. Irreversibility arrives only at the
very end and all at once — when `pre_max` goes negative for the *entire input space*, `h` freezes
(nothing upstream gets gradient) and recovery stops being possible. One continuous slide, then a
cliff. Permanent death is a property of the **end state**, verified on the production checkpoints
against arbitrary random inputs (`pre_max` = −1.45e−05, 32/32 channels), not of the trajectory.

Caveat kept honest: no arm reached true whole-input-space death inside 3000 steps, so "collapsing"
for 2e-4/3e-4 is an extrapolation from a monotone `pre_max` → 0 plus a `grad_aggregator` already at
1e-5. The production 3e-4 run needed ~16.5k steps to finish dying, so that is expected.

Figure: `figs/relu_death_vs_lr.png`. Raw per-step records: `scratch/relu_lr_exp/lr_*.jsonl`.

## 4. What was changed

- `sbatch/train_pooled1337_dpt_{aug,noaug}.sh`: `PEAK_LR="3e-4"` → `"5e-5"`. Verified by resolving
  the config through Hydra with the script's real override string — **all three** LR knobs move
  (`optim.optimizer.lr`, warmup `end_value`, cosine `start_value`); the script's own comment warns
  that setting only the optimizer is a silent no-op.
- Same files: `RESUME_FROM` cleared to `""` → fresh-from-base. **Resuming was a trap**: both
  `checkpoint_50.pt` and `checkpoint_last.pt` are post-collapse (the epoch 10–15 peak was never
  saved), and per docs/37 `resume_checkpoint_path` is a *full* resume, so it would have restored
  `prev_epoch=86` and restarted the dead model 29% into the cosine decay.
- The deviation comment now records what 3e-4 did, so it is not re-raised blind.

**Not done, recommended:** a tripwire. Any of these catches it within minutes —
`grad_aggregator < 1e-6` for N consecutive steps, `loss_diffusion == 0` exactly, or directly the
dead-unit fraction of `output_conv2[1]`. Gradient clipping is **not** the answer: `max_norm=1.0`
was already active on both param groups throughout.

A structural fix (LeakyReLU/GELU in place of that ReLU) would remove the trap entirely but deviates
from the pretrained VGGT head — deliberately not done here.

## 5. Provenance

Machine-measured on an L40S, 2026-08-04, against `checkpoint_last.pt` of both runs
(git `503a8f0`, dirty), the aborted native-z run, and `scratch/base_weights/vggt1b_base.pt`:

- `tools/probe_aggft_collapse.py` — forward/backward probe, base-model control included.
- Log analysis via `tools/load_run.py` over `metrics.jsonl` / `val_per_subject.csv`.
- The base control is what makes the zero readings trustworthy: the same probe reports
  `grad_aggregator = 1.81` and an input-dependent DVF on base weights, so it demonstrably **fires**
  on a healthy model (cf. the fault-injection rule in docs/59 F2 and docs/62 §11.7).
- Figure: `figs/pooled1337_collapse_check.png`.

**Cited, not re-measured here:** docs/58 (native-z), docs/37 (`resume_checkpoint_path` is a full
resume), docs/59 F9 (batch size pinned to 1), docs/60 (on-disk run logs).
