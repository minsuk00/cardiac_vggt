# 66 — Fresh-bootstrap failure: 4wok regression audit and controlled experiment plan

> **TL;DR & takeaway**
>
> Fresh-from-base training still has no proven explanation for its failure to learn target-phase
> conditioning/LV contraction. The strongest facts are: (1) the original 4wok checkpoint recovered
> EF, while a current 4wok warm-start retains that capability; (2) healthy fresh pooled runs remain
> near EF slope 0.18 after about 136k steps; (3) current direct-12 preprocessing is voxel-identical
> to the old implementation for 293/294 accessible CMRxRecon2024 subjects, so preprocessing is not
> a cohort-wide cause; (4) fused AdamW is numerically negligible; but (5) current per-block
> `torch.compile` changes the first-step gradient and AdamW update far beyond the eager rerun floor,
> so compile must be tested in training and can no longer be dismissed as numerically identical.
> The next campaign should use one current fixed-12 negative hub, one bundled faithful old-4wok
> positive control, and eight one-factor arms. The first priorities are faithful old direct-12,
> compile-off, old index-normalized z, respiratory-off, and exact legacy respiratory. Compare by
> optimizer step—not epoch—and use the same CorSeg EF phase sweep for every checkpoint. A causal
> claim requires a fresh fail→success flip, a repeat, and final confirmation on pooled1337 native-z.

**Date:** 2026-08-08  
**Status:** audit and preregistered plan; cause remains open; experiments in this document have not
yet been launched.  
**Research target:** make fresh-from-base pooled1337 + native-z + gather training recover
patient-specific LV contraction/EF.

---

## 1. The question and the success condition

The model receives scattered single-frame-per-slice cine images. Slot 0 is a real reference slice
at the requested cardiac phase. The model must read contraction state from that slice, propagate it
through global attention, and move the other slices so the reconstructed volume matches the same
phase. The practical failure is therefore not merely low PSNR: a fresh model reconstructs a heart
whose LV volume changes too little with the requested phase.

EF is the primary outcome because it directly tests whether the reconstruction contains LV
contraction. It must be accompanied by two diagnostics:

1. a paired reference-swap/phase-transfer test, which holds the non-reference slices fixed and
   changes only the target-phase reference; and
2. reconstruction guardrails (`recov_frac_heart`, motion PSNR, and `hole_frac_heart`) so an EF gain
   cannot be bought by anatomically implausible holes or unrelated deformation.

The investigation is complete only when all three conditions hold:

1. one isolated intervention flips fresh training from the failing trajectory to sustained EF
   recovery;
2. that flip replicates from another fresh initialization; and
3. the intervention works in the intended pooled1337 native-z gather run, not only in the CMRx24
   fixed-grid diagnostic.

Warm-starting from 4wok is a working deliverable, but it is not the answer to the fresh-bootstrap
question.

---

## 2. Evidence sources and how they were checked

This audit used four distinct sources. They must not be conflated.

### 2.1 The actual historical W&B run

The authoritative historical run is:

- W&B id: `4wokxzov`
- on-disk/log family: `217720691_mri_volume_diffusion_dynamic_axial_Cine_combined`
- launch: 2026-06-24 01:35
- launch command family: `mri_volume_diffusion`, 200 epochs

The W&B-recorded source snapshot and launch configuration were inspected, rather than assuming the
state of a later Git worktree. The snapshot establishes these historical facts:

- direct preprocessing to spacing `(1.4, 1.4, 12.0)` and shape `(256, 256, 12)`;
- effective GPU seed 8400;
- 1000 optimizer steps per epoch and 200 epochs, or 200k total steps;
- eager attention blocks—no `torch.compile`;
- non-fused AdamW;
- base-weight loading missing only the two newly introduced z-embedder tensors.

CMRxRecon2024 has 12 mm physical slice pitch. The historical direct-12 operation was therefore a
z identity with respect to pitch, while still enforcing a 12-plane tensor.

### 2.2 `/home/minsukc/vggt-oldcode-4wok` is not the W&B snapshot

`/home/minsukc/vggt-oldcode-4wok` is a Git worktree at commit `13e37fb`. It is useful for reading
the old implementation, but it is not a byte-for-byte export of the source that W&B logged for the
actual run. In particular, its preprocessing constant is currently 8 mm. Replaying that worktree
against the now-correct 12 mm affines resamples 12→8 mm and is not a faithful reproduction of
4wok.

Any positive-control replay must be built in an isolated copy/worktree from the actual W&B source
snapshot and launch configuration. Merely editing the old worktree's spacing constant is not
sufficient provenance, and the existing worktree must not be changed in place.

### 2.3 Current source and current on-disk run records

Current behavior was established from the source plus each run's `run_meta.jsonl`,
`metrics.jsonl`, `val_per_subject.csv`, and checkpoint metadata. W&B was not used as the metric
source because the local logs contain the full scalar history and per-subject records.

### 2.4 Read-only GPU probes

Two requested audits were run on the current GPU node without editing repository code or source
data:

- old direct-12 preprocessing versus current fixed12 tensors, over every accessible CMRx24
  subject; and
- eager/compiled plus fused/non-fused numerical comparisons on one real fixed12 training batch,
  including forward outputs, objective, raw gradients, clipping, and one optimizer step.

Probe scripts and detailed machine-readable output were written only under `/tmp`. The probes did
not modify a checkpoint, cached source tensor, NIfTI, repository file, or Git branch.

---

## 3. What is already proven about the failure

### 3.1 The capability existed in 4wok

The conclusive historical analysis in doc 33 measured:

- raw EF slope 0.773 and Pearson 0.765;
- robust/honest Spearman about 0.55;
- leak-excluded Pearson about 0.68 and slope about 0.74.

Thus 4wok learned real patient-specific contraction, although it still under-contracted and the
small healthy validation cohort made exact regression estimates noisy.

### 3.2 Current warm-start works, while healthy fresh starts do not

Late-window results from the relevant local logs are:

| run | initialization / cohort | latest step | last-30 EF slope | last-30 Spearman | last-30 EF MAE | respiratory slope | recov. frac. |
|---|---|---:|---:|---:|---:|---:|---:|
| `213966986_..._4wok_..._pooled1337` | 4wok warm-start / pooled | 97,240 | **0.881 ± 0.060** | **0.578 ± 0.030** | **19.35 ± 2.07** | 0.931 | 0.513 |
| `214141126_..._dpt_aug_..._pooled1337` | fresh base / pooled / aug | 136,510 | **0.189 ± 0.041** | 0.129 ± 0.035 | 48.38 ± 0.76 | 0.906 | 0.422 |
| `214141126_..._dpt_noaug_..._pooled1337` | fresh base / pooled / no aug | 136,510 | **0.181 ± 0.031** | 0.106 ± 0.021 | 48.68 ± 0.60 | 0.912 | 0.415 |
| `213953345_..._nogather_..._pooled1337` | fresh base / pooled / gather=0 | 88,825 | 0.180 ± 0.033 | 0.123 ± 0.027 | 48.95 | 0.398 | 0.235 |

The warm-start proves that the current objective, native-z data, new splat, compile path, and
pooled cohort can preserve and reward an already discovered conditioning circuit. It does **not**
prove that these choices are neutral during discovery from base weights.

The two fresh pooled arms are the strongest negative evidence: they are healthy 5e-5 runs, not
the dying-ReLU failures from doc 64, and their EF plateau persists through roughly 136k steps.
Undertraining is therefore not a plausible primary explanation for those runs.

### 3.3 The earlier `cond_ratio` basin diagnosis is invalid

The earlier claim that training drove phase conditioning below its untrained value came from
`214366719_*` checkpoints trained at 3e-4. Those are the proven dying-ReLU runs in doc 64:
diffusion loss is exactly zero, respiratory slope is effectively zero, and the head emits a
spatially constant bias. A phase-sensitivity ratio near zero is mechanically guaranteed for that
dead model.

No causal conclusion about the healthy 5e-5 fresh-training plateau may use those `cond_ratio`
measurements. The bootstrap problem is supported by the warm/fresh outcome gap, not by that probe.

### 3.4 The current CMRx24-only EF readout is weak

The CMRx24 validation subset has only 28 usable CorSeg EF cases and GT EF standard deviation about
4.29 percentage points. Its slope standard error is approximately 0.5. The two diagnostic runs
therefore cannot reliably distinguish a moderate positive slope from zero:

| run | latest step | last-30 EF slope | Spearman | EF MAE | respiratory slope | recov. frac. |
|---|---:|---:|---:|---:|---:|---:|
| current CMRx24 fixed12, gather=0 | 45,120 | −0.501 ± 0.208 | −0.136 | 56.62 | 0.154 | 0.413 |
| current CMRx24 native-z, gather=0 | 45,355 | −0.333 ± 0.320 | −0.079 | 55.76 | 0.217 | 0.296 |

These are negative-looking trajectories, but the between-patient EF regression is underpowered
and both arms accidentally used gather=0, which also weakens their breathing learning. They are
supporting evidence only, not a decisive native-z or fixed12 verdict.

### 3.5 The 8 mm old-code replay is a confounded diagnostic

The existing old-worktree replay resamples the relabeled 12 mm data to 8 mm, increasing adjacent
slice correlation and then cropping to 12 planes. A prior matched-step offline scoring at about
29k steps gave EF slope 0.913, Spearman 0.449, and MAE 23.61, versus slope −1.108, Spearman
−0.282, and MAE 55.87 for the current fixed12 arm. This is evidence that the old path can produce
a contraction-like result under a denser/redundant z representation. It is **not** evidence that
the old recipe succeeds on the real 12 mm problem. The replay also lacked current on-disk metrics,
so all comparisons must be made by a common offline evaluator.

---

## 4. Requested preprocessing audit: old direct-12 versus current fixed12

### 4.1 Exact comparison performed

For all 294 accessible CMRxRecon2024 subjects, the audit compared tensors at the float16 cache
boundary:

- legacy direct pipeline: old preprocessing code with the actual W&B constants
  `(1.4,1.4,12.0)` and `(256,256,12)`;
- current fixed12 pipeline: current native-z preprocessing followed by the existing fixed12
  symmetric crop/pad wrapper.

The comparison used the current on-disk NIfTI arrays. It therefore isolates the preprocessing
implementation; it does not reconstruct the historical pre-flip/pre-roll array contents that
4wok saw in June.

Native D distribution in the 294 subjects was:

```text
D=6: 2, D=8: 9, D=9: 65, D=10: 89,
D=11: 118, D=12: 10, D=14: 1
```

### 4.2 Result

- phase tensors were exactly equal for **293/294** subjects;
- content masks were exactly equal for **294/294** subjects;
- the only exception was `CMRx24_Val_P011`, the sole D=14 subject;
- that subject is assigned to the current training split;
- for that subject, 6,361,429 of 9,437,184 phase voxels differed;
- maximum absolute difference: 0.01123046875;
- mean absolute difference over the tensor: 0.0008380030.

### 4.3 Mechanism of the sole mismatch

The legacy direct path center-crops D=14→12 before computing the phase-00 non-zero percentile
normalization. The current native path normalizes all 14 planes and the fixed12 wrapper crops
afterward. The two removed edge planes therefore slightly change the normalization statistics.

For D≤12, the legacy path pads before normalization, but the normalization ignores zeros. Padding
therefore does not change the percentiles, and the tensors are exact.

### 4.4 Verdict

Direct preprocessing is ruled out as a cohort-wide explanation. Only 1/235 current CMRx24
training subjects differs, and its discrepancy is understood. A full GPU arm devoted to old
versus current preprocessing is not justified.

This verdict has a precise scope: historical array reversals/rolls, cohort membership, and the old
z coordinate convention remain separate factors.

---

## 5. Requested numerical audit: compile and fused AdamW

### 5.1 Setup

The numerical test used the production model and loss on a real current fixed12 batch:

- subject `CMRx24_Test_P001`;
- S=10 observed slices, D=12 output planes, `z_scale=7.5`;
- staged `scratch/base_weights/vggt1b_base.pt`;
- effective seed 12600;
- bf16 autocast;
- 930 trainable tensors, 637,402,404 trainable elements;
- patch embed frozen;
- production compile configuration: 48 attention blocks, `mode=default`, `dynamic=True`;
- AdamW: lr 5e-5, weight decay 0.05;
- actual separate gradient clipping for aggregator and point head, both at norm 1.0.

The test included two independent eager backward passes to measure the same-code numerical floor.
This matters because GPU scatter-add in the splat is not bit-deterministic even when the model
outputs are exact.

### 5.2 Eager versus eager: the numerical floor

The two eager model forwards produced exactly equal DVFs, world points, and confidence. Only the
splat volume and coverage differed at floating-point scatter noise:

| measurement | eager vs eager |
|---|---:|
| output global relative L2 | 4.93e-9 |
| output cosine | 0.99999999999985 |
| output max absolute difference | 4.77e-7 |
| gradient global relative L2 | 4.10e-4 |
| gradient cosine | 0.999999916 |
| gradient max absolute difference | 1.84e-5 |
| post-step parameter relative L2 | 3.24e-5 |
| update-vector relative L2 | 0.0233 |
| update-vector cosine | 0.999729 |

The first Adam step amplifies small differences in near-zero coordinates, but the eager update
directions remain almost identical.

### 5.3 Compiled versus eager

| measurement | compiled vs eager |
|---|---:|
| objective | 0.102794148 vs 0.102799460 |
| objective absolute / relative difference | 5.31e-6 / 5.17e-5 |
| DVF relative L2 | 6.63e-4 |
| DVF max absolute difference | 0.002387 |
| all-output relative L2 | 0.01487 |
| raw gradient norms | 3.12567 vs 1.01950 |
| raw-gradient relative L2 | **2.370** |
| raw-gradient cosine | **0.7799** |
| post-step parameter relative L2 | 0.001440 |
| update-vector relative L2 | **1.034** |
| update-vector cosine | **0.4775** |

The forward objective is close, consistent with prior compile audits. The backward/update is not.
The difference is far above the eager rerun floor.

The supported mechanism is: bf16 reassociation across the compiled attention blocks makes small
DVF changes; the splat then uses `floor`, in-bounds tests, trilinear weights, scatter-add, and
coverage normalization. Small coordinate changes can cross a voxel-cell or validity boundary,
making the piecewise backward path much less similar than the scalar losses suggest. The observed
near-unit V-canonical maximum difference and 2.91 coverage maximum difference are consistent with
that amplification.

This is a one-batch, first-update measurement. It proves that compile is **not gradient-identical**
for this objective. It does not prove that compile prevents EF bootstrap over thousands of steps.
The warm-start's success under compile also shows that compile does not erase an already learned
conditioning circuit. The live hypothesis is narrower: compile may alter which basin fresh
optimization enters.

The probe ran on an L40S node, while the proposed `spgpu` jobs generally use A40s. Kernel and
compiler behavior can be hardware-dependent, which is an additional reason to test compile-off in
the actual training environment rather than extrapolate from this probe.

### 5.4 Fused versus non-fused AdamW

Using identical eager parameters and identical clipped gradients:

| measurement | fused vs non-fused |
|---|---:|
| aggregator clip norm | identical: 0.104486 |
| point-head clip norm | identical: 1.014131 |
| post-step parameter relative L2 | 1.74e-8 |
| post-step cosine | 0.99999999999984 |
| post-step max absolute difference | 4.77e-7 |
| update-vector relative L2 | 1.25e-5 |
| update-vector cosine | 0.999999999923 |

The production compiled+fused versus legacy eager+non-fused result was effectively identical to
compile-only: update relative L2 1.034 and cosine 0.4775. Compile dominates; fused AdamW does not
deserve a separate long training arm.

---

## 6. Full old-4wok versus current comparison

The table separates verified deviations from factors that are already neutralized or remain live.

| axis | actual 4wok | current fixed12 / current pipeline | evidence and status |
|---|---|---|---|
| z preprocessing | direct 12 mm, D=12 | native-z cache, then fixed12 crop/pad | Measured exact for 293/294; **cleared** as systemic cause. |
| physical pitch | CMRx24 12 mm | CMRx24 12 mm | Same physical acquisition. The 8 mm replay is confounded. |
| z coordinate supplied to model | `(z-5.5)/5.5`, one unit=66 mm | physical `z_mm/90`, then fixed D wrapper | **Live.** Fixed12 does not restore old z normalization. |
| splat z conversion | index-grid convention, scale 5.5 | physical convention, `z_scale=7.5` at 12 mm | Coupled to the z-coordinate arm; must change consistently. |
| variable D / native z | always D=12 | fixed12 diagnostic pads/crops; target pipeline native D | Fixed12 neutralizes D but not physical-z encoding. CMRx EF readout is weak, so not fully resolved. |
| cohort | CMRx24 only | fixed12 CMRx24; final target pooled1337 | Cohort is not conclusively ruled out by CMRx EF because n=28 is underpowered. |
| phase reference | real slot-0 target-phase slice | same | Contract is unchanged. |
| trainable modules | aggregator finetune + point head; patch embed frozen | same effective regime | Verified relevant contract. |
| base weights | VGGT-1B; z embed missing and random | same base; same two missing z tensors | Base is the same; z initialization differs through seed. |
| effective seed | 8400 | 12600 | **Live but narrow:** the main random model difference is z embed initialization. |
| subject sampling | 1000 with-replacement draws/epoch | one exact pass, about 235 CMRx steps/epoch | **Live.** Changes exposure counts and RNG stream. |
| epoch length | fixed 1000 | dataset size (~235 CMRx, 935 pooled) | **Live.** Never compare by epoch. |
| total schedule | 200k steps, 10k warmup | fixed12 70.5k, 3,525 warmup | **Live but lower priority.** Pooled warmup is 14,025 and still fails. |
| respiratory simulator | per-slot phase, amplitude, direction; tilt ≤30° | per-slot phase, one subject-wide amplitude/direction; tilt ≤45° | **Live.** Current one-frame sampling makes per-plane phase equivalent to per-slot phase. |
| moderate affine/photometric augmentation | absent/different legacy path | current moderate GPU augmentation | **Refuted** by the healthy pooled aug/noaug pair. |
| gather loss | absent/zero in old recipe | fixed12 arm gather=0; final target gather=0.5 | Pooled gather0 and gather0.5 both fail EF; gather strongly affects breathing, not the fresh EF failure. |
| attention execution | eager | 48 blocks compiled | **Live and promoted** by the gradient/update probe. |
| AdamW implementation | non-fused | fused | **Cleared** by one-update measurement. |
| splat implementation | legacy boundary behavior | rewritten physical/native-z splat | Lower priority: fixed12 cuts off-slab slots and warm-start succeeds with new splat. Bootstrap-only interaction remains possible. |
| slice direction on disk | historical base-at-z0 | current apex-at-z0 | Actual data deviation; conditional in-memory test only. |
| odd-D roll | historical pre-roll state | current corrected roll | Actual data deviation; lower priority, no contraction-specific mechanism established. |
| split | old 240/30/31 naming/availability | current 235/29/30 CMRx subset | Small membership deviation; low priority. |
| one-frame-per-slice semantics | one in-FOV frame/plane | same effective fixed12 semantics | Cleared by code comparison. |
| inactive t embeddings | present but disabled | removed | No-op; both active configurations did not use them. |
| RoPE table size | larger legacy table | smaller table | Used rows are identical; cleared. |
| input deferral | legacy behavior | current deferral path | Numerically negligible/no-op for this question. |
| precision | bf16 training | bf16 training | Same high-level precision. Compile changes reassociation within it. |

### 6.1 Important z-coordinate arithmetic

For 12 planes at 12 mm pitch:

```text
old:     z_old = (k - 5.5) / 5.5 = z_mm / 66
current: z_new = z_mm / 90
```

Thus old z inputs span 90/66 = 1.3636 times the current values. A valid one-factor old-z arm must
change all coupled representations together:

```text
z_indices                 *= 90/66
scanner_coords[..., 2]    *= 90/66
z_scale                   /= 90/66   # 7.5 -> 5.5
```

Changing only the z embedding input while leaving scanner coordinates or splat scale unchanged
would create an inconsistent geometry and would not test the historical convention.

### 6.2 Respiratory difference precisely stated

Under one-frame-per-slice, both eras effectively sample a separate respiratory phase per slot.
The real change is the displacement field's amplitude and direction:

- legacy: amplitude, tilt, and azimuth are sampled independently per slot; tilt up to 30°;
- current: amplitude, tilt, and azimuth are sampled once per subject and expanded to all slots;
  only the phase-dependent scalar varies; tilt up to 45°.

The current simulator is therefore subject-coherent/rank-1 across slots. Whether that specifically
occupies the global-attention pathway needed for phase conditioning is a plausible but unproven
hypothesis. The counterargument is strong: legacy per-slot breathing was less coherent and arguably
harder, yet 4wok bootstrapped.

### 6.3 Steps per epoch are not cosmetic

Old 4wok used 1000 steps per epoch; current CMRx fixed12 uses about 235. Consequently:

```text
old epoch 15       = 15,000 optimizer steps
current epoch 15   =  3,525 optimizer steps
old epoch 100      = 100,000 optimizer steps
current epoch 100  = 23,500 optimizer steps
```

Epoch boundaries can also affect sampler reseeding, random streams, validation cadence, saving,
and the learning-rate schedule. Every comparison below is preregistered by global optimizer step.

---

## 7. What has been ruled out, weakened, and left open

### 7.1 Cleared or not worth a dedicated long run

- **Current fixed12 preprocessing implementation:** 293/294 phase tensors exact; sole mismatch
  understood and small in cohort scope.
- **Fused AdamW:** update-vector relative difference 1.25e-5 on identical gradients.
- **Moderate augmentation:** pooled aug/noaug fresh pair is a clean null.
- **Gather as the EF cause:** fresh gather0.5 and gather0 both fail EF, though gather matters for
  breathing and reconstruction coverage.
- **High LR collapse:** doc 64 is a separate solved failure; all runs considered here are healthy
  5e-5 runs.
- **T embeddings, used RoPE rows, fixed/derived grid shape, and input deferral:** verified no-ops or
  negligible.
- **A systematic splat off-slab failure:** fixed12 reduced off-slab slots from about 11% to about
  1.6% without recovering EF.

### 7.2 Weakened but not logically eliminated

- **Undertraining:** refuted for the fresh pooled runs through 136k steps, but a 15k/29k historical
  positive-control comparison is still needed.
- **LR warmup:** current pooled warmup (14,025 steps) is longer than old 4wok's 10k and still fails.
  The full old 200k-step cosine trajectory remains an actual deviation.
- **Native/variable z:** fixed12 is negative-looking, but it keeps physical z and its EF evaluator
  is underpowered. It does not rule out the z coordinate convention.
- **Pooled cohort:** CMRx-only arms are negative-looking but underpowered. The faithful old direct12
  control is required before cohort size can be cleared.
- **New splat:** warm-start works and fixed12 reduces its obvious boundary exposure. A subtle
  fresh-bootstrap interaction is still possible but lower priority than compile/z/respiratory.

### 7.3 Live hypotheses

Ranked by current evidence:

1. **Compile changes fresh optimization.** Directly measured first-step gradient/update divergence;
   long-run causal effect untested.
2. **Old index-normalized z is needed for bootstrap.** Every current failure, including fixed12,
   still uses physical z/90.
3. **Respiratory restructuring competes with the phase pathway.** Never tested respiratory-off;
   exact old behavior also untested in current code.
4. **Historical z-embed initialization.** Only two base-missing tensors are randomly initialized;
   seed 8400 versus 12600 can matter even in a large pretrained model.
5. **Sampling/exposure and epoch/RNG clock.** Old used with-replacement 1000-step epochs; current
   sees exact dataset passes.
6. **Full old LR trajectory.** Lower priority because pooled warmup already exceeds the old warmup.
7. **Historical array ordering or another unrecorded environment interaction.** Conditional only
   if the faithful old control fails on current arrays.

No evidence presently supports the claim that “seed is the cause.” Seed is one isolated arm. If
the historical seed succeeds, it must succeed again at another seed before stochasticity is called
causal.

---

## 8. Proposed ten-run campaign

The campaign has one repeated negative hub, one bundled historical positive control, and eight
one-factor deviations from the hub. The bundled positive control is deliberately **not** a causal
arm: it answers whether the regression premise is real before attributing it to one knob.

### Shared hub contract

Except where a row explicitly changes one item, current-code arms use:

- CMRx24 current split and fixed12 tensors;
- fresh VGGT-1B base initialization;
- effective seed 12600;
- current physical z/90 and `z_scale=7.5`;
- current reference-slot conditioning and aggft freeze pattern;
- one-frame-per-slice;
- current coherent respiratory simulator;
- current augmentation setting from the existing fixed12 hub;
- gather=0, matching the existing fixed12 hub and historical no-gather objective;
- bf16, compile on, fused AdamW;
- lr 5e-5 and current fixed12 schedule;
- identical train/validation subject lists;
- step checkpoints and common offline CorSeg evaluation.

If resolved-config or Git-SHA comparison shows the existing fixed12 run cannot serve as the exact
hub, C0 is mandatory. Otherwise C0 still provides a reproducibility estimate for a stochastic
bootstrap outcome.

### C0 — current fixed12 hub repeat

**Change:** none.  
**Tests:** whether the existing negative trajectory is reproducible under the exact code and
evaluation harness used for the new arms; establishes run-to-run variance and catches an invalid
control setup.  
**Interpretation:** if this repeat recovers EF, no single treatment arm can be interpreted without
additional baseline seeds.

### P0 — faithful actual-4wok direct-12 positive control

**Change:** bundled historical recipe, reconstructed from W&B source/config—not from the 8 mm
worktree replay. Use direct 12 mm/D12, old index z, legacy respiratory, eager blocks, non-fused
AdamW, seed 8400, old sampling/1000-step clock, and old 200k-step schedule. Use current on-disk
arrays initially.  
**Tests:** whether the actual old recipe can freshly bootstrap on real 12 mm data today. This is
the premise test.  
**Interpretation:**

- succeeds: there is a reproducible recipe regression, and the one-factor arms can localize it;
- fails: do not call a current-code bug. First test historical array order in memory and dependency
  provenance; 8 mm redundancy/information content becomes a leading explanation.

### A1 — compile off

**Only change:** `compile_attention_blocks=false`; keep fused AdamW on.  
**Tests:** whether the measured compile-induced gradient/update trajectory changes fresh phase
bootstrap on production A40s.  
**Why high priority:** it is the only formerly “numerically identical” knob now measured to produce
a first-update direction with cosine 0.4775 versus eager.  
**Interpretation:** a win makes compile a bootstrap-sensitive optimization factor, not a generally
broken forward path. Replicate before changing production defaults.

### A2 — exact old index-normalized z

**Only change:** use the coupled old z convention: scale `z_indices` and scanner z by 90/66, and
scale `z_scale` by 66/90.  
**Tests:** whether the old coordinate range and positional-embedding excitation are required for
phase bootstrap.  
**Guard:** apply all three coupled changes; changing only the token embedding is invalid geometry.

### A3 — respiratory simulation off

**Only change:** disable respiratory simulation; leave moderate affine/photometric augmentation
unchanged.  
**Tests:** the whole hypothesis that respiratory learning competes with discovery of target-phase
conditioning. No production run has tested this with statistical power.  
**Interpretation:**

- no EF recovery: close the respiratory-competition family;
- recovery: run A4 and, if necessary, decompose amplitude/direction/tilt.

Respiratory metrics are expected to be degenerate in this arm and are not failure criteria.

### A4 — exact legacy per-slot respiratory simulator

**Only change:** replace current subject-wide amplitude/direction with independently sampled
per-slot amplitude, azimuth, and tilt; restore the 30° tilt maximum. Keep respiratory phase and
all other training choices fixed.  
**Tests:** whether the specific coherent/rank-1 simulator—not respiratory motion generally—is the
regression.  
**Implementation:** runtime/in-memory code in an isolated worktree; no data-on-disk changes.

### A5 — historical z-embed initialization only

**Only change:** initialize the two base-missing z-embedder tensors exactly as effective seed 8400
would, then restore the current seed/RNG state before any sampling or augmentation.  
**Tests:** whether the historical random z basis provided a rare favorable bootstrap.  
**Why this is not “large networks are seed-sensitive”:** almost the entire 1B model comes from
pretrained weights. The isolated random difference is a small positional module at a critical new
input channel.  
**Interpretation:** one successful seed is insufficient. Confirm with at least one independent
fresh seed before concluding stochastic initialization is causal.

### A6 — with-replacement subject sampling

**Only change:** sample subjects with replacement as in the legacy loader, while retaining the
current 235-step epoch boundary, RNG clock, and LR schedule.  
**Tests:** whether unequal/repeated early subject exposure helps discover phase conditioning.  
**Separation from A7:** this arm changes the subject distribution but not what one epoch means.

### A7 — legacy 1000-step epoch/RNG clock

**Only change:** define training/validation/reseeding boundaries at 1000 optimizer steps while
preserving the current flattened exact-pass subject stream and current LR as a function of global
step.  
**Tests:** effects caused by epoch boundaries—reseed cadence, augmentation sequence, validation,
and checkpoint clock—separately from replacement sampling or LR.  
**Guard:** compare at identical global steps; “epoch 15” is not a matched point.

### A8 — legacy global-step LR trajectory

**Only change:** 200k total-step cosine schedule with 10k warmup and peak 5e-5, keyed directly to
global optimizer steps. Preserve current sampling and 235-step epoch boundaries.  
**Tests:** whether early integrated learning rate or late decay, rather than the nominal peak LR,
controls bootstrap.  
**Priority:** lower than compile/z/respiratory because the failed pooled run already had a 14,025
step warmup.

### Recommended launch order

Do not spend all ten allocations before the premise test is readable.

**Wave 1:** C0, P0, A1, A2, A3, A4, A5.  
**Wave 2 if Wave 1 has no clear winner:** A6, A7, A8.  
**Conditional reserve, not in the ten:** historical base-at-z0 in-memory array reversal.

The lower-priority arms can be queued immediately if capacity would otherwise sit idle, but they
must not delay P0/A1/A2.

---

## 9. Common evaluation protocol

### 9.1 Compare checkpoints by step

Save and score at least:

- 15k steps: the approximate point where the successful replay reportedly began showing LV
  contraction;
- 29k steps: matched old-replay comparison;
- 45k steps: current fixed12 mature checkpoint;
- 60k steps if the trajectory remains ambiguous.

Validation frequency may still be expressed in epochs internally, but analysis tables and plots
must use global optimizer step.

### 9.2 Use CorSeg everywhere

Use the same fast CorSeg evaluator that training uses, not the slower nnU-Net offline evaluator.
For every checkpoint:

1. reconstruct a full 12-phase sweep under a fixed deterministic validation sampling protocol;
2. run CorSeg on every predicted and GT phase;
3. derive LV volume, EDV, ESV, and EF identically;
4. record per-subject outputs, not only aggregate scalars;
5. bootstrap subject-level confidence intervals for slope, Spearman, and MAE.

Old and current checkpoints must be loaded with `strict=false` and scored by this one evaluator.
Their in-training validation numbers are not comparable because the historical validation protocol
and logger differ.

### 9.3 Primary and diagnostic outcomes

**Primary:**

- EF slope;
- EF Spearman;
- EF MAE.

**Direct conditioning diagnostic:** hold all non-reference slices fixed, switch only the reference
between ED and ES or across all phases, and measure:

- predicted LV-volume response versus GT response;
- reconstructed change outside the reference plane;
- a phase-transfer coefficient normalized to the GT change.

This paired test has more power than between-person EF on the narrow healthy CMRx cohort and
directly asks whether slot 0 controls the rest of the volume.

**Guardrails:**

- `recov_frac_heart` must increase;
- motion PSNR must not decrease materially;
- `hole_frac_heart` must not increase;
- monitor respiratory slope/EPE for every respiratory-enabled arm;
- monitor dead-ReLU/aggregator-gradient alarms to keep doc 64's separate failure out of this study.

### 9.4 Decision rule

The pooled healthy fresh runs cluster near slope 0.18/Spearman 0.1, while working warm-starts are
near slope 0.88/Spearman 0.58. For screening, a candidate should show a sustained movement into the
middle of that gap—approximately slope ≥0.45 and Spearman ≥0.30—at multiple step-aligned
checkpoints, with the paired phase-transfer diagnostic agreeing and no guardrail failure.

The CMRx-only n=28 slope must not be used as a binary early-stop rule because its uncertainty is
too large. A strong visual/paired phase-transfer signal at 15k should continue to 29k even if the
single slope estimate is noisy.

A candidate becomes causal only after:

1. it beats C0 under the identical protocol;
2. the result repeats from a fresh seed; and
3. the same one-factor change makes a fresh pooled1337 native-z gather=0.5 run recover EF on the
   n≈132 pooled validation cohort.

The final ship rule from doc 38 still applies after EF recovery: `recov_frac_heart` and motion PSNR
must rise without an increase in heart holes.

---

## 10. Deviations intentionally not assigned a first-wave GPU arm

### Direct old preprocessing

Not run: voxel audit cleared 293/294 subjects and explained the one D=14 mismatch.

### Fused AdamW

Not run: identical-gradient one-update difference is negligible and compile dominates the
production-versus-legacy numerical difference.

### Moderate augmentation

Not run: already refuted by the healthy pooled aug/noaug pair.

### Gather loss

Not run as the EF cause: fresh pooled gather0 and gather0.5 both fail. Gather remains required for
the final intended breathing/coverage behavior.

### Native-z versus fixed D alone

Not repeated immediately: the existing pair is suggestive but underpowered. The more precise live
factor is old index z, which fixed12 never restored. The final winner must still be verified on
native-z.

### Base-first array direction

Conditional only. Most current CMRx arrays were reversed after the historical run. If P0 succeeds
on current apex-first arrays, this factor is automatically ruled out. If P0 fails, reverse arrays
and masks only in memory for one faithful-control rerun. Never edit affines or data on disk.

### Odd-D roll

Conditional and below base-first. It affected many subjects, but no contraction-specific mechanism
has been demonstrated. Undo only in memory and only according to the recorded affected-subject
manifest.

### Historical split

Low priority. Old/current training membership differs by only a few subjects. Test only after
sampling/clock factors fail.

### Legacy splat boundary code

Low priority. Warm-start succeeds under the new splat and fixed12 removes most off-slab exposure.
If compile/z/respiratory/sampling all fail despite a successful P0, a one-factor old-splat arm is a
reasonable second campaign.

### Exact old software environment

Not a first-wave arm because the source-level deviations above are more direct and an exact 2026-06
CUDA/PyTorch environment may not be reconstructable. Preserve dependency versions from W&B and
SLURM logs. If P0 fails under the current environment after historical array-order tests, software
drift becomes a live premise-level factor.

---

## 11. Isolation, safety, and reproducibility requirements

- Do not modify the main working tree's model, trainer, or dataset code.
- Create a dedicated Git worktree/copy per implementation family; never `git switch` the shared
  working tree.
- Do not edit NIfTI arrays, affines, cached tensors, checkpoints, or split files in place.
- Implement z/order/respiratory treatments in memory in the isolated experiment worktree.
- Submit on account `jjparkcv0`, partition `spgpu`.
- From an interactive allocation, clear `SLURM_*` before invoking self-submitting scripts.
- Stage large checkpoints to node-local `/tmp` before `torch.load`.
- Record the resolved Hydra config, Git SHA, source snapshot hash, GPU model, dependency versions,
  effective seed, subjects/step, and optimizer step in each log directory.
- Mirror all scalar and per-subject measurements to disk; W&B is visualization/remote tracking, not
  the only record.
- Keep old/current scorer inputs and random validation samples identical and save their identifiers.
- No experiments were running in `squeue -u minsukc` at the time this document was written; the
  results above are the final available local records, not live-job estimates.

---

## 12. Decision tree after the first results

1. **P0 fails at 12 mm:** the historical-regression premise is not established. Test base-first
   arrays in memory, then dependency provenance. Do not tune current knobs based on the 8 mm replay.
2. **P0 succeeds and A1 succeeds:** replicate compile-off, then confirm compile-off on fresh pooled
   native-z gather. Compile is a bootstrap-sensitive factor.
3. **P0 succeeds and A2 succeeds:** replicate old-z, then test whether a physically meaningful
   alternative scaling can preserve pooled/native-z generality. Do not blindly ship an index ruler
   across variable pitches.
4. **A3 succeeds:** respiratory competition is real. A4 distinguishes “any respiratory” from the
   coherent current simulator; only then decompose amplitude/direction/tilt.
5. **A5 alone succeeds:** test at least two more z-embed seeds. Treat success probability, not seed
   identity, as the result.
6. **A6/A7/A8 succeeds:** repeat the smallest winning behavioral change and verify at matched steps;
   avoid bundling replacement, epoch length, and LR until their individual effects are known.
7. **P0 succeeds but no one-factor arm succeeds:** inspect the exact P0-versus-hub source/config diff
   again. Next candidates are base-first data, legacy splat, historical split, and dependency drift,
   still one at a time.
8. **Any candidate wins fixed12:** immediately run the decisive fresh pooled1337 native-z gather=0.5
   confirmation. Fixed12 is a diagnostic bridge, not the project destination.

---

## 13. Current conclusion

The available evidence does not justify blaming seed, native-z, pooled data, respiratory motion,
or compile as the cause yet. It does justify narrowing the next work:

- the old 8 mm replay cannot serve as the positive control;
- current direct-12 preprocessing and fused AdamW do not merit long arms;
- compile is not numerically gradient-identical and is now a top training hypothesis;
- old index-normalized z remains untested even in fixed12;
- respiratory-off has never been tested at useful scale;
- epochs must disappear from all cross-run causal comparisons in favor of optimizer steps.

The fastest route to an answer is therefore not another broad pooled run. It is a faithful W&B
old-4wok direct-12 premise test plus the compile/z/respiratory one-factor arms, evaluated at 15k and
29k steps by the same CorSeg phase-sweep protocol.
