# 47 — Train-step profiling & `torch.compile` evaluation

> **TL;DR & takeaway**
> Full per-step profiling of the active 1-frame training pipeline (`mri_volume_diffusion`,
> S≈10, aggft, bf16+TF32+gradient-checkpointing) shows the step is **68% backward / 29% forward**,
> and the forward is **~87% the aggregator's 24×24 attention blocks** (`point_head` ~14%, splat
> <1%, dataloader ~0 once the /tmp monai cache is warm). ~33% of kernel time is **FlashAttention +
> cuBLAS GEMM that `torch.compile` cannot accelerate on any torch**. **On the pinned torch 2.3.1,
> `torch.compile` is a net LOSS: 4–6% SLOWER in every whole-model mode** (default AND max-autotune,
> model-only AND full-step), it **FAILS under DDP** (autocast/fp32-bias bug in the DPT head — and the
> trainer is DDP-wrapped even on 1 GPU), and **compiling the attention Blocks — where ~90% of the
> compute is — CRASHES** (`FakeTensor`/RNG-state error: the 2.3.1 `torch.compile`+activation-
> checkpoint composition bug). The one part that compiles cleanly (frozen DINO backbone, forward-only)
> is only ~4% of the step and already optimal → 0 gain. So the badness is **partly a memory wall
> (checkpointing is mandatory at S≥10; min-cut recompute still OOMs) and substantially two FIXABLE
> torch-2.3.1 bugs.** **Concrete path if you want compile to help: upgrade to torch 2.4+ and compile
> the `Block` submodules (keeping checkpoint)** — the composition bug is fixed post-2.3, so it should
> run, for an expected **modest ~1.15–1.3×** (the FlashAttention floor caps it — NOT the "much
> faster"/2× hoped for), at preserved memory, but it cascades the pinned monai/triton/VGGT deps.
> **UNTESTED prediction — needs a torch-2.4+ isolated-env run to confirm.** On 2.3.1 as-is: do not
> use compile. Repro: `tools/profile_trainstep_compile.py {eager,compile,proper,parts}`.

## Why / context

Follow-up to the training-speed debate (see the memory notes on gradient checkpointing being
mandatory at S≥10). The open question: is `torch.compile` (model-only or full-train-step, default
or max-autotune) a worthwhile speedup for the current pipeline? To answer it honestly we needed a
**measured** per-step breakdown and a real eager-vs-compile A/B, not a mechanism argument.

## Methodology

`tools/profile_trainstep_compile.py` — a standalone harness that reproduces the real trainer's one
train step **without editing repo code**: it Hydra-composes the same config, `instantiate()`s the
same model / loss / data / optimizer exactly as `training/trainer.py`, applies the same aggft
freeze (`freeze_modules(["*patch_embed*"])`), uses the same bf16 autocast scope (aggregator bf16,
heads+splat fp32 per `vggt.py:98`), and captures **real batches from the real `DataLoader`**. Only
difference vs a real run: **random init** (skips the 8 GB base-weights load — shapes are
checkpoint-independent). Config = the most-recently-run 1-frame variant
(`oneframe_baseline_gather05.sh`): `max_img_per_gpu=12 one_frame_per_slice=true gather_weight=0.5`.

Timing = CUDA events with `synchronize()` before every `elapsed_time` read, median over 20 iters
after warmup; per-phase events on the default stream (contamination-free — events mark GPU-side
stream order). Kernel breakdown via `torch.profiler`. Hardware: one A40 (interactive session).

**Verified by the `prove-it` skill** (3 independent reviewers + fixes). Two reportable-number bugs
were found and fixed before the numbers below: (F1) the component split timed `point_head` in bf16
instead of the real fp32 → corrected 84→137 ms; (F2) the numeric-match compared losses on
optimizer-drifted weights → fixed with a weight snapshot/restore (rel_err then dropped to 3.9e-6,
confirming compile is bit-faithful). Reviewers confirmed the **speedup ratios are unbiased** and,
if anything, the only ordering asymmetry makes compile look *faster* than it is — so "compile is
slower" is robust. Caveats: absolute step time is a few-% optimistic (omits grad-clip, the
per-iter respiratory GPU aug, and the DDP `find_unused_parameters` graph traversal); the phase
**proportions** and the compile **ratios** are the trustworthy outputs, not the absolute ms (which
also depend on the shared interactive GPU).

## Results

### Baseline eager (S=10, one A40, bf16+TF32+gradient-checkpointing)

| phase | ms | % |
|---|---|---|
| forward | 991 | 28.6% |
| loss (splat+L1+diffusion+gather+TV) | 14 | 0.4% |
| **backward** | **2372** | **68.4%** |
| optimizer (AdamW) | 93 | 2.7% |
| **total** | **3470** | **0.29 it/s** |

Component split of the forward: **aggregator 867 ms**, `point_head` 137 ms (fp32 DPT), splat 9.7 ms.
Per-subject S varies (one-frame): S=9 → 3129 ms, S=10 → 3518 ms, S=11 → 3922 ms/step.

**Dataloader is NOT a bottleneck:** first batch 7.9 s (lazy monai cache build), then 615 ms, then
~1 ms once the node-local `/tmp` cache is warm and the 4 workers prefetch — negligible vs the ~3.5 s
GPU step. Cross-validates the earlier memory probe (`measure_ckpt.py`: S=8 2.56 s, S=12 4.19 s WITH
checkpointing).

### Kernel breakdown (`torch.profiler`, operator / Self-CUDA view, no double-count)

| category | ~% self-CUDA | compile can help? |
|---|---|---|
| FlashAttention fwd+bwd | 18.8% | **No** — dispatches to the same flash kernel |
| GEMM (`addmm`+`mm`, projections/head) | 14.3% | **No** — already cuBLAS |
| LayerNorm fwd+bwd | ~9% | partially (fusable) |
| Elementwise (`mul`/`add`/`gelu`) | ~11% | yes (fusable) |
| Memory `copy_` + `cat` (reshapes between frame/global attn + `output_list` concat) | ~10% | partially |
| Convolution (DPT head + patch_embed) | ~6% | marginal |
| AdamW | ~2% | no |

So **~33% of GPU time is FlashAttention + cuBLAS GEMM that `torch.compile` cannot touch**, and the
~20% nominally-fusable elementwise/norm is scattered in small windows between attention / GEMM /
checkpoint / graph-break boundaries.

### `torch.compile` A/B (same batch replayed; valid weight-snapshot numeric match)

| variant | speedup vs eager | loss rel-err | notes |
|---|---|---|---|
| eager | 1.00× (ref) | — | baseline |
| model-only, default | **0.95×** | 3.9e-6 | slower |
| full-step, default | **0.96×** | 3.9e-6 | slower |
| model-only, max-autotune | **0.95×** | — | slower |
| full-step, max-autotune | **0.95×** | — | slower |
| model-only, **+DDP** default | **FAILS** | — | `BackendCompilerFailed` |
| full-step, **+DDP** default | **FAILS** | — | `BackendCompilerFailed` |

- **Every no-DDP mode is 4–6% SLOWER than eager.** max-autotune (which autotunes GEMM templates)
  does not help — it can't beat cuBLAS on these shapes and can't touch FlashAttention.
- **Numerically faithful:** on identical weights the compiled loss matches eager to **3.9e-6** (bf16
  reassociation only). Compile is correct; it's just slower.
- **DDP fails:** `RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same`
  in the DPT head conv (`point_head_scratch_refinenet4_res_conf_unit2_conv1`) — a torch-2.3.1
  Inductor/DDPOptimizer bug where autocast's bias-cast doesn't survive the graph split. Graph
  breaks jump **4 → 13** under DDP. Since the trainer DDP-wraps even on 1 GPU (`trainer.py:396`,
  unconditional), this is the deployment-relevant path.
  - *Qualification (per prove-it):* only the naive wiring was tested — compiling the **outer DDP
    wrapper** with autocast **inside** the compiled region. Compiling the inner module before the
    DDP wrap, or moving autocast outside, was not tried and might avoid this specific failure. But
    given the no-DDP result is already net-negative, that path was not pursued.

### Why compile is net-negative here (mechanism, backed by the measurement)

1. **~33% is FlashAttention + cuBLAS GEMM** — Inductor dispatches to the identical kernels; zero
   headroom on a third of the work.
2. **Gradient checkpointing is mandatory** at S≥10 (no-checkpoint OOMs above S=8 — memory probe).
   The backward re-runs the forward, and Inductor **cannot fuse across checkpoint boundaries**, so
   the 68% backward stays FlashAttention-bwd + recompute-bound.
3. The **~20% fusable elementwise is fragmented** across attention/GEMM/checkpoint/graph-break
   boundaries, so Inductor fuses only in small local windows; the savings are **smaller than
   compile's guard/dispatch/wrapper overhead** → net −5%.

### "Proper" compile attempt — remove manual checkpointing, let AOTAutograd manage recompute

The obvious objection to the above is that the manual `checkpoint()` calls are exactly what
strangles compile — so the *proper* setup is to remove them and let AOTAutograd's min-cut
rematerialization partitioner decide recompute (the standard way to compile a transformer). Tested
directly (monkeypatch `aggregator.checkpoint` → passthrough, then `torch.compile`), at S=10 on a
clean-memory process (compiled case run FIRST to rule out fragmentation):

| setup | step ms | peak mem |
|---|---|---|
| eager + manual checkpoint (baseline) | 3479 | **22.5 GB** |
| **compiled, NO manual ckpt (AOTAutograd min-cut)** | — | **OOM** |
| eager, NO manual ckpt (store-everything) | — | OOM |
| compiled + manual checkpoint kept | 3694 | **44.1 GB** |

**Result: the proper setup OOMs at S=10, on clean memory.** Mechanism: min-cut rematerialization
reduces the forward→backward *saved set*, but **not the peak forward activation memory** — without
per-block checkpointing, all 48 blocks' activations stay live through the forward, which exceeds
46 GB (independently: `measure_ckpt.py` shows eager no-ckpt is 43 GB at S=8 and OOMs by S=10).
Only manual per-block `checkpoint()` frees activations mid-forward. And keeping manual checkpointing
*and* compiling **doubles memory (22.5 → 44.1 GB, near the 46 GB ceiling) and is slower** — the
torch-2.3.1 compile+checkpoint interaction inflates memory rather than preserving the saving.

**So the barrier isn't a misconfiguration — it's a memory wall.** At S≥10, checkpointing is
mandatory, and compile cannot help a checkpointed model here (can't fuse across the boundaries, and
inflates memory). Compile could only help at **S≤8** (where a non-checkpointed model fits ~43 GB) —
and there its ceiling is just the fusion win on the ~20% fusable ops (FlashAttention's 33% stays
immune), i.e. ~1.1–1.2×, at the cost of abandoning the multi-frame budget.

### Partial compilation (compile the frozen DINO / the attention blocks separately)

The natural next idea: don't compile the whole model — compile the pieces that don't have the
checkpoint problem. Tested at S=10 (`tools/profile_trainstep_compile.py parts`):

| target | step ms | peak | outcome |
|---|---|---|---|
| eager baseline | 3463 | 22.4 GB | — |
| compile **DINO backbone only** (`aggregator.patch_embed`, frozen, forward-only) | 3464 | 22.5 GB | **runs, 0 gain** |
| compile **48 attention Blocks** (checkpoint kept around them) | — | — | **inductor ERROR** |
| compile DINO + blocks | — | — | inductor ERROR |

- **DINO compiles cleanly but buys nothing.** The frozen DINOv2 ViT-L is forward-only (no backward,
  no checkpoint) — the ideal target — but its forward is only **148 ms (~4% of the step)** and is
  already FlashAttention + cuBLAS internally, so there's nothing to fuse: 3463 → 3464 ms.
- **Compiling the attention Blocks — where ~90% of the compute is — FAILS on torch 2.3.1.** The
  `Block` is invoked inside `torch.utils.checkpoint`, and compiling it throws
  `Exception: Please convert all Tensors to FakeTensors first ... Found in aten.clone.default(
  tensor(size=(16,), dtype=torch.uint8))` — the non-reentrant checkpoint's saved **RNG-state tensor**
  isn't fakeified during Dynamo tracing. This is the **torch-2.3.1 `torch.compile` + activation-
  checkpoint composition bug**, not a fundamental limit.

**This reframes the whole result.** The only part that compiles is too small to matter; the part
that matters (checkpointed attention blocks) is blocked by a *torch-version bug*, not physics. The
memory ballooning in the whole-model case (44 GB) is the same immaturity: 2.3.1's compile mishandles
checkpoint. So the honest bottom line is: **on torch 2.3.1, compile cannot help — partly by memory
wall, but substantially by two fixable 2.3.1 bugs (checkpoint+compile, and DDP+autocast).**

### What a torch upgrade would (and wouldn't) buy

torch 2.4+ adds `torch._functorch.config.activation_memory_budget` (a knob to tune the min-cut
partitioner's memory target), better compile+checkpoint memory handling, and fixes the DDP+autocast
bias bug that made the DDP path fail. So a newer torch *might* let a budget-tuned compiled model fit
at S=10 and reach **parity or a small win**. But it cannot beat the two hard floors: FlashAttention
(~33%, dispatches to the same kernel on any torch) and the recompute-dominated backward. It will not
be "much faster," and the upgrade cascades the pinned monai/triton/VGGT dependency web (docs:
`requirements.txt` rationale) — real reproducibility risk mid-project for a parity-to-modest gain.

### Full-compile requirements (documented even though the ceiling is ≤1.0×)

A clean `fullgraph=True` compile would additionally require: (a) removing the data-dependent guard
`if (z_indices == 0.0).all()` at `aggregator.py:314` (the sole source of the 4 graph breaks — a
debug warning); (b) handling `splat.py:108` `if intensity.max() > 2.0` + the dynamic-index
`scatter_add` (breaks a forward+loss fullgraph); (c) the DDP + `find_unused_parameters=true`
interaction; (d) **variable S** per subject (one-frame ⇒ S∈{9..14}) forcing recompilation per
distinct S or dynamic-shape overhead — a real blocker; (e) the checkpoint+compile interaction on
the pinned torch 2.3.1. Even after all of that, the measured ceiling is net-negative, so it is not
worth doing.

## Conclusion / recommendation

- **Do not pursue `torch.compile`** for this pipeline. Measured 4–6% slower in every no-DDP mode
  and a hard failure under the DDP path the trainer actually uses, on torch 2.3.1.
- The pipeline is already near its practical optimum (bf16 + TF32 + FlashAttention + gradient
  checkpointing). The dominant cost is the aggregator attention backward + mandatory checkpoint
  recompute = FlashAttention (compile-immune) + memory-forced recompute.
- The dataloader is not the bottleneck; `cudnn.benchmark`/`persistent_workers` would be single-digit
  at best.
- The **only** large lever is **reducing S** (fewer input frames — a research/quality decision, not
  free), which also happens to be what would block compile anyway.

## Repro

```bash
PYTHONPATH=training:. python tools/profile_trainstep_compile.py eager              # phase+component+profiler
PYTHONPATH=training:. python tools/profile_trainstep_compile.py compile default    # eager vs compile (default)
PYTHONPATH=training:. python tools/profile_trainstep_compile.py compile maxautotune
PYTHONPATH=training:. python tools/profile_trainstep_compile.py compile default --ddp   # reproduces the DDP failure
```
