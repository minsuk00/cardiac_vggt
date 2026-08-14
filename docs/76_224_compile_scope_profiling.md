# 224-resolution `torch.compile` scope profiling

> **TL;DR & takeaway** — At `img_size=224`, disable activation checkpointing and compile the
> transformer blocks individually. The single existing toggle
> `cuda.compile_attention_blocks=true` now compiles **all 72 blocks**: 24 frozen DINOv2 blocks +
> 24 VGGT frame-attention blocks + 24 VGGT global-attention blocks. On an A40 this measured
> **961.8 → 716.1 ms/step** and **26.24 → 23.01 GiB allocated**, with five Dynamo graphs, zero
> graph breaks, and no per-subject-`S` recompilation stalls after ~58 s cold compilation. Do not
> compile the DPT head, whole aggregator, whole VGGT, or loss. Compilation is **not bitwise or
> mathematically identical to eager**: the 72-block output differed by up to `9.33e-5`, and the
> non-smooth splat can amplify tiny coordinate changes into materially different gradients.
> Compiled and eager runs are separate numerical series.

## 1. Why this investigation was run

The native-first port (`docs/73`) decoupled model input resolution from the native 256² splat,
making `img_size=224` a valid perception/computation knob. At 224, activations are small enough
that VGGT no longer needs activation checkpointing on an A40. This opened two questions:

1. Which `torch.compile` boundary actually improves a complete training step?
2. Can compilation preserve eager outputs and gradients closely enough to call the paths equal?

The tested boundaries were eager, DINO blocks, VGGT frame/global blocks, both block families,
DPT head, whole aggregator, whole VGGT, and VGGT+loss/splat. Forced-dynamic, automatic-dynamic,
default, and max-autotune behavior were also exercised.

Everything was prototyped under `/tmp/deep-check/`; no profiling scripts were added to the repo.

## 2. Current production semantics

There is deliberately **one compile toggle**, not separate DINO/VGGT toggles:

```yaml
cuda:
  compile_attention_blocks: true
```

`Trainer._compile_attention_blocks()` applies `nn.Module.compile(mode="default", dynamic=True)`
in place to:

```text
aggregator.patch_embed.blocks   24 frozen DINOv2 blocks
aggregator.frame_blocks         24 VGGT frame-attention blocks
aggregator.global_blocks        24 VGGT global-attention blocks
                               ──
                                72 total
```

`false` compiles none of them. `nn.Module.compile()` preserves module identity and state-dict
keys; checkpoints remain interchangeable. The DPT head, token/reshape orchestration, splat,
loss, metrics, and optimizer remain eager.

Activation checkpointing is a separate model option:

```yaml
model:
  gradient_checkpointing: true
```

For a future 224 launch, use `model.gradient_checkpointing=false`. Keep it enabled at high
resolution: 518 OOMed without checkpointing, while 294 was already near the A40 ceiling.

The two 224 jobs launched on 2026-08-13 (`57357517`, `57366221`) predate both changes: their logs
confirm the old **48 VGGT-block compile + checkpointing-on** path. A running Python process does
not pick up these edits. Do not describe those jobs as 72-block/checkpoint-off runs.

## 3. Measurement setup and caveats

Hardware and step:

- A40, 44.4 GiB usable
- Real 941M-parameter VGGT architecture, DPT head, BF16 autocast
- Aggft freeze pattern (`*patch_embed*` frozen; ~637M trainable)
- Fused AdamW
- Native-splat loss with diffusion, gather, and heart-L1 terms
- Worst-case `S=21`, plus `S∈{5,12,17,21}` for dynamic-shape checks
- Synthetic inputs and randomly initialized weights; this is a systems benchmark, not a quality
  experiment on a trained checkpoint

The final timing comparison used three warmup steps followed by ten synchronized timed steps in
clean, separate processes. Cold compilation used independent `/tmp` Inductor caches per scope.

The suite's first-pass per-scope timings used three steady steps and are marked preliminary below.
They are sufficient to reject large regressions or multi-minute shape stalls, but the ship numbers
come from the final ten-step runs.

## 4. Activation-checkpoint result

At 224 and `S=21`, model forward/backward/optimizer timing was:

| Checkpointing | Forward | Backward | Optimizer | Total | Peak allocated |
|---|---:|---:|---:|---:|---:|
| on | 376 ms | 777 ms | 38 ms | 1191 ms | 13.8 GiB in the original probe |
| off | 366 ms | 541 ms | 37 ms | 944 ms | 26.8 GiB |

The ~247 ms penalty is almost entirely backward recomputation. The earlier claim that forward was
the dominant cost was false; backward is larger. Checkpoint-off fits with substantial A40 margin.

Pre-cache, eager resolution ceiling without checkpointing (historical; superseded below):

| Input | Peak allocated / result |
|---|---:|
| 224 | 26.8 GiB |
| 280 | 36.6 GiB |
| 294 | 39.1 GiB |
| 322 | OOM |

After the docs/75 selective-intermediate cache and the shipped 72-block compile path, a complete
production-step rerun (`S=D=21`, initialized fused-AdamW state, real loss, resident 12-phase batch)
measured **350 as the hard A40 ceiling**: six steps completed at 42.02 GiB allocated / 43.86 GiB
reserved, while 364 OOMed on its second step. The practical maximum is **336** at 39.48 / 41.12
GiB, leaving ~3.3 GiB reserved margin. See docs/75 §7.1 for the full table and caveats.

## 5. Compile-scope results

### 5.1 Stabilized comparison

| Scope | Mean step | Median | Std. dev. | Peak alloc. | Peak reserved |
|---|---:|---:|---:|---:|---:|
| eager | 961.8 ms | 961.6 ms | 1.58 ms | 26.24 GiB | 27.46 GiB |
| 48 VGGT blocks, default | 729.6 ms | 722.6 ms | 13.30 ms | 23.01 GiB | 23.90 GiB |
| **24 DINO + 48 VGGT blocks, default** | **716.1 ms** | **715.7 ms** | **2.08 ms** | **23.01 GiB** | **23.90 GiB** |
| 48 VGGT blocks, max-autotune | 716.0 ms | 714.1 ms | 4.78 ms | unreliable¹ | 24.21 GiB |

¹ Max-autotune enabled CUDA-graph behavior that made `max_memory_allocated()` under-report
replayed allocations; reserved memory is the honest comparison.

The 72-block default path is ~25.5% faster than eager and saves ~3.2 GiB allocated. Adding DINO to
the already-compiled VGGT blocks saves another ~13.5 ms mean (~1–2%) at no additional steady VRAM.

### 5.2 Cold compilation and graph behavior

| Scope | Initial cold call | First `S=21` | Dynamo graphs | Breaks | Later `S` stalls |
|---|---:|---:|---:|---:|---|
| 48 VGGT blocks, default | 29.5 s | 21.3 s | 4 | 0 | none |
| **72 blocks, default** | **36.5 s** | **21.9 s** | **5** | **0** | **none** |
| 48 VGGT blocks, max-autotune | 196.8 s | 209.3 s | 4 | 0 | CUDA-graph warnings |

The 72-block path pays ~58 s total on a cold cache, then handles `S=5,12,17,21` without Dynamo
recompiles. DINO adds one reusable graph. The four VGGT-block graphs are reusable layout/code-path
variants, not one graph per block or one graph per slice count: all 48 instances share compatible
block code, and `S` remains symbolic. Exact graph-to-layout attribution was not traced, so do not
invent a more specific mapping. The measured facts are five unique graphs total, zero graph breaks,
and zero Dynamo recompiles.

Max-autotune gained at most ~1–2% over default while taking roughly eight times longer to compile.
It is not worth deploying.

### 5.3 Rejected scopes

| Scope | Result | Why rejected |
|---|---|---|
| DINO blocks only | 948.8 ms preliminary, 26.24 GiB, 6.8 s compile | Only ~1.3%; useful only as the cheap addition to VGGT blocks |
| whole DINO module | compile failure | DINO's training-time checkpoint rematerialization contains RNG ops unsupported by current PyTorch |
| DPT head only | 971.6 ms, 26.77 GiB | Slower than eager |
| whole aggregator, `dynamic=True` | failed after ~16 min | Inductor symbolic-backward `CantSplit` failure |
| whole aggregator, automatic dynamic | 718.4 ms preliminary, 23.92 GiB | 184–282 s compilation for shapes; larger numerical drift |
| whole VGGT | 726.5 ms preliminary, 23.90 GiB | 198–245 s per unseen `S`; no benefit over blocks |
| VGGT + loss/splat | 714.2 ms preliminary, 23.90 GiB | Only ~12 ms beyond model; 13 graphs and 11 break events |

The full-step breaks came from tensor-to-Python scalar extraction in the `z_scale` guard/path and
the external `fused_ssim` extension. A separate `torch._dynamo.explain()` audit found six breaks
across four source locations, including the no-grad metrics resume path. Event counts differ by
how often the harness executes/resumes a broken frame; the conclusion is unchanged.

Loss+splat alone measured ~15.2 ms forward+backward. Even a hypothetical perfect full compile can
save only ~1.6% of a ~944–962 ms eager step, so refactoring loss guards/metrics for compilation is
not justified.

## 6. Numerical equivalence: compiled is not eager

Do not call compiled and eager training mathematically identical.

Baseline complication: CUDA scatter-add in the native splat is nondeterministic. Repeated eager
runs had exactly equal model outputs/objective but one loss-gradient tensor differed by up to
`8.83e-6`. Bitwise end-to-end gradient equality is therefore impossible even for eager vs eager.

For the 48-block default path, a smooth model-only objective (no splat) gave:

- objective difference: `2.98e-8`
- maximum output difference: `2.21e-5`
- maximum gradient difference across every trainable parameter: `1.41e-6`
- zero parameters outside the chosen numerical tolerance (`rtol=2e-3`, `atol=2e-6`)

This shows the compiled backward is numerically close on a smooth objective. It is not exact.

Through the real splat, small coordinate changes can cross a voxel boundary and change which eight
neighbors receive mass. The piecewise/discrete indexing amplifies tiny model-output differences
into materially different gradients:

| Compile scope | Max model-output difference | Objective difference | End-to-end gradient result |
|---|---:|---:|---|
| 48 VGGT blocks | `2.21e-5` | `8.05e-6` | materially different after splat |
| 72 DINO+VGGT blocks | `9.33e-5` | `1.20e-5` | materially different after splat |
| whole VGGT | `1.19e-4` | `2.62e-5` | materially different after splat |

The 72-block path is therefore a fresh numerical series relative to both eager and the older
48-block compile path. This is expected BF16/Inductor reassociation plus splat sensitivity, not
evidence of gross compiler failure.

## 7. Decision and future-agent rules

For future fresh 224 runs:

```bash
... img_size=224 model.gradient_checkpointing=false \
    cuda.compile_attention_blocks=true
```

This selects checkpoint-off + all 72 blocks compiled with default mode.

Rules:

1. **Do not split DINO into another config toggle.** The measured incremental compile cost is ~7 s,
   it adds no graph breaks or steady VRAM, and it saves ~1–2%. The existing block toggle owns both
   transformer families.
2. **Do not compile the whole DINO module.** Compile its 24 internal blocks so the outer checkpoint
   wrappers remain eager.
3. **Do not change `dynamic=True` for per-block compilation.** It works at this small boundary and
   handles native subject-to-subject `S` variation. The forced-dynamic failure applies to the whole
   aggregator, not individual blocks.
4. **Do not compile the DPT head or loss.** Both were measured and provide no useful gain.
5. **Do not use max-autotune.** Default reaches essentially the same speed with far less cold time.
6. **Do not compare compiled and eager curves as one uninterrupted numerical series.** Checkpoints
   remain structurally compatible, but optimization trajectories are not identical.
7. **Do not retrofit interpretation onto running jobs.** A process uses the code loaded at launch.

## 8. `/tmp` provenance

The session used these throwaway probes (paths may disappear after node cleanup):

```text
/tmp/deep-check/compile_scope_suite.py
/tmp/deep-check/stabilized_compile_bench.py
/tmp/deep-check/compile_smooth_equivalence.py
/tmp/deep-check/verify_gradient_checkpointing_real.py
/tmp/deep-check/test_gradient_checkpointing_option.py
/tmp/deep-check/test_compile_all_attention_blocks.py
```

Representative result logs used for the tables:

```text
/tmp/deep-check/compile_{eager,head,model,full,aggregator_auto,blocks_default,
                         blocks_maxauto,dino_blocks_default,all_blocks_default}.log
/tmp/deep-check/stable_{eager,blocks_default,blocks_maxauto,all_blocks_default}.log
/tmp/deep-check/compile_smooth_equivalence.log
```

All production code changes are confined to the checkpointing option in `VGGT`/`Aggregator`, the
72-block loop in `Trainer._compile_attention_blocks()`, and their `default.yaml` documentation.
