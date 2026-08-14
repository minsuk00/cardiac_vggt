# VGGT-Ω selective intermediate-output caching

> **TL;DR & takeaway** — Adopted VGGT-Ω's selective aggregator-output caching: the
> 24-layer aggregator now materializes concatenated frame/global outputs only for layers
> `[4, 11, 17, 23]`, the four layers consumed by DPT. The B-spline head remains covered because
> it consumes the final layer (`23`). This primarily reduces VRAM; it does not reduce attention or
> MLP FLOPs. On an A40, peak allocated memory fell by **2.10 GiB at S=10** and **4.40 GiB at
> S=21** for full-model forward+backward, while steady-state training runtime improved only
> **0.16–0.24%**. Full-model GPU predictions and focused CPU gradients were bit-identical to the
> previous all-output behavior. Follow-up full-step testing with checkpointing off + the 72-block
> compile path raises the A40 hard ceiling to **350** (six steps at 42.02 GiB allocated / 43.86
> GiB reserved); **364 OOMs** after optimizer state exists. Use **336** as the practical maximum
> (39.48 / 41.12 GiB) rather than operating within 0.6 GiB of the card limit.

## 1. Source and decision

VGGT-Ω's official aggregator exposes
`cached_layer_indices=(4, 11, 17, 23)`, converts it to a set, appends the concatenated
frame/inter-frame tensor only for those block indices, and appends `None` for all other indices:

- Official implementation: <https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/models/aggregator.py#L19-L29>
- Official caching branch: <https://github.com/facebookresearch/vggt-omega/blob/main/vggt_omega/models/aggregator.py#L118-L142>
- The original VGGT repository also describes its May 2026 fix as removing redundant retained
  intermediate tensors: <https://github.com/facebookresearch/vggt#readme>

We adopted that behavior rather than changing the aggregator architecture. Our alternating-attention
implementation is different from Ω's, so this is a behavior-level port, not a verbatim code copy.

## 2. Local consumers and retained layers

Before this change, `Aggregator.forward()` returned 24 concatenated tensors, each shaped
`[B, S, P, 2C]`. Only the following outputs are consumed in the active codebase:

- `DPTHead.intermediate_layer_idx = [4, 11, 17, 23]`.
- `BSplineWarpHead` reads `aggregated_tokens_list[-1]`, which is layer `23` for the production
  24-layer aggregator.

No other active Python consumer indexes the aggregator output list. Therefore the Ω default set
retains everything used by both shipped warp heads.

## 3. Implementation

The change is deliberately small:

- `vggt/models/aggregator.py`
  - adds `cached_layer_indices=(4, 11, 17, 23)`;
  - stores it as a set;
  - keeps the output list length equal to the transformer depth;
  - concatenates frame/global intermediates only for requested indices and inserts `None`
    elsewhere.
- `vggt/heads/dpt_head.py`
  - raises a clear `ValueError` if one of its required layers was not cached.
- `tests/test_reference_conditioning.py`
  - explicitly requests layer `1` for its two-layer test aggregators;
  - checks the `None`/tensor layout and exact equality of a retained layer.
- `tools/benchmark_aggregator_cache.py`
  - runs the matched full-architecture GPU equivalence, memory, and timing benchmark recorded below.

`cached_layer_indices` is ordinary Python state, not a parameter or persistent buffer, so checkpoint
state dictionaries are unchanged.

## 4. What changes and what does not

The raw aggregator return value is intentionally not identical:

- Before: all 24 entries were tensors.
- After: layers `[4, 11, 17, 23]` are tensors and the other 20 entries are `None`.

The values used by the model are unchanged. All frame/global attention blocks still execute in the
same order. The cache branch runs only after a block pair has produced its tokens; it controls whether
that layer's frame/global outputs are concatenated and retained. It does not modify the token stream
fed to subsequent blocks. Under the same runtime and determinism settings, the head therefore receives
the same tensors as before.

## 5. Expected resource effect

This is primarily a VRAM optimization:

- It removes 20 large concatenated output tensors from the returned list.
- At the production 518 input size, `P = 37×37 + 5 = 1374` tokens and `C = 1024`. Although
  the aggregator runs under bf16 autocast, the returned concatenated tensors were measured as
  **FP32** in the production architecture. The 20 skipped tensors therefore contain about
  `1.05 GiB` at `S=5`, `2.10 GiB` at `S=10`, and `4.40 GiB` at `S=21` (`B=1`), before
  allocator alignment. The measured peak deltas match this payload.
- Actual peak-memory reduction can differ because training still needs autograd state for the
  computation itself.

It does **not** skip attention blocks, patch embedding, MLPs, or DPT work, so it provides no material
model-FLOP reduction. It avoids 20 concatenations and their memory traffic. Section 7 measures only a
negligible steady-state runtime benefit; VRAM is the reason to keep this change.

## 6. CPU-only verification (2026-08-13)

No GPU was used. CUDA was hidden explicitly with `CUDA_VISIBLE_DEVICES=''`.

1. Focused repository suite:

   ```text
   tests/test_reference_conditioning.py: 12 passed
   ```

2. Independent grouped-block probe (`depth=4`, `aa_block_size=2`):

   - all-output cache `(0, 1, 2, 3)` versus selective cache `(1, 3)`;
   - identical weights and inputs;
   - retained outputs at layers `1` and `3` matched with `rtol=0, atol=0`;
   - selective layout was exactly `[None, tensor, None, tensor]`.

3. Backward-equivalence probe on the same models:

   - losses used only retained layers `1` and `3`;
   - every corresponding parameter gradient matched with `rtol=0, atol=0`.

4. `git diff --check` passed.

These checks established control-flow and autograd equivalence before the full-architecture GPU run.

## 7. Full-architecture A40 measurement (2026-08-13)

Reproduction:

```bash
PYTHONPATH=training:. python tools/benchmark_aggregator_cache.py --s 10 --repeats 6 --train
PYTHONPATH=training:. python tools/benchmark_aggregator_cache.py --s 21 --repeats 6 --train
```

Setup: one NVIDIA A40 (46,068 MiB), PyTorch `2.13.0+cu130`, full 941M production architecture,
518² inputs, batch size 1, DPT head, bf16 aggregator autocast, FP32 DPT path, aggregator gradient
checkpointing enabled, and frozen patch embed as in aggft. The same randomly initialized in-memory
model was toggled between all 24 cached layers and the selective four; weights never changed.
Random versus checkpoint weights does not change tensor shapes or the cache branch.

Peak memory and runtime were measured separately. Memory is `torch.cuda.max_memory_allocated()` after
clearing the caching allocator. Timing is the median of six steady-state repetitions with a warmed
allocator and balanced `AB, BA, AB, BA, AB, BA` ordering; ranges are retained by the tool. “Train F+B”
is full VGGT forward plus backward through a simple squared-world-point loss; it excludes splatting,
the production volume loss, optimizer state, and an optimizer step. Therefore the table directly
measures the model-side cache saving, but it does not prove that a complete production step's overall
peak falls by the same amount if a later splat/loss/optimizer phase becomes the new peak.

### Peak allocated memory (GiB)

| S | Scope | All 24 | Selective 4 | Saved |
|---:|---|---:|---:|---:|
| 10 | Aggregator inference | 6.54 | 4.43 | 2.11 |
| 10 | Full-model inference | 9.59 | 7.48 | 2.11 |
| 10 | Train F+B | 15.75 | 13.65 | **2.10 (13.4%)** |
| 21 | Aggregator inference | 9.84 | 5.42 | 4.41 |
| 21 | Full-model inference | 12.46 | 8.05 | 4.41 |
| 21 | Train F+B | 27.72 | 23.32 | **4.40 (15.9%)** |

### Median runtime (ms)

| S | Scope | All 24 | Selective 4 | Improvement |
|---:|---|---:|---:|---:|
| 10 | Aggregator inference | 843.1 | 835.7 | 0.87% |
| 10 | Full-model inference | 1021.5 | 1079.7 | inconclusive/noisy |
| 10 | Train F+B | 3249.5 | 3241.6 | **0.24%** |
| 21 | Aggregator inference | 2148.1 | 2131.7 | 0.76% |
| 21 | Full-model inference | 2477.9 | 2461.4 | 0.67% |
| 21 | Train F+B | 8623.6 | 8609.9 | **0.16%** |

The `S=10` full-inference samples had large timing outliers and a reversed median, so they do not
support a speed claim. The other steady-state deltas are all below 1%. Verdict: this is a substantial
and linearly S-scaled **VRAM** win, but not a meaningful speed or model-FLOP win.

### GPU numerical equivalence

At both `S=10` and `S=21`:

- retained aggregator layers `[4, 11, 17, 23]` passed `torch.equal`;
- final `dvfs`, `world_points`, and `world_points_conf` passed `torch.equal`;
- maximum absolute difference for every final output was exactly `0.0`;
- matched all/selective training losses were exactly equal.

Together with the CPU all-parameter gradient comparison and three independent code reviews, this
provides direct evidence that the optimization is numerically exact for the shipped architecture.

### 7.1 Complete production-step ceiling with checkpointing off

A follow-up `/tmp` probe closed the caveat above using the current production path: selective four-
layer cache, all 24 DINO + 48 VGGT blocks compiled individually (`dynamic=True`, default mode),
checkpointing off, fused AdamW including initialized optimizer state, the real native-splat loss,
and a worst-case native subject payload (`S=D=21`, all 12 phases resident). Every tested resolution
is a multiple of 14.

| `img_size` | Patch grid | Steady peak allocated | Steady peak reserved | Result |
|---:|---:|---:|---:|---|
| 322 | 23² | 37.03 GiB | 39.04 GiB | fits comfortably |
| **336** | 24² | **39.48 GiB** | **41.12 GiB** | **practical maximum (~3.3 GiB reserved margin)** |
| 350 | 25² | 42.02 GiB | 43.86 GiB | hard maximum; six steps stable, only ~0.56 GiB margin |
| 364 | 26² | 43.60 GiB at failure | ~44.41 GiB process use | OOM on step 2 after optimizer state existed |

Therefore **350 is the measured hard ceiling on this A40**, but **336 is the maximum that should be
used for a real job**. The 350 result leaves too little room for allocator variation, logging, or a
future small tensor addition. This supersedes the pre-cache eager ceiling of ~294 in docs/76 §4.
The throwaway probe was `/tmp/deep-check/resolution_ceiling_current.py`; no test code was added to
the repository.

## 8. Caveats for future changes

- If DPT's `intermediate_layer_idx` changes, update the aggregator cache set too; the explicit DPT
  error is intended to make a mismatch fail immediately.
- If a new head consumes another layer, add that layer to `cached_layer_indices`.
- A custom aggregator with depth below 24 must pass valid cache indices explicitly, as the tiny tests
  now do.
- The B-spline head assumes the final output is cached. The production default satisfies this.
