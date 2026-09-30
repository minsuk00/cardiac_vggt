# Model interpretability: literature sweep + analysis plan

> **TL;DR & takeaway** — We have never looked at what the fine-tuned VGGT does internally.
> The original VGGT paper has **no interpretability content**: it offers only ablations (alternating
> attention beats global-only). VGGT-Ω (arXiv 2605.15195) and 2025–26 follow-ups agree on one
> picture of VGGT's global attention:
> - **Early** global layers (~0–8) are near-uniform and positional. They are almost no-ops: converting them to frame attention costs only ~0.15 AUC.
> - **Middle** layers (~9–16) do sparse cross-view alignment. Dropping them hurts most.
> - **Late** layers only refine.
> - **Special tokens** get dense attention.
>
> "Attention ≠ explanation", so the plan leads with **causal** tests and uses attention maps only
> to form hypotheses. Ranked by value per effort for our dense-regression, slot-based setting:
> 1. **Slot-level counterfactuals**: reference swap, drop-one-slice vs |Δz|, z-token shuffle, respiratory dose-response, slot-0 key knockout per layer.
> 2. **Global-layer drop / global→frame swap, plus "model souping"**: revert module subsets to base VGGT-1B and measure the val metric drop.
> 3. **Slot×slot global-attention affinity per layer**, regressed on |Δz|, phase gap, and "is slot 0". Check for high-norm/sink tokens first.
> 4. **Per-slot probes by layer** for the never-given input phase `t_i` and respiratory shift, with base-model and control baselines, plus CKA drift vs base.
> 5. **ROI-scalar gradient×input / Integrated Gradients per slot**, cross-checked against drop-one-slice.
>
> Skip for now: SAEs, full AttnLRP, and the confidence channel (it is untrained, see §3).
> Status: literature only; no experiments run yet.

## 1. What the VGGT papers contain

**VGGT (arXiv 2503.11651, CVPR'25).** None of its figures shows attention or feature
analysis. Analysis-type elements:
- **Alternating-attention ablation:** ETH3D Chamfer AA 0.709 < global-only 0.827 < cross-attn 1.061.
- **Loss ablation:** the camera loss helps point maps most.
- **Special tokens:** the first frame's special tokens are the reference mechanism, the same one our slot-0 reference reuses.
- **Single-image input:** global attention "transitions to frame-wise attention".

**VGGT-Ω (arXiv 2605.15195, CVPR'26 oral per README).** Architecture changes:
- DINOv3 backbone, patch size 16.
- 16 register "scene tokens" per frame.
- "Register attention" replaces 25% of global layers. Cross-frame traffic runs only through registers.
- Selective caching of layers `[4,11,17,23]`; we ported this in docs/75.

Analysis elements, all transferable:
- **Global attention is sparse:** they visualize the layer-13 attention matrix and its value distribution (Fig. 3). This motivates register routing.
- **Emergent motion clusters:** space-time-normalized tokens → PCA → k-means, no labels (Fig. 9). Motion segmentation is cleanest at **layer 4** and becomes semantic by layer 23. Layer numbers come from the fetched summary, not a verbatim quote.
- **Model souping as a probe:** average weight *subsets* between VGGT and VGGT-Ω with no retraining. Depth/FOV knowledge lives mainly in the **frame-block FFNs**, not in Q/K/V.
- **Register content:** registers carry scene semantics; a register-only head aligns to text at 76.8% top-1.

## 2. Follow-ups that analyze multi-view transformer internals

| Paper | Method | Finding relevant to us |
|---|---|---|
| Block-Sparse Global Attn (2509.07120) | Head-avg post-softmax global attention; per-layer mean/max split into special↔patch; layer-drop (front/back/middle) | Middle layers are sparsest and match-like. Pruning the center layers hurts; pruning early/late layers is tolerated. Special-token attention must stay dense. Works at ~75% sparsity. |
| AVGGT (2512.02541) | Top-k attention arrows; 180° rotation test; global→frame (G2F) and mean-token (G2M) layer swaps | Layers 0–8 are positional (G2F: 88.13→87.98 AUC). Layers 9–16 align the same token / same position across views. Layers 17–23 refine. |
| RobustVGGT (2512.04012) | Per-view-pair mean global-attention score + feature cosine, per layer, with distractor views | The model rejects outlier views on its own, and the rejection signal peaks in the last layer. |
| VGGT4D (2511.19971) | Gram QQᵀ/KKᵀ from global layers over a temporal window | Shallow layers are semantic, middle layers (4–8) carry motion, deep layers are spatial. Masking keys in layers 1–5 removes motion contamination. |
| Easi3R (2503.24391) | DUSt3R cross-attention mean/std maps | Dynamic and under-observed regions get low cross-attention. |
| Understanding Multi-View Transformers (2510.24907) | Per-patch MLP probes on the residual stream after every sublayer (DUSt3R) | Self-attention does most of the geometry work. 5/12 heads are "correspondence heads". Sink heads can be knocked out harmlessly. |
| ZeroCo (2412.09072) | Cross-attention as a zero-shot correspondence field | Attention is much sharper than feature cosine for matching. |
| RoMa-Ω (2609.09507) | NN matching vs probes on VGGT-Ω features | Raw feature cosine is poor, especially late; probes are strong. Read attention or use a probe, not raw cosine. |
| FastVGGT (2509.02560), Attn-collapse (2512.21691) | Token collapse, effective rank | These are long-sequence effects (hundreds of frames). Probably irrelevant at S≤21 (untested). |
| Probe3D (2404.08636), Lexicon3D (2409.03757), Feat2GS (2412.09606) | Frozen-feature probes for 3D awareness | They assume the same surface is seen from several views. That mostly fails across z-planes. |

## 3. General toolbox → our setting

**Scalar target.** Our output is a dense Δ, not a class, so every attribution needs a scalar. Options:
- ROI L1 of `V_canon` vs `V_gt` over the LV/myocardium;
- mean ‖Δ‖ or mean Δz of one slot's heart pixels;
- LV cavity volume, which ties to EF.

Reporting which target was used matters, because each answers a different question.

**Mechanics, with code refs:**
- **Attention weights are never materialized.** SDPA is the default (`vggt/layers/attention.py:61`). The explicit fallback at `:62-67` (`fused_attn=False`) is where to capture them; there, q and k already have QK-norm and RoPE applied.
- **Global attention is large.** It spans S·1374 tokens (1 camera + 4 register + 37² patches per slot; `patch_start_idx=5`). Reduce it on the fly to slot×slot mass plus chosen query rows; do not store the full matrix.
- **Hook template:** `tools/probe_aggft_collapse.py` (forward hooks on aggregator blocks).
- **All 24 layers:** set `aggregator.cached_layer_indices` to all 24 (see `tools/benchmark_aggregator_cache.py`).
- **Forward pass on one val subject:** the `evaluation/src/engine/run_vggt.py` docstring path (`load_model_from_run` → `get_data` → forward).
- **The z embedding is not a separate token.** It is added to the camera token. Slot 0 gets `camera_token[0]`; the others get `camera_token[1]`.
- **The confidence channel is unused.** The point head returns `world_points_conf`, but no loss references it, so it is untrained and not a usable uncertainty map. The training-free alternative is ensemble variance over per-slot t draws and respiratory seeds.

| Method | Canonical ref | Fit for us |
|---|---|---|
| Slot drop / swap / z-shuffle / reference swap | intervention analysis | **Highest.** S forward passes per subject. Zero slots already occur as padding, so zero-ablation is less out-of-distribution than usual. `tools/e0_dump_phase_sweep.py` already does the reference-phase sweep. |
| Activation patching (clean/corrupt differ only in slot-0 phase) | ROME 2202.05262; Zhang & Nanda 2309.16042; attribution patching 2310.10348 | High. Localizes where target-phase info enters the other slots' tokens. |
| Head mean-ablation (768 heads) | Michel 1905.10650; Gandelsman 2310.05916 | Medium. Screen with attribution patching first. |
| Raw attention / rollout / entropy | Abnar & Zuidema 2005.00928 | Hypothesis-forming only (Jain & Wallace 1902.10186). Rollout is cheap at slot level. |
| Sinks / high-norm tokens | Darcet 2309.16588; Sun 2402.17762 | Do this **before** attention maps. Background and padding zeros are where artifact tokens form. |
| Linear/MLP probes, CKA | Alain & Bengio 1610.01644; Hewitt & Liang 1909.03368; Kornblith 1905.00414 | High. Probe the input phase `t_i` (never given), respiratory shift, and target phase in non-reference slots; z is the positive control. Decodable ≠ used, so pair probes with interventions. |
| Grad×input / IG / SmoothGrad / Grad-CAM | 1703.01365; 1706.03825; Seg-Grad-CAM 2002.11434 | Medium. Aggregate per slot. Needs a sanity check (Adebayo 1810.03292) and deletion curves. |
| AttnLRP / Chefer | 2402.05602; 2012.09838; 2103.15679 | Low for now. It needs custom rules for the DPT head and the splat. |
| PCA / k-means of tokens | DINOv2 2304.07193; VGGT-Ω Fig. 9 | Cheap and visual. Fit one joint PCA across slots. |
| Logit/tuned lens (feed layer-ℓ tokens into DPT) | Belrose 2303.08112 | Medium. The raw lens is out-of-distribution for the head. |
| SAEs | Stevens 2502.06755 | Skip (GPU-days, low value for single-domain regression). |
| Jacobian det / folding of the Δ map | Learn2Reg 2112.04489; VoxelMorph 1809.05231 | Cheap physical diagnostic. |

## 4. Our-setting hypotheses to test (untested)

- **H1: early global layers do "same (x,y) across slices".** The in-plane grid and RoPE positions are shared across slots, so the attention peaks should sit at the identical (x,y). The interesting signal is the *deviation* from identical (x,y) near moving myocardium or under respiratory shift.
- **H2: target phase enters through global attention to slot 0 in the middle layers.** Test with slot-0 key knockout per layer and with activation patching.
- **H3: the model infers each slot's own phase `t_i` internally.** Test with probes. This bears on the information contract (docs/04, docs/65).
- **H4: the aggregator fine-tune changed mostly the global blocks' FFNs.** Test with weight-delta norms and souping against VGGT-1B. The prior from docs/26 is that cross-slice fusion is the mechanism.

Prior internal evidence:
- docs/09 and docs/26: aggft beats head-only because of cross-slice attention.
- docs/22 and docs/25: slot 0 carries the target phase.
- docs/64: dead DPT ReLU found via hooks.

None of these looked at attention.
