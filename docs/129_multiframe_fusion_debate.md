# 129 — Between one frame and all frames: a literature sweep and 5-agent debate on multi-frame write/read

> **TL;DR & takeaway** (2026-10-01)
> - **Question.** CardioStitch reads one frame per slice; real acquisitions offer F frames per slice (12 in our test arms, possibly hundreds in real data). The hypothesis: use only the *necessary* frames, so the output stays sharp but holes get filled.
> - **What was done.** 5 independent researchers each worked one angle: VGGT/many-view 3D, video/4D fusion, cardiac & fetal SVR/self-gating, learned token selection, and splatting/hole-fill as devil's advocate. They then debated and an impartial judge synthesised the result.
> - **Verdict on the framing: right.** Separate *read* (all frames may inform selection; the aggregator still sees S = D slots) from *write* (few frames splat). Decide per voxel: the near-phase frame owns every voxel it covers, and others only fill its holes. That is front-to-back priority compositing (3DGS-style transmittance), not coverage averaging.
> - **Verdict on the evidence: weak.** "Splatting all frames blurs" was never measured. The cited "frozen recon" was a regime artifact (1-frame model fed a multi-frame burst; `run_vggt.py` docstring). Holes look minor (hole_frac_heart 0.064–0.091; −29% holes from gather-aux bought 0 dB, docs/44, older ckpt).
> - **Biggest measured lever: selection, not fusion.** Selector 25.25 dB vs oracle single-frame 26.3 dB.
> - **Recommended order:**
>   - (0) a training-free probe on frozen diff1000: K-pass priority composite vs rank-1 + pull-push fill. Go only if it gains > 0.2 dB with no EF loss.
>   - (1) a set-aware selector: ISAB over a plane's frames, shuffled order, appearance-distance labels.
>   - (2) only then the adaptive write window plus a low-res read cascade for F in the hundreds.
> - **Open question for the user:** may within-plane acquisition order be used as a self-gating cue? It is not t or r; docs/04 is silent on it.

## 1. Setup

- **Problem brief.** Given to every agent; see the raw output for the full text. It covered:
  - the pipeline (DINO patch_embed → 24 alternating blocks → DPT Δ + unused confidence → forward splat `acc/(cov+eps)`)
  - the info contract (z only)
  - the selector/oracle numbers (handoff §2)
  - prior findings: holes, the "blur" claim, stop-grad denominator refuted, the appearance wall, the padding leak
  - the user's idea: per-voxel priority, confidence as splat weight, summary tokens
- **Workflow.**
  - 5 research agents with their own web search and repo reading. Read-only: no jobs, no edits.
  - Each then critiqued all five dossiers, revised its own and ranked them all.
  - A judge wrote the report.
  - 11 agents, ~15 min.
- **Raw output:** `temp/framesel/multiframe_fusion_debate_raw.json` (gitignored). It holds every dossier, critique and ranking.

**Claims I re-checked myself before recording:**
- `evaluation/src/engine/run_vggt.py` docstring: the "frozen recon" artifact = a 1-frame model fed a multi-frame reference burst. Constant companions are averaged away by the coverage mean. Confirmed.
- docs/44: hole_frac_heart 0.091 → 0.064 (−29%, p=1e-3) under the gather auxiliary, at no PSNR gain. Confirmed. These numbers come from the one-frame-ablation-era checkpoint, **not diff1000**. They have not been re-measured on diff1000.
- `tools/framesel_data.py` docstring: candidate frames are presented in "cyclic order from a random start". Confirmed. Any set-aware selector must shuffle, or it could read phase from order.

## 2. Taxonomy of approaches

| Family | Idea | Key paper |
|---|---|---|
| Priority / soft z-buffer compositing | The closest-phase frame wins; others fill transmittance | Softmax Splatting; 3DGS |
| Learned splat confidence | Unused conf channel → importance weight | DUSt3R conf loss; NeSVoR |
| Robust post-alignment merge | Weight secondaries by agreement with the anchor | Wronski handheld SR; EDVR TSA; Kuklisova-Murgasova EM |
| Second-source hole fill | Splat a second frame into holes | Niklaus 2023 splatting-based synthesis |
| Render-space hole fill | Pull-push on (acc, cov), or a masked refiner | Pull-Push (GRAPP 2009); SynSin |
| Set-aware self-gating selector | Score frames relative to their plane's own cycle | Set Transformer (ISAB/PMA); XD-GRASP |
| Compressed read | All F frames → R memory tokens per plane | Perceiver(-IO); CUT3R; STM |
| Query-only passenger slots | Extra frames read the primary KV but are not read back | StreamVGGT; PaceVGGT |
| Token merging inside VGGT | Merge redundant tokens in global attention | FastVGGT; HTTM; LiteVGGT |
| Time-anchored pointmaps | Map every frame to a common query time | Dynamic Point Maps; V-DPM |
| Same-phase multi-beat averaging | Gate by phase, motion-correct, average | Kellman 2009; Hansen 2012; van Amerom 2019 |

## 3. Recommended design — "read wide & cheap, select with context, write narrow by priority"

Converged in the debate. In 4 of 5 final rankings the K-pass priority composite came first as the *first experiment*; the dissenter (splat-fusion) ranked the set-aware selector first.

1. **Read cascade.** A cheap pre-scorer (pixel or 224-px) over all F frames picks the top-M (4–8) per plane. Only those get 518-px DINO tokens.
   - Reason: one 518-px DINO ViT-L pass ≈ 0.8–1 TFLOP, so F = 100 × D = 12 ≈ 1 PFLOP, larger than the aggregator itself.
   - This is unmeasured arithmetic, to be measured.
2. **Set-aware selector.** Today's reference cross-attention scorer plus 1–2 ISAB layers over the plane's pooled, plane-centred frame embeddings.
   - Example cue: the smallest-LV frame of a plane is ES.
   - Output: a calibrated posterior p_{z,f}. Content only, so the information contract holds.
3. **Adaptive write window.** Frames with p ≥ ρ·max, capped at K_max = 2–3. At F = 12 this is usually the argmax alone; at F in the hundreds it is same-phase frames from other beats.
4. **K independent passes.** Pass k puts the rank-k frame on every plane, with the same slot-0 reference, batched as B = K. Cached tokens enter via `patch_tokens`. These are in-distribution for the frozen model, with no joint (K·D)² attention.
5. **Front-to-back priority composite:**
   - v_k = acc_k/(cov_k+ε)
   - α_k = 1−exp(−cov_k/τ)
   - T_{k+1} = T_k(1−α_k)
   - V = Σ T_k α_k v_k + T_{K+1}·pullpush(acc_1, cov_1)

   Rank-1-covered voxels are unchanged from today. Pull-push also removes the low-coverage `acc/(cov+eps)` leak.
6. **Optional, last:** learned per-pixel α logit = conf + β·selector_logit − γ·(rank−1), initialised with γ large (starts exactly at the rule in step 5). Ship it only if EF and sharpness hold: a learned weight can hedge, raising PSNR while flattening EF.

**Physiology.**
- Near-phase frames have the right shape and short transports, so they own covered voxels.
- Far-phase frames carry wrong-phase appearance even when perfectly placed, so they must never overwrite covered voxels.
- Same-phase frames from different breaths sample different through-plane positions. This is the classical source of z-gap fill and SNR.
- Small static holes are better filled from neighbouring correct tissue (pull-push) than from wrong-phase myocardium.

**Training notes:**
- Selector only, on cached tokens.
- Shuffle frame order.
- Vary F from 2 to 48 and re-time frames.
- Soft labels from appearance distance to the GT target-phase slice (train-time only), not phase-index distance.
- No reconstruction gradient into the selector: through the coverage division it rewards dilated phases.

**Risks:**
- Pass k ≥ 2 conditions the whole stack on worse frames, so seams may appear at α boundaries (unmeasured).
- K× latency.
- Respiration-diverse writers smear along z where through-plane breathing correction is weak (docs/34).

## 4. Runner-ups and when they would win

- **Rank-1 + pull-push only.** Ship this if Stage 0 shows no write headroom (composite − max(fill, supersample) ≤ 0.2 dB).
- **Query-only passenger slots.** These replace the K−1 extra passes once the write side has proven value.
  - Primary slots stay bit-identical. Passenger global attention costs ~k/3.8 of today's.
  - Passenger DVFs are conditioned on the primaries, so they are more consistent than independent passes.
  - Needs a custom attention mask and finetuning.
- **Per-plane cine memory tokens** (Perceiver, zero-gated, in the special-token zone the DPT head drops).
  - Help EF and OOD amplitude, not PSNR: one writer means the single-frame oracle ceiling still holds.
  - Gate: build only if a linear probe shows pooled cine tokens predict within-plane phase rank and LV-area range.
- **Same-phase multi-beat writes.** Only for real-time F in the hundreds, and only after the sim adds per-frame noise, RT blur and multi-beat rhythms. The noise-free gated sim cannot reward averaging. Evaluate on MIITT RT.
- **Deprioritised:**
  - Anchor-composed satellites (≈ supersampling the anchor; 2D flow undefined under basal through-plane motion).
  - Temporal-median fill (diastole-biased: paints an ED ghost into ES holes).
  - Token merging with K frames inside the aggregator (most expensive; untrainable from gated cine).

## 5. Unresolved

1. **Order of work.** The K-pass composite probe vs the set-aware selector first. Judge: run the probe first (training-free, cheaper), but expect the selector to be the bigger lever.
2. **Acquisition order as a cue.** May within-plane acquisition order be used for self-gating? **The user must decide.**
3. **Strict priority vs posterior-weighted writes.** Strict priority is the default; posterior weighting is allowed only inside the same-phase tier at large F.
4. **Whether α should be learned at all** (hedging trap).
5. **Source of secondary DVFs:** independent passes vs passengers. Decide from Stage 0 seam measurements.

## 6. Experiment ladder (cheapest first)

| # | Experiment | Cost | Go / no-go |
|---|---|---|---|
| 0a | Heart-ROI SSE share in voxels with cov < 0.5, oracle & selector rank 1, ED vs ES | Script over existing recons | ≤ ~10% of SSE ⇒ hole strategies can buy ≤ ~0.4 dB; skip to 1 |
| 0b | Frozen diff1000, K = 1..3 passes, oracle & selector ranks. Arms: A rank-1; B Σacc/Σcov (**first real blur measurement**); C priority composite; D rank-1 + pull-push; E C + pull-push; F rank-1 2× supersampled | ~1 GPU-h | Go to the write side iff oracle-C − max(D, F) > 0.2 dB psnr_motion with EF & sharpness held. Never use hole_frac as a win signal (docs/44: it falls for free) |
| 0c | 0b with per-frame complex noise at RT SNR | Same | Gains only under noise ⇒ write side is a real-data feature → step 4 |
| 1 | Set-aware selector (ISAB, shuffled, re-timed, appearance labels, variable F) | Minutes | Phase err 2.0 → ≲ 1.3 and PSNR toward 26.3 dB, EF held |
| 1b | Read cascade: low-res pre-scorer → top-M at 518 | Selector-only | Top-M recall of the 518 argmax ≥ 95% at M = 4–8 |
| 2 | Composite C with the new posterior window (K_max = 3) | Inference | docs/38 rule + EF & sharpness held |
| 3a | Linear probe: cine tokens → within-plane phase rank, LV-area range | Minutes | No ⇒ drop memory tokens |
| 3b | Learned α logit, head-only, initialised at C | ~10 epochs | Beats C with EF held; ablate the conf term |
| 3c | Δ downsampled to 19×19, re-splatted | Inference | < 0.1 dB drop ⇒ passengers viable |
| 4 | Sim realism (noise, RT blur, multi-beat, F ≥ 48) + same-phase averaging; MIITT RT | Large | Must beat a single frame on psnr_motion & EF on MIITT |

## 7. References (cited with URLs by the researchers; not all independently re-read)

FastVGGT https://arxiv.org/abs/2509.02560 · HTTM https://arxiv.org/abs/2511.21317 · LiteVGGT https://arxiv.org/abs/2512.04939 · StreamVGGT https://arxiv.org/abs/2507.11539 · CUT3R https://arxiv.org/abs/2501.12387 · TTT3R https://arxiv.org/abs/2509.26645 · Point3R https://arxiv.org/abs/2507.02863 · VGGT-Long https://arxiv.org/abs/2507.16443 · Dynamic Point Maps https://arxiv.org/abs/2503.16318 · V-DPM https://arxiv.org/abs/2601.09499 · π³ https://arxiv.org/abs/2507.13347 · DUSt3R https://arxiv.org/abs/2312.14132 · MapAnything https://arxiv.org/abs/2509.13414 · MonST3R https://arxiv.org/abs/2410.03825 · Perceiver https://arxiv.org/abs/2103.03206 · Perceiver IO https://arxiv.org/abs/2107.14795 · Set Transformer http://proceedings.mlr.press/v97/lee19d/lee19d.pdf · ToMe https://arxiv.org/abs/2210.09461 · PaceVGGT https://arxiv.org/abs/2605.08371 · Evict3R https://arxiv.org/html/2509.17650v2 · STM (ICCV 2019) · XMem https://arxiv.org/abs/2207.07115 · Cutie (CVPR 2024) · Mixture-of-Depths https://arxiv.org/abs/2404.02258 · Reparameterizable subset sampling https://arxiv.org/abs/1901.10517 · Gumbel-top-k https://arxiv.org/pdf/1903.06059 · EDVR https://arxiv.org/abs/1905.02716 · Deep Burst SR https://arxiv.org/abs/2101.10997 · Handheld multi-frame SR https://arxiv.org/abs/1905.03277 · Softmax Splatting https://arxiv.org/abs/2003.05534 · Splatting-based Synthesis http://sniklaus.com/splatsyn · BasicVSR++ https://arxiv.org/abs/2104.13371 · RVRT https://arxiv.org/abs/2206.02146 · 4DGS https://arxiv.org/abs/2310.08528 · SynSin https://arxiv.org/abs/1912.08804 · 3DGS https://arxiv.org/abs/2308.04079 · Pull-Push (GRAPP 2009) · IBRNet https://arxiv.org/abs/2102.13090 · SVS https://arxiv.org/abs/2011.07233 · van Amerom 2019 https://pubmed.ncbi.nlm.nih.gov/31081250/ · Kuklisova-Murgasova 2012 https://pubmed.ncbi.nlm.nih.gov/22939612/ · NeSVoR https://ieeexplore.ieee.org/document/10015091/ · XD-GRASP https://onlinelibrary.wiley.com/doi/10.1002/mrm.25665 · Hansen 2012 https://onlinelibrary.wiley.com/doi/10.1002/mrm.23284 · Kellman 2009 https://onlinelibrary.wiley.com/doi/10.1002/mrm.22153 · Soft-gating comparison https://pmc.ncbi.nlm.nih.gov/articles/PMC11182662/ · EMORe https://arxiv.org/pdf/2507.23224 (abstract only) · SPARC https://arxiv.org/pdf/2608.18616 (not read).
