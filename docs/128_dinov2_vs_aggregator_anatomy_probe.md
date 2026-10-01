# Does DINOv2 understand heart anatomy, and does the trained model need it to? (patch-token anatomy probes)

> **TL;DR & takeaway** — **DINOv2 is good enough as the backbone. Anatomy is not the bottleneck;
> patch resolution may be.**
> 1. **Frozen DINOv2 is contrast-robust but only coarsely anatomical (E17).**
>    - Setup: 40 Siemens CMRx24 subjects, each paired with an LV-volume-matched GE/Philips/Canon
>      M&Ms subject. Each patch of A is labelled by copying the label of its nearest-neighbour patch
>      in B ("1-NN label transfer").
>    - DINOv2 transfers myocardium at Dice **0.60**, where raw pixels give **0.03**. On a
>      contrast-remapped copy of the same slice, DINOv2 gives 0.97 / 0.96 / 0.93 and raw pixels give
>      0.11 / 0.00 / 0.10 (LV / MYO / RV).
>    - It mostly encodes *blood pool vs muscle vs outside*, not *which ventricle*. RV Dice is
>      **0.49 ± 0.23**; it beats the "same grid position" baseline in 36/40 pairs, but LV
>      transfer is *worse* than that baseline (0.74 vs 0.91).
> 2. **The trained model adds chamber identity by global block 3 (E18).**
>    - Setup: final518 model, 25 cross-dataset test-split pairs, real val input pipeline.
>    - RV Dice goes from **0.33** (DINOv2 output) to **0.70** at block 3, better in **25/25 pairs**
>      (Wilcoxon p = 6e-8). LV goes 0.60 → 0.83 (25/25) and MYO 0.41 → 0.59 (24/25).
>    - Anatomy holds through block 11 and collapses from block 15. At blocks 19 and 23, nearest
>      neighbours concentrate on ~120–150 "hub" patches instead of ~480, and Dice falls to ≤ 0.18.
>    - Block 3 is the same layer where docs/126 found the reference-phase read.
> 3. **That anatomy reaches the output.**
>    - The DPT point head reads blocks **4, 11, 17 and 23** and puts 4 and 11 on its
>      highest-resolution paths. Block 23 enters only at 19×19.
>    - The late-layer anatomy loss is expected, not a defect. Hypothesis, untested: those layers are
>      re-encoded for the displacement Δ.
>    - **Truncating to 11 blocks is not recommended.**
> 4. **Resolution, which this doc does not test, is the open lever.**
>    - One token covers about 9.7 mm, roughly the myocardial thickness.
>    - The DPT head never sees raw pixels.
>    - docs/86 shows 224 → 336 → 518 improving PSNR monotonically without flattening.
>    - The proposed next step, untested, is a zero-initialised **raw-pixel skip into the DPT head's
>      final 518² stage**, warm-started from the final checkpoint, with a matched no-skip control
>      (§5).
>
> Scripts: `tools/interp/e17_*.py` and `e18_*.py`. Outputs (gitignored, regenerable):
> `temp/dinov2_probe/` and `temp/aggft_probe/`. Date: 2026-09-30.

---

## 1. Question and why it matters

The model is VGGT, and every slice first goes through VGGT's **DINOv2 ViT-L/14 `patch_embed`**. It is
frozen under aggft (`optim.frozen_module_names = ["*patch_embed*"]`), so DINOv2's output is the only
view of the image the rest of the network gets. Two questions:

- Does frozen DINOv2 encode **cardiac anatomy** (LV / myocardium / RV) *independently of contrast*
  (vendor, field strength)?
- If not, does the **finetuned aggregator** make up for it, or is the backbone a bottleneck?

Related: docs/77 + docs/86 (a DINOv3 backbone swap did not help), docs/125 (interpretability
literature), docs/126 (causal interpretability of the same final model), docs/72 (resolution
experiments).

## 2. Background facts used throughout (verified from code)

- **Token geometry.** At a 518² input with patch 14, each slice has a **37×37 = 1369** patch grid.
  `x_norm_patchtokens` has shape `(N, 1369, 1024)`; CLS + 4 register tokens are excluded.
  - In the aggregator, each slot's token sequence is 5 special tokens (camera + 4 registers;
    `harness.N_SPECIAL = 5`) followed by the 1369 patch tokens.
  - Global-block outputs are reshaped `(S, 5+1369, 1024)` and the first 5 are dropped.
- **Patch footprint.** The canonical grid is 256² at 1.4 mm (358.4 mm FOV), bilinearly resized to
  518². One patch is **358.4 / 37 ≈ 9.7 mm**, about 7 canonical pixels.
- **Image normalisation into DINOv2.** `(img − resnet_mean) / resnet_std` with ImageNet constants
  `[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`, inside `Aggregator.forward`
  (`vggt/models/aggregator.py`). E17 applies the same constants by hand.
- **Labels.** `heart_seg*.nii.gz` uses 1 = LV blood pool, 2 = MYO, 3 = RV
  (`inference/seg_metrics_cmrxrecon.py:36`).
  - `heart_seg_canonical.nii.gz` is `(256, 256, D, 12)` at 1.4 mm and native dz, i.e. the same grid
    as the model input.
  - It is transposed `(X,Y,Z) → (Z,Y,X)` like `heart_roi_canonical` in
    `MRIDataset.get_data` (`training/data/datasets/mri_dataset.py:554`).
  - **Not verified:** whether the CMRx labels are manual or automatic (nnU-Net?). E17/E18 treat them
    as truth.

## 3. Method: 1-NN patch label transfer (shared by E17 and E18)

For a pair (A, B) at a given layer:

1. **Patch labels.** Resize the label slice to 518² (nearest neighbour), then give each 14×14 patch
   its **majority label** → 1369 labels per image (`avg_pool2d(one_hot, 14).argmax`).
2. **Features.** Take one feature vector per patch (`fa`, `fb`: `(1369, C)`). Subtract the
   **joint mean** token over A ∪ B, then L2-normalise.
   - The mean subtraction is needed: without it every cosine similarity is ≈ 0.8 (a shared DC
     component) and similarity maps are flat.
3. **Transfer.** For each A patch, take the most cosine-similar B patch (1-NN, no training) and copy
   that B patch's label: `pred_A = lb[(fa @ fb.T).argmax(1)]`.
4. **Score.** Per class c ∈ {LV, MYO, RV}:
   `Dice = 2|pred=c ∧ true=c| / (|pred=c| + |true=c|)`, counted in **patches** on the 37×37 grid
   (not pixels). Report the mean ± sd over pairs.
   - Use **Dice, not recall.** An early single-pair version reported per-class recall, which
     ignores false positives: every blood pool labelled "LV" still scored LV recall 0.96.
5. **Baselines.**
   - **Raw pixels:** the same 1-NN on the raw 14×14 pixel patches (196-d). Tests whether intensity
     alone suffices.
   - **Same location:** A's patch i takes B's label at patch i, i.e. pure geometry with no image
     content. **Interpret DINOv2 or aggregator numbers relative to this baseline.** In E17 the crops
     are LV-centred, so this baseline is strong for LV (0.91).

## 4. E17 — frozen DINOv2, native LV-centred crops (`tools/interp/e17_dinov2_frozen_crossvendor.py`)

### 4.1 Setup

- **Model.** DINOv2 = `aggregator.patch_embed.*` weights from `scratch/base_weights/vggt1b_base.pt`
  (strict load), staged to `/tmp`. Since `patch_embed` is frozen under aggft, these are the same
  weights the trained model uses.
- **Subjects.** The first 40 `CMRx24_Train_*` subjects with `sax/heart_seg.nii.gz` (Siemens, 3 T).
  Each is paired, greedily and without replacement, with the **non-Siemens** M&Ms subject
  (GE / Philips / Canon, vendor from `MNMs1/*information*.csv`) of the closest ED LV volume.
  Median |ΔLV| is **0.2 mL**.
  - These CMRx subjects may be in the model's training set. That doesn't matter here: frozen
    DINOv2 never trained on them.
- **Slice.** ED (`3d_recon/sax_frame_00`, `heart_seg[..., 0]`), at the middle of the LV's z-extent.
  A **160 mm square centred on the LV centroid** is resampled to 518² (`grid_sample`), and
  intensities are normalised to the 0.5 / 99.5 percentiles.
  - This is **not** the training pipeline; E18 uses the real one.
- **Conditions**, all transferring onto A:
  - **cross:** B → A.
  - **ctrl:** A′ → A, where A′ = A with a non-monotonic contrast remap
    `(1 − (2a−1)²)^0.7·0.9 + 0.1(1−a)`, so the anatomy is identical and the contrast differs.
  - **ctrl_rot:** A′ rotated 90° → A. The anatomy is identical but positions change, which
    removes any positional shortcut.
- **Runtime.** About 5 min on CPU.

### 4.2 Results

40 pairs; Dice mean ± sd:

| condition | method | LV | MYO | RV |
|---|---|---|---|---|
| cross (B→A) | DINOv2 | 0.74 ± 0.12 | 0.60 ± 0.18 | 0.49 ± 0.23 |
| | raw pixels | 0.25 ± 0.08 | 0.03 ± 0.05 | 0.29 ± 0.08 |
| | same location | **0.91 ± 0.04** | 0.62 ± 0.17 | 0.18 ± 0.21 |
| ctrl (A′→A) | DINOv2 | 0.97 ± 0.03 | 0.96 ± 0.05 | 0.93 ± 0.09 |
| | raw pixels | 0.11 ± 0.07 | 0.00 ± 0.00 | 0.10 ± 0.06 |
| | same location | 1.00 | 1.00 | 1.00 |
| ctrl_rot (A′rot→A) | DINOv2 | 0.82 ± 0.09 | 0.78 ± 0.08 | 0.51 ± 0.17 |
| | raw pixels | 0.11 ± 0.07 | 0.00 ± 0.01 | 0.11 ± 0.07 |
| | same location | 0.94 ± 0.04 | 0.78 ± 0.12 | 0.05 ± 0.03 |

**Paired statistics, cross condition:**
- **RV:** DINOv2 beats same location in **36/40** pairs (Wilcoxon p = 8e-10), but 7/40 pairs have
  RV Dice < 0.2.
- **LV:** DINOv2 is **below** same location in 35/40 pairs.

### 4.3 Interpretation

1. **Contrast invariance is robust.** Raw pixels collapse under any contrast change (MYO ≈ 0), while
   DINOv2 holds.
2. **The features are coarse and tissue-level.** DINOv2 separates blood pool, myocardial ring and
   extracardiac tissue across vendors, but it often confuses the LV and RV blood pools. Some
   out-of-heart blood pools (e.g. great vessels) also get matched as "LV", which is why LV Dice is
   below the geometry prior.
3. **The near-perfect ctrl score is partly positional.** Rotation drops RV from 0.93 to 0.51 with
   identical anatomy.
   - DINOv2 tokens carry position information: raw channels 272 and 35 show the same spatial
     gradient in every image regardless of content (qualitative, below).
   - DINOv2 is also not rotation-invariant. This probe cannot separate the two effects.

**Qualitative views (scratchpad only, not tracked; reproduce from E17 tokens if needed).**
- **Raw channels.** The 8 highest-variance channels split into positional (272, 35),
  structural/edge (78, 527, 600, which trace the myocardial ring and blood-pool edges in all
  vendors), a probable blood-pool detector (888) and noisy channels (1017, 554, 766).
- **Similarity maps.** Cosine similarity to a query patch (LV / MYO / RV / background):
  - LV and RV queries light up **every blood pool**, LV, RV and vessels alike.
  - The MYO query lights up the ring strongly only within the same image; cross-subject it is faint
    (~0.1–0.2), yet still the argmax, which is why 1-NN works.
  - Some RV query patches give scattered speckle across the whole image. Hypothesis, unverified:
    these are high-norm "artifact" tokens.
- **Shared-basis PCA.** One PCA fit on 120 images (40 × A, B, A′rot), with one global 1–99 %
  colour range. PC1–3 explain 27 %.
  - Blood pools are the same colour in every vendor, and the LV and RV are indistinguishable.
  - The background colour follows the grid position, even in the rotated A′.

**Lesson for future probes:** the very first single-pair test (P105 ↔ MNMs_R8V0Y4) suggested "DINOv2
cannot tell LV from RV" (RV ≈ 0.06). That pair is one of the 7/40 failures, and the claim was
wrong at n = 40. **Do not conclude from one pair.**

## 5. E18 — the trained model's tokens, layer by layer (`tools/interp/e18_aggregator_anatomy_probe.py`)

### 5.1 Setup

- **Model.** `scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt` (the
  `harness.CKPT` default), loaded with `load_model_from_run` (no attention hooks) and run in fp32.
  **Only `model.aggregator` is run**; the DPT head isn't needed. Captures are forward hooks:
  - the patch_embed output → `dino`;
  - `global_blocks[i]` outputs for i ∈ {0, 3, 7, 11, 15, 19, 23} → `g{i}`.
- **Inputs.** Exactly the val path: `harness.Subject` + `harness.build_batch(subj, zeros(S))`, i.e.
  canonical grid → 518², **all slots at ED**, no breathing, slot 0 = reference at the mid plane, one
  frame per slice (S = D).
- **Subjects.** The pooled_curated_v2 **test split** (180 subjects: MNMs 49, CMRx25 46, CMRx24 38,
  CMRx23 26, ACDC 21).
  - Pairs are **cross-source** (different dataset) with the closest ED LV volume, greedy, shuffled
    with `random_state=0`; median |ΔLV| is 0.6 mL.
  - Cross-source is not always cross-vendor (e.g. CMRx23 ↔ CMRx24 are both Siemens).
- **Probed slots.**
  - **ref** = slot 0, the reference at the mid plane; 25 pairs.
  - **comp** = the companion slot at plane ref+1; 24 pairs. One pair was dropped because a class
    was missing on that plane; pairs are skipped unless all three classes appear on both slices.
- **Runtime.** About 3 min on one shared GPU (an aggregator forward is 3–4 s per subject).
  - A full `model(...)` forward on CPU was about 70–120 s per subject: don't do that.

### 5.2 Results

**Reference slot** (25 pairs). Dice mean ± sd, plus the collapse diagnostics:
- **`uniq`:** how many distinct B patches serve as anybody's 1-NN.
- **`offset`:** the mean grid distance, in patches, from an A heart patch to its 1-NN.

| layer | LV | MYO | RV | uniq (of 1369) | offset |
|---|---|---|---|---|---|
| same location | 0.40 ± 0.29 | 0.16 ± 0.14 | 0.04 ± 0.08 | – | – |
| DINOv2 (frozen) | 0.60 ± 0.18 | 0.41 ± 0.18 | 0.33 ± 0.23 | 480 | 4.6 |
| global 0 | 0.61 ± 0.16 | 0.42 ± 0.19 | 0.34 ± 0.23 | 489 | 4.6 |
| **global 3** | **0.83 ± 0.08** | **0.59 ± 0.18** | **0.70 ± 0.16** | 495 | 4.9 |
| global 7 | 0.81 ± 0.12 | 0.59 ± 0.20 | 0.70 ± 0.17 | 478 | 4.9 |
| **global 11** | **0.84 ± 0.09** | **0.63 ± 0.19** | **0.70 ± 0.16** | 457 | 4.9 |
| global 15 | 0.62 ± 0.25 | 0.45 ± 0.28 | 0.48 ± 0.22 | 409 | 4.9 |
| global 19 | 0.18 ± 0.22 | 0.04 ± 0.09 | 0.14 ± 0.15 | 156 | 8.8 |
| global 23 | 0.14 ± 0.18 | 0.03 ± 0.06 | 0.08 ± 0.13 | 123 | 9.2 |

**Companion slot** (24 pairs): the same shape.

| layer | LV | MYO | RV |
|---|---|---|---|
| DINOv2 | 0.63 | 0.42 | 0.37 |
| global 3 | 0.84 | 0.61 | 0.72 |
| global 11 | 0.79 | 0.59 | 0.62 |
| global 19 | 0.31 | 0.23 | 0.17 |
| global 23 | 0.40 | 0.26 | 0.12 |

**Paired Wilcoxon, block vs DINOv2:**

| slot | class | g3 | g11 |
|---|---|---|---|
| ref | LV | 25/25, p = 6e-8 | 25/25, p = 6e-8 |
| ref | MYO | 24/25, p = 2e-7 | 25/25, p = 6e-8 |
| ref | RV | 25/25, p = 6e-8 | 25/25, p = 6e-8 |
| comp | LV | 24/24, p = 1e-7 | 22/24, p = 6e-5 |
| comp | MYO | 20/24, p = 8e-5 | 18/24, p = 5e-4 |
| comp | RV | 24/24, p = 1e-7 | 22/24, p = 4e-6 |

### 5.3 Interpretation

1. **Global block 0 ≈ DINOv2.** The first frame+global pair barely changes the anatomy content.
2. **Blocks 0 → 3 add chamber identity.** RV doubles and LV/MYO rise, consistently across pairs.
   - This is the same depth where docs/126 found the companions reading the reference's heart
     tokens (global layer 3, heads 2/10/12/14).
   - The probe is correlational. It does **not** show whether the gain comes from within-slice
     (frame) or cross-slice (global) attention; a frame-only vs global knockout would.
3. **Blocks 3–11 keep the anatomy. From 15 on it is overwritten.**
   - Late layers show hubness (the 1-NN collapses onto ~120–150 B patches) and match distances
     double.
   - In the pair figures, at block 19 nearly every B heart patch has similarity ≈ 0.6 to every
     query: the features are nearly uniform.
   - **Hypothesis, untested:** late layers encode what the point head needs, the per-pixel
     displacement Δ, under which an LV patch and an RV patch moving alike look alike.
   - A test would correlate late-layer feature similarity with the similarity of the predicted Δ.
   - This also fits docs/126: breathing alignment needs layers 12–17, and 18–23 are not needed.
4. **The absolute numbers differ from E17,** because E18 uses the full canonical FOV, not
   LV-centred crops. Same-location LV is 0.40 here vs 0.91 in E17. Compare layers within one
   experiment, not across E17 and E18.

### 5.4 Figures (`tools/interp/e18_viz.py`)

- **`temp/aggft_probe/dice_vs_layer.png`:** Dice vs depth; thin lines = individual pairs, thick =
  mean, dashed = same-location baseline.
- **`temp/aggft_probe/pairNN_layers.png`:** saved pairs 03, 13 and 15, i.e. the smallest, median
  and largest g3 − DINOv2 RV gain.
  - **Rows:** DINOv2, then blocks 3, 11, 19 and 23.
  - **Column 1:** labels copied B → A by 1-NN, with Dice.
  - **Columns 2–4:** cosine similarity on B to A's LV / MYO / RV query patch (the centroid-nearest
    patch), with GT outlines, zoomed to the heart.
- **What they show:** label transfer at blocks 3 and 11 is clean, while DINOv2 scatters RV labels
  onto background. The similarity maps carry the anatomy only subtly; even at blocks 3 and 11 the
  LV and RV queries both light B's LV most. **Use the 1-NN label maps and Dice, not raw similarity
  maps, as the evidence.**

## 6. How the anatomy reaches the output: the DPT head (verified from `vggt/heads/dpt_head.py`, `vggt/models/vggt.py`)

1. **Sources.** `point_head = DPTHead(dim_in=2*embed_dim, output_dim=4, activation="linear",
   conf_activation="expp1")`, with `intermediate_layer_idx` defaulting to **[4, 11, 17, 23]**.
   Each source is the same 37×37 token grid: special tokens are dropped, then LayerNorm.
2. **Per-source projection and resize.** A 1×1 conv projects channels to [256, 512, 1024, 1024].
   A small sinusoidal UV position embedding is added (ratio 0.1). Then a learned resize:

   | source | resize | grid |
   |---|---|---|
   | block 4 | `ConvTranspose2d` ×4 | 148 |
   | block 11 | `ConvTranspose2d` ×2 | 74 |
   | block 17 | identity | 37 |
   | block 23 | stride-2 conv | 19 |

   These grids carry **no new spatial information**; they are learned resamplings of 37×37.
3. **Fusion, coarse → fine** (`FeatureFusionBlock`). Each step: `out = RCU2(out + RCU1(skip))`, then
   bilinear upsample, then a 1×1 conv, where RCU = `x + conv3(relu(conv3(relu(x))))`.
   Chain: 19 → 37 (+ block 17) → 74 (+ block 11) → 148 (+ block 4) → 296.
4. **Output.**
   - `output_conv1`: 3×3 conv, 256 → 128 channels.
   - Bilinear upsample to 518², plus the position embedding.
   - `output_conv2`: 3×3 conv 128 → 32, ReLU, then a **1×1 conv 32 → 4**.
   - Channels 0–2 = Δ (linear, unbounded); channel 3 = confidence `1 + exp(·)` (unused).
5. **Residual DVF.** `world_points = scanner_coords + Δ` (`vggt.py:63-66`), which is then splatted.
   Δ is a **per-input-pixel** displacement in normalised units: 1.0 in x/y ≈ 178.5 mm
   (0.5·255·1.4), 1.0 in z = 90 mm (`Z_HALF_MM`).
6. **No raw pixels.** `images` is used only for its shape. All sub-patch detail must be decoded from
   the tokens.

**Consequence for this doc.** The anatomy-rich blocks (4, 11) feed the high-resolution decoder paths
(148, 74), and the anatomy-poor block 23 enters only at 19×19. So the late-layer anatomy loss is
not a problem, and **truncating the transformer at block 11 is not recommended.** That would remove
inputs the head uses; the only real test would be a training ablation, which has not been run.

## 7. Resolution: the remaining lever (analysis, not an experiment here)

### 7.1 Existing evidence

docs/86 trained the same model at three input sizes (pooled-1337 ep300, val breath arm, 144 shared
subjects). Patch mm is computed as 358.4 / grid.

| input | grid | ≈ mm / patch | PSNR | EF MAE % | LV Dice ES | pooled EF slope |
|---|---|---|---|---|---|---|
| 224 | 16² | 22.4 | 19.41 | 12.1 | 0.804 | 0.785 |
| 336 | 24² | 14.9 | 19.52 | 11.7 | 0.798 | 0.795 |
| 518 | 37² | 9.7 | **19.64** | **9.6** | **0.820** | 0.744 |

- **Paired PSNR:**

  | comparison | ΔPSNR | subjects better | p |
  |---|---|---|---|
  | 224 → 336 | +0.11 dB | 69 % | 3e-6 |
  | 336 → 518 | +0.15 dB | 74 % | 1e-7 |
  | 224 → 518 | +0.26 dB | 83 % | 2e-12 |

- **Not flattening:** the step to 518 is not smaller than the step to 336, so the model plausibly
  still loses sub-patch detail at 518. This is inferred from three points.
- **Trade-off:** 518 has the weakest EF slope; docs/86 reads it as regression to the mean.

### 7.2 Why not a larger input

- **Cost, estimated and not measured.**
  - Tokens per slice grow with resolution²: 742² gives 2809 tokens, about 2× those at 518.
  - Global attention runs over S × P tokens, so it grows about 4×; MLPs grow about 2×.
  - Overall about 2–4× time and activation memory. aggft already uses about 27 GB per A40
    (CLAUDE.md).
- **No new information.** The data is 256² at 1.4 mm. **518 is already a 2× upsample**; going larger
  only adds tokens over interpolated pixels.
- **Pretraining.** 518 is VGGT/DINOv2's pretraining resolution, so pos-embed interpolation is least
  OOD there.

### 7.3 Proposed next experiment (untested): raw-pixel skip into the DPT head

Add the missing high-frequency path without touching the transformer. Insert just before
`output_conv2`, at 518², where `out` has 128 channels:

```python
self.pix = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(64, 128, 3, padding=1))   # ZERO-INIT this last conv
...
out = out + self.pix(images.reshape(B * S, 3, H, W))         # then output_conv2 → 4 ch
```

- **Size.** About 93k parameters (3·32·9 + 32·64·9 + 64·128·9), against about 941M total and about
  33M for the head. Compute is small next to the 24 attention blocks; this is an estimate, not
  profiled.
- **Add, don't concatenate,** and zero-init the last conv. At step 0 the model is then *exactly* the
  current final model, so warm-starting is safe.
- **Warm-start, don't retrain from scratch.** Load the final checkpoint **weights only** with
  `strict=false`; the new `pix.*` keys are missing.
  - Gotcha (docs/37): `CKPT_ONLY` is a FULL resume (optimizer + `prev_epoch`) and can silently do
    zero training. Strip first: `torch.save({'model': torch.load(ck)['model']}, out)`.
- **Matched control (mandatory).** The same checkpoint finetuned for the same number of epochs
  without the skip. Otherwise any gain could be extra training.
- **Score** with the docs/38 decision rule (recov_frac ↑ and psnr_motion ↑ without hole_frac ↑), plus
  EPE and EF MAE as in docs/86.
- **Optional second skip** into the 148/296 fusion stages. Start with the single 518 skip.

## 8. Caveats (what this doc does NOT show)

- **1-NN label transfer** is a lower bound on what's linearly decodable. A trained linear probe
  could score higher at every layer, including DINOv2. The *relative* layer ordering is the
  finding, not the absolute Dice.
- **Scope.** ED only; mid and mid+1 planes only; one checkpoint (final518_diff1000); no breathing
  displacement in E18. Other phases, apical/basal slices and breathing-perturbed inputs are
  untested.
- **Labels.** E17 uses native-grid `heart_seg`; E18 uses `heart_seg_canonical`. Their provenance
  (manual vs automatic) for CMRx is unverified.
- **E17's contrast remap is synthetic,** one fixed non-monotonic curve. Real vendor differences are
  covered by the cross condition.
- **The late-layer hypothesis is not tested,** in either direction: that late layers encode Δ, and
  that the speckle patches are high-norm tokens.

## 9. Reproduce

Run from the repo root.

```bash
# E17 — frozen DINOv2 (CPU, ~5 min) + figures
CUDA_VISIBLE_DEVICES= micromamba run -n svr python tools/interp/e17_dinov2_frozen_crossvendor.py   # [N_PAIRS=40]
micromamba run -n svr python tools/interp/e17_viz.py
#   -> temp/dinov2_probe/{results.csv, maps.pt, summary.png, examples.png}

# E18 — trained aggregator (one GPU, ~3 min) + figures
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=training:. micromamba run -n svr python tools/interp/e18_aggregator_anatomy_probe.py   # [N_PAIRS=25]
micromamba run -n svr python tools/interp/e18_viz.py
#   -> temp/aggft_probe/{results.csv, pair03/13/15.pt, dice_vs_layer.png, pairNN_layers.png}
```

- **Determinism.** Both scripts are deterministic: no sampling beyond the fixed `random_state=0`
  pairing. A rerun after moving them into `tools/interp/` reproduced every number above exactly.
- **E18 env vars.** `INTERP_CKPT` (another checkpoint, via harness), `PROBE_DEV` (default `cuda:0`),
  `PROBE_SAVE_PAIRS` (default `3,13,15`).
- **GPU.** Per `~/CLAUDE.md`, never occupy all 4 caesar GPUs. Sharing a GPU that's already busy is
  fine (E18 needs only a few GB).
- **Probing another layer set.** Edit `GLOBAL`; frame blocks can be hooked the same way via
  `agg.frame_blocks[i]`.
