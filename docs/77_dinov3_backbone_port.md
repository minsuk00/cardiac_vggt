> **TL;DR & takeaway:** DINOv3 ViT-L/16 is fully integrated as a config-selectable
> training/validation backbone at 256²/patch-16 without changing the default DINOv2
> 518²/patch-14 path. Code, full CPU tests, real checkpoint metadata assembly, and an A40
> forward/backward smoke all pass. Transformers 5.3.0 is installed cleanly in `svr`; gated
> Hugging Face access is authenticated; and the physical 3.76 GB hybrid seed has been built,
> checksum-verified, strict-loaded, and exercised in a real-seed A40 forward/backward smoke.
> **The DINOv3 configuration is ready for a bounded training pilot.** No training has been
> launched yet.

# DINOv3 ViT-L/16 backbone port

Date: 2026-08-13

## 1. Provenance and isolation

All work lives in a dedicated worktree and branch:

```text
worktree: /home/minsukc/vggt-dinov3
branch:   experiment/dinov3
base:     a031ba74cbe1af877d8159746327ef2a3e442644
```

No tracked source in `/home/minsukc/vggt` (`main`) was edited, staged, committed, or merged.
The main worktree was already dirty with unrelated user changes; none were copied into this
branch. The DINOv3 worktree has an ignored `scratch` symlink to the same GPFS target as the
main worktree so caches/checkpoints never become heavy files under `$HOME`.

No commit exists yet. Everything described here is an uncommitted worktree diff.

## 2. Goal and non-goals

Goal: train/validate the existing MRI VGGT pipeline with a frozen, web-pretrained
DINOv3 ViT-L/16 image backbone while keeping VGGT's alternating-attention aggregator and
DPT trainable.

The two selectable configurations are:

| Config | Backbone | Input | Patch grid | Checkpoint loading |
|---|---|---:|---:|---|
| `default` | `dinov2_vitl14_reg` | 518² | 37×37 | original VGGT base, `strict=false` |
| `exp_dinov3` | `dinov3_vitl16` | 256² | 16×16 | exact hybrid seed, `strict=true` |

Non-goals:

- no feature-alignment bridge between DINOv3 and VGGT;
- no scratch-downstream or unfrozen-DINOv3 arm;
- no changes to the trainer, optimizer, loss, splat, augmentation, or validation logic;
- no changes to OOD inference adapters or their existing 518² assumptions;
- no actual training in this worktree yet.

## 3. Runtime switching contract

The same training entry point is used for both variants:

```bash
# Standing DINOv2 behavior (unchanged)
PYTHONPATH=training:. torchrun --nproc_per_node=1 \
  training/launch.py --config default

# DINOv3 variant (only after completing §11)
PYTHONPATH=training:. torchrun --nproc_per_node=1 \
  training/launch.py --config exp_dinov3
```

`training/config/default.yaml` now exposes:

```yaml
img_size: 518
patch_size: 14
backbone: "dinov2_vitl14_reg"
```

These values are threaded into the model and dataset. `exp_dinov3.yaml` inherits the whole
default and overrides only experiment identity, backbone, input/patch size, and checkpoint:

```yaml
backbone: "dinov3_vitl16"
img_size: 256
patch_size: 16
checkpoint:
  resume_checkpoint_path: "./scratch/base_weights/vggt1b_dinov3_vitl16_seed.pt"
  strict: true
```

This keeps DINOv2 the default. Once the optional dependency and seed exist, switching is a
config choice rather than a different trainer or launch path.

## 4. Runtime code changes

### 4.1 Optional DINOv3 adapter

New file: `vggt/models/dinov3.py`.

`DINOv3ViTL16PatchEmbed` lazily imports Transformers and constructs the official ViT-L/16
architecture:

- hidden width 1024;
- 24 transformer layers;
- 16 attention heads;
- MLP width 4096;
- patch size 16;
- four DINOv3 register tokens;
- `use_gated_mlp=false`;
- model image size set to the configured input (256 for this experiment).

The adapter receives the ImageNet-normalized tensor already produced by `Aggregator`.
Transformers' `last_hidden_state` is ordered as:

```text
[CLS] [register 0] [register 1] [register 2] [register 3] [patches...]
```

The adapter removes the first five tokens, validates input divisibility and the exact
spatial-token count, and returns only `(B*S, (H/16)*(W/16), 1024)` patch tokens. At 256² the
result is `(B*S, 256, 1024)`.

The whole adapter remains at `aggregator.patch_embed`. Therefore the existing
`optim.frozen_module_names: ["*patch_embed*"]` freezes and eval-locks all DINOv3 parameters.
VGGT's own `camera_token`, `register_token`, z embedder, frame/global blocks, and DPT are
outside that subtree and remain trainable.

Transformers is not imported at module-import time. The default DINOv2 path still imports
and composes without the optional package installed.

### 4.2 Aggregator selection

`vggt/models/aggregator.py` adds one selection branch:

```text
dinov2_vitl14_reg -> existing in-repo DINOv2 implementation
dinov3_vitl16     -> DINOv3ViTL16PatchEmbed
```

The DINOv3 branch rejects any patch size other than 16. Unknown backbone names now raise an
explicit `ValueError` instead of failing through a dictionary lookup.

VGGT still adds its own camera token plus four VGGT register tokens after backbone patch
extraction. DINOv3's internal CLS/register tokens are not passed into VGGT attention.

### 4.3 Patch-size plumbing

`patch_size` is now explicit through all resolution-sensitive components:

1. `MRIDataset` validates `target_size % patch_size == 0`.
2. `Aggregator` uses it for the 2-D RoPE grid.
3. `DPTHead` receives it explicitly from `VGGT` and reshapes the patch grid accordingly.
4. `BSplineWarpHead` already accepted it; `VGGT` continues passing it.

Default values remain 518/14, so the standing DINOv2 path preserves its 37×37 token grid
and state-dict namespace.

### 4.4 Positional API preservation

The first implementation inserted the new arguments into the middle of existing public
constructor signatures. A correctness review caught that old positional calls would bind
incorrectly. The final signatures append the new options after every legacy parameter:

- `VGGT(..., bspline_grid_size=32, backbone="dinov2_vitl14_reg", **kwargs)`;
- `MRIDataset(..., defer_input_images=False, patch_size=14)`.

Hydra always uses keywords, so active training was never affected by the temporary ordering
bug. The final ordering also preserves external/legacy positional callers.

## 5. Why a hybrid checkpoint is needed

The seed builder is **not used by training**. It is an offline, one-time conversion utility
under `tools/`:

```text
tools/build_dinov3_seed.py                 (manual, run once)
        ↓
scratch/base_weights/vggt1b_dinov3_vitl16_seed.pt
        ↓
training/trainer.py                       (ordinary checkpoint load)
```

The original DINOv2 run needs no conversion because `vggt1b_base.pt` already contains one
coherent upstream model: DINOv2 patch embed + VGGT aggregator + VGGT DPT.

For DINOv3, required pretrained weights come from two sources:

- `vggt1b_base.pt`: VGGT camera/register tokens, alternating-attention blocks, and DPT;
- `facebook/dinov3-vitl16-pretrain-lvd1689m`: DINOv3 ViT-L/16.

Loading only the old VGGT base with `strict=false` would not initialize DINOv3: the new
backbone namespace would remain random. The hybrid seed combines the two sources once, then
the existing trainer loads one ordinary weights-only checkpoint. This avoids modifying the
shared trainer to perform two-source initialization or network downloads at every launch.

The tool belongs under `tools/` because it is a reusable offline checkpoint-conversion step,
not standing runtime code and not evaluation code.

## 6. Exact seed-builder algorithm

`tools/build_dinov3_seed.py` performs these steps on CPU:

1. Load `scratch/base_weights/vggt1b_base.pt`.
2. Accept either the actual raw `OrderedDict` format or a wrapped `{"model": state}` format.
3. Construct the exact DINOv3 target model on the meta device and record every target key and
   tensor shape without allocating the ~1B target weights.
4. Copy only old-base keys requested by that target. This automatically keeps current VGGT
   aggregator/DPT tensors while discarding:
   - the old `aggregator.patch_embed.*` DINOv2 subtree;
   - 525 retired `camera_head.*`, `depth_head.*`, and `track_head.*` tensors.
5. If z-embedder tensors are absent, inject the deterministic initialization described in §7.
6. Load official DINOv3 ViT-L/16 and insert its 415 tensors under
   `aggregator.patch_embed.model.*`.
7. Require exact target-key equality and exact tensor-shape equality.
8. Write a weights-only `{"model": hybrid_state}` checkpoint.
9. Publish atomically. Without `--overwrite`, `os.link(temp, output)` provides atomic
   no-clobber behavior; with `--overwrite`, `os.replace` is explicit.

Default paths:

```text
input:    scratch/base_weights/vggt1b_base.pt
HF cache: scratch/huggingface
output:   scratch/base_weights/vggt1b_dinov3_vitl16_seed.pt
```

All resolve through the worktree's `scratch` symlink to GPFS, never to heavy `$HOME` files.

## 7. Strict loading and the z embedder

### 7.1 Original DINOv2 behavior

The standing config uses `strict=false`. The upstream VGGT checkpoint differs intentionally
from the current MRI model:

- it lacks `aggregator.z_embedder.proj.weight` and `.bias` (added by this project);
- it contains 525 deleted camera/depth/track-head tensors.

The original initialization sequence is:

```text
construct current model -> randomly initialize z embedder under the training seed
load upstream base with strict=false -> tolerate missing z and unexpected old heads
train the retained z initialization
```

### 7.2 DINOv3 behavior

`strict=true` is not an architectural DINOv3 requirement. It is a deliberate safety guard
for a checkpoint that we control and construct to be exact. It ensures that a stale or
mis-namespaced hybrid cannot silently leave some/all DINOv3 weights random.

Strict loading requires every target tensor, including z. The official VGGT base has no z
weights, so the builder supplies them. These are not pretrained DINOv3 weights. They recreate
the same default initialization that a fresh original-style run would retain:

```text
effective training seed = seed_value * max_epochs = 42 * 200 = 8400
```

`ZIndexEmbedder` is the first parameterized module constructed in the aggregator, so creating
it alone under Torch seed 8400 reproduces that initialization. `torch.random.fork_rng` keeps
the builder from mutating its caller's RNG state. If an input checkpoint already contains z
tensors, the builder preserves them instead of replacing them.

A valid alternative would be `strict=false`, omit z from the hybrid, and assert that exactly
those two keys are missing. The current trainer only logs missing/unexpected keys; it does not
enforce an expected set. The exact seed plus `strict=true` therefore provides the safer
failure mode while leaving the default DINOv2 config unchanged at `strict=false`.

## 8. Four issues found by prove-it and their final disposition

### Issue 1 — blocking seed construction

Initial defects:

- loader required a wrapped `{"model": ...}` checkpoint, but the actual base is raw;
- all non-DINOv2 source keys were retained, including 525 retired heads;
- the exact target required two z keys absent from the base.

Final fix: raw/wrapped loading, target-driven key selection, deterministic missing-z
injection, and exact key/shape validation. This was the only issue that blocked a DINOv3
launch.

### Issue 2 — `VGGT` positional compatibility

Initial `backbone` placement shifted the legacy `embed_dim` positional slot. Active Hydra
training used keywords and was unaffected. Final fix: append `backbone` after all legacy
parameters.

### Issue 3 — `MRIDataset` positional compatibility

Initial `patch_size` placement shifted the legacy `mri_mode` positional slot. Active Hydra
training used keywords and was unaffected. Final fix: append `patch_size` after all legacy
parameters.

### Issue 4 — concurrent offline-builder publication

The initial existence check plus unconditional `os.replace` allowed two simultaneous manual
builders to overwrite one another even without `--overwrite`. This never affected training
and was irrelevant to a normal single invocation. Final fix: atomic hard-link publication
for no-clobber mode. A real two-process probe confirmed one writer succeeds, the other gets
`FileExistsError`, the output remains valid, and no temp file remains.

## 9. Dependency installation and isolation

The normal `requirements.txt` remains untouched. DINOv3's dependency stays isolated in the
optional file:

```text
requirements-dinov3.txt -> transformers==5.3.0
```

The dependency dry-run against the original `svr` environment showed that installation would
add only:

- `transformers==5.3.0`;
- `tokenizers==0.22.2`;
- `regex==2026.7.19`;

and would retain the installed `huggingface-hub==1.4.1`. After explicit user authorization on
2026-08-13, the real install matched that plan exactly:

```text
installed: transformers==5.3.0, tokenizers==0.22.2, regex==2026.7.19
upgraded:  none
downgraded: none
removed:   none
torch:     2.13.0+cu130 (unchanged)
hub:       1.4.1 (unchanged)
```

Direct imports of Transformers, Tokenizers, Regex, and `DINOv3ViTModel` pass. Pip's dependency
graph check finds no missing dependency introduced by these packages. The stock `pip check`
command itself crashes while parsing a pre-existing malformed platform tag in
`itk-5.4.5.dist-info/WHEEL`; its separate dependency phase reports only the pre-existing
Gradio/MoviePy versus Pillow/MarkupSafe conflicts. None involves the three packages above.

## 10. Verification evidence

### 10.1 CPU tests

With CUDA hidden explicitly:

```text
focused DINOv3/fix tests: 12 passed
full suite before dependency install: 359 passed, 399 warnings, 183.08 s
full suite after install + seed build: 359 passed, 399 warnings, 93.90 s
git diff --check:        passed
static compilation:      passed
```

Warnings were pre-existing Torch/Monai/NumPy/SimpleITK warnings, not failures.

### 10.2 Real checkpoint and Transformers metadata

A read-only `FakeTensorMode` + `mmap` inspection of the actual VGGT base proved:

```text
format:                raw OrderedDict
base keys:             1797
z-embedder keys:       0
retired SfM-head keys: 525
```

Transformers 5.3 constructed the full ViT-L/16 model on the meta device. Combining its real
state namespace with actual VGGT base metadata through the repaired builder produced:

```text
DINOv3 source keys: 415
target keys:        1345
hybrid keys:        1345
key equality:       exact
shape equality:     exact
```

This first validated the conversion algorithm and strict-loading schema without downloading
the official tensor values or writing the heavy hybrid file. After gated access was
authenticated, the authorized CPU builder downloaded the official values to GPFS and wrote:

```text
path:          scratch/base_weights/vggt1b_dinov3_vitl16_seed.pt
bytes:         3,762,602,088
SHA-256:       318ab2acf264cb47fce906605c15da40166d2d58f101dd446a2b0bce4543fb4f
payload keys:  model (and nothing else)
state tensors: 1345
DINOv3 tensors: 415
old DINOv2 tensors remaining: 0
```

The persistent official-model cache is GPFS-backed under `scratch/huggingface/` and occupies
approximately 1.2 GB. The GPFS seed and its node-local `/tmp` staging copy had identical
SHA-256 hashes. Both deterministic z-embedder tensors matched the builder initializer exactly.
Loading the physical state into the exact target model with `strict=true` reported zero missing
and zero unexpected keys.

### 10.3 Real Transformers CPU forward

A two-layer reduced DINOv3 model using the real Transformers 5.3 implementation produced the
expected `(2, 4, 32)` patch-token output for two 32²/patch-16 inputs after stripping five
special tokens. Repeated frozen/eval forwards were bit-identical. Full ViT-L/16 construction
and target-state enumeration succeeded on the meta device.

### 10.4 A40 forward/backward smoke

The full random-initialized DINOv3 ViT-L/16 + VGGT aggregator + DPT model ran on the allocated
NVIDIA A40 outside the sandbox:

```text
input:       B=1, S=2, 3×256×256
world shape: (1, 2, 256, 256, 3)
conf shape:  (1, 2, 256, 256)
loss:        5.059440
peak memory: 5.97 GiB
runtime:     25.0 s
```

Backward checks:

```text
DINOv3 patch_embed: frozen, eval-locked, no gradients
frame blocks:       432 finite gradient tensors, norm 0.0137834
global blocks:      432 finite gradient tensors, norm 0.0137476
z embedder:           2 finite gradient tensors, norm 4.38138e-06
DPT point head:       62 finite gradient tensors, norm 9.18100
```

Outputs and all checked gradients were finite and nonzero where required.

### 10.5 Independent review

The initial three-reviewer prove-it pass found the four issues in §8. Each survived a fresh
adversarial verifier before being accepted. After the fixes, three independent deep-check
reviews covered the seed builder, `VGGT`, and `MRIDataset`; none found a remaining correctness
issue. The seed review included a true two-process no-clobber probe and exact seeded-z
comparison.

### 10.6 Real-seed A40 forward/backward

After the physical hybrid was built, the exact checkpoint was loaded with `strict=true` into a
normally constructed model, frozen using the shipped `optim.frozen_module_names` pattern, and
run for one synthetic forward/backward on the NVIDIA A40:

```text
input:                       B=1, S=2, 3×256×256
world shape:                 (1, 2, 256, 256, 3)
confidence shape:            (1, 2, 256, 256)
loss:                        0.1655164212
patch_embed mode:            eval
patch_embed gradient tensors: 0
frame-block gradients:       432 tensors, norm 0.2804114
global-block gradients:      432 tensors, norm 0.2894640
z-embedder gradients:          2 tensors, norm 0.00147342
DPT gradients:                62 tensors, norm 6.8525357
peak allocated GPU memory:   5.97 GiB
forward + backward:          0.81 s
```

Outputs and every checked trainable gradient were finite. This used the downloaded official
DINOv3 weights and the completed hybrid seed, not the earlier random-initialized smoke.

Verification-harness gotcha: the first real-seed probe constructed the model on `meta`, loaded
the state with `assign=true`, and then failed at `.cuda()` because Transformers retains a
non-persistent buffer that is absent from the state dict and therefore remained a meta tensor.
That is not the training construction path and did not expose a checkpoint mismatch. The probe
was corrected to construct a normal CPU model exactly like training, strict-load it, and then
move it to CUDA; the results above are from that successful corrected run. Do not use
meta-construction plus `assign=true` for a runnable device-transfer smoke.

## 11. Current launch gate and exact next steps

As of this document's date:

```text
transformers installed in shared svr:                     YES (5.3.0)
official DINOv3 repository access authenticated:         YES
official DINOv3 weights downloaded to persistent GPFS:   YES
physical hybrid seed file built:                         YES
strict load of the physical seed exercised:              YES
actual DINOv3 training launched:                         NO
```

All launch prerequisites are now complete. `--config exp_dinov3` is ready for the bounded pilot
in Step 6. The first authorized builder attempt did stop before writing output with
`401 Unauthorized`; after the user authenticated gated access, the retry completed and all
physical-checkpoint checks above passed.

### Step 1 — optional dependency: completed

The authorized installation completed with the exact versions in §9. No further package
installation is required.

### Step 2 — gated-repository access: completed

1. Sign in at
   `https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m` and accept/request access.
2. Create a read-only user access token at `https://huggingface.co/settings/tokens`.
3. In a private terminal, authenticate with `HF_HOME` on GPFS. Do not paste the token into an
   agent chat:

```bash
HF_HOME=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/huggingface \
  hf auth login
```

Confirm access without exposing the token:

```bash
HF_HOME=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/huggingface \
  hf auth whoami
```

The token file and downloaded weights stay on GPFS, not under `$HOME`. `hf auth whoami`
successfully confirmed the authenticated account before the download.

### Step 3 — stage the large VGGT source checkpoint locally: completed

Run on the compute node, not a login node:

```bash
cp scratch/base_weights/vggt1b_base.pt /tmp/vggt1b_base.pt
```

This avoids the documented pathological small-read `torch.load` behavior on GPFS.

### Step 4 — build the hybrid seed on CPU: completed

```bash
CUDA_VISIBLE_DEVICES='' \
HF_HOME=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/huggingface \
HF_HUB_CACHE=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/huggingface/hub \
PYTHONPATH=training:. python tools/build_dinov3_seed.py \
  --base /tmp/vggt1b_base.pt \
  --hf-cache scratch/huggingface \
  --output scratch/base_weights/vggt1b_dinov3_vitl16_seed.pt
```

The Hugging Face cache and output are GPFS-backed through `scratch`. Do not redirect them to
`$HOME`. Do not pass `--overwrite` unless replacing an inspected bad seed intentionally.

### Step 5 — inspect and strict-load the physical seed: completed

Copy the completed seed to node-local `/tmp` before `torch.load`. Confirm:

- top-level payload is exactly `{"model": state_dict}`;
- state has 1345 keys;
- strict model loading reports no missing/unexpected keys;
- DINOv3 remains frozen after the normal freeze call;
- aggregator and DPT remain trainable.

All five checks passed; exact values are recorded in §10.2 and §10.6.

### Step 6 — launch a bounded pilot before a long run

Use the existing training entry point and `exp_dinov3`, first with a short bounded override
appropriate for a smoke run. Only after checkpoint load, one train step, validation, logging,
and checkpoint save/resume are proven should a long training job be launched.

## 12. File inventory

Runtime/config changes:

- `vggt/models/dinov3.py` — optional DINOv3 adapter;
- `vggt/models/aggregator.py` — backbone selection;
- `vggt/models/vggt.py` — backbone and DPT patch-size plumbing;
- `training/data/datasets/mri_dataset.py` — configurable grid validation;
- `training/config/default.yaml` — explicit DINOv2 defaults;
- `training/config/exp_dinov3.yaml` — DINOv3 experiment override.

Offline/dependency/test/docs additions:

- `tools/build_dinov3_seed.py` — one-time hybrid checkpoint builder;
- `requirements-dinov3.txt` — optional dependency pin;
- `tests/test_dinov3_variant.py` — DINOv3 and issue-regression tests;
- `docs/77_dinov3_backbone_port.md` — this record (renumbered from 75 at merge: main had taken 75/76);
- `docs/README.md` — index entry.

No trainer, loss, splat, augmentation, evaluation, or inference-adapter file was modified for
this port.
