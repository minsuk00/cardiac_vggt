# 28 — Multi-frame-per-slice input + continuous physical z

> **TL;DR & takeaway**
>
> Two staged changes to the input sampling, both implemented + test/smoke-validated (2026-06-30),
> **gains not yet confirmed by a real training run.**
>
> **(A) Multi-frame per slice (default ON, `mri_finetune.yaml` budget now S=20).** The old pipeline
> fed *one frame per z-plane* (sparse, blind input phase). Now each sample is a **fixed S=20** set:
> **full z-coverage** (every in-bbox plane ≥once) **+ uniform-random extra frames**, a multi-frame
> regime closer to real acquisition (many frames per slice) and
> closer to original VGGT's 2–24-view training. **Why:** a single SAX slice gives almost no cue about
> its own through-plane (Δz) displacement, so the model output ≈0 Δz; a per-slice *cine* makes phase,
> contraction **amplitude**, and through-plane motion observable from *content* — without any
> cardiac-phase label (still blind input `t`), and without the flat-EF collapse (docs 24/25). Fixed
> S=20 (not random) avoids allocator fragmentation. Full coverage keeps the **full-volume L1 loss
> valid unchanged** (no holes).
>
> **(B) Continuous physical z (default OFF, gated `continuous_z`).** Each non-reference slot's z is
> jittered off its integer plane (±`z_jitter`=0.5, clamped to `[0, D-1-eps]`) and the slice is
> extracted by 2-plane linear interpolation. Teaches z as a **continuous physical coordinate** so
> off-grid real-acquisition slices splat correctly (the splat is already continuous). It is a
> **train-time augmentation** (valid because the gated source volume is spatially coherent — you can
> interpolate a slice at z=5.3; at inference you feed real scattered slices at their measured z, no
> interpolation). Default OFF ⇒ numerically identical to the discrete-grid pipeline.
>
> **Cost on an A40 (measured):** S=20 train ≈ **37 GB / 45.5 GB** (no creep, fixed-S); inference fits
> ~96 slots. **Untouched:** loss, model/aggregator, splat, confidence. **Next:** a real multi-frame
> training run to confirm the z-motion / EF gains, then ablate `continuous_z`.

---

## Why (the problem these solve)

The headline goal is fast free-breathing cine from a *few scattered single-frame-per-slice*
acquisitions. The prior pipeline simulated exactly that — but two limitations surfaced:

1. **Through-plane motion is starved (docs 21/22).** With one blind-phase frame per plane, Δz is
   nearly unobservable from image content (a SAX slice slid in z just looks like a slice at a nearby
   z). The model defaults to ≈0 Δz. The old explicit-`target_t` path recovered *some* Δz but caused
   the flat-EF amplitude collapse (docs 24/25) — a genuine fork.
2. **Amplitude/EF is only weakly observable.** The reference slot (docs 25) fixes the *query*, but a
   sparse input still under-observes the patient's contraction.

**Multi-frame per slice resolves the fork instead of trading sides:** a temporal sequence at a plane
makes both amplitude (the cine shows contraction) and through-plane motion (structures enter/leave
the plane over time) observable **from content** — no explicit `t` (honest: at inference you have the
stream, not labels) and no flat-EF (amplitude no longer guessed). It is also the *classical SVR*
regime and closer to VGGT's many-view training. Caveat: SAX is fundamentally weak through-plane, so
this should *improve* Δz, not fully solve it — to be measured.

**Continuous z** addresses inference reality: real slices land off the 12-plane training grid, and you
**cannot resample free-breathing slices onto the grid** (through-plane interpolation needs a coherent
stack; scattered free-breathing slices are each at a different respiratory/cardiac state — and
resampling would presuppose the very alignment the model exists to produce). So the model must accept
continuous z natively; the jitter augmentation exercises the z-embedding/splat across the continuum.

## What changed (precise)

All sampling lives in `MRIDataset.get_data` (`training/data/datasets/mri_dataset.py`). The model
already consumes arbitrary per-slot `(t, z, scanner_coords)`, so it is **untouched**.

- **S budget:** dropped `S = min(T_total, bbox_z_size, …)`; now `S = img_per_seq or num_slices` (=20).
- **z sampling:** slot 0 = `(t_target, z_mid)` reference (unchanged); then **coverage** of every other
  in-bbox plane once; then **uniform-random extras** via `rng.choices(in_bbox_z, k=n_extra)` to
  fill S. Guards: empty-bbox (`or [z_mid]`), `img_per_seq < n_planes` (subsample coverage). Static
  mode + `n_forced_target` ablation hook preserved. The non-slot-0 tail is shuffled (order is
  irrelevant to the set-attention model; keeps val varying per `seq_index`).
- **`t` is extraction-only:** a per-slot phase is sampled solely to pull the right slice content; it is
  **never** fed as model conditioning (`t_indices`/`target_t` stay inert — real-time CMR has no phase
  label). The model reads phase from the reference slot + cine content.
- **Continuous z (gated):** `continuous_z` + `z_jitter` kwargs/config. When on, non-reference slots get
  `z + uniform(-j, j)` clamped to `[0, D-1-eps]`; extraction blends the two bracketing planes
  (`(1-f)·plane_z0 + f·plane_z1`). Slot 0 stays integer (keeps the filmstrip's integer reference gather
  valid). The `[0,D-1-eps]` clamp also **fixes a pre-existing splat top-plane drop** (a slice at exactly
  z=D-1 was discarded by the in-bounds gate).

**Two index namespaces (key to the staging).** `slice_indices`/`timesteps` are the integer-gather z/t;
`z_indices`/`t_indices` are the float conditioning scalars. Continuous z only required making
`slice_indices` **float32** through collation (`composed_dataset.py`) + the trainer's two diagnostic
casts, and upgrading the one integer gather (`gpu_aug.extract_slices_from_phases`) to interpolate
(exact at integer z ⇒ discrete pipeline numerically unchanged). The default-on **respiratory** reslice
was already float-safe (grid_sample), so it needed no change.

**Config:** `mri_finetune.yaml` — `max_img_per_gpu`/`img_nums`/`num_slices` 12→20; new top-level
`continuous_z: false` + `z_jitter: 0.5` wired into both dataset blocks. No `PYTORCH_CUDA_ALLOC_CONF`
change needed (fixed S ⇒ no fragmentation; `expandable_segments` is already set in `trainer.py`).

## Verification (done)

- **205 tests pass.** Rewrote the 3 S-cap/distinct-z contract tests; added multi-frame coverage /
  LV-bias / reference-slot-0 tests; added 6 continuous-z dataset tests + a re-extraction float-safety
  test (`extract_slices_from_phases` float-z exact at integer, blends at z=5.5). Fixed the now-obsolete
  "scattered slots exclude the reference plane / distinct z" assertion in `test_reference_conditioning`.
- **Real-loop smokes (A40, this node):** S=20 multi-frame trained end-to-end (loss 0.051→0.020, val
  per-phase logging intact, **peak 36.7 GB, no creep**); `continuous_z=true` trained end-to-end + saved
  a checkpoint (float jitter + float32 collation + respiratory float-safe + interp extraction, no
  errors, 36 GB). `img_per_seq == n_planes` reproduces the old full-coverage/distinct-z behavior.

## Status / next

Implemented + validated mechanically. **Not yet confirmed to improve metrics** — needs a real
multi-frame training run (then re-eval Δz / per-patient EF). Sequencing: train Phase A (S=20,
`continuous_z=false`) first to attribute the multi-frame gain, then ablate `continuous_z=true`.

**Deferred (decide by ablation later):** a data-driven "reference-cine plane" flag marking the dense
LV plane as the temporal clock (the LV cine is a *complete* observation while off-LV slices are
*partial* — explicit signalling may help, given the flat-EF precedent that implicit global inference
can fail); Level-B varying-pitch resampling; confidence-weighted splatting (only once real artifacted
inputs exist).
