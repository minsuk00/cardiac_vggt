# 87 — Real real-time free-breathing MIITT inference: `run_vggt_rt.py`

> **TL;DR & takeaway**
> The one-off Aug-18 probe that fed REAL MIITT real-time free-breathing recordings to the model
> is now a standing harness script, `evaluation/src/engine/run_vggt_rt.py`: per subject it
> reconstructs a full 3D volume for **every** real acquired frame (180) by sweeping the reference
> slot over the recording while each companion slot keeps ONE fixed real frame of its own z-plane
> (the trained one-frame-per-slice contract). Zero-shot for non-MIITT-pooled checkpoints,
> **qualitative only** (retrospective subsampling, no GT). Geometry comes from the standard
> `MRIDataset` path pointed at the RT data itself, so subjects whose RT/gated slice counts differ
> (the AFib patient) just work. Run so far: Volunteer1/2/3 on `augaggr224hw2_ep300` +
> Volunteer1/2/3 + AFib on the base `augaggr224_ep300` (awrobewn). Outputs (recon NIfTI + input +
> 4-row GIF + DVF panel) under `evaluation/volumes/miitt_rt/out/`. Prove-it-reviewed clean.

## 1. What it does (2026-08-20)

Promoted from `temp/miitt_rt_probe_full/miitt_rt_driver_full.py` (2026-08-18 session, README
there). Per subject:

1. Load `scratch/data/MIITT/nifti/<subj>/realtime/sax/4d_recon.nii.gz` (~180 frames × D slices,
   golden-angle spiral R=12, 2.3×2.3 mm, 25 ms/frame — protocol values confirmed by the data
   author 2026-08-20, replacing the old 2.6 mm placeholder). Normalize [p0.5, p99.9]→[0,1],
   resample 2.3→1.4 mm, center crop/pad to the canonical 256×256 → `(180, D, 256, 256)` bundle.
2. Build ONE batch via the standard `MRIDataset.get_data` (docs/79 one-implementation rule),
   pointed at the RT recording itself: `make_rt_scaffold` writes 12 evenly-spaced RT frames as a
   temp MIITT_sax-layout subject; dz=10 mm (8 mm + 2 mm gap) comes from the RT header. Companion
   slots each get one fixed representative RT frame (gated-t → linspace index remap); slot 0 =
   mid-plane reference.
3. Sweep slot 0 over all 180 real frames — one forward + splat per frame (~100 ms/frame on A40;
   full recording ≈ 19 s GPU). Companions never change: same scattered acquisition, 180 queries.
4. Save per subject to `evaluation/volumes/miitt_rt/out/<subj>/`: `rt_input.nii.gz` (canonical
   resampled input) and `<model_name>/{recon_rt.nii.gz, gif_rt.gif, panel_dvf_rt.png}`. GIF rows:
   gated context (only when the subject's own gated D matches — cycled, NOT phase-aligned) |
   raw RT | model input (companions frozen, reference plane animating, red-starred) | recon;
   played at the true rate (25 ms/frame → 40 fps ÷ display stride).

```
micromamba run -n svr env PYTHONPATH=training:. python evaluation/src/engine/run_vggt_rt.py \
    --ckpt scratch/logs/<run>/ckpts/checkpoint_last.pt --model-name vggt_<slug> \
    [--subjects MIITT_Volunteer1 ...] [--stride 3]
```

## 2. Framing (what this is and is not)

- **Closest data to the deployment target in the project**: real breathing, no gating, genuinely
  sparse per-slice acquisition — the docs/83 §6.6 "load-bearing" OOD direction.
- **Still retrospective + unscored**: companions are subsampled post-hoc from a densely sampled
  recording, and real-time data has NO ground truth — judge by eye; no metrics are produced and
  none should be cited from this path.
- **Zero-shot** for checkpoints trained on `pooled.txt` (zero MIITT lines — verified from
  run_meta in the probe session). Volunteer1/2/3's *gated* scans are TRAIN subjects in
  `pooled_miitt.txt`, so for the miitt-pooled arm prefer val subjects (Volunteer6/7/8) if their
  RT data qualifies.
- The recon's timestamp is the mid-slice's frame time (per-slice cines are not simultaneous
  across z; the script never assumes they are).

## 3. Decisions & gotchas

- **Geometry from RT, not gated**: an early version reused the subject's gated `MRIDataset` entry
  (valid — same prescription) but broke on the AFib patient (RT D=12 vs gated D=15) and needed a
  donor-subject hack. Replaced by the temp-scaffold approach: self-contained per subject, no
  gated dependency, same single geometry implementation.
- **NIfTI has no float16** — recons saved float32 (gzip absorbs most of it; ~357 MB/subject).
- **`build_batch_rt` deliberately duplicates ~15 collate lines of `run_vggt.build_batch`**: the
  original hard-requires bundle T == gated T, the exact constraint a non-cyclic RT recording must
  drop. Not worth touching `run_vggt.py` for.
- Missing heart ROI is safe at eval: `get_data` just omits the key (the docs/78 hard-raise is
  train-loss-side).
- Prove-it review (1 reviewer + adversarial verification + real GPU runs of every path): 0 bugs;
  `rt_input.nii.gz` round-trip byte-exact vs independent re-derivation; RT-vs-gated z-order
  parity settled by provenance (docs/78: no flips applied to either MIITT arm, one converter).

## 4. Runs on disk

| arm | subjects | note |
|---|---|---|
| `vggt_augaggr224hw2_ep300` (cfvoed6b) | Volunteer1/2/3 | re-rendered to the 4-row layout |
| `vggt_augaggr224_ep300` (awrobewn) | Volunteer1/2/3 + Patient_2024Jan04_Cardiomyopathy_AFib | AFib: D=12 own-RT geometry; gated row = central 12 of its 15 gated slices (approx alignment, display only) |

Open/next: other arms (518/p2p98/miitt-pooled) are one command each; val-split MIITT subjects for
the miitt-pooled arm; any future scoring (self-consistency, seg-EF on RT) would consume
`recon_rt.nii.gz` — none defined yet.
