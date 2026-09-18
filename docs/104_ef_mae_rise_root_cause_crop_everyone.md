# 104 — Why EF MAE rose from ~12 pp to ~22 pp: crop-everyone segmentation (~5 pp on the recon) + a ~3 pp checkpoint regression, NOT subjects, centres, or curation

> **TL;DR & takeaway** (2026-09-15). The scatter-arm EF MAE is a **cohort-wide under-contraction**
> (predicted ESV ≈ 1.5× GT, EDV ≈ GT; median error 20 pp, 135/180 test subjects > 15 pp, every
> cohort 17–27 pp) — not particular subjects or centres. The rise from the docs/86-era ~12 pp is two
> things, both measured on the 19 cmrx2024 val subjects shared by the old and new pipelines, with
> the same nnU-Net (Task114) and byte-identical GT: **(1) docs/98's crop-everyone segmentation
> inflates the recon's ES LV volume far more than GT's** (recon ESV ratio 1.29 → 1.65, GT ESV
> +18 %) — +5.4 pp paired, Wilcoxon p = 0.002, 15/19 worse; the 6-DOF registration changes nothing
> (±0.2 pp). **(2) The `curation_*` checkpoints are ~2.5–3 pp worse than `augaggr224` under
> full-FOV scoring** (p ≈ 0.02, 13–16/19 worse), and this is **not curation** — curated (628 train)
> and uncurated (1066 train) ep300 score the same (17.5 vs 17.0) — nor the diffusion regulariser
> (`noreg224` = `augaggr224` at 12.9 vs 13.0 in the old pipeline) nor a split leak (all 19 are val
> in every split). What's left is training-code drift between commits d2693c1 (2026-08-21) and
> 77f968d (2026-09-11) and the training-set change; not isolated.
> **Every method is scored the same way, so the ranking stands, but absolute EF MAEs under v2
> are ~5 pp pessimistic for all arms and should be read as such.**

## 1. Question

The v2 campaign (docs/103) reports VGGT scatter EF MAE 22.5 pp test / 21.6 val, vs 10–13 pp for
every arm in the pre-v2 archive (`evaluation/_archive/metric_results_prev2_20260913/_ef/`). Is it
specific subjects, a centre, training on the curated split, evaluating on curated data, or the
pipeline?

## 2. Not subjects, not centres (per-subject JSONs, no re-run)

Test, `vggt_curation_curated_ep300` scatter, n=180: MAE 22.5, median 20.0, signed −22.4,
135 subjects > 15 pp, dropping every subject > 30 pp still leaves 18.1. Per cohort: cmrx2023 24.3,
cmrx2024 22.0, cmrx2025 23.8, acdc 16.7, mnms 23.3 (val adds miitt 17.9, ocmr 17.6). EDV pred/GT
0.96, **ESV pred/GT 1.56**. Gated diagnostic (same ckpt, gated stack): 4.6 pp, ESV ratio ~1.0 —
the segmenter and scorer are fine; the scatter recon does not contract enough at ES.

## 3. Not curation

| val arm (n=111) | EF MAE | ESV pred/GT |
|---|---|---|
| curation_uncurated ep300 (1066 train) | 19.9 | 1.42 |
| curation_curated ep300 (628 train) | 21.6 | 1.54 |
| curation_uncurated ep177 | 22.6 | 1.55 |

Curated ≈ uncurated. Also every old-pipeline arm sits at 10–13 pp (`noreg224` 12.9,
`augaggr224` 13.0, `augaggr518` 10.2, `augaggr518splat518_ep250` 10.2), so the
`diffusion_weight` 1000 → 0 change in the curation recipe is not it either.

## 4. Pipeline component isolated: it's the crop, not the registration

Setup: the 19 cmrx2024 val subjects present in both the pre-v2 archive and today's bundles; arm
`vggt_docs86_repro_224` (the docs/86 `augaggr224_ep300` checkpoint run through today's engine).
GT `gt_t*.nii.gz` is **byte-identical** between `scratch/eval/_archive_prev2_20260913/` and
`scratch/eval/cmrx2024/` (checked P012, max |Δ| = 0), as is `mask_heart`. Six variants per phase
segmented with the standard `run_seg.sh` (Task114 2d, `nnUNetTrainerV2_MMS`): GT full-FOV / GT
cropped to `mask_heart`; recon raw `recon_breath/vol_t*` (unregistered) full / cropped; recon
`cine_breath` (6-DOF registered) full / cropped. LV EF from each curve's own max/min.

| recon variant | vs GT full: MAE / signed / ESV ratio | vs GT cropped: MAE / signed / ESV ratio |
|---|---|---|
| raw, full FOV (≈ old pipeline) | **14.6** / −13.7 / 1.29 | 12.2 / −11.1 / 1.10 |
| raw, cropped | 19.9 / −19.9 / 1.65 | 17.3 / −17.3 / 1.38 |
| registered, full FOV | 14.8 / −14.8 / 1.33 | 12.2 / −12.2 / 1.13 |
| registered, cropped (= v2) | 19.4 / −19.4 / 1.66 | **16.8** / −16.8 / 1.40 |

- raw-full vs GT-full reproduces the **archived old-pipeline number on these 19 subjects (14.6)**
  exactly: today's `vggt_docs86_repro_224` raw volumes are bit-identical to the archived August
  `vggt_augaggr224_ep300` raw volumes (max |Δ| = 0.0000 over all 19×12), and re-segmenting the
  archived `cine_breath` itself gives the same 14.6 with every subject within 0.1 pp of the
  archived JSON. (The "~12–13 pp" of docs/86/103 is the 143-subject pooled-val cohort; these 19
  cmrx2024 subjects were always harder.) So the like-for-like ledger is just two steps:
  14.6 (Aug ckpt, old scoring) → 16.8 (Aug ckpt, v2 scoring, +2.2 net crop) → 22.1 (Sep ckpt,
  v2 scoring, +5.3).
- registered-cropped vs GT-cropped reproduces the official v2 number for this arm (16.8) exactly.
- Registration: 14.6 → 14.8, 19.9 → 19.4 — no effect.
- **Cropping the recon: +5.4 pp paired (Wilcoxon p = 0.002, 15/19 worse); recon ESV ratio
  1.29 → 1.65.** Cropping GT also inflates GT ESV (54.6 → 64.4 mL mean, GT EF 63.9 → 61.3), which
  cancels ~2.5 pp of it, so net v2-vs-old is +2.2 pp for this checkpoint and +4.6 pp for the
  curated one (17.5 → 22.1).
- Mechanism (from the numbers, not a story): on a black-exterior crop nnU-Net over-segments the LV
  cavity at ES, and does so more on a blurry recon than on GT. docs/98 §3 measured the crop as
  "uniform noise, no bias" on Dice; for the ES *volume* it is a bias that scales with recon
  blur, so it penalises every reconstruction method more than it penalises GT.

Per-subject ESV (mL), GT full vs recon raw-full vs recon raw-crop, worst movers: P120 48.8 / 67.9 /
104.5; P127 32.7 / 39.1 / 63.8; P091 33.8 / 21.4 / 72.1; Val_P054 31.4 / 36.4 / 61.5.

## 5. Checkpoint component: real, ~3 pp, not curation, not isolated further

Same 19 subjects, raw full-FOV recon vs GT full-FOV (old-style scoring):

| checkpoint | EF MAE | ESV ratio | paired vs augaggr224 |
|---|---|---|---|
| augaggr224_ep300 (pooled1337, 935 train, diffusion 1000, commit a031ba7) | 14.6 | 1.29 | — |
| curation_curated_ep300 (628 train, diffusion 0, e7e3944) | 17.5 | 1.41 | +2.9, p = 0.023, 13/19 |
| curation_uncurated_ep300 (1066 train, diffusion 0, 77f968d) | 17.0 | 1.27 | +2.4, p = 0.026, 16/19 |

Ruled out: curation (curated ≈ uncurated), diffusion weight (§3), split leak (all 19 are `[val]`
in `pooled.txt`, `pooled_uncurated_ablation.txt` and `pooled_curated_v2.txt`), aug tier / resp sim /
img_size (identical in `run_meta.jsonl`). Training-code diff a031ba7..77f968d (`training/`, `vggt/`) read
in full: every change is inert at these runs' settings (`splat_res=None`, `motion_l1_weight=0`,
gradient checkpointing default-on, DPT cached layers unchanged, DINOv3 path unused, cache
signature same percentiles) **except one line** — `Aggregator` now builds the DINOv2 patch embed
with a hard-coded `img_size=518` instead of the run's `img_size` (224 here); its effect at 224 is
untested. Remaining candidates therefore: that line, and the train-set composition (935 `pooled`
train vs 628 curated / 1066 uncurated). Separating them needs a retrain. The final518 sweep trains on the
same code as the curation arms, so if this is code drift it carries over; the old-pipeline
trend (`augaggr518` 10.2 vs `augaggr224` 13.0, ep250 vs ep50 27 → 10) says resolution and
training length both help ES amplitude.

## 5b. Where the under-contraction sits: the end planes don't empty at ES (2026-09-16)

Follow-up measurements, all on existing volumes (no eval/ edits; scratch dumps + `run_seg.sh`).

**(a) In-plane pad of the crop, 19 subjects, August ckpt** (same GT/recon, mask dilated in-plane
only, z-extent unchanged): `mask_heart` 16.8 → +10 mm **14.3** (= full-FOV 14.8) → +20 mm 15.3
→ +30 mm 16.1. **(b) +10 mm on all 111 val subjects, curated ep300 (224):** 21.6 → **18.4**
(signed −17.8); per source cmrx2023 26.9→20.7, cmrx2024 22.1→16.4, cmrx2025 22.7→20.3, acdc
15.0→12.3, mnms 23.6→18.7, ocmr 17.6→15.0, **miitt 17.9→22.2** (the exception, see (e)).
**(c) Expert check, ACDC val+test (n=32, expert ED/ES labels):** nnU-Net-on-cropped-GT vs expert
EF MAE **5.0** (signed −2.2); gated-diag VGGT vs expert 6.3; scatter VGGT (curated ep300) vs
expert **18.0**; SVRTK scatter 31.0. The segmenter+crop is ~5 pp of noise with a small *negative*
bias; the scatter under-contraction is real against experts too.

**(d) z-restriction test** (19 subjects, both crops; prediction counted only on planes where GT
has LV — fixed slab = union over phases, per-phase = at that phase):

| crop | all planes | fixed GT slab | per-phase GT planes |
|---|---|---|---|
| tight | 16.8 | 16.9 | 7.3 |
| +10 mm | 14.3 | 14.5 | 4.7 |

On 111 val subjects (+10 mm, 224 arm): all planes 18.4 → per-phase 9.9. A fixed slab changes
nothing; a per-phase restriction halves the error. **Why:** the planes VGGT keeps a cavity on at ES
are planes where GT has LV at ED but not at ES (`figs/ef_crop_104/P120_ED_ES_allplanes.png`,
`basal_planes_zoom.png`: P012/P028/P063/P064/P011 z=9 — GT ES shows atria/valve plane and nnU-Net
finds nothing; VGGT's ES plane is its ED anatomy slightly shrunk, LV 570–830 px). I.e. **the recon
does not reproduce longitudinal shortening: the basal/apical end planes never empty at ES.** ESV
keeps ~2 planes of blood; that is the "ESV ≈ 1.3–1.5× GT" of §2, plane by plane. Per-phase
restriction removes exactly that error by borrowing GT's own shortening, so it is an oracle, not a
fair ROI — usable as a labelled diagnostic ("EF within GT slice coverage" ≈ the in-plane part),
not as the headline. One counter-example (P068 z=7: GT ES still shows a cavity, nnU-Net gave 0)
shows GT-side misses exist; (c) says they don't dominate (a systematic basal drop on GT would
push GT EF *up*, expert comparison says it is 2 pp *down*). The true apex (z=0, GT has no LV at
any phase; VGGT's blurry disc is labelled LV at ED and ES) inflates EDV and ESV alike and barely
moves EF.

**(e) MIITT regression under the pad** (`miitt_MIITT_Volunteer2_pad.png`): 10 mm pitch, larger
apex/base discs; on the padded crop nnU-Net labels VGGT's structureless apical/basal discs as
whole-disc LV at every phase. The tight crop had suppressed that by accident. Same mechanism as
(d), larger, and a 518 arm should be re-checked for it.

**Protocol conclusion:** keep the original ROI z-extent, add a +10 mm in-plane pad for every arm
(needs baselines regenerated with the padded box — `MASK_FILE=` in the runners — and every arm
re-scored; no VGGT re-inference, no training). Don't restrict z. The remaining ~15 pp (224) is
the model: ~half missing through-plane shortening, ~half in-plane under-contraction; the gated
diagnostic (4.6) shows both are recoverable when the phases are given, so it is the
one-frame-per-slice information limit at base/apex, and a training-signal target.

## 6. Consequences

1. **The v2 EF MAEs are ~5 pp pessimistic for every arm** because of the crop, and the penalty
   grows with recon blur — so it hits SVRTK/NeSVoR/NiftyMIC at least as hard as VGGT. Ranking
   and the gated-diagnostic conclusion (docs/103 §3) are unaffected. Whether to keep
   crop-everyone (fair to baselines that only reconstruct inside the ROI, docs/98) or report
   full-FOV EF alongside is a protocol decision for the user; the toggle is one line in
   `ef_dice._dump_cine` and GT segs would need a re-cache.
2. The remaining ~3 pp is a training-side regression between the August and September recipes;
   isolating it means a bisect over d2693c1..77f968d or an ablation on train-set composition —
   not done.

## 7. Repro

Scratch scripts (session scratchpad, not in repo): dump the six variants per phase for the 19
shared subjects → `bash evaluation/src/engine/run_seg.sh <in> <seg>` (needs a manifest with
`meta.dump_id`) → per-variant LV curves. 1368 + 456 volumes, ~25 min on one L40S. The 19 subjects
= `set(old.per_subject) ∩ set(new.per_subject)` of
`_archive/metric_results_prev2_20260913/_ef/vggt_augaggr224_ep300.json` and
`metric_results/val/_ef/vggt_docs86_repro_224.json`.
