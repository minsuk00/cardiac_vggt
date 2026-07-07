# 31 — Updated baseline roster: the four SVR tools, single- vs multi-stack, and what our data can actually support

> **TL;DR & takeaway**
>
> A consolidated, decision-oriented roster of the reconstruction baselines for VGGT-MRI,
> synthesizing docs `24`(report)/`29`/`30` plus a long design discussion (2026-07-02). Read
> this to know *which* baselines to run, *on what input*, and *why*.
>
> **The roster — four runnable SVR-family baselines + a floor.** `NiftyMIC` (classical, rigid,
> **already run**, n=2), `SVRTK` (classical, rigid — same class as NiftyMIC, uses an EM-robust
> recon, community-standard citation), `NeSVoR` (the *only* usable **ML** SVR — learned pose +
> neural field), and `fetal_cmr_4d` (the *only* self-gating end-to-end **4D cardiac** pipeline).
> Plus the **floor** = stack + trilinear/B-spline interpolation (≈ identity-Δ). **DMCVR is
> dropped** — it's learned through-plane *interpolation* on a coherent breath-hold stack, not SVR
> (no motion model), so it can't be a free-breathing/motion comparator.
>
> **The single most important fact: through-plane reconstruction from a *single* orientation is
> mathematically underdetermined.** Classical SVR super-resolves the coarse (through-plane) axis
> *only* if something samples that axis at sub-slice offsets — a second **orientation** (LAX/LVOT),
> overlapping slices, or motion scatter. On single-orientation SAX **every** SVR tool (NiftyMIC /
> SVRTK / NeSVoR / fetal_cmr_4d) degrades to through-plane **interpolation** — this is a property
> of the *input*, not the tool. Switching tools does not fix it.
>
> **This is exactly VGGT-MRI's thesis.** Classical SVR earns through-plane resolution from *extra
> acquisition* (multi-orientation stacks); VGGT-MRI earns it from a *cohort-trained prior* on one
> orientation. So the contribution is "trade acquisition redundancy for a learned prior." It's a
> real, unpublished combination (gating-free + respiratory-corrupted + phase-scattered +
> single-orientation + intensity), **but scope it honestly**: the through-plane is prior-driven,
> so the model recovers **EF/timing and observed-phase content**, while per-voxel through-plane
> **motion PSNR is acquisition-limited** (docs `19`/`20`/`21`/`22`/`24`). Don't sell "recovers
> arbitrary through-plane motion" or "proven real-time transfer" (still simulated).
>
> **Baseline-input decision:** run classical SVR **single-stack** (matched input — its degradation
> to interpolation is the *evidence* the prior does real work) **AND** multi-stack (its designed
> best, only possible on **gated CMRxRecon SAX+LAX+LVOT with GT**) as an upper bound, so you can
> claim "single-orientation prior ≈ multi-orientation classical SVR" without being accused of
> strawmanning. Keep the *through-plane* claim (multi-stack, gated) separate from the
> *free-breathing* claim (single-stack, no GT).
>
> **Data reality:** every real free-breathing source we have (OCMR, Göttingen, MIITT) is
> **SAX-only** — and that's structural, not a collection gap (multi-orientation SVR needs a slow
> stack-of-stacks protocol that defeats the point of going real-time). The combination
> **multiplanar + free-breathing + ground-truth does not exist** in our catalog, which is the
> fundamental reason no baseline cleanly matches VGGT-MRI's task.

---

## 1. The roster

| Baseline | Family | Learned? | Output → metric | Status | Role |
|---|---|---|---|---|---|
| **Floor** (stack + interp) | naive | no | intensity → PSNR/SSIM | your own code | lower bound ≈ identity-Δ; always include |
| **NiftyMIC** | A · classical SVR | no | intensity → PSNR/SSIM/EF | **run** (n=2, doc `29`) | "no-ML optimization SVR" comparator |
| **SVRTK** (`mirtk reconstruct`) | A · classical SVR | no | intensity → PSNR/SSIM/EF | not started | 2nd classical point; EM-robust recon; standard citation |
| **NeSVoR** (+SVoRT) | B · learned INR-SVR | yes | intensity → PSNR/SSIM/EF | **run** (n=2, doc `32`) | the *only* usable ML SVR; per-scan (no cohort prior) |
| **fetal_cmr_4d** | A · classical 4D cardiac | no | 4D intensity → PSNR/SSIM/EF | not started | *only* self-gating end-to-end pipeline; cite/limited-run |
| ~~DMCVR~~ | B · learned diffusion | yes | intensity | **dropped** | interpolation on coherent stack, not SVR — see §2 |

The four SVR-family tools are the live options. VGGT-MRI is itself in family B (learned), targeting
family A's output (intensity volume).

### 1.1 Why DMCVR is dropped

`_html/24` listed DMCVR as "closest ML baseline (clean-input upper bound)." On reflection it is not
a usable comparator for the free-breathing / motion-correction question: DMCVR assumes a **coherent
breath-hold stack** and learns to synthesize through-plane content to fill the slice gap — i.e.
**learned through-plane super-resolution / interpolation, with no registration, no motion model, no
gating.** It reconstructs whatever (mis)aligned configuration it is handed. It can still be *cited*
as "even given a clean stack, a learned method tops out at X," but it does not test motion
correction, so it is not in the SVR baseline set. (It remains adjacent prior art: "a learned prior
fills through-plane from sparse SAX" is not itself novel — DMCVR did it in breath-hold; VGGT's
novelty is the harder free-breathing/gating-free/scattered combination, not single-stack-prior in
the abstract.)

## 2. Capability matrix (verified; extends doc `30` §2)

| | NiftyMIC | SVRTK | NeSVoR | fetal_cmr_4d |
|---|---|---|---|---|
| **Self-gating** | No | No (cardiac mode needs labels supplied) | No (you bin the phase) | **Yes** — `cardsync_intraslice/interslice` |
| **Cardiac-phase concept** | None | `reconstructCardiacVelocity` = velocity/4D-flow only, **unusable for magnitude cine** | None | Yes (full 4D pipeline) |
| **Registration engine** | RegAladin (NiftyReg, rigid block-matching) | MIRTK (rigid `reconstruct`; deformable `reconstructFFD`) | SVoRT transformer (learned pose; rigid+deformable) | Hierarchical (stack↔stack, slice↔volume, frame↔volume) over MIRTK |
| **Reconstruction (SRR)** | PSF-aware, Tikhonov (L2) + LSMR; outlier rejection **off** by default | **EM robust** statistics (Kuklišová-Murgašová lineage), on by default | Implicit neural representation (neural field) | **Uses SVRTK's recon engine** + robust EM |
| **Needs multi-orientation for through-plane?** | Yes (else interpolates) | Yes | Yes (single-stack variants exist, still limited) | Yes (many orientations) |
| **Runs on single SAX stack?** | ✅ (we did) | ✅ | ✅ | ⚠️ off-design, probably; stack↔stack step becomes a no-op |
| **Effort on our data** | done (3 bugs fixed) | new Singularity pull, expect same-class bugs | Docker, GPU | highest: MATLAB + SVRTK build + manual MITK masks, ~6 h/case |

**Key relationships:**
- **fetal_cmr_4d uses SVRTK as its reconstruction engine.** Its contribution *over* plain SVRTK is
  the **self-gating + hierarchical registration** wrapper, not a different reconstruction. So
  fetal_cmr_4d's recon ≈ SVRTK's recon.
- **NiftyMIC's recon genuinely differs** from SVRTK/fetal (RegAladin + Tikhonov-L2 vs MIRTK +
  EM-robust). NiftyMIC is less outlier-robust by default — this will matter once input is corrupted.
- All are **classical SVR** in the same sense: the alternating **register ⇄ reconstruct** loop over
  the PSF forward model `y_k = D · B · M_k · X + noise`. NeSVoR replaces the hand-tuned loop with
  learned pose + a neural field but chases the same output.

## 3. The core mechanism: why single-orientation ⇒ interpolation (not a tool flaw)

Super-resolution reconstruction (the SRR half of SVR) can only recover the coarse (through-plane)
axis at higher resolution than the slices **if something samples that axis at sub-slice-thickness
offsets.** Diversity of sampling in the coarse direction = new information.

A single SAX stack is a set of **parallel** planes sharing **one** through-plane axis, sampled only
on the slice grid (e.g. our canonical Z = 12 mm pitch), with **gaps no slice ever touched**. So:

1. **SRR has nothing to super-resolve through-plane** → it falls back on the regularizer `λ·R(X)`
   → smooth **interpolation**, no resolution gain.
2. **Through-plane registration is ill-posed** — the volume built from parallel slices has no
   through-plane texture to lock onto, so `M_k`'s z-component is unconstrained.

**What fixes it:** sub-slice sampling diversity in the coarse axis — from (a) a second
**orientation** (LAX/LVOT, whose fine in-plane axis runs *along* SAX's coarse axis), (b) overlapping
/ interleaved slices, or (c) motion scatter (the fetal case — subject motion itself provides the
diversity once registered out). Our SAX data has **none** of these.

**Corollary:** on single-orientation SAX, *all four* tools interpolate through-plane. The limitation
is the **input**, not the tool — you cannot escape it by swapping NiftyMIC → SVRTK → NeSVoR. This is
the exact gap VGGT-MRI's learned prior is meant to fill.

### 3.1 The n=2 clean-stack run measures pipeline round-trip loss, not motion correction

Important caveat carried from doc `29`/`30`: the n=2 NiftyMIC run fed `phases[t_target=0]`, which
**is our `V_gt` itself** (already-aligned ED SAX stack from the canonical cache). There is **no
misalignment to correct** — so the ~24 dB `PSNR_anat` is NiftyMIC's *self-inflicted round-trip
damage* (resample onto its 1 mm isotropic grid + Tikhonov smoothing + PSF deconvolution of a blur
that isn't there + LSMR stopped at 10 iters + intensity drift), not a motion-correction result. A
super-resolution/registration pipeline is **not** an identity map; pushing the GT through it dents
it. The comparison is also degenerate: NiftyMIC got the *answer* as input (single-phase clean
stack) while VGGT's eval got phase-scattered S=20 input, yet VGGT scored higher only because `V_gt`
is literally its training target. **The clean-stack run is a runs-end-to-end sanity check, not a
comparison.** A meaningful test needs real misalignment (§6).

## 4. Single-stack vs multi-stack — run BOTH

The contribution is "single-orientation free-breathing reconstruction via a learned prior." Proving
it well needs two comparison points, not one:

- **Single-stack classical SVR (matched input).** Hand NiftyMIC/SVRTK the *same* single-orientation
  input as VGGT and show they degrade to through-plane interpolation. **The baseline's failure is
  the evidence the prior does real work.** This is the honest, matched-input comparison and the
  direct support for the claim.
- **Multi-stack classical SVR (baseline at its designed best).** Give classical SVR its full
  redundancy (SAX+LAX+LVOT) and show **VGGT-on-single-stack approaches SVR-on-multi-stack.** This is
  the *stronger, more defensible* result ("our prior on one orientation matches what classical SVR
  needs three orientations to achieve") and it **preempts the strawman objection** ("of course
  single-orientation NiftyMIC fails — that's not what it's for").

**Target table:** (1) floor, (2) classical SVR single-stack, (3) classical SVR multi-stack, with
VGGT single-stack throughout. This structure defeats both "your baseline is a strawman" and "your
method isn't better than classical SVR."

**Constraint:** the multi-stack point only exists where multi-orientation stacks *and* GT coexist —
i.e. **gated CMRxRecon** (raw `ChallengeData` has `cine_sax/lax/lvot.mat` at the same real T=12
phases; doc `30` finding 4). So the multi-stack comparison demonstrates the **through-plane**
argument on *gated* data; it does **not** test the *free-breathing* argument. Keep the two claims
separate in any write-up.

## 5. Real free-breathing data is structurally SAX-only

Confirmed for all three OOD eval sources:
- **Göttingen** — SAX only (each subject = one SAX stack; doc `16`).
- **OCMR** — we selected `viw=sax` real-time stacks; non-SAX OCMR series are single real-time slices
  (e.g. a real-time 4-chamber), **not** co-registered multi-orientation stacks of the same heart
  (doc `06`).
- **MIITT** — SAX only (`gated/sax`, `realtime/sax`; doc `23`).

This is not a collection gap to fill with LAX. An LAX *real-time* cine is trivially acquirable, but
the **multi-orientation, co-registered, SVR-ready set** SVR needs is a **special slow
stack-of-stacks protocol**: many orientations acquired sequentially, each self-gated, then
**cross-orientation respiratory motion correction** because breathing moved the heart between the
SAX and LAX acquisitions (so their nominal geometry is wrong). "Co-registered" spans three
alignments: shared geometry (free within a session), cardiac-phase sync across orientations (requires
self-gating each stack), and respiratory motion correction (the hard part). You *acquire* the
multi-orientation streams; the SVR pipeline *does* the cardiac-sync + motion-correction to produce
the aligned set.

Requiring this is **antithetical to the project's goal** of fast real-time from few scattered
single-orientation acquisitions. So real free-breathing is inherently single-orientation — which is
the regime the learned prior targets, and where classical multi-orientation SVR structurally cannot
follow. **Multiplanar + free-breathing + GT does not exist in our catalog.**

## 6. Using classical SVR on *real-time* data: self-gate → then SVR

To run NiftyMIC/SVRTK on real free-breathing (e.g. Göttingen), the input must first be gated. The
correct decomposition is **self-gate → bin frames into phases → SVR each phase** — which is exactly
what fetal_cmr_4d does internally (it just uses SVRTK's recon after its `cardsync`). So you could:

1. Self-gate the real-time frames (fetal_cmr_4d's `cardsync`, **or** a simple custom image-based
   self-gater — temporal Fourier / PCA on the heart ROI; not hard).
2. Feed the per-phase binned stacks to NiftyMIC/SVRTK.

**Caveats (doc `30` §3 flagged this hybrid as possible-but-not-clearly-better):**
- **Through-plane doesn't improve** — Göttingen is single-orientation, so the recon still
  interpolates `dz` regardless of gating. Gating fixes the *cardiac* axis, not the *through-plane*
  axis.
- **No GT on Göttingen** → self-consistency / leave-one-slice-out eval only.
- Extracting fetal_cmr_4d's gating still needs its whole MATLAB+SVRTK build (or write your own
  gater); plus custom glue to export bins in NiftyMIC's expected NIfTI layout.

So it's a sound way to get a "classical SVR on real free-breathing" data point, but the
single-orientation and no-GT limits persist.

## 7. Timing / cost — where the hours go

`fetal_cmr_4d`'s ~6 h/case is **not** the self-gating. Self-gating is cheap signal processing
(per-slice temporal-frequency + cross-slice sync = seconds–minutes). The cost is dominated by:
1. **Iterative SVR recon run *per cardiac phase*** (register⇄reconstruct, nested at 3 levels, ×~12
   phases) — the compute bottleneck.
2. **Manual MITK heart/chest masking** per stack — *human* time.
3. Multiple orientations (more slices to register); k-t recon front-end if not bypassed.

**NiftyMIC/SVRTK are not inherently faster** — they are phase-agnostic and reconstruct **one** volume
per run. Our n=2 run was fast (~20 min) only because it did **one** phase; a full 4D cine would cost
~12× (~4 h single-orientation). Rough scale:
- NiftyMIC, 1 phase, 1 orientation: **~20 min** (measured, CPU).
- NiftyMIC, 4D (12 phases), 1 orientation: **~4 h**.
- fetal_cmr_4d, 4D, multi-orientation: **~6 h** (extra ≈ hierarchical registration + multiple
  orientations + manual masking).

No classical SVR is "fast" in absolute terms — the iterative solve runs to convergence per scan from
scratch, no weights carried between subjects. That minutes-to-hours cost vs VGGT's single forward
pass is itself a reportable contrast.

## 8. Recommended experiment plan (priority order)

1. **Fix the VGGT-side eval** — `num_slices=12` correction (doc `30` #5: the live checkpoint trained
   pre-multi-frame). The 32.7 dB number is suspect until this lands.
2. **Respiratory-corrupted same-phase test** (the real classical-SVR test, doc `30` §4.2). Apply the
   rigid SI/AP respiratory shift per real z-slice before export so NiftyMIC's registration has a
   *real* job; evaluate VGGT through the real val-time respiratory path (`seq_index`-deterministic).
   Score both vs `V_gt`. Report **both** `PSNR_anat` **and** `val_motion` (dynamic voxels) —
   `PSNR_anat` alone flatters both and hides the motion story (docs `13`/`22`).
3. **SVRTK-rigid** (`mirtk reconstruct`) as the 2nd classical point on the same protocol — its
   EM-robust recon should handle the corrupted slices better than NiftyMIC's Tikhonov default.
4. **NeSVoR** (Docker) — the ML-SVR point; per-scan, no cohort prior (contrast with VGGT's amortized
   prior). **Both clean-stack and respiratory-corrupted runs done** (n=2, doc `32`): beats NiftyMIC
   on the clean (non-meaningful) protocol; on the respiratory-corrupted protocol (the real test),
   mean PSNR_anat drops 31.14→22.99 dB (-8.15 dB) — a real, non-trivial degradation. NiftyMIC/SVRTK
   still need the corrupted re-run (NiftyMIC's `run_niftymic.sh` needs its stale bind-mount fixed
   first). VGGT-side comparison on this protocol deferred (checkpoint/sampling-regime question,
   doc `32`).
5. **Multi-orientation NiftyMIC** on gated CMRxRecon SAX+LAX+LVOT (doc `30` finding 4) — the
   upper-bound / anti-strawman point. IFFT+SOS the raw fully-sampled `cine_{lax,lvot}.mat`, use real
   spacing from `cine_{lax,lvot}_info.csv`, align to the SAX stack, feed 3 orientations.
6. **More subjects** — n=2 is proof-of-concept, not a result.
7. **fetal_cmr_4d on Göttingen** — lowest priority (cost + no GT); the only *real* (not simulated)
   free-breathing data point, worth an eventual self-consistency run, not urgent.

Everything else (k-space DL, 5D free-running, XD-GRASP, STACOM2024 for the separate EF lane) stays
cite-not-run per `_html/24`.

## 9. Contribution framing (honest scoping)

- **The problem is genuinely ill-posed and the learned-prior solution is principled.**
  Single-orientation through-plane cannot be recovered from data alone; only a prior can. Trading
  *acquisition redundancy* for a *cohort prior* is clinically valuable (no stack-of-stacks, no gating
  hardware, one fast pass). The specific combination — gating-free + respiratory-corrupted +
  phase-scattered + single-orientation + intensity — is unpublished (`_html/24`).
- **Scope it to what is recoverable.** The through-plane is prior-driven, so the model recovers
  **EF/timing and observed-phase content** (docs `24`/`25`/`22`), while **per-voxel through-plane
  motion PSNR is acquisition-coverage-limited** (docs `19`/`20`/`21`) — do not claim "recovers
  arbitrary through-plane motion." Lead with EF/functional recovery + free-breathing correction, not
  raw motion PSNR.
- **Real-time transfer is still simulated** (scattered sampling + in-plane aug + rigid respiratory
  sim; not bSSFP transient / single-shot artifacts). "Works on real free-breathing" is aspirational
  until shown on OCMR/Göttingen/MIITT — which have no GT, making quantitative proof hard (hence
  self-consistency eval).

"Solid, publishable contribution if scoped precisely" — not "huge" on the idea alone; the value is in
execution + honest scoping.

## References

- `_html/24_svr_baselines.html` — the full baseline landscape/taxonomy/RUN-CITE list.
- `_html/25_niftymic_baseline_comparison.html` — n=2 NiftyMIC visual report (VGGT numbers stale per
  doc `30` #5).
- `docs/29_niftymic_baseline_first_run.md` — the NiftyMIC run, bugs, results.
- `docs/30_baseline_testing_strategy_and_corrections.md` — tool-capability corrections, false starts,
  fair-comparison protocol, the `num_slices` bug.
- `docs/22`, `docs/24`, `docs/25` — reference-conditioning / flat-EF / EF-vs-motion-PSNR (what's
  recoverable).
- `docs/19`, `docs/20`, `docs/21` — the acquisition-coverage motion-PSNR ceiling.
- `docs/06` (OCMR), `docs/16` (Göttingen), `docs/23` (MIITT + `eval/` adapters) — the SAX-only OOD
  sources.
- `baselines/niftymic/` — reusable scripts (`export_stack.py`, `run_niftymic.sh`, `score.py`,
  `eval_vggt_same_subjects.py`).
