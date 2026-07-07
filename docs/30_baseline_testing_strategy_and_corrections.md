# 30 — Free-breathing SVR baseline testing: tool capabilities, corrections, and the fair-comparison protocol

> **TL;DR & takeaway**
>
> Extended session (2026-07-02, same day as doc 29) working out **how to fairly test
> classical SVR baselines against VGGT-MRI**, with several real corrections along the
> way — this doc exists so a fresh agent doesn't re-litigate the same mistakes.
>
> **Confirmed facts (verified from code/docs, not assumed):**
> 1. **Neither NiftyMIC nor SVRTK self-gates, in any mode.** `mirtk reconstruct`/
>    `reconstructFFD` have zero cardiac-phase concept. SVRTK's `reconstructCardiacVelocity`
>    needs pre-computed `-cardphase`/`-rrinterval` labels as *input* — it bins+reconstructs
>    given phase info, it doesn't estimate phase. **Self-gating code exists in exactly one
>    place in this whole toolchain: `fetal_cmr_4d`'s MATLAB layer** (`cardsync_intraslice/
>    interslice`).
> 2. **`reconstructCardiacVelocity` is velocity/4D-flow-specific — not usable for our
>    plain magnitude cine at all.** It needs phase-contrast velocity-encoded input +
>    slice transforms from a *prior* magnitude reconstruction. There is no native SVRTK
>    command for magnitude-cine → 4D; that's `fetal_cmr_4d`'s own orchestration (which
>    calls SVRTK's generic commands underneath).
> 3. **Our respiratory-motion sim is a PURE RIGID translation**, confirmed by reading
>    `training/data/respiratory.py`'s own docstring ("SI/AP rigid translation... no
>    rotation, no scaling") — despite being implemented via a "deform-then-reslice"
>    `grid_sample` mechanism, the displacement is spatially uniform (same vector
>    everywhere), i.e. mathematically rigid, not a local/non-rigid warp. **Correction:**
>    earlier in this session I incorrectly called it "genuine deformation" and
>    recommended testing against SVRTK's deformable `reconstructFFD` mode — wrong.
>    **Rigid** registration (NiftyMIC / SVRTK's `reconstruct`) is the mechanistically
>    correct match for this corruption.
> 4. **CMRxRecon's raw `ChallengeData` (not the `Cine_combined` SAX-only set our
>    pipeline uses) has real multi-orientation cine per subject**: `cine_sax.mat`,
>    `cine_lax.mat`, `cine_lvot.mat`, all sharing the **same T=12 real ECG-gated
>    phases** (confirmed via `h5py` inspection of the raw k-space: `cine_lax.mat`
>    → `kspace_full` shape `(12, 3, 10, 168, 448)` = T×Z×coils×readout×PE, fully
>    sampled, needs only a standard IFFT+SOS recon, no undersampling to solve). This
>    is a **much cheaper and more faithful way to give NiftyMIC genuine multi-orientation
>    redundancy** than anything Göttingen-related — same real gated phase, no self-gating
>    needed, ground truth intact. **Not yet built** (see Next Steps).
> 5. **A real bug found and fixed**: the VGGT-MRI same-subject eval (`baselines/niftymic/
>    eval_vggt_same_subjects.py`) originally used `num_slices=20` (the *current*
>    multi-frame config, `docs/28`). But the checkpoint being evaluated
>    (`217721337_mri_volume_reference_dynamic_axial_Cine_combined`) **started training
>    2026-06-24, six days before the multi-frame commit (`9aeb760`, 2026-06-30)
>    landed** — so it was actually trained under the *prior* commit's sampling
>    (`845e11f`, carrying `1856a82`'s reference-slot config): `num_slices=12`, **z
>    sampled WITHOUT replacement (one frame per z-plane, no repeats)** — genuinely the
>    project's "single-frame-per-slice" regime, not multi-frame. The eval was silently
>    testing the model **outside its trained input distribution**. Fixed to
>    `num_slices=12`; corrected number pending re-run (see docs/29's live addendum or
>    check `baselines/niftymic/vggt_volumes/` timestamps).
>
> **Lesson for future agents: always check which commit a *live/in-progress* training
> run actually started from before configuring an eval — don't assume the current
> working-tree config matches an old or still-running job.** `git log -1
> --format="%H %ad" -- <config file>` cross-referenced against the run's `log.txt`
> first-line timestamp is the check.

---

## 1. What's already run (see `docs/29` for full detail)

- **NiftyMIC**, real end-to-end run, 2 val subjects (`Train_P053`, `Val_P055`), clean
  real per-phase (t=0/ED) stack (12 slices, all real z, single SAX orientation, NOT our
  synthetic scattering — the fair "clean input" protocol per `_html/24_svr_baselines.html`
  §4). Three bugs found+fixed (Singularity bind-mount path, unconditional SimpleITK
  API-drift crash, missing intensity calibration). Result: **corr 0.886/0.901 with true
  anatomy, PSNR_anat 23.6/25.3 dB** (raw uncalibrated PSNR was a meaningless ~1.7 dB —
  scale/offset artifact, not structural failure). Runtime 18–25 min/subject, CPU-only,
  LSMR hit its 10-iteration cap without converging both times.
- **VGGT-MRI**, same 2 subjects/phase, same `PSNR_anat` metric — **but using the WRONG
  `num_slices=20`** (bug #5 above, being corrected as this doc is written). The
  originally-reported 32.70 dB mean is **suspect** until the `num_slices=12` re-run
  confirms/revises it. Do not cite the 32.70 dB number without checking whether the
  correction has landed (see `baselines/niftymic/eval_vggt_same_subjects.py`'s current
  `num_slices` value and re-run timestamp).
- Full visual report: `_html/25_niftymic_baseline_comparison.html` (built before the
  `num_slices` bug was found — **its VGGT-MRI numbers/figures are stale** and need
  regenerating once the corrected eval + a proper apples-to-apples visual are ready).
- **SVRTK: not yet run at all.** No Singularity image pulled, no data exported for it.

## 2. Tool capability matrix (verified, not assumed)

| | NiftyMIC | SVRTK | `fetal_cmr_4d` |
|---|---|---|---|
| Self-gating | No | No (even cardiac mode needs labels supplied) | **Yes** — `cardsync_intraslice/interslice` |
| Cardiac-phase awareness | None | `reconstructCardiacVelocity` — but **velocity/4D-flow only**, unusable for magnitude cine | Yes, full pipeline (uses SVRTK's generic commands as its own reconstruction back-end) |
| Registration model | Rigid only (what we ran; `RegAladin`) | Rigid (`reconstruct`) **and** deformable/FFD (`reconstructFFD`) | Inherits SVRTK's |
| Regularizer | Tikhonov + LSMR (confirmed from our run log) | EM-based robust statistics (Kuklišová-Murgašová 2012 lineage) | Inherits SVRTK's |
| Effort to run on our data | Done (this session) | Not started — new Singularity pull, likely new bugs to find | Highest in the whole catalog — MATLAB + from-source SVRTK build + manual MITK masks, ~6h/case (per `_html/24` §2) |
| Right input regime we have | CMRxRecon (already gated) | CMRxRecon (already gated) | **Göttingen** (real ungated stream) — NOT CMRxRecon (self-gating is moot on already-gated data) |
| Ground truth available? | Yes (CMRxRecon `V_gt`) | Yes (CMRxRecon `V_gt`) | Only on CMRxRecon (moot use case); **no GT on Göttingen** (self-consistency eval only) |

## 3. The corrected mental model of "what's a fair test"

Three false starts, each corrected in-session — recorded so they aren't repeated:

1. **False start: "give NiftyMIC more frames via Göttingen's 127-frame real-time streams."**
   Wrong, because NiftyMIC's registration model assumes ONE static object; Göttingen
   slices share **no cardiac or respiratory clock** (`"each slice is its own independent
   ~4s acquisition"` per `scratch/data/goettingen/README.md`), so raw multi-frame input
   there is phase-scattered — exactly what breaks a static-object registration model,
   the same failure mode as feeding it mixed cardiac phases anywhere else.
2. **False start: "self-gate Göttingen ourselves, or hybridize `fetal_cmr_4d`'s gating
   with NiftyMIC's reconstruction."** Technically possible but the wrong next move —
   `fetal_cmr_4d` is the costliest item in the whole catalog, Göttingen has no ground
   truth to score against, and a NiftyMIC/fetal_cmr_4d hybrid needs custom glue code
   nobody has published (higher risk than just using `fetal_cmr_4d`'s own native
   reconstruction).
3. **False start: "feed NiftyMIC/SVRTK our own respiratory-corrupted, phase-scattered
   VGGT-style input to test the real hard task."** Wrong for the SAME reason as #1 —
   VGGT's native S-slot sampling has most slots at *different* real cardiac phases, and
   classical SVR tools have zero phase concept, so this isn't a meaningful test, it just
   breaks them.

**The corrected, current plan** (not yet executed): keep every classical-SVR input
slice at the **same known real phase** (t=0), respiratory-corrupt each slice
independently (rigid SI/AP shift, matching what the trainer actually applies to VGGT's
input) — this is temporally coherent (satisfies the static-object assumption) while
still containing the exact kind of real spatial misalignment SVR's registration is
built to correct. **Feed VGGT-MRI its own native sampling** (whatever the actual
checkpoint was trained with — see bug #5, currently `num_slices=12`, one frame per
z-plane, slot semantics per `reference_slot=True`) with the same respiratory corruption
applied. Score both against the same `V_gt`. This is NOT identical input for both
methods — it's each method's own natural acquisition contract, under a comparable
corruption model, aimed at the same real target. That asymmetry is intentional: it's
testing whether VGGT's much cheaper acquisition (mostly off-target-phase content + one
real reference) matches what classical SVR gets from a full dedicated same-phase stack.

**Open prediction (recorded before running, so we can check calibration after):**
Classical SVR should do *relatively* better on this test than on the clean-stack test
(it's being given a real, well-posed problem to solve, unlike before), but the
respiratory sim's SI-dominant displacement lands on the through-plane axis — the known
weak point of single-orientation SVR — so it's not a trivial win for NiftyMIC/SVRTK
either. VGGT has a concrete (not hand-wavy) edge: it was directly trained against this
exact corruption model (`docs/05`), not a novel problem for it. Expect VGGT to still
win, by a smaller margin than the clean-stack test's 8.3 dB gap. **Unconfirmed — this
is a prediction, not a result.**

## 4. Next steps, in rough priority order

1. **Finish the `num_slices=12` correction** — confirm the corrected VGGT-MRI number,
   update `docs/29` and `_html/25_niftymic_baseline_comparison.html` (currently stale).
2. **Build the respiratory-corrupted same-phase test** (§3's corrected plan) —
   extend `baselines/niftymic/export_stack.py` to apply `training/data/respiratory.py`'s
   rigid SI/AP shift per real z-slice before export; run NiftyMIC on it; evaluate
   VGGT-MRI through the real val-time respiratory path (`sample_resp_disp(...,
   train=False, seq_index=...)`) instead of the respiration-free construction used so
   far. Score both against `V_gt`.
3. **Get SVRTK running** — new Singularity pull, expect new bugs (same class as
   NiftyMIC's — dependency drift, bind-mount issues). Test both `reconstruct` (rigid,
   matches the respiratory corruption) as the primary comparator.
4. **Build the SAX+LAX+LVOT multi-orientation NiftyMIC test** (§ finding 4) — IFFT+SOS
   reconstruct `cine_lax.mat`/`cine_lvot.mat` from raw k-space (`FullSample/`, fully
   sampled, no undersampling to solve), get real physical spacing from
   `cine_{lax,lvot}_info.csv` (NIfTI headers currently have placeholder 1.0mm), align
   geometrically with the existing SAX stack, feed NiftyMIC 3 real orientations instead
   of 1. This tests NiftyMIC's actual designed redundancy mechanism (cross-orientation
   through-plane recovery), cheaply, with ground truth intact — likely higher-value than
   the Göttingen route for the same reason self-gating isn't needed here.
5. **`fetal_cmr_4d` on Göttingen** — lowest priority given cost (~6h/case, MATLAB+SVRTK
   build) and no ground truth (self-consistency/leave-one-out eval only). Worth doing
   eventually for a real (not simulated) free-breathing data point, not urgent.

## References

- `docs/29_niftymic_baseline_first_run.md` — the actual NiftyMIC run, bugs, results.
- `_html/24_svr_baselines.html` — the original baseline landscape/taxonomy/RUN-CITE list.
- `_html/25_niftymic_baseline_comparison.html` — visual report (VGGT numbers stale, see §1).
- `training/data/respiratory.py` — the rigid respiratory sim, verified §item 3.
- `baselines/niftymic/` — all reusable scripts (`export_stack.py`, `run_niftymic.sh`,
  `score.py`, `eval_vggt_same_subjects.py`).
- `scratch/data/goettingen/README.md` — Göttingen data facts (independent per-slice
  acquisition, no shared clock, no ground truth).
- `scratch/data/CMRxRecon2024/ChallengeData/Cine/TrainingSet/FullSample/` — raw
  multi-orientation k-space (§ finding 4), not yet used.
