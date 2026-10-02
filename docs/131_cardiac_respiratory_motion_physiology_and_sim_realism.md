# 131 — Cardiac vs respiratory motion: independence, how the heart moves with breathing, and how realistic our sim is

> **TL;DR & takeaway**
> (2026-10-01, literature + code-read session; **nothing run, no code changed**.)
> 1. **Cardiac and respiratory motion are weakly coupled, not independent.** The field (XD-GRASP, 5D free-running, CiNeVol) treats them as separable components, and that is defensible: coupling is a few % in healthy subjects (preload / ventricular interdependence / RSA), larger in patients. **Our sim has zero coupling by construction** (a geometric shift applied to a gated, single-breath-state cine).
> 2. **Best-supported model of the heart's breathing motion = per-slice 6-DOF rigid transform**, SI-dominant (~5 mm tidal, SI 2–8 mm; AP ~1–3.5 mm; LR ≲3 mm; rotation ~1–3.5°), with **3–4 mm (up to 7 mm) local LV/RA deformation left over** and subject-dependent hysteresis. 3D translation alone was sufficient in only 2/10 patients; the heart/diaphragm slope varies 0.17–0.93 across subjects (so "0.6" is not reliable per subject).
> 3. **Our sim is larger than tidal.** `default.yaml`: SI amplitude 18.8 ± 7.35 mm (11–26 mm, "tidal → deep"), tilt θ~U(0°,45°) of the **SI+AP shift vector** (not a heart rotation). At 45° AP = SI, vs literature AP ≈ 0.3–0.5·SI. So it is a stress-test regime, not typical tidal breathing, with no rotation / hysteresis / deformation.
> 4. **If we add a two-component head:** claim only a *global rigid respiratory* component (supervisable from the sim's known per-slice shifts) + a dense residual (cardiac + non-rigid breathing). The split is enforced by supervision, not architecture. A *composed* form `R_resp ∘ C_card` is more principled than additive; additive is fine to first order. Both **untested**.
> 5. **MONAI Physio is NOT usable** for this (CT-only mesh surrogate, no explicit heart-breathing model, no public heart weights, no MRI training). Only its vertex→dense-field warp is reusable.
>
> Companion: `docs/132_isotropic_recon_lax_multiview_and_highres_gt_survey.md`.

**Evidence marks used below.** `READ` = a subagent opened the source (often the abstract only — publisher full texts were frequently 403) ; `RECALLED` = background knowledge, not verified this session ; `INFER` = our inference. All literature below was gathered by subagents; numbers are second-hand until someone re-opens the papers before citing.

## 1. Are cardiac and respiratory motion independent?

**Bottom line:** no, but separable enough that the field models them as separate components. I found **no paper that tests additive vs composed deformation** or that quantifies whether cardiac phase biases respiratory-shift estimates.

### 1.1 What separation-based methods assume
- Feng et al., **XD-GRASP**, MRM 2016 — cardiac and respiratory states become separate dimensions of one multi-dimensional image, each with its own sparsity regulariser. Separability is built into the binning. (READ, abstract.)
- Di Sopra et al., **5D free-running whole-heart**, MRM 2019 — cardiac and respiratory self-gating signals extracted from raw data and binned into a cardiac × respiratory grid. (READ, abstract.)
- Vogt et al., **CiNeVol**, Med Biol Eng Comput 2026 — ECG + respiration-belt states as inputs; "decoupled estimates of cardiac and respiratory motion". Full text not opened, so whether the two fields are **summed or composed is unknown**. (READ, abstract.)
- Fayad/Odille et al., dual-gated PET/MR (EJNMMI Phys 2015) — respiratory displacement is interpolated as a function of cardiac phase (a weak, conditional coupling). (READ, PMC.)
- Xu et al., JCMR 2015 — respiratory model for MRI-guided intracardiac procedures: rigid translation+rotation fit linearly to respiratory data, cardiac-gated; residual 3.80 → 2.24 mm. (READ, PMC.)

### 1.2 Physiological coupling (what breaks independence)
Mechanisms (standard physiology; magnitudes where read):
1. **Intrathoracic pressure → preload.** Inspiration raises venous return to the right heart: Claessen et al., Am J Physiol Heart 2014 (real-time CMR) — expiration→inspiration **RV EDV +18 ± 8 %, RV SV +21 ± 10 %**; LV ESV 9 ± 7 % higher at expiration; LV EDV/SV changes not significant. (READ.)
2. **Ventricular interdependence.** Shared septum + pericardium: Chitiboi et al., FIMH 2017 — septum position and LV end-diastolic diameter vary with the respiratory cycle (end-systolic diameter ~constant); functional response 5.8 ± 3.6 % normals vs 2.5 ± 3.4 % patients. (READ, PMC.)
3. **Stroke-volume mismatch** — instantaneous LV and RV SV differ and converge over the breath cycle (Claessen). The pulmonary-transit lag explanation is RECALLED.
4. **Respiratory sinus arrhythmia / cardioventilatory coupling** — heart rate rises on inspiration (vagal modulation of the sinus node). Elstad et al., Am J Physiol Heart 2018 (READ, abstract). Magnitude (tens of ms of R–R modulation) is RECALLED. ⇒ **R–R-fraction phase labels are slightly respiratory-dependent.**
5. **Mechanical** (diaphragm/lung expansion moves and tilts the heart) — this is the bulk displacement our sim models; it is a position effect, not a coupling of cardiac *state*.

Healthy-subject null result: Liu et al., Sci Rep 2019 (5D cine, 15 volunteers) — no significant difference in ESV/EDV/SV/EF between end-expiration and end-inspiration. (READ, abstract.) ⇒ coupling is small in healthy cohorts (most of our training pool is healthy CMRx) and larger in patients (ACDC / arrhythmia eval).

### 1.3 Composition order
No source tests it. INFER: respiratory = near-rigid body shift, cardiac = non-rigid contraction ⇒ the principled form is `x → R_resp(C_card(x))` (contract in the heart frame, then apply the body shift). Additive `Δ_card + Δ_resp` is accurate to first order for small near-rigid shifts and degrades with large amplitude or respiration-induced shape change. **Cheap ablation if we ever build the two-head model.**

### 1.4 Does breathing affect the ECG? (background knowledge, not searched)
ECG = skin-electrode voltage differences caused by the heart's electrical wave. Breathing (a) modulates **timing** (RSA: vagal control of the sinus node ⇒ R–R varies) and (b) changes the **measurement** (heart moves/tilts relative to the electrodes; air-filled lungs change conduction ⇒ baseline wander and QRS amplitude/axis modulation) without changing the heart's electrical activity. Relevant to gated baselines: an ECG-derived phase is slightly breathing-contaminated; our model uses no ECG.

## 2. How the heart moves with breathing (physiological model)

**Best-supported model for per-slice correction: a per-slice 6-DOF rigid transform** (translation dominated by SI, small AP/LR, small rotation).

| Claim | Source | Mark |
|---|---|---|
| Tidal-breathing heart motion is dominated by SI, linear in diaphragm SI motion, ≈ a global translation | Wang, Riederer, Ehman, MRM 1995 | READ (abstract) |
| Heart/diaphragm factor 0.6 is Wang's, as cited; AP/LR multipliers (0.4/0.3) are Dasari's own simulation assumptions | Dasari et al., PLOS ONE 2013 | READ (as cited there) |
| Coronary-to-diaphragm slope ≈0.6 on average but **0.17–0.93, r² 0.04–0.87** (12 volunteers) ⇒ patient-specific factors needed | Danias et al., AJR 1999 | READ (abstract) |
| 10 patients, free-breathing biplane angiograms: caudal translation on inspiration **4.9 ± 1.9 mm** (2.4–8.0); cranio-dorsal rotation 1.5 ± 0.9°; anterior translation 1.3 ± 1.8 mm and caudo-dextral rotation 1.2 ± 1.3° in 8/10; **3D-translation-only sufficient in only 2/10** | Shechter et al., IEEE TMI 2004 | READ (abstract) |
| Rigid motion mainly craniocaudal; **deformation after rigid alignment largest at RA free wall and LV, typically 3–4 mm, up to 7 mm**; PCA motion/deformation model built | McLeish et al., IEEE TMI 2002 | READ (abstract) |
| LV electro-anatomical mapping (catheter, different population): 8.8 ± 2.3 mm 3D = FH 7.0 / AP 3.6 / RL 3.3 mm; intra-patient variability 3.5 ± 1.8 mm | Dasari et al., PLOS ONE 2013 | READ |
| Hysteresis strongly subject-dependent | Nehrke et al., Radiology 2001 | READ (abstract) |
| 3D translation and affine both beat SI-only translation; calibrated 3D affine reduces model RMS 1.51 → 0.74 mm (coronaries) | Manke et al., TMI/JMRI 2002, MRM 2003 | READ (abstracts) |
| Subject-specific **elliptical** (hysteresis-aware) affine beats linear affine; 9 volunteers, 25 s prescan | Burger et al., MRM 2013 | READ (abstract) |
| Slice-wise in-plane rigid correction of SAX stacks leaves ~**2 mm** MAD (3.1 → 1.9 mm with LAX target; 3.1 → 2.6 mm without) | Tarroni et al., arXiv 1810.02201 | READ (full text) |
| Navigator drift 1.7–3.0 mm over a scan; diaphragm excursion ~14.7 mm | PMC3679310 | READ |
| Respiration modulates mid-SAX LV EDD 5.8 ± 3.6 % (normals) vs 2.5 ± 3.4 % (patients) | PMC6258012 | READ |

Typical numbers (tidal): SI ~5 mm (2–8), AP ~1–3.5 mm (sign can differ per subject), LR ≲3 mm, rotation ~1–1.5° (up to 3.5°). **Not found:** a measured hysteresis magnitude in mm, deep-breath tail behaviour, age/disease effects (RECALLED/unverified).

**SVR works' motion models:** van Amerom/Lloyd/Deprez MRM 2018–19 (fetal cine) — per-frame in-plane rigid, global then a second registration for residual heart twisting, composed `A = U·V`; Roberts et al., Nat Commun 2020 — rigid body per frame with outlier rejection (READ). Tian MRM 2025 and CiNeVol: transform model **not verified** (abstract/snippet only).

### 2.1 Simulators
- **XCAT** (Segars, Med Phys 2010; READ, full text): respiratory motion driven by two time curves (diaphragm height, AP chest expansion); heart translations/rotations user-set; defaults from 20 respiratory-gated 4D CTs; **no hysteresis described**; default heart amplitudes not stated in the text.
- Older NCAT-based SPECT studies moved the heart with the same maximum as the diaphragm (2–4 cm), which is **not** realistic (measured ≈ 0.6×).

## 3. Our simulator vs the literature

Source: `training/data/respiratory.py`, `training/config/default.yaml` (read this session).
- Waveform: Lujan `d(r) = A·sin(πr)^(2n)`, n=3, SI displacement; per-slot SI + smaller AP; one tilt θ per subject.
- `amplitude_mm: 18.8`, `amplitude_jitter: 7.35` ⇒ **11.45–26.15 mm** (comment: "tidal → deep"); `tilt_min_deg: 0`, `tilt_max_deg: 45` = direction of the SI+AP shift vector off the D axis; `amplitude_breath_jitter: 0`.
- Applied per **input slice only**; targets stay at the unshifted end-expiration reference.

| | Literature (tidal) | Our sim |
|---|---|---|
| SI amplitude | ~5 mm (2–8) | 11–26 mm (mean 18.8) |
| AP / SI | ~0.3–0.5, random sign | up to tan 45° = 1.0 |
| Rotation of the heart | 1–3.5° | none (tilt = shift direction, not rotation) |
| Hysteresis / temporal correlation | present, subject-dependent | none (per-slice iid) |
| Local deformation | 3–4 mm (≤7) | none (pure rigid translation, verified in docs/30) |
| Cardiac–respiratory coupling | few % healthy, more in patients | none |

**Takeaways.** (i) Amplitude is ~2–4× tidal: it covers "deep" breathing and is a deliberate-looking stress test (we did not check the docs for the rationale). (ii) Per-slice-iid amplitude is realistic for stacks acquired in separate breath-holds, less so for continuous free-breathing. (iii) Any claim of "disentangling" must be scoped to the **rigid** respiratory component, since the sim cannot test the rest. (iv) Earlier project findings (docs/20/21/34, memory notes: breathing is not the bottleneck, oracle transport ≤ ~1.4 dB) suggest a **disentangled head is more likely an interpretability / clean-respiratory-output win than a PSNR win** — not re-measured for a two-head design.

## 4. Proposed (not implemented) two-component design
- `Δ_resp`: low-dimensional **per-slot** output (3 translations + optionally 3 rotations, pooled from the slot's token), supervised **directly** with the sim's known per-slice shifts. `Δ_card`: the existing dense DPT head. Splat uses the sum (or the composed form).
- Identifiability comes from the Δ_resp supervision, not structure (a dense field can also represent a rigid shift).
- Consistency test: same stack/frames, resimulate breathing at a different amplitude ⇒ `Δ_card` should not change.
- Do **not** force `Δ_card` to zero mean (long-axis shortening is legitimate SI motion) — INFER, untested.
- Ablation vs the single-Δ model on respiratory EPE/slope and motion PSNR with hole fraction as a guard (docs/38 ship rule).

## 5. MONAI Physio (Project-MONAI/monai-physio, release 2026.09.1) — verdict: not usable
Code cloned and read by a subagent (nothing run). Verdict **no for direct use**:
- Pipeline: segmentation → PCA statistical shape model fit → learned **PhysicsNeMo MeshGraphNet/MLP surrogate** (per-vertex displacement from `[mean-shape xyz, PCA coeffs, stage]`, stage = R–R fraction for heart / 4D-CT breathing phase for lung). Output VTK meshes / animated USD / optionally warped images.
- **Separate lung and heart networks**, independent time variables, composed by hand in a tutorial. The heart moves with breath only because the lung's smoothed, normal-restricted surface field is composed onto the heart surface (`respiratory_sigma_mm=15`) — a diffusion artifact, **not validated against measured heart displacement**. No rigid/translational heart model.
- Training data: lung = TCIA 4D-Lung (CT, ~83 cases per tutorial text); heart = Duke-Heart-4DLabelmaps (**not public**, email the author). Only a lung checkpoint ships; no heart weights found. **CT-only; no MRI motion model.** Beta, one main developer, "not validated for clinical use".
- **Reusable piece:** `WorkflowInferMovement.create_deformation_field` / `smooth_deformation_field_transform` / `transform_image` (`workflow_infer_movement.py:383` and nearby) — vertex displacements → dense ITK field → warp, with an optional sliding-boundary restriction. Usable if we ever want a **non-rigid** breathing sim fed with our own displacements.

## 6. Open questions / next steps (none started)
1. If a respiratory head is built: 6-DOF vs SI+AP only; additive vs composed; supervise with sim shifts.
2. Should the sim add rotation, subject-specific AP sign/ratio, temporal correlation / hysteresis, a tidal-amplitude arm? (Stress-test amplitude vs realism trade-off.)
3. Re-open key papers (Shechter 2004, McLeish 2002, Claessen 2014, Chitiboi 2017, Danias 1999) before citing numbers in the paper — most were abstract-only reads.
