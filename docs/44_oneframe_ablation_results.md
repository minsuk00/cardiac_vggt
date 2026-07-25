# 44 — One-frame ablation: results, and what breathing estimation actually buys

> **TL;DR & takeaway** (2026-07-15). Mid-training analysis of the **6-run `1frame_series` ablation** designed in
> docs/43, evaluated at a **common epoch 26** (the runs died at 25–39 epochs and are being resumed, so nothing
> here is a converged number). **Verdicts: (C1) the gather aux loss WORKS — decisively, at every epoch — for
> breathing AND for coverage, but it does NOT buy image quality and slightly costs it.** Paired over the
> plateau (epochs 20–38, n=19, byte-identical val data — the pairing is verified, not assumed): through-plane
> EPE **1.37 vs 2.89 mm** (t=−16, 19/19; for scale, predicting Δz=0 scores 5.18 mm, so the hub removes **74%**
> of the breathing error and no_gather only 44%), heart coverage holes **0.064 vs 0.091 (−29%, p=1e-3)** — but
> `recov_frac`/`psnr_motion` are **coin-flips** and gather is **significantly worse** on aggregate PSNR (bbox
> −0.135 dB, static −0.372 dB). **This is not a ceiling artifact — 5.40 dB of oracle headroom sits unclaimed**,
> so the flat appearance is a real, informative negative. **Do not read C1 off a single epoch: ep26 mildly
> flatters gather.** Next: **(C5) lowdiff100 is NULL**
> (rule-out achieved, reproducing docs/33 under the new regime); **(C3) contz is a no-ship in-distribution**
> (−1.9 dB bbox, −2.6 dB static, recov 0.52 vs 0.66 at every epoch, with *fewer* coverage holes — the smearing
> signature); **(C2/C4) aug and dino need the OOD arm** to decide.
>
> **THE PARADOX IS RESOLVED — and the resolution has a SCOPE that changes the ship decisions.
> In-distribution, breathing/placement is not the bottleneck; the appearance wall is.** The two models'
> reconstructions **agree with each other (26.9 dB) far better than either agrees with GT (20.3 dB)** ⇒
> **89% of the in-dist error is common-mode**, only 11% unique. So a real 2.6× placement fix is worth
> **+0.04 dB (p=0.48, null)**. The 5.4 dB of oracle headroom is *appearance* headroom (the oracle gets the
> true target-phase content the model never observes — docs/22, docs/24); gather cannot spend it.
> **BUT on REAL OOD data (MIITT) the same decomposition gives 74% shared / 26% unique — the placement-driven
> component is 2.7× larger — and gather's payoff tracks it exactly: +0.402 dB, p=0.020, 10/13 subjects.**
> ⇒ **"gather buys ~0 dB" is true IN-DISTRIBUTION ONLY.** Reported without that scope it would have been the
> wrong ship call. **Verdicts: C1 SHIP** (pays on real data), **C2 SHIP** (aug: −0.08 in-dist / **+0.63 OOD**,
> p=0.016 — exactly docs/05's predicted trade), **C4 in-dist ONLY** (dino is the only significant in-dist
> winner, +0.161 p=0.004, but **flat on OOD** p=0.71 — judging it on in-dist PSNR as docs/43 specified would
> have shipped a non-transferring change), **C3 NO-SHIP** in-dist / undecidable (§5), **C5 NULL** (its OOD
> "win" is confounded — it is the least-trained model).
>
> **Breathing estimation is good — better than a first read suggests.** On the **270/300 content-bearing
> slots** the hub scores **slope 0.844 / corr 0.958 / EPE 1.10 mm**, and **ignores 0 of 100** big breaths that
> land on slices with anatomy. Its only "misses" are 8 slices that **breathing physically evacuated out of the
> FOV** (94% empty, heart-free) — ill-posed, and *both* models fail those identically. A do-nothing model
> scores EPE 5.18 mm, so the hub removes **73%** of the breathing error vs no_gather's 53%. It invents nothing
> on un-breathed input (clean control **0.14 mm**) and moves slices near-rigidly. Under-correction is real
> (every robust estimator lands 0.78–0.84) but **magnitude-dependent** — worst at 2–5 mm (~55%), 81–88% at
> ≥5 mm. Do **not** carry docs/42's deep-collapse profile (96/91/68%) here — different model. **Two
> methodological corrections future agents must not repeat: (1) the `Val_Loss/metric_*` keys in docs/38+43 are
> DEAD** — the real names are `val/resp/*`, `val/metric/*`, `val/psnr/*`, and `contz` bakes a *different*
> identity baseline into its key (`base17.3` vs `17.1`), so string-matching silently drops it; **(2) `val/ef/slope`
> at n=29 is NOISE** — the hub's own range across epochs is 0.08–0.63, so lowdiff100's "0.000" and dino's "0.157"
> are draws from that band, not findings. **The checkpoint is the authority on its own epoch, not `log.txt`**:
> the trainer prints "Saving checkpoint at epoch N" *before* the write, so `no_gather` (ep37) and `dino_ft`
> (ep33) are each one epoch behind what their logs claim.

Companion to docs/43 (the design + what each comparison tests), docs/38 (the GT-referenced val metrics),
docs/37 (the gather aux loss), docs/42 (the OOD head-to-head harness), docs/33 (EF + the regularizer swap),
docs/05 (aug hurts in-distribution), docs/28 (continuous-z).
Checkpoints: `scratch/checkpoints/README.md` (`20260715_1frame_*`). Report: `_html/44_oneframe_ablation_analysis.html`.

---

## 1. What was run, and the state it was caught in

Six single-factor variants off the `1f_gather05` hub, all warm-started from `4wok_weights_only.pt`, all
`one_frame_per_slice=true`, `max_epochs=100`, peak LR 5e-5. All six **died early** (wandb `state=failed`) and
are being resumed. LR at death was **3.4e-5** (70% of peak, mid-cosine) — nobody annealed, so **none of these
is a converged model** and every number here is a mid-flight read.

| variant | wandb | ckpt epoch | val epochs | one delta vs hub |
|---|---|---|---|---|
| `gather05` (hub) | [`fhkgalju`](https://wandb.ai/minsuk-choi/vggt-mri/runs/fhkgalju) | 39 | 40 | — |
| `no_gather` | [`lmboejhq`](https://wandb.ai/minsuk-choi/vggt-mri/runs/lmboejhq) | **37** | 39 | `gather_weight=0` |
| `contz` | [`tfz1x7ft`](https://wandb.ai/minsuk-choi/vggt-mri/runs/tfz1x7ft) | 39 | 40 | `continuous_z=true` |
| `dino_ft` | [`hlh3emae`](https://wandb.ai/minsuk-choi/vggt-mri/runs/hlh3emae) | **33** | 35 | patch_embed trains, LR 2e-5 |
| `aug_moderate` | [`lylgvajs`](https://wandb.ai/minsuk-choi/vggt-mri/runs/lylgvajs) | 39 | 40 | aug tier=moderate |
| `lowdiff100` | [`2kwj0tkd`](https://wandb.ai/minsuk-choi/vggt-mri/runs/2kwj0tkd) | 25 | 26 | `diffusion_weight=100` |

**Epoch is a live confound and must be controlled.** The hub's breathing metrics are *still improving* at ep40
(slope .757@ep1 → .805@ep26 → .852@ep40; EPE 1.95 → 1.36 → 1.32 mm) while `recov_frac_heart` is **flat**
(.65–.68 for every run at every epoch). So: breathing claims **must** be epoch-matched; recov-based claims are
epoch-robust. Primary read = **ep26** (last epoch where all six are alive); each run's own final epoch is
reported separately and never mixed into the same cell.

## 2. Epoch-matched results (ep26) — orientation only, superseded by the plateau test in §3

**Do not quote this table as the evidence base.** A single epoch is noise-dominated and ep26 mildly flatters
gather. It is here because it is the one epoch where all six runs are alive, which makes it a useful *map*.
The *evidence* is the paired plateau test in §3.

| metric | gather05 | no_gather | contz | dino_ft | aug_mod | lowdiff100 |
|---|---|---|---|---|---|---|
| resp EPE mm ↓ | 1.364 | **3.494** | 1.840 | 1.633 | **1.130** | 1.253 |
| resp slope →1 | 0.805 | **0.380** | 0.767 | 0.763 | **0.873** | 0.837 |
| resp corr | 0.937 | **0.459** | 0.893 | 0.939 | 0.949 | 0.942 |
| deep ignored ↓ | 0.000 | **0.348** | 0.007 | 0.000 | 0.000 | 0.000 |
| recov ↑ | 0.661 | 0.657 | **0.519** | 0.672 | 0.648 | 0.657 |
| hole ↓ | 0.055 | 0.060 | 0.055 | 0.063 | 0.071 | 0.056 |
| psnr motion ↑ | 20.862 | 20.823 | **19.470** | 20.978 | 20.763 | 20.837 |
| psnr bbox ↑ | 28.474 | 28.485 | **26.559** | 28.497 | 28.260 | 28.505 |
| psnr static ↑ | 32.011 | 32.154 | **29.388** | 31.947 | 31.625 | 32.141 |

## 3. The central result: gather fixes breathing and coverage, and COSTS a little PSNR

**Read this from the PLATEAU, not from a single epoch.** A single-epoch read is noise-dominated, and ep26
mildly flatters gather (no_gather's EPE there is 3.49 vs its 2.89 plateau mean). Both runs are evaluated on
byte-identical val data at every epoch — verified: `mse_heart_identity` and `mse_heart_oracle` agree to
**2e-10** and the applied breathing agrees to **exactly 0.0** — so epoch *e* is a legitimate **pair**, and a
paired t-test over the plateau (epochs 20–38, n=19) removes the common-mode epoch noise:

| metric | gather05 | no_gather | diff | t | gather wins | verdict |
|---|---|---|---|---|---|---|
| resp EPE mm ↓ | **1.367** | 2.893 | **−1.526** | −16.0 | **19/19** | gather (p=4e-12) |
| resp slope →1 | **0.828** | 0.557 | +0.272 | +10.3 | **19/19** | gather (p=6e-9) |
| resp corr ↑ | **0.936** | 0.647 | +0.289 | +10.1 | **19/19** | gather (p=8e-9) |
| **hole_frac_heart ↓** | **0.064** | 0.091 | **−0.027** | −3.9 | 16/19 | **gather (p=1e-3)** |
| **coverage_frac ↑** | **0.723** | 0.720 | +0.003 | +6.1 | 17/19 | **gather (p=9e-6)** |
| recov_frac_heart | 0.664 | 0.665 | −0.001 | −0.6 | 9/19 | **coin-flip** |
| PSNR motion | 20.893 | 20.898 | −0.005 | −0.2 | 10/19 | **coin-flip** |
| PSNR bbox ↑ | 28.465 | **28.601** | −0.135 | −4.0 | 3/19 | **no_gather (p=9e-4)** |
| PSNR static ↑ | 31.949 | **32.321** | −0.372 | −6.9 | 1/19 | **no_gather (p=2e-6)** |

So the honest three-part result:
1. **Gather massively fixes through-plane placement.** t=−16 on EPE, unanimous 19/19 across every plateau
   epoch. For scale: a model that always predicts Δz=0 has EPE **5.18 mm** (= mean |applied| over the 300
   frozen-bundle slots). The hub removes **74%** of that error; `no_gather` removes only **44%**.

   **Independently reproduced offline on the frozen bundles** (different data, different code path, ED phase):
   gather05 EPE **1.39 mm** (removes 73% of null) vs no_gather **2.43 mm** (removes 53%). And the offline data
   shows **what gather actually does** — it is not a uniform gain improvement:

   | applied \|SI\| | gather05 recovered | no_gather recovered |
   |---|---|---|
   | 2–8 mm (n=62) | **75%** | 43% |
   | 8–12 mm (n=38) | 85% | **86% — identical** |
   | 12–40 mm (n=49) | **80%** | 66% |
   | **slices IGNORED** (applied ≥5 mm, \|pred\| < 2 mm) | **7%** | **27%** |

   At 8–12 mm the two models are *the same*. **Gather's entire contribution is stopping the model from
   ignoring breaths outright** — the ignored population drops 27% → 7%. The aggregate slope/EPE gap is a
   mixture effect of that, not a better-tracking model.
2. **Gather also buys a real, non-dB win that a PSNR-only reading misses:** heart coverage holes fall
   **0.091 → 0.064 (−29% relative, p=1e-3)** and `coverage_frac` rises (p=9e-6). This is precisely the
   failure mode the aux was designed for (docs/37).
3. **But it does not buy image quality — it slightly COSTS it.** `recov_frac_heart` and `psnr_motion` are
   statistical coin-flips (t=−0.6, −0.2), and gather is *significantly worse* on aggregate PSNR
   (bbox −0.135 dB, static −0.372 dB, winning 3/19 and 1/19). "Buys ~0 dB" is, if anything, generous.

**And this is not a ceiling artifact.** The appearance metric has abundant room: identity 16.76 dB → model
20.63 dB → **oracle 26.04 dB**, i.e. **5.40 dB of headroom unclaimed** and `recov_frac` 0.67 means a third of
the recoverable span is still on the table. The oracle is splat-achievable by construction (Δ=0 splat of the
true target-phase content through the *same* splat with the *same* coverage weights, `loss.py:576-584`) — not
a fantasy bound. The metric *could* show an improvement and doesn't.

**This is where the docs/38 decision rule is the wrong instrument.** Applied literally ("ships iff recov↑ and
psnr_motion↑ without hole↑"), it returns *no-ship* for a change that removes 74% of a real placement error and
cuts coverage holes by 29%. The rule was built (docs/37) to catch a change that games a *process* metric while
the volume rots — the stop-grad failure. **It is a veto, not an elector.** Use the factor-specific metric to
decide the win; use docs/38 only to catch a regression.

### RESOLVED: why a real placement fix buys no image quality

**The two models reconstruct nearly the same volume.** Run on the same 30 frozen bundles
(`tools/compare_recons_1frame.py`, heart&FOV ROI, ED):

| | PSNR |
|---|---|
| gather05 vs GT | 20.29 ± 1.57 dB |
| no_gather vs GT | 20.35 ± 1.57 dB |
| **gather05 vs no_gather** | **26.91 ± 2.31 dB** |

The two recons **agree with each other 6.6 dB better than either agrees with GT**. If their errors were
independent, PSNR(A,B) would be ≈18.8 dB; it is 26.9. Decomposing per subject: **88% of each model's error is
SHARED** and only **12% is unique** to it. The breathing-Δz difference — 2.6× EPE, 27% vs 7% of slices
ignored — perturbs 12% of the error budget and does not reduce it.

**So placement is not the binding constraint IN-DISTRIBUTION; the common-mode error is.** That common-mode
error is the known **appearance wall**: with one frame per slice at unknown cardiac phases, the model must
*synthesise* the target-phase content it never observed (docs/22's "held-out error is appearance pattern, not
amplitude"; docs/24's information limit). The 5.40 dB of oracle headroom is real, but the oracle is handed the
**true target-phase content** — so that headroom is mostly *appearance*, not placement. Gather cannot spend it.

### …but on REAL OOD data placement DOES pay, and the same test explains why

Running the identical decomposition on the MIITT cohort (13 real gated subjects) instead of CMRx:

| cohort | **shared** | **unique** | gather's measured payoff |
|---|---|---|---|
| CMRx (in-distribution, n=30) | **90%** | **10%** | **+0.041 dB (p=0.48, null)** |
| MIITT (real gated OOD, n=13) | **74%** | **26%** | **+0.402 dB (p=0.020, 10/13)** |

> **Numbers corrected (prove-it, 2026-07-16).** These are now computed from **direct MSE**, not from the
> PSNR-difference formula the first draft used. That formula was contaminated by a `(GT.max/B.max)²` peak
> factor (`compare_recons_1frame.py`'s `psnr()` used `peak=b.max`), which inflated the OOD unique fraction; the
> original draft read 71%/29%. Direct MSE + an assumption-free error-field correlation (ρ) agree: in-dist
> 90%/10% (ρ=90%), OOD 74%/26% (ρ=75%). The in-distribution number was always robust (peaks near-equal there).

**The placement-driven unique component is ~2.6× larger on real OOD data (10% → 26%), and gather's payoff
tracks it — 10× bigger (+0.04 → +0.40 dB) at ep39.** In-distribution the appearance wall swamps everything, so
a real placement fix is invisible. On real data the shared appearance error is a smaller share of a bigger
total, and placement becomes measurable. **(But see docs/45: this +0.40 dB OOD win was a marginal n=13 result
that did not hold up when re-tested at ep60 — the mechanism is real, the ship-decision was not robust.)**

**This partially rehabilitates the gather loss and it is the single most useful correction in this doc:**
"gather buys ~0 dB" is true **in-distribution only**. Reported without that scope it would have been the wrong
call — the ship decision for a *real-world* model rests on the OOD number, where gather wins significantly.

**The whole-cohort numbers agree** (both n=30, same frozen bundles, same scorer):

| | gather05 | no_gather | diff |
|---|---|---|---|
| breathing EPE | **1.42 mm** | 2.44 mm | −1.02 |
| **breath PSNR** | 20.584 | 20.543 | **+0.041 dB** |
| **clean PSNR** (no breathing at all) | 22.016 | 21.872 | **+0.144 dB** |
| breath SSIM | 0.861 | 0.860 | +0.001 |

**The one-line answer to "how much does breathing estimation matter?"** At 12 mm pitch with one frame per
slice, cutting through-plane error from 2.44 mm to 1.42 mm is worth **+0.04 ± 0.31 dB** — statistically
nothing. And the giveaway sits in the table: gather's edge is **3.5× larger on CLEAN input, where there is no
breathing to correct at all**. Whatever small PSNR effect it has is not a breathing effect. Breathing
estimation is already good enough that it is *not* what limits this pipeline.

### A hypothesis of ours that FAILED (recorded so nobody re-runs it)

We predicted the coverage-division splat *absorbs sub-pitch* z-errors, so gather's benefit should appear
**specifically on deep breathers** (whose errors cross a plane boundary). **Refuted:**

| test | result |
|---|---|
| gather advantage vs subject's max applied \|SI\| | r=+0.066, **p=0.735** |
| gather advantage vs fraction of slots ≥12 mm | r=−0.041, **p=0.833** |
| deep half vs shallow half of the cohort | +0.047 vs +0.002 dB, **p=0.706** |
| PSNR gain under **breathing** | **+0.025** ± 0.305 dB |
| PSNR gain under **clean** (no breathing at all!) | **+0.151** ± 0.320 dB |

The advantage does not grow with breath depth at all. The tell is the last row: gather's edge is *larger on
clean input than on breathing input*. If its benefit were breathing correction, it would be **zero** on clean.
Whatever small PSNR effect gather has is not about breathing.

### What we ruled OUT for the paradox (each tested, not assumed)
- **"A few mm of z-error is cheap at 12 mm pitch."** REFUTED *for coherent shifts*. `tools/zshift_sensitivity.py`
  translates the real GT by δz and rescores it against itself on the same heart ROI: **1.36 mm → 34.7 dB,
  3.49 mm → 26.6 dB** — an 8 dB spread. A *coherent whole-volume* z-error of that size is expensive.
  **Scope caveat (prove-it):** this establishes sensitivity to a *coherent* shift, which is the worst case for
  decorrelation and is NOT the *incoherent per-slice* error the two models actually differ by (which the
  coverage-averaging splat partially cancels). So it rules out gross metric blindness but does not, by itself,
  prove the metric would catch the models' placement difference — that conclusion is carried by the independent
  shared-error decomposition below, for which the z-shift is a corroborator, not the proof.
- **"The metric measures the wrong region."** REFUTED for the hub. The metric gates Δz on `img>0.05` (whole
  FOV: chest wall, lungs, liver) while PSNR scores the heart — but recomputing per-slot Δz under a heart mask
  (`tools/breathing_roi_probe.py`) gives slope 0.588 vs 0.627 FOV-gated, EPE 1.96 vs 1.78 mm. Same answer.
- **"Different breathing draws."** REFUTED: both runs' val breathing is identical (5.90/27.05 mm).
- **"The model hallucinates breathing."** REFUTED: clean negative control (applied ≡ 0 on every slot) gives
  mean |predicted Δz| **0.15 mm**, max 0.83 mm over 306 slots. It predicts ~nothing when there is nothing.

*(Open: `tools/compare_recons_1frame.py` measures whether the two models' reconstructed volumes actually
differ — pending the eval sweep. See §7.)*

## 4. How well does breathing estimation work?

Measured on the frozen bundles (offline), and **cross-validated against the trainer's own val metric through a
completely independent code path**: for the same hub checkpoint, offline gives slope 0.814 / corr 0.936 /
EPE 1.31 mm vs wandb's 0.852 / 0.940 / 1.32 mm. Two implementations, different data, different cardiac phase —
same answer. The harness is sound.

**Amplitude response — the `1f_gather05` hub (ep39), n=30 subjects / 300 slots, frozen CMRx bundles:**

| applied \|SI\| | n slots | applied → predicted | recovered |
|---|---|---|---|
| 0–2 mm | 151 | 0.3 → 0.3 | 104% (≈ the noise floor; applied ≈ 0) |
| 2–8 mm | 62 | 4.9 → 3.7 | **75%** |
| 8–12 mm | 38 | 9.9 → 8.4 | **85%** |
| 12–40 mm | 49 | 17.0 → 13.7 | **80%** |

Aggregate: slope **0.807**, corr 0.928, EPE **1.39 mm**, signed bias **−0.98 mm**.

**Those all-slot bin means UNDERSTATE the model, because 30 of the 300 slots are ill-posed.** An earlier
version of this doc called the response "bimodal — tracks 93%, completely misses 7%". **That framing was
refuted by adversarial review and is retracted.** The 8 slots where the hub predicts ≈0 despite a big applied
shift are not a model failure mode — they are slices **breathing physically emptied out of the FOV**:

| slots with applied ≥5 mm | gather05 | no_gather |
|---|---|---|
| ignored (\|pred\| < 2 mm) | 8/116 (7%) | 31/116 (27%) |
| …their position in the stack (`z_norm`) | **0.90** (the far end) | 0.48 (**mid-stack**) |
| …their heart content (`heart_frac`) | **0.0021** (heart-free) | 0.0401 (**contains heart**) |
| …their FOV content after breathing | **0.059** — evacuated (was 0.216 clean) | 0.206 (**intact**) |
| **content-bearing breaths ignored** (`fov ≥ 0.15`) | **0 / 100 = 0%** | **23 / 100 = 23%** |
| low-content breaths ignored (`fov < 0.15`) | 8/16 = 50% | 8/16 = **50% — identical** |

The reslice samples at `z + d/12`; on an end-of-stack plane a deep breath pulls that past the last content
plane into zeros, leaving a ~94%-empty, heart-free slice. **Predicting nothing there is an ill-posed slot, and
both models fail those identically (8/16 each).** So the honest headline, restricted to the 270/300
content-bearing slots:

| | slope | corr | EPE |
|---|---|---|---|
| **gather05 (hub)** | **0.844** | **0.958** | **1.10 mm** |
| no_gather | 0.668 | 0.843 | 2.24 mm |

**On slices that actually contain anatomy, the hub never ignores a breath (0/100).** And C1 gets *stronger*,
not weaker: `no_gather`'s failures are **real** — 23 mid-stack slices *with heart in them*, ignored outright.

Two further corrections from the same review, both verified:
- **Under-correction is real but magnitude-dependent, not a uniform 80%.** Dropping the 8 ill-posed slots
  moves the slope only 0.807 → 0.829, and every robust estimator (Theil–Sen 0.812, through-origin 0.809,
  heart-ROI-only 0.782) lands in 0.78–0.84 — so the mixture-artifact hypothesis is dead and the EIV argument
  holds. But recovery is **~55% at 2–5 mm** (the *worst* bin) and 81–88% at ≥5 mm.
- **The 96/91/68% deep-collapse profile belongs to the OLDER docs/42 `gather05`**, a different model on a
  different series. Do not carry it here.

Three supporting facts, each measured:
- **slope < 1 is real under-correction, not regression attenuation.** `x` is the exact simulated shift with
  zero measurement error, so the OLS slope is unbiased no matter how noisy `y` is (noise in `y` inflates the
  standard error; it does not bias the slope). The −0.98 mm signed bias says the same thing.
- **The motion is a near-rigid slab shift, not a smear**: within-slot Δz std 0.57 mm (hub), 0.48 mm (prior).
- **The reference slot is not doing the work**: leave-slot-0-out moves the slope only 0.807 → 0.789. (This
  matters because `tools/exp_4wok_analysis.py` excludes slot 0 while the trainer and `resp_diag` include it —
  and docs/38 §4's claim that slot 0 is excluded is **stale**; the shipped code includes it.)

## 5. contz — UNDECIDABLE in-distribution (an earlier "it smears" reading was wrong)

**This section previously concluded "contz is a no-ship because it smears content". That conclusion was
REFUTED by adversarial review and is retracted.** What follows is the corrected reading; the retraction is
kept deliberately, because the error is instructive.

**The killer fact: contz's ORACLE is 3.04 dB worse than everyone else's.**

| run | `mse_heart_oracle` | → dB | unique values across all 40 epochs |
|---|---|---|---|
| gather05 / no_gather / dino_ft / aug_moderate / lowdiff100 | 0.00249 | **26.04** | 1 |
| **contz** | **0.00502** | **22.99** | 1 |

The oracle (`loss.py:576-584`) is a **Δ=0 splat of the TRUE target-phase content through the same splat with
the same coverage weights** — zero misplacement, perfect content, **no model at all**. contz's ceiling is
3.04 dB lower, it is **constant at every epoch including ep1**, and it is **larger than contz's entire
measured deficit** (bbox −1.9 dB, static −2.6 dB). So contz is not competing on the same scale: a *perfect*
model under contz's configuration would still lose ~3 dB on this val set.

Two model-independent causes, both in code, neither about learning:
- **The input itself is destroyed before the model sees it.** `continuous_z` builds each input slice as a
  2-plane blend across *different z at the same t* (`mri_dataset.py:434-440`:
  `canon_slices = (1-frac)*s0 + frac*s1`). Information is gone at the dataloader.
- **Off-grid coordinates pay a double resample** (sample + splat) against a V_gt that lives on the discrete
  grid — visible as the `full` identity baseline shifting 25.4 → 24.9.

**Why my "smearing" inference failed** — three errors worth remembering:
1. **I epoch-matched every PSNR at ep26 but quoted `hole_frac` at ep40.** At ep26 the hole fractions are
   **0.0552 vs 0.0553 — identical**. The "fewer holes" premise, the whole basis of the smearing story, does
   not exist at my own chosen epoch. Never mix epochs within one argument.
2. **"Fewer holes + worse PSNR" is reproduced by a PERFECT model.** Running the real `splat_to_volume` with
   Δ=0 and true content, off-grid z *always* cuts holes (−0.075) and costs PSNR at realistic through-plane
   decorrelation (−1.9 dB). The signature I attributed to the model is splat geometry.
3. **"Worse from ep1" argues FOR a structural cause, not against it.** A model that learns something bad
   diverges *over* training; a handicap present before training could differentiate is structural — and the
   constant oracle proves it is.

Also: `coverage_frac` moves +0.94 pp for contz, which contradicts "content spread over more voxels".

**Corrected verdict: the in-distribution val CANNOT decide contz, in either direction.** Worse, it is
structurally incapable of measuring what contz is *for*: continuous-z exists to model **off-grid inference
slices**, but this val's V_gt sits on the discrete grid, so the setup can only ever measure contz's cost and
never its benefit. **C3 must be decided on the frozen OOD bundles** (MIITT's native 10 mm pitch), where every
method sees byte-identical input. Note contz also carries a genuine eval asymmetry there: it gets
`--continuous-z` to match its training, so its OOD arm differs from the hub's by two things, not one.

**The one surviving general lesson:** `hole_frac`↓ is *not* automatically good — off-grid splatting lowers it
for free — so the docs/38 rule's "without hole↑" clause has a blind side.

## 6. EF: no power, so C5 cannot be decided by it (and doesn't need to be)

`val/ef/slope` across epochs, n=29 at **every** point (nothing degenerate):

| run | ep1 | ep6 | ep11 | ep16 | ep21 | ep26 | ep31 | ep36 |
|---|---|---|---|---|---|---|---|---|
| gather05 | .626 | .592 | .426 | .383 | **.083** | .404 | .558 | .574 |
| contz | .428 | .729 | .548 | .760 | .670 | **1.308** | .892 | .704 |
| lowdiff100 | .653 | .720 | .542 | .397 | .575 | **.000** | — | — |
| dino_ft | .464 | .374 | .249 | .237 | .541 | .428 | **.157** | — |

The hub's own within-run swing (**0.08–0.63**) exceeds every between-run difference, and the analytic CI makes
it concrete: at the observed Spearman (0.20–0.45), `SE(b)/b = sqrt((1/r²−1)/(n−2))` puts a single slope
estimate's 95% CI at roughly **±0.6** — wider than the entire between-run spread. So 0.000 vs 0.626 is **one
confidence interval**. **C5 is NULL on power grounds**, matching docs/33's prediction (swapping the regularizer
entirely left EF unchanged) — a successful rule-out, now confirmed under the one-frame + per-subject-breathing
regime.
**Consequence:** nnU-Net EF/Dice (docs/43 Layer C) is **not run** — at n=29 with this variance it cannot
resolve the differences in play. Revisit at epoch 100 if something ships.

**Two things adversarial review corrected here — the noise band is not a licence to dismiss everything:**
- **contz's EF is genuinely separated, not noise.** Paired against the mean of the other runs: **+0.319,
  t=2.71, p=0.030**, above the pack **7/8** epochs, and `val/ef/spearman` corroborates **independently**
  (p=0.033, 7/8). Its slope reaches **1.308** — an *over*-shoot of EF dynamic range. Sweeping it into the
  noise band was wrong. (It is still not a clean win: per §5 contz reconstructs from different inputs against
  a 3.04 dB-handicapped ceiling, so its EF is not comparable either.)
- **lowdiff100's 0.000 is TRUNCATED, not noise.** It sits **5.3σ** outside its own prior band (mean 0.577,
  sd 0.109) — that is an extreme excursion at the **last logged point of a killed run**, not a typical draw.
  The right word is *uninterpretable*. Tellingly **the hub made the identical excursion** (0.083 at ep21,
  z=−4.57) **and recovered** to 0.558/0.574 — such excursions happen and resolve; lowdiff100 was simply cut
  off before it could. (Its Spearman of −0.018 is *not* independent corroboration — same 29 pairs.)
  dino_ft's 0.157 (z=−2.05) *is* within band.

## 7. In-distribution head-to-head on the frozen bundles (all n=30, identical inputs)

| model | clean PSNR | breath PSNR | cost | vs hub (breath) |
|---|---|---|---|---|
| **1f_dino_ft** (ep33, LR 2e-5) | **22.239** | **20.745** | 1.494 | **+0.161** |
| **1f_gather05** (hub, ep39) | 22.016 | 20.584 | 1.432 | — |
| 1f_no_gather (ep37) | 21.872 | 20.543 | 1.329 | −0.041 |
| 1f_aug_moderate (ep39) | 21.981 | 20.501 | 1.480 | −0.083 |
| *gather05 (docs/42, ep100, prior series)* | *22.09* | *20.57* | *1.52* | *−0.01* |
| *SVRTK (classical)* | *28.23* | *17.63* | *10.60* | *−2.95* |

Three things worth noting.

1. **The whole ablation spans 0.24 dB.** Four models differing by real, statistically significant amounts in
   *placement* (2.6× EPE), *coverage* (29% holes), and *training regime* land within a quarter of a dB of each
   other. That is §3's finding restated on independent data: they all hit the same appearance wall.
2. **`dino_ft` is the best, and it is the underdog** — +0.161 dB over the hub while sitting at **ep33 vs 39**
   and **half the LR** (2e-5). It independently confirms the wandb read (+0.15–0.25 dB epoch-matched). It is
   still the largest single effect in the series, and still small. **C4 needs the OOD arm**, and it is a
   2-knob package (unfreeze + LR), so a win is "DINO finetuning done properly", not attributable to the
   unfreeze alone.
3. **The hub at epoch 39 already matches the fully-trained docs/42 gather05 under breathing** (20.584 vs
   20.57). Unlike the wandb val numbers these *are* comparable across series, because the frozen bundles are
   byte-identical (`breath_disp_mm` 6.668 for both). And **aug's in-dist cost is only −0.08 dB** — the
   expected docs/05 dip, nowhere near collapse, so C2 stays alive for the OOD arm.

## 8. ⚠️ On real OOD (MIITT), the breathing metric measures RELOCATION, not breathing

**Read this before interpreting any MIITT breathing number.** `1f_aug_moderate` on MIITT (13 real gated
subjects, native 10 mm pitch):

| | in-dist (CMRx) | **real OOD (MIITT)** |
|---|---|---|
| slope | 0.873 | 0.860 |
| corr | 0.958 | **0.715** |
| EPE | ~1.03 mm | **4.15 mm** (removes only 22% of the 5.31 mm null) |
| **clean negative control** | **0.14 mm** | **8.25 mm** (max 24.4) |

The clean control — where the applied shift is **exactly zero** — degrades **59×**. But this is **not
hallucinated breathing**: the predicted Δz is **strongly structured in z** (r=−0.78, **p=3e-28**, −2.36 mm per
plane), concentrated at one end of the stack:

| z-plane | 1–4 | 5 | 6–11 |
|---|---|---|---|
| mean predicted Δz on un-breathed input | **+13 to +22 mm** | +6.7 | ≈0 |

That is a **large, systematic, whole-field through-plane RELOCATION** of one end of the stack — applied on data
with no breathing at all. It reproduces, on this new series, the known failure mode from
`project_gather_contz_relocation_not_flicker` (*"a large whole-field CONSTANT through-plane Δz that RELOCATES
the volume, NOT flicker"*).

**It is systematic across independent training series, which points at geometry.** The docs/42 models — a
*different* series trained months earlier — show the same z-structured relocation on the same MIITT bundles
(free to check; their `resp_diag.json` already exist):

| model | series | clean Δz on MIITT | vs z-plane |
|---|---|---|---|
| `gather05` | docs/42 (prior) | **+4.22 mm** | r=−0.457, p=4e-08 |
| `gather05_contz` | docs/42 (prior) | +4.59 mm | r=−0.399, p=8e-08 |
| `1f_aug_moderate` | this series | **+8.22 mm** | r=−0.780, p=3e-28 |

Every model relocates, and always in the same z-structured direction.

**The obvious explanation — the 10 mm vs 12 mm pitch snap — is REFUTED by measurement.** `MIITTGatedAdapter`
snaps MIITT's native 10 mm slices onto the canonical 12 mm grid
(`inference/adapters/base.py:69-73`: `idx = floor(d/12 + 5.5 + 0.5)`). Reproducing that snap exactly for a
13-slice MIITT stack, the induced displacement is a **sawtooth bounded at ±6 mm** — half the canonical pitch,
by construction — and it *oscillates* rather than ramping:

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| snap error (mm) | +6 | −4 | −2 | 0 | +2 | +4 | +6 | −4 | −2 | 0 | +2 | +4 | +6 |

The observed clean Δz reaches **+21 mm — 3.5× the maximum possible snap error**, and is a smooth hump, not a
sawtooth. **Snap correction cannot explain it.** (An earlier draft of this doc asserted the pitch mismatch as
the likely cause; that was wrong and is retracted.)

**The cause is genuinely UNKNOWN.** Three candidates, and they do not all have the same sign:
1. **Domain-shift relocation (bad).** The model, trained only on CMRx geometry, pulls MIITT anatomy toward
   where CMRx hearts sit in the canonical cube. This is the `gather_contz_relocation_not_flicker` failure mode.
2. **Correcting real acquisition drift (GOOD — would invert the reading).** MIITT gated is a **real**
   acquisition: its slices are acquired over minutes in separate breath-holds, which genuinely drift. If the
   clean stack contains real inter-slice misalignment, the model's nonzero Δz is it **doing its job**, and
   MIITT's "clean arm" is **not a valid negative control at all** — `applied=0` only means *we* added no
   simulated shift, not that the data is unshifted.
3. **Simply undertrained (most likely, and cheap to test).** See the epoch pattern below.

**The epoch pattern is the strongest lead.** Compare *within* this series (fair — same series, same epoch)
against the docs/42 model (different series, **ep100**):

| model | series / epoch | clean Δz (MIITT) | breath EPE | breath corr |
|---|---|---|---|---|
| `1f_gather05` (hub) | this series, **ep39** | **8.84 mm** | 4.71 mm | 0.614 |
| `1f_aug_moderate` | this series, **ep39** | **8.22 mm** | **4.15 mm** | **0.715** |
| `gather05` | docs/42, **ep100** | **4.22 mm** | — | — |

Both **ep39** models sit at ~8–9 mm; the **ep100** model at 4.2 mm — **half**. That is consistent with the
relocation *shrinking as training matures*, which would make this largely an artifact of reading mid-training
checkpoints rather than a fundamental OOD failure. **Directly testable:** re-run
`tools/fig_ood_relocation.py` when the resumed runs reach ep100. If the clean Δz falls to ~4 mm, candidate 3
explains most of it.

**A correction worth recording:** an earlier draft said "aug doubles the relocation (8.22 vs the prior
gather05's 4.22)". That was a **cross-series comparison** — aug@ep39 against a different series@ep100 —
exactly the mistake docs/43 §5 warns about. **Within-series, aug is BETTER than the hub on every OOD measure**
(clean 8.22 vs 8.84, EPE 4.15 vs 4.71, corr 0.715 vs 0.614), which is what C2 predicts aug should do.

Candidates 1 and 2 remain distinguishable by a concrete experiment: (1) predicts the recon becomes *less*
anatomically coherent than the input stack; (2) predicts *more*. **Until that runs, do not read the MIITT
relocation as a failure.**

**Consequences, both important:**
1. **MIITT breathing slope/corr/EPE do NOT measure breathing quality.** They are dominated by a constant
   relocation ~2–5× the breathing signal. The seemingly-healthy MIITT slope (0.86) is especially misleading:
   a large systematic offset barely moves the *slope* while destroying the *correlation* (0.958 → 0.715) and
   the EPE. **Never quote MIITT `resp_diag` slope as an OOD breathing result** without the clean control
   beside it.
2. **The MIITT clean control is the single most diagnostic OOD number we have** — far more so than PSNR. It
   is free (already written by every eval run) and it is the thing that exposed this.

## 9. THE VERDICTS — paired per-subject, both cohorts, all 12 passes complete

Every model ran on **CMRx (30)** and **MIITT (13)** with byte-identical frozen inputs and one scorer.
Paired per-subject t-tests on **breath PSNR** (the metric that matters — all methods see the same corrupted
input and the same GT):

| comparison | in-dist (n=30) | **real OOD, MIITT (n=13)** | verdict |
|---|---|---|---|
| **C1** gather05 − no_gather | +0.041, p=0.48, 15/30 — **null** | **+0.402, p=0.020, 10/13** | **SHIP** — pays on real data |
| **C2** aug_moderate − gather05 | −0.083, p=0.09 — n.s. | **+0.632, p=0.016, 10/13** | **SHIP** — exactly docs/05's trade |
| **C3** contz − gather05 | **−0.559** (worst of six) | see §5 — not decidable | **NO-SHIP** in-dist; OOD confounded |
| **C4** dino_ft − gather05 | **+0.161, p=0.004, 22/30** | +0.067, p=0.71 — **n.s.** | **in-dist only; does NOT transfer** |
| **C5** lowdiff100 − gather05 | +0.073, p=0.08 — n.s. | +0.525, p=0.008, 12/13 ⚠️ | **NULL** in-dist; OOD **confounded** (ep25) |

Full cohort table (breath PSNR, dB):

| model | CMRx (30) | MIITT (13) |
|---|---|---|
| **1f_aug_moderate** | 20.501 | **16.786** ← best OOD |
| **1f_dino_ft** | **20.745** ← best in-dist | 16.087 |
| 1f_lowdiff100 (ep25) | 20.511 | 16.679 |
| **1f_gather05** (hub) | 20.584 | 16.153 |
| 1f_no_gather | 20.543 | 15.751 |
| 1f_contz | 20.025 | — |
| *NeSVoR (classical)* | — | *15.118* |
| *SVRTK (classical)* | *17.629* | *14.835* |

**Reading the verdicts:**
- **C1 SHIPS.** Null in-distribution, significant on real data (+0.40 dB, 10/13). §3 explains the mechanism:
  the appearance wall hides it in-distribution.
- **C2 SHIPS, and vindicates docs/05.** aug costs in-distribution (−0.08, n.s.) and wins OOD (+0.63, p=0.016)
  — precisely the trade it was justified on. It also has the best breathing of the series and, within-series,
  the *lowest* OOD relocation.
- **C4 inverts and is the cautionary tale.** dino is the **only** significant in-dist winner (+0.161,
  p=0.004, 22/30) and is **flat on OOD** (p=0.71). Judging C4 on in-distribution PSNR alone — which docs/43
  §2 specified — would have shipped a change that doesn't transfer. Also a 2-knob package (unfreeze + LR).
- **C5 stays NULL.** Its OOD "win" (+0.525, 12/13) is **confounded**: lowdiff100 is the least-trained model
  (ep25 vs 39). Note the direction — *less* training ⇒ *better* OOD — which is the same pattern as the OOD
  relocation shrinking with epochs (§8). Both are consistent with OOD performance being **non-monotonic in
  training**, i.e. in-distribution overfitting. That is a hypothesis worth testing at ep100, not a result.

**⚠️ One caveat on every MIITT number:** several models have **negative** `cost_psnr` — they score *better*
from breathing-corrupted input than from clean input (hub −0.51, aug −0.65). That is not physical and is
further evidence (with §8's relocation) that **MIITT's clean arm is not a valid reference**. The *breath*
comparisons above remain sound — all methods see identical corrupted input and identical GT — but do not quote
MIITT `clean_psnr` or `cost_psnr` as results.

## 10. Status / open items

- **C2 (aug) and C4 (dino) are still open** — both are decided by the MIITT OOD arm, in progress. C3 (contz)
  likewise, and now *only* on OOD (§5).
- Everything here is **mid-training** (epochs 25–39 of 100, mid-cosine at ~70% peak LR). Re-read after the
  resumed runs reach epoch 100 — especially C2/C4, whose in-dist margins (~0.1–0.2 dB) are inside the
  run-to-run band.
- **The appearance wall is the target now, not breathing.** §3 shows 88% of the reconstruction error is
  common-mode and placement-independent. Any further work on through-plane accuracy is optimising a variable
  the reconstruction has been measured not to care about — at 12 mm pitch, in this regime. The levers worth
  pulling are the ones that attack target-phase appearance synthesis (docs/36's roadmap), not placement.
