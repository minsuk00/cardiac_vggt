# 82 — Eval harness: display gamma, one shared vmax, 3-axis DVF, and prove-it round 2

> **TL;DR & takeaway**
> Two things happened on `eval/nativez-transition` after the docs/81 rename (2026-08-18).
> **(1) Display work:** eval's rendered outputs now match training's own panels — gamma 0.7 on
> every intensity display (`GAMMA=0` restores linear), the DVF panel shows Δx/Δy/Δz instead of Δz
> alone (the data was always in `ed_dvf.npz`; the renderer just never plotted it), and all five
> outputs for a subject share ONE `vmax` instead of three drifting formulas — the gifs looked ~2×
> brighter than the panels because their windows genuinely differed (0.328 vs 0.675 ROI-masked
> p99.9 vs whole-image p99.5 on the same subject). **(2) A second 2-agent prove-it review** of that
> work found 6 real defects, all since fixed and each fault-injected to prove the fix fires. The one
> that mattered: `render_lookup`'s canvas was narrower than its own title, so **`panel_lookup.png`
> was silently unwritable for all 37 cmrx2025 subjects** (and a hard crash from the standalone CLI).
> Display-only changes throughout — no scored metric moved.

## 1. Display gamma

Ported from training's `trainer_viz.py:_display_gamma` so an eval panel and a wandb panel of the
same volume look alike: `clip(x / vmax, 0, 1) ** 0.7`. Defined identically in
`src/analysis/slice_panels.py` and `src/engine/assemble_and_gif.py` — deliberately duplicated
rather than shared, matching the existing convention where both files carry their own copies of
small constants. `GAMMA_ON = os.environ.get("GAMMA", "1") != "0"` (on by default), and every
rendered file appends a `[gamma 0.7]` tag to its title so the image self-documents its display mode.

Applied to intensity displays only. **Not** applied to the Δx/Δy/Δz maps (physical mm on a diverging
colormap, not intensities) or to `render_lookup`'s `|V_canon−V_gt|` magma error map. Measured effect
on a real slice: mean displayed brightness 0.082 → 0.149.

## 2. The DVF panel shows all three axes

`render_dvf` went from 2 rows to 4: input intensity, Δx, Δy, Δz. `run_vggt` always stored the full
3-channel Δ in `ed_dvf.npz` — only the renderer discarded x and y. Δx/Δy share one in-plane colour
limit and Δz gets its own, because in-plane (256 vox @ 1.4 mm) and through-plane (native `D` @
`dz_mm`) have very different mm-per-normalized-unit; one shared range would make Δz look ~4× larger
than it is. Same reasoning as training's fixed `IN_PLANE_R`/`THROUGH_R`, kept percentile-based here
to match this file's style.

## 3. One vmax for all five outputs

**The symptom:** `gif_breath.gif` looked much brighter than `panel_dvf.png` for the same subject.
**The cause, measured not guessed:** the gif used a heart-ROI-masked p99.9 window and the panel used
a whole-image p99.5 one — 0.328 vs 0.675 on the same subject, a 2.06× difference.

**The fix:** `assemble_and_gif.py` computes its (unchanged) ROI-masked p99.9 `vmax` and passes it
into `slice_panels.build(..., vmax=vmax)`.

**The mechanism worth knowing before touching this code.** `gif_clean/breath.gif` and
`render_lookup`'s `V_canon`/`V_gt` cells live on the canonical `(256,256,D)` grid — the same grid as
the heart mask, so ROI-masking is a plain boolean index. `panel_input.gif` and `panel_dvf.png`'s
input row appear to be a different representation (per-slot crops at model-input resolution), which
looks like it would need new coordinate-projection code to mask. It doesn't: `inputs_ed` is the RAW
pre-upsample `fetch()` output, still on the canonical grid, and `up_model()` only resizes the
*displayed* pixels. Since `vmax` is a normalization SCALAR, computing it from the raw canonical
pixels and applying it to the resized display pixels is exact — resizing doesn't change the value
range. That is what made "one vmax" a scalar hand-off instead of a projection problem.

**Known, deliberate divergence.** `build(vmax=None)` — the standalone-CLI path — computes its own
window from ED input slices with a positivity filter, where the wired path pools GT+recon over all
T phases unfiltered. Measured 0.328 vs 0.344 (~4.8%) on `CMRx24_Test_P012`. Not unified because the
standalone path has no recon/GT stacks loaded; the code says so rather than claiming otherwise
(that claim was itself defect #5 below).

## 4. Prove-it round 2 — 6 defects, all fixed

Two reviewers split by failure surface (A: data integrity / split enforcement / shell; B:
visualization / vmax / coordinate space). All 6 findings were verified against real data and real
runs before being accepted, then fixed. **Every fix was fault-injected** — a fix isn't trusted until
the check has been seen to fire on a broken input.

| # | Defect | Fix | Proof the fix fires |
|---|---|---|---|
| 1 | `render_lookup`'s 4×1.35in canvas (6.34in) is narrower than its suptitle → `assert_layout` raises → **no panel written**. Measured over the cohort: **37/37 cmrx2025 subjects overflowed, 0/107 elsewhere**; widest real title needs 7.44in. Silent via `assemble_and_gif`'s try/except; a hard crash from the standalone CLI (how every image this session was made) | `min_w=8.0` on the `grid()` call, matching `render_dvf` | Widest real subject now renders; a 300-char title still raises, so the guard is widened, not disabled |
| 2 | `filter_by_split` fails OPEN: `m.get("split", split)` defaults to the requested split, keeping an unlabelled bundle for ANY split — the opposite of the function's stated purpose | `m.get("split")`, no default | Synthetic manifests: `has_split` kept; `no_split` AND `wrong_split` both dropped |
| 3 | No NaN guard on `gt` before the vmax percentile (recons are hardened, GT wasn't). One NaN voxel → `vmax=nan` → every frame renders blank with no error (`max(nan, 1e-8)` is `nan`, so the existing guard can't rescue it) | `np.nan_to_num` on `gt` | Injected one NaN: `vmax` nan → 0.998 |
| 4 | `stamps_agree` written to `metrics.json` and **never read**. `cost_psnr` differences two separately reconstructed volumes, so a stale `recon_clean/` makes it measure checkpoint drift; the soft cases (pre-stamp run, `ALLOW_MIXED_ARMS=1`) only warned, and the summary was indistinguishable from a verified one | `aggregate.py` reads it, warns, and writes `cost_psnr_unverified` to the summary. **Flags, doesn't fail** — `ALLOW_MIXED_ARMS` exists precisely so a known-good mix can still score | Real run: `cost_psnr UNVERIFIED for 29/29` on cmrx2024; regenerated summary byte-identical except the two new fields |
| 5 | Comment claimed the standalone vmax computes "the SAME number" as the wired one; it doesn't (§3) | Comment states the real divergence and why it isn't unified | — (comment only) |
| 6 | `for SUBJ in $(ls ...)` under `set -e`: a command-substitution failure doesn't trip `-e`, so a missing `out/` dir scores nothing silently | Glob + `N_SCORED` counter that fails loudly at the point of failure | Empty dir now exits 1 with `[fatal]`; populated dir still counts 15 |

Standing checks after the fixes: `pytest tests/` **363 passed**, `check_paths.py` **ALL PASS**.

**What #4 revealed:** 29/29 cmrx2024 subjects' `cost_psnr` is unverified (4 explicitly `False`, 25
predating the key). Those numbers aren't necessarily wrong — they are simply no longer *claimed* to
be verified.

## 5. Gotchas

- **`find` misses the whole volumes tree without `-L`.** `evaluation/volumes` is a symlink onto
  GPFS; a plain `find evaluation/volumes -name 'panel_*'` returns **zero** while the files exist.
- **Panels are absent, not stale.** 139 of 144 scored arm dirs have no gifs and no panels at all;
  the only 5 that do were rendered 2026-08-18 with current code. Every older rendered file lives
  under `_archive_prenativez_20260712/`. So the render sweep is a FIRST render, not a refresh.
- **A `GAMMA=0` test run silently overwrote gamma-on files** already sent to the user — these
  renderers write in place, so a probe run with a different display setting destroys the artefact
  it was probing. Render probes to a temp `outdir=`.
- **The baseline shells are not native-z ready** (pre-existing, unfixed; also flagged in docs/81).
  `run_svrtk3d.sh` and `run_nesvor.sh` hardcode `THICK=8`, reasoning from CMRxRecon2024's
  12 mm pitch = 8 mm thickness + 4 mm gap. Under native-z the live cohort spans **8 distinct
  pitches** — dz = 12.0 (×72), 10.0 (×57), 8.0 (×5), 5.0 (×4), 8.8 (×2), 9.6 (×2), 7.8, 7.0 — so
  the default is right for only 72/144 subjects. `-thickness` sets the PSF through-plane width, so
  a wrong value is a silently degraded reconstruction, not a crash. `dz_mm` is in every manifest;
  the dz→thickness RULE still needs deciding (`dz − 4` gives an implausible 1 mm at dz = 5.0).
  **This blocks the baseline campaign** — no baseline arm exists on the native-z contract yet.
