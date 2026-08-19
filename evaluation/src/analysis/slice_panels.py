#!/usr/bin/env python
"""slice_panels.py — per-arm diagnostic panels for the frozen eval bundles.

Three panels (choose via --panel; default all), written INTO the arm dir beside the gifs
(volumes/<ds>/out/<subj>/<arm>/) and auto-rendered by ../engine/assemble_and_gif.py for VGGT arms.
(The old panel_cycle GIF was dropped — it duplicated the engine gif_clean/breath montage.)

  panel_input.gif   2 rows (clean / breath input) x N_SLOTS cols, ANIMATED over t. Only the
                    REFERENCE column cycles (slot 0 = (t_target, z_mid)); companions stay fixed at
                    their acquired phase — 'the middle slice is the only one moving', the one-frame
                    input contract. Valid on BOTH arms (rows fetch clean and breath directly).

  panel_dvf.png     4 rows (input / predicted Δx / Δy / Δz map) x N_SLOTS cols, ED only, BREATH
                    ARM ONLY. Δ = ed_dvf.npz delta[...,{0,1,2}] * MM_PER_NORM[{0,1,2}], diverging;
                    Δx/Δy share one in-plane color limit, Δz gets its own (very different mm/norm —
                    ports training's `_log_volume_and_dvf_to_wandb` DVF figure, trainer_viz.py).
                    header z{k|k.k} · t={phase} [REF] · applied +X.X mm, pred vs true below.

  panel_lookup.png  round-trip / analysis-by-synthesis (BREATH ARM, ED). ≤4 slot rows x 4 cols:
                    input | V_canon@pred | V_gt@pred | |Δ|@pred, where pred = scanner_coords + Δ
                    (sample_volume). Port of training's _log_lookup_to_wandb: col1≈col2 by
                    construction (renderer blur), col2≈col3 by training (recon error).

All grayscale intensity displays (panel_input, panel_dvf's input row, panel_lookup's 3 gray
cells) apply the SAME gamma=0.7 curve as training's own panels (`_display_gamma`,
trainer_viz.py) — ON by default; set GAMMA=0 to restore plain linear display.

N_SLOTS VARIES and is NOT 12: it is this SUBJECT'S OWN D (native-z, docs/58 — 9 to 18 across the
pooled cohort), and continuous-z keeps fractional z with no collision dedup so slots can share a
rounded plane. dvf/lookup are breath-only: run_vggt saves the per-pixel Δ field only for
t == ED_PHASE under `if breathing:`, so `--arm clean` renders panel_input only.

Pure disk read — no GPU, no VGGT model load (lookup uses torch sample_volume on CPU). Input
frames come from the frozen bundle itself (`prep_bundle`), which IS what the model was fed, so
there is no second placement implementation that could drift.

Run:
  PY=/home/minsukc/micromamba/envs/svr/bin/python
  $PY evaluation/src/analysis/slice_panels.py                # hub x 4 cohorts, rep subject
  $PY .../slice_panels.py --cohort acdc --subject patient042 --method vggt_20260719_1f_dino_ft_ep99
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))                   # evaluation/ (paths.py)
import paths                                                                    # noqa: E402
sys.path.insert(0, str(paths.EVAL_ROOT / "src" / "engine"))                    # evaluation/src/engine (run_vggt)
import run_vggt as R                                                            # noqa: E402
sys.path.insert(0, str(paths.EVAL_ROOT.parent / "training"))                   # training/ (preprocess)
from data.preprocess import Z_HALF_MM                                           # noqa: E402

# normalized [-1,1] -> mm. Through-plane is Z_HALF_MM, a CONSTANT for every subject under physical
# z (docs/58); the retired `MM_PER_NORM[2] = 66.0` encoded the old fixed 12-plane x 12 mm cube.
MM_PER_NORM = R.MM_PER_NORM

HUB = "vggt_augaggr224hw2_ep300"
COHORTS = list(paths.DATASETS)
FOV_GATE = 0.05                                    # matches run_vggt.resp_diag's `imgs > 0.05`

# Same curve as training/trainer_viz.py:_display_gamma — mid-tones brightened for readability.
# ON by default (matches training's own panels); GAMMA=0 restores the old linear display.
GAMMA_ON = os.environ.get("GAMMA", "1") != "0"
GAMMA_TAG = "  [gamma 0.7]" if GAMMA_ON else ""     # appended to every panel's title, so the
                                                     # rendered image self-documents its own display mode


def display_gamma(image, vmax, gamma=0.7):
    return np.clip(image / max(float(vmax), 1e-8), 0.0, 1.0) ** gamma


def prep_bundle(subj_dir, T, D, dz_mm):
    """(gt, entries, fetch) straight from the frozen bundle — ONE reader for every source.

    Replaces the old four-entry PREP table, which existed only because three sources reached the
    model through adapters and had to be placed onto a shared 12-plane cube. Every bundle is now
    written on the SUBJECT'S OWN native grid by `build_inputs/pooled.py`, so a plane index IS the
    slice index and `z_val` is physical: `(z - (D-1)/2) * dz / Z_HALF_MM` — the same formula
    `MRIDataset.get_data` uses, not the retired index-normalized `z/(D-1)*2-1`.
    """
    gt = R.load_bundle(subj_dir, T, "gt")
    clean = R.load_bundle(subj_dir, T, "clean")
    breath = R.load_bundle(subj_dir, T, "breath")
    entries = [{"z_plane": z, "slice_idx": z,
                "z_val": (z - (D - 1) / 2.0) * dz_mm / Z_HALF_MM} for z in range(D)]

    def fetch(phase, slice_idx, breathing):
        return (breath if breathing else clean)[phase, slice_idx]

    return gt, entries, fetch


# ── locating things on disk ──────────────────────────────────────────────────────────────────
def method_dir(cohort, subject, method):
    """contz's OOD dirs carry a DOUBLED `_contz` suffix (the --continuous-z run appends it);
    on CMRx they don't. Try both or contz is silently dropped (docs/45 gotcha)."""
    base = str(paths.subject_dir(cohort, subject))
    for cand in (f"{base}/{method}", f"{base}/{method}_contz"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"no method dir for {method} under {base}")


def rep_subject(cohort, method):
    """Subject closest to the cohort's median breath PSNR (same choice docs/46 makes)."""
    cands = [paths.summary(cohort, method), paths.summary(cohort, method + "_contz"),
             paths.legacy_summary(cohort, method), paths.legacy_summary(cohort, method + "_contz")]
    hits = [p for p in cands if p.is_file()]
    if not hits:
        raise FileNotFoundError(
            f"no summary json for method '{method}' in cohort '{cohort}' "
            f"(looked in metric_results/ and out/ for {method}[_contz]). "
            f"That cohort/method pair was never scored — pass --subject explicitly.")
    per = json.loads(hits[0].read_text())["per_subject"]
    # An unscorable subject is written with breath_psnr null (aggregate's NaN->null contract);
    # drop those rather than crashing the median.
    per = [r for r in per if r.get("breath_psnr") is not None]
    if not per:
        raise ValueError(f"every per_subject row of {hits[0]} has null breath_psnr — "
                         f"pass --subject explicitly")
    med = np.median([r["breath_psnr"] for r in per])
    return min(per, key=lambda r: abs(r["breath_psnr"] - med))["subject"]


# ── slot bookkeeping ─────────────────────────────────────────────────────────────────────────
def load_slots(md, entries, D, dz_mm):
    """-> (slots, ref_k), slots in npz order.

    The order is READ from `ed_dvf.npz` (`slot_z`), not reconstructed. It used to be re-derived by
    replaying the runner's own slot-building rule, guarded by an equality check that was explicitly
    "necessary but not sufficient" — invariant to permutation within a group of slots sharing a
    rounded plane. `run_vggt` now records the realized draw directly, so there is nothing to
    reproduce and nothing to drift: slot i sat at plane `slot_z[i]`, phase `slot_t[i]`.

    Slot 0 is the reference by construction (`MRIDataset.get_data` puts the target-phase reference
    there when `reference_slot` is on), so ref is simply the entry at `slot_z[0]`.
    """
    npz = np.load(os.path.join(md, "ed_dvf.npz"))
    slot_z = np.asarray(npz["slot_z"]).reshape(-1)
    # Under native-z, entries are indexed BY PLANE (entry k == plane k), so the npz's plane index
    # is the entry index. Round because continuous_z stores fractional z.
    order = [int(round(float(z))) for z in slot_z]
    if max(order) >= len(entries) or min(order) < 0:
        raise AssertionError(f"npz slot_z {slot_z.tolist()} outside this subject's D={len(entries)}")
    ref_k = order[0]

    dx = npz["delta"][..., 0].astype(np.float32) * MM_PER_NORM[0]               # (S,hw,hw) mm
    dy = npz["delta"][..., 1].astype(np.float32) * MM_PER_NORM[1]               # (S,hw,hw) mm
    dz = npz["delta"][..., 2].astype(np.float32) * MM_PER_NORM[2]               # (S,hw,hw) mm
    slots = []
    for i, k in enumerate(order):
        # z_cont = the slot's TRUE (possibly fractional) depth IN PLANE UNITS on this subject's own
        # grid. Physical z (docs/58) inverts as z_index = z_norm * Z_HALF_MM / dz + (D-1)/2 — the
        # exact inverse of `MRIDataset.get_data`'s z_val. The retired form
        # `(z_val+1)/2*(D_CANON-1)` assumed the fixed 12-plane cube and silently mis-places every
        # subject whose dz != 12 mm. Under continuous_z several slots can share a rounded plane,
        # so keying anything by z_plane still drops slices — hence z_cont + per-slot columns.
        z_cont = entries[k]["z_val"] * Z_HALF_MM / dz_mm + (D - 1) / 2.0
        slots.append(dict(i=i, k=k, z=entries[k]["z_plane"], z_cont=float(z_cont),
                          z_val=float(entries[k]["z_val"]),   # physical normalized z, carried through
                          slice_idx=entries[k]["slice_idx"], phase=int(npz["slot_t"][i]),
                          applied=float(npz["applied_disp_mm"][i, 0]),
                          dx=dx[i], dy=dy[i], dz=dz[i], is_ref=(i == 0)))
    return slots, ref_k


def slot_cols(slots):
    """Columns for the Δz panel: one per SLOT, ordered by true canonical depth. NEVER key by
    z_plane — continuous-z slots share rounded planes and a z_plane-keyed dict drops them."""
    return sorted(slots, key=lambda s: s["z_cont"])


def splat_z_weights(z_cont, D):
    """The trilinear z-weights splat_to_volume ACTUALLY deposits for a slice at depth z_cont.

    Mirrors vggt/utils/splat.py exactly, including its in-bounds gate `z0f >= 0 & z0f <= D-2`.
    That bound is ASYMMETRIC: a slice sitting exactly on the TOP plane (z_cont == D-1) has
    floor == D-1 > D-2 and is dropped ENTIRELY — it deposits nothing anywhere — whereas z_cont == 0
    is fine. Verified: at D=12, z_cont 10.999 -> {10: .001, 11: .999} but z_cont 11.0 -> {}.
    A naive `max(0, 1-|z_cont-p|)` would claim 1.00 there. `D` is this subject's own native slice
    count (docs/58), not a fixed 12."""
    z0 = int(np.floor(z_cont))              # int() alone would truncate -0.5 to 0 and wrongly admit it
    if not (0 <= z0 <= D - 2):
        return {}
    frac = z_cont - z0
    # Drop negligible weights: z_cont is a float round-trip (z/11*2-1 -> back), so a slot on plane 1
    # yields 1.0000000000000004 -> a 4e-16 weight on plane 2. plane_note tests dict MEMBERSHIP for
    # the REF tag, so leaving that in would stamp REF on a plane the reference never touches.
    return {z0: 1.0} if frac <= 1e-9 else (
        {z0 + 1: 1.0} if frac >= 1.0 - 1e-9 else {z0: 1.0 - frac, z0 + 1: frac})


def plane_coverage(slots, p, D):
    """Splat mass the input slices deposit on canonical plane p, in slice-equivalents.

    Two things the splat does that a naive rounded-plane count gets wrong, both corrected here:
      z-weighting  trilinear — a slice at z_cont=9.25 puts 75% on plane 9, 25% on plane 10. Bucketing
                   by the ROUNDED plane mislabels: on ACDC contz (z_cont = 1.75, 2.58 ... 9.25)
                   planes 1 and 10 would read "no input" while each receives ~25% of a slice.
      area         the intensity gate is per-PIXEL (`intensity > 1e-3`, splat.py), so a slot deposits
                   (gated pixel fraction) x w_z, NOT 1.0 x w_z. Counting whole slices overstates by
                   up to 1.8x across cohorts — measured gated fraction: CMRx 0.93, MIITT 0.63,
                   OCMR 0.55 — which would print an identical "1.00 slice-eq" for very different
                   real evidence. `s["mass"]` carries that fraction, so empty slots also drop out
                   automatically (no separate has_fov test needed here). NB `mass` does not model the
                   splat's IN-PLANE bound (`x0f <= W-2`), which drops the last row/column — an
                   overstatement of ~0.4%, below the printed precision.

    REMAINING CAVEAT, unavoidable: this is ACQUISITION GEOMETRY ONLY (pre-Δ). The splat consumes
    world_points = scanner_coords + Δ, and the predicted Δz relocates mass by up to ~1.3 planes, so
    a low weight here is NOT proof the recon lacked evidence at that depth (measured counter-case:
    ocmr exam_fs_0012 z2 reads 0.55 slice-eq while the recon there is essentially blank)."""
    return sum(s["mass"] * w for s in slots
               for q, w in splat_z_weights(s["z_cont"], D).items() if q == p)


def plane_note(slots, p, D):
    w = plane_coverage(slots, p, D)
    # A fractional-z reference legitimately lands on BOTH flanking planes, so REF can mark two.
    ref = " REF" if any(s["is_ref"] and p in splat_z_weights(s["z_cont"], D) for s in slots) else ""
    return "no input" if w < 1e-3 else f"{w:.2f} slice-eq{ref}"


def up_model(img, res):
    """Canonical 256 frame -> the model-input grid the model actually saw, so the FOV gate lands on
    the same pixels `run_vggt.resp_diag` gated on.

    `res` is NOT a constant any more: `img_size` is a real config knob (the arm under test trains at
    224, not 518), and the Δ field in ed_dvf.npz is stored at exactly that resolution. Callers pass
    `delta.shape[1]` so the two can never disagree."""
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.as_tensor(img)[None, None].float(), size=(res, res),
                         mode="bilinear", align_corners=True)[0, 0].numpy()


def check_against_resp_diag(md, slots):
    """Cross-check our Δz against resp_diag.json's breath pred_dz_mm. NOT index-aligned in general:
    resp_diag drops slots failing the FOV gate (`if not m.any(): continue`), so compare only the
    slots that pass the same gate, in the same order. Expect ~1e-5 mm (float16 storage floor)."""
    rd = json.load(open(os.path.join(md, "resp_diag.json"))).get("breath", {})
    want = rd.get("pred_dz_mm")
    ours = [s["pred"] for s in slots if s["has_fov"]]
    if want is None:            # `not want` would also swallow [] — exactly the gate disagreement
        return f"SKIPPED — resp_diag has no breath pred_dz_mm key (ours has {len(ours)})"
    # RAISE, don't just print: if this disagrees, every `pred` label on panel B is untrustworthy.
    # Matches the slot-order assert's severity — a printed warning in a multi-cohort log gets lost.
    if len(ours) != len(want):
        raise AssertionError(
            f"resp_diag cross-check n mismatch: ours={len(ours)} resp_diag={len(want)} — the FOV "
            f"gate disagrees with the recorded run (stale bundle?); panel B labels would be wrong")
    d = float(np.max(np.abs(np.asarray(ours) - np.asarray(want))))
    if d > 1e-2:                                   # float16 storage floor is ~1e-5; 1e-2 is real drift
        raise AssertionError(f"resp_diag cross-check max|Δ| = {d:.3e} mm — far above the ~1e-5 "
                             f"float16 floor; this bundle's Δ does not match its recorded run")
    return f"max|Δ| = {d:.2e} mm over {len(ours)} slots"


# ── layout ───────────────────────────────────────────────────────────────────────────────────
def grid(nrow, ncol, cell, top_in, bot_in, left_in, right_in, hgap_in, wgap_in, min_w=0.0):
    """Explicit inch-based grid: every axes box is exactly `cell` x `cell`.

    Do NOT use tight_layout here. With imshow's aspect='equal', tight_layout grows the axes boxes
    to satisfy the aspect and can return OVERLAPPING boxes; the later-drawn row then paints over
    the earlier one, silently cropping ~29% off the top row (measured). Square boxes + square
    images means aspect='equal' is already satisfied, so nothing gets resized."""
    W = left_in + ncol * cell + (ncol - 1) * wgap_in + right_in
    if W < min_w:               # else a few-column panel is narrower than its suptitle and
        pad = (min_w - W) / 2.0  # assert_layout raises instead of writing the figure
        left_in += pad; right_in += pad; W = min_w
    H = bot_in + nrow * cell + (nrow - 1) * hgap_in + top_in
    fig, axes = plt.subplots(nrow, ncol, figsize=(W, H), squeeze=False)
    fig.subplots_adjust(left=left_in / W, right=1.0 - right_in / W,
                        bottom=bot_in / H, top=1.0 - top_in / H,
                        wspace=wgap_in / cell, hspace=hgap_in / cell)
    return fig, axes, W, H


def assert_layout(fig, axes):
    """Guard BOTH layout defects: rows overlapping (the top row gets silently cropped), and — the
    risk created by dropping bbox_inches='tight' — any text or colorbar running off the canvas.
    Call LAST, after every artist incl. the colorbar exists, or the check misses them.
    Column 0 suffices for the overlap test: grid() makes all columns share y-extents."""
    fig.canvas.draw()
    for r in range(axes.shape[0] - 1):
        hi, lo = axes[r, 0].get_position(), axes[r + 1, 0].get_position()
        if lo.y1 > hi.y0 + 1e-9:
            raise AssertionError(
                f"row {r} and {r+1} axes overlap by {lo.y1 - hi.y0:.4f} fig-fraction "
                f"— the top row would be silently cropped")
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    W, H = fig.get_size_inches()
    if tb.x0 < -1e-3 or tb.y0 < -1e-3 or tb.x1 > W + 1e-3 or tb.y1 > H + 1e-3:
        raise AssertionError(
            f"content overflows the canvas (savefig has no bbox_inches='tight' to rescue it): "
            f"tight bbox x[{tb.x0:.3f},{tb.x1:.3f}] y[{tb.y0:.3f},{tb.y1:.3f}] vs figure {W:.3f}x{H:.3f}")


# ── rendering ────────────────────────────────────────────────────────────────────────────────
def col_header(s):
    # show the fractional depth when continuous-z put the slice off the integer plane
    zs = f"z{s['z_cont']:.1f}" if abs(s["z_cont"] - s["z"]) > 1e-6 else f"z{s['z']}"
    tag = " REF" if s["is_ref"] else ""
    return f"{zs} · t={s['phase']}{tag}\n{s['applied']:+.1f} mm"


def render_dvf(inputs_ed, cols, out, title, vmax):
    """One column per INPUT SLOT (count varies; continuous-z can exceed D_CANON), ordered by true
    canonical depth. Deliberately decoupled from the GT/RECON gif, which is per-plane.

    4 rows: input intensity, Δx, Δy, Δz — ports training's `_log_volume_and_dvf_to_wandb` DVF
    figure (trainer_viz.py). Δx/Δy share ONE in-plane color limit and Δz gets its OWN: in-plane
    (256 vox @ 1.4mm) and through-plane (native D @ dz_mm, via MM_PER_NORM) have very different
    mm-per-normalized-unit, so a single shared range would make Δz look ~4x bigger than it is
    (same reasoning as training's fixed IN_PLANE_R/THROUGH_R — kept percentile-based here instead
    of training's fixed mm range, matching this file's existing style and adapting to whatever
    displacement scale the eval bundle actually has). `vmax` (input-row intensity ceiling) is
    caller-supplied — see `build()` — so this panel shares ONE scale with the gifs and every other
    panel for the same subject, rather than each computing its own independently."""
    def _lim(vals):
        lim = float(np.percentile(np.abs(vals), 99.5))
        return lim if (np.isfinite(lim) and lim > 0) else 1.0   # `or 1.0` would pass NaN through

    lim_xy = _lim(np.concatenate([np.concatenate([s["dx"].ravel(), s["dy"].ravel()]) for s in cols]))
    lim_z = _lim(np.concatenate([s["dz"].ravel() for s in cols]))
    n = len(cols)
    fig, axes, W, H = grid(4, n, cell=1.02, top_in=0.70, bot_in=0.34,
                           left_in=0.46, right_in=0.62, hgap_in=0.20, wgap_in=0.04, min_w=8.0)
    m_xy = m_z = None
    for j, s in enumerate(cols):
        for r in range(4):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        im = inputs_ed[s["i"]]
        shown = display_gamma(im, vmax) if GAMMA_ON else im / max(vmax, 1e-3)
        axes[0, j].imshow(shown, cmap="gray", origin="lower", vmin=0, vmax=1)
        m_xy = axes[1, j].imshow(s["dx"], cmap="RdBu_r", origin="lower", vmin=-lim_xy, vmax=lim_xy)
        axes[2, j].imshow(s["dy"], cmap="RdBu_r", origin="lower", vmin=-lim_xy, vmax=lim_xy)
        m_z = axes[3, j].imshow(s["dz"], cmap="RdBu_r", origin="lower", vmin=-lim_z, vmax=lim_z)
        axes[0, j].set_title(col_header(s), fontsize=5.4,
                             color=("#d2691e" if s["is_ref"] else "0.15"))
        # A slot with no in-FOV pixels has NO estimate — resp_diag drops it. Say so instead of
        # printing an ungated whole-field mean that reads as a catastrophic miss.
        if s["pred"] is None:
            axes[3, j].set_xlabel(f"no FOV — n/a\ntrue {s['applied']:+.1f}", fontsize=5.2,
                                  color="#b22222")
        else:
            axes[3, j].set_xlabel(f"pred {s['pred']:+.1f}\ntrue {s['applied']:+.1f}", fontsize=5.2)
    axes[0, 0].set_ylabel("INPUT\n(at ED)", fontsize=6.5)
    axes[1, 0].set_ylabel("pred Δx\n(mm)", fontsize=6.5)
    axes[2, 0].set_ylabel("pred Δy\n(mm)", fontsize=6.5)
    axes[3, 0].set_ylabel("pred Δz\n(mm)", fontsize=6.5)
    fig.suptitle(f"{title}\npredicted Δ at ED (breath arm) — per-pixel, mm; "
                 f"±{lim_xy:.2f} mm in-plane, ±{lim_z:.2f} mm through-plane (saturating)"
                 f"{GAMMA_TAG}", fontsize=8,
                 y=1.0 - 0.05 / H, va="top")
    if m_xy is not None:                                    # own axes in the reserved right margin
        p1 = axes[1, -1].get_position()
        cax1 = fig.add_axes([1.0 - 0.42 / W, p1.y0, 0.055 / W, p1.height])
        fig.colorbar(m_xy, cax=cax1, extend="both").ax.tick_params(labelsize=5)
    if m_z is not None:
        p3 = axes[3, -1].get_position()
        cax3 = fig.add_axes([1.0 - 0.42 / W, p3.y0, 0.055 / W, p3.height])
        fig.colorbar(m_z, cax=cax3, extend="both").ax.tick_params(labelsize=5)
    assert_layout(fig, axes)                                # AFTER the colorbars exist
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def render_input(cols, fetch, T, out, title, res, vmax):
    """The one-frame INPUT the model is fed, animated over the target-phase sweep. Row 1 = clean,
    row 2 = breathing-corrupted; one column per slot (depth-ordered). Only the REFERENCE column
    cycles with t — slot 0 = (t_target, z_mid) — while companion slots stay fixed at their acquired
    phase. So 'the middle slice is the only one moving', exactly the one-frame contract; the breath
    row shows that reference slice wobbling under the respiratory corruption. `vmax` is
    caller-supplied (see `build()`) so this shares ONE display scale with the gifs and every other
    panel for the same subject."""
    def img(s, t, breathing):                                # up-sampled model-input grid, [0,1]
        ph = t if s["is_ref"] else s["phase"]                        # only the reference tracks t
        return up_model(fetch(ph, s["slice_idx"], breathing), res)
    n = len(cols)
    ref = next((s for s in cols if s["is_ref"]), cols[0])
    # gamma applied at DISPLAY time only; vmax is caller-supplied from the raw ROI-masked pixels.
    disp = ((lambda a: display_gamma(a, vmax)) if GAMMA_ON else (lambda a: a / max(vmax, 1e-8)))
    fig, axes, W, H = grid(2, n, cell=1.02, top_in=0.60, bot_in=0.12,
                           left_in=0.52, right_in=0.10, hgap_in=0.22, wgap_in=0.04)
    handles = {}
    for j, s in enumerate(cols):
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        handles[("c", j)] = axes[0, j].imshow(disp(img(s, R.ED_PHASE, False)), cmap="gray",
                                              origin="lower", vmin=0, vmax=1)
        handles[("b", j)] = axes[1, j].imshow(disp(img(s, R.ED_PHASE, True)), cmap="gray",
                                              origin="lower", vmin=0, vmax=1)
        axes[0, j].set_title(col_header(s), fontsize=5.4,
                             color=("#d2691e" if s["is_ref"] else "0.15"))
    axes[0, 0].set_ylabel("clean\ninput", fontsize=6.5)
    axes[1, 0].set_ylabel("breath\ninput", fontsize=6.5)
    sup = fig.suptitle("", fontsize=8, y=1.0 - 0.05 / H, va="top")
    frames = []
    for t in range(T):
        for j, s in enumerate(cols):
            if s["is_ref"]:                                          # only the ref changes per frame
                handles[("c", j)].set_data(disp(img(s, t, False)))
                handles[("b", j)].set_data(disp(img(s, t, True)))
        sup.set_text(f"{title}\ninput fed to the model — reference plane (z{ref['z']}) at phase "
                     f"t = {t}/{T - 1}; companions fixed at their acquired phase{GAMMA_TAG}")
        if t == T - 1:
            assert_layout(fig, axes)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(out, frames, duration=180, loop=0)
    print("wrote", out)


def render_lookup(inputs518, cols, delta_full, V_canon, V_gt, out, title, z_scale, vmax):
    """Round-trip / analysis-by-synthesis (breath arm, at ED). For ≤4 slots across depth, sample the
    reconstruction V_canon AND the GT volume V_gt back at the model's predicted coords
    p = scanner_coords + Δ, beside the input slice and the |V_canon−V_gt|@p error. Port of training's
    `_log_lookup_to_wandb`: col1≈col2 by construction (renderer blur), col2≈col3 by training (recon
    error). Reference row (Δ≈0) is the phase-matched control. inputs518 keyed by SLOT index i.
    `vmax` (all 3 intensity cells) is caller-supplied (see `build()`) — the same scale as the gifs
    and every other panel, replacing the old per-call `max(V_canon.max(), V_gt.max())`."""
    import torch
    from vggt.utils.splat import sample_volume
    ref = [s for s in cols if s["is_ref"]]
    nonref = [s for s in cols if not s["is_ref"]]
    if nonref:
        pick = np.linspace(0, len(nonref) - 1, min(3, len(nonref))).round().astype(int)
        sel = (ref[:1] + [nonref[i] for i in pick])[:4]
    else:
        sel = ref[:4] or cols[:4]
    hw = int(delta_full.shape[1])            # model-input resolution, from the stored Δ field
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2.0 - 1.0).astype(np.float32)          # matches assemble_batch exactly
    y_norm = (py / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    Vc = torch.as_tensor(np.ascontiguousarray(V_canon))[None].float()   # (1,D,H,W)
    Vg = torch.as_tensor(np.ascontiguousarray(V_gt))[None].float()
    ERR = 0.1
    # min_w: 4 columns of 1.35in is only 6.34in wide, narrower than the suptitle for long subject
    # names — assert_layout then raises and NO panel is written (silently, via assemble_and_gif's
    # try/except). Measured over the whole 144-subject cohort: the widest real title needs 7.44in
    # (all 37 cmrx2025 subjects overflowed 6.34; every other source fit). 8.0 matches render_dvf and
    # leaves ~12 chars of headroom; a longer name still fails loudly rather than cropping silently.
    fig, axes, W, H = grid(len(sel), 4, cell=1.35, top_in=0.92, bot_in=0.10,
                           left_in=0.66, right_in=0.10, hgap_in=0.12, wgap_in=0.06, min_w=8.0)
    titles = ["input I", "V_canon @ pred", "V_gt @ pred", "|V_canon−V_gt| @ pred"]
    for r, s in enumerate(sel):
        z_val = s["z_val"]        # physical normalized z, carried from entries (docs/58)
        d = delta_full[s["i"]].astype(np.float32)                  # (hw,hw,3) normalized Δ
        pos = np.stack([x_norm + d[..., 0], y_norm + d[..., 1],
                        np.full_like(x_norm, z_val) + d[..., 2]], -1)   # scanner_coords + Δ
        pos_t = torch.as_tensor(pos.reshape(1, -1, 3)).float()
        # z_scale is REQUIRED (docs/58): [-1,1] is physical z, not this volume's own depth.
        rc = sample_volume(Vc, pos_t, z_scale).reshape(hw, hw).numpy()
        rg = sample_volume(Vg, pos_t, z_scale).reshape(hw, hw).numpy()
        cells = [(inputs518[s["i"]], "gray", 0, max(vmax, 1e-3)),
                 (rc, "gray", 0, vmax), (rg, "gray", 0, vmax),
                 (np.abs(rc - rg), "magma", 0, ERR)]
        for c, (data, cmap, vmin, vm) in enumerate(cells):
            # gamma on the 3 grayscale intensity cells only, NOT the magma error map
            if cmap == "gray" and GAMMA_ON:
                data, vmin, vm = display_gamma(data, vm), 0, 1
            axes[r, c].imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vm)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=7)
        axes[r, 0].set_ylabel(("REF " if s["is_ref"] else "") + f"slot{s['i']}\nz{s['z']}", fontsize=6.5)
    fig.suptitle(f"{title}\nround-trip @pred (breath, ED): input | V_canon | V_gt | |Δ|  — "
                 f"col1≈col2 renderer blur, col2≈col3 recon error"
                 + (f"\n{GAMMA_TAG.strip()}" if GAMMA_ON else ""),
                 fontsize=8, y=1.0 - 0.04 / H, va="top")
    assert_layout(fig, axes)
    fig.savefig(out, dpi=170); plt.close(fig)
    print("wrote", out)


# ── driver ───────────────────────────────────────────────────────────────────────────────────
def build(cohort, subject, method, arm, outdir=None, panels=("dvf",), vmax=None):
    subj_dir = str(paths.subject_dir(cohort, subject))
    md = method_dir(cohort, subject, method)
    meta = json.load(open(os.path.join(md, "metadata.json")))
    breathing = (arm == "breath")
    man = json.loads(paths.manifest(cohort, subject).read_text())
    T = int(man["T"])                                                          # per-subject, not 12
    D, dz_mm = int(man["D"]), float(man["dz_mm"])                              # native-z, per subject

    gt, entries, fetch = prep_bundle(subj_dir, T, D, dz_mm)                    # gt (T,D,H,W) clean
    slots, ref_k = load_slots(md, entries, D, dz_mm)
    cols = slot_cols(slots)
    # Model-input resolution: `img_size` is a real knob now (this arm is 224, not 518). Prefer the
    # arm's own metadata; fall back to the stored Δ field, which is written at exactly that size.
    res = int(meta.get("img_size") or np.load(os.path.join(md, "ed_dvf.npz"))["delta"].shape[1])

    # Input frames at ED, keyed by SLOT index (never by z_plane — continuous-z slots collide there).
    # The reference slot's recorded phase is ED by construction (run_vggt sets slot 0's timestep
    # per queried phase and dumps the Δ field at t == ED_PHASE), which is exactly the frame the Δ
    # field was computed from.
    # NOTE on --arm clean: `phase`/`slice_idx` come from the BREATH npz but select frames from the
    # CLEAN stack. Sound because run_vggt reconstructs BOTH arms from the same name-seeded draw
    # (one `make_dataset` + one seq_index per subject), so the (plane, phase) pairs are identical.
    inputs_ed = [fetch(R.ED_PHASE if s["is_ref"] else s["phase"], s["slice_idx"], breathing)
                 for s in slots]
    for s, im in zip(slots, inputs_ed):
        up = up_model(im, res)                             # the model-input grid this arm used
        g = up > FOV_GATE                             # 0.05 — resp_diag's gate, for the cross-check
        s["has_fov"] = bool(g.any())
        s["mass"] = float((up > 1e-3).mean())         # 1e-3 — the SPLAT's gate, for plane_coverage
        # No in-FOV pixel => no estimate exists (resp_diag drops the slot). Do NOT substitute the
        # ungated whole-field mean: it is background-dominated and reads as a huge prediction error.
        s["pred"] = float(s["dz"][g].mean()) if s["has_fov"] else None

    if vmax is None:
        # STANDALONE fallback only — the wired path (assemble_and_gif) always passes vmax explicitly.
        # Same SHAPE as the gifs' formula (heart-ROI-masked p99.9) but NOT the same number: this pools
        # ED input slices, positivity-filtered; the caller pools GT+recon over all T phases, unfiltered.
        # Measured 0.328 vs 0.344 on CMRx24_Test_P012 (~4.8%). Not unified because the standalone path
        # has no recon/GT stacks loaded — so a standalone panel is very slightly off the gifs' scale.
        # Falls back to the FOV mask (heart_mask is optional per-source) or, if every slot's z-plane
        # happens to miss the ROI entirely, to the unmasked whole-slot pool.
        heart_p = str(paths.heart_mask(cohort, subject))
        mask_p = heart_p if os.path.exists(heart_p) else str(paths.fov_mask(cohort, subject))
        roi3d = R._load_xyz_to_dhw(mask_p) > 0.5                                   # (D,H,W)
        roi_pool = [im[roi3d[s["slice_idx"]]] for s, im in zip(slots, inputs_ed)
                   if roi3d[s["slice_idx"]].any()]
        roi_vals = (np.concatenate(roi_pool) if roi_pool
                   else np.concatenate([im.ravel() for im in inputs_ed]))
        roi_vals = roi_vals[roi_vals > 0]
        vmax = float(np.percentile(roi_vals, 99.9)) if roi_vals.size else 1.0

    n_nofov = sum(not s["has_fov"] for s in slots)
    ndup = len(slots) - len({s["z"] for s in slots})
    chk = check_against_resp_diag(md, slots) if breathing else "n/a (clean arm)"
    print(f"  {len(slots)} slots ({ndup} sharing a z-plane, {n_nofov} with no FOV), "
          f"ref z{entries[ref_k]['z_plane']}; Δz vs resp_diag: {chk}")

    # panel_cycle (GT-vs-recon cardiac gif) was dropped — it duplicated the engine gif_clean/breath.
    # basename(md) — NOT meta['model_name'], which drops both the date and the `_contz` suffix and
    # so collides (miitt vggt_20260713_gather05 vs ..._contz both -> 'gather05').
    mtag = os.path.basename(md)
    # `z_mode` (snapped|continuous) is retired — there is no 12 mm snap left to opt out of. Label
    # the geometry that actually varies now: this subject's own native D and dz.
    zlbl = f"D={D} dz={dz_mm:g}mm" + (" contz" if meta.get("continuous_z") else "")
    base = f"{cohort} · {subject} · {mtag} · {zlbl}"                            # arm-neutral prefix
    ttl = f"{base} · {arm} input"
    # Panels co-locate with the gifs in the arm dir (volumes/<ds>/out/<subj>/<arm>/); --out overrides.
    def dst(name):
        p = f"{outdir}/{name}" if outdir else str(paths.arm_dir(cohort, subject, mtag) / name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    # panel_input: the model's INPUT animated (clean+breath rows); arm-independent content, so a
    # neutral title (no clean/breath suffix) — the file is the same whichever arm triggered it.
    if "input" in panels:
        render_input(cols, fetch, T, dst("panel_input.gif"), base, res, vmax)

    # panel_dvf / panel_lookup need the ED Δ field, which run_vggt dumps ONLY for the breath arm.
    if not breathing:
        if any(p in panels for p in ("dvf", "lookup")):
            print("  [skip] panel_dvf/lookup — ed_dvf.npz is breath-arm only (run_vggt.py:390)")
        return
    if "dvf" in panels:
        render_dvf(inputs_ed, cols, dst("panel_dvf.png"), ttl, vmax)
    if "lookup" in panels:
        V_canon = R._load_xyz_to_dhw(os.path.join(md, f"recon_{arm}", f"vol_t{R.ED_PHASE:02d}.nii.gz"))
        delta_full = np.load(os.path.join(md, "ed_dvf.npz"))["delta"].astype(np.float32)   # (S,hw,hw,3) norm
        res = int(delta_full.shape[1])                                   # model-input resolution
        inputs518 = [up_model(im, res) for im in inputs_ed]              # keyed by slot i
        render_lookup(inputs518, cols, delta_full, V_canon, gt[R.ED_PHASE],
                      dst("panel_lookup.png"), ttl, Z_HALF_MM / dz_mm, vmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", nargs="+", default=COHORTS, choices=COHORTS)
    ap.add_argument("--subject", default=None, help="default: cohort's median-breath-PSNR subject")
    ap.add_argument("--method", default=HUB)
    ap.add_argument("--arm", default="breath", choices=["breath", "clean"])
    ap.add_argument("--out", default=None,
                    help="override output dir (default: the arm dir, volumes/<ds>/<subj>/<arm>/)")
    ap.add_argument("--panel", nargs="+", default=["dvf", "input", "lookup"],
                    choices=["dvf", "input", "lookup"],
                    help="which panels to render (default: all three)")
    a = ap.parse_args()
    if a.subject and len(a.cohort) != 1:
        ap.error("--subject names one subject, so pass exactly one --cohort "
                 f"(got {len(a.cohort)}: {a.cohort}). Subject namespaces are cohort-specific.")
    for c in a.cohort:
        subj = a.subject or rep_subject(c, a.method)
        print(f"[{c}] {subj} · {a.method} · {a.arm}")
        build(c, subj, a.method, a.arm, a.out, panels=tuple(a.panel))


if __name__ == "__main__":
    main()
