#!/usr/bin/env python
"""slice_panels.py — per-arm diagnostic panels for the frozen eval bundles.

Three panels (choose via --panel; default all), written INTO the arm dir beside the gifs
(volumes/<ds>/out/<subj>/<arm>/) and auto-rendered by engine/assemble_and_gif.py for VGGT arms.
(The old panel_cycle GIF was dropped — it duplicated the engine gif_clean/breath montage.)

  panel_input.gif   2 rows (clean / breath input) x N_SLOTS cols, ANIMATED over t. Only the
                    REFERENCE column cycles (slot 0 = (t_target, z_mid)); companions stay fixed at
                    their acquired phase — 'the middle slice is the only one moving', the one-frame
                    input contract. Valid on BOTH arms (rows fetch clean and breath directly).

  panel_dvf.png     2 rows (input / predicted Δz map) x N_SLOTS cols, ED only, BREATH ARM ONLY.
                    Δz = ed_dvf.npz delta[...,2] * MM_PER_NORM[2], diverging, shared scale.
                    header z{k|k.k} · t={phase} [REF] · applied +X.X mm, pred vs true below.

  panel_lookup.png  round-trip / analysis-by-synthesis (BREATH ARM, ED). ≤4 slot rows x 4 cols:
                    input | V_canon@pred | V_gt@pred | |Δ|@pred, where pred = scanner_coords + Δ
                    (sample_volume). Port of training's _log_lookup_to_wandb: col1≈col2 by
                    construction (renderer blur), col2≈col3 by training (recon error).

N_SLOTS VARIES and is NOT 12: continuous-z keeps fractional z with no collision dedup, so slots can
share a rounded plane and outnumber D_CANON (up to 17 on ACDC). dvf/lookup are breath-only: run_vggt
saves the per-pixel Δ field only for t==ED_PHASE under `if breathing:` (run_vggt.py:390), so `--arm
clean` renders panel_input only.

Pure disk read — no GPU, no VGGT model load (lookup uses torch sample_volume on CPU). Reuses
run_vggt's own prep_* so the input frames are the exact pixels the model saw rather than a
reimplementation that could silently drift (OOD cohorts resample native->canonical in `fetch`).

Run:
  PY=/home/minsukc/micromamba/envs/svr/bin/python
  $PY evaluation/analysis/slice_panels.py                    # hub x 4 cohorts, rep subject
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                   # evaluation/ (paths.py)
import paths                                                                    # noqa: E402
sys.path.insert(0, str(paths.EVAL_ROOT / "engine"))                            # evaluation/engine (run_vggt)
import run_vggt as R                                                            # noqa: E402
from inference.adapters.base import D_CANON, MM_PER_NORM                        # noqa: E402

# output dir defaults to paths.figure_dir (figures/<ds>/<subj>/<arm>/ on GPFS); --out overrides
HUB = "vggt_20260719_1f_gather05_ep99"
COHORTS = list(paths.DATASETS)
FOV_GATE = 0.05                                    # matches run_vggt.resp_diag's `imgs > 0.05`

# T is per-SUBJECT, read from manifest.json — it is NOT 12. CMRx resamples to 12, but the OOD
# cohorts keep their native phase count (measured: ACDC 13-35, OCMR 18-30, MIITT 30), and slot_t
# indexes into that. Hardcoding 12 silently blows up on OOD. Only D_CANON (the z depth) is fixed.
PREP = {"cmrxrecon": lambda sd, subj, T, cz: R.prep_cmrxrecon(sd, T, cz),
        "miitt":     lambda sd, subj, T, cz: R.prep_miitt(sd, subj, T, cz),
        "ocmr":      lambda sd, subj, T, cz: R.prep_ocmr(sd, subj, T, cz),
        "acdc":      lambda sd, subj, T, cz: R.prep_acdc(sd, subj, T, cz)}


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
            f"(looked in results/ and out/ for {method}[_contz]). "
            f"That cohort/method pair was never scored — pass --subject explicitly.")
    per = json.loads(hits[0].read_text())["per_subject"]
    med = np.median([r["breath_psnr"] for r in per])
    return min(per, key=lambda r: abs(r["breath_psnr"] - med))["subject"]


# ── slot bookkeeping ─────────────────────────────────────────────────────────────────────────
def slot_entry_order(entries):
    """Reproduce run_vggt.build_slots' ENTRY ORDER for regime='onef':
         slots = [(ref_k, 0)] + [(k, <random phase>) for k in range(n) if k != ref_k]   (ascending k)
    The phases come from the npz (slot_t), so build_slots' rng — hence seq_index — is not needed;
    only the order is. ref_k is computed exactly as reconstruct() does (bbox-centre plane, NOT
    argmin|z_val|)."""
    z_planes = [e["z_plane"] for e in entries]
    z_mid = (min(z_planes) + max(z_planes) + 1) // 2
    ref_k = int(np.argmin([abs(e["z_plane"] - z_mid) for e in entries]))
    return [ref_k] + [k for k in range(len(entries)) if k != ref_k], ref_k


def load_slots(md, entries):
    """-> (slots, ref_k), slots in npz order.

    Sanity-checks the rebuilt order against the npz's own slot_z. NOTE this check is
    necessary-but-NOT-sufficient: it is invariant to permutation within a group of entries sharing
    a rounded z_plane (continuous-z produces such groups). Correctness ultimately rests on both
    orders being ascending-k over the same deterministic `entries` (assign_canonical_z returns a
    totally-ordered `sorted(...)`), which was verified against build_slots. The check still earns
    its keep: it fires loudly on a multiframe dir (length mismatch) and on stale bundles whose
    geometry drifted across a rounding boundary."""
    npz = np.load(os.path.join(md, "ed_dvf.npz"))
    order, ref_k = slot_entry_order(entries)
    got = [entries[k]["z_plane"] for k in order]
    want = npz["slot_z"].tolist()
    if got != want:
        raise AssertionError(
            f"slot order mismatch: rebuilt {got} != npz slot_z {want}. "
            f"(This script reproduces regime='onef' only — a multiframe method dir lands here.)")

    dz = npz["delta"][..., 2].astype(np.float32) * MM_PER_NORM[2]               # (S,hw,hw) mm
    slots = []
    for i, k in enumerate(order):
        # z_cont = the slot's TRUE (possibly fractional) canonical depth. Under continuous_z,
        # assign_canonical_z returns fractional z with NO collision dedup, so several slots can
        # share a rounded z_plane and the slot count can exceed D_CANON (seen up to 17 on ACDC).
        # Keying anything by z_plane silently drops those slices — hence z_cont + per-slot columns.
        z_cont = (entries[k]["z_val"] + 1.0) / 2.0 * (D_CANON - 1)
        slots.append(dict(i=i, k=k, z=entries[k]["z_plane"], z_cont=float(z_cont),
                          slice_idx=entries[k]["slice_idx"], phase=int(npz["slot_t"][i]),
                          applied=float(npz["applied_disp_mm"][i, 0]), dz=dz[i], is_ref=(k == ref_k)))
    return slots, ref_k


def slot_cols(slots):
    """Columns for the Δz panel: one per SLOT, ordered by true canonical depth. NEVER key by
    z_plane — continuous-z slots share rounded planes and a z_plane-keyed dict drops them."""
    return sorted(slots, key=lambda s: s["z_cont"])


def splat_z_weights(z_cont):
    """The trilinear z-weights splat_to_volume ACTUALLY deposits for a slice at depth z_cont.

    Mirrors vggt/utils/splat.py exactly, including its in-bounds gate `z0f >= 0 & z0f <= D-2`.
    That bound is ASYMMETRIC: a slice sitting exactly on the TOP plane (z_cont == D_CANON-1) has
    floor == D-1 > D-2 and is dropped ENTIRELY — it deposits nothing anywhere — whereas z_cont == 0
    is fine. Verified: z_cont 10.999 -> {10: .001, 11: .999} but z_cont 11.0 -> {} (35 of 1118
    method dirs have a slot at plane 11). A naive `max(0, 1-|z_cont-p|)` would claim 1.00 there."""
    z0 = int(np.floor(z_cont))              # int() alone would truncate -0.5 to 0 and wrongly admit it
    if not (0 <= z0 <= D_CANON - 2):
        return {}
    frac = z_cont - z0
    # Drop negligible weights: z_cont is a float round-trip (z/11*2-1 -> back), so a slot on plane 1
    # yields 1.0000000000000004 -> a 4e-16 weight on plane 2. plane_note tests dict MEMBERSHIP for
    # the REF tag, so leaving that in would stamp REF on a plane the reference never touches.
    return {z0: 1.0} if frac <= 1e-9 else (
        {z0 + 1: 1.0} if frac >= 1.0 - 1e-9 else {z0: 1.0 - frac, z0 + 1: frac})


def plane_coverage(slots, p):
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
               for q, w in splat_z_weights(s["z_cont"]).items() if q == p)


def plane_note(slots, p):
    w = plane_coverage(slots, p)
    # A fractional-z reference legitimately lands on BOTH flanking planes, so REF can mark two.
    ref = " REF" if any(s["is_ref"] and p in splat_z_weights(s["z_cont"]) for s in slots) else ""
    return "no input" if w < 1e-3 else f"{w:.2f} slice-eq{ref}"


def up518(img):
    """Canonical 256 frame -> the 518 grid the model actually saw (matches assemble_batch, so the
    FOV gate lands on the same pixels resp_diag gated on)."""
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.as_tensor(img)[None, None].float(),
                         size=(R.INPUT_IMG_SIZE, R.INPUT_IMG_SIZE),
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


def render_dvf(inputs_ed, cols, out, title):
    """One column per INPUT SLOT (count varies; continuous-z can exceed D_CANON), ordered by true
    canonical depth. Deliberately decoupled from the GT/RECON gif, which is per-plane."""
    dz_all = np.concatenate([s["dz"].ravel() for s in cols])
    lim = float(np.percentile(np.abs(dz_all), 99.5))
    if not np.isfinite(lim) or lim <= 0:                    # `or 1.0` would pass NaN through
        lim = 1.0
    n = len(cols)
    fig, axes, W, H = grid(2, n, cell=1.02, top_in=0.70, bot_in=0.34,
                           left_in=0.46, right_in=0.62, hgap_in=0.20, wgap_in=0.04, min_w=8.0)
    # SHARED input-row scale. Per-column `vmax=im.max()` stretches every slot to full white, so a
    # dim apical/basal slot looks as strong as a mid-ventricular one — hiding exactly the variation
    # in input quality across depth this panel exists to show — so share one vmax across slots.
    ivals = np.concatenate([inputs_ed[s["i"]].ravel() for s in cols])
    ivmax = float(np.percentile(ivals[ivals > 0], 99.5)) if (ivals > 0).any() else 1.0
    m = None
    for j, s in enumerate(cols):
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        im = inputs_ed[s["i"]]
        axes[0, j].imshow(im, cmap="gray", origin="lower", vmin=0, vmax=max(ivmax, 1e-3))
        m = axes[1, j].imshow(s["dz"], cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
        axes[0, j].set_title(col_header(s), fontsize=5.4,
                             color=("#d2691e" if s["is_ref"] else "0.15"))
        # A slot with no in-FOV pixels has NO estimate — resp_diag drops it. Say so instead of
        # printing an ungated whole-field mean that reads as a catastrophic miss.
        if s["pred"] is None:
            axes[1, j].set_xlabel(f"no FOV — n/a\ntrue {s['applied']:+.1f}", fontsize=5.2,
                                  color="#b22222")
        else:
            axes[1, j].set_xlabel(f"pred {s['pred']:+.1f}\ntrue {s['applied']:+.1f}", fontsize=5.2)
    axes[0, 0].set_ylabel("INPUT\n(at ED)", fontsize=6.5)
    axes[1, 0].set_ylabel("pred Δz\n(mm)", fontsize=6.5)
    fig.suptitle(f"{title}\npredicted through-plane Δz at ED (breath arm) — "
                 f"per-pixel, mm; ±{lim:.2f} mm scale (saturating)", fontsize=8,
                 y=1.0 - 0.05 / H, va="top")
    if m is not None:                                       # own axes in the reserved right margin
        p = axes[1, -1].get_position()
        cax = fig.add_axes([1.0 - 0.42 / W, p.y0, 0.055 / W, p.height])
        fig.colorbar(m, cax=cax, extend="both").ax.tick_params(labelsize=5)
    assert_layout(fig, axes)                                # AFTER the colorbar exists
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def render_input(cols, fetch, T, out, title):
    """The one-frame INPUT the model is fed, animated over the target-phase sweep. Row 1 = clean,
    row 2 = breathing-corrupted; one column per slot (depth-ordered). Only the REFERENCE column
    cycles with t — slot 0 = (t_target, z_mid) — while companion slots stay fixed at their acquired
    phase. So 'the middle slice is the only one moving', exactly the one-frame contract; the breath
    row shows that reference slice wobbling under the respiratory corruption."""
    def img(s, t, breathing):                                        # up-sampled 518 grid, [0,1]
        ph = t if s["is_ref"] else s["phase"]                        # only the reference tracks t
        return up518(fetch(ph, s["slice_idx"], breathing))
    n = len(cols)
    ref = next((s for s in cols if s["is_ref"]), cols[0])
    allpix = np.concatenate([img(s, R.ED_PHASE, b).ravel() for s in cols for b in (False, True)])
    vmax = float(np.percentile(allpix[allpix > 0], 99.5)) if (allpix > 0).any() else 1.0
    fig, axes, W, H = grid(2, n, cell=1.02, top_in=0.60, bot_in=0.12,
                           left_in=0.52, right_in=0.10, hgap_in=0.22, wgap_in=0.04)
    handles = {}
    for j, s in enumerate(cols):
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        handles[("c", j)] = axes[0, j].imshow(img(s, R.ED_PHASE, False), cmap="gray",
                                              origin="lower", vmin=0, vmax=vmax)
        handles[("b", j)] = axes[1, j].imshow(img(s, R.ED_PHASE, True), cmap="gray",
                                              origin="lower", vmin=0, vmax=vmax)
        axes[0, j].set_title(col_header(s), fontsize=5.4,
                             color=("#d2691e" if s["is_ref"] else "0.15"))
    axes[0, 0].set_ylabel("clean\ninput", fontsize=6.5)
    axes[1, 0].set_ylabel("breath\ninput", fontsize=6.5)
    sup = fig.suptitle("", fontsize=8, y=1.0 - 0.05 / H, va="top")
    frames = []
    for t in range(T):
        for j, s in enumerate(cols):
            if s["is_ref"]:                                          # only the ref changes per frame
                handles[("c", j)].set_data(img(s, t, False))
                handles[("b", j)].set_data(img(s, t, True))
        sup.set_text(f"{title}\ninput fed to the model — reference plane (z{ref['z']}) at phase "
                     f"t = {t}/{T - 1}; companions fixed at their acquired phase")
        if t == T - 1:
            assert_layout(fig, axes)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(out, frames, duration=180, loop=0)
    print("wrote", out)


def render_lookup(inputs518, cols, delta_full, V_canon, V_gt, out, title):
    """Round-trip / analysis-by-synthesis (breath arm, at ED). For ≤4 slots across depth, sample the
    reconstruction V_canon AND the GT volume V_gt back at the model's predicted coords
    p = scanner_coords + Δ, beside the input slice and the |V_canon−V_gt|@p error. Port of training's
    `_log_lookup_to_wandb`: col1≈col2 by construction (renderer blur), col2≈col3 by training (recon
    error). Reference row (Δ≈0) is the phase-matched control. inputs518 keyed by SLOT index i."""
    import torch
    from vggt.utils.splat import sample_volume
    ref = [s for s in cols if s["is_ref"]]
    nonref = [s for s in cols if not s["is_ref"]]
    if nonref:
        pick = np.linspace(0, len(nonref) - 1, min(3, len(nonref))).round().astype(int)
        sel = (ref[:1] + [nonref[i] for i in pick])[:4]
    else:
        sel = ref[:4] or cols[:4]
    hw = R.INPUT_IMG_SIZE
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2.0 - 1.0).astype(np.float32)          # matches assemble_batch exactly
    y_norm = (py / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    Vc = torch.as_tensor(np.ascontiguousarray(V_canon))[None].float()   # (1,D,H,W)
    Vg = torch.as_tensor(np.ascontiguousarray(V_gt))[None].float()
    vmax = float(max(V_canon.max(), V_gt.max(), 1e-3)); ERR = 0.1
    fig, axes, W, H = grid(len(sel), 4, cell=1.35, top_in=0.92, bot_in=0.10,
                           left_in=0.66, right_in=0.10, hgap_in=0.12, wgap_in=0.06)
    titles = ["input I", "V_canon @ pred", "V_gt @ pred", "|V_canon−V_gt| @ pred"]
    for r, s in enumerate(sel):
        z_val = s["z_cont"] / (D_CANON - 1) * 2.0 - 1.0             # recover normalized scanner z
        d = delta_full[s["i"]].astype(np.float32)                  # (hw,hw,3) normalized Δ
        pos = np.stack([x_norm + d[..., 0], y_norm + d[..., 1],
                        np.full_like(x_norm, z_val) + d[..., 2]], -1)   # scanner_coords + Δ
        pos_t = torch.as_tensor(pos.reshape(1, -1, 3)).float()
        rc = sample_volume(Vc, pos_t).reshape(hw, hw).numpy()
        rg = sample_volume(Vg, pos_t).reshape(hw, hw).numpy()
        cells = [(inputs518[s["i"]], "gray", 0, max(vmax, 1e-3)),
                 (rc, "gray", 0, vmax), (rg, "gray", 0, vmax),
                 (np.abs(rc - rg), "magma", 0, ERR)]
        for c, (data, cmap, vmin, vm) in enumerate(cells):
            axes[r, c].imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vm)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=7)
        axes[r, 0].set_ylabel(("REF " if s["is_ref"] else "") + f"slot{s['i']}\nz{s['z']}", fontsize=6.5)
    fig.suptitle(f"{title}\nround-trip @pred (breath, ED): input | V_canon | V_gt | |Δ|  — "
                 f"col1≈col2 renderer blur, col2≈col3 recon error", fontsize=8, y=1.0 - 0.04 / H, va="top")
    assert_layout(fig, axes)
    fig.savefig(out, dpi=170); plt.close(fig)
    print("wrote", out)


# ── driver ───────────────────────────────────────────────────────────────────────────────────
def build(cohort, subject, method, arm, outdir=None, panels=("dvf",)):
    subj_dir = str(paths.subject_dir(cohort, subject))
    md = method_dir(cohort, subject, method)
    meta = json.load(open(os.path.join(md, "metadata.json")))
    cz = meta.get("z_mode") == "continuous"                                     # contz: OOD only
    breathing = (arm == "breath")
    T = int(json.loads(paths.manifest(cohort, subject).read_text())["T"])       # per-subject, not 12

    gt, entries, fetch, _ = PREP[cohort](subj_dir, subject, T, cz)              # gt (T,D,H,W) clean
    slots, ref_k = load_slots(md, entries)
    cols = slot_cols(slots)

    # Input frames at ED, keyed by SLOT index (never by z_plane — continuous-z slots collide there).
    # The reference slot's recorded phase is 0 == ED_PHASE by construction (build_slots seeds
    # slots[0] = (ref_k, 0)), which is exactly the frame the Δ field was computed from.
    # NOTE on --arm clean: `phase`/`slice_idx` come from the BREATH npz but select frames from the
    # CLEAN stack. Sound only because run_vggt calls reconstruct with the same seq_index for both
    # arms (run_vggt.py:383-384), so build_slots' rng yields identical (k, phase) pairs. If the arms
    # ever get separate seeds this silently mislabels — nothing here can detect it.
    inputs_ed = [fetch(R.ED_PHASE if s["is_ref"] else s["phase"], s["slice_idx"], breathing)
                 for s in slots]
    for s, im in zip(slots, inputs_ed):
        up = up518(im)                                                         # the 518 grid
        g = up > FOV_GATE                             # 0.05 — resp_diag's gate, for the cross-check
        s["has_fov"] = bool(g.any())
        s["mass"] = float((up > 1e-3).mean())         # 1e-3 — the SPLAT's gate, for plane_coverage
        # No in-FOV pixel => no estimate exists (resp_diag drops the slot). Do NOT substitute the
        # ungated whole-field mean: it is background-dominated and reads as a huge prediction error.
        s["pred"] = float(s["dz"][g].mean()) if s["has_fov"] else None

    n_nofov = sum(not s["has_fov"] for s in slots)
    ndup = len(slots) - len({s["z"] for s in slots})
    chk = check_against_resp_diag(md, slots) if breathing else "n/a (clean arm)"
    print(f"  {len(slots)} slots ({ndup} sharing a z-plane, {n_nofov} with no FOV), "
          f"ref z{entries[ref_k]['z_plane']}; Δz vs resp_diag: {chk}")

    # panel_cycle (GT-vs-recon cardiac gif) was dropped — it duplicated the engine gif_clean/breath.
    # basename(md) — NOT meta['model_name'], which drops both the date and the `_contz` suffix and
    # so collides (miitt vggt_20260713_gather05 vs ..._contz both -> 'gather05').
    mtag = os.path.basename(md)
    base = f"{cohort} · {subject} · {mtag} · z={meta.get('z_mode','?')}"        # arm-neutral prefix
    ttl = f"{base} · {arm} input"
    # Panels co-locate with the gifs in the arm dir (volumes/<ds>/out/<subj>/<arm>/); --out overrides.
    def dst(name):
        p = f"{outdir}/{name}" if outdir else str(paths.arm_dir(cohort, subject, mtag) / name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    # panel_input: the model's INPUT animated (clean+breath rows); arm-independent content, so a
    # neutral title (no clean/breath suffix) — the file is the same whichever arm triggered it.
    if "input" in panels:
        render_input(cols, fetch, T, dst("panel_input.gif"), base)

    # panel_dvf / panel_lookup need the ED Δ field, which run_vggt dumps ONLY for the breath arm.
    if not breathing:
        if any(p in panels for p in ("dvf", "lookup")):
            print("  [skip] panel_dvf/lookup — ed_dvf.npz is breath-arm only (run_vggt.py:390)")
        return
    if "dvf" in panels:
        render_dvf(inputs_ed, cols, dst("panel_dvf.png"), ttl)
    if "lookup" in panels:
        V_canon = R._load_xyz_to_dhw(os.path.join(md, f"recon_{arm}", f"vol_t{R.ED_PHASE:02d}.nii.gz"))
        delta_full = np.load(os.path.join(md, "ed_dvf.npz"))["delta"].astype(np.float32)   # (S,hw,hw,3) norm
        inputs518 = [up518(im) for im in inputs_ed]                                         # keyed by slot i
        render_lookup(inputs518, cols, delta_full, V_canon, gt[R.ED_PHASE], dst("panel_lookup.png"), ttl)


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
