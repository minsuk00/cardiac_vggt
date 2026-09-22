"""Build the rhythm-arm eval cohorts (docs/110) from an existing frozen bundle.

Each arm becomes its OWN cohort dir in the standard bundle layout, so the whole evaluation
harness (run_vggt, fetal4d_gate, score/*) consumes it unchanged:

    scratch/eval/<source>_<arm>/out/<subject>/
        manifest.json          source manifest + a "rhythm" block; "source" keeps the BASE name
        gt/gt_t{00..T-1}        per-target GT: the clean, unbreathed volume at the REFERENCE
                                slice's true cardiac position for frame f
        breath/stack_t{00..T-1} the simulated input (rhythm + breathing)
        rolled -> breath        symlink: fetal4d_gate.py reads rolled/ and needs no change
        mask*.nii.gz heart_seg.nii.gz   copied verbatim (geometry is untouched)
        seg_gt/                 permuted copy when positions are integers; otherwise NOT written
                                (regenerate with ef_dice.py dump --gt-only + run_seg.sh)

## The index change

`t` stops meaning "cardiac phase" and starts meaning "the reference slice's frame f". Under a
periodic rhythm the reference slice's frame f sits at integer position (f - roll_ref) mod T, so GT
is a pure cyclic permutation of the source gt_t* files and every metric is invariant. Under hrv/af
positions go fractional and GT is a genuine blend -- which is the point: a short beat is cut off
before reaching ED, so scoring against the clean ED-first cine would charge every method for a
volume nobody photographed (docs/110 section 7).

## Arms

    regular_frozen  periodic beats, breathing frozen per plane  -> must reproduce the source rolled/
    regular         periodic beats, frame-wise breathing        -> isolates frame-wise breathing
    hrv             beat length ~ N(1, 0.15), frame-wise        -> mild irregularity (+/-15%, which
                                                                   is ~4x real resting HRV; the
                                                                   name is historical, docs/110 s4)
    af              beat length ~ N(1, 0.25), frame-wise        -> cut-off / hold / volume-matched
                                                                   resume (docs/110 section 5)

    regular24 / hrv24 / af24 / regular_frozen24  -- the SAME rhythms at 24 frames per slice
    (2 nominal beats) instead of 12. New cohort names, so the 12-frame bundles behind docs/111-112
    are never overwritten. See the ARMS table for why 24 frames is the lever.

Usage:
    PYTHONPATH=training:. python tools/build_af_bundle.py --source cmrx2024 \
        --subjects CMRx24_Test_P016,CMRx24_Test_P004,CMRx24_Test_P017
    PYTHONPATH=training:. python tools/build_af_bundle.py --source cmrx2024 --check \
        --subjects CMRx24_Test_P016
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT]
from data.respiratory import lujan_displacement, reslice_volume_vec        # noqa: E402

T = 12
DT = 1.0 / T            # frame time, in units of the nominal R-R
T_BREATH = 5.0          # one breath ~ 5 beats (Furnrohr & Heckel 2026, App. A.6)
N_BEATS = 16            # beats drawn per slice, FIXED (see the RNG note in simulate)
BURN = 3                # beats played before the window opens
HRV_MODEL = "precedent"  # "precedent" = iid N(1, 0.15) | "jansz" = bounded walk, ~+/-4% (a null arm)
INPLANE_MM = 1.4
RR_MIN, RR_MAX = 0.45, np.inf   # only a physiological floor (~systole). Markl's [0.57, 1.55] range
                                # must NOT be used as a clip: measured, it shrinks the realised
                                # sd from 0.25 to 0.216 (docs/110 section 5).
HOLD_MAX = 0.55         # longest full-relaxation hold, in nominal R-R. A MODELLING BOUND, not a cited
                        # physiological value (none exists): it is the rest a full beat gets at
                        # Markl's longest observed R-R (1.55 - 1.0). Only weak resumed beats, whose
                        # nominal duration is short, would otherwise rest longer. docs/114.
HOLD_MAX_FRAMES = int(HOLD_MAX / DT) + 1    # most frames a hold of HOLD_MAX can be sampled at
JUMP_MAX = 1.5          # largest forward phase step across a beat boundary (a beat cut during
                        # contraction legitimately continues ~1 phase into the next beat)

# arm -> (rhythm model, breathing model, frames per slice)
# af_rvr / af_pause are the DISAMBIGUATION pair. `af` draws R-R around 1.0, so a short beat is cut
# off mid-cycle and a long one parks at full ED -- two faces of one mechanic, measured at
# r(cutoff rate, hold_frac) = -0.50, so `af` alone can never say which of them costs EF. These two
# split them by bounding the draw on one side only:
#   af_rvr   every beat <= 1.0 -> the trigger always arrives before the cycle finishes, so beats
#            are cut mid-contraction and resume from a partly-contracted state. AF with RAPID
#            ventricular response. Measured: 0.88 cut-offs/slice, hold_frac 0.071.
#   af_pause every beat >= 1.03 -> traversing all T phases takes 1.0 s at the fixed rate, so every
#            beat COMPLETES before the next trigger and nothing is ever cut; the surplus is spent
#            parked at full ED. AF with SLOW ventricular response / pauses. Measured: 0.00
#            cut-offs/slice, hold_frac 0.245.
# Both are real AF phenotypes, not synthetic extremes. See docs/113.
ARMS = {
    "regular_frozen": ("regular", True,  T),
    "regular":        ("regular", False, T),
    "hrv":            ("hrv",     False, T),
    "af":             ("af",      False, T),
    "af_rvr":         ("af_rvr",   False, T),
    "af_pause":       ("af_pause", False, T),
    # 24-frame (2 nominal beats) arms -- the final paper run. Same rhythms, NEW cohort names so the
    # 12-frame bundles that docs/111-112 rest on are never overwritten. 24 frames is the lever the
    # rhythm parameters are not: it hands Fetal CMR 4D TWO images per output bin per slice instead
    # of one, so its uniform-ramp gate averages them. Under a regular rhythm those two are exactly
    # one beat apart (same cardiac phase -> denoising); under af they are 2.04-4.24 of 12 phases
    # apart (-> temporal blur). Measured bin_phase_sd 37.2deg -> 59.7deg with 0% empty bins, versus
    # +2deg for any rhythm-parameter change. The baseline gets MORE data, not less.
    # `numcardphase` stays T=12: it is the output bin count, and letting it follow the frame count
    # would give every frame its own bin and silently delete the whole mechanism.
    "regular_frozen24": ("regular", True,  2 * T),
    "regular24":        ("regular", False, 2 * T),
    "hrv24":            ("hrv",     False, 2 * T),
    "af24":             ("af",      False, 2 * T),
    # 48-frame (4 nominal beats) AF: the frames-per-slice PROBE (tools/cinevol_nframes_probe.py),
    # built only under temp/, never into scratch/eval. Per-plane RNG streams do not depend on
    # n_frames, so frames 0..23 are bit-identical to af24 -- the probe asserts this.
    "af48":             ("af",      False, 4 * T),
}

# Rhythms solved with pos_physio (cut-off / hold / volume-matched resume) rather than the uniform
# stretch of pos_compress. Keyed here, not by `== "af"`, so a new arm cannot silently fall through
# to the wrong position model.
AF_RHYTHMS = ("af", "af_rvr", "af_pause")


# ───────────────────────── R-R sequences (units of nominal R-R) ─────────────────────────
def _iid_gauss(sd, n, rng, mu=1.0, lo=None, hi=None):
    """iid per-cycle Gaussian factor (the form used by Furnrohr & Heckel 2026). Out-of-range draws
    are REDRAWN, not clipped, so no probability mass piles up at the bound. The lower bound is
    load-bearing: a beat shorter than systole (~0.45) is unphysiological (refractory period).

    `mu`/`lo`/`hi` default to the shipped behaviour (mean 1.0, [RR_MIN, RR_MAX]), so `regular`,
    `hrv` and `af` draw EXACTLY as before -- verified by the regular_frozen bit-exactness check.
    The disambiguation arms override them to bound the draw on one side.
    """
    lo = RR_MIN if lo is None else lo
    hi = RR_MAX if hi is None else hi
    rr = rng.normal(mu, sd, n)
    bad = (rr < lo) | (rr > hi)
    while bad.any():
        rr[bad] = rng.normal(mu, sd, int(bad.sum()))
        bad = (rr < lo) | (rr > hi)
    return rr


def rr_sequence(rhythm, n, rng):
    if rhythm == "regular":
        return np.ones(n)
    if rhythm == "hrv" and HRV_MODEL == "precedent":            # Furnrohr & Heckel 2026, App. A.6:
        return _iid_gauss(0.15, n, rng)                         # iid factor ~ N(1.0, 0.15) per cycle
    if rhythm == "hrv":                                         # Jansz 2010 walk (~+/-4% overall)
        s, d = 0.0175, 0.1
        rr = np.empty(n)
        prev = 1.0 + rng.normal(0, s / np.sqrt(1 - (1 - d) ** 2))
        for i in range(n):
            prev = prev + d * (1.0 - prev) + rng.normal(0, s)
            rr[i] = prev
        return rr
    if rhythm == "af_rvr":                                      # every beat cut off mid-cycle
        # mu=0.60 (100 bpm), not the first cut at 0.80: MEASURED, short beats cannot lower phase
        # coverage (bins_cov 9.43-9.70 for every candidate, all ABOVE regular_frozen's 9.31, which
        # does zero EF damage), because a short beat starts a NEW beat and keeps sampling new
        # phases -- only a hold re-samples the same one. So this arm can only ever test "do
        # cut-offs cost EF at CONSTANT coverage", and the one thing tuning buys is cut-off rate.
        # At 0.60 that is 1.10 cut-offs/slice (2.5x `af`) with hold_frac 0.054 (2.4x BELOW `af`),
        # so a null here means cut-offs are harmless rather than "the arm was too mild".
        return _iid_gauss(0.10, n, rng, mu=0.60, hi=0.85)
    if rhythm == "af_pause":                                    # every beat completes, then parks
        # mu=1.20, not 1.30: MEASURED, 1.30 drives hold_frac to 0.229 but makes 14/38 subjects
        # DEGENERATE (4-7 consecutive frozen target frames -> duplicate GT -> excluded), which is
        # both the unphysiological freeze this project is trying to avoid and a cut of the paired
        # set to n=24. At 1.20 the hold dose is still 1.35x the shipped `af` arm (0.177 vs 0.131)
        # with zero cut-offs, while the degenerate rate falls to 5/38 -- exactly `af`'s own rate,
        # so the hold arm is no more artifact-prone than the arm already published.
        return _iid_gauss(0.15, n, rng, mu=1.20, lo=1.03)
    return _iid_gauss(0.25, n, rng)                             # af: Markl 2015, 161/643


# ───────────────────────── time -> cardiac position (float, in GT frames) ─────────────────────────
def pos_compress(times, rr):
    """The precedent's uniform stretch: the whole beat is played faster/slower."""
    starts = np.concatenate([[0.0], np.cumsum(rr)])
    b = np.searchsorted(starts, times, side="right") - 1
    return (times - starts[b]) / rr[b] * T, b


class DegenerateLV(ValueError):
    """This subject's LV curve cannot drive the AF model. main() skips the subject, not the batch."""


def physio_beats(rr, vol):
    """The AF model's beat bookkeeping -> (realised rr, p_start per beat, es).

    Every beat plays at the stored cine's rate, contraction AND expansion. A LONG beat finishes
    its cycle and then HOLDS at full relaxation until the next trigger (diastasis absorbs the
    surplus cycle length -- Chung 2004, PMID 15217800). A SHORT one is cut off, and the beat after
    it resumes from where the heart actually is: the phase reached if it was cut during
    contraction, else the systolic frame whose whole-LV volume matches the volume reached (a
    weaker beat).

    The hold is capped at HOLD_MAX by shortening the beat itself, so the REALISED rr differs from
    the draw on the clipped beats; callers must build their timeline from the returned rr.
    Idempotent: feeding the realised rr back in returns it unchanged.

    History (docs/114): the first model also held, uncapped, and at 12 frames left P004 with 11 of
    12 target frames byte-identical. It was replaced by stretching expansion across the long beat,
    which froze late-ES subjects instead (12/180 at 24 frames, T-es phases spread over ~1 s) and
    has no physiological support. The same rewrite clamped a beat cut during contraction forward
    to `es`, teleporting the heart up to 5 phases (36/180).
    """
    es = int(np.argmin(vol))
    if not 1 <= es <= T - 2:
        raise DegenerateLV(f"degenerate LV curve: ES at index {es}")
    vol_p = np.r_[vol, vol[0]]                                   # periodic, index T == phase 0
    sys_v = np.minimum.accumulate(vol[:es + 1])                  # enforce monotone-decreasing systole
    rr = np.array(rr, float)
    p_start = np.zeros(len(rr))
    for i in range(len(rr)):
        # A beat's OWN nominal duration: contraction p_start->es plus expansion es->T, both at the
        # stored cine's rate. A resumed beat has less contraction left, so its nominal is shorter.
        t_c = max(es - p_start[i], 0.0) * DT
        t_nom = t_c + (T - es) * DT
        rr[i] = min(rr[i], t_nom + HOLD_MAX)
        if i + 1 == len(rr):
            break
        if rr[i] <= t_c:                        # cut during contraction: already a systolic phase
            p_start[i + 1] = p_start[i] + rr[i] / DT
            continue
        p_end = min(es + (rr[i] - t_c) / DT, float(T))
        vv = np.interp(p_end, np.arange(T + 1), vol_p)
        p_start[i + 1] = 0.0 if vv >= sys_v[0] else np.interp(vv, sys_v[::-1], np.arange(es + 1)[::-1])
    return rr, p_start, es


def pos_physio(times, rr, vol):
    """Cardiac position under the AF model (see physio_beats). `rr` must be the REALISED sequence.

    vol: (T,) whole-LV volume per GT phase. Returns pos in [0, T] (T == phase 0) and beat idx.

    Positions are in STORED phase units, with a beat running 0 -> T. That convention is
    load-bearing: `pos_compress` uses it too, and `simulate`'s `t0` encodes the bundle's own
    per-plane roll in those units. Do not re-express this in an ED-anchored frame -- an earlier
    attempt to roll the curve so "full" meant the true ED frame shifted the whole af timeline by
    `ed` relative to every other arm (measured on P046, ed=11: off by exactly 11), silently
    breaking the control for the very subjects it meant to help.

    `t0` equalises the window's opening TIME, not its opening PHASE. Under `regular`/`hrv`
    (pos_compress) the stored phase at t0 is roll_frac * T whatever the beat lengths, so those
    arms do open at the same phase. Under an AF arm the beat playing at t0 has already been cut
    and volume-matched-resumed during the BURN-in, so it opens elsewhere -- measured on
    MNMs_L8N7Z0's mid plane: 2.00 for regular/hrv, 4.53 for af. That is the burn-in working as
    intended (the window must not always open on a fresh beat), not a control defect.

    KNOWN LIMITATION, disclosed rather than fixed: `sys_v[0] = vol[0]` is taken as "full". The
    stored cine is ED-first by convention but not always -- the seg-derived ED is frame 10/11 on
    4 of the 38 cmrx2024 test subjects and ~14% of CMRx25 (docs/105 s5d). For those, "full" is
    ~96-99% of the true maximum, so the volume-matched resume measures against a slightly low
    reference and a completed beat holds a hair short of true ED. A ~1-frame effect on ~10% of
    subjects, versus a whole-arm misalignment if corrected by rolling. Revisit only with a scheme
    that keeps the stored-phase time origin.
    """
    rr_real, p_start, es = physio_beats(rr, vol)
    assert np.array_equal(rr_real, rr), "pos_physio needs the realised rr (physio_beats(rr, vol)[0])"
    starts = np.concatenate([[0.0], np.cumsum(rr)])
    b = np.searchsorted(starts, times, side="right") - 1
    el = times - starts[b]
    pos = np.empty(len(times))
    for k in range(len(times)):
        p0 = p_start[int(b[k])]; e = el[k]
        t_c = max(es - p0, 0.0) * DT
        if e <= t_c:                                    # CONTRACTION at the stored rate
            pos[k] = p0 + e / DT
        else:                                           # EXPANSION at the stored rate, then HOLD at T
            pos[k] = min(es + (e - t_c) / DT, float(T))
    return pos, b


# ───────────────────────── rendering ─────────────────────────
def render_phase(gt, p, blend):
    """The heart at fractional cardiac position `p` (in GT-frame units). ONE definition, used for
    BOTH the input slices and the GT volumes -- so the reference plane of GT[f] is exactly the
    reference slice's input frame f (modulo breathing). Do not inline this."""
    p = p % T
    if not blend:
        return gt[int(np.round(p)) % T]
    lo = int(np.floor(p))
    w = p - lo
    return gt[lo % T] * (1 - w) + gt[(lo + 1) % T] * w


# ───────────────────────── subject I/O ─────────────────────────
def load_subject(d, need_vol):
    man = json.load(open(os.path.join(d, "manifest.json")))
    gt = np.stack([np.asarray(nib.load(os.path.join(d, "gt", f"gt_t{t:02d}.nii.gz")).dataobj,
                              dtype=np.float32).transpose(2, 1, 0) for t in range(T)])  # (T,D,H,W)
    segs = sorted(glob.glob(os.path.join(d, "seg_gt", "seg_t*.nii.gz")))
    vol = None
    if len(segs) == T:
        vol = np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in segs], float)
        if vol.max() <= 0:
            vol = None
    if need_vol and vol is None:
        raise SystemExit(f"{d}: the af arm needs a usable seg_gt/ LV curve (found {len(segs)} segs) "
                         f"-- run ef_dice.py dump --gt-only + run_seg.sh for this subject first")
    return man, torch.from_numpy(gt), vol


def breathing_model(man):
    """Recover the subject's fixed breathing direction + amplitude from the manifest's stored
    per-plane displacement draw, so the simulated breathing IS the bundle's frozen realization."""
    disp = np.asarray(man["breath"]["disp_dhw_mm"], float)       # (D,3)
    r0 = np.asarray(man["breath"]["r_per_plane"], float)         # (D,)
    n = int(man["breath"]["config"]["cos2n"])
    norm = np.linalg.norm(disp, axis=1)
    k = int(np.argmax(norm))
    u = disp[k] / norm[k]                                        # unit direction (one per subject)
    amp = norm[k] / float(lujan_displacement(float(r0[k]), 1.0, n=n))
    # Recovering ONE direction + ONE amplitude from the max-norm plane is valid only on
    # respiratory.py's group_by_burst branch with zero tidal jitter, where both are per SUBJECT.
    # On the per-slot branch amplitude is drawn per slot, so this would be wrong for every other
    # plane (up to 2.3x over the shipped 11.45-26.15 mm range). check 1 cannot catch it, because
    # regular_frozen uses the stored disp0[z] directly and never touches `amp` -- so assert the
    # recovery instead, on every plane.
    resid = float(np.abs(u * amp * np.array([float(lujan_displacement(float(r), 1.0, n=n))
                                             for r in r0])[:, None] - disp).max())
    if resid > 1e-3:
        raise SystemExit(
            f"breathing_model: per-plane recovery residual {resid:.3g} mm > 1e-3. The stored "
            f"displacement is not one subject-level (direction, amplitude) pair -- check "
            f"data.augmentation.respiratory.{{group_by_burst, amplitude_breath_jitter, per_slot}} "
            f"in the source bundle's breath.config.")
    return u, amp, r0, n, disp


def simulate(man, gt, vol, rhythm, seed, frozen_breath, n_frames=T):
    """-> frames (n_frames, D, H, W) float32, info list (one dict per plane, EXACT positions)."""
    D = gt.shape[1]
    spacing = (float(man["dz_mm"]), INPLANE_MM, INPLANE_MM)
    u, amp, r0, n, disp0 = breathing_model(man)
    roll = man["rolled"]["roll_per_plane"]
    blend = rhythm != "regular"                     # regular lands on integers anyway
    out = torch.zeros((n_frames, D) + tuple(gt.shape[2:]))
    info = []
    for z in range(D):
        # One private stream per (subject, plane), always drawing the same N_BEATS: a slice's beats
        # must not depend on n_frames or on how many other slices came before it. Any change that
        # makes the draw depend on either re-introduces a bug that was found and fixed in design.
        rng = np.random.default_rng([seed, z])
        rr_drawn = rr_sequence(rhythm, N_BEATS, rng)
        # The hold cap shortens beats, so the timeline (t0, starts) must use the REALISED rr.
        rr = physio_beats(rr_drawn, vol)[0] if rhythm in AF_RHYTHMS else rr_drawn
        assert rr[BURN:].sum() > (n_frames + T) * DT, "raise N_BEATS"
        # The window opens inside beat BURN, not beat 0: without burn-in the timeline's first beat
        # always starts from a full heart, so the first beat in EVERY window would be a normal one
        # -- exactly where a weak/held beat matters most.
        # Start fraction = the bundle's own per-plane roll => a periodic rhythm reproduces rolled/.
        t0 = rr[:BURN].sum() + ((-roll[z]) % T) / T * rr[BURN]
        times = t0 + np.arange(n_frames) * DT
        pos, beat = (pos_physio(times, rr, vol) if rhythm in AF_RHYTHMS
                     else pos_compress(times, rr))
        info.append({"z": z, "pos": pos.tolist(), "beat": beat.tolist(), "rr": rr.tolist(),
                     "rr_drawn": rr_drawn.tolist()})
        for f in range(n_frames):
            V = render_phase(gt, pos[f], blend)
            if frozen_breath:
                d = disp0[z]
            else:
                r = (r0[z] + f * DT / T_BREATH) % 1.0
                d = u * amp * float(lujan_displacement(float(r), 1.0, n=n))
            out[f, z] = reslice_volume_vec(V, d, spacing=spacing)[z]
    return out.numpy(), info


def physio_postconditions(info, vol):
    """Hard build-time checks on EVERY plane's trajectory -> stats dict for the manifest.

    check() only ever looked at the reference plane's GT targets, which is how a freeze and a
    teleport on the INPUT planes both shipped (docs/114). Under the hold rule a held frame sits at
    exactly float(T), so both tests are exact -- no tolerance to tune.
    """
    holds, clipped, n_beats = [], 0, 0
    for pl in info:
        pos, beat = np.asarray(pl["pos"]), np.asarray(pl["beat"])
        rr, rr_drawn = np.asarray(pl["rr"]), np.asarray(pl["rr_drawn"])
        run = longest = 0
        for at_T in pos == float(T):
            run = run + 1 if at_T else 0
            longest = max(longest, run)
        if longest > HOLD_MAX_FRAMES:
            raise SystemExit(f"plane {pl['z']}: {longest} consecutive held frames > {HOLD_MAX_FRAMES}")
        jump = np.diff(pos)[np.diff(beat) != 0]
        if (jump > JUMP_MAX).any():
            raise SystemExit(f"plane {pl['z']}: forward jump of {jump.max():.2f} phases across a "
                             f"beat boundary > {JUMP_MAX}")
        _, p_start, es = physio_beats(rr, vol)
        win = np.unique(beat)
        t_nom = (np.maximum(es - p_start, 0.0) + (T - es)) * DT
        holds += np.maximum(rr - t_nom, 0.0)[win].tolist()
        clipped += int((rr < rr_drawn)[win].sum())
        n_beats += len(win)
    return {"max_hold_ms": 1000 * max(holds), "beats_clipped": clipped, "beats_in_window": n_beats,
            "rr_sd_drawn": float(np.std(np.concatenate([pl["rr_drawn"] for pl in info]))),
            "rr_sd_realised": float(np.std(np.concatenate([pl["rr"] for pl in info]))),
            "note": "ms assume a 1 s nominal R-R; sd over all N_BEATS of every plane"}


# ───────────────────────── writer ─────────────────────────
# Pure geometry: unaffected by which cardiac phase a frame shows, so copied verbatim.
# heart_seg.nii.gz is deliberately NOT here -- it is PHASE-INDEXED against the source GT order,
# which no longer holds. Copying it would let dice_floors.py (:61) pair the wrong phase and emit a
# silently wrong floor. Under an integer permutation it is permuted with the GT; otherwise it is
# omitted and recorded in heart_siblings_missing so a consumer fails loudly instead of quietly.
SIBLINGS = ("mask.nii.gz", "mask_fov.nii.gz", "mask_heart.nii.gz", "mask_heart_pad10.nii.gz")


def _sha(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _save_dhw(dhw, affine, path):
    """(D,H,W) splat order -> (X,Y,Z) on disk. Inverse of load_subject's transpose."""
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.asarray(dhw, dtype=np.float32)
                                                  .transpose(2, 1, 0)), affine), path)


def build(src_dir, dst_dir, arm, source, overwrite):
    """-> status string. Writes one (subject, arm) cohort dir."""
    # dst_dir is <eval_root>/<cohort>/out/<subject>; the guard is on the COHORT component.
    cohort = os.path.basename(os.path.dirname(os.path.dirname(dst_dir.rstrip("/"))))
    if not any(cohort.endswith("_" + a) for a in ARMS):
        raise SystemExit(f"refusing to write to cohort '{cohort}' (from {dst_dir}): it must end in "
                         f"one of {sorted('_' + a for a in ARMS)} -- this guard exists so a typo "
                         f"can never write into a real bundle")
    if os.path.exists(os.path.join(dst_dir, "manifest.json")) and not overwrite:
        return "skipped"
    rhythm, frozen, nf = ARMS[arm]
    man, gt, vol = load_subject(src_dir, need_vol=(rhythm in AF_RHYTHMS))
    D = gt.shape[1]
    seed = int(man["seed"])
    affine = nib.load(os.path.join(src_dir, "gt", "gt_t00.nii.gz")).affine

    frames, info = simulate(man, gt, vol, rhythm, seed, frozen_breath=frozen, n_frames=nf)
    physio = physio_postconditions(info, vol) if rhythm in AF_RHYTHMS else None   # before any write
    ref = int(man["scatter"]["ref_plane"])
    pos = np.asarray(info[ref]["pos"], float)          # the nf target positions
    blend = rhythm != "regular"

    os.makedirs(os.path.join(dst_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "breath"), exist_ok=True)

    # --- input stacks -------------------------------------------------------------------
    for f in range(nf):
        _save_dhw(frames[f], affine, os.path.join(dst_dir, "breath", f"stack_t{f:02d}.nii.gz"))

    # rolled/ == breath/ (the unknown per-slice start phase is already baked into the rhythm).
    # A symlink, so fetal4d_gate.py's hardcoded rolled/ needs no change and costs no bytes.
    link = os.path.join(dst_dir, "rolled")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink("breath", link)

    # --- per-target GT ------------------------------------------------------------------
    # Integer position => GT is literally one of the source files: COPY it, so the permutation is
    # byte-identical rather than merely numerically equal (and re-runs are trivially idempotent).
    gt_map = []
    for f in range(nf):
        p = float(pos[f]) % T
        dst = os.path.join(dst_dir, "gt", f"gt_t{f:02d}.nii.gz")
        k = int(np.round(p))
        if abs(p - k) < 1e-9:
            shutil.copyfile(os.path.join(src_dir, "gt", f"gt_t{k % T:02d}.nii.gz"), dst)
            gt_map.append(k % T)
        else:
            _save_dhw(render_phase(gt, p, blend), affine, dst)
            gt_map.append(None)

    # --- geometry siblings (unchanged by the rhythm) ------------------------------------
    for name in SIBLINGS:
        p = os.path.join(src_dir, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(dst_dir, name))

    # --- seg_gt: only when every target is an existing phase ----------------------------
    # The cache is segmentations OF THE GT. A blended GT needs fresh nnU-Net segs, so we write
    # nothing rather than a stale cache; ef_dice's content-keyed freshness check would reject it
    # anyway, but an absent cache is unambiguous.
    seg_note = "regenerate: ef_dice.py dump --gt-only + run_seg.sh"
    missing = list(man.get("heart_siblings_missing", []))
    integral = all(k is not None for k in gt_map)

    # heart_seg.nii.gz: (T,...) phase-indexed. Permute it with the GT when every target is an
    # existing phase; otherwise omit it and say why.
    hs = os.path.join(src_dir, "heart_seg.nii.gz")
    # On an --overwrite rebuild an arm can flip integral -> blended (a different seed, or a
    # regenerated source seg_gt changing the af trajectory). Clear the phase-indexed siblings
    # first, or a stale permuted copy survives next to a blended GT and dice_floors.py pairs the
    # wrong phase -- exactly what omitting them is meant to prevent.
    for stale in (os.path.join(dst_dir, "heart_seg.nii.gz"), os.path.join(dst_dir, "seg_gt")):
        if os.path.isdir(stale):
            shutil.rmtree(stale)
        elif os.path.exists(stale):
            os.remove(stale)
    if os.path.exists(hs):
        if integral:
            im = nib.load(hs)
            a = np.asarray(im.dataobj)
            if a.ndim == 4 and a.shape[3] == T:
                nib.save(nib.Nifti1Image(np.ascontiguousarray(a[..., gt_map]), im.affine),
                         os.path.join(dst_dir, "heart_seg.nii.gz"))
            else:                      # not phase-indexed after all -> copy unchanged
                shutil.copyfile(hs, os.path.join(dst_dir, "heart_seg.nii.gz"))
        else:
            missing.append("heart_seg.nii.gz (phase-indexed; invalid once GT targets are blended)")

    src_segs = sorted(glob.glob(os.path.join(src_dir, "seg_gt", "seg_t*.nii.gz")))
    have_seg_gt = len(src_segs) == T and os.path.isfile(os.path.join(src_dir, "seg_gt", "src.json"))
    if not integral:
        missing.append("seg_gt (GT targets are blended; regenerate with ef_dice.py dump --gt-only)")
    elif not have_seg_gt:
        # Distinguish "blended, so a cache is invalid" from "the source simply has no cache".
        # Also stops a partial seg_gt/ from raising mid-write and abandoning the whole build.
        missing.append(f"seg_gt (source has {len(src_segs)}/{T} segs; nothing to permute)")
    if integral and have_seg_gt:
        sd = os.path.join(dst_dir, "seg_gt")
        os.makedirs(sd, exist_ok=True)
        for f, k in enumerate(gt_map):
            shutil.copyfile(os.path.join(src_dir, "seg_gt", f"seg_t{k:02d}.nii.gz"),
                            os.path.join(sd, f"seg_t{f:02d}.nii.gz"))
        src_json = json.load(open(os.path.join(src_dir, "seg_gt", "src.json")))
        json.dump({"gt_sha256": _sha(os.path.join(dst_dir, "gt", "gt_t00.nii.gz")),
                   "roi_sha256": src_json["roi_sha256"], "T": nf, "crop": src_json.get("crop"),
                   "note": f"permuted copy of {source}'s seg_gt: seg_t{{f}} = source seg_t{{gt_map[f]}}"},
                  open(os.path.join(sd, "src.json"), "w"), indent=1)
        seg_note = "permuted copy of the source cache"

    # --- manifest -----------------------------------------------------------------------
    # `source` keeps the BASE name so run_baselines.thickness_mm(m["source"]) resolves, and so the
    # add_scatter/add_rolled seed assertions stay meaningful.
    out = {k: man[k] for k in ("source", "subject", "rel_path", "split_file", "split", "seed",
                               "seed_basis", "T", "D", "H", "W", "spacing_xyz_mm", "dz_mm",
                               "content_mask_frac", "breath", "rolled")
           if k in man}
    out["cohort"] = f"{source}_{arm}"
    out["heart_siblings_missing"] = missing
    # Every consumer (ef_dice, image_metrics, run_vggt, fetal4d_gate) reads manifest["T"] as "how
    # many phase volumes does this bundle have", which is now the FRAME count. The nominal beat
    # length in frames -- fetal's -numcardphase, and the modulus of the gate's uniform ramp -- is a
    # separate number and stays at the stored cine's T. Conflating them deletes the 2-images-per-bin
    # mechanism, so they are written as two explicit keys rather than one reused one.
    out["T"] = nf
    out["n_cardphase"] = T
    roll = man["rolled"]["roll_per_plane"]
    ppp = man["scatter"]["phase_per_plane"]
    # Remap the frozen companion draw from PHASE indices to FRAME indices, so plane z still shows
    # the cardiac phase it shows today: frame j of plane z is at phase (j - roll_z) mod T, so
    # j = (ppp[z] + roll_z) mod T. This makes regular_frozen a true replication of the source
    # scatter input, leaving the rhythm as the only delta. (pin_scatter skips slot 0.)
    # Which BEAT each companion plane's single frame is taken from. VGGT reads exactly one frame
    # per slice; with nf > T that frame must be allowed anywhere in the acquisition, or VGGT's
    # scatter spans 1 s while the baselines integrate 2 s of the same slices. Under a PERIODIC
    # rhythm the cardiac phase is untouched (j and j+T are the same stored phase) and only the
    # respiratory state changes (it advances every frame by DT/T_BREATH). Under hrv/af that is
    # FALSE: frames j and j+T sit at different cardiac positions, so the beat draw also changes
    # which phase the companion plane shows. Its own RNG stream, so adding it cannot
    # perturb the per-plane rhythm draws; for nf == T it is all zeros and the arm is unchanged.
    n_beats_win = nf // T
    beat = (np.random.default_rng([seed, 991]).integers(0, n_beats_win, D)
            if n_beats_win > 1 else np.zeros(D, dtype=int))
    out["scatter"] = {**man["scatter"],
                      "phase_per_plane": [int((ppp[z] + roll[z]) % T + T * beat[z])
                                          for z in range(D)],
                      "phase_per_plane_source": list(map(int, ppp)),
                      "beat_per_plane_scatter": [int(b) for b in beat],
                      "note": "FRAME index per plane (not cardiac phase): stack_t{j}[z] is that "
                              "slice's j-th real-time frame. Remapped from the source bundle's "
                              "phase draw by j = (phase + roll_z) mod T + T*beat_z."}
    out["rhythm"] = {
        "arm": arm, "rhythm": rhythm, "frozen_breath": frozen, "seed": seed,
        "ref_plane": ref, "ref_pos": pos.tolist(),        # <- the oracle-timing file
        "gt_map": gt_map,
        # Frames sharing a cardiac position: under `af` a beat longer than nominal reaches full ED
        # and HOLDS there, so consecutive camera frames photograph the same state and their GT
        # volumes are byte-identical. Expected, not a bug -- but that target is double-weighted in
        # the per-phase PSNR/SSIM/NCC mean (EF is unaffected: (max-min)/max ignores repeats).
        # On a 24-frame arm this list is LONG BY DESIGN and means something different: under
        # `regular` frames f and f+12 are the same stored phase, so every phase pairs up and all 12
        # pairs appear here. That is the whole point of the design (two images per output bin), NOT
        # a degenerate window -- so the old run-length exclusion rule must NOT be applied to these
        # arms. The degeneracy signal remains window LV-excursion coverage (docs/110 s11.8).
        "duplicate_targets": [[int(i) for i in np.where(np.abs((pos % T) - p) < 1e-9)[0]]
                              for p in sorted({round(float(x) % T, 9) for x in pos})
                              if int((np.abs((pos % T) - p) < 1e-9).sum()) > 1],
        "pos_per_plane": [i["pos"] for i in info],
        "rr_per_plane": [i["rr"] for i in info],           # REALISED (after the hold cap)
        "rr_drawn_per_plane": [i["rr_drawn"] for i in info],
        "physio": physio,
        "beat_per_plane": [i["beat"] for i in info],
        "params": {"T": T, "n_frames": nf, "DT": DT, "T_BREATH": T_BREATH, "N_BEATS": N_BEATS,
                   "BURN": BURN, "HRV_MODEL": HRV_MODEL, "RR_MIN": RR_MIN, "HOLD_MAX": HOLD_MAX,
                   "blend": blend},
        "builder": os.path.basename(__file__), "source_cohort": source,
        "gt": "clean unbreathed volume at the REFERENCE slice's true position for frame f",
        "seg_gt": seg_note,
    }
    json.dump(out, open(os.path.join(dst_dir, "manifest.json"), "w"), indent=1)
    return "built"


# ───────────────────────── checks ─────────────────────────
def check(src_dir, dst_root, subject, source, suffix=""):
    """Verification 1-4 of the plan. Prints PASS/FAIL per check; returns the number of failures.

    `suffix` selects the arm family ("" = the 12-frame arms, "24" = the 2-beat paper arms). Every
    loop below runs to the BUILT arm's own frame count, read from its manifest, so the same checks
    cover both families rather than silently testing only the first beat of a 24-frame bundle.
    """
    fails = 0

    def report(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    def arm_dir(a):
        return os.path.join(dst_root, f"{source}_{a}{suffix}", "out", subject)

    def n_frames_of(a):
        return int(json.load(open(os.path.join(arm_dir(a), "manifest.json")))["T"])

    man = json.load(open(os.path.join(src_dir, "manifest.json")))
    ref = int(man["scatter"]["ref_plane"])
    roll = man["rolled"]["roll_per_plane"]

    # 1. regular_frozen reproduces the source rolled/ stacks bit-exactly.
    #    The source has only T frames, so compare the FIRST beat; 1b covers the rest.
    d = arm_dir("regular_frozen")
    nf = n_frames_of("regular_frozen")
    worst = 0.0
    for k in range(T):
        a = np.asarray(nib.load(os.path.join(d, "breath", f"stack_t{k:02d}.nii.gz")).dataobj,
                       dtype=np.float32)
        b = np.asarray(nib.load(os.path.join(src_dir, "rolled", f"stack_t{k:02d}.nii.gz")).dataobj,
                       dtype=np.float32)
        worst = max(worst, float(np.abs(a - b).max()))
    report("regular_frozen input == source rolled/ (first beat)", worst == 0.0,
           f"max|diff|={worst:.2e}")

    # 1b. Under regular_frozen the rhythm is periodic AND breathing is frozen per plane, so frame f
    #     and frame f+T are the same cardiac phase at the same displacement -> byte-identical. This
    #     is the only check that looks past frame T, and it is what catches an n_frames plumbing bug
    #     that leaves the second beat empty, stale or misindexed.
    if nf > T:
        worst = 0.0
        for k in range(nf - T):
            a = np.asarray(nib.load(os.path.join(d, "breath", f"stack_t{k:02d}.nii.gz")).dataobj,
                           dtype=np.float32)
            b = np.asarray(nib.load(os.path.join(d, "breath",
                                                 f"stack_t{k + T:02d}.nii.gz")).dataobj,
                           dtype=np.float32)
            worst = max(worst, float(np.abs(a - b).max()))
        report("regular_frozen frame f == frame f+T (periodic, frozen breath)", worst == 0.0,
               f"max|diff|={worst:.2e}  nf={nf}")

    # 2. regular* GT is a byte-identical cyclic permutation of the source gt_t*
    for arm in ("regular_frozen", "regular"):
        d = arm_dir(arm)
        ok = True
        for f in range(n_frames_of(arm)):
            k = (f - roll[ref]) % T
            ok &= _sha(os.path.join(d, "gt", f"gt_t{f:02d}.nii.gz")) == \
                _sha(os.path.join(src_dir, "gt", f"gt_t{k:02d}.nii.gz"))
        report(f"{arm} GT == cyclic permutation (sha256)", ok, f"shift={-roll[ref] % T}")

    # 3. hrv/af actually applied an irregular rhythm.
    #    NOT "the GT is blended": a window that happens to contain one long beat starting on an
    #    integer position yields 12 integer positions and a GT identical to regular's (measured on
    #    CMRx24_Test_P017). The falsifiable property is that the RHYTHM differs -- the drawn beat
    #    lengths are not all 1.0, and that difference reaches the pixels on at least one plane.
    reg = arm_dir("regular")
    for arm in ("hrv", "af"):
        d = arm_dir(arm)
        rr = np.asarray(json.load(open(os.path.join(d, "manifest.json")))["rhythm"]["rr_per_plane"])
        differs = max(
            float(np.abs(
                np.asarray(nib.load(os.path.join(d, "breath", f"stack_t{f:02d}.nii.gz")).dataobj,
                           dtype=np.float32)
                - np.asarray(nib.load(os.path.join(reg, "breath", f"stack_t{f:02d}.nii.gz")).dataobj,
                             dtype=np.float32)).max())
            for f in range(n_frames_of(arm)))
        report(f"{arm} rhythm is irregular and reaches the pixels",
               rr.std() > 1e-6 and differs > 0,
               f"rr sd={rr.std():.3f}  max|{arm}-regular|={differs:.3e}")

    # Informational: how much of this subject's LV excursion the af window actually observes, and
    # which targets are duplicated by a full-ED hold. Not pass/fail -- a weak window is a legitimate
    # draw -- but a near-static window makes that subject's EF uninformative, so it must be visible.
    dm = json.load(open(os.path.join(arm_dir("af"), "manifest.json")))
    segs = sorted(glob.glob(os.path.join(src_dir, "seg_gt", "seg_t*.nii.gz")))
    if len(segs) == T:
        vol = np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in segs], float)
        p = np.asarray(dm["rhythm"]["ref_pos"]) % T
        lv = np.interp(p, np.arange(T + 1), np.r_[vol, vol[0]])
        span = (lv.max() - lv.min()) / (vol.max() - vol.min()) * 100
        print(f"  [info] af window sees {span:.0f}% of the LV excursion; "
              f"duplicate targets: {dm['rhythm']['duplicate_targets'] or 'none'}")

    # 4. reference-plane consistency: plane z_ref of GT[f] must reslice onto the reference slice's
    #    input frame f. Run it on regular_frozen (integer positions) AND on hrv, where pos[0] is
    #    fractional -- otherwise the BLENDED GT path, which is the central docs/110 section 7
    #    claim, is never exercised by any check. Always via the re-reslice comparison with a
    #    tolerance: reslice_volume_vec is NOT the identity at zero displacement (grid_sample's
    #    float32 coordinate round-trip leaves ~2e-6), so an exact-equality branch would raise a
    #    spurious FAIL on any subject whose reference plane happens to sit at end-expiration.
    for arm in ("regular_frozen", "hrv"):
        d = arm_dir(arm)
        man_d = json.load(open(os.path.join(d, "manifest.json")))
        nf_a = int(man_d["T"])
        pos_ref = np.asarray(man_d["rhythm"]["ref_pos"], float)
        frozen = bool(man_d["rhythm"]["frozen_breath"])
        disp0 = np.asarray(man_d["breath"]["disp_dhw_mm"], float)[ref]
        u, amp, r0, n, _ = breathing_model(man_d)
        worst = 0.0
        for f in range(nf_a):
            gtf = np.asarray(nib.load(os.path.join(d, "gt", f"gt_t{f:02d}.nii.gz")).dataobj,
                             dtype=np.float32)
            disp = disp0 if frozen else (
                u * amp * float(lujan_displacement(float((r0[ref] + f * DT / T_BREATH) % 1.0),
                                                   1.0, n=n)))
            got = reslice_volume_vec(torch.from_numpy(gtf.transpose(2, 1, 0).copy()),
                                     torch.tensor(np.asarray(disp), dtype=torch.float32),
                                     spacing=(float(man["dz_mm"]), INPLANE_MM, INPLANE_MM))[ref]
            inp = np.asarray(nib.load(os.path.join(d, "breath", f"stack_t{f:02d}.nii.gz")).dataobj,
                             dtype=np.float32)[:, :, ref]
            worst = max(worst, float(np.abs(got.numpy().T - inp).max()))
        blended = int((np.abs(pos_ref - np.round(pos_ref)) > 1e-9).sum())
        report(f"{arm}: ref plane of GT[f] reslices onto input frame f, all f",
               worst < 1e-5, f"max={worst:.2e}  ({blended}/{nf_a} targets blended)")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="base cohort name, e.g. cmrx2024")
    ap.add_argument("--subjects", required=True, help="comma-separated subject names")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--eval-root", default=os.path.join(ROOT, "scratch/eval"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify an already-built set, write nothing")
    ap.add_argument("--check-suffix", default="", choices=["", "24"],
                    help="which arm family --check verifies ('' = 12-frame arms, '24' = 2-beat arms)")
    a = ap.parse_args()

    subjects = [s.strip() for s in a.subjects.split(",") if s.strip()]
    if a.check:
        bad = 0
        for s in subjects:
            print(f"\n=== {s} ===")
            bad += check(os.path.join(a.eval_root, a.source, "out", s), a.eval_root, s, a.source,
                         suffix=a.check_suffix)
        print(f"\n{'ALL CHECKS PASSED' if not bad else str(bad) + ' CHECK(S) FAILED'}")
        raise SystemExit(1 if bad else 0)

    n = {"built": 0, "skipped": 0}
    for s in subjects:
        src = os.path.join(a.eval_root, a.source, "out", s)
        if not os.path.exists(os.path.join(src, "manifest.json")):
            raise SystemExit(f"no source bundle at {src}")
        for arm in a.arms:
            dst = os.path.join(a.eval_root, f"{a.source}_{arm}", "out", s)
            os.makedirs(dst, exist_ok=True)
            try:
                st = build(src, dst, arm, a.source, a.overwrite)
            except DegenerateLV as e:          # one bad subject must not abort the whole batch
                st = "degenerate"
                print(f"  {s}: {e}", flush=True)
            n[st] = n.get(st, 0) + 1
            print(f"  {s:24} {arm:15} {st}", flush=True)
    print(f"built={n['built']} skipped={n['skipped']}")


if __name__ == "__main__":
    main()
