#!/usr/bin/env python
"""Breathing-motion EPE for CiNeVol on a rhythm cohort: the fitted model's OWN respiratory offset
field vs the TRUE applied through-plane shift, on the same input slices VGGT was scored on.

CiNeVol warps every query point  warped = xyz + dc(xyz, phi) + dr(xyz, psi)  (cinevol/model.py,
offsets()). `dr` depends only on the breath label psi (the TRUE breath level, tools/cinevol_prepare
.resp_labels -- an oracle input, hence the row label "psi-conditioned"), so per input slice (z, f)
the model's belief about where that slice's content sits along the stack axis is

    pred_dz[z, f] = mean over the slice's content pixels of  dr(xyz, phi[z, f], psi[z, f]) . z_hat

with z_hat the stack's through-plane unit vector (affine column 2) and xyz the pixel's scanner-mm
position -- exactly the coordinates the fit used (cinevol/prepare.py:80). The truth is the same
expression evaluation/src/analysis/motion_epe/vggt.py scores the VGGT arms against:

    applied_dz[z, f] = u * amp * lujan((r0[z] + f*DT/T_BREATH) % 1)  [through-plane component]

Default = the D scatter slices (manifest["scatter"]["phase_per_plane"][z] is plane z's frame index),
i.e. the one-frame-per-plane input VGGT received, so both rows have D terms per subject on identical
slices. --all-frames scores every (z, f) of the D x n_frames stack CiNeVol was actually fit on.

Every subject is scored. A plane with no heart-mask content pixels (breathing shift sampled beyond
the stack, or the heart mask does not reach that plane) is dropped from CiNeVol's row only, so the
plane sets differ slightly from VGGT's (vggt.py drops only blank input slices); the per-subject dump
has `n_planes`. The error is demeaned per subject (the fit's canonical breathing state is a free
constant, and the applied shifts are relative to an unknown end-expiration too); one GLOBAL sign
is fixed once from the pooled slope (CiNeVol's dr convention vs the sim's) and reported. The
"predict_nothing_<method>" row is pred = 0 on the same planes.

Output (--json) merges into the same {method: {epe, epe_demeaned, slope, corr, applied_abs, n,
skipped}} shape evaluation/src/analysis/motion_epe/vggt.py writes, so tools/rhythm24_ef_table.py
picks the row up.

Usage (needs the CiNeVol env + a GPU: the Grid4D encoder is CUDA-only):
    source baselines/cinevol/env.sh
    $CINEVOL_PY evaluation/src/analysis/motion_epe/cinevol.py af12 [--arm-name cinevol_motion]
        [--all-frames] [--subjects cmrx2024:CMRx24_Test_P001 ...]
        [--json temp/rhythm24_ef/af12_resp_epe.json]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = str(next(p for p in HERE.parents if (p / "evaluation").is_dir()))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools"),
                os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "baselines", "cinevol")]
import paths                                                             # noqa: E402
from build_af_bundle import breathing_model                              # noqa: E402
from cinevol_prepare import cardiac_labels, resp_labels                  # noqa: E402
from data.respiratory import lujan_displacement                          # noqa: E402

VARIANT = "breath"
INTENSITY_GATE = 0.01
SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")


def applied_dz(man):
    """(D, n_frames) true through-plane shift in mm, frame-wise, same formula as rhythm24_resp_epe."""
    u, amp, r0, n, _ = breathing_model(man)
    P = man["rhythm"]["params"]
    f = np.arange(P["n_frames"])
    out = np.empty((man["D"], P["n_frames"]))
    for z in range(man["D"]):
        rz = np.full(P["n_frames"], r0[z]) if man["rhythm"]["frozen_breath"] \
            else (r0[z] + f * P["DT"] / P["T_BREATH"]) % 1.0
        out[z] = [(u * amp * float(lujan_displacement(float(x), 1.0, n=n)))[0] for x in rz]
    return out


@torch.no_grad()
def pred_dz(model, sd, man, pairs, device, chunk=65536):
    """Per (z, f) pair: mean dr . z_hat (mm) over the slice's content pixels (intensity > 0.01) INSIDE
    the fit's footprint, mask_heart_pad10. The footprint gate is load-bearing: both hash encoders
    return 0 outside the fit's bounding box (mask bbox + 20 mm; cinevol/encodings.py `valid`,
    Grid4D hashencoder.cu) and untrained entries inside it, so dr over the whole plane is mostly a
    bias-driven constant (measured: only 23-36 % of a plane's body pixels are in-box, and a plane
    with no mask pixels returned -7.9 mm of pure extrapolation). None for a pair with no such
    pixels -- the model has no information about that plane."""
    phi, psi = cardiac_labels(man), resp_labels(man)
    mask_img = nib.load(sd / paths.HEART_MASK_PAD)
    mask = np.asarray(mask_img.dataobj) > 0
    affine = mask_img.affine
    z_hat = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    z_hat_t = torch.tensor(z_hat, dtype=torch.float32, device=device)
    stacks = {}
    out = []
    for z, f in pairs:
        if f not in stacks:
            im = nib.load(sd / VARIANT / f"stack_t{f:02d}.nii.gz")
            assert np.allclose(im.affine, affine, atol=1e-4)
            stacks[f] = np.asarray(im.dataobj, dtype=np.float32)
        sl = stacks[f][:, :, z]
        ij = np.argwhere(mask[:, :, z] & (sl > INTENSITY_GATE))
        if len(ij) == 0:
            out.append(None)
            continue
        xyz = nib.affines.apply_affine(affine, np.column_stack((ij, np.full(len(ij), z))))
        xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
        acc = 0.0
        for s in range(0, len(xyz), chunk):
            x = xyz[s:s + chunk]
            c = torch.full((len(x),), float(phi[z, f]), device=device)
            r = torch.full((len(x),), float(psi[z, f]), device=device)
            _dc, dr = model.offsets(x, c, r)
            acc += float((dr @ z_hat_t).sum())
        out.append(acc / len(xyz))
    return out


def subject_series(model, ds, subj, arm_name, all_frames, device):
    sd = paths.subject_dir(ds, subj)
    man = json.load(open(sd / "manifest.json"))
    appl = applied_dz(man)
    if all_frames:
        pairs = [(z, f) for z in range(man["D"]) for f in range(appl.shape[1])]
    else:
        # VGGT's slot 0 is the reference plane at the QUERY frame (frame 0 in the pass resp_diag
        # saves), not that plane's scatter frame -- match it so the D slices are identical.
        pairs = [(z, int(f)) for z, f in enumerate(man["scatter"]["phase_per_plane"])]
        pairs[man["rhythm"]["ref_plane"]] = (man["rhythm"]["ref_plane"], 0)
    pred = pred_dz(model, sd, man, pairs, device)
    keep = [i for i, p in enumerate(pred) if p is not None]
    return (np.array([pred[i] for i in keep], float),
            np.array([appl[pairs[i][0], pairs[i][1]] for i in keep], float),
            len(pairs) - len(keep), [pairs[i][0] for i in keep])


def score(pred, appl):
    """Keys of rhythm24_resp_epe.subject_row plus `epe_raw`; `applied_abs` raw. `pred` already
    carries the global sign."""
    err = pred - appl
    dm = float(np.abs(err - err.mean()).mean())
    # `epe` is what tools/rhythm24_ef_table.py reads for the Motion EPE column; for CiNeVol the
    # raw offset is a free constant, so `epe` carries the demeaned value and the raw one is `epe_raw`
    # (for VGGT the two agree to ~0.03 mm, so the column stays comparable).
    row = {"epe": dm, "epe_demeaned": dm, "epe_raw": float(np.abs(err).mean()),
           "applied_abs": float(np.abs(appl).mean()), "n_planes": int(pred.size)}
    if appl.std() > 1e-6 and pred.size >= 2:
        row["slope"] = float(np.polyfit(appl, pred, 1)[0])
        row["corr"] = float(np.corrcoef(appl, pred)[0, 1]) if pred.std() > 1e-9 else 0.0
    return row


def summarise(rows, skipped):
    mean = lambda k: float(np.nanmean([r.get(k, np.nan) for r in rows]))       # noqa: E731
    med = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in rows]))      # noqa: E731
    iqr = lambda k: [float(x) for x in np.nanpercentile([r.get(k, np.nan) for r in rows], [25, 75])]  # noqa: E731
    out = {k: mean(k) for k in ("epe", "epe_demeaned", "epe_raw", "slope", "corr", "applied_abs")}
    out.update({f"{k}_median": med(k) for k in ("epe", "epe_demeaned", "slope", "corr")})
    out.update({f"{k}_iqr": iqr(k) for k in ("epe", "epe_demeaned")})
    out.update(n=len(rows), skipped=skipped)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--arm-name", default="cinevol_motion", help="arm dir holding last.pt per subject")
    ap.add_argument("--all-frames", action="store_true", help="score every (z, f), not only the scatter slices")
    ap.add_argument("--subjects", nargs="+", default=None, metavar="SOURCE:SUBJECT")
    ap.add_argument("--json", default=None, help="merge the rows into this file (rhythm24_resp_epe.py shape)")
    ap.add_argument("--per-subject", default=None, help="also dump per-subject rows + series here")
    ap.add_argument("--slices", default=None, help="per-slice dump for common_set.py")
    a = ap.parse_args()
    from cinevol.fit import model_from_checkpoint
    device = torch.device("cuda")

    if a.subjects:
        work = [tuple(s.split(":")) for s in a.subjects]
        work = [(f"{src}_{a.arm}" if not src.endswith(a.arm) else src, subj) for src, subj in work]
    else:
        work = []
        for src in SOURCES:
            ds = f"{src}_{a.arm}"
            for p in sorted(glob.glob(str(paths.dataset_root(ds) / "*" / a.arm_name / f"recon_{VARIANT}" / "last.pt"))):
                work.append((ds, os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))))
    print(f"[{a.arm_name}/{a.arm}] {len(work)} subjects, {'all frames' if a.all_frames else 'scatter slices (VGGT-matched)'}",
          flush=True)

    series, planes, skipped_pairs, missing = {}, {}, 0, 0
    for k, (ds, subj) in enumerate(work):
        ck = paths.recon_dir(ds, subj, a.arm_name, VARIANT) / "last.pt"
        if not ck.is_file():
            missing += 1
            continue
        model, _state = model_from_checkpoint(str(ck), device)
        pred, appl, skip, zs = subject_series(model, ds, subj, a.arm_name, a.all_frames, device)
        skipped_pairs += skip
        if pred.size >= 2:
            series[f"{ds}/{subj}"] = (pred, appl)
            planes[f"{ds}/{subj}"] = zs
        del model
        if (k + 1) % 20 == 0 or k + 1 == len(work):
            print(f"  {k + 1}/{len(work)} fitted models queried", flush=True)

    # One global sign for the method: the pooled slope's sign (demeaned per subject first).
    if not series:
        sys.exit(f"no scorable subject ({missing} missing checkpoints)")
    pool_p = np.concatenate([p - p.mean() for p, _ in series.values()])
    pool_a = np.concatenate([q - q.mean() for _, q in series.values()])
    if pool_a.std() < 1e-6:
        sys.exit("applied shift is constant across every kept plane -- no slope, no sign, nothing to score")
    sign = 1.0 if np.polyfit(pool_a, pool_p, 1)[0] >= 0 else -1.0
    print(f"global sign = {sign:+.0f} (pooled slope {np.polyfit(pool_a, pool_p, 1)[0]:+.3f} before sign, "
          f"{np.polyfit(pool_a, sign * pool_p, 1)[0]:+.3f} after)")

    rows = {key: score(sign * p, q) for key, (p, q) in series.items()}
    zero = {key: score(np.zeros_like(q), q) for key, (_, q) in series.items()}
    method = a.arm_name + ("_allframes" if a.all_frames else "")
    out = {method: summarise(list(rows.values()), skipped_pairs),
           f"predict_nothing_{method}": summarise(list(zero.values()), skipped_pairs)}   # per-run: its own kept planes
    out[method].update(sign=sign, missing_checkpoints=missing,
                       input="all_frames" if a.all_frames else "scatter")

    print(f"\n{'row':34}{'n':>5}{'EPE':>8}{'demean':>8}{'med[IQR]':>22}{'slope':>8}{'corr':>7}{'|applied|':>11}")
    for m, o in out.items():
        print(f"{m:34}{o['n']:5d}{o['epe']:8.2f}{o['epe_demeaned']:8.2f}"
              f"{o['epe_demeaned_median']:8.2f} [{o['epe_demeaned_iqr'][0]:.2f}, {o['epe_demeaned_iqr'][1]:.2f}]"
              f"{o['slope']:8.2f}{o['corr']:7.2f}{o['applied_abs']:11.2f}")
    if a.slices:
        if a.all_frames:
            sys.exit("--slices needs the scatter slots (one per plane), not --all-frames")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import common                                                      # noqa: E402
        common.write_slices(a.slices, a.arm_name, ["dz"],
                            {k: {int(z): {"applied": [float(q_)], "pred": [float(sign * p_)]}
                                 for z, p_, q_ in zip(planes[k], p, q)} for k, (p, q) in series.items()})
    if a.per_subject:
        json.dump({k: dict(rows[k], pred=(sign * p).tolist(), applied=q.tolist())
                   for k, (p, q) in series.items()}, open(a.per_subject, "w"), indent=1)
        print(f"wrote {a.per_subject}")
    if a.json:
        prev = json.load(open(a.json)) if os.path.isfile(a.json) else {}
        prev.update(out)
        json.dump(prev, open(a.json, "w"), indent=1)
        print(f"wrote {a.json} ({len(prev)} rows)")


if __name__ == "__main__":
    main()
