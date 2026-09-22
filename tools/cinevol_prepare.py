"""Rhythm bundle (tools/build_af_bundle.py, docs/110+115) -> CiNeVol input. READS the bundle only.

CiNeVol (Vogt et al. 2026, baselines/cinevol/) is fitted per subject from every acquired frame plus
two scalar states per frame. It never estimates those states; the paper takes them from an ECG and
a respiration belt. Here they come from the SIMULATOR's sensors-equivalent, never from its answer:

  cardiac  phi in [0,1)  the paper's Eq.16 (Feinstein 1997 bilinear systole/diastole phase) applied
                         to the simulated R-PEAK TIMES (= the beat starts). `pos_per_plane`, the
                         heart's true position, is NEVER read here -- it depends on the LV-volume
                         curve (volume-matched resume, hold), which no ECG carries.
  resp     psi in [0,1]  the breath level sin^{2n}(pi r): what a perfect, hysteresis-free belt would
                         report. 0 = end-expiration = the unbreathed GT. The displacement direction
                         and amplitude stay hidden (but see DEVIATIONS.md: one 3-vector per subject).

Query states (one per GT volume f): the reference plane's OWN phi at frame f, psi = 0. Same rule as
the fit -- a label from any other rule addresses a different point of the learned phase axis.

Training pixels are restricted to the bundle's padded heart mask (every other baseline's recon ROI;
the closest single-stack analogue of the paper's multi-stack intersection mask) by cropping to the
mask's bounding box and writing NaN outside it, which cinevol.prepare drops -- its code is untouched.

    PYTHONPATH=baselines/cinevol:training:. python tools/cinevol_prepare.py \
        --dataset cmrx2023_af24 --subject CMRx23_Test_P005 --out /tmp/cinevol_prep/P005
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("evaluation", "evaluation/src/engine", "baselines/cinevol", "training"):
    sys.path.insert(0, os.path.join(ROOT, p))
import paths  # noqa: E402
from run_baselines import thickness_mm  # noqa: E402  (the harness's one slice-THICKNESS rule)

RES_MM = 1.4          # output voxel, isotropic: matches SVRTK / NeSVoR / NiftyMIC / Fetal CMR 4D


# ───────────────────────── beat timeline (mirrors build_af_bundle.simulate) ─────────────────────────
def timeline(man, z):
    """-> (frame times, beat start times, beat index per frame) for plane z, rebuilt from the
    manifest exactly as `simulate` builds it: REALISED rr, window opening inside beat BURN."""
    r, P = man["rhythm"], man["rhythm"]["params"]
    rr = np.asarray(r["rr_per_plane"][z], float)
    roll = man["rolled"]["roll_per_plane"][z]
    starts = np.concatenate([[0.0], np.cumsum(rr)])
    t0 = rr[:P["BURN"]].sum() + ((-roll) % P["T"]) / P["T"] * rr[P["BURN"]]
    times = t0 + np.arange(P["n_frames"]) * P["DT"]
    beat = np.searchsorted(starts, times, side="right") - 1
    return times, starts, beat


def systole_s(rr):
    """Paper Eq.16: systolic duration (s) of a beat from ITS OWN heart rate, 60/rr."""
    return (546.0 - 2.1 * 60.0 / rr) / 1000.0


def cardiac_labels(man):
    """(D, n_frames) Feinstein phase from R-peak times. Record means are per SUBJECT over every
    simulated beat of every plane (a real ECG spans the whole scan, not one slice's window)."""
    rr_all = np.asarray(man["rhythm"]["rr_per_plane"], float)
    s_all = systole_s(rr_all)
    assert (s_all > 0).all() and (rr_all - s_all > 0).all(), "beat shorter than its estimated systole"
    ms, mr = s_all.mean(), rr_all.mean()
    md = mr - ms
    out = []
    for z in range(man["D"]):
        times, starts, beat = timeline(man, z)
        rr = rr_all[z][beat]
        el = times - starts[beat]
        s = systole_s(rr)
        phi = np.where(el < s, el / s * (ms / mr), (el - s) / (rr - s) * (md / mr) + ms / mr)
        out.append(phi)
    out = np.asarray(out)
    assert ((out >= 0) & (out < 1)).all()
    return out


def resp_labels(man):
    """(D, n_frames) breath level in [0,1]; identical to the factor `simulate` multiplies u*amp by."""
    from data.respiratory import lujan_displacement
    r, P = man["rhythm"], man["rhythm"]["params"]
    n = int(man["breath"]["config"]["cos2n"])
    r0 = np.asarray(man["breath"]["r_per_plane"], float)
    f = np.arange(P["n_frames"])
    out = np.empty((man["D"], P["n_frames"]))
    for z in range(man["D"]):
        rz = np.full(P["n_frames"], r0[z]) if r["frozen_breath"] else (r0[z] + f * P["DT"] / P["T_BREATH"]) % 1.0
        out[z] = [float(lujan_displacement(float(x), 1.0, n=n)) for x in rz]
    return np.clip(out, 0.0, 1.0)


# ───────────────────────── geometry ─────────────────────────
def crop_and_grid(mask, affine):
    """Mask bbox (inclusive voxel index ranges) and the RES_MM-isotropic output grid covering it."""
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    rot = affine[:3, :3] / spacing
    extent = (hi - lo + 1) * spacing                       # mm, voxel edge to voxel edge
    shape = np.ceil(extent / RES_MM - 1e-6).astype(int)
    start_mm = (lo - 0.5) * spacing + RES_MM / 2           # first output voxel CENTRE, stack-index mm
    grid = np.eye(4)
    grid[:3, :3] = rot * RES_MM
    grid[:3, 3] = affine[:3, 3] + rot @ start_mm
    return lo, hi, [int(x) for x in shape], grid


def prepare(dataset, subject, out):
    sd = paths.subject_dir(dataset, subject)
    man = json.load(open(sd / "manifest.json"))
    assert "rhythm" in man, f"{dataset}/{subject}: not a rhythm bundle"
    NF, D, ref = man["rhythm"]["params"]["n_frames"], man["D"], man["rhythm"]["ref_plane"]
    thick = thickness_mm(man.get("source", dataset), man["rel_path"], float(man["dz_mm"]))

    imgs = [nib.load(sd / "breath" / f"stack_t{f:02d}.nii.gz") for f in range(NF)]
    affine = imgs[0].affine
    mask_img = nib.load(sd / paths.HEART_MASK_PAD)
    assert np.allclose(mask_img.affine, affine, atol=1e-4) and mask_img.shape == imgs[0].shape
    mask = np.asarray(mask_img.dataobj) > 0
    lo, hi, shape, grid = crop_and_grid(mask, affine)
    sl = tuple(slice(a, b + 1) for a, b in zip(lo, hi))
    cine = np.stack([np.asarray(im.dataobj, dtype=np.float32)[sl] for im in imgs], -1)   # X,Y,Z,frame
    cine[~mask[sl]] = np.nan                                  # cinevol.prepare drops non-finite pixels
    keep = [z for z in range(lo[2], hi[2] + 1)]               # planes inside the bbox (contiguous)
    assert all(mask[:, :, z].any() for z in keep), "mask has an empty plane inside its z-range"
    assert lo[2] <= ref <= hi[2], "reference plane lies outside the heart mask"
    crop_aff = affine.copy()
    crop_aff[:3, 3] = nib.affines.apply_affine(affine, lo)

    phi, psi = cardiac_labels(man), resp_labels(man)
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError(f"{out} is not empty -- never overwrite; choose a new --out")
    os.makedirs(out, exist_ok=True)
    nib.save(nib.Nifti1Image(cine, crop_aff), os.path.join(out, "sax_observed.nii.gz"))
    np.save(os.path.join(out, "cardiac_states.npy"), phi[keep].astype(np.float32))
    np.save(os.path.join(out, "respiratory_states.npy"), psi[keep].astype(np.float32))
    queries = {"cardiac": [float(x) for x in phi[ref]], "respiratory": 0.0, "ref_plane": int(ref),
               "rule": "reference plane's own Feinstein phase at frame f; psi=0 (end-expiration)"}
    json.dump(queries, open(os.path.join(out, "query_states.json"), "w"), indent=1)
    sim = {"name": "vggt_rhythm_bundle", "cohort": dataset, "arm": man["rhythm"]["arm"],
           "builder": man["rhythm"]["builder"], "seed": man["seed"],
           "cardiac_state": "Feinstein bilinear phase (paper Eq.16) from simulated R-peak times; "
                            "record means per subject; true position never read",
           "respiratory_state": "true breath level sin^{2n}(pi r) in [0,1]; 0 = end-expiration",
           "training_pixels": f"inside {paths.HEART_MASK_PAD} (bbox crop + NaN outside)",
           "slice_thickness_mm": thick, "dz_mm": float(man["dz_mm"]), "n_frames": NF}
    spec = {"subject": subject, "protocol": "acdc_single_sax_simulated",   # = cinevol's single-stack switch
            "state_provenance": sim["cardiac_state"] + " | " + sim["respiratory_state"],
            "simulation": sim,
            "adaptations": ["single SAX stack replaces the paper's multi-stack intersection",
                            "training pixels restricted to the padded heart mask",
                            "states from a simulator: noise-free R-peaks, noise-free breath level"],
            "reference_cardiac_state": queries["cardiac"][0], "end_expiration_state": 0.0,
            "output_grid": {"shape": shape, "affine": grid.tolist()},
            "stacks": [{"stack_id": "SAX", "image": "sax_observed.nii.gz", "slice_thickness_mm": thick,
                        "cardiac_states": "cardiac_states.npy", "respiratory_states": "respiratory_states.npy"}]}
    json.dump(spec, open(os.path.join(out, "acquisitions.json"), "w"), indent=1)
    from cinevol.prepare import prepare as cinevol_prepare
    return cinevol_prepare(os.path.join(out, "acquisitions.json"), os.path.join(out, "prepared"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prepare(a.dataset, a.subject, a.out)


if __name__ == "__main__":
    main()
