"""E4 — the mechanism in the paper's own functional metric (LV volume-time curve via nnU-Net Task114).

Per test subject (same companions + breathing draw as E1), reference-phase sweep t = 0..T-1 under:
  full          unmodified
  ko_ref_L3     companions may not attend to ANY slot-0 token in global layer 3 only (1 of 24)
  self_L12_17   no cross-slice attention in global layers 12-17
  identity      Δ = 0 (the naive stack at the same inputs)
and the GT cine. Every volume (all arms AND GT) is multiplied by the SAME padded heart ROI
(heart_roi_canonical dilated 10 mm in-plane and ±ceil(10/dz) planes) and written as
<name>__<arm>__t{tt}_0000.nii.gz for nnU-Net. NOTE: this crop is close to, but NOT identical to, the
paper's ef_dice.py mask (tools/build_padded_heart_mask.py: in-plane only, FOV-clamped, from
mask_heart). E4 numbers are for WITHIN-E4 arm comparisons only, not comparable to the paper tables.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nibabel as nib       # noqa: E402
import numpy as np          # noqa: E402
import torch                # noqa: E402
from scipy.ndimage import binary_dilation   # noqa: E402
import harness as H         # noqa: E402
from e2_breathing import slot_allow         # noqa: E402


def padded_roi(subj):
    roi = subj.heart.cpu().numpy()
    r_in = int(round(10 / 1.4))
    st = np.zeros((1, 2 * r_in + 1, 2 * r_in + 1), bool)
    yy, xx = np.mgrid[-r_in:r_in + 1, -r_in:r_in + 1]
    st[0] = yy ** 2 + xx ** 2 <= r_in ** 2
    roi = binary_dilation(roi, st)
    nz = int(np.ceil(10 / subj.dz))
    roi = binary_dilation(roi, np.ones((2 * nz + 1, 1, 1), bool))
    return roi


def save(vol_dhw, roi, dz, path):
    v = (vol_dhw * roi).astype(np.float32).transpose(2, 1, 0)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(v), np.diag([1.4, 1.4, dz, 1.0])), path)


def run_subject(model, cfg, ds, idx, dev, out):
    subj = H.Subject(ds, idx, dev)
    if subj.heart is None:
        return None
    rng = np.random.default_rng(1000 + idx)                 # SAME draw as E1/E1b/E3/E5
    comp = rng.integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
    roi = padded_roi(subj)
    S = subj.S
    ko3 = torch.ones(2 * S, 2 * S, dtype=torch.bool); ko3[2:, 0:2] = False
    eye = slot_allow(torch.eye(S, dtype=torch.bool))
    tag = f"s{idx:03d}"
    for t in range(subj.T):
        save(subj.phases[t].cpu().numpy(), roi, subj.dz, f"{out}/{tag}__gt__t{t:02d}_0000.nii.gz")
        ts = comp.copy(); ts[0] = t
        b = H.build_batch(subj, ts, disp)
        for arm in ("full", "ko_ref_L3", "self_L12_17", "identity"):
            if arm == "identity":
                wp = b["scanner_coords"].clone()
            elif arm == "full":
                wp = H.forward(model, subj, b)["wp"]
            elif arm == "ko_ref_L3":
                wp = H.forward(model, subj, b, allow=ko3, layers={3})["wp"]
            else:
                wp = H.forward(model, subj, b, allow=eye, layers=set(range(12, 18)))["wp"]
            V, _ = H.splat(subj, b, wp)
            save(V.cpu().numpy(), roi, subj.dz, f"{out}/{tag}__{arm}__t{t:02d}_0000.nii.gz")
    return {"idx": idx, "subject": subj.name, "D": subj.D, "dz": subj.dz, "T": subj.T,
            "ref_plane": subj.ref_plane, "slot_z": subj.slot_z.tolist(), "comp_t": comp.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    idxs = list(range(len(ds.subjects)))[args.shard::args.nshard]
    if args.limit:
        idxs = idxs[: args.limit]
    os.makedirs(args.out + "/in", exist_ok=True)
    os.makedirs(args.out + "/meta", exist_ok=True)
    for i in idxs:
        fp = f"{args.out}/meta/s{i:03d}.json"
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev, args.out + "/in")
        json.dump(rec, open(fp, "w"))
        print(f"[{i}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
