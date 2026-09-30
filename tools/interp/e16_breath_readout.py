"""E16 — per-slice breathing readout on a rhythm bundle (paper engine), one reference frame.

For each slot: the through-plane breathing shift the engine applied to that plane at the frame it was
sampled at (build_af_bundle's own breathing_model + lujan_displacement), and the model's predicted mean
through-plane displacement (mm, over in-FOV pixels) under full / ko_ref_L3 / self_L12_17.
Same batch path as E15 (run_vggt.prepare_batch). Saves <out>/breath_readout_t{frame}.json.
"""
import argparse
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src"))
sys.path.insert(0, os.path.join(H.ROOT, "tools"))
import run_vggt as rv       # noqa: E402
import build_af_bundle as bab   # noqa: E402
from e2_breathing import slot_allow   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=os.path.join(H.ROOT, "scratch/eval"))
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sd = os.path.join(args.eval_root, args.cohort, "out", args.subject)
    man = json.load(open(os.path.join(sd, "manifest.json")))
    T, D, dz = man["T"], man["D"], man["dz_mm"]
    ppp, ref = man["scatter"]["phase_per_plane"], int(man["scatter"]["ref_plane"])
    u, amp, r0, n, _ = bab.breathing_model(man)

    def applied(z, f):
        r = (r0[z] + f * bab.DT / bab.T_BREATH) % 1.0
        return float((u * amp * float(bab.lujan_displacement(float(r), 1.0, n=n)))[0])   # through-plane (d)

    dev = torch.device("cuda:0")
    model, cfg = H.load("cuda:0")
    bundle = rv.load_bundle(sd, T, "breath")
    with tempfile.TemporaryDirectory() as tmp:
        ds = rv.make_dataset(cfg, man["rel_path"], man.get("split", "test"), tmp)
        seq = rv.name_seed(man.get("source", args.cohort), args.subject)
        batch = rv.prepare_batch(ds, seq, bundle, dev, dz_bundle=dz, scatter=man["scatter"])
    S = batch["timesteps"].shape[1]
    batch["timesteps"][0, 0] = int(args.frame)
    rv._extract(batch, dev)
    z_slots = [int(round(float(v))) for v in batch["slice_indices"][0].tolist()]
    f_slots = [args.frame if z == ref else int(ppp[z]) for z in z_slots]
    ko3 = torch.ones(2 * S, 2 * S, dtype=torch.bool); ko3[2:, 0:2] = False
    eye = slot_allow(torch.eye(S, dtype=torch.bool))
    arms = {"full": (None, None), "ko_ref_L3": (ko3, {3}), "self_L12_17": (eye, set(range(12, 18)))}
    fake = SimpleNamespace(R=int(batch["scanner_coords"].shape[-2]))
    b = {"images": batch["images"], "z_indices": batch["z_indices"], "scanner_coords": batch["scanner_coords"]}
    rec = {"z": z_slots, "frame": f_slots, "applied_mm": [applied(z, f) for z, f in zip(z_slots, f_slots)]}
    for arm, (allow, layers) in arms.items():
        wp = H.forward(model, fake, b, allow=allow, layers=layers)["wp"]
        dzn = (wp - b["scanner_coords"])[0, ..., 2]                         # (S,R,R) normalized z
        pred = []
        for s in range(S):
            m = b["images"][0, s].mean(0) > 0.05
            pred.append(float(dzn[s][m].mean()) * H.Z_HALF_MM)
        rec[arm] = pred
    os.makedirs(args.out, exist_ok=True)
    json.dump(rec, open(os.path.join(args.out, f"breath_readout_t{args.frame:02d}.json"), "w"), indent=1)
    order = np.argsort(z_slots)
    for k in ["applied_mm", "full", "ko_ref_L3", "self_L12_17"]:
        print(f"{k:12s}", np.round(np.asarray(rec[k])[order], 1).tolist())


if __name__ == "__main__":
    main()
