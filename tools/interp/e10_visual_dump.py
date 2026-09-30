"""E10 — full-FOV volumes for the image-based paper figure (no ROI crop, unlike E4).

Same companions + breathing draw as E1/E1b/E4. For each listed subject, the reference is set to ED and
to ES (GT LV-volume extremes from E4), under: full, ko_ref_L3, self_L12_17, identity. Saves one npz per
subject with the (D,H,W) volumes, the GT phases at ED/ES, and the breathing displacements.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402
from e2_breathing import slot_allow         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    e4 = {r["idx"]: r for r in json.load(open(os.path.join(H.ROOT, "scratch/interp/e4/e4_curves.json")))["per_subject"]}
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    for idx in args.idx:
        subj = H.Subject(ds, idx, dev)
        rng = np.random.default_rng(1000 + idx)
        comp = rng.integers(0, subj.T, subj.S)
        disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
        S = subj.S
        ko3 = torch.ones(2 * S, 2 * S, dtype=torch.bool); ko3[2:, 0:2] = False
        eye = slot_allow(torch.eye(S, dtype=torch.bool))
        g = np.array(e4[idx]["gt"]["all"])
        ed, es = int(np.argmax(g)), int(np.argmin(g))
        rec = {"ed": ed, "es": es, "ref_plane": subj.ref_plane, "dz": subj.dz,
               "slot_z": np.asarray(subj.slot_z), "comp_t": comp,
               "disp": disp.cpu().numpy() if torch.is_tensor(disp) else np.asarray(disp)}
        for name, t in (("ed", ed), ("es", es)):
            rec[f"gt_{name}"] = subj.phases[t].cpu().numpy().astype(np.float16)
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
                rec[f"{arm}_{name}"] = V.cpu().numpy().astype(np.float16)
        np.savez_compressed(f"{args.out}/s{idx:03d}.npz", **rec)
        print(f"[{idx}] ed={ed} es={es} D={subj.D}", flush=True)


if __name__ == "__main__":
    main()
