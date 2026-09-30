"""E3 — descriptive global-attention statistics (all 24 global layers, unmodified model).

Per subject (random companions + val-style breathing, same draw as E1): record per layer/head the
slot->slot attention mass from patch queries (all, and heart-ROI queries), the mass on each slot's
special tokens, on the SAME (row,col) patch, and on its 3x3 neighbourhood; plus mean entropy.
Also save full attention maps for a few query points (heart-centroid patch of the reference and of
a companion) for figures. Two input variants: breathing on / off (paired).
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402


def heart_centroid_patch(subj, b, slot):
    hp = H.heart_patch_mask(subj, b)[slot]
    G = subj.R // 14
    if not hp.any():
        return None
    ii = hp.nonzero()[:, 0].float()
    r, c = (ii // G).mean(), (ii % G).mean()
    return int(round(float(r))), int(round(float(c)))


def run_subject(model, cfg, ds, idx, dev, save_maps):
    subj = H.Subject(ds, idx, dev)
    if subj.heart is None:
        return None
    rng = np.random.default_rng(1000 + idx)
    comp = rng.integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
    rec = {"subject": subj.name, "idx": idx, "D": subj.D, "S": subj.S, "dz": subj.dz,
           "slot_z": subj.slot_z.tolist(), "comp_t": comp.tolist(), "disp": disp.cpu().numpy(),
           "stats": {}, "maps": {}}
    for var in ("breath", "clean"):
        b = H.build_batch(subj, comp, disp if var == "breath" else None)
        qp = []
        if save_maps:
            for s in (0, int(np.argmin(np.abs(subj.slot_z - (subj.ref_plane + 1)) + (np.arange(subj.S) == 0) * 99))):
                c = heart_centroid_patch(subj, b, s)
                if c is not None:
                    qp.append((s, c[0], c[1]))
        o = H.forward(model, subj, b, record=range(24), query_points=qp)
        rec["stats"][var] = o["attn"]
        rec["pred_dz"] = H.slot_dz_mm(subj, b, o["dvf"]).tolist() if var == "breath" else rec.get("pred_dz")
        if save_maps:
            rec["maps"][var] = {k: v.half().numpy() for k, v in o["maps"].items()}
            rec["images_" + var] = b["images_splat"][0].cpu().numpy().astype(np.float16)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--maps_every", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    idxs = list(range(len(ds.subjects)))[args.shard::args.nshard]
    if args.limit:
        idxs = idxs[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    for i in idxs:
        fp = os.path.join(args.out, f"s{i:03d}.pkl")
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev, save_maps=(i % args.maps_every == 0))
        pickle.dump(rec, open(fp, "wb"))
        print(f"[{i}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
