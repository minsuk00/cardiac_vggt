"""E12 — attention-map gallery: full global-attention maps (all 24 layers, 16 heads) for several query
points per slice, unmodified model, same companion/breathing draw as E1/E3.

Query points (patch grid 37x37) on a near and a far non-reference slice and on the reference:
  lv       heart-ROI centroid (LV cavity)
  edge_l / edge_r   leftmost / rightmost heart-ROI patch on the centroid row (wall / RV side)
  body     8 patches below the centroid (outside the heart, inside the body)
  air      patch (2, 2) (image corner)
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import harness as H         # noqa: E402


def points(subj, b, slot):
    G = subj.R // 14
    hp = H.heart_patch_mask(subj, b)[slot].reshape(G, G).cpu().numpy()
    if not hp.any():
        return {}
    rr, cc = np.nonzero(hp)
    r, c = int(round(rr.mean())), int(round(cc.mean()))
    row = cc[rr == r] if (rr == r).any() else cc
    return {"lv": (r, c), "edge_l": (r, int(row.min())), "edge_r": (r, int(row.max())),
            "body": (min(r + 8, G - 1), c), "air": (2, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid_step", type=int, default=0,
                    help="if >0: instead of the 5 named points, query every k-th heart patch on the near slice")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    for idx in args.idx:
        subj = H.Subject(ds, idx, dev)
        rng = np.random.default_rng(1000 + idx)
        comp = rng.integers(0, subj.T, subj.S)
        disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
        b = H.build_batch(subj, comp, disp)
        z = np.asarray(subj.slot_z)
        near = int(np.argmin(np.abs(z - (subj.ref_plane + 1)) + (np.arange(subj.S) == 0) * 99))
        far = int(np.argmax(np.abs(z - subj.ref_plane)))
        qnames, qp = [], []
        if args.grid_step:
            G = subj.R // 14
            hp = H.heart_patch_mask(subj, b)[near].reshape(G, G).cpu().numpy()
            for r, c in zip(*np.nonzero(hp)):
                if r % args.grid_step == 0 and c % args.grid_step == 0:
                    qnames.append(("near", near, f"g{r}_{c}", int(r), int(c))); qp.append((near, int(r), int(c)))
        else:
            for tag, s in (("ref", 0), ("near", near), ("far", far)):
                for name, (r, c) in points(subj, b, s).items():
                    qnames.append((tag, s, name, r, c)); qp.append((s, r, c))
        o = H.forward(model, subj, b, record=range(24), query_points=qp)
        rec = {"idx": idx, "S": subj.S, "slot_z": z, "ref_plane": subj.ref_plane, "comp_t": comp,
               "disp": disp.cpu().numpy(), "queries": qnames,
               "images": b["images_splat"][0].cpu().numpy().astype(np.float16),
               "maps": {k: v.half().numpy() for k, v in o["maps"].items()}}
        pickle.dump(rec, open(f"{args.out}/s{idx:03d}.pkl", "wb"))
        print(f"[{idx}] {len(qp)} queries, {len(rec['maps'])} maps", flush=True)


if __name__ == "__main__":
    main()
