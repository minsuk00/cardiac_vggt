"""E6 — does the breathing correction work by cross-slice CONTENT MATCHING?

No breathing on any slot except the probe slot k (a companion whose planes k-1 and k+1 are both in
the stack and neither is the reference), shifted through-plane by δ = f * dz for
f in FRACS = {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}. Record, in every global layer, slot k's per-head attention mass
to the slot holding plane z_k+1 ("up", the plane its content moves toward for δ>0), to the slot
holding z_k-1 ("down"), and to itself, from heart-ROI queries. Plus slot k's predicted dz.
If mid-layer attention follows the content (up rises with f) the correction is correspondence-based.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import harness as H         # noqa: E402

FRACS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def run_subject(model, cfg, ds, idx, dev):
    subj = H.Subject(ds, idx, dev)
    if subj.heart is None:
        return None
    z = subj.slot_z
    cands = [s for s in range(1, subj.S) if (z[s] + 1 in z) and (z[s] - 1 in z)
             and z[s] + 1 != subj.ref_plane and z[s] - 1 != subj.ref_plane]
    if not cands:
        return None
    # prefer a probe near the middle of the heart (largest heart area)
    area = [int(subj.heart[int(z[s])].sum()) for s in cands]
    k = cands[int(np.argmax(area))]
    up = int(np.where(z == z[k] + 1)[0][0]); dn = int(np.where(z == z[k] - 1)[0][0])
    rng = np.random.default_rng(5000 + idx)
    ts = rng.integers(0, subj.T, subj.S)
    rec = {"subject": subj.name, "idx": idx, "k": int(k), "up": up, "down": dn, "dz": subj.dz,
           "slot_z": z.tolist(), "rows": []}
    for f in FRACS:
        disp = np.zeros((subj.S, 3), np.float32); disp[k, 0] = f * subj.dz
        b = H.build_batch(subj, ts, disp)
        o = H.forward(model, subj, b, record=range(24))
        st = o["attn"]
        row = {"f": f, "pred_dz": H.slot_dz_mm(subj, b, o["dvf"]).tolist(),
               "mass_k": np.stack([st[l]["mass_h"][:, k, :] for l in range(24)]),  # (24,16,S) heart queries
               "up": np.stack([st[l]["mass_h"][:, k, up] for l in range(24)]),     # (24,16)
               "down": np.stack([st[l]["mass_h"][:, k, dn] for l in range(24)]),
               "self": np.stack([st[l]["mass_h"][:, k, k] for l in range(24)]),
               "ref": np.stack([st[l]["mass_h"][:, k, 0] for l in range(24)])}
        rec["rows"].append(row)
    return rec


def main():
    import argparse
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
    os.makedirs(args.out, exist_ok=True)
    for i in idxs:
        fp = os.path.join(args.out, f"s{i:03d}.pkl")
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev)
        pickle.dump(rec, open(fp, "wb"))
        print(f"[{i}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
