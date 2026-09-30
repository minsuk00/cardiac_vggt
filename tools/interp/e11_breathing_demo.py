"""E11 — visual demo that breathing correction is relative (docs/126 §3).

All slots at the same cardiac phase (ED), so only breathing differs. Through-plane shift (dz, mm) cases:
  none   no shift
  some   every 3rd non-reference slot shifted +shift
  all    every slot (incl. reference) shifted +shift
Saves the input (naive, Δ=0) and full-model volumes plus the predicted per-slot dz for each case.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import harness as H         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, nargs="+", required=True)
    ap.add_argument("--shift", type=float, nargs="+", default=[8.0, 15.0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    for idx in args.idx:
        subj = H.Subject(ds, idx, dev)
        S = subj.S
        ts = np.zeros(S, dtype=int)
        rec = {"gt": subj.phases[0].cpu().numpy().astype(np.float16), "slot_z": np.asarray(subj.slot_z),
               "ref_plane": subj.ref_plane, "dz": subj.dz}
        cases = [("none", 0.0, np.zeros(S, bool))]
        some = np.zeros(S, bool); some[1::3] = True
        for sh in args.shift:
            cases += [(f"some{sh:g}", sh, some), (f"all{sh:g}", sh, np.ones(S, bool))]
        for name, sh, mask in cases:
            disp = np.zeros((S, 3), np.float32); disp[mask, 0] = sh
            b = H.build_batch(subj, ts, disp)
            o = H.forward(model, subj, b)
            V, _ = H.splat(subj, b, o["wp"])
            V0, _ = H.splat(subj, b, b["scanner_coords"].clone())
            pred = H.slot_disp_mm(subj, b, o["dvf"], "fov")[:, 0]
            rec[f"{name}_out"] = V.cpu().numpy().astype(np.float16)
            rec[f"{name}_in"] = V0.cpu().numpy().astype(np.float16)
            rec[f"{name}_applied"] = disp[:, 0]
            rec[f"{name}_pred"] = pred
            print(f"[{idx}] {name}: applied mean {disp[:, 0].mean():.1f}  pred on shifted {np.nanmean(pred[mask]) if mask.any() else 0:.1f}"
                  f"  pred on unshifted {np.nanmean(pred[~mask]) if (~mask).any() else float('nan'):.1f}", flush=True)
        np.savez_compressed(f"{args.out}/s{idx:03d}.npz", **rec)


if __name__ == "__main__":
    main()
