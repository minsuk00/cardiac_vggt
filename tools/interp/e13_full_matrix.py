"""E13 — full global-attention matrix (all tokens x all tokens, mean of 16 heads) for chosen layers, one
subject, unmodified model, same companion/breathing draw as E1/E3/E12. Pooled 6x6 tokens for storage/display.
Token order = slot order (slot 0 = reference), each slot = 5 special tokens + 37x37 patches (row-major).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import harness as H         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=90)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 8])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    subj = H.Subject(ds, args.idx, dev)
    comp = np.random.default_rng(1000 + args.idx).integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + args.idx)
    b = H.build_batch(subj, comp, disp)
    o = H.forward(model, subj, b, full_layers=args.layers)
    rec = {f"L{l}": o["full"][l].numpy().astype(np.float32) for l in args.layers}
    rec.update(S=subj.S, slot_z=np.asarray(subj.slot_z), pool=H.CTL.full_pool if hasattr(H.CTL, "full_pool") else 6,
               heart=H.heart_patch_mask(subj, b).cpu().numpy())
    np.savez_compressed(f"{args.out}/s{args.idx:03d}.npz", **rec)
    print("saved", {k: v.shape for k, v in rec.items() if hasattr(v, "shape")}, flush=True)


if __name__ == "__main__":
    main()
