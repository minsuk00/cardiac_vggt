"""Write a weights-only copy of a checkpoint (keeps ONLY the model state_dict).

Used to seed the frozen-refiner run from an existing VGGT checkpoint: the new model has
a refiner submodule the old optimizer state never saw, so loading the full checkpoint's
optimizer would mismatch. Stripping to {"model": ...} makes the trainer load weights only
(strict=False tolerates the missing refiner.* keys) and start a fresh optimizer/epoch — while
SLURM requeue still does a full resume from the run's OWN later checkpoint_last.pt.

Usage: python tools/make_weights_only_ckpt.py <in.pt> <out.pt>
"""
import os
import sys

import torch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    inp, out = sys.argv[1], sys.argv[2]
    ck = torch.load(inp, map_location="cpu", weights_only=False)
    model = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"model": model}, out)
    print(f"wrote {out}  ({len(model)} tensors, weights-only; optimizer/epoch/scaler stripped)")


if __name__ == "__main__":
    main()
