"""Fine-tuning weight drift of the final model vs base VGGT-1B, per aggregator block and module.
Relative Frobenius change ||W_ft - W_base|| / ||W_base|| for attn.qkv / attn.proj / mlp in every
frame and global block, plus the camera/register tokens. CPU only. Writes JSON + prints a table."""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local   # noqa: E402

FT = os.path.join(ROOT, "scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt")
BASE = os.path.join(ROOT, "scratch/base_weights/vggt1b_base.pt")


def load(p):
    ck = torch.load(stage_checkpoint_to_local(p), map_location="cpu", weights_only=False, mmap=True)
    return ck["model"] if "model" in ck else ck


def main(out):
    ft, base = load(FT), load(BASE)
    res = {}
    for kind in ("frame_blocks", "global_blocks"):
        for i in range(24):
            for mod in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"):
                k = f"aggregator.{kind}.{i}.{mod}"
                if k not in base:
                    continue
                a, b = ft[k].float(), base[k].float()
                res[k] = float((a - b).norm() / b.norm())
    for k in ("aggregator.camera_token", "aggregator.register_token"):
        a, b = ft[k].float(), base[k].float()
        res[k] = float((a - b).norm() / b.norm())
        res[k + "[ref_vs_other]_ft"] = float((a[:, 0] - a[:, 1]).norm() / a[:, 1].norm())
        res[k + "[ref_vs_other]_base"] = float((b[:, 0] - b[:, 1]).norm() / b[:, 1].norm())
    json.dump(res, open(out, "w"), indent=1)
    print("block   frame(qkv proj fc1 fc2)            global(qkv proj fc1 fc2)")
    for i in range(24):
        f = [res.get(f"aggregator.frame_blocks.{i}.{m}", float('nan')) for m in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")]
        g = [res.get(f"aggregator.global_blocks.{i}.{m}", float('nan')) for m in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")]
        print(f"{i:3d}  " + " ".join(f"{x:.3f}" for x in f) + "     " + " ".join(f"{x:.3f}" for x in g))
    for k, v in res.items():
        if "token" in k:
            print(k, round(v, 3))


if __name__ == "__main__":
    main(sys.argv[1])
