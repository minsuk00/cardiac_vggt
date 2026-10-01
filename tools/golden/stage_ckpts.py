"""Stage the 4 paper checkpoints to /tmp as weights-only copies for the golden harness.

Copies each checkpoint_last.pt off GPFS sequentially (one big read), strips it to
{"model": state_dict}, and deletes the full copy.
"""
import os
import shutil
import sys

import torch

LOGS = "/home/minsukc/vggt/scratch/logs"
RUNS = {
    "diff1000": "210823094_final518_diff1000_curated898",
    "base": "210823094_final518_base_curated898",
    "nogather": "210498476_final518_nogather_curated898",
    "hw0": "210484679_final518_hw0_curated898",
}
OUT = "/tmp/vggt_golden/ckpts"


def main():
    os.makedirs(OUT, exist_ok=True)
    for arm, run in RUNS.items():
        dst = os.path.join(OUT, f"{arm}.pt")
        if os.path.exists(dst):
            print(f"{arm}: exists, skip", flush=True)
            continue
        raw = os.path.join(OUT, f"{arm}.full.pt")
        shutil.copyfile(os.path.join(LOGS, run, "ckpts", "checkpoint_last.pt"), raw)
        ck = torch.load(raw, map_location="cpu", weights_only=False)
        torch.save({"model": ck["model"], "prev_epoch": ck.get("prev_epoch")}, dst + ".part")
        os.replace(dst + ".part", dst)
        os.remove(raw)
        print(f"{arm}: staged epoch={ck.get('prev_epoch')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
