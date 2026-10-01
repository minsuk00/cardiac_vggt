"""G5: a val-only trainer run on a paper checkpoint, then fingerprint every file it wrote.

Runs training/launch.py main() in-process with mode=val on the 4-subject g5_split.txt (the EF
sweep -> 8 val batches), deterministic kernels, compile off.
  --variant off : no wandb, EF, visuals or volume dumps (the metric path only)
  --variant on  : every monitoring cadence = 1, wandb offline, EF + volume dumps on

  python tools/golden/run_val.py --variant off --out temp/golden/<tag>/g5_off
"""
import argparse
import csv
import glob
import json
import os
import sys

from common import ARMS, RECIPE, REPO, dump, sha

import torch

CKPT = "/tmp/vggt_golden/ckpts/diff1000.pt"
VARIANTS = {
    "off": ["logging.wandb_writer=null", "logging.ef_eval_enable=false",
            "logging.log_visuals=false", "logging.save_val_volumes=false"],
    "on": ["logging.filmstrip_every_n_val_epochs=1", "logging.visual_panels_every_n_val_epochs=1",
           "logging.ef_eval_every_n_val_epochs=1", "logging.ef_eval_enable=true",
           "logging.log_visuals=true", "logging.save_val_volumes=true"],
}


def fingerprint_run(log_dir):
    res = {"metrics": []}
    for line in open(os.path.join(log_dir, "metrics.jsonl")):
        if line.strip():
            row = json.loads(line)
            row.pop("t", None)                    # wall-clock timestamp
            res["metrics"].append(row)
    for name in ("val_per_subject.csv", "baseline_identity.json"):
        path = os.path.join(log_dir, name)
        if os.path.exists(path):
            res[name] = list(csv.reader(open(path))) if name.endswith(".csv") else json.load(open(path))
    import nibabel as nib
    for path in sorted(glob.glob(os.path.join(log_dir, "val_volumes", "**", "*.nii*"), recursive=True)):
        res.setdefault("val_volumes", {})[os.path.relpath(path, log_dir)] = sha(
            nib.load(path).get_fdata(dtype="float32"))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--out", required=True, help="log dir for this run (must not exist)")
    a = p.parse_args()

    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"--out {out} is not empty; refusing to reuse it")
    os.environ.setdefault("WANDB_MODE", "offline")
    torch.use_deterministic_algorithms(True, warn_only=True)
    overrides = RECIPE + ARMS["diff1000"] + VARIANTS[a.variant] + [
        "mode=val", f"split_file={REPO}/tools/golden/g5_split.txt",
        f"checkpoint.resume_checkpoint_path={CKPT}", "checkpoint.strict=true",
        f"checkpoint.save_dir={out}/ckpts", f"logging.log_dir={out}", "exp_name=golden_val",
        "cuda.compile_attention_blocks=false",
    ]

    from training import launch
    sys.argv = ["launch.py", "--config", "default", *overrides]
    launch.main()

    dump(fingerprint_run(out), os.path.join(out, "golden.json"))
    print("wrote", os.path.join(out, "golden.json"))


if __name__ == "__main__":
    main()
