"""G2: a short training run of one paper arm, then fingerprint what it produced.

Runs training/launch.py main() in-process with the arm's paper overrides plus a short schedule
(2 epochs x 10 steps, 3 val batches; val runs after epoch 0). wandb runs offline so every
visual/diagnostic path still executes. --det (default) adds torch.use_deterministic_algorithms
and eager attention so old-vs-new can be compared bit for bit.
"""
import argparse
import json
import os
import sys

from common import ARMS, RECIPE, dump, sha

import torch


def fingerprint_run(log_dir):
    rows = [json.loads(l) for l in open(os.path.join(log_dir, "metrics.jsonl")) if l.strip()]
    ck = torch.load(os.path.join(log_dir, "ckpts", "checkpoint_last.pt"), map_location="cpu",
                    weights_only=False)
    log = open(os.path.join(log_dir, "log.txt")).read() if os.path.exists(os.path.join(log_dir, "log.txt")) else ""
    return {
        "metrics": rows,
        "model": {k: sha(v) for k, v in sorted(ck["model"].items())},
        "optimizer": sha(json.dumps(ck["optimizer"]["param_groups"] if "param_groups" in ck["optimizer"]
                                    else str(ck["optimizer"].keys()), default=str)),
        "n_warnings": sum("WARNING" in l for l in log.splitlines()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--out", required=True, help="log dir for this run")
    p.add_argument("--paper_mode", action="store_true", help="keep torch.compile, no deterministic algos")
    a = p.parse_args()

    os.environ.setdefault("WANDB_MODE", "offline")
    overrides = RECIPE + ARMS[a.arm] + [
        "max_epochs=2", "limit_train_batches=10", "limit_val_batches=3",
        f"exp_name=golden_{a.arm}", f"logging.log_dir={os.path.abspath(a.out)}",
    ]
    if not a.paper_mode:
        torch.use_deterministic_algorithms(True)
        overrides.append("cuda.compile_attention_blocks=false")

    import launch
    sys.argv = ["launch.py", "--config", "default", *overrides]
    launch.main()

    dump(fingerprint_run(a.out), os.path.join(a.out, "golden.json"))
    print("wrote", os.path.join(a.out, "golden.json"))


if __name__ == "__main__":
    main()
