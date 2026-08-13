"""Deterministic fixed-ED validation for the cardiac Stage-1 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import models
from pipeline import make_dataset, validate
from runtime import configure_reproducibility, positive_int
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local


DEFAULT_RUN_DIR = Path(
    "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/logs/"
    "fcsvr_cardiac_stage1_cmrx24_ed"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", default="/home/minsukc/vggt/scratch/data")
    parser.add_argument("--split-file", default="training/splits/cmrx24only.txt")
    parser.add_argument("--output", default=str(DEFAULT_RUN_DIR / "evaluation.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args()

    configure_reproducibility(args.seed, strict=True)
    device = torch.device("cuda")
    model = models.flow_SNet3d0_1024().to(device).float()
    load_path = stage_checkpoint_to_local(args.checkpoint)
    state = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    dataset = make_dataset(args.data_root, args.split_file, "val")
    rows = validate(model, dataset, device, seed=args.seed, limit=args.limit)
    if args.verify_determinism:
        repeated = validate(model, dataset, device, seed=args.seed, limit=args.limit)
        if rows != repeated:
            raise RuntimeError("deterministic validation mismatch")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
