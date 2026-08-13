"""Minimal float32 PyTorch trainer for FC-SVR cardiac Stage 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import models
from models.losses import l22_loss_affine_invariant
from pipeline import append_jsonl, make_dataset, prepare_sample, validate
from runtime import (
    EpochShuffle,
    clip_gradients_like_author,
    configure_reproducibility,
    deterministic_validation,
    ensure_fresh_output,
    positive_int,
    truncate_jsonl_after,
)
from tracking import init_wandb
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local


DEFAULT_OUTPUT = (
    "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/logs/"
    "fcsvr_cardiac_stage1_cmrx24_ed"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/minsukc/vggt/scratch/data")
    parser.add_argument("--split-file", default="training/splits/cmrx24only.txt")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=positive_int, default=256_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=positive_int, default=5_000)
    parser.add_argument("--val-limit", type=positive_int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def save_checkpoint(path, model, optimizer, step):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def main():
    args = parse_args()
    configure_reproducibility(args.seed)
    device = torch.device("cuda")
    torch.set_default_dtype(torch.float32)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(args.data_root, args.split_file, "train")
    val_set = make_dataset(args.data_root, args.split_file, "val")
    model = models.flow_SNet3d0_1024().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = 0
    checkpoint = out / "checkpoint_last.pt"
    if args.resume and not checkpoint.exists():
        raise FileNotFoundError(f"--resume requested but {checkpoint} does not exist")
    if not args.resume:
        ensure_fresh_output(out)
    if args.resume:
        load_path = stage_checkpoint_to_local(str(checkpoint))
        state = torch.load(load_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        truncate_jsonl_after(out / "train_metrics.jsonl", start)
        truncate_jsonl_after(out / "val_metrics.jsonl", start)

    run = init_wandb(out, {
        **vars(args),
        "model": "Flow_SNet3d0_1024",
        "loss": "l22_loss_affine_invariant",
        "batch_size": 1,
        "precision": "float32",
        "label": "FC-SVR Stage 1, GT-pose-normalized",
    }, resume_step=start if args.resume else None)

    model.train()
    subject_order = EpochShuffle(len(train_set), args.seed)
    for step in range(start, args.steps):
        sample = prepare_sample(
            train_set, subject_order.index(step), device,
            seed=args.seed * 1_000_003 + step,
        )
        lr = args.lr * (1.0 - step / args.steps) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        prediction = model(sample["inputs"].float())
        loss = l22_loss_affine_invariant(prediction, sample["target"].float())
        loss.backward()
        clip_gradients_like_author(model)
        optimizer.step()
        append_jsonl(out / "train_metrics.jsonl", {"step": step + 1, "loss": loss.item(), "lr": lr})
        run.log({"train/loss": loss.item(), "train/lr": lr}, step=step + 1)

        due = (step + 1) % args.val_every == 0 or step + 1 == args.steps
        if due:
            with deterministic_validation():
                rows = validate(
                    model, val_set, device, seed=args.seed + 10_000, limit=args.val_limit
                )
            for row in rows:
                append_jsonl(out / "val_metrics.jsonl", {"step": step + 1, **row})
            val_log = {}
            for mode in ("raw", "compensated"):
                for key in rows[0][mode]:
                    val_log[f"val/{mode}/{key}"] = sum(row[mode][key] for row in rows) / len(rows)
            run.log(val_log, step=step + 1)
            # Publish the checkpoint last. If disk/W&B logging fails first, the
            # previous checkpoint remains authoritative and resume_from rewinds
            # W&B before the interval is regenerated.
            save_checkpoint(checkpoint, model, optimizer, step + 1)
    run.finish()


if __name__ == "__main__":
    main()
