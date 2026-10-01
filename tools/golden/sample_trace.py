"""G1: fingerprint the train/val batches the dataloader yields (CPU, no model).

Mirrors Trainer: set_seeds(seed + epoch*100, max_epochs) then the epoch's loader.
--variant switches on a non-paper sampler path (multi-frame, static, fixed t, no reference slot).
"""
import argparse

from common import ARMS, RECIPE, compose, dump, fingerprint, train_val_loaders
from train_utils.general import set_seeds

VARIANTS = {
    "paper": [],
    "multiframe": ["one_frame_per_slice=false"],
    "static": ["mri_data_mode=static"],
    "tfixed0": ["t_target_fixed=0"],
    "noref": ["reference_slot=false"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--variant", default="paper", choices=VARIANTS)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--val_steps", type=int, default=10)
    a = p.parse_args()

    cfg = compose(RECIPE + ARMS["diff1000"] + VARIANTS[a.variant])
    trace = {"train": [], "val": []}
    train_loader, val_loader = train_val_loaders(cfg)
    for epoch in range(a.epochs):
        set_seeds(cfg.seed_value + epoch * 100, cfg.max_epochs, 0)
        for i, batch in enumerate(train_loader(epoch)):
            if i >= a.steps:
                break
            trace["train"].append({"epoch": epoch, "step": i, **fingerprint(batch)})

    for i, batch in enumerate(val_loader(0)):
        if i >= a.val_steps:
            break
        trace["val"].append({"step": i, **fingerprint(batch)})

    dump(trace, a.out)
    print(f"wrote {a.out}: {len(trace['train'])} train, {len(trace['val'])} val batches")


if __name__ == "__main__":
    main()
