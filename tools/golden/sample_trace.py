"""G1: fingerprint the train/val batches the dataloader yields (CPU, no model).

Mirrors Trainer: set_seeds(seed + epoch*100, max_epochs) then train_dataset.get_loader(epoch).
"""
import argparse

from common import ARMS, RECIPE, compose, dump, fingerprint
from hydra.utils import instantiate
from train_utils.general import set_seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--val_steps", type=int, default=10)
    a = p.parse_args()

    cfg = compose(RECIPE + ARMS["diff1000"])
    trace = {"train": [], "val": []}

    train = instantiate(cfg.data.train, _recursive_=False)
    train.seed = cfg.seed_value
    for epoch in range(a.epochs):
        set_seeds(cfg.seed_value + epoch * 100, cfg.max_epochs, 0)
        for i, batch in enumerate(train.get_loader(epoch=epoch)):
            if i >= a.steps:
                break
            trace["train"].append({"epoch": epoch, "step": i, **fingerprint(batch)})

    val = instantiate(cfg.data.val, _recursive_=False)
    val.seed = cfg.seed_value
    for i, batch in enumerate(val.get_loader(epoch=0)):
        if i >= a.val_steps:
            break
        trace["val"].append({"step": i, **fingerprint(batch)})

    dump(trace, a.out)
    print(f"wrote {a.out}: {len(trace['train'])} train, {len(trace['val'])} val batches")


if __name__ == "__main__":
    main()
