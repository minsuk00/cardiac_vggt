"""G1b: fingerprint gpu_augment_batch outputs on real batches, the way the trainer calls it
(train: affine transforms + private respiratory generator; val: no transforms, train=False).

  python tools/golden/aug_trace.py --out temp/golden/<tag>/g1b_<device>.json --device cpu|cuda
"""
import argparse

from common import ARMS, RECIPE, compose, dump, fingerprint, train_val_loaders

import torch
from data.gpu_aug import build_gpu_transforms, gpu_augment_batch
from data.respiratory import RespiratoryConfig
from train_utils.general import copy_data_to_device, set_seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--val_steps", type=int, default=6)
    a = p.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)

    cfg = compose(RECIPE + ARMS["diff1000"])
    aug_cfg = cfg.data.augmentation
    transforms = build_gpu_transforms(aug_cfg)
    resp_cfg = RespiratoryConfig.from_cfg(aug_cfg.get("respiratory"))
    gen = torch.Generator(device=a.device).manual_seed(int(cfg.seed_value))
    train_loader, val_loader = train_val_loaders(cfg)

    trace = {"train": [], "val": []}
    set_seeds(cfg.seed_value, cfg.max_epochs, 0)
    for i, batch in enumerate(train_loader(0)):
        if i >= a.steps:
            break
        batch = copy_data_to_device(batch, a.device)
        out = gpu_augment_batch(batch, transforms, a.device, respiratory_cfg=resp_cfg,
                                train=True, resp_generator=gen)
        trace["train"].append({"step": i, **fingerprint(out)})

    for i, batch in enumerate(val_loader(0)):
        if i >= a.val_steps:
            break
        batch = copy_data_to_device(batch, a.device)
        out = gpu_augment_batch(batch, None, a.device, respiratory_cfg=resp_cfg, train=False)
        trace["val"].append({"step": i, **fingerprint(out)})

    dump(trace, a.out)
    print(f"wrote {a.out}: {len(trace['train'])} train, {len(trace['val'])} val batches")


if __name__ == "__main__":
    main()
