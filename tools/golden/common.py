"""Shared helpers for the golden-output harness (tools/golden/).

Run everything from the repo root with PYTHONPATH=training:.
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by deterministic cuBLAS

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(REPO, "training"), REPO]

# The four paper arms: default.yaml + RECIPE + ARM overrides (sbatch/train_final_518.sh).
RECIPE = ["max_epochs=300", "loss.volume.splat_res=518",
          "data.augmentation.enable=true", "data.augmentation.tier=aggressive"]
# Every arm spells out its diffusion weight so the overrides mean the same thing whether
# default.yaml is the base recipe (old) or the diff1000 recipe (new).
ARMS = {
    "diff1000": ["loss.volume.diffusion_weight=1000.0"],
    "base": ["loss.volume.diffusion_weight=0.0"],
    "nogather": ["loss.volume.diffusion_weight=0.0", "loss.volume.gather_weight=0.0"],
    "hw0": ["loss.volume.diffusion_weight=0.0", "loss.volume.heart_weight=0.0"],
}


def sha(x):
    """Bit-exact fingerprint of a tensor / array / python value."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.view(torch.int16)
        x = x.contiguous().numpy()
    if isinstance(x, np.ndarray):
        h = hashlib.sha256(np.ascontiguousarray(x).tobytes())
        h.update(str((x.dtype, x.shape)).encode())
        return h.hexdigest()[:16]
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fingerprint(batch):
    return {k: sha(v) for k, v in sorted(batch.items())}


def compose(overrides):
    from resolvers import register_all
    from hydra import compose as _compose, initialize_config_dir
    register_all()
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "training", "config")):
        return _compose(config_name="default", overrides=overrides)


def train_val_loaders(cfg):
    """-> (get_train_loader(epoch), get_val_loader(epoch)) built the way the trainer builds them."""
    from hydra.utils import instantiate
    train = instantiate(cfg.data.train, _recursive_=False)
    train.seed = cfg.seed_value
    val = instantiate(cfg.data.val, _recursive_=False)
    val.seed = cfg.seed_value
    return train.get_loader, val.get_loader


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
