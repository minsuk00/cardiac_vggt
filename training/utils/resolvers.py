"""The OmegaConf custom resolvers every config in training/config/ uses — one definition.

`register_all()` is idempotent: names already registered are left alone, so importing this from
several entry points (launch.py, data/__init__.py, tests, tools) never raises or swaps a resolver.
"""
import time

from omegaconf import OmegaConf

from vggt.models.backbones import backbone_patch_size

# Computed ONCE per process. `rev_ts` sorts runs reverse-chronologically; interpolations such as
# `log_dir: ...${exp_name}` re-resolve on every read, so a resolver returning a fresh timestamp
# would make exp_name, log_dir and the wandb name drift apart within one run.
REVERSE_TS = str(2000000000 - int(time.time()))


def _backbone_tag(name):
    """Compact WandB tag for the backbone."""
    return "dinov3" if str(name).startswith("dinov3_") else "dinov2"


def _aug_tag(enabled, tier):
    """Compact WandB tag for the augmentation setting."""
    return "noaug" if not enabled else {"aggressive": "aug_agg"}[str(tier)]


RESOLVERS = {
    "rev_ts": lambda: REVERSE_TS,
    "backbone_tag": _backbone_tag,
    "aug_tag": _aug_tag,
    # Patch size derives from the backbone so config cannot express a mismatch (docs/77).
    "backbone_ps": backbone_patch_size,
}


def register_all():
    for name, fn in RESOLVERS.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
