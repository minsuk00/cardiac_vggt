"""Runtime controls kept outside the released FC-SVR model implementation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def ensure_fresh_output(path: Path) -> None:
    if any(path.iterdir()):
        raise FileExistsError(
            f"output already contains run state: {path}; use --resume or a new --output"
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def truncate_jsonl_after(path: Path, step: int) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if int(record["step"]) <= step:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


@contextmanager
def deterministic_validation():
    was_enabled = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled)


def clip_gradients_like_author(model: torch.nn.Module) -> None:
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)


def configure_reproducibility(seed: int, *, strict: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(strict)


class EpochShuffle:
    def __init__(self, size: int, seed: int):
        self.size = size
        self.seed = seed
        self.epoch = -1
        self.order = None

    def index(self, step: int) -> int:
        epoch, offset = divmod(step, self.size)
        if epoch != self.epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.order = torch.randperm(self.size, generator=generator)
            self.epoch = epoch
        return int(self.order[offset])
