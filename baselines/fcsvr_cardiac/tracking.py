"""Resumable Weights & Biases initialization for the cardiac baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def get_or_create_wandb_id(output_dir: Path, generate_id: Callable[[], str]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "wandb_id.txt"
    if path.exists():
        run_id = path.read_text(encoding="utf-8").strip()
        if not run_id:
            raise RuntimeError(f"empty W&B run id: {path}")
        return run_id
    run_id = generate_id()
    path.write_text(run_id, encoding="utf-8")
    return run_id


def init_wandb(output_dir: Path, config: dict, *, resume_step: int | None):
    import wandb

    run_id = get_or_create_wandb_id(output_dir, lambda: wandb.util.generate_id(length=16))
    resume_args = (
        {"resume": "allow"}
        if resume_step is None
        else {"resume_from": f"{run_id}?_step={resume_step}"}
    )
    return wandb.init(
        project="fcsvr-cardiac",
        id=run_id,
        name=output_dir.name,
        **resume_args,
        dir=str(output_dir),
        config=config,
        tags=["fcsvr", "stage1", "cmrx24", "ed", "gt-pose-normalized"],
    )
