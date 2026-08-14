#!/usr/bin/env python3
"""Build the weights-only VGGT-1B + DINOv3 ViT-L/16 hybrid seed on CPU.

This intentionally has no automatic download/install path. The official DINOv3 repository is
gated: accept its terms and authenticate first, then run only with the optional dependency
installed and a GPFS-backed Hugging Face cache directory.
"""

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

import torch


DEFAULT_BASE = Path("scratch/base_weights/vggt1b_base.pt")
DEFAULT_OUTPUT = Path("scratch/base_weights/vggt1b_dinov3_vitl16_seed.pt")
DEFAULT_HF_CACHE = Path("scratch/huggingface")
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
OLD_BACKBONE_PREFIX = "aggregator.patch_embed."
NEW_BACKBONE_PREFIX = "aggregator.patch_embed.model."
Z_EMBEDDER_SEED = 42 * 200


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_model_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{path} must contain a state dict, got {type(checkpoint).__name__}")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError(f"{path} model state must be a mapping, got {type(state).__name__}")
    if not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in state.items()):
        raise TypeError(f"{path} model state must map string names to tensors")
    return state


def _z_embedder_state(seed=Z_EMBEDDER_SEED):
    from vggt.models.aggregator import ZIndexEmbedder

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        state = ZIndexEmbedder(embed_dim=1024).state_dict()
    return {f"aggregator.z_embedder.{name}": tensor.clone() for name, tensor in state.items()}


def _assemble_hybrid(base_state, dinov3_state, target_shapes):
    hybrid = {name: base_state[name] for name in target_shapes if name in base_state}
    for name, tensor in _z_embedder_state().items():
        if name in target_shapes and name not in hybrid:
            hybrid[name] = tensor
    for name, tensor in dinov3_state.items():
        hybrid[f"{NEW_BACKBONE_PREFIX}{name}"] = tensor

    target_keys = set(target_shapes)
    hybrid_keys = set(hybrid)
    missing = sorted(target_keys - hybrid_keys)
    unexpected = sorted(hybrid_keys - target_keys)
    if missing or unexpected:
        raise ValueError(
            f"Hybrid key mismatch: {len(missing)} missing, {len(unexpected)} unexpected; "
            f"missing[:10]={missing[:10]}, unexpected[:10]={unexpected[:10]}"
        )

    shape_errors = [
        (name, tuple(hybrid[name].shape), target_shapes[name])
        for name in sorted(target_keys)
        if tuple(hybrid[name].shape) != target_shapes[name]
    ]
    if shape_errors:
        raise ValueError(f"Hybrid tensor shape mismatch (first 10): {shape_errors[:10]}")
    return hybrid


def _save_weights_only(state, output_path: Path, overwrite=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save({"model": state}, temp_path)
        if overwrite:
            os.replace(temp_path, output_path)
        else:
            os.link(temp_path, output_path)
            temp_path.unlink()
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _target_shapes():
    from vggt.models.vggt import VGGT

    with torch.device("meta"):
        model = VGGT(
            img_size=256,
            patch_size=16,
            backbone="dinov3_vitl16",
            embed_dim=1024,
            enable_point=True,
            use_z_pose_embedding=True,
            use_reference_token=True,
            train_on_residual_dvf=True,
            warp_head_type="dpt",
        )
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def build_seed(base_path: Path, output_path: Path, hf_cache: Path, overwrite: bool = False):
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing seed: {output_path}")
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if output_path.resolve() == base_path.resolve():
        raise ValueError("Output must differ from the VGGT-1B source checkpoint")

    try:
        from transformers import DINOv3ViTModel
    except ImportError as exc:
        raise ImportError("Install requirements-dinov3.txt before running this tool") from exc

    hf_cache.mkdir(parents=True, exist_ok=True)
    base_state = _load_model_state(base_path)
    dinov3 = DINOv3ViTModel.from_pretrained(MODEL_ID, cache_dir=str(hf_cache))
    dinov3_state = dinov3.state_dict()

    target_shapes = _target_shapes()
    hybrid = _assemble_hybrid(base_state, dinov3_state, target_shapes)
    _save_weights_only(hybrid, output_path, overwrite=overwrite)
    print(
        f"Wrote {output_path} with {len(hybrid)} tensors; replaced "
        f"{sum(name.startswith(OLD_BACKBONE_PREFIX) for name in base_state)} DINOv2 tensors "
        f"with {len(dinov3_state)} DINOv3 tensors."
    )


def main():
    args = _parse_args()
    build_seed(args.base, args.output, args.hf_cache, args.overwrite)


if __name__ == "__main__":
    main()
