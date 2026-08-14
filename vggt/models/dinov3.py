"""Transformers-backed DINOv3 ViT-L/16 patch-token adapter."""

import torch
import torch.nn as nn


DINOV3_VITL16_CONFIG = {
    "patch_size": 16,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_register_tokens": 4,
    "image_size": 256,
    "use_gated_mlp": False,
}


def extract_patch_tokens(
    last_hidden_state: torch.Tensor,
    *,
    height: int,
    width: int,
    patch_size: int = 16,
    num_register_tokens: int = 4,
) -> torch.Tensor:
    """Remove CLS/register tokens and verify the remaining spatial token grid."""
    if patch_size <= 0:
        raise ValueError(f"patch_size={patch_size} must be positive.")
    if num_register_tokens < 0:
        raise ValueError(f"num_register_tokens={num_register_tokens} must be non-negative.")
    if height <= 0 or width <= 0 or height % patch_size or width % patch_size:
        raise ValueError(
            f"Input size {(height, width)} must be positive and divisible by patch_size={patch_size}."
        )
    if last_hidden_state.ndim != 3:
        raise ValueError(
            f"last_hidden_state must have shape (batch, tokens, channels), got {tuple(last_hidden_state.shape)}"
        )

    special_tokens = 1 + num_register_tokens
    expected_patches = (height // patch_size) * (width // patch_size)
    expected_total = special_tokens + expected_patches
    if last_hidden_state.shape[1] != expected_total:
        raise ValueError(
            f"DINOv3 returned {last_hidden_state.shape[1]} tokens; expected {expected_total} "
            f"({special_tokens} special + {expected_patches} patches) for {(height, width)}."
        )
    return last_hidden_state[:, special_tokens:]


class DINOv3ViTL16PatchEmbed(nn.Module):
    """DINOv3 ViT-L/16 whose public output is patch tokens only.

    Transformers is imported only when this optional backbone is constructed, so the
    default DINOv2 path does not require the optional dependency.
    """

    patch_size = 16
    num_register_tokens = 4

    def __init__(self, img_size: int = 256):
        super().__init__()
        if img_size <= 0 or img_size % self.patch_size:
            raise ValueError(
                f"img_size={img_size} must be a positive multiple of {self.patch_size}."
            )
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError as exc:
            raise ImportError(
                "The DINOv3 backbone requires requirements-dinov3.txt. "
                "The default DINOv2 backbone does not require Transformers."
            ) from exc

        config = DINOv3ViTConfig(**{**DINOV3_VITL16_CONFIG, "image_size": img_size})
        self.model = DINOv3ViTModel(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must have shape (batch, 3, H, W), got {tuple(images.shape)}")
        _, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected 3 input channels, got {channels}")
        output = self.model(pixel_values=images)
        return extract_patch_tokens(
            output.last_hidden_state,
            height=height,
            width=width,
            patch_size=self.patch_size,
            num_register_tokens=self.num_register_tokens,
        )
