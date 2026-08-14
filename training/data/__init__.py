from omegaconf import OmegaConf

from .datasets.mri_dataset import MRIDataset


def backbone_patch_size(backbone: str) -> int:
    """Patch size is a fixed property of the backbone, not a free knob (docs/77)."""
    if backbone == "dinov3_vitl16":
        return 16
    if backbone.startswith("dinov2_") or "conv" in backbone:
        return 14
    raise ValueError(f"Unknown backbone {backbone!r}: cannot derive patch_size")


# Registered here (imported by every consumer that instantiates datasets/models from the
# config) so `patch_size: ${backbone_ps:${backbone}}` resolves in standalone compose()
# scripts too, not just training/launch.py.
OmegaConf.register_new_resolver("backbone_ps", backbone_patch_size, replace=True)
