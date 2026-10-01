def backbone_patch_size(backbone: str) -> int:
    """Patch size is a fixed property of the backbone, not a free knob (docs/77)."""
    if backbone == "dinov3_vitl16":
        return 16
    if backbone.startswith("dinov2_"):
        return 14
    raise ValueError(f"Unknown backbone {backbone!r}: cannot derive patch_size")
