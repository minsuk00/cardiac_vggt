from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    size: int = 192
    spacing_mm: float = 1.5625
    center: tuple[float, float] = (96.0, 96.0)
    epochs: int = 100
    learning_rate: float = 1e-3  # Assumption: optimizer/rate not specified in paper.
    batch_size: int = 32  # Implementation choice.
    min_bp_pixels: int = 1  # Keep all nonempty ACDC blood pools by default.
    seed: int = 2018
