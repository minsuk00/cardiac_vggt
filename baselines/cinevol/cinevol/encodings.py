"""Anisotropic 3D grids, smoothstep interpolation, and optional Grid4D CUDA.

The torch implementation follows the mathematical indexing/interpolation convention
of the cited encoder, with ordinary autograd for derivative checks. No CUDA source
is vendored. The external Grid4D dependency is pinned by scripts/setup.sh.
"""
import importlib
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn


class TorchHashGrid(nn.Module):
    def __init__(self, levels, features, base, finest, log2_size):
        super().__init__()
        base = np.asarray(base, dtype=np.float64)
        growth = np.exp2(np.log2(np.asarray(finest) / base) / (levels - 1))
        # Match Grid4D's host-side allocation and float32 device scale.
        sizes = np.minimum(2 ** log2_size,
                           np.prod(np.ceil(base[None] * growth[None] ** np.arange(levels)[:, None]), axis=1)).astype(np.int64)
        scale = torch.exp2(torch.arange(levels)[:, None] * torch.log2(torch.tensor(growth, dtype=torch.float32))) * torch.tensor(base, dtype=torch.float32) - 1
        self.register_buffer("scale", scale)
        self.register_buffer("resolution", (torch.ceil(scale) + 1).long())
        self.register_buffer("sizes", torch.from_numpy(sizes))
        self.register_buffer("offsets", torch.from_numpy(np.concatenate(([0], np.cumsum(sizes)[:-1]))))
        self.register_buffer("corners", torch.tensor([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)]))
        self.embeddings = nn.Parameter(torch.empty(int(sizes.sum()), features))
        nn.init.uniform_(self.embeddings, -1e-4, 1e-4)
        self.output_dim = levels * features

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, 3)
        valid = ((x >= 0) & (x <= 1)).all(-1)
        p = x[:, None, :] * self.scale
        floor = p.floor()
        fraction = p - floor
        smooth = fraction.square() * (3 - 2 * fraction)
        corners = self.corners[None, None]
        integer = floor.long()[:, :, None] + corners
        weights = torch.where(corners.bool(), smooth[:, :, None], 1 - smooth[:, :, None]).prod(-1)
        dense = integer[..., 0] + self.resolution[None, :, None, 0] * (integer[..., 1] + self.resolution[None, :, None, 1] * integer[..., 2])
        hashed = (integer[..., 0] ^ (integer[..., 1] * 2654435761) ^ (integer[..., 2] * 805459861)) & 0xFFFFFFFF
        index = torch.where((self.resolution.prod(-1) > self.sizes)[None, :, None], hashed, dense)
        index = index.remainder(self.sizes[None, :, None]) + self.offsets[None, :, None]
        out = (self.embeddings[index] * weights[..., None]).sum(2).flatten(1)
        return (out * valid[:, None]).reshape(*shape, self.output_dim)


class Grid4DHashGrid(nn.Module):
    def __init__(self, levels, features, base, finest, log2_size):
        super().__init__()
        root = Path(os.environ.get("CINEVOL_GRID4D", Path(__file__).resolve().parents[1] / "third_party/Grid4D"))
        if not (root / "hashencoder/hashgrid.py").is_file():
            raise RuntimeError("Grid4D dependency missing. Run bash scripts/setup.sh on the GPU node, or use --backend torch.")
        # Calling .venv/bin/python does not activate its bin directory. PyTorch's
        # extension builder must find the Ninja executable installed beside it.
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        sys.path.insert(0, str(root))
        module = importlib.import_module("hashencoder.hashgrid")
        self.grid = module.HashEncoder(input_dim=3, num_levels=levels, level_dim=features,
                                       base_resolution=base, desired_resolution=finest,
                                       log2_hashmap_size=log2_size)
        self.output_dim = levels * features

    def forward(self, x):
        # Our interface takes [0,1]; the upstream wrapper maps [-1,1] to [0,1].
        # PSF batches can have non-contiguous strides; upstream uses .view().
        shape = x.shape[:-1]
        encoded = self.grid((2 * x - 1).reshape(-1, 3).contiguous(), size=1)
        return encoded.reshape(*shape, self.output_dim)


def make_grid(spec, backend):
    cls = {"torch": TorchHashGrid, "grid4d": Grid4DHashGrid}[backend]
    return cls(spec.get("levels", spec.get("levels_each")), spec["features_per_level"],
               spec["base_resolution"], spec["finest_resolution"], spec["log2_hashmap_size"])
