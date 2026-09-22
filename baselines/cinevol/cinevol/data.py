"""Prepared observations contain no target images or target segmentation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def resolve(manifest_path, value):
    p = Path(value)
    return p if p.is_absolute() else Path(manifest_path).resolve().parent / p


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Observations:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.meta = json.loads(self.path.read_text())
        if self.meta.get("schema_version") != 1:
            raise ValueError("Unsupported observation manifest schema")
        p = resolve(self.path, self.meta["observations"])
        if p.is_dir():
            files = sorted(p.glob("*.npy"))
            # Names and file digests identify the complete prepared observations.
            hashes = {f.name: digest(f) for f in files}
            observation_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            self.arrays = {f.stem: np.load(f, mmap_mode="r", allow_pickle=False) for f in files}
        else:
            observation_digest = digest(p)
            with np.load(p, allow_pickle=False) as z:
                self.arrays = {k: np.array(z[k], copy=True) for k in z.files}
        self.fingerprint = {"manifest_sha256": digest(self.path), "observations_sha256": observation_digest}
        required = ("xyz", "value", "cardiac", "respiratory", "slice_index", "frame_index", "rotation", "spacing")
        if any(k not in self.arrays for k in required):
            raise ValueError(f"Observations require {required}")
        a = self.arrays
        self.n = len(a["value"])
        if not self.n or a["xyz"].shape != (self.n, 3):
            raise ValueError("Empty observations or invalid physical coordinates")
        for k in required:
            if not np.isfinite(a[k]).all():
                raise ValueError(f"Non-finite observations: {k}")
        for k in ("value", "cardiac", "respiratory", "slice_index", "frame_index"):
            if a[k].shape != (self.n,):
                raise ValueError(f"Incorrect shape for {k}")
        for k in ("cardiac", "respiratory"):
            if np.any((a[k] < 0) | (a[k] > 1)):
                raise ValueError(f"{k} must be supplied in [0,1]")
        for k in ("slice_index", "frame_index"):
            if not np.issubdtype(a[k].dtype, np.integer) or a[k].min() < 0:
                raise ValueError(f"{k} must be non-negative integer IDs")
        self.n_slices = int(a["slice_index"].max()) + 1
        self.n_frames = int(a["frame_index"].max()) + 1
        if a["rotation"].shape != (self.n_slices, 3, 3) or a["spacing"].shape != (self.n_slices, 3):
            raise ValueError("One rotation and spacing/thickness vector required per slice")
        if (a["spacing"] <= 0).any() or not np.allclose(a["rotation"].transpose(0, 2, 1) @ a["rotation"], np.eye(3), atol=1e-4):
            raise ValueError("Invalid PSF spacing or non-orthogonal acquisition orientation")
        single_sax = self.meta.get("protocol") == "acdc_single_sax_simulated"
        if single_sax:
            if (self.meta.get("minimum_stack_coverage") != 1
                    or self.meta.get("acquisition_stack_count") != 1
                    or not self.meta.get("simulation") or not self.meta.get("state_provenance")):
                raise ValueError("Single-SAX coverage requires one acquisition and documented simulated states")
        elif self.meta.get("minimum_stack_coverage", 0) < 2:
            raise ValueError("Native CiNeVol requires coverage by at least two distinct stacks; no silent single-stack adaptation")
        grid = self.meta["output_grid"]
        affine = np.asarray(grid["affine"], dtype=float)
        if affine.shape != (4, 4) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError("Invalid output affine")
        for key in ("reference_cardiac_state", "end_expiration_state"):
            if not 0 <= self.meta[key] <= 1:
                raise ValueError(f"Invalid {key}")
        if self.meta.get("intensity_scale", 0) <= 0:
            raise ValueError("A positive observed-input intensity scale is required")

    def bounds(self, padding=20.0):
        return np.stack((self.arrays["xyz"].min(0) - padding, self.arrays["xyz"].max(0) + padding))

    def sample(self, count, generator, device):
        idx = torch.randint(self.n, (count,), generator=generator).numpy()
        result = {}
        for k in ("xyz", "value", "cardiac", "respiratory", "slice_index", "frame_index"):
            dtype = torch.long if k.endswith("index") else torch.float32
            result[k] = torch.as_tensor(self.arrays[k][idx], dtype=dtype, device=device)
        si = self.arrays["slice_index"][idx]
        for k in ("rotation", "spacing"):
            result[k] = torch.as_tensor(self.arrays[k][si], dtype=torch.float32, device=device)
        return result


def chunks(batch, size):
    for start in range(0, len(batch["value"]), size):
        yield start, {k: v[start:start + size] for k, v in batch.items()}
