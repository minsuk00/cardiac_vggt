"""Export full end-expiration cine and the exact reference-state 3D query."""
import argparse
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from .config import write_json
from .fit import model_from_checkpoint
from .psf import FWHM_TO_SIGMA


@torch.no_grad()
def query(model, shape, affine, cardiac, respiratory, samples=16, chunk=4096, seed=19):
    device = next(model.parameters()).device
    total = int(np.prod(shape))
    out = np.empty(total, np.float32)
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    if not np.allclose(spacing, spacing[0], rtol=1e-4):
        raise ValueError("Paper inference requires an isotropic output grid")
    # CPU RNG makes sampling independent of accelerator and reproducible on reload.
    rng = torch.Generator().manual_seed(seed)
    for start in range(0, total, chunk):
        ijk = np.stack(np.unravel_index(np.arange(start, min(start + chunk, total)), shape), -1)
        xyz = torch.tensor(nib.affines.apply_affine(affine, ijk), dtype=torch.float32, device=device)
        noise = torch.randn(len(xyz), samples, 3, generator=rng).to(device)
        psf = xyz[:, None] + noise * (float(spacing[0]) * FWHM_TO_SIGMA)
        phi = torch.full(psf.shape[:-1], float(cardiac), device=device)
        psi = torch.full_like(phi, float(respiratory))
        intensity = model.intensity(psf, phi, psi)[0].mean(1)
        out[start:start + len(xyz)] = intensity.cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite reconstruction")
    return out.reshape(shape)


def reconstruct(checkpoint, output=None, device="cuda", chunk=4096, seed=19):
    if chunk < 1:
        raise ValueError("Chunk size must be positive")
    t_start = time.perf_counter()
    model, state = model_from_checkpoint(checkpoint, device)
    meta, config = state["subject_metadata"], state["config"]
    out = Path(output) if output else Path(checkpoint).resolve().parent / "reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    grid = meta["output_grid"]
    shape, affine = tuple(grid["shape"]), np.asarray(grid["affine"])
    phi = np.arange(config["full_cine_export_cardiac_frames"]) / config["full_cine_export_cardiac_frames"]
    psi = meta["end_expiration_state"]
    scale = meta["intensity_scale"]
    frames = []
    for i, phase in enumerate(phi):
        frames.append(query(model, shape, affine, phase, psi, config["psf_samples"]["inference"], chunk, seed) * scale)
        print(f"Reconstructed cardiac state {i+1}/{len(phi)}", flush=True)
    cine = nib.Nifti1Image(np.stack(frames, -1), affine)
    cine.header.set_xyzt_units("mm", "unknown")
    nib.save(cine, out / "cine_end_expiration.nii.gz")
    t_cine = time.perf_counter() - t_start
    ref = query(model, shape, affine, meta["reference_cardiac_state"], psi,
                config["psf_samples"]["inference"], chunk, seed) * scale
    img = nib.Nifti1Image(ref, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, out / "reference_state.nii.gz")
    write_json(out / "states.json", {
        "cardiac_states": phi.tolist(), "respiratory_state": psi,
        "reference_cardiac_state": meta["reference_cardiac_state"],
        "state_provenance": meta["state_provenance"], "checkpoint_step": state["step"],
        "profile": config["profile"], "smoke": config["smoke"], "protocol": meta["protocol"],
        "intensity_units": "original_observation_units", "seed": seed,
        "time_axis": "normalized cardiac phase; not seconds", "source_manifest": state["manifest"],
    })
    write_json(out / "timing_export.json", {"load_and_full_cine_seconds": t_cine,
               "additional_reference_query_seconds": time.perf_counter() - t_start - t_cine})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--seed", type=int, default=19)
    reconstruct(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
