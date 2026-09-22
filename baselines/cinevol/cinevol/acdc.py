"""ACDC single-SAX adaptation with reproducible, supplied simulated motion states."""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import shift

from .config import write_json
from .data import digest
from .prepare import image_geometry, prepare


def respiratory_states(slices, frames, rng):
    # Independent acquisition starts per slice, with smooth breathing over its cine.
    starts = rng.uniform(0, 2 * np.pi, (slices, 1))
    cycles = rng.uniform(0.5, 1.5, (slices, 1))
    return ((1 - np.cos(starts + 2 * np.pi * cycles * np.arange(frames)[None] / frames)) / 2).astype(np.float32)


def corrupt_cine(clean, respiratory, spacing, displacement_mm):
    """Observed(x,y,z,t) = clean(x - dx(psi), y - dy(psi), z,t).

    The displacement vector is in acquisition in-plane axes (mm), from psi=0
    to psi=1. No ground-truth displacement is passed to the fitting model.
    """
    clean = np.asarray(clean, dtype=np.float32)
    respiratory = np.asarray(respiratory)
    displacement_mm = np.asarray(displacement_mm, dtype=float)
    if clean.ndim != 4 or not np.isfinite(clean).all():
        raise ValueError("Source must be a finite 4D cine")
    if respiratory.shape != clean.shape[2:] or not np.isfinite(respiratory).all() or np.any((respiratory < 0) | (respiratory > 1)):
        raise ValueError("Respiratory states must have shape (slices, frames) and values in [0,1]")
    if displacement_mm.shape != (2,) or not np.isfinite(displacement_mm).all():
        raise ValueError("Provide two finite in-plane displacement components in mm")
    shifts_mm = respiratory[..., None] * displacement_mm
    observed = np.empty_like(clean)
    for z in range(clean.shape[2]):
        for t in range(clean.shape[3]):
            observed[:, :, z, t] = shift(clean[:, :, z, t], shifts_mm[z, t] / np.asarray(spacing[:2]),
                                          order=1, mode="constant", cval=0., prefilter=False)
    return observed, shifts_mm.astype(np.float32)


def isotropic_grid(image, voxel_mm):
    spacing, rotation = image_geometry(image)
    # Preserve the centre and orientation; use a grid no larger than the source
    # voxel-centre span, avoiding an artificial padded reference border.
    shape = np.floor((np.asarray(image.shape[:3]) - 1) * spacing / voxel_mm).astype(int) + 1
    if np.any(shape < 3):
        raise ValueError("Output grid must have at least three voxels per dimension")
    affine = np.eye(4)
    affine[:3, :3] = rotation * voxel_mm
    centre = nib.affines.apply_affine(image.affine, (np.asarray(image.shape[:3]) - 1) / 2)
    affine[:3, 3] = centre - affine[:3, :3] @ ((shape - 1) / 2)
    return tuple(int(x) for x in shape), affine


def save_image(path, array, affine):
    image = nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine)
    image.header.set_xyzt_units("mm", "unknown")
    nib.save(image, path)


def generate(patient_dir, output, seed=7, displacement_mm=(5., 5.), voxel_mm=2.,
             slice_thickness_mm=None, reference_frame=None):
    started = time.perf_counter()
    patient_dir, output = Path(patient_dir).resolve(), Path(output).resolve()
    source = patient_dir / f"{patient_dir.name}_4d.nii.gz"
    image = nib.load(source)
    if len(image.shape) != 4 or image.shape[3] < 2:
        raise ValueError("ACDC input must be a full 4D cine")
    spacing, rotation = image_geometry(image)
    if not np.isfinite(voxel_mm) or voxel_mm <= 0:
        raise ValueError("Output voxel size must be positive and finite")
    thickness = float(spacing[2] if slice_thickness_mm is None else slice_thickness_mm)
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("Slice thickness must be positive and finite")
    if reference_frame is None:
        info = dict(line.split(":", 1) for line in (patient_dir / "Info.cfg").read_text().splitlines() if ":" in line)
        reference_frame = int(info["ED"])
    if not 1 <= reference_frame <= image.shape[3]:
        raise ValueError("Reference frame uses ACDC's one-based frame numbering")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory to preserve previous data: {output}")
    clean = image.get_fdata(dtype=np.float32)
    rng = np.random.default_rng(seed)
    z, t = image.shape[2:]
    respiratory = respiratory_states(z, t, rng)
    cardiac = np.broadcast_to(np.arange(t, dtype=np.float32) / t, (z, t)).copy()
    observed, shifts_mm = corrupt_cine(clean, respiratory, spacing, displacement_mm)
    selected = rng.integers(t, size=z)
    before = np.stack([observed[:, :, k, selected[k]] for k in range(z)], axis=2)
    grid_shape, grid_affine = isotropic_grid(image, voxel_mm)
    reference = nib.Nifti1Image(clean[..., reference_frame - 1], image.affine)
    reference = resample_from_to(reference, (grid_shape, grid_affine), order=1)
    output.mkdir(parents=True, exist_ok=True)
    save_image(output / "sax_observed.nii.gz", observed, image.affine)
    save_image(output / "before_native.nii.gz", before, image.affine)
    save_image(output / "reference.nii.gz", reference.get_fdata(dtype=np.float32), grid_affine)
    np.save(output / "cardiac_states.npy", cardiac)
    np.save(output / "respiratory_states.npy", respiratory)
    # Evaluation/provenance only. The model receives scalar states, not these shifts.
    np.savez(output / "simulation_gt.npz", respiratory_states=respiratory, cardiac_states=cardiac,
             shifts_inplane_mm=shifts_mm, shifts_world_mm=shifts_mm @ rotation[:, :2].T,
             selected_frame_indices=selected, selected_cardiac_states=cardiac[np.arange(z), selected],
             selected_respiratory_states=respiratory[np.arange(z), selected])
    simulation = {
        "name": "acdc_inplane_respiratory_translation_v1", "seed": seed,
        "source_cine": str(source), "source_sha256": digest(source),
        "source_shape": list(image.shape), "source_spacing_mm": spacing.tolist(),
        "displacement_at_state_one_inplane_mm": list(map(float, displacement_mm)),
        "respiratory_state_definition": "psi=(1-cos(start_z+2*pi*cycles_z*t/T))/2; psi=0 is unshifted end-expiration",
        "respiratory_sampling": "start_z uniform [0,2*pi); cycles_z uniform [0.5,1.5); independent per slice",
        "image_warp": "observed(x,y,z,t)=source(x-dx(psi),y-dy(psi),z,t); linear interpolation, zero outside FOV",
        "cardiac_state_definition": "zero-based frame index / number of frames; gated-cine proxy",
        "reference_frame_one_based": reference_frame,
        "slice_thickness_mm": thickness,
        "slice_thickness_source": "explicit user parameter" if slice_thickness_mm is not None else "assumed equal to header slice-centre spacing; true thickness unavailable",
        "output_voxel_mm": float(voxel_mm),
        "reference_definition": "linear resampling of original ACDC reference frame; not acquired isotropic high-resolution ground truth",
        "gt_file": str(output / "simulation_gt.npz"),
        "source_generation_seconds": time.perf_counter() - started,
    }
    spec = {
        "subject": patient_dir.name, "protocol": "acdc_single_sax_simulated",
        "state_provenance": "Known states from acdc_inplane_respiratory_translation_v1; cardiac phase from ACDC cine frame index",
        "simulation": simulation,
        "adaptations": ["single SAX support replaces native multistack intersection",
                        "supplied simulated respiration and frame-index cardiac phase",
                        "in-plane translation only; no simulated through-plane respiratory motion",
                        "reference is interpolated ACDC, not high-resolution ground truth",
                        "full observed cine for CiNeVol; before_native is a sparse asynchronous subset"],
        "reference_cardiac_state": (reference_frame - 1) / t, "end_expiration_state": 0.,
        "output_grid": {"shape": list(grid_shape), "affine": grid_affine.tolist()},
        "stacks": [{"stack_id": "SAX", "image": "sax_observed.nii.gz", "slice_thickness_mm": thickness,
                    "cardiac_states": "cardiac_states.npy", "respiratory_states": "respiratory_states.npy"}],
        "before": "before_native.nii.gz", "reference": "reference.nii.gz",
    }
    write_json(output / "simulation.json", simulation)
    write_json(output / "acquisitions.json", spec)
    del clean, observed, before, reference
    return prepare(output / "acquisitions.json", output / "prepared")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--displacement-mm", type=float, nargs=2, default=(5., 5.), metavar=("DX", "DY"))
    p.add_argument("--voxel-mm", type=float, default=2.)
    p.add_argument("--slice-thickness-mm", type=float)
    p.add_argument("--reference-frame", type=int, help="One-based ACDC frame; defaults to Info.cfg ED")
    generate(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
