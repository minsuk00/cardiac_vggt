"""Small analytic multiview cine fixture; not a patient or a paper result."""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from .config import write_json
from .prepare import prepare


def phantom(x, cardiac, respiratory):
    y = np.array(x, copy=True)
    y[..., 2] += 3 * respiratory
    y[..., :2] *= np.asarray(1 + 0.18 * (1 - np.cos(2 * np.pi * cardiac)))[..., None]
    radius = np.sqrt((y[..., 0] / 9) ** 2 + (y[..., 1] / 8) ** 2 + (y[..., 2] / 14) ** 2)
    body = 0.12 * np.exp(-np.sum((y / 22) ** 2, axis=-1))
    blood = 0.75 / (1 + np.exp(np.clip((radius - 0.65) * 24, -50, 50)))
    wall = 0.35 * np.exp(-((radius - 0.85) / 0.12) ** 2)
    return (body + blood + wall).astype(np.float32)


def generate(output, size=20, frames=8, seed=7):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    target_affine = np.diag([2., 2., 2., 1.])
    target_affine[:3, 3] = -(size - 1)
    shape = (size, size, size)
    indices = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    world = nib.affines.apply_affine(target_affine, indices)
    phi_ref = 0.23
    reference = phantom(world, phi_ref, 0)
    nib.save(nib.Nifti1Image(reference, target_affine), output / "reference.nii.gz")
    cine = np.stack([phantom(world, t / 20, 0) for t in range(20)], -1)
    nib.save(nib.Nifti1Image(cine, target_affine), output / "reference_cine.nii.gz")
    stacks = []
    # Independently sample three thick-slice orientations from the continuous field.
    for view, axes in enumerate(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
        sh = (size, size, max(3, size // 2))
        rotation = np.eye(3)[:, axes]
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([2., 2., 4.])
        affine[:3, 3] = -affine[:3, :3] @ ((np.asarray(sh) - 1) / 2)
        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in sh], indexing="ij"), -1)
        xyz = nib.affines.apply_affine(affine, ijk)
        cardiac = np.broadcast_to(np.arange(frames) / frames, (sh[2], frames)).copy()
        respiratory = (1 - np.cos(np.arange(frames)[None] * 0.8 + rng.uniform(0, 2 * np.pi, (sh[2], 1)))) / 2
        data = np.empty((*sh, frames), np.float32)
        for t in range(frames):
            value = np.zeros(sh, np.float32)
            for _ in range(32):
                noise = rng.standard_normal((*sh, 3)) * (np.array([2.4, 2.4, 4.]) / np.sqrt(8 * np.log(2)))
                value += phantom(xyz + noise @ rotation.T, cardiac[:, t], respiratory[:, t]) / 32
            data[..., t] = value + rng.normal(0, 0.005, sh)
        path = f"view{view}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), output / path)
        stacks.append({"stack_id": f"view{view}", "image": path, "slice_thickness_mm": 4.,
                       "cardiac_states": cardiac.tolist(), "respiratory_states": respiratory.tolist()})
        if view == 0:
            chosen = rng.integers(frames, size=sh[2])
            before = np.stack([data[:, :, z, chosen[z]] for z in range(sh[2])], 2)
            from nibabel.processing import resample_from_to
            before = resample_from_to(nib.Nifti1Image(before, affine), (shape, target_affine), order=1)
            nib.save(before, output / "before.nii.gz")
    spec = {"subject": "analytic_demo", "protocol": "analytic_fixture_not_benchmark",
            "state_provenance": "Known analytic cardiac phase and respiratory amplitude (oracle)",
            "stacks": stacks, "output_grid": {"shape": list(shape), "affine": target_affine.tolist()},
            "reference_cardiac_state": phi_ref, "end_expiration_state": 0.,
            "reference": "reference.nii.gz", "reference_cine": "reference_cine.nii.gz", "before": "before.nii.gz",
            "reference_cine_cardiac_states": (np.arange(20) / 20).tolist()}
    write_json(output / "acquisitions.json", spec)
    return prepare(output / "acquisitions.json", output / "prepared")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/demo")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--frames", type=int, default=8)
    a = p.parse_args()
    generate(a.output, a.size, a.frames)


if __name__ == "__main__":
    main()
