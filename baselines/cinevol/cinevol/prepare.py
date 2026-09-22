"""Prepare native multistack or explicit single-SAX cine with supplied states."""
import argparse
import json
import time
from pathlib import Path
import nibabel as nib
import numpy as np
from .config import write_json


def image_geometry(image):
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    if qc and sc and not np.allclose(q, s, atol=1e-3):
        raise ValueError("Conflicting NIfTI qform and sform: resolve acquisition geometry before preparation")
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    rotation = image.affine[:3, :3] / spacing
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("Sheared acquisition affine is unsupported")
    return spacing, rotation


def states(value, root, z, t):
    a = np.load(root / value) if isinstance(value, str) else np.asarray(value)
    if a.shape == (t,):
        a = np.broadcast_to(a, (z, t))
    if a.shape != (z, t) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError(f"States must be [frames] or [slices,frames] in [0,1], got {a.shape}")
    return a


def coverage(xyz, stacks):
    counts = np.zeros(len(xyz), dtype=np.int16)
    for stack in stacks:
        im = stack["image"]
        ijk = nib.affines.apply_affine(np.linalg.inv(im.affine), xyz)
        inside = np.all((ijk >= -0.5) & (ijk <= np.array(im.shape[:3]) - 0.5), axis=1)
        # Gaps between planes do not count as slice support.
        dz = np.abs(ijk[:, 2] - np.round(ijk[:, 2])) * stack["spacing"][2]
        inside &= dz <= stack["thickness"] / 2 + 1e-5
        counts += inside
    return counts


def prepare(spec_path, output):
    started = time.perf_counter()
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    root, output = spec_path.parent, Path(output).resolve()
    single_sax = spec.get("protocol") == "acdc_single_sax_simulated"
    minimum_coverage = 1 if single_sax else 2
    if single_sax and len(spec["stacks"]) != 1:
        raise ValueError("The ACDC single-SAX protocol requires exactly one stack")
    if single_sax and (not spec.get("simulation") or not spec.get("state_provenance")):
        raise ValueError("The ACDC adaptation requires simulation metadata and supplied-state provenance")
    if not single_sax and len(spec["stacks"]) < 2:
        raise ValueError("Provide independently acquired multiplanar cine stacks; SAX-only ACDC is not native CiNeVol input")
    stacks = []
    ids = [s["stack_id"] for s in spec["stacks"]]
    if len(set(ids)) != len(ids):
        raise ValueError("stack_id must identify a distinct acquisition")
    for s in spec["stacks"]:
        image = nib.load(root / s["image"])
        if len(image.shape) != 4 or image.shape[3] < 2:
            raise ValueError("Every stack must be 4D cine (X,Y,slice,frame)")
        spacing, rotation = image_geometry(image)
        thickness = float(s["slice_thickness_mm"])
        if thickness <= 0:
            raise ValueError("Specify actual slice thickness in mm")
        stacks.append(dict(image=image, spacing=spacing, rotation=rotation, thickness=thickness,
                           cardiac=states(s["cardiac_states"], root, *image.shape[2:]),
                           respiratory=states(s["respiratory_states"], root, *image.shape[2:])))
    def batches():
        slice_id = 0
        for s in stacks:
            im = s["image"]
            values = np.asarray(im.dataobj, dtype=np.float32)
            ij = np.stack(np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]), indexing="ij"), -1).reshape(-1, 2)
            for z in range(im.shape[2]):
                xyz = nib.affines.apply_affine(im.affine, np.column_stack((ij, np.full(len(ij), z))))
                valid = coverage(xyz, stacks) >= minimum_coverage
                if not valid.any():
                    continue
                for t in range(im.shape[3]):
                    v = values[:, :, z, t].reshape(-1)
                    use = valid & np.isfinite(v)
                    yield s, slice_id, z, t, xyz[use], v[use]
                slice_id += 1

    # Count first, then stream into memory-mapped files. Full ACDC cine can
    # contain tens of millions of pixels; concatenated copies exceed login RAM.
    count, rotations, spacings = 0, [], []
    for s, sid, z, t, xyz, v in batches():
        count += len(v)
        if sid == len(rotations):
            rotations.append(s["rotation"])
            spacings.append([*s["spacing"][:2], s["thickness"]])
    if not count:
        raise ValueError("No finite pixels satisfy the acquisition coverage requirement")
    output.mkdir(parents=True, exist_ok=True)
    observation_dir = output / "observations"
    observation_dir.mkdir(exist_ok=True)
    arrays = {}
    for key in ("xyz", "value", "cardiac", "respiratory", "slice_index", "frame_index"):
        shape = (count, 3) if key == "xyz" else (count,)
        dtype = np.int64 if key.endswith("index") else np.float32
        arrays[key] = np.lib.format.open_memmap(observation_dir / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
    cursor = 0
    for s, sid, z, t, xyz, v in batches():
        end = cursor + len(v)
        arrays["xyz"][cursor:end] = xyz
        arrays["value"][cursor:end] = v
        arrays["cardiac"][cursor:end] = s["cardiac"][z, t]
        arrays["respiratory"][cursor:end] = s["respiratory"][z, t]
        arrays["slice_index"][cursor:end] = sid
        arrays["frame_index"][cursor:end] = t
        cursor = end
    scale = float(np.percentile(np.abs(arrays["value"]), 99))
    if scale <= 0:
        raise ValueError("Observed input intensities are all zero")
    for start in range(0, count, 1000000):
        arrays["value"][start:start + 1000000] /= scale
    for a in arrays.values():
        a.flush()
    arrays.clear()
    np.save(observation_dir / "rotation.npy", np.asarray(rotations, np.float32))
    np.save(observation_dir / "spacing.npy", np.asarray(spacings, np.float32))
    meta = {"schema_version": 1, "subject": spec["subject"], "observations": "observations",
            "minimum_stack_coverage": minimum_coverage, "intensity_scale": scale,
            "acquisition_stack_count": len(stacks),
            "state_provenance": spec["state_provenance"], "output_grid": spec["output_grid"],
            "reference_cardiac_state": spec["reference_cardiac_state"],
            "end_expiration_state": spec["end_expiration_state"],
            "protocol": spec.get("protocol", "native_full_cine"), "source_spec": str(spec_path),
            "preparation_seconds": time.perf_counter() - started}
    for key in ("simulation", "adaptations"):
        if key in spec:
            meta[key] = spec[key]
    for key in ("reference", "reference_cine", "before", "evaluation_mask"):
        if spec.get(key):
            meta[key] = str((root / spec[key]).resolve())
    if "reference_cine_cardiac_states" in spec:
        meta["reference_cine_cardiac_states"] = spec["reference_cine_cardiac_states"]
    write_json(output / "subject.json", meta)
    from .data import Observations
    dataset = Observations(output / "subject.json")
    print(f"Prepared {dataset.n:,} covered pixels, {dataset.n_slices} slice acquisitions: {output / 'subject.json'}")
    return output / "subject.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.spec, a.output)


if __name__ == "__main__":
    main()
