"""Reference-state metrics on the shared physical evaluation grid."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from .config import write_json
from .data import resolve


def metrics(prediction, target, mask=None):
    from skimage.metrics import structural_similarity
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Metrics require matched 3D arrays")
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not valid.all():
        raise ValueError("Non-finite evaluation volumes")
    if mask is not None:
        valid &= mask.astype(bool)
    if not valid.any():
        raise ValueError("Empty evaluation mask")
    error = np.square(prediction[valid].astype(np.float64) - target[valid])
    mse = float(error.mean())
    energy = float(np.square(target[valid].astype(np.float64)).sum())
    data_range = float(target[valid].max() - target[valid].min())
    if data_range <= 0 or energy <= 0:
        raise ValueError("Reference has no dynamic range or energy")
    win = min(7, min(target.shape))
    win = win if win % 2 else win - 1
    if win < 3:
        raise ValueError("SSIM needs at least 3 voxels per output dimension")
    _, ssim_map = structural_similarity(target.astype(np.float64), prediction.astype(np.float64),
                                       data_range=data_range, win_size=win, full=True)
    # Explicit convention: mean full SSIM map over the same support as pixel metrics.
    return {"mse": mse, "psnr_db": None if mse == 0 else float(10 * np.log10(data_range ** 2 / mse)),
            "perfect_match": mse == 0, "nmse": float(error.sum() / energy),
            "ssim": float(ssim_map[valid].mean()), "voxels": int(valid.sum()),
            "data_range": data_range, "ssim_window": win}


def evaluate(manifest, reconstruction, output=None):
    manifest = Path(manifest).resolve()
    meta = json.loads(manifest.read_text())
    if not meta.get("reference"):
        raise ValueError("No ground-truth/reference volume provided; reference metrics are unavailable")
    prediction = nib.load(reconstruction)
    target = nib.load(resolve(manifest, meta["reference"]))
    if prediction.shape != target.shape or not np.allclose(prediction.affine, target.affine, atol=1e-4):
        raise ValueError("Reference and output must already share the benchmark grid; no target-driven registration is performed")
    mask = None
    if meta.get("evaluation_mask"):
        mask_img = nib.load(resolve(manifest, meta["evaluation_mask"]))
        if mask_img.shape != target.shape or not np.allclose(mask_img.affine, target.affine, atol=1e-4):
            raise ValueError("Evaluation mask is on a different grid")
        mask = mask_img.get_fdata() > 0
    result = metrics(prediction.get_fdata(), target.get_fdata(), mask)
    result.update(reference_cardiac_state=meta["reference_cardiac_state"],
                  end_expiration_state=meta["end_expiration_state"],
                  protocol=meta["protocol"], intensity_normalization="none for evaluation; common original units",
                  registration="none", ssim_reduction="full SSIM map averaged over evaluation support")
    output = Path(output) if output else Path(reconstruction).parent / "metrics.json"
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--reconstruction", required=True)
    p.add_argument("--output")
    evaluate(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
