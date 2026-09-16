"""Align 3D NIfTI stacks or every frame of a 4D cine with a trained checkpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import shift
import torch

from .config import Config
from .data import normalize, preprocess
from .model import DangiCenterNet


def load_model(checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = DangiCenterNet()
    model.load_state_dict(payload['model'])
    return model.to(device).eval(), Config(**payload['config'])


@torch.inference_mode()
def predict_centers(model, slices, batch_size=32):
    """Normalized (Z,192,192) arrays -> (Z,2) centres in (x,y) pixels."""
    if batch_size < 1 or len(slices) == 0:
        raise ValueError('Positive batch size and nonempty slice stack required')
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    for start in range(0, len(slices), batch_size):
        batch = np.ascontiguousarray(slices[start:start+batch_size, None], dtype=np.float32)
        predictions.append(model(torch.from_numpy(batch).to(device)).cpu().numpy())
    centers = np.concatenate(predictions)
    if not np.isfinite(centers).all():
        raise FloatingPointError('Model produced nonfinite centres')
    return centers


def translate_slices(slices, translations_xy, order=1):
    """Positive x/y moves content right/down; zero fill, no wraparound."""
    translations_xy = np.asarray(translations_xy)
    if slices.ndim != 3 or translations_xy.shape != (len(slices), 2):
        raise ValueError('Expected slices (Z,Y,X) and translations (Z,2)')
    if not np.isfinite(translations_xy).all():
        raise ValueError('Translations must be finite')
    return np.stack([shift(plane, (dy, dx), order=order, mode='constant',
                           cval=0, prefilter=False)
                     for plane, (dx, dy) in zip(slices, translations_xy)])


def anchor_point(centers, anchor, config=Config()):
    """Where every predicted centre is translated to.

    None -> image centre (paper Sec. 2.4). int -> that slice's predicted centre (reference-
    slice anchoring: same per-slice shifts up to one whole-stack constant). (2,) -> explicit.
    """
    if anchor is None:
        return np.asarray(config.center, dtype=np.float32)
    if np.ndim(anchor) == 0:
        return centers[int(anchor)]
    return np.asarray(anchor, dtype=np.float32)


def align_stack(model, slices, config=Config(), batch_size=32, anchor=None):
    """Input preprocessed (Z,192,192), in original intensity units.

    Returns corrected slices, predicted centres, and applied translations.
    Normalization is used only for CNN input; output intensities are preserved.
    """
    slices = np.asarray(slices, dtype=np.float32)
    centers = predict_centers(model, normalize(slices), batch_size)
    translations = anchor_point(centers, anchor, config) - centers
    return translate_slices(slices, translations), centers, translations


def align_nifti(model, volume, config=Config(), batch_size=32):
    if len(volume.shape) not in (3, 4):
        raise ValueError('Expected a 3D stack or 4D cine')
    frames = [volume] if len(volume.shape) == 3 else [
        nib.Nifti1Image(np.asarray(volume.dataobj[..., t]), volume.affine)
        for t in range(volume.shape[3])]
    outputs, centers, translations = [], [], []
    for frame in frames:
        slices, affine = preprocess(frame, config)
        corrected, c, d = align_stack(model, slices, config, batch_size)
        outputs.append(corrected.transpose(2, 1, 0))
        centers.append(c)
        translations.append(d)
    output = outputs[0] if len(volume.shape) == 3 else np.stack(outputs, axis=3)
    header = volume.header.copy()
    header.set_data_dtype(np.float32)
    corrected_volume = nib.Nifti1Image(output, affine, header)
    metadata = dict(config=asdict(config), coordinate_order='x,y',
                    units='preprocessed pixels',
                    centers_xy=np.asarray(centers).tolist(),
                    translations_xy=np.asarray(translations).tolist(),
                    input_affine=volume.affine.tolist(), output_affine=affine.tolist(),
                    note='Arrays indexed [frame][slice][xy]; output is the resampled/cropped grid.')
    return corrected_volume, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from input')
    model, config = load_model(args.checkpoint, args.device)
    output, metadata = align_nifti(model, nib.load(args.input), config, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, args.output)
    args.output.with_name(args.output.name + '.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {args.output} and transform metadata')


if __name__ == '__main__':
    main()
