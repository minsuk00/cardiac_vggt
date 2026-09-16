"""Paper-style centre errors on labelled held-out ACDC ED/ES frames."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
import torch

from .align import load_model, predict_centers
from .data import case_paths, read_case


def summary(values):
    return dict(mean=float(np.mean(values)), std=float(np.std(values)),
                median=float(np.median(values)), count=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('ACDC/database'))
    parser.add_argument('--subset', choices=('test', 'val', 'acdc-test'), default='test')
    parser.add_argument('--output', type=Path, default=Path('runs/dangi/evaluation.json'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    model, config = load_model(args.checkpoint, args.device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if args.subset == 'acdc-test':
        patients = sorted(p.name for p in (args.data / 'testing').glob('patient*'))
        subset = 'testing'
    else:
        patients, subset = payload['split'][args.subset], 'training'
    before, after, records = [], [], []
    for path in case_paths(args.data, patients, subset):
        images, true, valid, _ = read_case(path, config)
        predicted = predict_centers(model, images)
        initial = np.linalg.norm(true - true.mean(axis=0), axis=1)
        residual = np.linalg.norm(predicted - true, axis=1)
        before.extend(initial.tolist())
        after.extend(residual.tolist())
        records.append(dict(case=path.name, true_xy=true.tolist(),
                            predicted_xy=predicted.tolist(), measured_center=valid.tolist(),
                            initial_px=initial.tolist(), residual_px=residual.tolist()))
    if not before:
        raise ValueError('No evaluation frames found')
    ks = ks_2samp(before, after)
    result = dict(subset=args.subset, patients=patients, before_px=summary(before),
                  after_px=summary(after), ks_statistic=float(ks.statistic),
                  ks_pvalue=float(ks.pvalue), records=records,
                  note='Paper-style descriptive slice metrics, including propagated targets; '
                       'KS pools correlated slices. ACDC reimplementation, not original STACOM results.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
