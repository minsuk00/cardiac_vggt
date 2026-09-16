"""Train the Stage A baseline: python -m dangi.train --help."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import ACDCSlices, case_paths, make_split
from .model import DangiCenterNet
from .pool import PoolSlices, read_split


def ssd_loss(prediction, target):
    """Sum x/y squared error per slice, averaged over the minibatch."""
    return (prediction - target).square().sum(dim=1).mean()


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, (images, centers) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images, centers = images.to(device), centers.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = ssd_loss(model(images), centers)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training/validation loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
    if not count:
        raise ValueError("Empty epoch")
    return total / count


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('ACDC/database'))
    parser.add_argument('--output', type=Path, default=Path('runs/dangi'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--min-bp-pixels', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-train-batches', type=int,
                        help='Smoke test only: truncate each training epoch')
    parser.add_argument('--pool-split', type=Path,
                        help='VGGT pooled split file ([train]/[val] sections); --data is then the data root')
    parser.add_argument('--pool-cache', type=Path, help='npz cache dir for preprocessed pool subjects')
    parser.add_argument('--pool-limit', type=int, help='Smoke test only: first N subjects per section')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.min_bp_pixels) < 1 or args.workers < 0:
        parser.error('Epochs, batch size, minimum BP size must be positive; workers >= 0')
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error('--max-train-batches must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run within a Slurm GPU allocation')
    config = replace(Config(), epochs=args.epochs, batch_size=args.batch_size,
                     seed=args.seed, min_bp_pixels=args.min_bp_pixels)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.pool_split:
        split = dict(source='pool', split_file=str(args.pool_split.resolve()),
                     train=read_split(args.pool_split, 'train')[:args.pool_limit],
                     val=read_split(args.pool_split, 'val')[:args.pool_limit])
    else:
        split = make_split(args.data, config.seed)
    checkpoint = None
    if args.resume:
        if args.resume.parent.resolve() != args.output.resolve():
            raise ValueError('Resume in the original --output directory to retain the best checkpoint')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        previous = checkpoint['config']
        for key in asdict(config):
            if key != 'epochs' and previous[key] != asdict(config)[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        if checkpoint['split'] != split:
            raise ValueError('Resume split does not match current dataset')
        if checkpoint.get('max_train_batches') != args.max_train_batches:
            raise ValueError('Resume must use the same smoke-test/training budget')
    elif (args.output / 'last.pt').exists() or (args.output / 'best.pt').exists():
        raise FileExistsError('Output already has checkpoints; use --resume or a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'split.json').write_text(json.dumps(split, indent=2) + '\n')
    (args.output / 'config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
    if args.pool_split:
        print('Loading and preprocessing pooled cohort (all phases, nnU-Net LV centroids)...', flush=True)
        train = PoolSlices(args.data, args.pool_split, 'train', config, augmented=True,
                           limit=args.pool_limit, cache_dir=args.pool_cache)
        val = PoolSlices(args.data, args.pool_split, 'val', config,
                         limit=args.pool_limit, cache_dir=args.pool_cache)
    else:
        print('Loading and preprocessing labelled ED/ES frames...', flush=True)
        train = ACDCSlices(case_paths(args.data, split['train']), config, augmented=True)
        val = ACDCSlices(case_paths(args.data, split['val']), config)
    print(f'{len(train.images)} training slices; {len(train)} augmented examples/epoch; '
          f'{len(val)} validation slices', flush=True)
    generator = torch.Generator()
    training = DataLoader(train, batch_size=config.batch_size, shuffle=True,
                          generator=generator, num_workers=args.workers,
                          pin_memory=device.type == 'cuda')
    validation = DataLoader(val, batch_size=config.batch_size,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')
    model = DangiCenterNet().to(device)
    # Paper does not specify an optimizer; HANDOFF.md chooses Adam at 1e-3.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start, best = 0, float('inf')
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best = checkpoint['epoch'], checkpoint['best_val_ssd']
    for epoch in range(start, config.epochs):
        # Epoch-specific order makes epoch-boundary resume reproducible.
        generator.manual_seed(config.seed + epoch)
        begin = time.monotonic()
        training_loss = run_epoch(model, training, device, optimizer, args.max_train_batches)
        validation_loss = run_epoch(model, validation, device)
        improved = validation_loss < best
        best = min(best, validation_loss)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                       config=asdict(config), split=split, epoch=epoch+1,
                       best_val_ssd=best, max_train_batches=args.max_train_batches)
        if improved:
            save_checkpoint(args.output / 'best.pt', payload)
        save_checkpoint(args.output / 'last.pt', payload)
        record = dict(epoch=epoch+1, train_ssd=training_loss, val_ssd=validation_loss,
                      seconds=time.monotonic()-begin, best=improved)
        with (args.output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
