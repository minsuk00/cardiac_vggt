"""Small orchestration/plotting helpers for the central notebook.

Training, preprocessing, inference and metrics remain in their original modules.
"""
import json
from pathlib import Path
import shlex
import subprocess


def run_command(command, root):
    """Stream a module's progress and propagate failure to the notebook."""
    print(shlex.join(map(str, command)), flush=True)
    process = subprocess.Popen(list(map(str, command)), cwd=root,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        process.stdout.close()
    if status:
        raise subprocess.CalledProcessError(status, command)


def training_command(python, data, run, epochs=100, batch_size=32,
                     workers=4, seed=2018, min_bp_pixels=1, resume=True):
    run = Path(run).resolve()
    command = [str(python), '-u', '-m', 'dangi.train', '--data', str(Path(data).resolve()),
               '--output', str(run), '--epochs', str(epochs), '--batch-size', str(batch_size),
               '--workers', str(workers), '--seed', str(seed),
               '--min-bp-pixels', str(min_bp_pixels), '--device', 'cuda']
    if resume and (run / 'last.pt').exists():
        command += ['--resume', str(run / 'last.pt')]
    return command


def submit_training(command, root, run, partition='spgpu', walltime='01:30:00', cpus=4):
    """Submit exactly the notebook configuration; no hidden default run directory."""
    root, run = Path(root).resolve(), Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    script = '\n'.join([
        '#!/bin/bash', 'set -euo pipefail', f'cd {shlex.quote(str(root))}',
        'export OMP_NUM_THREADS=1', 'export MKL_NUM_THREADS=1',
        'srun ' + shlex.join(list(map(str, command))), '',
    ])
    script_path = run / 'notebook_train.slurm'
    script_path.write_text(script)
    result = subprocess.run([
        'sbatch', '--parsable', '--account=jjparkcv98', f'--partition={partition}',
        '--gres=gpu:1', '--nodes=1', '--ntasks=1', '--mem=48G',
        f'--cpus-per-task={cpus}', f'--time={walltime}', '--job-name=dangi-stage-a',
        f'--output={run / "train-%j.log"}', str(script_path),
    ], cwd=root, capture_output=True, text=True, check=True, timeout=30)
    job_id = result.stdout.strip().split(';')[0]
    if not job_id.isdigit():
        raise RuntimeError(f'Unexpected sbatch response: {result.stdout}')
    (run / 'job.json').write_text(json.dumps(dict(job_id=job_id, command=command), indent=2)+'\n')
    return job_id


def job_status(run):
    path = Path(run) / 'job.json'
    if not path.exists():
        return 'No notebook-submitted job recorded for this run.'
    job_id = json.loads(path.read_text())['job_id']
    result = subprocess.run(['squeue', '--jobs', str(job_id), '--format=%.18i %.9P %.20j %.8T %.10M %R'],
                            capture_output=True, text=True, check=True, timeout=15)
    return result.stdout + '\nIf no job row appears, inspect the log or sacct for completion status.'


def read_history(run):
    path = Path(run) / 'history.jsonl'
    if not path.exists():
        return []
    # A notebook can read while training appends. Ignore only an incomplete tail.
    lines = path.read_text().splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1 or line.endswith('\n'):
                raise
    return records


def plot_history(records):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    for key, label in [('train_ssd', 'Training'), ('val_ssd', 'Validation')]:
        ax.plot([r['epoch'] for r in records], [r[key] for r in records], label=label)
    ax.set(xlabel='Epoch', ylabel='SSD (pixels²)', title='Centre-regression learning curves')
    ax.legend()
    fig.tight_layout()
    return fig


def plot_alignment(before, after, centers, target, z):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    low, high = float(before.min()), float(before.max())
    axes[0].imshow(before[z], cmap='gray', vmin=low, vmax=high)
    axes[0].scatter(*centers[z], marker='+', c='tab:orange', s=100)
    axes[0].set_title(f'Before · slice {z} · predicted centre')
    axes[1].imshow(after[z], cmap='gray', vmin=low, vmax=high)
    axes[1].scatter(*target, marker='+', c='tab:orange', s=100)
    axes[1].set_title('After · target centre')
    axes[2].plot(centers[:, 0], range(len(centers)), label='Predicted x')
    axes[2].plot(centers[:, 1], range(len(centers)), label='Predicted y')
    axes[2].axvline(target[0], color='gray', linestyle='--', label='Target')
    axes[2].invert_yaxis()
    axes[2].set(xlabel='Centre coordinate (pixels)', ylabel='Slice', title='Centres across stack')
    axes[2].legend()
    for ax in axes[:2]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def reference_on_output_grid(reference, shape_zyx, output_affine):
    """Sample a 3D reference intensity volume on the output's physical grid.

    This uses NIfTI geometry only; no registration or image-driven adjustment.
    """
    import numpy as np
    import nibabel as nib
    from nibabel.processing import resample_from_to
    if len(reference.shape) != 3:
        raise ValueError('Select a single 3D ground-truth frame before comparing')
    floating_reference = nib.Nifti1Image(reference.get_fdata(dtype=np.float32), reference.affine)
    sampled = resample_from_to(floating_reference, (tuple(shape_zyx[::-1]), output_affine),
                               order=1, mode='constant', cval=np.nan)
    return sampled.get_fdata(dtype=np.float32).transpose(2, 1, 0)


def plot_ground_truth_comparison(reference, before, after, affine, z, y, x,
                                predicted=None, true_centers=None, measured=None):
    """Matched SAX/XZ/YZ views with one intensity window and an absolute error map."""
    import numpy as np
    import matplotlib.pyplot as plt
    if reference.shape != before.shape or reference.shape != after.shape:
        raise ValueError('Ground truth, input and output must share the same grid')
    finite = reference[np.isfinite(reference)]
    if not len(finite):
        raise ValueError('Ground truth does not overlap the output grid; check NIfTI affines')
    low, high = np.percentile(finite, [1, 99.5])
    if high <= low:
        low, high = float(finite.min()), float(finite.max()) + 1e-6
    error = np.abs(after-reference)
    limit = max(float(np.nanpercentile(error, 99.5)), 1e-6)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout='constrained')
    volumes = (reference, before, after, error)
    titles = ('Ground-truth reference', 'Input', 'Dangi output', '|Output − ground truth|')
    planes = (lambda v: v[z], lambda v: v[:, y, :], lambda v: v[:, :, x])
    aspects = (spacing[1]/spacing[0], spacing[2]/spacing[0], spacing[2]/spacing[1])
    labels = (f'SAX · z={z}', f'XZ · y={y}', f'YZ · x={x}')
    for row, (plane, aspect) in enumerate(zip(planes, aspects)):
        for col, volume in enumerate(volumes):
            ax = axes[row, col]
            im = ax.imshow(plane(volume), cmap='magma' if col == 3 else 'gray',
                           vmin=0 if col == 3 else low, vmax=limit if col == 3 else high,
                           aspect=aspect if row == 0 else 'auto', interpolation='nearest')
            ax.set_title(titles[col] if row == 0 else '')
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(labels[row])
            if col == 3:
                fig.colorbar(im, ax=ax, shrink=.65, label='Intensity error')
    if predicted is not None:
        axes[0, 1].scatter(*predicted[z], c='tab:orange', marker='+', s=90, label='Predicted centre')
    if true_centers is not None:
        label = 'GT centre' if measured is None or measured[z] else 'Propagated centre'
        axes[0, 1].scatter(*true_centers[z], facecolors='none', edgecolors='lime',
                           marker='o', s=90, label=label)
    if predicted is not None or true_centers is not None:
        axes[0, 1].legend(fontsize=7, loc='upper right')
    fig.suptitle('Same physical grid and intensity window · XZ/YZ stretched vertically for visibility',
                 fontsize=12)
    return fig
