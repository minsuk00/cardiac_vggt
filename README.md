# Cardiac 4D MRI Slice-to-Volume Reconstruction

Adapts [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025) for **unsupervised slice-to-volume reconstruction of cardiac cine MRI**. The input is one 2D slice per short-axis plane, each from an arbitrary cardiac phase, plus one reference slice at the target phase. The model reconstructs the full 3D volume at that target phase. Trained on the pooled curated-v2 split (`training/splits/pooled_curated_v2.txt`: 628 / 90 / 180 train / val / test subjects).

## Setup

1. **Environment** (Python 3.10; the sbatch scripts expect an env named `svr`):
   ```bash
   micromamba create -n svr python=3.10 -y && micromamba activate svr
   pip install -r requirements.txt
   pip install --no-deps git+https://github.com/halleewong/batchaug.git@52163bd15ca0ddaabca0ec3c71fbf09b9a883a4b
   pip install --no-deps -e .     # the vggt, training and inference packages (editable)
   ```
   `--no-deps` keeps pip from re-resolving the pinned torch stack. Optional: `fused_ssim` (the
   per-slice SSIM val metric; training runs without it, the metric is just skipped) —
   `module load gcc/11.2.0 cuda/13.1.0 && pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim/@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38`.
2. **Storage.** Every path in the configs and scripts is relative to the repo root, and runs always
   launch from there. Data, base weights and run logs live under `./scratch/` (gitignored): make it a
   symlink to your own storage, `ln -s /path/to/your/vggt-scratch scratch`, then
   - `scratch/data/` — the prepared cohort (LPS, apex at z0, per-source trees named in
     `training/splits/pooled_curated_v2.txt`) and `scratch/data/whs/cardiac_phase.csv`. Copy or link it
     from the lab GPFS copy at `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/data`.
   - `scratch/base_weights/vggt1b_base.pt` — the VGGT-1B weights,
     `https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`.
   - `mkdir -p slurm_logs` (SLURM logs).
3. **Logging.** `wandb login` once (the cluster script runs wandb online). Set `VGGT_NOTIFY_EMAIL`
   to get the gradient-collapse alarm by email; unset, it is only logged.

The `sbatch/` scripts run the clone they are submitted from and use `$MAMBA_ROOT_PREFIX`
(default `~/micromamba`) / `$MAMBA_EXE` (default `~/.local/bin/micromamba`); SLURM mails the
submitting user. Check the `#SBATCH --account` / partition lines match an allocation you belong to.

## Training

Entry point: `training.launch` (Hydra, single GPU). Config: `training/config/default.yaml` = the paper recipe (the `diff1000` arm); ablation arms are `training/config/ablation_<arm>.yaml`.

```bash
torchrun --nproc_per_node=1 -m training.launch --config default
```

Cluster (paper recipe, self-submitting, auto-requeue): `ARM=diff1000 bash sbatch/train_final_518.sh`.
`ARM` is one of `base | diff1000 | hw2 | hw0 | nogather | diff1000_nogather`; `diff1000` is the paper model.

## Evaluation

`CKPT=<run>/ckpts/checkpoint_last.pt MODEL_NAME=<name> sbatch sbatch/eval_pooled_val.sh` (from the repo root) scores a checkpoint on the frozen gated + breathing-simulated bundles; without `CKPT` it scores the paper checkpoint. It writes under `evaluation/volumes/`, so first link that to storage: `ln -s ../scratch/eval evaluation/volumes`. See `evaluation/README.md`.

## Acknowledgements

Built on top of [VGGT](https://github.com/facebookresearch/vggt) (Wang et al., CVPR 2025). Thanks to its authors for open-sourcing the model and pretrained weights, which we adapted for the cardiac MRI setting.

## License

See [LICENSE.txt](./LICENSE.txt).
