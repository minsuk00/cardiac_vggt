#!/bin/bash
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=01:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=tier1_gather_ab
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%x_%j.log
#SBATCH --open-mode=append

# Tier-1 offline validation of the gather-aux A/B (treatment gw=0.5 vs control gw=0.0).
# Reproduces the online breathing metric offline on the frozen checkpoints + renders a
# self-contained HTML report with many qualitative examples. ~30 min, single A40.
#
# Points at the LIVE checkpoint_last.pt in each exp dir (GPFS, visible from any node). These are
# updated by the running training jobs; the script copies each ckpt to node-local $TMPDIR first so
# a mid-write never corrupts the read. Re-run at convergence by resubmitting this script unchanged.
#
# Submit (from a login node OR an interactive GPU node — plain sbatch queues fine either way):
#   sbatch sbatch/tier1_gather_ab_report.sh

set -euo pipefail

REPO=/home/minsukc/vggt
TREAT_DIR="$REPO/scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined"
CTRL_DIR="$REPO/scratch/logs/216539845_mri_volume_diffusion_ftctrl_gather0_1frame_dynamic_axial_Cine_combined"

eval "$(micromamba shell hook --shell bash)"
micromamba activate svr

cd "$REPO"

# Snapshot the live checkpoints to node-local scratch to avoid reading a half-written file.
mkdir -p "$TMPDIR/ckpts"
cp "$TREAT_DIR/ckpts/checkpoint_last.pt" "$TMPDIR/ckpts/treatment.pt"
cp "$CTRL_DIR/ckpts/checkpoint_last.pt"  "$TMPDIR/ckpts/control.pt"

PYTHONPATH=training:. python tools/tier1_gather_ab_report.py \
    --treatment "$TMPDIR/ckpts/treatment.pt" \
    --control   "$TMPDIR/ckpts/control.pt" \
    --seqs 0-29 \
    --out "$REPO/_html" \
    --n_panels 12

echo "Report: $REPO/_html/gather_aux_ab_tier1_validation.html"
