#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpu_cmode=shared

# --- Configuration ---
# mri_volume: direct V_canon vs V_gt supervised loss, per-axis normalization,
# splat-based volume output, residual-DVF head, frozen aggregator.
CONFIG="mri_volume"
NGPU=1
MASTER_PORT=29521

# --- Resume settings (set ONE of these or leave both empty for a fresh run) ---
# RESUME_FROM: path to a previous run's experiment directory.
#   Continues SAME exp_name → overwrites that run's checkpoints.
#   Reuses SAME wandb run id → metrics overlay on the old run's wandb dashboard.
#   Use this when continuing the same experiment (e.g., a crash recovery).
RESUME_FROM=""

# CKPT_ONLY: load model weights from this checkpoint but start a FRESH exp dir
# + FRESH wandb run. Use this when pipeline semantics changed (e.g., multi-phase
# vs ED-only) and you don't want metrics to overlay onto the old dashboard.
# Ignored if RESUME_FROM is set.
CKPT_ONLY="/home/minsukc/vggt/scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_${CONFIG}"
    if [ ! -z "$RESUME_FROM" ]; then
        JOB_NAME="${JOB_NAME}_resume"
    elif [ ! -z "$CKPT_ONLY" ]; then
        JOB_NAME="${JOB_NAME}_ckptonly"
    fi

    mkdir -p /home/minsukc/vggt/slurm_logs/

    echo "Submitting: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" \
           --output="/home/minsukc/vggt/slurm_logs/${TIMESTAMP}_${JOB_NAME}_%j.log" \
           "$0"
    exit
fi

# --- Environment Setup ---
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate svr

cd /home/minsukc/vggt

sleep $((SLURM_PROCID * 2))  # stagger startup

export WANDB_MODE=online

# --- Build Hydra overrides ---
OVERRIDES=""
if [ ! -z "$RESUME_FROM" ]; then
    # Mode 1 — continue same experiment dir + same wandb run.
    EXP_NAME=$(basename "$RESUME_FROM")
    CKPT_PATH="${RESUME_FROM}/ckpts/checkpoint_last.pt"
    if [ ! -f "$CKPT_PATH" ]; then
        echo "ERROR: RESUME_FROM is set but $CKPT_PATH does not exist."
        exit 1
    fi
    OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_PATH}"
    echo "Resuming (same exp + wandb) from: $CKPT_PATH"

    # Auto-extract WandB run id from <RESUME_FROM>/wandb/wandb/{run|offline-run}-<ts>-<id>/
    WANDB_DIR=$(ls -dt "${RESUME_FROM}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
    if [ ! -z "$WANDB_DIR" ]; then
        WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
        OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
        echo "Auto-detected WandB resume_id: $WANDB_RESUME_ID"
    else
        echo "Warning: no WandB run dir found under $RESUME_FROM/wandb/wandb/ — a new WandB run will be created."
    fi
elif [ ! -z "$CKPT_ONLY" ]; then
    # Mode 2 — load model weights only; fresh exp dir + fresh wandb run.
    if [ ! -f "$CKPT_ONLY" ]; then
        echo "ERROR: CKPT_ONLY is set but $CKPT_ONLY does not exist."
        exit 1
    fi
    OVERRIDES="checkpoint.resume_checkpoint_path=${CKPT_ONLY}"
    echo "Loading weights only from: $CKPT_ONLY (fresh exp_name + fresh wandb run)"
fi

CMD="PYTHONPATH=training:. torchrun \
    --nproc_per_node=$NGPU \
    --master_port=$MASTER_PORT \
    training/launch.py \
    --config $CONFIG $OVERRIDES"

echo "Running: $CMD"
eval $CMD
