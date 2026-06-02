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
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --gpu_cmode=shared
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# --- Configuration ---
# ED-only variant of train_mri_volume.sh: forces t_target=0 every sample
# (matches the pre-multi-phase pipeline that produced the 4-day baseline).
CONFIG="mri_volume"
NGPU=1
MASTER_PORT=29522

# Force ED-only (t_target=0 every sample). Baked in for this script; do not remove.
PHASE_OVERRIDE="t_target_fixed=0"

# --- Resume settings (set ONE of these or leave both empty for a fresh run) ---
# RESUME_FROM: path to a previous run's experiment directory.
#   Continues SAME exp_name → overwrites that run's checkpoints.
#   Reuses SAME wandb run id → metrics overlay on the old run's wandb dashboard.
RESUME_FROM=""

# CKPT_ONLY: load model weights from this checkpoint but start a FRESH exp dir
# + FRESH wandb run. Ignored if RESUME_FROM is set.
CKPT_ONLY=""

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_${CONFIG}_ed"
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
# REQUEUE_STATE persists the pinned exp_name (and inherited overrides) across requeues.
# exp_name normally embeds a per-launch reverse timestamp (${rev_ts:}) that would change
# on every requeue → trainer's checkpoint auto-detect points at a fresh empty dir →
# restart-from-scratch. Pin it once on first launch, reuse on every restart.
REQUEUE_STATE="/home/minsukc/vggt/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    # --- Requeue restart: reuse pinned exp_name, resume from THIS run's own
    #     checkpoint_last.pt (auto-detected from save_dir). ---
    source "$REQUEUE_STATE"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    WANDB_DIR=$(ls -dt "./scratch/logs/${EXP_NAME}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
    if [ ! -z "$WANDB_DIR" ]; then
        WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
        OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
    fi
    echo "Requeue restart #${SLURM_RESTART_COUNT}: exp_name=${EXP_NAME}, resume_id=${WANDB_RESUME_ID:-<new>}, extra='${EXTRA_OVERRIDES}'"
else
    # --- First launch: decide exp_name, then persist it for requeue restarts. ---
    EXTRA_OVERRIDES=""
    if [ ! -z "$RESUME_FROM" ]; then
        EXP_NAME=$(basename "$RESUME_FROM")
        CKPT_PATH="${RESUME_FROM}/ckpts/checkpoint_last.pt"
        if [ ! -f "$CKPT_PATH" ]; then
            echo "ERROR: RESUME_FROM is set but $CKPT_PATH does not exist."
            exit 1
        fi
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_PATH}"
        echo "Resuming (same exp + wandb) from: $CKPT_PATH"

        WANDB_DIR=$(ls -dt "${RESUME_FROM}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
        if [ ! -z "$WANDB_DIR" ]; then
            WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
            OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
            echo "Auto-detected WandB resume_id: $WANDB_RESUME_ID"
        else
            echo "Warning: no WandB run dir found under $RESUME_FROM/wandb/wandb/ — a new WandB run will be created."
        fi
    elif [ ! -z "$CKPT_ONLY" ]; then
        if [ ! -f "$CKPT_ONLY" ]; then
            echo "ERROR: CKPT_ONLY is set but $CKPT_ONLY does not exist."
            exit 1
        fi
        # Pin a stable exp_name; carry max_epochs + limit_val_batches across requeues.
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_mri_volume_dynamic_axial_Cine_combined"
        EXTRA_OVERRIDES="max_epochs=500 limit_val_batches=30"
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_ONLY} ${EXTRA_OVERRIDES}"
        echo "Loading weights only from: $CKPT_ONLY (exp_name=${EXP_NAME}, fresh wandb run, max_epochs=500, limit_val_batches=30)"
    else
        # Fresh run. Pin a stable exp_name for requeue restarts.
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_mri_volume_dynamic_axial_Cine_combined"
        OVERRIDES="exp_name=${EXP_NAME}"
        echo "Fresh run: exp_name=${EXP_NAME}"
    fi
    { echo "EXP_NAME=${EXP_NAME}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
fi

# Always-on ED-only override (appended last so it can't be accidentally clobbered).
OVERRIDES="$OVERRIDES $PHASE_OVERRIDE"

echo "Running: torchrun ... --config $CONFIG $OVERRIDES"

# --- SLURM auto-requeue signal forwarding ---
# SLURM sends SIGUSR1 to THIS batch shell 120s before walltime (--signal=B:USR1@120).
# torchrun runs in the background; the trap forwards USR1 to torchrun's worker child
# (launch.py installs a SIGUSR1 handler that runs `scontrol requeue` then exits 0).
# `wait` after forwarding keeps the job cgroup alive until the worker has exited.
_forward_usr1() {
    echo "[requeue] batch shell caught SIGUSR1 — forwarding to torchrun workers (children of ${TORCHRUN_PID})"
    pkill -USR1 -P "$TORCHRUN_PID" 2>/dev/null
    wait "$TORCHRUN_PID"
}
trap _forward_usr1 USR1

export PYTHONPATH=training:.
torchrun \
    --nproc_per_node=$NGPU \
    --master_port=$MASTER_PORT \
    training/launch.py \
    --config $CONFIG $OVERRIDES &
TORCHRUN_PID=$!
wait "$TORCHRUN_PID"
