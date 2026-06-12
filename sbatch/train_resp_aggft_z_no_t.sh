#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=96:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --gpu_cmode=shared
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# =============================================================================
# RESPIRATORY run 2/3 — breathing ON, z input embedding but NO input-t embedding.
# Head + aggregator finetune (only DINOv2 patch_embed frozen), all 12 cardiac
# target phases, fresh from VGGT-1B base, SLURM auto-requeue.
#
# Same as run 1/3 but use_t_pose_embedding=false: the model no longer knows which
# cardiac phase each INPUT slice came from (blind to input t — the realistic
# one-frame-per-slice regime where input t is unavailable). target_t stays ON
# (use_target_t_pose_embedding=true) — it's always a free query. Tests whether
# input-t conditioning matters once the model must content-infer cardiac phase.
#
# COMPARISON SET (identical except the embedding/aug knobs):
#   1        breathing, use_z=T use_t=T          → input z + input t conditioning
#   2 (this) breathing, use_z=T use_t=F          → drop input-t conditioning
#   3        breathing + affine aug, use_z=T use_t=F
# All three keep use_target_t_pose_embedding=true (target_t is always available).
#
# Unfreezing the aggregator exposes params with no gradient in the point-only
# forward → DDP needs find_unused_parameters=true even on 1 GPU. ~27GB on the A40,
# ~2.8x slower than head-only. 200 epochs (limit_train_batches=1000, 1 GPU).
# =============================================================================

# ---- PER-RUN CONFIG (the only block that differs across the run scripts) ----
CONFIG="mri_volume"
NGPU=1
EXP_TAG="mri_volume_resp_allphases_aggft_z_no_t"
RUN_OVERRIDES="max_epochs=200 data.augmentation.respiratory.enable=true use_z_pose_embedding=true use_t_pose_embedding=false use_target_t_pose_embedding=true optim.frozen_module_names=[*patch_embed*] distributed.find_unused_parameters=true logging.wandb_writer.tags=[mri_volume,allphases,multiphase,aggft,resp,no_t]"
MASTER_PORT=29552
# -------------------------------------------------------------------------------

# Fresh-run series: do NOT seed from the (non-transferable) 4-day baseline.
RESUME_FROM=""
CKPT_ONLY=""

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_${EXP_TAG}"
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
# REQUEUE_STATE persists the pinned exp_name + EXTRA_OVERRIDES across requeues.
REQUEUE_STATE="/home/minsukc/vggt/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    source "$REQUEUE_STATE"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    WANDB_DIR=$(ls -dt "./scratch/logs/${EXP_NAME}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
    if [ ! -z "$WANDB_DIR" ]; then
        WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
        OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
    fi
    echo "Requeue restart #${SLURM_RESTART_COUNT}: exp_name=${EXP_NAME}, resume_id=${WANDB_RESUME_ID:-<new>}, extra='${EXTRA_OVERRIDES}'"
else
    REV_TS=$((2000000000 - $(date +%s)))
    EXP_NAME="${REV_TS}_${EXP_TAG}"
    EXTRA_OVERRIDES="${RUN_OVERRIDES}"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    echo "Fresh run: exp_name=${EXP_NAME}  overrides='${EXTRA_OVERRIDES}'"
    { echo "EXP_NAME=${EXP_NAME}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
fi

echo "Running: torchrun ... --config $CONFIG $OVERRIDES"

# --- SLURM auto-requeue signal forwarding ---
_forward_usr1() {
    echo "[requeue] batch shell caught SIGUSR1 — forwarding to torchrun workers (children of ${TORCHRUN_PID})"
    pkill -USR1 -P "$TORCHRUN_PID" 2>/dev/null
    wait "$TORCHRUN_PID"
}
trap _forward_usr1 USR1

export PYTHONPATH=training:.
set -f
torchrun \
    --nproc_per_node=$NGPU \
    --master_port=$MASTER_PORT \
    training/launch.py \
    --config $CONFIG $OVERRIDES &
TORCHRUN_PID=$!
wait "$TORCHRUN_PID"
