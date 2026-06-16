#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=14-00:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --gpu_cmode=shared
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# =============================================================================
# REFINER run A (FROZEN) — 3D UNet refiner on the splat, trained with ALL of VGGT
# FROZEN. Seeds VGGT weights from the resp run t59w6nqy (resp, z, no input-t, aggft;
# exp 218747856_..._z_no_t) and trains ONLY the refiner. Isolates the pure splat-deblur
# gain (geometry fixed). Breathing ON, two-term loss L_pre(V_canon)+λ·L_post(V_refined).
#
# Weights-only seed: the new model has a refiner the old optimizer never saw, so we load
# WEIGHTS ONLY (strict=False) + fresh optimizer/epoch. Put on RUN_OVERRIDES so SLURM
# requeue still full-resumes from this run's OWN checkpoint_last.pt.
# frozen_module_names freezes patch_embed+camera_token+aggregator+point_head ⇒ only
# refiner.* trains. find_unused_parameters=true (frozen forward emits grad-less params).
# =============================================================================

# ---- PER-RUN CONFIG ----
CONFIG="mri_volume"
NGPU=1
EXP_TAG="mri_refiner_frozen"
SEED_FULL="/home/minsukc/vggt/scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
SEED_WEIGHTS_ONLY="/home/minsukc/vggt/scratch/base_weights/t59w6nqy_resp_no_t_weights_only.pt"
RUN_OVERRIDES="max_epochs=200 enable_refiner=true refiner_use_coverage=true data.augmentation.respiratory.enable=true use_z_pose_embedding=true use_t_pose_embedding=false use_target_t_pose_embedding=true optim.frozen_module_names=[*patch_embed*,*camera_token*,*aggregator*,*point_head*] distributed.find_unused_parameters=true loss.volume.refiner_lambda=1.0 checkpoint.resume_checkpoint_path=${SEED_WEIGHTS_ONLY} logging.wandb_writer.tags=[mri_volume,allphases,multiphase,resp,no_t,refiner,frozen]"
MASTER_PORT=29557
# ------------------------

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

# --- Create the weights-only seed once (skips optimizer/epoch ⇒ no param-set mismatch) ---
if [ ! -f "$SEED_WEIGHTS_ONLY" ]; then
    if [ ! -f "$SEED_FULL" ]; then
        echo "ERROR: seed checkpoint $SEED_FULL not found."
        exit 1
    fi
    echo "Creating weights-only seed: $SEED_FULL -> $SEED_WEIGHTS_ONLY"
    python tools/make_weights_only_ckpt.py "$SEED_FULL" "$SEED_WEIGHTS_ONLY" \
        || { echo "ERROR: failed to create weights-only seed"; exit 1; }
fi

# --- Build Hydra overrides ---
REQUEUE_STATE="/home/minsukc/vggt/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    source "$REQUEUE_STATE"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    WANDB_DIR=$(ls -dt "./scratch/logs/${EXP_NAME}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
    if [ ! -z "$WANDB_DIR" ]; then
        WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
        OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
    fi
    echo "Requeue restart #${SLURM_RESTART_COUNT}: exp_name=${EXP_NAME}, resume_id=${WANDB_RESUME_ID:-<new>}"
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
