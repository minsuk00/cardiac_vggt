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
# REFINER run B (JOINT + SSIM) — 3D UNet refiner on the splat, trained JOINTLY with an
# aggregator-finetune from VGGT-1B base. Refiner from scratch; aggregator + point_head
# finetuned (only DINOv2 patch_embed frozen). Breathing ON, z embedding, NO input-t,
# target_t ON. Two-term deep-supervised loss L_pre(V_canon)+L_post(V_refined), with
# L_post = λ·L1 + 0.1·(1−SSIM_2d), λ=1. NOTE: in the joint run the SSIM gradient reaches
# the point head/aggregator via the residual V_refined=V_canon+Δ + the differentiable
# splat (only the frozen run isolates it) — intentional co-adaptation.
#
# Counterpart to run A (frozen): A isolates the pure splat-deblur gain (VGGT frozen);
# B lets the geometry co-adapt with the refiner. Refiner ≈ 0.35M params (anisotropic,
# coverage-aware). find_unused_parameters=true (disabled heads emit grad-less params).
# =============================================================================

# ---- PER-RUN CONFIG ----
CONFIG="mri_volume"
NGPU=1
EXP_TAG="mri_refiner_joint_ssim"
RUN_OVERRIDES="max_epochs=200 enable_refiner=true refiner_use_coverage=true data.augmentation.respiratory.enable=true use_z_pose_embedding=true use_t_pose_embedding=false use_target_t_pose_embedding=true optim.frozen_module_names=[*patch_embed*] distributed.find_unused_parameters=true loss.volume.refiner_lambda=1.0 loss.volume.refiner_ssim_weight=0.1 logging.wandb_writer.tags=[mri_volume,allphases,multiphase,aggft,resp,no_t,refiner,joint,ssim]"
MASTER_PORT=29566
# ------------------------

# Fresh from VGGT-1B base (config default resume_checkpoint_path). Refiner from scratch.
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
