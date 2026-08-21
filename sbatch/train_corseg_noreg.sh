#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:l40s:1
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

# ============================================================================================
# CORSEG-DICE SWEEP ARM, 2026-08-21 — runs from the vggt-dice WORKTREE (branch
# arm/corseg-dice, the CorSeg soft-Dice port). Recipe byte-identical to the awrobewn
# baseline (train_pooled1337_dpt_augaggressive_224.sh / run 213340611): img 224, aggressive
# aug, 300 epochs, LR 5e-5 (three knobs), heart_weight 0.5, diffusion 1000 — the ONLY
# change is ARM_OVERRIDES below. Judge vs awrobewn: SV-ratio (0.750 baseline), EF bias
# (-10.4), per-patient r (0.816); re-score decisive ckpts with nnU-Net Task114, NOT CorSeg
# (training signal and verdict must stay decoupled). Watch corseg_sat_frac: if most
# Dice-grad elements pin at the 5e-6 clamp, weight is exhausted — raise grad_clamp instead.
# ============================================================================================
REPO="/home/minsukc/vggt-dice"
CONFIG="default"

# --- THE ARM VARIABLE ---
ARM_OVERRIDES="loss.volume.diffusion_weight=0.0"
VARIANT_TAG="noreg224"

# --- Recipe (awrobewn, verbatim) ---
PEAK_LR="5e-5"
RECIPE_OVERRIDES="max_epochs=300 \
img_size=224 \
model.gradient_checkpointing=false \
optim.optimizer.lr=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.0.end_value=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.1.start_value=${PEAK_LR}"
AUG_OVERRIDES="data.augmentation.enable=true data.augmentation.tier=aggressive"
RESUME_FROM=""

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_dice_${VARIANT_TAG}"
    [ ! -z "$RESUME_FROM" ] && JOB_NAME="${JOB_NAME}_resume"
    mkdir -p ${REPO}/slurm_logs/
    echo "Submitting: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" \
           --output="${REPO}/slurm_logs/${TIMESTAMP}_${JOB_NAME}_%j.log" \
           "$0"
    exit
fi

# --- Environment Setup ---
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate svr

cd "$REPO"
export WANDB_MODE=online

# --- Build Hydra overrides (requeue-safe: EXTRA_OVERRIDES persisted verbatim) ---
REQUEUE_STATE="${REPO}/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

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
    if [ ! -z "$RESUME_FROM" ]; then
        EXP_NAME=$(basename "$RESUME_FROM")
        CKPT_PATH="${RESUME_FROM}/ckpts/checkpoint_last.pt"
        if [ ! -f "$CKPT_PATH" ]; then
            echo "ERROR: RESUME_FROM is set but $CKPT_PATH does not exist."; exit 1
        fi
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES} ${ARM_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_PATH} ${EXTRA_OVERRIDES}"
        echo "Resuming (same exp + wandb) from: $CKPT_PATH"
        WANDB_DIR=$(ls -dt "${RESUME_FROM}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
        if [ ! -z "$WANDB_DIR" ]; then
            WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
            OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
        fi
    else
        # Fresh-from-base VGGT-1B (config default resume path, strict=false).
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_${VARIANT_TAG}_pooled1337"
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES} ${ARM_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
        echo "Fresh-from-base run: exp_name=${EXP_NAME}"
        echo "  recipe: ${EXTRA_OVERRIDES}"
    fi
    { echo "EXP_NAME=${EXP_NAME}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
fi

echo "Running: python ... --config $CONFIG $OVERRIDES"

# --- SLURM auto-requeue signal forwarding ---
_forward_usr1() {
    echo "[requeue] batch shell caught SIGUSR1 — forwarding to python worker (${TRAIN_PID})"
    kill -USR1 "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID"
}
trap _forward_usr1 USR1

export PYTHONPATH=training:.
python training/launch.py \
    --config $CONFIG $OVERRIDES &
TRAIN_PID=$!
wait "$TRAIN_PID"
