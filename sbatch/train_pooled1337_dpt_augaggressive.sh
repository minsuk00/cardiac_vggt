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

# --- Configuration ---
# ============================================================================================
# POOLED-1337 NATIVE-Z — AGGRESSIVE-AUG ARM, 2026-08-12. Identical to train_pooled1337_dpt_aug.sh
# except AUG_OVERRIDES tier=aggressive: escalated affine (rotate ±180 p.9, translate ±32, no
# per-axis scale) + flip + gamma 0.6-1.5 + bias ±0.4 + the docs/63 §5 acquisition-artifact
# post-ops (isotropic zoom 0.8-1.2, low-res, Gibbs, phase-encode ghosting). A/B vs the
# moderate promotion run (same cohort, seed, LR, epochs, heart-L1, resp).
# Its partner `train_pooled1337_dpt_noaug.sh` is byte-identical except `AUG_OVERRIDES`.
# The two together are a clean A/B on affine+photometric augmentation: everything else —
# cohort, seed, epochs, LR schedule, freeze, head, loss, respiratory — is identical.
# ============================================================================================
#
# Target-phase REFERENCE-SLICE conditioning (docs/24, docs/25): slot 0 = a real target-phase
# reference slice (mid-ventricular plane), marked via VGGT's native camera_token anchor; the
# model reads the target phase from slot-0's image content instead of a content-free target_t
# index. Fixes the flat-EF amplitude regression + the target_t=k/12 timing ambiguity.
#
# WARM-START: FRESH FROM BASE VGGT-1B (the config default resume path,
# ./scratch/base_weights/vggt1b_base.pt, strict=false) — NOT a cardiac ckpt. Leave RESUME_FROM
# and CKPT_ONLY empty for that. aggft (only `*patch_embed*` frozen ⇒ the 24+24 attention
# blocks, z_embedder, camera_token and point_head all train).
#
# Regularizer arm = L2 diffusion (`tv=0`, `diffusion=1000`) — deliberate, docs/62 §4 #3.
# Respiration is ON in BOTH arms via default.yaml (`data.augmentation.respiratory.enable=true`,
# the proven "resp, z-only" recipe, docs/05) — it is a SEPARATE toggle from affine aug and is
# NOT part of this A/B.
#
# DEVIATIONS FROM default.yaml, applied via RECIPE_OVERRIDES below:
#   max_epochs 200 -> 300
#   LR         5e-5 (unchanged — see below; must be set in THREE places, note at RECIPE_OVERRIDES)
# ⚠️ LR WAS 3e-4 AND IT KILLED BOTH ARMS. Jobs 55996915/16 (epoch 86/300) collapsed at epoch
# 16-17, 1-2 epochs after warmup put the LR at its 3e-4 peak: a single outlier batch
# (grad_point 34x normal, loss_diffusion 80x) drove the predicted DVF to a spatially CONSTANT
# field, which zeroes loss_diffusion exactly and severs the gradient to the aggregator for
# good (grad_aggregator < 1e-6 for the next ~70 epochs, while weight_decay=0.05 ground every
# trainable aggregator gamma down ~3x). Existing max_norm=1.0 clipping did NOT prevent it.
# 3e-4 is 6x default.yaml's 5e-5 for ~636M PRETRAINED params at batch size 1. Do not raise it
# again without a zero-gradient tripwire.
CONFIG="default"

# --- Recipe (shared by both arms; keep in sync with the noaug partner) ---
# ⚠️ LR IS THREE KNOBS, NOT ONE. `optim.optimizer.lr` only sets AdamW's initial value; the
# CompositeParamScheduler overwrites the LR every step from `where`, so setting the optimizer
# alone leaves the schedule at 5e-5 and the override silently does nothing after step 0.
# schedulers.0 = LinearParamScheduler (warmup 1e-8 -> peak, first 5%),
# schedulers.1 = CosineParamScheduler (peak -> 1e-8, remaining 95%).
PEAK_LR="5e-5"
RECIPE_OVERRIDES="max_epochs=300 \
optim.optimizer.lr=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.0.end_value=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.1.start_value=${PEAK_LR}"

# --- The A/B variable: affine + photometric augmentation (tier=moderate, docs/46 §3 C2) ---
AUG_OVERRIDES="data.augmentation.enable=true data.augmentation.tier=aggressive"
VARIANT_TAG="augaggr"
# --- Resume settings (leave BOTH empty for the fresh-from-base reference run) ---
# RESUME_FROM: continue a previous run's exp dir + same wandb run (crash recovery).
RESUME_FROM=""
# CKPT_ONLY: load weights from a checkpoint into a fresh exp dir. EMPTY here on purpose →
# fresh-from-base (the config's base-weights resume path is used). Ignored if RESUME_FROM set.
CKPT_ONLY=""

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_pooled1337_dpt_${VARIANT_TAG}"
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
# REQUEUE_STATE pins exp_name (and EXTRA_OVERRIDES) across requeues so checkpoint auto-detect
# finds checkpoint_last.pt instead of a fresh rev_ts dir. (See _archive/legacy_sbatch/train_mri_volume.sh for detail.)
REQUEUE_STATE="/home/minsukc/vggt/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    # Requeue restart: reuse pinned exp_name, resume from THIS run's checkpoint_last.pt.
    source "$REQUEUE_STATE"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    WANDB_DIR=$(ls -dt "./scratch/logs/${EXP_NAME}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
    if [ ! -z "$WANDB_DIR" ]; then
        WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
        OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
    fi
    echo "Requeue restart #${SLURM_RESTART_COUNT}: exp_name=${EXP_NAME}, resume_id=${WANDB_RESUME_ID:-<new>}, extra='${EXTRA_OVERRIDES}'"
else
    EXTRA_OVERRIDES=""
    if [ ! -z "$RESUME_FROM" ]; then
        EXP_NAME=$(basename "$RESUME_FROM")
        CKPT_PATH="${RESUME_FROM}/ckpts/checkpoint_last.pt"
        if [ ! -f "$CKPT_PATH" ]; then
            echo "ERROR: RESUME_FROM is set but $CKPT_PATH does not exist."
            exit 1
        fi
        # The recipe must be replayed here too: without it Hydra falls back to default.yaml
        # (max_epochs 200, lr 5e-5, augmentation.enable=true) and a resume would silently
        # change the run — including flipping aug ON for the noaug arm. Only the requeue
        # branch above replayed it; a manual RESUME_FROM restart did not.
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_PATH} ${EXTRA_OVERRIDES}"
        echo "Resuming (same exp + wandb) from: $CKPT_PATH"
        echo "  recipe: ${EXTRA_OVERRIDES}"
        WANDB_DIR=$(ls -dt "${RESUME_FROM}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
        if [ ! -z "$WANDB_DIR" ]; then
            WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
            OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
            echo "Auto-detected WandB resume_id: $WANDB_RESUME_ID"
        fi
    elif [ ! -z "$CKPT_ONLY" ]; then
        if [ ! -f "$CKPT_ONLY" ]; then
            echo "ERROR: CKPT_ONLY is set but $CKPT_ONLY does not exist."
            exit 1
        fi
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_${VARIANT_TAG}_pooled1337"
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_ONLY} ${EXTRA_OVERRIDES}"
        echo "Loading weights only from: $CKPT_ONLY (exp_name=${EXP_NAME}, fresh wandb run)"
    else
        # Mode 0 — FRESH FROM BASE VGGT-1B (config default resume path, strict=false).
        # The full recipe is spelled out in EXTRA_OVERRIDES so it persists VERBATIM across
        # requeues (the requeue branch replays this string; anything left implicit in the
        # config would silently revert if default.yaml were edited mid-run).
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_${VARIANT_TAG}_pooled1337"
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
        echo "Fresh-from-base run: exp_name=${EXP_NAME}"
        echo "  recipe: ${EXTRA_OVERRIDES}"
    fi
    { echo "EXP_NAME=${EXP_NAME}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
fi

echo "Running: python ... --config $CONFIG $OVERRIDES"

# --- SLURM auto-requeue signal forwarding (see _archive/legacy_sbatch/train_mri_volume.sh) ---
# Single-GPU / plain-python launch: forward SIGUSR1 straight to the python worker (it installs
# the requeue handler itself) instead of to torchrun's children.
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
