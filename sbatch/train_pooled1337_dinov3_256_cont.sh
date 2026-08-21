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
# POOLED-1337 — DINOv3 ViT-L/16 SECOND CYCLE (warm restart), 2026-08-16. Byte-identical to
# train_pooled1337_dinov3_256.sh except CKPT_ONLY + VARIANT_SUFFIX below.
#
# WHY: the first DINOv3 cycle (213317182_dinov3_256_pooled1337, job 57455588) COMPLETED all
# 300 epochs but was still climbing when the cosine ran out — over its last 50 epochs EF slope
# +0.068/50ep (t=+6.7) and heart-seg PSNR +0.089 dB/50ep, at LR ~1e-8. The three finished
# DINOv2-224 arms were flat over the same window (EF slope trend -0.008..+0.009, |t|<=1.3), so
# the DINOv3 arm stopped because the SCHEDULE ended, not because it converged. It also only
# escaped flat-EF at epoch 174 vs epoch 33-64 for the DINOv2 arms => slow optimization, not
# obviously a capacity ceiling. This run gives it a second full cosine cycle to test that.
#
# ⚠️ INTERPRETATION CAVEAT: the result is a 600-epoch DINOv3 vs a 300-epoch DINOv2. Parity is
# still a NEGATIVE verdict on the backbone (2x budget for a tie); only a clear win over the
# DINOv2-224 arms (seg gain 2.89-3.13 dB, EF slope 0.81-0.85) counts as a positive.
#
# ⚠️ THE SEED MUST BE STRIPPED. CKPT_ONLY is NOT weights-only despite the name: it feeds
# checkpoint.resume_checkpoint_path, and _load_resuming_checkpoint (trainer.py:433-451)
# restores optimizer/scaler/steps/prev_epoch from whatever file it is given. Pointed at the
# raw ckpts/checkpoint_last.pt (prev_epoch=299) it would set self.epoch=300, and the loop
# `while self.epoch < self.max_epochs` (trainer.py:708) exits immediately => ZERO training in
# a fresh exp dir. docs/37. The seed below was stripped to {"model": ...} only, so self.epoch
# stays 0 and `where = epoch/max_epochs` (trainer.py:1086) replays the full warmup + cosine:
#   torch.save({'model': torch.load('<ckpts>/checkpoint_last.pt')['model']}, SEED)
# ============================================================================================
# POOLED-1337 — DINOv3 ViT-L/16 ARM, img_size=256, 2026-08-14. Byte-identical recipe to
# train_pooled1337_dpt_augaggressive_224.sh except the DINOv3-related knobs, all carried by
# `--config exp_dinov3` (docs/77): backbone=dinov3_vitl16, img_size=256 (16² patches),
# patch_size auto-derived (backbone_ps resolver), warm-start from the hybrid seed
# vggt1b_dinov3_vitl16_seed.pt with strict=true (NOT vggt1b_base.pt — the DINOv3 backbone
# would stay random). Splat/render stays at native 256² (docs/73). A/B vs the 224 DINOv2
# aggressive arm isolates the backbone (+256-vs-224 perception res).
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
# WARM-START: FRESH FROM THE DINOv3 HYBRID SEED (exp_dinov3's resume path,
# ./scratch/base_weights/vggt1b_dinov3_vitl16_seed.pt, strict=true, weights-only) — NOT a
# cardiac ckpt. Leave RESUME_FROM and CKPT_ONLY empty for that. aggft (only `*patch_embed*`
# frozen ⇒ the 24+24 attention blocks, z_embedder, camera_token and point_head all train;
# the DINOv3 backbone lives under patch_embed and stays frozen).
#
# Regularizer arm = L2 diffusion (`tv=0`, `diffusion=1000`) — deliberate, docs/62 §4 #3.
# Respiration is ON in BOTH arms via default.yaml (`data.augmentation.respiratory.enable=true`,
# the proven "resp, z-only" recipe, docs/05) — it is a SEPARATE toggle from affine aug and is
# NOT part of this A/B.
#
# DEVIATIONS FROM default.yaml, applied via RECIPE_OVERRIDES below:
#   max_epochs 200 -> 300
#   gradient checkpointing true -> false (224 fits on A40; docs/76)
#   LR         5e-5 (unchanged — see below; must be set in THREE places, note at RECIPE_OVERRIDES)
# ⚠️ LR WAS 3e-4 AND IT KILLED BOTH ARMS. Jobs 55996915/16 (epoch 86/300) collapsed at epoch
# 16-17, 1-2 epochs after warmup put the LR at its 3e-4 peak: a single outlier batch
# (grad_point 34x normal, loss_diffusion 80x) drove the predicted DVF to a spatially CONSTANT
# field, which zeroes loss_diffusion exactly and severs the gradient to the aggregator for
# good (grad_aggregator < 1e-6 for the next ~70 epochs, while weight_decay=0.05 ground every
# trainable aggregator gamma down ~3x). Existing max_norm=1.0 clipping did NOT prevent it.
# 3e-4 is 6x default.yaml's 5e-5 for ~636M PRETRAINED params at batch size 1. Do not raise it
# again without a zero-gradient tripwire.
CONFIG="exp_dinov3"

# --- Recipe (shared by both arms; keep in sync with the noaug partner) ---
# ⚠️ LR IS THREE KNOBS, NOT ONE. `optim.optimizer.lr` only sets AdamW's initial value; the
# CompositeParamScheduler overwrites the LR every step from `where`, so setting the optimizer
# alone leaves the schedule at 5e-5 and the override silently does nothing after step 0.
# schedulers.0 = LinearParamScheduler (warmup 1e-8 -> peak, first 5%),
# schedulers.1 = CosineParamScheduler (peak -> 1e-8, remaining 95%).
PEAK_LR="5e-5"
# PINNED EF LABELS. compute_cardiac_phase.py rewrites scratch/data/whs/cardiac_phase.csv
# WHOLESALE (`open(OUT, "w")`) and a concurrent session is adding OCMR/MIITT to it — the md5
# moved twice on 2026-08-16 alone (c171fb -> e3ac90 -> 87cd43). The EF sweep reads this file
# once at startup and the EF eval is try/except-wrapped, so a mid-write read would degrade EF
# logging SILENTLY rather than crash. Pin to a frozen copy so this run is reproducible.
# Verified 2026-08-16: vs the pre-OCMR version the file only ADDS rows (8 ocmr_sax, and
# earlier 26 miitt/miitt_sax) — 0 existing rows changed, 0 of our cohort's rows changed — and
# the only generator diff since cycle 1 is one line gating `miitt_sax`. So these labels are
# identical to cycle 1's for all 1337 pooled subjects (848 cmrx + 341 mnms + 148 acdc); EF
# stays comparable to 213317182 and to the DINOv2 arms.
CARDIAC_PHASE_PIN="/home/minsukc/vggt/temp/cardiac_phase_pin_dinov3cont.csv"
RECIPE_OVERRIDES="max_epochs=300 \
img_size=256 \
model.gradient_checkpointing=false \
cardiac_phase_csv=${CARDIAC_PHASE_PIN} \
optim.optimizer.lr=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.0.end_value=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.1.start_value=${PEAK_LR} \
${EXPERIMENT_OVERRIDES:-}"

# --- The A/B variable: affine + photometric augmentation (tier=moderate, docs/46 §3 C2) ---
AUG_OVERRIDES="data.augmentation.enable=true data.augmentation.tier=aggressive"
VARIANT_TAG="dinov3_256${VARIANT_SUFFIX:-_cont}"
# --- Resume settings (leave BOTH empty for the fresh-from-base reference run) ---
# RESUME_FROM: continue a previous run's exp dir + same wandb run (crash recovery).
# EMPTY on purpose: RESUME_FROM would continue 213317182's exp dir + wandb run, and would need
# max_epochs=600 to train at all — which stretches the cosine so the LR jumps back to ~2.5e-5
# at epoch 300 anyway, while overwriting the first cycle's log_dir. Fresh dir instead.
RESUME_FROM=""
# CKPT_ONLY: load weights from a checkpoint into a fresh exp dir + fresh wandb run, replaying
# the full 300-epoch warmup+cosine. MUST be a {"model": ...}-only file — see the stripping
# note at the top of this script.
CKPT_ONLY="./scratch/base_weights/dinov3_256_ep299_weights_only.pt"

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
