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
# POOLED-1337 NATIVE-Z LONG RUN — arm A/B pair, 2026-08-01. THIS SCRIPT = **AUG ON**.
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
# ============================================================================================
# THE ONE DEVIATION FROM `train_pooled1337_dpt_aug.sh`: loss.volume.gather_weight 0.5 -> 0.0
# ============================================================================================
# WHY (2026-08-06): the fresh-from-base aug/noaug pair (56257009/11) learns breathing well
# (resp slope 0.87, r 0.91, EPE 1.44mm) but NEVER learns target-phase conditioning: predicted
# EF 8.3% vs 58.7% true, and the cross-slot response to the reference phase collapses from
# 0.030 (base weights) to 0.005 by epoch 50 and is FLAT to epoch 87. Measured facts:
#   * the objective PRESERVES conditioning if the model already has it (start from 4wok: stays
#     ~0.10 over 300 steps) -> this is a BOOTSTRAP failure, not destruction, not undertraining
#   * breathing can remove 7.2x more full-volume L1 than cardiac phase (model-free, all 5
#     sources) -> the phase-blind "just correct breathing" solution is a deep local optimum
#   * gather exists to restore the sharp THROUGH-PLANE placement gradient (loss.py, docs/37/38)
#     and docs/46 C1 calls it decisive for BREATHING quality -> it sharpens the task that was
#     already winning, and is degenerate for in-plane cardiac motion (pure intensity matching:
#     inside blood pool / myocardium many positions satisfy it equally, including unmoved)
# HYPOTHESIS UNDER TEST: gather is what deepens the basin so conditioning never bootstraps.
#   conditioning emerges here  -> gather is the lever; fresh training is fixable
#   it does not                -> gather is CLEARED; remaining suspects are the pooled cohort
#                                 (~87 vs 4wok's ~833 exposures/subject) and the resp params
# READ IT EARLY with the cross-slot probe (cond_ratio), not EF: healthy ~0.08 vs stuck ~0.005,
# separable within a few epochs, whereas EF took 87 epochs to be unambiguous.
# NOTE 4wok itself had NO gather (the term did not exist until 2026-07-07, commit d925ddd).
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

# ============================================================================================
# CMRX24-ONLY COHORT CAUSE TEST (2026-08-07). Same as train_pooled1337_dpt_aug_nogather.sh
# except the cohort: training/splits/cmrx24only.txt = the CMRxRecon2024 subset of pooled.txt
# (235 train / 29 val / 30 test, uniform D=12, single source) — reverting the pooled-cohort
# change (size, 5-source heterogeneity, variable S=D 5-21) in one run, the closest match to
# 4wok's setup under today's code. gather stays 0 here only because this script was cloned
# from the nogather arm AFTER gather was refuted as the cause (job 56616177 collapsed
# identically) — with gather cleared, it is not a confound.
#   conditioning emerges (cond_ratio ~0.08 by ep20-30) -> the pooled cohort change is the cause
#   it collapses (~0.005)                              -> cohort ALSO cleared; suspect deeper
# ============================================================================================
# --- The A/B variable: affine + photometric augmentation (tier=moderate, docs/46 §3 C2) ---
# ============================================================================================
# FIXED-12 VARIANT (2026-08-07): identical to train_cmrx24only_dpt_aug_nogather.sh except the
# dataset class — temp/fixed12_dataset.py::MRIDatasetFixed12 pads/crops every subject's cached
# bundle to D=12 @ 12mm (CMRx24 is natively 12mm, so no resampling), reproducing the
# pre-native-z fixed (256,256,12) grid geometry through today's code. Needs temp/ on
# PYTHONPATH (set below). 2x2 readout vs the native-z partner (56722507):
#   both bootstrap        -> cohort was the cause (native-z fine)
#   only fixed12 does     -> native-z variable-D/z geometry is the cause
#   both collapse         -> loss/splat/trainer code rewrite is the suspect -> old-code run
# NOTE: heart_roi_canonical is native-D on disk, so recov_frac/heart-ROI metrics are skipped
# for padded subjects here; EF + cond_ratio (the verdict metrics) are unaffected.
# ============================================================================================
AUG_OVERRIDES="data.augmentation.enable=true data.augmentation.tier=moderate \
loss.volume.gather_weight=0.0 \
split_file=training/splits/cmrx24only.txt dataset_name=cmrx24only \
limit_train_batches=235 limit_val_batches=58 \
data.train.dataset.dataset_configs.0._target_=fixed12_dataset.MRIDatasetFixed12 \
data.val.dataset.dataset_configs.0._target_=fixed12_dataset.MRIDatasetFixed12"
# Tagged `aug_nogather` so the exp dir + job name are distinguishable from the running aug arm.
# VARIANT_TAG feeds ONLY EXP_NAME and JOB_NAME — it is not a hyperparameter.
VARIANT_TAG="aug_nogather_cmrx24_fixed12"
# --- Resume settings (leave BOTH empty for the fresh-from-base reference run) ---
# RESUME_FROM: continue a previous run's exp dir + same wandb run (crash recovery).
RESUME_FROM=""
# CKPT_ONLY: load weights from a checkpoint into a fresh exp dir. EMPTY here on purpose →
# fresh-from-base (the config's base-weights resume path is used). Ignored if RESUME_FROM set.
CKPT_ONLY=""

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_cmrx24only_dpt_${VARIANT_TAG}"
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
        EXP_NAME="${REV_TS}_mri_volume_dpt_${VARIANT_TAG}_dynamic_axial_cmrx24only"
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_ONLY} ${EXTRA_OVERRIDES}"
        echo "Loading weights only from: $CKPT_ONLY (exp_name=${EXP_NAME}, fresh wandb run)"
    else
        # Mode 0 — FRESH FROM BASE VGGT-1B (config default resume path, strict=false).
        # The full recipe is spelled out in EXTRA_OVERRIDES so it persists VERBATIM across
        # requeues (the requeue branch replays this string; anything left implicit in the
        # config would silently revert if default.yaml were edited mid-run).
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_mri_volume_dpt_${VARIANT_TAG}_dynamic_axial_cmrx24only"
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

export PYTHONPATH=temp:training:.
python training/launch.py \
    --config $CONFIG $OVERRIDES &
TRAIN_PID=$!
wait "$TRAIN_PID"
