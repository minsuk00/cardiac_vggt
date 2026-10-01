#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --gpu_cmode=shared
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# --- Configuration ---
# ============================================================================================
# FINAL-MODEL SWEEP, 2026-09-11 — six arms on the curated-v2 split (docs/97, 628/90/180).
# Base recipe (ARM=base) = the then-paper 224 recipe (no-reg, gather 0.5, heart-L1 0.5, aggressive aug, resp on,
# cosine 5e-5 -> 0 over 300 epochs, seed 42) at img_size=518 with the DENSE training splat
# (loss.volume.splat_res=518). Rationale: on pooled1337 the 518-input + 518-splat combination
# is the only thing that reproduced the old-518 EF (MAE ~10.3 vs 12.9 for 224 no-reg); 224+518
# splat and 518+native splat did not (session FINAL-MODEL-RUNS, real nnU-Net EF, n=144).
# Every arm below differs from `base` by exactly the listed override(s), and each arm is its
# own config in training/config/ (diff1000 shipped as the paper model, so it is default.yaml):
#   ARM=base          diffusion 0, heart 0.5                    ablation_base.yaml
#   ARM=diff1000      + loss.volume.diffusion_weight=1000       default.yaml  (THE PAPER MODEL)
#   ARM=hw2           + loss.volume.heart_weight=2.0            ablation_hw2.yaml
#   ARM=hw0           + loss.volume.heart_weight=0.0            ablation_hw0.yaml
#   ARM=nogather      + loss.volume.gather_weight=0.0           ablation_nogather.yaml
#   ARM=diff1000_nogather  diff1000 + nogather                  ablation_diff1000_nogather.yaml
# Score with sbatch/eval_pooled_val.sh (v2 val is the default split) -> sbatch/eval_ef_dice.sh.
# Submit: ARM=<arm> bash $0  (from a login node, or `unset ${!SLURM_@}` first inside an
# interactive job). ~30 min/epoch on an L40S at 518/518 -> ~6 days per arm.
# ============================================================================================
#
# Target-phase REFERENCE-SLICE conditioning (docs/24, docs/25); WARM-START fresh from base
# VGGT-1B (config default resume path, strict=false) — leave RESUME_FROM and CKPT_ONLY empty.
# aggft (only `*patch_embed*` frozen). ⚠️ LR: 5e-5 peak, never 3e-4 (killed two arms, see
# sbatch/_archive/train_pooled1337_dpt_augaggressive_224.sh).
PEAK_LR="5e-5"

ARM="${ARM:-diff1000}"   # the paper model (= default.yaml)
case "$ARM" in
  base)         CONFIG="ablation_base" ;;
  diff1000)     CONFIG="default" ;;
  hw2)          CONFIG="ablation_hw2" ;;
  hw0)          CONFIG="ablation_hw0" ;;
  nogather)     CONFIG="ablation_nogather" ;;
  diff1000_nogather) CONFIG="ablation_diff1000_nogather" ;;
  *) echo "ERROR: ARM must be base|diff1000|hw2|hw0|nogather|diff1000_nogather (got '$ARM')"; exit 1 ;;
esac

# The shared recipe is spelled out (even though it equals default.yaml) so it persists
# VERBATIM across requeues and cannot drift if default.yaml is edited mid-run. The loss
# weights are NOT listed: they are what distinguishes the arms, so they live in $CONFIG
# (CONFIG is pinned in REQUEUE_STATE too); an override here would shadow the arm's config.
RECIPE_OVERRIDES="max_epochs=300 \
img_size=518 \
split_file=training/splits/pooled_curated_v2.txt \
dataset_name=curated898 \
limit_train_batches=628 \
model.gradient_checkpointing=true \
loss.volume.splat_res=518 \
optim.optimizer.lr=${PEAK_LR} \
${EXPERIMENT_OVERRIDES:-}"

AUG_OVERRIDES="data.augmentation.enable=true data.augmentation.tier=aggressive"
VARIANT_TAG="final518_${ARM}${VARIANT_SUFFIX:-}"
# --- Resume settings (leave BOTH empty for the fresh-from-base reference run) ---
# RESUME_FROM: continue a previous run's exp dir + same wandb run (crash recovery).
RESUME_FROM="${RESUME_FROM:-}"
# CKPT_ONLY: load weights from a checkpoint into a fresh exp dir. EMPTY here on purpose →
# fresh-from-base (the config's base-weights resume path is used). Ignored if RESUME_FROM set.
# GOTCHA (docs/37): a full checkpoint_last.pt is a FULL resume (optimizer + prev_epoch), strip
# to {"model": ...} for a real warm-start.
CKPT_ONLY="${CKPT_ONLY:-}"

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_${VARIANT_TAG}"
    if [ ! -z "$RESUME_FROM" ]; then
        JOB_NAME="${JOB_NAME}_resume"
    elif [ ! -z "$CKPT_ONLY" ]; then
        JOB_NAME="${JOB_NAME}_ckptonly"
    fi

    # The repo = this script's clone. Submitting FROM it makes SLURM_SUBMIT_DIR the repo, which
    # the job (a spool copy of this script, so no usable BASH_SOURCE) reads back below — on
    # first launch and on every requeue alike.
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    mkdir -p "$REPO/slurm_logs/"

    echo "Submitting: $JOB_NAME"
    cd "$REPO" && sbatch --job-name="$JOB_NAME" \
           --chdir="$REPO" \
           --output="$REPO/slurm_logs/${TIMESTAMP}_${JOB_NAME}_%j.log" \
           "$REPO/sbatch/$(basename "${BASH_SOURCE[0]}")"
    exit
fi

REPO="$SLURM_SUBMIT_DIR"
if [ ! -f "$REPO/training/launch.py" ]; then
    echo "ERROR: submit dir $REPO is not the repo root — submit with \`bash sbatch/train_final_518.sh\`."
    exit 1
fi

# --- Environment Setup ---
export MAMBA_EXE="${MAMBA_EXE:-$HOME/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate svr

cd "$REPO"

sleep $((SLURM_PROCID * 2))  # stagger startup

export WANDB_MODE=online

# --- Build Hydra overrides ---
# REQUEUE_STATE pins exp_name (and CONFIG, EXTRA_OVERRIDES) across requeues so checkpoint auto-detect
# finds checkpoint_last.pt instead of a fresh rev_ts dir. (See _archive/legacy_sbatch/train_mri_volume.sh for detail.)
# State files written before CONFIG was pinned carry no CONFIG line; those runs used
# CONFIG=default with every changed key (epochs, splat_res, loss weights) in EXTRA_OVERRIDES.
REQUEUE_STATE="${REPO}/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    # Requeue restart: reuse pinned exp_name, resume from THIS run's checkpoint_last.pt.
    CONFIG="default"  # pre-CONFIG-pin state files (see above); overwritten by `source` otherwise
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
        # and a resume would silently change the run.
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
        EXP_NAME="${REV_TS}_${VARIANT_TAG}_curated898"
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_ONLY} ${EXTRA_OVERRIDES}"
        echo "Loading weights only from: $CKPT_ONLY (exp_name=${EXP_NAME}, fresh wandb run)"
    else
        # Mode 0 — FRESH FROM BASE VGGT-1B (config default resume path, strict=false).
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_${VARIANT_TAG}_curated898"
        EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${AUG_OVERRIDES}"
        OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
        echo "Fresh-from-base run: exp_name=${EXP_NAME}"
        echo "  recipe: ${EXTRA_OVERRIDES}"
    fi
    { echo "EXP_NAME=${EXP_NAME}"; echo "CONFIG=${CONFIG}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
fi

echo "Running: python ... --config $CONFIG $OVERRIDES"

# --- SLURM auto-requeue signal forwarding (see _archive/legacy_sbatch/train_mri_volume.sh) ---
_forward_usr1() {
    echo "[requeue] batch shell caught SIGUSR1 — forwarding to python worker (${TRAIN_PID})"
    kill -USR1 "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID"
}
trap _forward_usr1 USR1

python -m training.launch \
    --config $CONFIG $OVERRIDES &
TRAIN_PID=$!
wait "$TRAIN_PID"
