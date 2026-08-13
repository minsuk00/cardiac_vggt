# ============================================================================================
# CORSEG-DICE ARM — shared launch body. Sourced by corsegdice_w*.sh.
# ============================================================================================
# THE EXPERIMENT (docs/69 follow-up, 2026-08-11; sibling of the heart-L1 series). The heart-L1
# arms upweight heart-ROI *intensity* fidelity; this arm instead adds a soft-Dice through a
# FROZEN CorSeg-CineSAX (MedNeXt-L, 2D per-slice) on every z-slice of V_canon against the
# on-disk per-phase GT labels (heart_seg_canonical[..., t_target], batch key heart_seg_t):
#     objective += corseg_weight * (1 - mean soft Dice over {LV_myo, LV_cav, RV})
# Rationale: Dice rewards putting the endocardial BOUNDARY in the right place — which is
# literally what contraction amplitude (amp_ratio) measures — where L1 rewards intensity
# fidelity anywhere in the ROI. Mechanism, label remap (GT 1=LV_cav/2=myo/3=RV vs CorSeg
# 1=myo/2=LV_cav/3=RV, verified IoU 0.90/0.71/0.89), differentiable preprocessing (replicates
# corseg_infer paper mode, validated Dice 0.918 on GT): training/corseg_dice.py in the tree.
#
# WEIGHT SCALE — do NOT reason from loss magnitudes. The Dice term is O(1) while loss_volume
# is ~0.01, but its gradient on V_canon is boundary-concentrated and enormous per unit weight:
# measured on a REAL early-training prediction, ||dDice/dV|| / ||dL1/dV|| ~= 370x, so
# GRAD-PARITY with the full L1 sits at corseg_weight ~= 0.0027. w002 (~0.74x parity) is the
# moderate arm; w100 (0.1, ~37x) is the Dice-dominant arm.
#
# ⚠️ POST-MORTEM (job 57023101, 2026-08-11; full record + falsified theories docs/70 §3b).
# PROVEN root cause: a FINITE loss whose backward produced non-finite grads passed the
# trainer's loss-side NaN guard, and under bf16 the GradScaler (which skips inf-grad steps
# for fp16) is disabled -> one optimizer.step() wrote NaN into 930 weight tensors -> every
# forward NaN forever (docs/64-lookalike: grads log 0, meters freeze bit-identical). Fixes
# baked into THIS TREE: (1) trainer.py skip-step guard on non-finite grad norms (the
# load-bearing fix; logs "SKIPPING optimizer step"); (2) corseg branch is an fp32 island
# with z-score sd floor 1e-3; (3) per-voxel grad clamp 5e-6 on the Dice branch (insurance
# for its measured ~100x spike tail; keeps w100 bounded — 39% of voxels saturate at cap);
# (4) chunked+checkpointed MedNeXt forward (4 slices/chunk): peak 6.85 GB @ D=21 (was ~20GB,
# OOM'd) at ~386 ms/step worst case. VERIFIED: 100-step smokes at w=0.002 AND w=0.1 — zero
# NaN objectives, 2/1 skipped steps, healthy grads throughout, loss_corseg decreasing in
# both. A few "SKIPPING optimizer step" lines per epoch are EXPECTED and benign; a streak
# of them is not — investigate before assuming.
#
# TREE: /home/minsukc/vggt-arm-corseg — copy of vggt-arm-heart (incl. its gpu_aug ROI-warp
# fix) + corseg files: training/corseg_dice.py (new), training/loss.py, training/config/
# default.yaml, training/data/gpu_aug.py (heart_seg_t co-warp), training/data/datasets/
# mri_dataset.py (heart_seg_t loading), training/data/composed_dataset.py (key whitelist —
# GOTCHA: sample keys are explicitly whitelisted there; a new batch key silently vanishes
# without that entry). heart_weight defaults 0.0 in this tree — the Dice term is isolated;
# a combined arm would set both.
#
# COHORT: identical to the heart-L1 series — cmrx24only.txt (235/29/30) on CURRENT v2 data,
# NATIVE-Z (required: heart_seg_canonical is on the native canonical grid; the fixed12
# wrapper would drop the key and loss.py raises). CONTROL: judge against heartl1_w000
# (job 56990551) — same tree lineage, corseg_weight=0 == heart_weight=0 == production loss.
#
# ⚠️ HYGIENE: amp_ratio (tools/e0_*) is CorSeg-derived and this arm TRAINS on CorSeg — the
# decisive checkpoints MUST be re-scored with nnU-Net (Task114) before any ship decision.
# ⚠️ metric_psnr_3d_heartseg region overlaps the training signal here too — judge on
# amp_ratio (nnU-Net-confirmed) and motion PSNR.
# ============================================================================================

TREE="${TREE:-/home/minsukc/vggt-arm-corseg}"
CONFIG="default"

MAX_EPOCHS=300

# ⚠️ LR IS THREE KNOBS, NOT ONE (see heartl1_common.sh).
PEAK_LR="5e-5"
RECIPE_OVERRIDES="max_epochs=${MAX_EPOCHS} \
optim.optimizer.lr=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.0.end_value=${PEAK_LR} \
optim.options.lr.0.scheduler.schedulers.1.start_value=${PEAK_LR}"

COHORT_OVERRIDES="split_file=training/splits/cmrx24only.txt dataset_name=cmrx24only \
limit_train_batches=235 limit_val_batches=58"

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_corsegdice_${VARIANT_TAG}"
    mkdir -p /home/minsukc/vggt/slurm_logs/
    echo "Submitting: $JOB_NAME  (tree=$TREE, corseg_weight=$CORSEG_WEIGHT)"
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

cd "$TREE"
sleep $((SLURM_PROCID * 2))

export WANDB_MODE=online

REQUEUE_STATE="/home/minsukc/vggt/slurm_logs/.requeue_${SLURM_JOB_ID}.env"
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    source "$REQUEUE_STATE"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    WANDB_DIR=$(ls -dt "./scratch/logs/${EXP_NAME}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
    if [ ! -z "$WANDB_DIR" ]; then
        WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
        OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
    fi
    echo "Requeue restart #${SLURM_RESTART_COUNT}: exp_name=${EXP_NAME}"
elif [ -n "$RESUME_EXP_NAME" ]; then
    # Manual resume of an existing exp dir (e.g. after moving partitions): same exp_name
    # -> the trainer picks up ckpts/checkpoint_last.pt; reuse the wandb run id as above.
    EXP_NAME="$RESUME_EXP_NAME"
    EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${COHORT_OVERRIDES} loss.volume.corseg_weight=${CORSEG_WEIGHT}"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    WANDB_DIR=$(ls -dt "./scratch/logs/${EXP_NAME}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
    if [ ! -z "$WANDB_DIR" ]; then
        WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
        OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
    fi
    echo "Manual resume: exp_name=${EXP_NAME}"
    { echo "EXP_NAME=${EXP_NAME}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
else
    REV_TS=$((2000000000 - $(date +%s)))
    EXP_NAME="${REV_TS}_mri_volume_corsegdice_${VARIANT_TAG}_dynamic_axial_cmrx24only"
    EXTRA_OVERRIDES="${RECIPE_OVERRIDES} ${COHORT_OVERRIDES} loss.volume.corseg_weight=${CORSEG_WEIGHT}"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    echo "Fresh-from-base run: exp_name=${EXP_NAME}"
    echo "  recipe: ${EXTRA_OVERRIDES}"
    { echo "EXP_NAME=${EXP_NAME}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
fi

echo "Running from $TREE: python training/launch.py --config $CONFIG $OVERRIDES"

export PYTHONPATH=training:.
python training/launch.py --config $CONFIG $OVERRIDES &
TRAIN_PID=$!
wait "$TRAIN_PID"
