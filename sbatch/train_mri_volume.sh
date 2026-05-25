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
# mri_volume: direct V_canon vs V_gt supervised loss, per-axis normalization,
# splat-based volume output, residual-DVF head, frozen aggregator.
CONFIG="mri_volume"
NGPU=1
MASTER_PORT=29521

# --- Resume settings (set ONE of these or leave both empty for a fresh run) ---
# RESUME_FROM: path to a previous run's experiment directory.
#   Continues SAME exp_name → overwrites that run's checkpoints.
#   Reuses SAME wandb run id → metrics overlay on the old run's wandb dashboard.
#   Use this when continuing the same experiment (e.g., a crash recovery).
RESUME_FROM=""

# CKPT_ONLY: load model weights from this checkpoint but start a FRESH exp dir
# + FRESH wandb run. Use this when pipeline semantics changed (e.g., multi-phase
# vs ED-only) and you don't want metrics to overlay onto the old dashboard.
# Ignored if RESUME_FROM is set.
CKPT_ONLY="/home/minsukc/vggt/scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="vggt_${CONFIG}"
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
# REQUEUE_STATE persists the pinned exp_name (and any inherited overrides) across
# requeues. exp_name normally embeds a per-launch reverse timestamp (${rev_ts:}),
# which would change on every requeue and point the trainer's checkpoint auto-detect
# at a fresh, empty dir → restart-from-scratch. We pin it once on the first launch and
# reuse it on every restart so get_resume_checkpoint() finds checkpoint_last.pt.
REQUEUE_STATE="/home/minsukc/vggt/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    # --- Requeue restart: reuse pinned exp_name, resume from THIS run's own
    #     checkpoint_last.pt (auto-detected from save_dir). Ignore the original
    #     RESUME_FROM/CKPT_ONLY seed — we want the latest weights, not the seed. ---
    source "$REQUEUE_STATE"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    # Re-attach the same wandb run so requeued segments overlay on one dashboard.
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
        # Mode 1 — continue same experiment dir + same wandb run.
        EXP_NAME=$(basename "$RESUME_FROM")
        CKPT_PATH="${RESUME_FROM}/ckpts/checkpoint_last.pt"
        if [ ! -f "$CKPT_PATH" ]; then
            echo "ERROR: RESUME_FROM is set but $CKPT_PATH does not exist."
            exit 1
        fi
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_PATH}"
        echo "Resuming (same exp + wandb) from: $CKPT_PATH"

        # Auto-extract WandB run id from <RESUME_FROM>/wandb/wandb/{run|offline-run}-<ts>-<id>/
        WANDB_DIR=$(ls -dt "${RESUME_FROM}/wandb/wandb/"{run,offline-run}-*/ 2>/dev/null | head -1)
        if [ ! -z "$WANDB_DIR" ]; then
            WANDB_RESUME_ID=$(basename "$WANDB_DIR" | sed -E 's|^(offline-)?run-[0-9_]+-||; s|/$||')
            OVERRIDES="$OVERRIDES +logging.wandb_writer.resume_id=${WANDB_RESUME_ID}"
            echo "Auto-detected WandB resume_id: $WANDB_RESUME_ID"
        else
            echo "Warning: no WandB run dir found under $RESUME_FROM/wandb/wandb/ — a new WandB run will be created."
        fi
    elif [ ! -z "$CKPT_ONLY" ]; then
        # Mode 2 — load model weights from ckpt; fresh exp dir + fresh wandb run.
        # The checkpoint's epoch + step counters ARE inherited (so wandb x-axis starts at
        # the resumed step, e.g. 200K — makes it visible that the model came from a ckpt
        # and isn't from-scratch). The fresh exp_name + wandb run_id provide separation
        # from the source experiment's dashboard.
        # Bump max_epochs so the inherited epoch counter (e.g. 200 from the 4-day ckpt)
        # leaves headroom for further training — otherwise `while epoch < max_epochs`
        # exits immediately and the job does zero train steps.
        if [ ! -f "$CKPT_ONLY" ]; then
            echo "ERROR: CKPT_ONLY is set but $CKPT_ONLY does not exist."
            exit 1
        fi
        # Multi-phase fine-tune: keep the config default limit_val_batches=200 so every one
        # of the 12 target phases gets ~16-17 val samples. The 30 val subjects are revisited
        # ~7× across the loop, but each revisit is at a DIFFERENT t_target (= seq_index % 12),
        # so the extra iters are genuine per-phase coverage, not redundancy. (A fixed-phase
        # run WOULD be redundant past one pass — there val_epoch auto-caps in code.)
        # Pin a stable exp_name (rev_ts prefix matches the config naming convention) so
        # requeue restarts reuse this dir. EXTRA_OVERRIDES carries max_epochs across requeues
        # (without it, the restart reverts to config max_epochs=200 < inherited epoch → no-op).
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_mri_volume_dynamic_axial_Cine_combined"
        EXTRA_OVERRIDES="max_epochs=500"
        OVERRIDES="exp_name=${EXP_NAME} checkpoint.resume_checkpoint_path=${CKPT_ONLY} ${EXTRA_OVERRIDES}"
        echo "Loading weights only from: $CKPT_ONLY (exp_name=${EXP_NAME}, fresh wandb run, max_epochs=500)"
    else
        # Mode 0 — fresh run. Pin a stable exp_name for requeue restarts.
        REV_TS=$((2000000000 - $(date +%s)))
        EXP_NAME="${REV_TS}_mri_volume_dynamic_axial_Cine_combined"
        OVERRIDES="exp_name=${EXP_NAME}"
        echo "Fresh run: exp_name=${EXP_NAME}"
    fi
    # Persist for requeue restarts.
    { echo "EXP_NAME=${EXP_NAME}"; echo "EXTRA_OVERRIDES=\"${EXTRA_OVERRIDES}\""; } > "$REQUEUE_STATE"
fi

echo "Running: torchrun ... --config $CONFIG $OVERRIDES"

# --- SLURM auto-requeue signal forwarding ---
# SLURM sends SIGUSR1 to THIS batch shell 120s before walltime (--signal=B:USR1@120).
# torchrun runs in the background; the trap forwards USR1 to torchrun's worker child
# (launch.py installs a SIGUSR1 handler that runs `scontrol requeue` then exits 0).
# `wait` after forwarding keeps the shell — and thus the job cgroup — alive until the
# worker has exited, so it isn't torn down mid-handler.
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
