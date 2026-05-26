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
# 200k-step head-only training run. Fresh run (loads VGGT-1B base on cold start,
# then trains only point_head — aggregator + patch_embed stay frozen) with SLURM
# auto-requeue across the walltime cap, resuming from its own ckpts/checkpoint_last.pt.
#
# 200k optimizer steps = 200 epochs  (limit_train_batches=1000, accum_steps=1, 1 GPU).
# =============================================================================

# ---- PER-RUN CONFIG (the only block that differs across the 4 run scripts) ----
CONFIG="mri_volume"
NGPU=1
EXP_TAG="mri_volume_t0t7"                 # → exp_name = <rev_ts>_${EXP_TAG}
RUN_OVERRIDES="max_epochs=200 t_target_phases=[0,7] logging.wandb_writer.tags=[mri_volume,t0t7,multiphase,headonly]"   # multiphase restricted to target phases {0,7}, head-only
MASTER_PORT=29536
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
# exp_name normally embeds a per-launch reverse timestamp, which would change on
# every requeue and point the trainer's checkpoint auto-detect at a fresh, empty
# dir. We pin it once on the first launch and reuse it on every restart so the
# trainer's resolve_resume_checkpoint() finds ckpts/checkpoint_last.pt.
REQUEUE_STATE="/home/minsukc/vggt/slurm_logs/.requeue_${SLURM_JOB_ID}.env"

if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    # --- Requeue restart: reuse pinned exp_name + EXTRA_OVERRIDES. The trainer
    #     auto-detects this run's own ckpts/checkpoint_last.pt from save_dir
    #     (resolve_resume_checkpoint prefers it over the config's base model.pt). ---
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
    # --- First launch: fresh run. Pin exp_name + EXTRA_OVERRIDES for requeue restarts. ---
    REV_TS=$((2000000000 - $(date +%s)))
    EXP_NAME="${REV_TS}_${EXP_TAG}"
    EXTRA_OVERRIDES="${RUN_OVERRIDES}"
    OVERRIDES="exp_name=${EXP_NAME} ${EXTRA_OVERRIDES}"
    echo "Fresh run: exp_name=${EXP_NAME}  overrides='${EXTRA_OVERRIDES}'"
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
# Disable filename globbing so Hydra list overrides like
# optim.frozen_module_names=[*patch_embed*] pass through to argv literally (word-split
# on spaces, but '*'/'[]' are NOT expanded against the cwd). Word-splitting of $OVERRIDES
# is still intended. Placed after all globbing (wandb-dir detection) above is done.
set -f
torchrun \
    --nproc_per_node=$NGPU \
    --master_port=$MASTER_PORT \
    training/launch.py \
    --config $CONFIG $OVERRIDES &
TORCHRUN_PID=$!
wait "$TORCHRUN_PID"
