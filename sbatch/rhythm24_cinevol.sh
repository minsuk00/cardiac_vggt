#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --job-name=rhythm24_cinevol
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/rhythm24_cinevol_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# CiNeVol (Vogt et al. 2026; baselines/cinevol, DEVIATIONS.md) on a 24-frame rhythm arm, all 180
# test subjects. One from-scratch 500-step fit + 24-volume export per subject (tools/run_cinevol.py).
#
#   ARM=af24 (default) | hrv24 | regular24      -> cohorts {cmrx2023,...,mnms}_$ARM
#
# Measured on 1 subject (L40S, 2026-09-21): 149 s total = prepare 9 + fit 123 + export 17, 2.4 GB.
# TIMING FAIRNESS: total_wall.sec is only comparable on ONE GPU model -> this refuses anything but
# an A40 (spgpu), the same card rhythm24_metrics.sh pins.
#
# NEVER OVERWRITES: a stamped subject is skipped; an existing unstamped arm dir is refused; every
# run writes to recon_breath.partial<pid>_<time> and publishes by one rename.
#
# The Grid4D encoder .so must already be built (it is, for sm_86+sm_89). To rebuild, on a GPU node:
#   source baselines/cinevol/env.sh && $CINEVOL_PY tools/run_cinevol.py --build-only
#
# NOT self-submitting:
#   sbatch --export=ALL,ARM=af24 sbatch/rhythm24_cinevol.sh        dry-run: bash ... --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}}
[ -d "$REPO/baselines/cinevol" ] || REPO=/home/minsukc/vggt-afsim
cd "$REPO" || exit 1
source baselines/cinevol/env.sh
ARM=${ARM:-af24}
ARM_NAME=${ARM_NAME:-cinevol}   # output arm dir; ARM_NAME=cinevol_motion KEEP_CKPT=1 = resp-EPE refit
N_SHARDS=${N_SHARDS:-8}
IDX=${SLURM_ARRAY_TASK_ID:-0}
DRYRUN=""; [ "${1:-}" = "--dry-run" ] && DRYRUN="--dry-run"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=== task $IDX/$N_SHARDS  arm=$ARM  arm_name=$ARM_NAME  keep_ckpt=${KEEP_CKPT:-0}  gpu=$GPU  build=$GRID4D_BUILD_DIR ==="
if [ -z "$DRYRUN" ]; then
    [[ "$GPU" == *A40* ]] || [ -n "${ALLOW_ANY_GPU:-}" ] || { echo "not an A40 ($GPU): timing would not be comparable, refusing"; exit 4; }
    [ -f "$GRID4D_BUILD_DIR/_hash_encoder.so" ] || { echo "encoder not built: run tools/run_cinevol.py --build-only first"; exit 5; }
fi
sleep $((IDX * 5))          # stagger the 8 extension loads / GPFS opens
$CINEVOL_PY -u tools/run_cinevol.py --arm "$ARM" --arm-name "$ARM_NAME" --shard "$IDX" "$N_SHARDS" \
    ${KEEP_CKPT:+--keep-checkpoint} ${SUBJECTS:+--subjects $SUBJECTS} $DRYRUN
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
