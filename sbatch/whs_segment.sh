#!/usr/bin/env bash
# Whole-heart seg + ROI across all datasets. 3 GPU workers (array 0-2); each worker strides through
# the worklist and processes its units SEQUENTIALLY: prep (svr) -> nnUNet_predict Task114 2d (nnunet)
# -> assemble seg/ROI siblings (svr). Idempotent (skips units whose heart_seg.nii.gz already exists),
# so it's safe to re-run / resume. Small intermediates live on node-local $TMPDIR; only the 4D masks +
# manifest rows hit GPFS.
#
# Generate the worklist first:  micromamba run -n svr python tools/nnunet_mnms_eval/make_whs_worklist.py
# Smoke (1 worker, first few units):  N_UNITS=4 sbatch --array=0-0 sbatch/whs_segment.sh
# Full:  sbatch sbatch/whs_segment.sh
#
#SBATCH --job-name=whs_seg
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/minsukc/vggt/slurm_logs/whs_seg_%A_%a.log
set -euo pipefail

REPO=/home/minsukc/vggt
MAMBA=/home/minsukc/.local/bin/micromamba
ENV_SH=$REPO/tools/nnunet_mnms_eval/env.sh
WORKLIST=$REPO/scratch/data/whs/worklist.txt
ROWS=$REPO/scratch/data/whs/rows
mkdir -p "$REPO/slurm_logs" "$ROWS"

NWORKERS=3
WORKER=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$(grep -c . "$WORKLIST")
# Optional cap for smoke tests: N_UNITS limits how many worklist lines are considered.
TOTAL=${N_UNITS:-$TOTAL}
echo "worker $WORKER/$NWORKERS over $TOTAL units"

# Retry wrapper: `micromamba run` shares ~/.cache/mamba/proc across concurrent workers and can
# transiently fail to acquire that lock ("libmamba Could not set lock"). Retry with backoff so a
# lock collision doesn't silently drop a unit.
mrun () { local i; for i in 1 2 3 4 5; do "$@" && return 0; echo "  mamba retry $i (rc=$?)"; sleep $((i * 8)); done; return 1; }

process_unit () {   # $1 = 1-based worklist line number; returns non-zero on failure (unit left un-done)
    local LINE DS REGIME P SIB WORK
    LINE=$(sed -n "${1}p" "$WORKLIST")
    [ -z "$LINE" ] && return 0
    read -r DS REGIME P <<< "$LINE"
    if [ "$DS" = "cmrx" ]; then SIB="$P/heart_seg.nii.gz"; else SIB="$(dirname "$P")/heart_seg.nii.gz"; fi
    if [ -f "$SIB" ]; then echo "  [$1] exists, skip $DS $REGIME"; return 0; fi
    echo "  [$1] $DS $REGIME $P"

    WORK="${TMPDIR:-/tmp}/whs_${SLURM_ARRAY_JOB_ID:-x}_${WORKER}_$1"
    rm -rf "$WORK"; mkdir -p "$WORK/in" "$WORK/seg"
    mrun "$MAMBA" run -n svr python "$REPO/tools/nnunet_mnms_eval/prep_one.py" \
        --dataset "$DS" --regime "$REGIME" --path "$P" --out_dir "$WORK/in" || { rm -rf "$WORK"; return 1; }
    mrun "$MAMBA" run -n nnunet bash -c "source '$ENV_SH' && \
        nnUNet_predict -i '$WORK/in' -o '$WORK/seg' -t 114 -m 2d -tr nnUNetTrainerV2_MMS" || { rm -rf "$WORK"; return 1; }
    mrun "$MAMBA" run -n svr python "$REPO/tools/nnunet_mnms_eval/assemble_whs.py" \
        --dataset "$DS" --regime "$REGIME" --path "$P" --seg_dir "$WORK/seg" --manifest_dir "$ROWS" || { rm -rf "$WORK"; return 1; }
    rm -rf "$WORK"
}

# Stride: worker w handles worklist lines w, w+N, w+2N, ... (1-based lines = idx+1)
for (( idx=WORKER; idx<TOTAL; idx+=NWORKERS )); do
    process_unit $((idx + 1)) || echo "  [$((idx+1))] FAILED (continuing)"
done
echo "worker $WORKER done"
