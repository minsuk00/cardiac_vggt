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
# RUNTIME, MEASURED IN SITU (job 55577444, 3x L40S on spgpu2, 2026-07-30): ~54 s/unit/worker.
#   1343 pool units x 54 s = ~20 GPU-hours  ->  ~8 h each on a 3-way array.
# Cost is dominated by FIXED per-unit overhead, not volume size: each unit spawns a fresh
# nnUNet_predict that loads all 5 fold checkpoints, plus 3 micromamba env switches, against only
# ~126 2D slices of actual inference. So per-unit time is near-constant across very different
# volume shapes -- do NOT estimate this from slice counts (doing so was 7x wrong).
# Two earlier figures, for calibration context: 116 s/unit measured on a CONTENDED interactive A40
# (another job on the node, cgroup-limited to 1 CPU) -- an upper bound, not representative; and
# 109 s/unit from the previous full run on A40 (job 53298619, 140 units in 4h15m).
# --time=20:00:00 (not 12h) leaves headroom: all array tasks can land on one node and the rate may
# degrade on the larger M&Ms volumes. The job is idempotent anyway (skips units whose
# heart_seg.nii.gz exists), so an overrun costs one free resubmit.
# OPTIMISATION LEFT ON THE TABLE: nnUNet_predict accepts a DIRECTORY of cases, so batching N
# subjects per call would amortise the 5-fold model load across N units and could cut this
# several-fold. Not done -- it changes the unit/idempotency structure.
#SBATCH --job-name=whs_seg
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=20:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/minsukc/vggt/slurm_logs/whs_seg_%A_%a.log
set -euo pipefail

REPO=/home/minsukc/vggt
MAMBA=/home/minsukc/.local/bin/micromamba
ENV_SH=$REPO/tools/nnunet_mnms_eval/env.sh
# Overridable so a smoke run can target a hand-picked subset (N_UNITS only caps to the FIRST N
# lines, which are all one dataset — useless for exercising a newly added source token).
WORKLIST=${WORKLIST:-$REPO/scratch/data/whs/worklist.txt}
ROWS=$REPO/scratch/data/whs/rows
mkdir -p "$REPO/slurm_logs" "$ROWS"

# NWORKERS must equal the array size, because the loop below strides by it. It used to be
# hardcoded 3, which made `sbatch --array=0-0` (1 GPU) silently process only every 3rd line and
# still print "worker 0 done" — two thirds of the cohort left unsegmented with no error. Derive it
# from SLURM_ARRAY_TASK_COUNT so it tracks --array automatically; override explicitly if needed.
NWORKERS=${NWORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}
WORKER=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=$(grep -c . "$WORKLIST")
# Optional cap for smoke tests: N_UNITS limits how many worklist lines are considered.
TOTAL=${N_UNITS:-$TOTAL}
MINE=$(( (TOTAL - WORKER + NWORKERS - 1) / NWORKERS ))
echo "worker $WORKER of $NWORKERS  ->  $MINE of $TOTAL units (stride $NWORKERS)"
if [ "$WORKER" -ge "$NWORKERS" ]; then
    echo "FATAL: worker id $WORKER >= NWORKERS $NWORKERS — units would be skipped"; exit 1
fi

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
    # FORCE=1 bypasses the existence skip. Needed because this test checks a PARTIAL output:
    # heart_seg.nii.gz is written early in assemble_whs.py, so a unit that dies later (as 75 did in
    # job 55577444, when the native-z TARGET_SPACING made the canonical affine degenerate) still
    # looks "done" and would silently never get a manifest row. See docs/39 §11c.
    # Use with a hand-built WORKLIST of just the affected units:
    #   WORKLIST=/path/to/redo.txt FORCE=1 sbatch --array=0-2 sbatch/whs_segment.sh
    if [ -f "$SIB" ] && [ -z "${FORCE:-}" ]; then echo "  [$1] exists, skip $DS $REGIME"; return 0; fi
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
