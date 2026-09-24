#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --array=0-24%10
#SBATCH --job-name=rhythm24_metrics
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/rhythm24_metrics_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# Image metrics (PSNR/SSIM/NCC) for one method per task on a 24-frame rhythm arm, all 5 source
# cohorts. Writes each subject's metrics.json + the scored cine_* volumes that the EF chain
# (sbatch/rhythm24_seg.sh STAGE=pred) segments -- so this runs AFTER the recons, BEFORE pred seg.
#
# NEEDS A GPU. pose_psf.py's 6-DOF registration search (evaluation/src/score/pose_psf.py:71,
# `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`) is what image_metrics
# spends its time in -- and falls back to CPU SILENTLY, no error, just slow. First submission of
# this job went to `standard` (no GPU): measured marginal rate 1.4-2.8 min/subject on CPU (~2x for
# 24 vs 12 frames on top of that), which projected to hours. Refuses to run without an A40 below.
#
# Always through score/run.py --datasets <cohort>: a bare image_metrics.py defaults to cmrx2024
# and the rhythm cohorts reuse its subject AND arm names, so it would overwrite live campaign
# metrics (docs/110 s11.8). No skip-if-exists guard on metrics.json -- safe to resubmit/overwrite.
#
#   sbatch --export=ALL,ARM=af24 sbatch/rhythm24_metrics.sh
# ============================================================================================
set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=/home/minsukc/micromamba/envs/svr/bin/python

ARM=${ARM:-af24}
METHODS=(fetal_cmr_4d vggt_final518_base_ep300 vggt_final518_diff1000_ep300
         vggt_final518_nogather_ep300 vggt_final518_hw0_ep300)
# METHOD_LIST="a b c" overrides the list (e.g. the classical baselines); size --array to 5 x its length.
[ -n "${METHOD_LIST:-}" ] && read -ra METHODS <<< "$METHOD_LIST"
# One task per (method, source) -- the SAME index layout as rhythm24_seg.sh STAGE=pred, so tasks
# 0-4 are fetal and 5-24 the four VGGT arms. run.py scores + aggregates per (dataset, method), so
# tasks share no output.
SOURCES=(cmrx2023 cmrx2024 cmrx2025 acdc mnms)
NS=${#SOURCES[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
[ "$IDX" -lt $(( ${#METHODS[@]} * NS )) ] || { echo "task $IDX: nothing to do"; exit 0; }
METHOD=${METHODS[$((IDX / NS))]}
DATASETS="${SOURCES[$((IDX % NS))]}_${ARM}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=== task $IDX  method=$METHOD  datasets=$DATASETS  gpu=$GPU ==="
if [ "${1:-}" = "--dry-run" ]; then echo "dry-run: nothing run"; exit 0; fi
[[ "$GPU" == *A40* ]] || [ -n "${ALLOW_ANY_GPU:-}" ] || { echo "not an A40 ($GPU) -- pose_psf falls back to CPU silently, refusing"; exit 4; }
$PY evaluation/src/score/run.py --method "$METHOD" --datasets $DATASETS --split test
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
