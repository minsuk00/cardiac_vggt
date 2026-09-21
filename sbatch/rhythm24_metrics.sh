#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=08:00:00
#SBATCH --array=0-4
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
# Always through score/run.py --datasets <cohort>: a bare image_metrics.py defaults to cmrx2024
# and the rhythm cohorts reuse its subject AND arm names, so it would overwrite live campaign
# metrics (docs/110 s11.8).
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
IDX=${SLURM_ARRAY_TASK_ID:-0}
[ "$IDX" -lt "${#METHODS[@]}" ] || { echo "task $IDX: nothing to do"; exit 0; }
METHOD=${METHODS[$IDX]}
DATASETS=$(for s in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do printf '%s_%s ' "$s" "$ARM"; done)
echo "=== task $IDX  method=$METHOD  datasets=$DATASETS ==="
if [ "${1:-}" = "--dry-run" ]; then echo "dry-run: nothing run"; exit 0; fi
$PY evaluation/src/score/run.py --method "$METHOD" --datasets $DATASETS --split test
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
