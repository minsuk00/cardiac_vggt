#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --job-name=cinevol_nframes
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/cinevol_nframes_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# CiNeVol frames-per-slice probe (tools/cinevol_nframes_probe.py): one subject per array task,
# fitted on the first 12 / 24 / 48 frames of its af48 record, each scored against the af24 GT.
# Writes ONLY under temp/cinevol_nframes_probe/. Run `... setup` once on any node first.
#
#   sbatch sbatch/cinevol_nframes_probe.sh
# ============================================================================================
set -uo pipefail
REPO=${REPO:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
source baselines/cinevol/env.sh
IDX=${SLURM_ARRAY_TASK_ID:-0}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=== task $IDX  gpu=$GPU  build=$GRID4D_BUILD_DIR ==="
[ -f "$GRID4D_BUILD_DIR/_hash_encoder.so" ] || { echo "encoder not built"; exit 5; }
sleep $((IDX * 5))
$CINEVOL_PY -u tools/cinevol_nframes_probe.py run --index "$IDX" --ns ${NS:-12 24 48}
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
