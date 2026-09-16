#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --job-name=svrtk_v2
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/svrtk_v2_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# SVRTK baseline campaign on the curated-v2 bundles: BOTH splits (val + test) x BOTH inputs
# (gated = the baseline upper bound; scatter = the SAME-INPUT stack VGGT sees -> arm
# svrtk3d_scatter). CPU-only mirtk reconstruct, J=8 phases in parallel x OMP=2 threads = 16
# threads on a 16-CPU allocation; this allocation is what the reported per-cine wall-clock
# (total_wall.sec + provenance.txt) is measured on. Fair-timing DEBUG=0.
#
# Array shards the subject list (--shard <task> <count>); resizing --array just rebalances.
# The driver skips already-stamped subjects, so requeues/re-submissions only do the remainder.
# NOT self-submitting — submit with `sbatch sbatch/eval_baseline_svrtk_v2.sh`.
# Restrict with SPLITS="val" / INPUTS="scatter" / SOURCES="cmrx2024 acdc".
# ============================================================================================
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt}
[ -d "$REPO/evaluation/src/engine" ] || REPO=/home/minsukc/vggt
cd "$REPO"
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export J=${J:-8} OMP=${OMP:-2} DEBUG=${DEBUG:-0}
SPLITS=${SPLITS:-"val test"}
INPUTS=${INPUTS:-"gated scatter"}
SOURCES=${SOURCES:-"cmrx2023 cmrx2024 cmrx2025 acdc mnms miitt ocmr"}
echo "shard $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT  (J=$J OMP=$OMP DEBUG=$DEBUG)  splits=[$SPLITS] inputs=[$INPUTS]"
rc=0
for SPLIT in $SPLITS; do
  for INPUT in $INPUTS; do
    echo "=== svrtk3d  split=$SPLIT  input=$INPUT ==="
    $PY evaluation/src/engine/run_baselines.py --method svrtk3d --variant breath --split "$SPLIT" \
        --input "$INPUT" --sources $SOURCES --shard "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" || rc=1
  done
done
exit $rc
