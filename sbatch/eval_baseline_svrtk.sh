#!/bin/bash
#SBATCH --account=jjparkcv0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --job-name=eval_svrtk_val
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/eval_svrtk_val_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# SVRTK baseline generation over the 144-subject val cohort, breath arm only (docs/83).
# CPU-only: mirtk reconstruct, J phases in parallel per subject. Fair-timing config:
# DEBUG=0 (no -debug; ~6x faster, volume identical — rerun a subset under METHOD=svrtk3d_debug
# if the per-slice .dof transforms are ever needed).
#
# Array shards the subject list: --shard <task_id> <task_count>, so resizing --array just
# rebalances. Driver skips already-stamped subjects, so requeues/re-submissions only do the
# remainder (and never clobber a finished subject's timing provenance).
#
# Measured (gl1701, 4 CPUs, J=4): 259 s/subject -> at 8 CPUs/J=8 expect ~2-3 min/subject;
# 4 shards x 36 subjects ~= 1.5-2 h each. Walltime 6 h = generous margin.
#
# Submit: sbatch sbatch/eval_baseline_svrtk.sh
# ⚠️ From inside an interactive GPU allocation, clear the SLURM env first (`unset ${!SLURM_@}`).
# ============================================================================================

set -euo pipefail

# Under `sbatch`, slurmd runs a COPY of this script from its spool dir, so BASH_SOURCE-derived
# paths point at /var/spool/... — fall back to the submit dir (validated), then the main tree.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt}
[ -d "$REPO/evaluation/src/engine" ] || REPO=/home/minsukc/vggt
cd "$REPO"
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}

export J=${J:-8}          # parallel phases per subject == --cpus-per-task at OMP=1
export OMP=${OMP:-1}
export DEBUG=${DEBUG:-0}  # fair-timing default (see header)

echo "shard $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT  (J=$J OMP=$OMP DEBUG=$DEBUG)"
$PY evaluation/src/engine/run_baselines.py --method svrtk3d --variant breath --split val \
    --shard "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT"
