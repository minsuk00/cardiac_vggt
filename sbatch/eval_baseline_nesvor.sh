#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=16:00:00
#SBATCH --array=0-7
#SBATCH --job-name=eval_nesvor_val
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/eval_nesvor_val_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# NeSVoR baseline generation over the 144-subject val cohort, breath arm only (docs/83).
# One GPU per array task; each cardiac phase is an independent single-GPU INR fit. J=1 on
# purpose: J=1 per-phase times are the fair compute-cost unit (provenance.txt documents that
# J>1 times are contention-inflated). Parallelism comes from the array, not from sharing a GPU.
#
# Array shards the subject list: --shard <task_id> <task_count>, so resizing --array just
# rebalances. Driver skips already-stamped subjects, so requeues/re-submissions only do the
# remainder. The shell stages the 5.3 GB .sif to node-local /tmp once per node (flock'd).
#
# Measured: 192 s/phase on V100 (archived — the container ships V100-era builds, so V100 is
# actually its FASTEST arch; an L40S smoke measured 242 s/phase) -> ~38 min/subject.
# 8 shards x 18 subjects ~= 11.5 h each; walltime 16 h = margin. Max 8 GPUs by request.
#
# Submit: sbatch sbatch/eval_baseline_nesvor.sh
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

export J=${J:-1}          # fits per GPU — keep 1 for clean per-phase timings (see header)

echo "shard $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT  (J=$J, gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))"
$PY evaluation/src/engine/run_baselines.py --method nesvor --variant breath --split val \
    --shard "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT"
