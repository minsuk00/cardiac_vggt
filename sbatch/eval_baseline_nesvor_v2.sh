#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --job-name=nesvor_v2
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/nesvor_v2_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# NeSVoR baseline campaign on the curated-v2 bundles: BOTH splits (val + test) x BOTH inputs
# (gated = upper bound; scatter = SAME-INPUT -> arm nesvor_scatter). NATIVE nesvor-t2 env
# (docs/90), one A40 (spgpu) per array task, J=1 so per-phase times are the true single-GPU
# cost — the SAME GPU class every VGGT arm is timed on (fairness = same GPU for all timed
# methods, not which GPU). Measured ~165 s/phase on A40 (docs/90) -> ~33 min/subject ->
# 291 subjects x 2 inputs ~= 320 GPU-h -> ~40 h wall on 8 GPUs (--time 48h; requeue-safe).
#
# Array shards the subject list; the driver skips already-stamped subjects (requeue-safe).
# NOT self-submitting — submit with `sbatch sbatch/eval_baseline_nesvor_v2.sh`.
# Restrict with SPLITS="val" / INPUTS="scatter" / SOURCES="cmrx2024 acdc".
# ============================================================================================
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt}
[ -d "$REPO/evaluation/src/engine" ] || REPO=/home/minsukc/vggt
cd "$REPO"
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export J=${J:-1}
SPLITS=${SPLITS:-"val test"}
INPUTS=${INPUTS:-"gated scatter"}
SOURCES=${SOURCES:-"cmrx2023 cmrx2024 cmrx2025 acdc mnms miitt ocmr"}
echo "shard $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT  (J=$J, gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))  splits=[$SPLITS] inputs=[$INPUTS]"
rc=0
for SPLIT in $SPLITS; do
  for INPUT in $INPUTS; do
    echo "=== nesvor  split=$SPLIT  input=$INPUT ==="
    $PY evaluation/src/engine/run_baselines.py --method nesvor --variant breath --split "$SPLIT" \
        --input "$INPUT" --sources $SOURCES --shard "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" || rc=1
  done
done
exit $rc
