#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --job-name=fetal4d_v2
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/fetal4d_v2_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# Fetal CMR 4D (van Amerom 2019 engine, LV-area self-gating) baseline campaign on the curated-v2
# bundles: BOTH splits (val + test), ONE input — the self-gated rolled/ stack built from breath/
# (docs/105; --input gated --variant breath is the only accepted combination). CPU-only
# `mirtk reconstructCardiac`, one joint 4D solve per subject, OMP=16 threads on a 16-CPU
# allocation — the same allocation SVRTK (16) and NiftyMIC (8x2) report their wall-clock on, so
# total_wall.sec is comparable even though this engine scales poorly past ~4 cores (docs/105).
#
# PREREQUISITES (once, before submitting; the driver refuses subjects that lack them):
#   pooled.py --add-rolled  per source x split          -> <subject>/rolled/
#   fetal4d_gate.py dump / seg (GPU, nnU-Net) / assemble -> <subject>/fetal_cmr_4d/gate/
#
# Array shards the subject list (--shard <task> <count>); resizing --array just rebalances.
# The driver skips already-stamped subjects, so requeues/re-submissions only do the remainder.
# NOT self-submitting — submit with `sbatch sbatch/eval_baseline_fetal4d_v2.sh`.
# Restrict with SPLITS="val" / SOURCES="cmrx2024 acdc".
# ============================================================================================
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt}
[ -d "$REPO/evaluation/src/engine" ] || REPO=/home/minsukc/vggt
cd "$REPO"
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export OMP=${OMP:-$SLURM_CPUS_PER_TASK} ROBUST=${ROBUST:-1} RES=${RES:-1.4}
SPLITS=${SPLITS:-"val test"}
SOURCES=${SOURCES:-"cmrx2023 cmrx2024 cmrx2025 acdc mnms miitt ocmr"}
echo "shard $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT  (OMP=$OMP ROBUST=$ROBUST RES=$RES)  splits=[$SPLITS]"
rc=0
for SPLIT in $SPLITS; do
  echo "=== fetal_cmr_4d  split=$SPLIT  input=gated(rolled) ==="
  $PY evaluation/src/engine/run_baselines.py --method fetal_cmr_4d --variant breath --split "$SPLIT" \
      --input gated --sources $SOURCES --shard "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" || rc=1
done
exit $rc
