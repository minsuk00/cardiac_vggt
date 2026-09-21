#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --job-name=rhythm24_fetal
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/rhythm24_fetal_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# Fetal CMR 4D (self-gated) on a 24-frame rhythm arm, all 180 test subjects (docs/115).
#
#   ARM=af24 (default) | hrv24 | regular24      -> cohorts {cmrx2023,...,mnms}_$ARM
#
# TIMING FAIRNESS: the allocation is the main benchmark's CPU config, byte for byte
# (eval_baseline_fetal4d_v2.sh / eval_baseline_svrtk_v2.sh: standard, 16 CPUs, 48 GB, OMP=16, one
# subject at a time), capped at 24 concurrent jobs. `standard` has no CPU feature tags, so the CPU
# model cannot be pinned: after the run, grep "cpu model" in every provenance.txt and re-run any
# subject that did not land on a Xeon Gold 6154.
#
# The gate must already exist (sbatch/gate_af24.sh). NOT self-submitting:
#   sbatch --export=ALL,ARM=af24 sbatch/rhythm24_fetal.sh      dry-run: bash ... --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export OMP=${OMP:-${SLURM_CPUS_PER_TASK:-16}} ROBUST=${ROBUST:-1} RES=${RES:-1.4}
unset T                      # run_fetal4d.sh reads NF/NCP from the manifest only (docs/114)

ARM=${ARM:-af24}
N_SHARDS=${N_SHARDS:-24}
IDX=${SLURM_ARRAY_TASK_ID:-0}
SOURCES=$(for s in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do printf '%s_%s ' "$s" "$ARM"; done)
DRYRUN=""; [ "${1:-}" = "--dry-run" ] && DRYRUN="--dry-run"

echo "=== task $IDX/$N_SHARDS  arm=$ARM  OMP=$OMP  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) ==="
$PY evaluation/src/engine/run_baselines.py --method fetal_cmr_4d --variant breath --split test \
    --input gated --sources $SOURCES --shard "$IDX" "$N_SHARDS" ${SUBJECTS:+--subjects $SUBJECTS} $DRYRUN
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
