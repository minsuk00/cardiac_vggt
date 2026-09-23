#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --job-name=af12_svrtk
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af12_svrtk_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# SVRTK 3D (per-phase, same-input scatter stack) on the five af12 cohorts, all 180 test
# subjects, for the docs/120 §6 timing table — arm svrtk3d_scatter.
#
# TIMING FAIRNESS: the allocation is the main benchmark's CPU config, byte for byte
# (eval_baseline_svrtk_v2.sh / rhythm24_fetal.sh: standard, 16 CPUs, 48 GB, J=8 phases x OMP=2 =
# 16 threads, one subject at a time), 24 concurrent array tasks (~7-8 subjects each, ~170 s per
# subject). `standard` has no CPU feature tags, so the CPU model cannot be pinned: after the run,
# grep "cpu model" in every provenance.txt and re-run any subject that did not land on a Xeon
# Gold 6154 (the existing svrtk3d_scatter + fetal_cmr_4d timings' CPU).
#
# Per-subject timing = <subject>/svrtk3d_scatter/recon_breath/total_wall.sec (all 12 phases at
# J=8; fresh run, no cached phases). A stamped subject is skipped by run_baselines.py, so a
# resubmission after a walltime cut only completes what is missing.
#
# NOT self-submitting:
#   sbatch sbatch/af12_svrtk.sh          dry-run (work list only): bash sbatch/af12_svrtk.sh --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export J=${J:-8} OMP=${OMP:-2} DEBUG=0

N_SHARDS=${N_SHARDS:-24}
IDX=${SLURM_ARRAY_TASK_ID:-0}
SOURCES=$(for s in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do printf '%s_af12 ' "$s"; done)
DRYRUN=""; [ "${1:-}" = "--dry-run" ] && DRYRUN="--dry-run"

echo "=== task $IDX/$N_SHARDS  J=$J OMP=$OMP  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)  host=$(hostname) ==="
$PY evaluation/src/engine/run_baselines.py --method svrtk3d --variant breath --split test \
    --input scatter --sources $SOURCES --shard "$IDX" "$N_SHARDS" ${SUBJECTS:+--subjects $SUBJECTS} $DRYRUN
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
