#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#SBATCH --job-name=af12_niftymic
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af12_niftymic_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# NiftyMIC v2 (per-phase, same-input scatter stack) on the five af12 cohorts, all 180 test
# subjects, for the docs/120 §6 timing table — arm niftymic_scatter.
#
# TIMING FAIRNESS: the allocation is the main benchmark's CPU config, byte for byte
# (eval_baseline_svrtk_v2.sh / af12_svrtk.sh: standard, 16 CPUs, 48 GB, J=8 phases x OMP=2 = 16
# threads, one subject at a time), 30 concurrent array tasks (6 subjects each, ~230 s per
# subject). `standard` has no CPU feature tags: after the run, grep "cpu model" in every
# provenance.txt and re-run any subject that did not land on a Xeon Gold 6154.
#
# Per-subject timing = <subject>/niftymic_scatter/recon_breath/total_wall.sec. A stamped subject
# is skipped by run_baselines.py, so a resubmission only completes what is missing.
#
# NOT self-submitting:
#   sbatch sbatch/af12_niftymic.sh       dry-run (work list only): bash sbatch/af12_niftymic.sh --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export J=${J:-8} OMP=${OMP:-2}

N_SHARDS=${N_SHARDS:-30}
IDX=${SLURM_ARRAY_TASK_ID:-0}
SOURCES=$(for s in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do printf '%s_af12 ' "$s"; done)
DRYRUN=""; [ "${1:-}" = "--dry-run" ] && DRYRUN="--dry-run"

echo "=== task $IDX/$N_SHARDS  J=$J OMP=$OMP  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)  host=$(hostname) ==="
$PY evaluation/src/engine/run_baselines.py --method niftymic --variant breath --split test \
    --input scatter --sources $SOURCES --shard "$IDX" "$N_SHARDS" ${SUBJECTS:+--subjects $SUBJECTS} $DRYRUN
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
