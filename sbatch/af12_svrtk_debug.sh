#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --array=0-29
#SBATCH --job-name=af12_svrtk_debug
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af12_svrtk_debug_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# SVRTK 3D with -debug (per-slice .dof motion transforms, for the motion-EPE analysis
# evaluation/src/analysis/motion_epe/svrtk3d.py) on the five af12 cohorts, all 180 test subjects.
#
# SEPARATE ARM: METHOD=svrtk3d_debug DEBUG=1 --input scatter -> arm svrtk3d_debug_scatter (its own
# dir per subject; run_baselines.py appends _scatter to a user METHOD). It never touches the
# timing arm svrtk3d_scatter (sbatch/af12_svrtk.sh). Its timings are NOT the fair speed number
# (-debug is ~6x slower in parallel, 13 s -> 81 s per phase) — use svrtk3d_scatter for docs/120.
# Same allocation as the timing run so the recon volumes are identical (16 CPUs, J=8 x OMP=2);
# 30 concurrent array tasks (6 subjects each).
#
# NOT self-submitting:
#   sbatch sbatch/af12_svrtk_debug.sh    dry-run (work list only): bash sbatch/af12_svrtk_debug.sh --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export J=${J:-8} OMP=${OMP:-2} DEBUG=1 METHOD=svrtk3d_debug

N_SHARDS=${N_SHARDS:-30}
IDX=${SLURM_ARRAY_TASK_ID:-0}
SOURCES=$(for s in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do printf '%s_af12 ' "$s"; done)
DRYRUN=""; [ "${1:-}" = "--dry-run" ] && DRYRUN="--dry-run"

echo "=== task $IDX/$N_SHARDS  METHOD=$METHOD DEBUG=$DEBUG J=$J OMP=$OMP  host=$(hostname) ==="
$PY evaluation/src/engine/run_baselines.py --method svrtk3d --variant breath --split test \
    --input scatter --sources $SOURCES --shard "$IDX" "$N_SHARDS" ${SUBJECTS:+--subjects $SUBJECTS} $DRYRUN
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
