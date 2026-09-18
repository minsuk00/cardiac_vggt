#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --array=0-23
#SBATCH --job-name=svrtk_debug_motion
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/svrtk_debug_motion_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# SEPARATE ARM, motion-EPE only: SVRTK test/scatter with DEBUG=1 (-debug), which dumps each
# phase's per-slice rigid transforms (transformation*.dof: tx,ty,tz,rx,ry,rz — SVR's estimated
# motion correction, incl. through-plane tz). METHOD=svrtk3d_debug -> arm svrtk3d_debug_scatter,
# its own dir under <subject>/svrtk3d_debug_scatter/ — does NOT touch/overwrite the campaign's
# svrtk3d_scatter arm (recons, timings, metric_results) in any way; see evaluation/paths.py
# HEART_MASK_PAD wiring + run_baselines.py's METHOD-aware arm naming for how the two stay
# separate. Purely a motion-analysis rerun; not rescored for image metrics / EF/Dice.
#
# DEBUG=1 measured ~2.8x slower per phase than DEBUG=0 (clean same-subject/phase A/B, no
# parallel I/O contention: 30s -> 84s); real wall-clock under this job's J=8 parallel writes may
# run higher (run_svrtk3d.sh's own comment: up to ~6x, from concurrent 200MB/phase debug I/O).
# 24 shards x ~7-8 test subjects/shard x 12 phases (2 J=8 batches/subject) -> est. 1-3h/shard.
#
# SHARD_N is fixed at 24 independent of SLURM_ARRAY_TASK_COUNT, so a partial submission (e.g.
# `sbatch --array=0 ...` for a 1-shard timing/correctness probe) still shards correctly against
# the full 24-way split; a later `--array=0-23` submission (the header default) completes it.
# NOT self-submitting — submit with `sbatch sbatch/eval_baseline_svrtk_debug_motion.sh`.
# Test/scatter only (motion-EPE scope, docs pending) — not val, not gated.
# ============================================================================================
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt}
[ -d "$REPO/evaluation/src/engine" ] || REPO=/home/minsukc/vggt
cd "$REPO"
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export J=${J:-8} OMP=${OMP:-2} DEBUG=1 METHOD=svrtk3d_debug
SOURCES=${SOURCES:-"cmrx2023 cmrx2024 cmrx2025 acdc mnms miitt ocmr"}
SHARD_N=${SHARD_N:-24}
echo "shard $SLURM_ARRAY_TASK_ID / $SHARD_N  (J=$J OMP=$OMP DEBUG=$DEBUG METHOD=$METHOD)  split=test input=scatter"
$PY evaluation/src/engine/run_baselines.py --method svrtk3d --variant breath --split test \
    --input scatter --sources $SOURCES --shard "$SLURM_ARRAY_TASK_ID" "$SHARD_N"
