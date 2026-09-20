#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24g
#SBATCH --time=8:00:00
#SBATCH --array=0-29
#SBATCH --job-name=af_disambig_fetal
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af_disambig_fetal_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# docs/113 — CUT-OFF vs HOLD disambiguation. Fetal CMR 4D (SELF-GATED ONLY) on the two new
# rhythm cohorts, all 38 cmrx2024 test subjects.
#
# WHY: `af` draws R-R around 1.0, so a short beat is cut off mid-cycle and a long one parks at
# full ED. Those are two faces of one mechanic and are anti-correlated at r=-0.50 across
# subjects, so the `af` cohort can NEVER say which of them costs Fetal CMR 4D its EF. These two
# cohorts bound the R-R draw on one side only and therefore separate them:
#     cmrx2024_af_rvr    R-R ~ N(0.80, 0.15) capped at 1.00  -> 0.88 cut-offs/slice, hold 0.071
#     cmrx2024_af_pause  R-R ~ N(1.30, 0.20) floored at 1.05 -> 0.00 cut-offs/slice, hold 0.245
# Both are real AF phenotypes (rapid vs slow ventricular response), not synthetic extremes.
#
# ONLY the self-gated arm is built. The oracle / oracle-balanced arms were diagnostics for the
# coverage-vs-timing question that docs/111 s3d already settled by intervention; re-running them
# here would add 152 reconstructions and answer nothing new.
# 2 cohorts x 1 arm = 2 combos x 15 shards = 30 array tasks, 76 reconstructions.
#
# 16 CPUs (OMP=16), matching the PRODUCTION fetal baseline sbatch/eval_baseline_fetal4d_v2.sh
# ("one joint 4D solve per subject, OMP=16 threads on a 16-CPU") rather than the 4 CPUs the
# earlier af_full_fetal.sh used -- ~8.5 min/subject instead of ~13.1.
#
# PREREQUISITES (the driver refuses subjects that lack them):
#   tools/build_af_bundle.py --source cmrx2024 --arms af_rvr af_pause --subjects <all 38>
#   fetal4d_gate.py dump / seg / assemble --work-tag afsim_disambig --sources <both cohorts>
#
# The driver skips already-stamped subjects, so a requeue only does the remainder.
# NOT self-submitting — submit with `sbatch sbatch/af_disambig_fetal.sh`.
# Dry-run the work list first: bash sbatch/af_disambig_fetal.sh --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export OMP=${OMP:-${SLURM_CPUS_PER_TASK:-16}} ROBUST=${ROBUST:-1} RES=${RES:-1.4}

COMBOS=(
  "cmrx2024_af_rvr fetal_cmr_4d"
  "cmrx2024_af_pause fetal_cmr_4d"
)
N_SHARDS=${N_SHARDS:-15}   # 2 combos x 15 shards = 30 array tasks, ~2-3 subjects each
IDX=${SLURM_ARRAY_TASK_ID:-0}
N_COMBOS=${#COMBOS[@]}
TOTAL=$((N_COMBOS * N_SHARDS))
if [ "$IDX" -ge "$TOTAL" ]; then
  echo "task $IDX >= $TOTAL (=$N_COMBOS combos x $N_SHARDS shards) — nothing to do"; exit 0
fi
COMBO_IDX=$((IDX / N_SHARDS))
SHARD_IDX=$((IDX % N_SHARDS))
read -r COHORT ARM <<< "${COMBOS[$COMBO_IDX]}"

DRYRUN=""
[ "${1:-}" = "--dry-run" ] && DRYRUN="--dry-run"

echo "=== task $IDX/$TOTAL  combo=$COMBO_IDX/$N_COMBOS  shard=$SHARD_IDX/$N_SHARDS  cohort=$COHORT  arm=$ARM  (OMP=$OMP) ==="
METHOD="$ARM" GATE_ARM="$ARM" \
  $PY evaluation/src/engine/run_baselines.py \
      --method fetal_cmr_4d --variant breath --split test --input gated \
      --sources "$COHORT" --shard "$SHARD_IDX" "$N_SHARDS" $DRYRUN
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
