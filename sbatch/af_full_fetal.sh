#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=8:00:00
#SBATCH --array=0-29
#SBATCH --job-name=af_full_fetal
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af_full_fetal_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# docs/110/111 arrhythmia FULL COHORT — Fetal CMR 4D on all 38 cmrx2024 test subjects, 4 rhythm
# cohorts, 3 gate arms. 10 (cohort, arm) combos x 3 shards each = 30 array tasks, using
# run_baselines.py's own --shard I N mechanism (the same one the production campaign uses,
# eval_baseline_fetal4d_v2.sh) for balanced subject-level sharding, not per-pair concurrency
# like the pilot's af_pilot_fetal.sh.
#
# fetal_cmr_4d_oracle_balanced is built ONLY for hrv/af, not regular_frozen/regular:
# rank_quantize on an already-integer, already-uniformly-spaced rhythm is the identity
# permutation up to floating point noise (VERIFIED: max|theta_raw - theta_balanced| = 5.3e-15
# on both regular cohorts), so building it there would reconstruct byte-for-byte the same
# thing as fetal_cmr_4d_oracle for zero additional information -- 76 of 456 reconstructions
# saved for free. 10 combos x 38 subjects = 380 reconstructions total.
#
# ARM PAIRING IS LOAD-BEARING. run_baselines.py refuses a GATE_ARM/METHOD mismatch (and, as of
# docs/111 s7 F1, ALSO refuses METHOD set to a variant arm with GATE_ARM left unset -- the
# silent-mispairing bug the fix closed). `--method fetal_cmr_4d` stays fixed (it selects the
# shell + the gate precondition check); METHOD/GATE_ARM select which gate is consumed and where
# the recon lands.
#
# PREREQUISITES (the driver refuses subjects that lack them):
#   tools/build_af_bundle.py --subjects <all 38, comma-sep>       -> scratch/eval/cmrx2024_<arm>/out/<subject>/
#   fetal4d_gate.py dump / seg / assemble --work-tag afsim_full             -> <subject>/fetal_cmr_4d/gate/
#   fetal4d_gate.py assemble --oracle --work-tag afsim_full                -> <subject>/fetal_cmr_4d_oracle/gate/
#   fetal4d_gate.py assemble --oracle --balanced --work-tag afsim_full     -> .../fetal_cmr_4d_oracle_balanced/gate/
#     (balanced gate needed only for --sources cmrx2024_hrv cmrx2024_af)
#
# The driver skips already-stamped subjects, so a requeue only does the remainder.
# NOT self-submitting — submit with `sbatch sbatch/af_full_fetal.sh`.
# Dry-run the work list first: bash sbatch/af_full_fetal.sh --dry-run (runs task 0 locally,
# prints the plan, no sbatch).
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export OMP=${OMP:-${SLURM_CPUS_PER_TASK:-4}} ROBUST=${ROBUST:-1} RES=${RES:-1.4}

# (cohort, arm) work items. Order: self-gated + oracle on every cohort, oracle_balanced only
# where it can differ from oracle (hrv, af).
COMBOS=(
  "cmrx2024_regular_frozen fetal_cmr_4d"
  "cmrx2024_regular_frozen fetal_cmr_4d_oracle"
  "cmrx2024_regular fetal_cmr_4d"
  "cmrx2024_regular fetal_cmr_4d_oracle"
  "cmrx2024_hrv fetal_cmr_4d"
  "cmrx2024_hrv fetal_cmr_4d_oracle"
  "cmrx2024_hrv fetal_cmr_4d_oracle_balanced"
  "cmrx2024_af fetal_cmr_4d"
  "cmrx2024_af fetal_cmr_4d_oracle"
  "cmrx2024_af fetal_cmr_4d_oracle_balanced"
)
N_SHARDS=${N_SHARDS:-3}   # 10 combos x 3 shards = 30 array tasks
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
