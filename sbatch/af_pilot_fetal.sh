#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=8:00:00
#SBATCH --array=0-7%7
#SBATCH --job-name=af_pilot_fetal
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af_pilot_fetal_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# docs/110 arrhythmia PILOT — Fetal CMR 4D on the four rhythm cohorts, two gating arms.
#
#   4 cohorts (cmrx2024_{regular_frozen,regular,hrv,af})
#     x 2 arms  (fetal_cmr_4d = self-gated, fetal_cmr_4d_oracle = true per-frame theta)
#     = 8 array tasks, each running all 4 pilot subjects sequentially.
#
# ARM PAIRING IS LOAD-BEARING. run_baselines.py refuses a GATE_ARM/METHOD mismatch, because
# mis-pairing them is otherwise silent: you would get an arm named _oracle holding self-gated
# timing, or an oracle recon written into (or skipped as already-stamped under) the plain arm
# dir. `--method fetal_cmr_4d` stays fixed (it selects the shell + the gate precondition check);
# METHOD/GATE_ARM select which gate is consumed and where the recon lands.
#
# CPU sizing: the engine scales poorly past ~4 cores (docs/105), so 4 cpus x 7 concurrent tasks
# = 28 CPUs, inside the 30-CPU budget for this campaign. `%7` caps concurrency, not the array.
#
# PREREQUISITES (the driver refuses subjects that lack them):
#   tools/build_af_bundle.py                      -> scratch/eval/cmrx2024_<arm>/out/<subject>/
#   fetal4d_gate.py dump / seg / assemble --work-tag afsim   -> <subject>/fetal_cmr_4d/gate/
#   fetal4d_gate.py assemble --oracle --work-tag afsim       -> <subject>/fetal_cmr_4d_oracle/gate/
#
# The driver skips already-stamped subjects, so a requeue only does the remainder.
# NOT self-submitting — submit with `sbatch sbatch/af_pilot_fetal.sh`.
# Restrict with COHORTS=... ARMS=... SUBJECTS=... ; widen the pilot with SUBJECTS="..." for the
# full-cohort run (drop --subjects entirely by setting SUBJECTS="ALL").
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export OMP=${OMP:-${SLURM_CPUS_PER_TASK:-4}} ROBUST=${ROBUST:-1} RES=${RES:-1.4}

COHORTS=${COHORTS:-"cmrx2024_regular_frozen cmrx2024_regular cmrx2024_hrv cmrx2024_af"}
ARMS=${ARMS:-"fetal_cmr_4d fetal_cmr_4d_oracle"}
SUBJECTS=${SUBJECTS:-"CMRx24_Test_P016 CMRx24_Test_P004 CMRx24_Test_P017 CMRx24_Train_P046"}

# Flatten (cohort, arm) into the array index. Order is cohort-major so a partial array still
# covers both arms of the cohorts it reached — the oracle gap is the quantity of interest, and
# it is only readable when both arms of the SAME cohort exist.
PAIRS=()
for c in $COHORTS; do for a in $ARMS; do PAIRS+=("$c|$a"); done; done
N=${#PAIRS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
[ "$IDX" -lt "$N" ] || { echo "task $IDX >= $N pairs — nothing to do"; exit 0; }
COHORT=${PAIRS[$IDX]%%|*}
ARM=${PAIRS[$IDX]##*|}

SUBJ_ARGS=()
[ "$SUBJECTS" = "ALL" ] || SUBJ_ARGS=(--subjects $SUBJECTS)

echo "=== task $IDX/$N  cohort=$COHORT  arm=$ARM  (OMP=$OMP ROBUST=$ROBUST RES=$RES) ==="
echo "    subjects: ${SUBJECTS}"
METHOD="$ARM" GATE_ARM="$ARM" \
  $PY evaluation/src/engine/run_baselines.py \
      --method fetal_cmr_4d --variant breath --split test --input gated \
      --sources "$COHORT" "${SUBJ_ARGS[@]}"
rc=$?
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
