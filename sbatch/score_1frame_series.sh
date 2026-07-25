#!/bin/bash
#SBATCH --job-name=score_1frame
#SBATCH --account=jjparkcv0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/score_1frame_%j.log
#
# CPU scoring sweep for the one-frame ablation: assemble_and_gif.py (SKIP_GIF=1 -> metrics.json
# only) over every (subject, method) pair, then aggregate.py per (dataset, method).
#
# Why sbatch and not inline: the interactive GPU node has 4 cores and is busy running run_vggt.py.
# Scoring is pure CPU and embarrassingly parallel across subjects, so it goes to `standard` with
# 32 cores and overlaps the GPU passes.
#
# Usage: sbatch --account=jjparkcv0 sbatch/score_1frame_series.sh [method ...]
set -uo pipefail
VGGT=/home/minsukc/vggt
cd "$VGGT" || exit 1

# Call the env's python DIRECTLY, never `micromamba run`, under xargs -P. Parallel `micromamba run`
# contends on the mamba cache lock and SILENTLY DROPS units ("Could not set lock (No locks
# available)") -- it dropped a subject from this very sweep, and dropped 16 units in docs/39.
PY="/home/minsukc/micromamba/envs/svr/bin/python"

METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  METHODS=(vggt_20260715_1f_gather05 vggt_20260715_1f_no_gather vggt_20260715_1f_aug_moderate \
           vggt_20260715_1f_dino_ft vggt_20260715_1f_contz vggt_20260715_1f_lowdiff100)
fi

score() {  # score <dataset> <method>
  local ds=$1 meth=$2
  local subs
  subs=$(ls -d "$VGGT/scratch/eval/$ds/out"/*/"$meth" 2>/dev/null | awk -F/ '{print $(NF-1)}')
  [ -z "$subs" ] && { echo "  ($ds/$meth: no recons yet, skip)"; return; }
  local n; n=$(echo "$subs" | wc -l)
  echo "### $ds / $meth  ($n subjects)"
  local extra=""
  [ "$ds" != "cmrxrecon" ] && extra="FOV_MASK=mask_fov.nii.gz"   # miitt/ocmr/acdc score over placed FOV
  echo "$subs" | xargs -P 16 -I{} env SKIP_GIF=1 EVAL_DATASET="$ds" $extra \
    "$PY" "$VGGT/scratch/eval/engine/assemble_and_gif.py" {} "$meth" > /dev/null 2>&1
  # Verify nothing was silently dropped before aggregating.
  local got; got=$(ls "$VGGT/scratch/eval/$ds/out"/*/"$meth"/metrics.json 2>/dev/null | wc -l)
  [ "$got" -ne "$n" ] && echo "  !! WARNING $ds/$meth: scored $got of $n subjects -- re-running the misses"
  if [ "$got" -ne "$n" ]; then
    for s in $subs; do
      [ -f "$VGGT/scratch/eval/$ds/out/$s/$meth/metrics.json" ] || \
        env SKIP_GIF=1 EVAL_DATASET="$ds" $extra "$PY" \
          "$VGGT/scratch/eval/engine/assemble_and_gif.py" "$s" "$meth" > /dev/null 2>&1
    done
  fi
  "$PY" "$VGGT/scratch/eval/engine/aggregate.py" "$ds" "$meth" 2>&1 | tail -3
}

for meth in "${METHODS[@]}"; do
  for ds in cmrxrecon miitt ocmr acdc; do
    score "$ds" "$meth"
  done
done
echo "SCORE_DONE $(date +%H:%M:%S)"
