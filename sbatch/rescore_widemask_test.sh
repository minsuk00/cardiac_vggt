#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --job-name=rescore_widemask_test
#SBATCH --output=/home/minsukc/vggt/slurm_logs/rescore_widemask_test_%j.log

# ============================================================================================
# One-off: rescore the 5 final518 test-set arms with the widened heart_mask_pad ROI
# (image_metrics.py's default as of 2026-09-16). CPU-only work (image_metrics.py has no torch/
# cuda import) -- --gres=gpu:1 is here only because spgpu2 requires it to schedule; scoring runs
# on the 32 requested CPUs via xargs, not the GPU. Reuses the existing reconstructed volumes
# (run_vggt.py output on disk) -- no reconstruction, scoring only.
# ============================================================================================

set -eo pipefail
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate svr
cd /home/minsukc/vggt
export PYTHONPATH=training:.
PY=/home/minsukc/micromamba/envs/svr/bin/python
SPLIT=test

JOBLIST=$(mktemp)
for MODEL_NAME in final518_base_ep300 final518_diff1000_ep300 final518_hw2_ep300 final518_motion10_ep300 final518_motion10_hw2_ep300; do
  for S in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do
    for SUBJ_DIR in "evaluation/volumes/$S/out"/*/; do
      [ -d "$SUBJ_DIR" ] || continue
      SUBJ="$(basename "$SUBJ_DIR")"
      [ -d "$SUBJ_DIR/vggt_$MODEL_NAME" ] || continue
      [ "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["split"])' "$SUBJ_DIR/manifest.json")" = "$SPLIT" ] || continue
      echo "$S|$SUBJ|$MODEL_NAME" >> "$JOBLIST"
    done
  done
done
echo "job count: $(wc -l < "$JOBLIST")"

DONE_LOG=/home/minsukc/vggt/slurm_logs/rescore_widemask_test_done_${SLURM_JOB_ID}.log
ERR_LOG=/home/minsukc/vggt/slurm_logs/rescore_widemask_test_err_${SLURM_JOB_ID}.log

cat "$JOBLIST" | xargs -P 8 -I{} bash -c '
  IFS="|" read -r S SUBJ MODEL_NAME <<< "{}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 EVAL_DATASET="$S" '"$PY"' evaluation/src/score/image_metrics.py "$SUBJ" "vggt_$MODEL_NAME" >/dev/null 2>>'"$ERR_LOG"' \
    && echo "OK $S $SUBJ $MODEL_NAME" >> '"$DONE_LOG"' \
    || echo "FAIL $S $SUBJ $MODEL_NAME" >> '"$DONE_LOG"'
'

echo "SCORING DONE, aggregating"
for MODEL_NAME in final518_base_ep300 final518_diff1000_ep300 final518_hw2_ep300 final518_motion10_ep300 final518_motion10_hw2_ep300; do
  for S in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do
    $PY evaluation/src/score/aggregate.py "$S" "vggt_$MODEL_NAME" >> "$ERR_LOG" 2>&1
  done
done
echo ALL_DONE
grep -c FAIL "$DONE_LOG" || true
