#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --array=0-24%10
#SBATCH --job-name=rhythm24_seg
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/rhythm24_seg_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# EF / Dice chain (ef_dice dump -> nnU-Net 3d_fullres -> ef_dice score) on a 24-frame rhythm arm.
#
#   STAGE=gt    5 tasks  (--array=0-4): GT only, one per source cohort. Fills the per-subject
#               seg_gt_3d_fullres/ cache. RUN THIS FIRST and let it finish: five method jobs
#               started against a cold cache would each re-segment the GT and race on the cache.
#   STAGE=pred  25 tasks (--array=0-24%10): 5 methods x 5 source cohorts, reusing the GT cache.
#               NEEDS sbatch/rhythm24_metrics.sh FIRST: ef_dice segments the SCORED cine_* volumes
#               (the exact gauge/pose/PSF treatment image_metrics scored), not the raw recons, and
#               skips any subject that has none.
#
# SEG_CFG is pinned to 3d_fullres here rather than inherited from a default. A40 / 5 CPUs / 48 GB =
# the benchmark's nnU-Net config (docs/101 s2.4a). Measured 4.0 s/volume (5 folds + mirror TTA).
#
#   sbatch --array=0-4      --export=ALL,ARM=af24,STAGE=gt   sbatch/rhythm24_seg.sh
#   sbatch --array=0-24%10  --export=ALL,ARM=af24,STAGE=pred sbatch/rhythm24_seg.sh
# ============================================================================================
set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}}
cd "$REPO"
export PYTHONPATH=training:. SPLIT=${SPLIT:-test} SEG_CFG=3d_fullres
PY=/home/minsukc/micromamba/envs/svr/bin/python

ARM=${ARM:-af24}
STAGE=${STAGE:?set STAGE=gt|pred}
METHODS=(fetal_cmr_4d vggt_final518_base_ep300 vggt_final518_diff1000_ep300
         vggt_final518_nogather_ep300 vggt_final518_hw0_ep300)
# METHOD_LIST="a b c" overrides the list (e.g. the classical baselines); size --array to 5 x its length.
[ -n "${METHOD_LIST:-}" ] && read -ra METHODS <<< "$METHOD_LIST"
# The five sources a rhythm arm was built for -- NOT paths.DATASETS (it also lists miitt/ocmr, and
# the task index is (method, source): a phantom source silently drops real pairs).
SOURCES=(cmrx2023 cmrx2024 cmrx2025 acdc mnms)
NS=${#SOURCES[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
DS="${SOURCES[$((IDX % NS))]}_${ARM}"
if [ "$STAGE" = gt ]; then
  [ "$IDX" -lt "$NS" ] || { echo "task $IDX: nothing to do"; exit 0; }
  TAG="gt"; DUMP_ARGS=(--gt-only)
else
  [ "$IDX" -lt $(( ${#METHODS[@]} * NS )) ] || { echo "task $IDX: nothing to do"; exit 0; }
  TAG="${METHODS[$((IDX / NS))]}"; DUMP_ARGS=(--method "$TAG")
fi
# Git-tracked beside the cohort's image-metric summaries: metric_results/<split>/<cohort>/ef/<method>.json
# (NOT test/_ef/<method>.json, the main campaign's cross-cohort file). Pre-2026-09-23 runs wrote
# temp/rhythm24_ef/<ARM>/<method>__<cohort>.json.
OUT="$REPO/evaluation/metric_results/$SPLIT/${DS}/ef/${TAG}.json"
WORK=/tmp/rhythm24_seg_${USER}_${SLURM_JOB_ID:-local}_${IDX}
echo "=== task $IDX  stage=$STAGE  $TAG  $DS  SEG_CFG=$SEG_CFG  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
echo "out: $OUT"
if [ "${1:-}" = "--dry-run" ]; then echo "dry-run: nothing run"; exit 0; fi
[ -e "$OUT" ] && { echo "REFUSING: $OUT already exists -- move it aside first"; exit 2; }
mkdir -p "$WORK/in" "$WORK/seg" "$(dirname "$OUT")"
trap 'rm -rf "$WORK"' EXIT

t0=$(date +%s)
$PY evaluation/src/score/ef_dice.py dump "$WORK/in" "${DUMP_ARGS[@]}" --cohorts "$DS"
t1=$(date +%s); echo "[timing] dump: $((t1-t0))s  ($(ls "$WORK/in" | grep -c _0000.nii.gz) volumes to segment)"
if ls "$WORK/in"/*_0000.nii.gz >/dev/null 2>&1; then
  # Stagger `micromamba run`: tasks released together collide on ~/.cache/mamba/proc ("LockFile
  # acquisition failed") and the task dies before segmenting anything.
  sleep $(( (IDX % 10) * 15 ))
  bash evaluation/src/engine/run_seg.sh "$WORK/in" "$WORK/seg"
fi
t2=$(date +%s); echo "[timing] seg: $((t2-t1))s"
# KEEP the segmentation masks. $WORK is node-local and deleted on exit, and the score json holds
# only scalars (EDV/ESV/EF/Dice) -- so without this, the per-frame LV/MYO/RV volume curves are gone
# and any other EF rule (temporal smoothing, GT-anchored ED/ES, no largest-component filter...)
# costs a full re-segmentation. Masks are uint8 and compress to a few KB each. ef_manifest.json
# maps the sidx in each file name back to (cohort, subject). Copied BEFORE score, so a scoring
# failure does not lose them either.
KEEP="$REPO/scratch/eval/_rhythm24_segs/$ARM/${TAG}__${DS}"
if ls "$WORK/seg"/*.nii.gz >/dev/null 2>&1; then
  mkdir -p "$KEEP" && cp "$WORK/seg"/*.nii.gz "$WORK/seg"/nnunet_config "$WORK/in/ef_manifest.json" "$KEEP"/ \
    && echo "kept $(ls "$KEEP" | grep -c nii.gz) seg masks -> $KEEP"
fi
$PY evaluation/src/score/ef_dice.py score "$WORK/seg" --input "$WORK/in" --out "$OUT"
echo "[timing] score: $(( $(date +%s) - t2 ))s"
echo "DONE -> $OUT"
