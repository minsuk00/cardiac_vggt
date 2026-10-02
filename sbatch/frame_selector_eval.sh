#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --job-name=frame_selector_eval
#SBATCH --output=slurm_logs/frame_selector_eval_%A_%a.out

# ============================================================================================
# Frame-selector eval on the rhythm arms (docs/130): diff1000 with companions picked by a
# per-slice selection table (run_vggt.py --input sel), then image metrics, then nnU-Net EF/Dice.
# One task per (arm, source): 2 arms x 5 sources = 10 tasks.
#
#   SEL=<name> -> table dir scratch/framesel/sel_tables/<name> (frame_selector.predict / .oracle)
#   mkdir -p slurm_logs && sbatch --export=ALL,SEL=oracle sbatch/frame_selector_eval.sh   # repo root
# ⚠️ From inside an interactive GPU allocation, clear the SLURM env first (`unset ${!SLURM_@}`).
# ============================================================================================
set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
cd "$REPO" || exit 1
export SPLIT=test SEG_CFG=3d_fullres
PY=${PY:-${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/svr/bin/python}
CKPT=scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt

SEL=${SEL:-oracle}
NAME=final518_diff1000_ep300_sel_${SEL}
METHOD=vggt_${NAME}
ARMS=(af12 hrv12)
SOURCES=(cmrx2023 cmrx2024 cmrx2025 acdc mnms)
IDX=${SLURM_ARRAY_TASK_ID:-0}
DS="${SOURCES[$((IDX % 5))]}_${ARMS[$((IDX / 5))]}"
echo "=== task $IDX  $DS  $METHOD  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

$PY evaluation/src/engine/run_vggt.py --dataset "$DS" --ckpt "$CKPT" --model-name "$NAME" \
    --split test --arms breath --input sel --sel-dir "scratch/framesel/sel_tables/$SEL" || exit 1
$PY evaluation/src/score/run.py --method "$METHOD" --datasets "$DS" --split test || exit 1

OUT="$REPO/evaluation/metric_results/test/${DS}/ef/${METHOD}.json"
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists"; exit 2; }
WORK=/tmp/frame_selector_seg_${USER}_${SLURM_JOB_ID:-local}_${IDX}
mkdir -p "$WORK/in" "$WORK/seg" "$(dirname "$OUT")"
trap 'rm -rf "$WORK"' EXIT
$PY evaluation/src/score/ef_dice.py dump "$WORK/in" --method "$METHOD" --cohorts "$DS" || exit 1
sleep $(( IDX * 15 ))
bash evaluation/src/engine/run_seg.sh "$WORK/in" "$WORK/seg" || exit 1
KEEP="$REPO/scratch/eval/_framesel_segs/${METHOD}__${DS}"
mkdir -p "$KEEP" && cp "$WORK/seg"/*.nii.gz "$WORK/seg"/nnunet_config "$WORK/in/ef_manifest.json" "$KEEP"/ || exit 1
$PY evaluation/src/score/ef_dice.py score "$WORK/seg" --input "$WORK/in" --out "$OUT" || exit 1
echo "DONE -> $OUT"
