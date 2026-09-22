#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --job-name=rhythm24_cinevol_score
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/rhythm24_cinevol_score_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# Score the CiNeVol arm on a 24-frame rhythm arm with THE SAME standing code every other method is
# scored by -- nothing CiNeVol-specific:
#   1. evaluation/src/score/run.py        image metrics (PSF + 6-DOF NCC registration, PSNR/SSIM/NCC)
#                                         -> <subject>/cinevol/{metrics.json,cine_breath.nii.gz}
#                                         -> evaluation/metric_results/test/<cohort>/cinevol.json
#   2. ef_dice dump -> run_seg.sh (nnU-Net 3d_fullres) -> ef_dice score     EF / EDV / ESV / Dice / HD95
#                                         -> temp/rhythm24_ef/<ARM>/cinevol__<cohort>.json
# This is rhythm24_metrics.sh + rhythm24_seg.sh STAGE=pred for ONE method, one task per source cohort,
# written as its own file so those two (their hardcoded METHODS lists) stay untouched.
#
# NEVER OVERWRITES: refuses if either output json exists. The shared GT seg cache must be warm
# (180/180 verified 2026-09-21) so this only READS it -- the task aborts if any subject is uncached.
# EXTRA: the per-frame nnU-Net segmentations (pred + GT copies) are KEPT under
# scratch/cinevol/segs/<ARM>/cinevol__<cohort>/ -- ef_dice keeps only ED/ES summaries, and the
# all-frame metrics of docs/117 (volume-curve error, 24-frame Dice) need the full set.
#
#   sbatch --export=ALL,ARM=af24 sbatch/rhythm24_cinevol_score.sh          dry-run: bash ... --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}}
[ -d "$REPO/evaluation/src/score" ] || REPO=/home/minsukc/vggt-afsim
cd "$REPO" || exit 1
export PYTHONPATH=training:. SPLIT=test SEG_CFG=3d_fullres
PY=/home/minsukc/micromamba/envs/svr/bin/python
ARM=${ARM:-af24}
METHOD=cinevol
SOURCES=(cmrx2023 cmrx2024 cmrx2025 acdc mnms)
IDX=${SLURM_ARRAY_TASK_ID:-0}
[ "$IDX" -lt ${#SOURCES[@]} ] || { echo "task $IDX: nothing to do"; exit 0; }
DS="${SOURCES[$IDX]}_${ARM}"
AGG="evaluation/metric_results/test/${DS}/${METHOD}.json"
EF="temp/rhythm24_ef/${ARM}/${METHOD}__${DS}.json"
SEGKEEP="scratch/cinevol/segs/${ARM}/${METHOD}__${DS}"
WORK=/tmp/rhythm24_cinevol_score_${USER}_${SLURM_JOB_ID:-local}_${IDX}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=== task $IDX  $METHOD  $DS  gpu=$GPU ==="
echo "out: $AGG | $EF | $SEGKEEP"
if [ "${1:-}" = "--dry-run" ]; then echo "dry-run: nothing run"; exit 0; fi
[ -n "$GPU" ] || { echo "no GPU: pose_psf would fall back to CPU silently, refusing"; exit 4; }
for f in "$AGG" "$EF" "$SEGKEEP"; do [ -e "$f" ] && { echo "REFUSING: $f already exists -- move it aside first"; exit 2; }; done

$PY evaluation/src/score/run.py --method "$METHOD" --datasets "$DS" --split test || { echo "image metrics failed"; exit 1; }

mkdir -p "$WORK/in" "$WORK/seg" "$(dirname "$EF")" "$(dirname "$SEGKEEP")"
trap 'rm -rf "$WORK"' EXIT
$PY evaluation/src/score/ef_dice.py dump "$WORK/in" --method "$METHOD" --cohorts "$DS" || exit 1
$PY - "$WORK/in/ef_manifest.json" <<'EOF' || exit 3
import json, sys
m = json.load(open(sys.argv[1]))["subjects"]; rows = m if isinstance(m, list) else list(m.values())
bad = [r for r in rows if not r.get("gt_cached")]
print(f"dumped {len(rows)} subjects; GT cache reused for {len(rows) - len(bad)}")
sys.exit(3 if bad else 0)          # an uncached GT would make score() WRITE the shared cache -> abort
EOF
sleep $(( IDX * 15 ))                                   # stagger `micromamba run` (proc-lock collisions)
bash evaluation/src/engine/run_seg.sh "$WORK/in" "$WORK/seg" || { echo "segmentation failed"; exit 1; }
$PY evaluation/src/score/ef_dice.py score "$WORK/seg" --input "$WORK/in" --out "$EF" || exit 1
mkdir "$SEGKEEP" && cp "$WORK"/seg/*.nii.gz "$WORK/in/ef_manifest.json" "$SEGKEEP"/ && echo "kept segs -> $SEGKEEP"
echo "=== task $IDX done ==="
