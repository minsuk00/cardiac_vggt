#!/usr/bin/env bash
# Step 2/3 of the EF/Dice chain — nnU-Net v1 Task114 (M&Ms) segmentation ACROSS THE ENV BOUNDARY.
#
# ef_dice.py runs in `svr`, but the segmenter is nnU-Net v1 in the SEPARATE `nnunet` env
# (RESULTS_FOLDER etc. are set by tools/nnunet_mnms_eval/env.sh). This wrapper is git-tracked so the
# middle step is reproducible instead of living only in a driver sbatch. Like run_svrtk3d.sh /
# run_nesvor.sh it crosses envs, so it is a shell wrapper, not an engine python step.
#
# Full chain (all git-tracked):
#   python evaluation/src/analysis/ef_dice.py dump  <IN> --method <m> --cohorts ...
#   bash   evaluation/src/engine/run_seg.sh          <IN> <SEG>
#   python evaluation/src/analysis/ef_dice.py score <SEG> --input <IN> --out <ef.json>
#   python evaluation/src/analysis/ef_dice.py plot  <ef.json> --out <ef.png>
#
# Usage: bash evaluation/src/engine/run_seg.sh <input_dir> <seg_dir>
set -euo pipefail
VGGT=/home/minsukc/vggt
IN=${1:?input_dir with ef_dice.py dump output (*_0000.nii.gz)}
SEG=${2:?seg_dir (nnU-Net output)}
ENV_SH=$VGGT/tools/nnunet_mnms_eval/env.sh
mkdir -p "$SEG"
shopt -s nullglob   # empty globs expand to nothing (not a literal), so the count is honest under set -e

_in=("$IN"/*_0000.nii.gz)
[ "${#_in[@]}" -gt 0 ] || { echo "run_seg: no *_0000.nii.gz in $IN — run ef_dice.py dump first" >&2; exit 1; }
echo "[run_seg] Task114 2d nnUNetTrainerV2_MMS   in=$IN (${#_in[@]} vols)  out=$SEG"

micromamba run -n nnunet bash -c "source '$ENV_SH' && \
  nnUNet_predict -i '$IN' -o '$SEG' -t 114 -m 2d -tr nnUNetTrainerV2_MMS"

_seg=("$SEG"/*.nii.gz)
echo "[run_seg] segmented ${#_seg[@]} volumes -> $SEG"
