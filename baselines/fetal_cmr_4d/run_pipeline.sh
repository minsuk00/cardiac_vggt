#!/usr/bin/env bash
# fetal_cmr_4d magnitude 4D recon on a MIITT (single-orientation) recon dir.
#
# WHY THIS IS NOT THE FULL 5-STAGE PIPELINE:
# The paper's staged flow (stack-stack reg -> static MC -> slice_cine -> interslice
# cardiac sync -> 4D recon) is built for MULTI-orientation acquisitions. On our
# single short-axis stack, three of those stages are inapplicable by construction:
#   - stack-stack registration : needs >=2 stacks (we have one) -> degenerate/segfault
#   - slice_cine + INTERSLICE sync : align cardiac phase across slices using their
#     spatial OVERLAP, which only exists between DIFFERENT orientations. Parallel
#     SAX slices don't overlap -> no basis for interslice sync (same root cause as
#     the through-plane blur).
# What remains, and is faithful for single-orientation, is:
#   1. intra-slice self-gating (per-slice heart rate + cardiac phase)   [MATLAB]
#   2. motion-corrected 4D reconstruction                               [SVRTK]
# The motion correction is reconstructCardiac's own register<->reconstruct
# iterations (the same rigid slice/frame->volume registration the staged static-MC
# does, integrated). Interslice offset is identity (intra-slice phases used directly).
#
# Usage: bash baselines/fetal_cmr_4d/run_pipeline.sh Volunteer1 [Volunteer2 ...]
#   env: ITERS (default 4), RESMM (default 1.5), NUMPHASE (default 25)
set -uo pipefail

VGGT=/home/minsukc/vggt
FCMR=$VGGT/baselines/fetal_cmr_4d
REPO=$VGGT/scratch/fetal_cmr_4d/repo
RECROOT=$VGGT/scratch/fetal_cmr_4d/recon
MYDIR=$FCMR/matlab
ITERS="${ITERS:-4}"; RESMM="${RESMM:-1.5}"; NUMPHASE="${NUMPHASE:-25}"

export PATH="$FCMR/bin:$PATH"
export FCMR_BIND="$VGGT/scratch/fetal_cmr_4d"
module load matlab/R2024b 2>/dev/null || true

for VOL in "$@"; do
  RD="$RECROOT/$VOL"
  if [ ! -f "$RD/data/s01_rlt_ab.nii.gz" ]; then
    echo "SKIP $VOL: no exported input (run export_miitt.py first)"; continue
  fi
  echo "===================== $VOL ====================="

  echo "[1/2] intra-slice self-gating (MATLAB)"
  ( cd "$MYDIR" && matlab -nodisplay -batch "miitt_gating('$RD','$REPO','$MYDIR')" ) \
      || { echo "FAIL gating $VOL"; continue; }

  echo "[2/2] motion-corrected 4D reconstruction (SVRTK, ITERS=$ITERS RES=${RESMM}mm)"
  OUT="$RD/faithful_cine"; rm -rf "$OUT"; mkdir -p "$OUT"
  MEANRR=$(cat "$RD/cardsync/mean_rrinterval.txt")
  RRINTERVALS=$(cat "$RD/cardsync/rrintervals.txt")
  CARDPHASES=$(cat "$RD/cardsync/cardphases_intraslice_cardsync.txt")
  NSLICE=$(wc -w < "$RD/cardsync/rrintervals.txt")
  NFRAME=$(wc -w < "$RD/cardsync/cardphases_intraslice_cardsync.txt")
  THICK=$(cat "$RD/data/slice_thickness.txt")
  ( cd "$OUT" && mirtk reconstructCardiac cine.nii.gz 1 ../data/s01_rlt_ab.nii.gz \
      -thickness "$THICK" -mask ../mask/mask_chest.nii.gz \
      -iterations "$ITERS" -rec_iterations 7 -rec_iterations_last 12 \
      -resolution "$RESMM" -numcardphase "$NUMPHASE" -rrinterval "$MEANRR" \
      -rrintervals "$NSLICE" $RRINTERVALS -cardphase "$NFRAME" $CARDPHASES \
      -no_robust_statistics > log-main.txt 2>&1 ) \
      || { echo "FAIL recon $VOL"; continue; }

  echo "DONE $VOL -> $OUT/cine.nii.gz"
done
