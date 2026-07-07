#!/usr/bin/env bash
# Author-parameter reconstructCardiac on a MIITT single-orientation stack, fed by our
# LV-area self-gated cardiac phases (cardphases_lvanchor_cardsync.txt, doc 35).
#
# vs run_pipeline.sh: (1) cardphase = LV-anchored (per-slice ED offset) not identity-offset
# intra-slice (doc 34/35); (2) reconstructCardiac params reverted to the AUTHORS' verbatim
# values (DEVIATIONS.md §C): resolution 1.25 mm, rec_iterations 10/last 20, robust stats ON;
# (3) tight heart mask (s01_mask_heart) not the oversized chest mask.
#
# CPU-only, memory-heavy — submit to SLURM `standard` (see sbatch/ wrapper). doc 34 §7.
# Usage: bash baselines/fetal_cmr_4d/run_selfgate_recon.sh Volunteer1
#   env: CARDPHASE_FILE (default cardphases_lvanchor_cardsync.txt), RESMM (1.25), MASK (s01_mask_heart)
set -uo pipefail

VGGT=/home/minsukc/vggt
FCMR=$VGGT/baselines/fetal_cmr_4d
RECROOT=$VGGT/scratch/fetal_cmr_4d/recon
export PATH="$FCMR/bin:$PATH"
export FCMR_BIND="$VGGT/scratch/fetal_cmr_4d"

CARDPHASE_FILE="${CARDPHASE_FILE:-cardphases_lvanchor_cardsync.txt}"
RESMM="${RESMM:-1.25}"; ITERS="${ITERS:-4}"; NUMPHASE="${NUMPHASE:-25}"
MASK="${MASK:-s01_mask_heart}"

for VOL in "$@"; do
  RD="$RECROOT/$VOL"
  CP="$RD/cardsync/$CARDPHASE_FILE"
  [ -f "$CP" ] || { echo "SKIP $VOL: no $CARDPHASE_FILE (run selfgate_lvarea_assemble.py)"; continue; }
  echo "===================== $VOL  (author params, mask=$MASK, res=${RESMM}mm) ====================="

  OUT="$RD/selfgate_cine"; rm -rf "$OUT"; mkdir -p "$OUT"
  MEANRR=$(tr -d '[:space:]' < "$RD/cardsync/mean_rrinterval.txt")   # strip trailing space (MATLAB writes '%.6f ')
  RRINTERVALS=$(cat "$RD/cardsync/rrintervals.txt")
  CARDPHASES=$(cat "$CP")
  NSLICE=$(wc -w < "$RD/cardsync/rrintervals.txt")
  NFRAME=$(wc -w < "$CP")
  THICK=$(tr -d '[:space:]' < "$RD/data/slice_thickness.txt")   # strip trailing space

  ( cd "$OUT" && mirtk reconstructCardiac cine.nii.gz 1 ../data/s01_rlt_ab.nii.gz \
      -thickness "$THICK" -mask "../mask/${MASK}.nii.gz" \
      -iterations "$ITERS" -rec_iterations 10 -rec_iterations_last 20 \
      -resolution "$RESMM" -numcardphase "$NUMPHASE" -rrinterval "$MEANRR" \
      -rrintervals "$NSLICE" $RRINTERVALS -cardphase "$NFRAME" $CARDPHASES \
      > log-main.txt 2>&1 ) \
      || { echo "FAIL recon $VOL (see $OUT/log-main.txt)"; continue; }
  echo "DONE $VOL -> $OUT/cine.nii.gz"
done
