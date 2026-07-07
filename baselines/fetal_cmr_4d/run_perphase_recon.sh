#!/usr/bin/env bash
# 3D-per-phase SVR baseline (doc 35). Each phase reconstructed INDEPENDENTLY (no temporal PSF).
# Runs phases in parallel (J at a time) across cores.
set -uo pipefail
VGGT=/home/minsukc/vggt; FCMR=$VGGT/baselines/fetal_cmr_4d
export PATH="$FCMR/bin:$PATH"; export FCMR_BIND="$VGGT/scratch/fetal_cmr_4d"
VOL="${1:-Volunteer1}"; P="${2:-25}"; K="${3:-4}"; J="${J:-8}"; RESMM="${RESMM:-1.25}"
OUTNAME="${OUTNAME:-perphase_cine}"; ROBUST="${ROBUST:-on}"    # ROBUST=off -> -no_robust_statistics
ROBFLAG=""; [ "$ROBUST" = "off" ] && ROBFLAG="-no_robust_statistics"
ITERS="${ITERS:-4}"                                              # match 4D-joint (run_selfgate_recon.sh)
RD=$VGGT/scratch/fetal_cmr_4d/recon/$VOL; OUT=$RD/$OUTNAME; mkdir -p "$OUT"
THICK=$(tr -d '[:space:]' < "$RD/data/slice_thickness.txt")     # true slice thickness from the data, not hardcoded
recon_one() {
  local p=$1; local pp=$(printf "%02d" $p)
  [ -f "$OUT/vol_p${pp}.nii.gz" ] && { echo "phase $pp cached"; return; }
  local ST="" TH=""; for k in $(seq 0 $((K-1))); do ST="$ST ../perphase_stacks/stack_p${pp}_k${k}.nii.gz"; TH="$TH $THICK"; done
  ( cd "$OUT" && OMP_NUM_THREADS=4 mirtk reconstruct vol_p${pp}.nii.gz $K $ST \
      -thickness $TH -mask ../mask/s01_mask_heart.nii.gz -resolution $RESMM -iterations $ITERS $ROBFLAG \
      > log_p${pp}.txt 2>&1 ) && echo "phase $pp done" || echo "phase $pp FAIL"
}
export -f recon_one; export OUT K RESMM ROBFLAG THICK ITERS
seq 0 $((P-1)) | xargs -P $J -I{} bash -c 'recon_one {}'
echo "PERPHASE_RECON_DONE"
