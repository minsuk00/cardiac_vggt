#!/usr/bin/env bash
# Run NiftyMIC's classical SVR+SR reconstruction on the exported per-phase stacks.
# Usage: bash baselines/niftymic/run_niftymic.sh <subject>_<t> [more subject_t ...]
#   e.g. bash baselines/niftymic/run_niftymic.sh CMRx24_Train_P053_t0 CMRx24_Val_P055_t0
set -euo pipefail

SIF=/home/minsukc/vggt/scratch/niftymic/sif/niftymic.sif
DATA_DIR=/home/minsukc/vggt/scratch/niftymic/data
RECON_DIR=/home/minsukc/vggt/scratch/niftymic/recon
BIND=/home/minsukc/vggt/scratch/niftymic

mkdir -p "$RECON_DIR"

for tag in "$@"; do
  stack="$DATA_DIR/${tag}_stack.nii.gz"
  mask="$DATA_DIR/${tag}_mask.nii.gz"
  out_dir="$RECON_DIR/$tag"
  mkdir -p "$out_dir"

  if [[ ! -f "$stack" ]]; then
    echo "SKIP $tag: $stack not found (run export_stack.py first)"
    continue
  fi

  echo "=== $tag ==="
  singularity exec --bind "$BIND" "$SIF" niftymic_reconstruct_volume \
    --filenames "$stack" \
    --filenames-masks "$mask" \
    --output "$out_dir/recon.nii.gz" \
    2>&1 | tee "$out_dir/run.log"
done
