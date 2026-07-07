#!/bin/bash
# Run NeSVoR (github.com/daviddmc/NeSVoR, Docker junshenxu/nesvor:v0.5.0) on
# NiftyMIC's already-exported per-phase stacks
# (scratch/niftymic/data/<tag>_{stack,mask}.nii.gz) -- same input, same 2
# subjects (Train_P053_t0, Val_P055_t0), no separate NeSVoR export step.
#
# STACK_SUFFIX selects clean vs respiratory-corrupted input (docs/30 sec4 step 2):
#   STACK_SUFFIX=stack       (default) -- the clean stack, IS V_gt, nothing to correct
#   STACK_SUFFIX=resp_stack  -- baselines/export_resp_stack.py's corrupted variant;
#                               mask is reused unshifted from the clean export either way.
# Writes to a suffix-specific recon dir so runs never clobber each other.
#
# Usage: STACK_SUFFIX=resp_stack bash baselines/nesvor/run_nesvor.sh <tag> [<tag> ...]
#   e.g. bash baselines/nesvor/run_nesvor.sh Train_P053_t0 Val_P055_t0
set -euo pipefail

module load singularity

STACK_SUFFIX="${STACK_SUFFIX:-stack}"

SIF_GPFS=/home/minsukc/vggt/scratch/nesvor/sif/nesvor.sif
NM_DATA=/home/minsukc/vggt/scratch/niftymic/data
# Default (clean) suffix keeps the ORIGINAL recon dir (Phase 1 results already there,
# already scored) -- only a non-default suffix gets its own recon_<suffix>/ dir.
if [ "$STACK_SUFFIX" = "stack" ]; then
    RECON_DIR=/home/minsukc/vggt/scratch/nesvor/recon
else
    RECON_DIR="/home/minsukc/vggt/scratch/nesvor/recon_${STACK_SUFFIX}"
fi

# Stage the .sif onto node-local /tmp so the container's internal file reads
# (torch/CUDA imports at exec time) don't repeatedly hit networked GPFS --
# mirrors the project's monai-cache pattern (CLAUDE.md: node-local NVMe vs
# GPFS). The durable copy stays on scratch/ so we don't re-pull 5.3GB per node.
LOCAL_SIF="/tmp/vggt-nesvor_${USER}/nesvor.sif"
mkdir -p "$(dirname "$LOCAL_SIF")"
if [ ! -f "$LOCAL_SIF" ]; then
    cp "$SIF_GPFS" "$LOCAL_SIF"
fi

if [ "$#" -eq 0 ]; then
    echo "usage: run_nesvor.sh <tag> [<tag> ...]"
    exit 1
fi

for tag in "$@"; do
    mkdir -p "$RECON_DIR/$tag"

    singularity exec --nv \
        --bind "$NM_DATA:/nm_data" \
        --bind "$RECON_DIR:/data" \
        "$LOCAL_SIF" \
        nesvor reconstruct \
        --input-stacks  "/nm_data/${tag}_${STACK_SUFFIX}.nii.gz" \
        --stack-masks   "/nm_data/${tag}_mask.nii.gz" \
        --sample-mask   "/nm_data/${tag}_mask.nii.gz" \
        --thicknesses   8.0 \
        --registration  stack \
        --output-volume "/data/${tag}/recon.nii.gz" \
        --output-model  "/data/${tag}/model.pt" \
        --device 0 \
        2>&1 | tee "$RECON_DIR/${tag}/run.log"
done
