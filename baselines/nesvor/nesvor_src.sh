#!/usr/bin/env bash
# Run NeSVoR from the LOCAL SOURCE CLONE (baselines/nesvor/NeSVoR, checked out at v0.5.0)
# instead of the source frozen inside the container image.
#
# Why this exists: the pinned image `junshenxu/nesvor:v0.5.0` installs NeSVoR as an EDITABLE
# install (nesvor.egg-link -> /usr/local/NeSVoR), so bind-mounting our clone over that path
# makes `import nesvor` resolve to the clone. Everything else -- python 3.10, torch 1.13.1+cu117,
# tiny-cuda-nn, the two prebuilt CUDA extensions -- still comes from the container. No new conda
# env, no rebuild, and the numerics stay identical to what we already published.
#
# VERIFIED (2026-08-27):
#   * the container's /usr/local/NeSVoR source is byte-identical to upstream tag v0.5.0
#     (diff -rq, only extra file = the Dockerfile-generated install.sh)
#   * with the bind, `nesvor.__file__` points at the clone; edits to the clone take effect
#     (marker-print test: present with the bind, absent without)
#   * end-to-end `nesvor reconstruct` runs on a real eval bundle (CMRx24_Test_P012, breath t00)
#
# Usage (drop-in for `singularity exec ... nesvor`):
#   bash baselines/nesvor/nesvor_src.sh [extra singularity binds...] -- reconstruct --input-stacks ...
# or source it and use the NESVOR_SRC_BIND array in your own singularity command.
#
# Example:
#   bash baselines/nesvor/nesvor_src.sh --bind "$SD:/data" --bind "$OUT:/out" -- \
#     reconstruct --input-stacks /data/breath/stack_t00.nii.gz \
#                 --stack-masks /data/mask_heart.nii.gz --sample-mask /data/mask_heart.nii.gz \
#                 --thicknesses 8 --registration none --output-resolution 1.4 \
#                 --output-volume /out/vol_t00.nii.gz --device 0
set -euo pipefail

VGGT=${VGGT:-/home/minsukc/vggt}
CLONE=${NESVOR_CLONE:-$VGGT/baselines/nesvor/NeSVoR}
SIF_GPFS=${NESVOR_SIF:-$VGGT/scratch/nesvor/sif/nesvor.sif}

module load singularity 2>/dev/null || true

[ -f "$CLONE/nesvor/__init__.py" ] || { echo "FATAL: no clone at $CLONE (see baselines/nesvor/readme.md §7)"; exit 1; }
# The two CUDA extensions + egg-info are NOT in the git tree -- they are copied out of the image
# (see scratch/nesvor/blobs_v050). Without them `import nesvor` fails at the C-extension import.
for f in "$CLONE"/nesvor/slice_acq_cuda.*.so "$CLONE"/nesvor/transform_convert_cuda.*.so; do
    [ -f "$f" ] || { echo "FATAL: missing vendored CUDA ext in $CLONE/nesvor (re-copy from scratch/nesvor/blobs_v050)"; exit 1; }
done

# Same node-local .sif staging as evaluation/src/engine/run_nesvor.sh (GPFS is slow for the
# container's internal small reads); flock'd + size-checked so concurrent jobs can't poison it.
LOCAL_SIF="/tmp/vggt-nesvor_${USER}/nesvor.sif"
mkdir -p "$(dirname "$LOCAL_SIF")"
exec 9>"${LOCAL_SIF}.lock"; flock 9
_want=$(stat -c %s "$SIF_GPFS")
if [ ! -f "$LOCAL_SIF" ] || [ "$(stat -c %s "$LOCAL_SIF" 2>/dev/null || echo -1)" != "$_want" ]; then
    _tmp="${LOCAL_SIF}.tmp.$$"
    cp "$SIF_GPFS" "$_tmp" && mv -f "$_tmp" "$LOCAL_SIF" || { rm -f "$_tmp"; echo "FATAL: sif staging failed"; exit 1; }
fi
exec 9>&-

# Split argv at the `--` separator: before it = extra singularity binds, after it = nesvor args.
BINDS=()
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do BINDS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift

exec singularity exec --nv \
    --bind "$CLONE:/usr/local/NeSVoR" \
    "${BINDS[@]}" \
    "$LOCAL_SIF" nesvor "$@"
