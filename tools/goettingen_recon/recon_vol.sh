#!/bin/bash
# Classical RT-NLINV reconstruction of one Göttingen radial volume (all SAX slices),
# matching mrirecon/nlinv-net's 10_reco_rt.sh (RT-NLINV + ROVir coil compression, NO neural net).
#
# Usage: recon_vol.sh <vol_basename_without_ext> <out_dir> [slice]
#   <vol_basename>: e.g. /scratch/.../goettingen_radial/vol0001_vis1  (no .cfl/.hdr)
#   <out_dir>:      output dir; produces img_rt{,_fil,_fil_nlmean}.{cfl,hdr}
#   [slice]:        optional single slice index; omit → all slices
set -eu

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/bart_env.sh"
NLINV_SCRIPTS=/home/minsukc/vggt/scratch/nlinv-net/01_scripts

VOL="$1"; OUT="$2"; SLICE="${3:-}"
mkdir -p "$OUT"

# skip the ~33min CPU nlmeans cosmetic denoise; img_rt_fil (median-filtered RT-NLINV) is the deliverable
export SKIP_NLMEANS=1

SLICE_ARG=""
[ -n "$SLICE" ] && SLICE_ARG="-s$SLICE"

# -R raw-input mode, -S13 = 13 spokes/frame binning, -V ROVir coil compression (matches example 67)
"$NLINV_SCRIPTS/10_reco_rt.sh" -R $SLICE_ARG -S13 -V "$VOL" "$OUT/img_rt"
