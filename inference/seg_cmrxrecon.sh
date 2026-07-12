#!/usr/bin/env bash
# Segment the per-phase pred/GT volumes dumped by inference/run_cmrxrecon.py --dump-volumes
# with the M&Ms nnU-Net (Task114, 2d) in the isolated `nnunet` env (docs/15), then the
# EF/Dice analysis is done by inference/seg_metrics_cmrxrecon.py.
#
# 2d + nnUNetTrainerV2_MMS matches the method used for the docs/17 + docs/24 EF analyses, so
# pred-EF vs GT-EF stays method-matched (same segmenter on both). `svr` is never activated here.
#
# Usage:
#   bash inference/seg_cmrxrecon.sh <dump_dir> <seg_out_dir>
set -euo pipefail
IN="${1:?usage: seg_cmrxrecon.sh <dump_dir> <seg_out_dir>}"
OUT="${2:?usage: seg_cmrxrecon.sh <dump_dir> <seg_out_dir>}"
ENV_SH="$(cd "$(dirname "$0")/.." && pwd)/tools/nnunet_mnms_eval/env.sh"
mkdir -p "$OUT"
micromamba run -n nnunet bash -c "source '$ENV_SH' && \
  nnUNet_predict -i '$IN' -o '$OUT' -t 114 -m 2d -tr nnUNetTrainerV2_MMS"
echo "segmentations -> $OUT"
