#!/bin/bash
# Per-GPU queue for the interpretability campaign (docs/126). Usage: bash tools/interp/queue.sh <gpu> <shard> <nshard>
# Waits for this shard's E1 process to finish, then runs E5, E1b, E2, E3 for the same shard.
set -u
G=$1; SH=$2; NS=$3
cd "$(dirname "$0")/../.."
export PYTHONPATH=training:. CUDA_VISIBLE_DEVICES=$G
PY=/home/minsukc/micromamba/envs/svr/bin/python
O=temp/interp
while pgrep -f "e1_reference.py --out $O/e1 --shard $SH " > /dev/null; do sleep 60; done
for E in e5_patching:e5 e1b_dissociation:e1b e2_breathing:e2 e3_attention:e3 e6_breath_attention:e6; do
    S=${E%%:*}; D=${E##*:}
    $PY tools/interp/$S.py --out $O/$D --shard $SH --nshard $NS > $O/${D}_shard$SH.log 2>&1
done
