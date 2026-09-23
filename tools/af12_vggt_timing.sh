#!/bin/bash
# Timed VGGT final518_diff1000 recon on the five af12 cohorts (docs/120).
#
#   GPUS=0,1,2 NAME=final518_diff1000_ep300_3gpu bash tools/af12_vggt_timing.sh   # 3-GPU phase sharding
#   GPUS=0     NAME=final518_diff1000_ep300_1gpu bash tools/af12_vggt_timing.sh   # same-machine 1-GPU reference
#
# Same run_vggt.py args as sbatch/rhythm24_vggt.sh (split test, arms breath, input scatter).
# Safety: refuses if vggt_$NAME exists in any af12 cohort (never overwrites), refuses unless every
# visible GPU is idle and the CPU load is low, snapshots every existing af12 file before/after and
# diffs them. Monitors GPUs + CPU load every 2 s into $LOG/monitor.csv (contamination audit).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=training:.
GPUS=${GPUS:-0,1,2}
NAME=${NAME:?set NAME (the arm name, e.g. final518_diff1000_ep300_3gpu)}
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
CKPT=${CKPT:-scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt}
MAX_LOAD=${MAX_LOAD:-4}
SOURCES=(cmrx2023 cmrx2024 cmrx2025 acdc mnms)
LOG=${LOG:-temp/af12_timing_${NAME}}; mkdir -p "$LOG"
# Only the GPUs we use are visible: run_vggt then never opens a context on any other GPU.
export CUDA_VISIBLE_DEVICES=$GPUS
LOCAL_GPUS=$(seq -s, 0 $(( $(tr ',' '\n' <<< "$GPUS" | wc -l) - 1 )))

for S in "${SOURCES[@]}"; do
    if compgen -G "evaluation/volumes/${S}_af12/out/*/vggt_${NAME}" > /dev/null; then
        echo "REFUSING: vggt_${NAME} already exists in ${S}_af12 (pick another NAME)"; exit 1; fi
done
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING: a process is already on a GPU"; nvidia-smi; exit 1; fi
LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
    echo "REFUSING: 1-min load average $LOAD > $MAX_LOAD (another job is using the CPU)"; exit 1; fi

snap() { find evaluation/volumes/*_af12/out -type f -printf '%p %s %T@\n' | grep -v "/vggt_${NAME}/" | sort; }
snap > "$LOG/before.txt"
( while true; do
    ts=$(date +%s.%N); la=$(awk '{print $1}' /proc/loadavg)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader,nounits | sed "s/^/$ts gpu,/;s/$/,load=$la/"
    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader | sed "s/^/$ts app,/"
    sleep 2
  done ) > "$LOG/monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
echo "host $(hostname)  gpus $GPUS  run pid $$  start $(date -Is)" | tee "$LOG/run.txt"
nvidia-smi --query-gpu=index,name --format=csv,noheader >> "$LOG/run.txt"

rc=0
for S in "${SOURCES[@]}"; do
    echo "=== ${S}_af12  $(date -Is)" | tee -a "$LOG/run.txt"
    $PY evaluation/src/engine/run_vggt.py --dataset "${S}_af12" --ckpt "$CKPT" --model-name "$NAME" \
        --split test --arms breath --input scatter --gpus "$LOCAL_GPUS" > "$LOG/${S}_af12.log" 2>&1 || rc=$?
done
kill $MON
snap > "$LOG/after.txt"
if diff -q "$LOG/before.txt" "$LOG/after.txt" > /dev/null; then echo "existing files untouched: OK"
else echo "!! existing af12 files CHANGED — see diff $LOG/before.txt $LOG/after.txt"; rc=1; fi
# other processes that appeared on the GPUs during the run (should be empty)
awk -F', ' '/ app,/{print $2}' "$LOG/monitor.csv" | sort -u > "$LOG/gpu_pids.txt"
echo "PIDs seen on GPUs during the run: $(tr '\n' ' ' < "$LOG/gpu_pids.txt")  (only ours => clean)"
echo "end $(date -Is) rc=$rc" | tee -a "$LOG/run.txt"
exit $rc
