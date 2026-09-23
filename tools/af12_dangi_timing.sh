#!/bin/bash
# Timed Dangi Stage-A recon on the five af12 cohorts (test split, scatter input), phases sharded
# across GPUs by run_dangi.py --gpus (docs/120 §6/§8: one model per GPU, contiguous phase blocks).
#
#   GPUS=0,1,2,3 NAME=dangi_scatter_4gpu bash tools/af12_dangi_timing.sh
#
# Dangi sibling of tools/af12_vggt_timing.sh, same safety net: refuses if <NAME> exists for any
# af12 subject (never overwrites), refuses unless every visible GPU is idle and the CPU load is
# low, snapshots every existing af12 file before/after and diffs them, monitors GPUs + CPU load
# every 2 s into $LOG/monitor.csv. Per-subject timing: <subject>/<NAME>/timing.json
# (breath.total_sec = preprocess+predict+translate for all T phases; stack reads / NIfTI writes
# are outside the span, recorded as io_load_sec / io_save_sec).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=training:.
GPUS=${GPUS:-0,1,2,3}
NAME=${NAME:?set NAME (the arm name, e.g. dangi_scatter_4gpu)}
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
MAX_LOAD=${MAX_LOAD:-8}          # node-wide load; other users' cgroups on a shared node count too
ARM=${ARM:-af12}                 # af12 | hrv12
SOURCES=${SOURCES:-cmrx2023_$ARM cmrx2024_$ARM cmrx2025_$ARM acdc_$ARM mnms_$ARM}
LOG=${LOG:-temp/${ARM}_timing_${NAME}}; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPUS
LOCAL_GPUS=$(seq -s, 0 $(( $(tr ',' '\n' <<< "$GPUS" | wc -l) - 1 )))

if compgen -G "evaluation/volumes/*_${ARM}/out/*/${NAME}" > /dev/null; then
    echo "REFUSING: ${NAME} already exists for an $ARM subject (pick another NAME):"
    ls -d evaluation/volumes/*_${ARM}/out/*/${NAME} | head; exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING: a process is already on a GPU"; nvidia-smi; exit 1; fi
LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
    echo "REFUSING: 1-min load average $LOAD > $MAX_LOAD (another job is using the CPU)"; exit 1; fi

snap() { find evaluation/volumes/*_${ARM}/out -type f -printf '%p %s %T@\n' | grep -v "/${NAME}/" | sort; }
snap > "$LOG/before.txt"
( while true; do
    ts=$(date +%s.%N); la=$(awk '{print $1}' /proc/loadavg)
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv,noheader,nounits | sed "s/^/$ts gpu,/;s/$/,load=$la/"
    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader | sed "s/^/$ts app,/"
    sleep 2
  done ) > "$LOG/monitor.csv" 2>&1 &
MON=$!
trap 'kill $MON 2>/dev/null' EXIT
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  gpus $GPUS  run pid $$  start $(date -Is)" | tee "$LOG/run.txt"
nvidia-smi --query-gpu=index,name --format=csv,noheader >> "$LOG/run.txt"

$PY evaluation/src/engine/run_dangi.py --split test --input scatter --arm-name "$NAME" \
    --sources $SOURCES --gpus "$LOCAL_GPUS" > "$LOG/run_dangi.log" 2>&1; rc=$?
kill $MON
snap > "$LOG/after.txt"
if diff -q "$LOG/before.txt" "$LOG/after.txt" > /dev/null; then echo "existing files untouched: OK"
else echo "!! existing $ARM files CHANGED — see diff $LOG/before.txt $LOG/after.txt"; rc=1; fi
awk -F', ' '/ app,/{print $2}' "$LOG/monitor.csv" | sort -u > "$LOG/gpu_pids.txt"
echo "PIDs seen on GPUs during the run: $(tr '\n' ' ' < "$LOG/gpu_pids.txt")  (only ours => clean)"
echo "end $(date -Is) rc=$rc" | tee -a "$LOG/run.txt"
exit $rc
