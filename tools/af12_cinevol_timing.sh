#!/bin/bash
# Timed CiNeVol recon on the five af12 cohorts with the fit step + export sharded across GPUs
# (docs/120 §6 timing table, docs/121 for the --gpus implementation).
#
#   GPUS=0,1,2,3 NAME=cinevol_4gpu bash tools/af12_cinevol_timing.sh
#
# CiNeVol sibling of tools/af12_vggt_timing.sh, same safety net: refuses if <NAME> exists for any
# af12 subject (never overwrites; run_cinevol.py also refuses per subject), refuses unless every
# visible GPU is idle and the CPU load is low, requires the Grid4D encoder to be built (the build
# must never land inside a timed subject), snapshots every existing af12 file before/after and
# diffs them, monitors GPUs + CPU load every 2 s into $LOG/monitor.csv (contamination audit).
# Per-subject timing: <subject>/<NAME>/recon_breath/provenance.txt "wall seconds : prepare | fit |
# export | total" and total_wall.sec (docs/121: total = prepare + fit + export).
set -uo pipefail
cd "$(dirname "$0")/.."
source baselines/cinevol/env.sh          # module load + CINEVOL_PY + GRID4D_BUILD_DIR + PYTHONPATH
GPUS=${GPUS:-0,1,2,3}
NAME=${NAME:?set NAME (the arm name, e.g. cinevol_4gpu)}
MAX_LOAD=${MAX_LOAD:-8}          # node-wide load; other users' cgroups on a shared node count too
LOG=${LOG:-temp/af12_timing_${NAME}}; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPUS
LOCAL_GPUS=$(seq -s, 0 $(( $(tr ',' '\n' <<< "$GPUS" | wc -l) - 1 )))

# RESUME=1: continue a run cut by the walltime (run_cinevol.py skips stamped subjects and refuses
# unstamped dirs, so nothing is overwritten either way); otherwise a pre-existing NAME is an error.
if [ -z "${RESUME:-}" ] && compgen -G "evaluation/volumes/*_af12/out/*/${NAME}" > /dev/null; then
    echo "REFUSING: ${NAME} already exists for an af12 subject (pick another NAME, or RESUME=1):"
    ls -d evaluation/volumes/*_af12/out/*/${NAME} | head; exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING: a process is already on a GPU"; nvidia-smi; exit 1; fi
LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
    echo "REFUSING: 1-min load average $LOAD > $MAX_LOAD (another job is using the CPU)"; exit 1; fi
[ -f "$GRID4D_BUILD_DIR/_hash_encoder.so" ] || { echo "REFUSING: Grid4D encoder not built — run: $CINEVOL_PY tools/run_cinevol.py --build-only"; exit 1; }

snap() { find evaluation/volumes/*_af12/out -type f -printf '%p %s %T@\n' | grep -v "/${NAME}/" | grep -v "/${NAME}.partial" | sort; }
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

# run_cinevol.py iterates all five af12 sources itself; a stamped subject is skipped, so a run cut
# by the walltime is resumed with RESUME=1 (same NAME) on the next allocation. A subject killed
# mid-fit leaves <NAME>.partial<pid>_<ts>/ next to it — remove that by hand before resuming.
$CINEVOL_PY -u tools/run_cinevol.py --arm af12 --arm-name "$NAME" --gpus "$LOCAL_GPUS" \
    > "$LOG/run_cinevol.log" 2>&1; rc=$?
kill $MON
snap > "$LOG/after.txt"
if diff -q "$LOG/before.txt" "$LOG/after.txt" > /dev/null; then echo "existing files untouched: OK"
else echo "!! existing af12 files CHANGED — see diff $LOG/before.txt $LOG/after.txt"; rc=1; fi
awk -F', ' '/ app,/{print $2}' "$LOG/monitor.csv" | sort -u > "$LOG/gpu_pids.txt"
echo "PIDs seen on GPUs during the run: $(tr '\n' ' ' < "$LOG/gpu_pids.txt")  (only ours => clean)"
echo "end $(date -Is) rc=$rc" | tee -a "$LOG/run.txt"
exit $rc
