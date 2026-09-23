#!/bin/bash
# Timed NeSVoR recon on the five af12 cohorts (test split, scatter input), the 12 per-phase fits
# sharded across GPUs by run_nesvor.sh's GPUS= mode (docs/120 §6/§8: phase p -> GPU p mod N, one
# fit per GPU at a time; T=12 on 4 GPUs = 3 sequential fits per GPU).
#
#   GPUS=0,1,2,3 NAME=nesvor_4gpu bash tools/af12_nesvor_timing.sh     # -> arm nesvor_4gpu_scatter
#   RESUME=1 ... same command ...     # continue after a walltime cut (complete subjects are skipped)
#   DRY_RUN=1 ... same command ...    # print the per-subject work list (THICK) only; no guards, no run
#
# Each cohort goes through run_baselines.py (NOT run_nesvor.sh directly): it supplies the per-subject
# THICK (thickness_mm of the base source) and MASK_FILE=mask_heart_pad10.nii.gz exactly as the main
# NeSVoR benchmark does. Calling run_nesvor.sh directly silently fell back to THICK=8 and the unpadded
# mask_heart.nii.gz (bug found 2026-09-23 on the first af12 test subject). GPUS passes through the
# environment to run_nesvor.sh. run_baselines.py names the arm METHOD + "_scatter".
#
# NeSVoR sibling of tools/af12_vggt_timing.sh, same safety net: refuses if the arm exists for any
# af12 subject (never overwrites; RESUME=1 bypasses only that check), refuses unless every visible
# GPU is idle and the CPU load is low, snapshots every existing af12 file before/after and diffs
# them, monitors GPUs + CPU load every 2 s into $LOG/monitor.csv. Per-subject timing:
# <subject>/<NAME>_scatter/recon_breath/total_wall.sec = wall of the 12 fits with N in flight (the span
# includes each `nesvor reconstruct` process start + its own stack read: inherent to the CLI);
# time_t*.sec = each fit alone. A subject interrupted mid-run keeps its finished vol_t*.nii.gz and
# is completed on RESUME — its total_wall.sec is then the resumed invocation only (provenance says
# so); exclude such subjects from the mean or delete their <NAME>/ dir before resuming.
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS=${GPUS:-0,1,2,3}
NAME=${NAME:?set NAME (the METHOD base, e.g. nesvor_4gpu -> arm nesvor_4gpu_scatter)}
ARMDIR="${NAME}_scatter"         # what run_baselines.py names it for --input scatter
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
MAX_LOAD=${MAX_LOAD:-8}          # node-wide load; other users' cgroups on a shared node count too
SOURCES=(${SOURCES:-cmrx2023 cmrx2024 cmrx2025 acdc mnms})
ARM=${ARM:-af12}                 # af12 | hrv12
export PYTHONPATH=training:.
COHORTS=$(for S in "${SOURCES[@]}"; do printf '%s_%s ' "$S" "$ARM"; done)
# SHARDS=a-b (with N_SHARDS=N): run only run_baselines.py shards a..b of N, so the cohort can be
# split STATICALLY across nodes (e.g. 0-6 here, 7-9 on a second 4x L40S job). Never let two nodes
# run overlapping shards: run_nesvor.sh has no lock, two writers on one subject corrupt it.
N_SHARDS=${N_SHARDS:-10}
SHARDS=${SHARDS:-0-$((N_SHARDS - 1))}
SHARD_IDS=$(seq "${SHARDS%-*}" "${SHARDS#*-}")
if [ -n "${DRY_RUN:-}" ]; then
    for I in $SHARD_IDS; do
        METHOD="$NAME" "$PY" evaluation/src/engine/run_baselines.py --method nesvor --variant breath \
            --split test --input scatter --sources $COHORTS --shard "$I" "$N_SHARDS" --dry-run || exit $?
    done; exit 0; fi
LOG=${LOG:-temp/${ARM}_timing_${ARMDIR}_shards${SHARDS}of${N_SHARDS}}; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPUS
LOCAL_GPUS=$(seq -s, 0 $(( $(tr ',' '\n' <<< "$GPUS" | wc -l) - 1 )))

if [ -z "${RESUME:-}" ] && compgen -G "evaluation/volumes/*_${ARM}/out/*/${ARMDIR}" > /dev/null; then
    echo "REFUSING: ${ARMDIR} already exists for an $ARM subject (pick another NAME, or RESUME=1):"
    ls -d evaluation/volumes/*_${ARM}/out/*/${ARMDIR} | head; exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING: a process is already on a GPU"; nvidia-smi; exit 1; fi
LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
    echo "REFUSING: 1-min load average $LOAD > $MAX_LOAD (another job is using the CPU)"; exit 1; fi
[ -x "${NESVOR_BIN:-/home/minsukc/micromamba/envs/nesvor-t2/bin/nesvor}" ] || { echo "REFUSING: nesvor binary missing (docs/90)"; exit 1; }

snap() { find evaluation/volumes/*_${ARM}/out -type f -printf '%p %s %T@\n' | grep -v "/${ARMDIR}/" | sort; }
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

# run_baselines.py: split filter (manifest split=test), per-subject THICK + padded MASK_FILE, skips
# stamped subjects; GPUS reaches run_nesvor.sh through the environment.
echo "shards $SHARDS of $N_SHARDS" | tee -a "$LOG/run.txt"
rc=0
for I in $SHARD_IDS; do
    METHOD="$NAME" GPUS="$LOCAL_GPUS" "$PY" evaluation/src/engine/run_baselines.py --method nesvor \
        --variant breath --split test --input scatter --sources $COHORTS --shard "$I" "$N_SHARDS" \
        >> "$LOG/run_baselines.log" 2>&1 || rc=$?
done
grep -E "^--- \[|RECON_DONE|FAILED" "$LOG/run_baselines.log" | tail -3 | tee -a "$LOG/run.txt"
kill $MON
snap > "$LOG/after.txt"
if diff -q "$LOG/before.txt" "$LOG/after.txt" > /dev/null; then echo "existing files untouched: OK"
else echo "!! existing $ARM files CHANGED — see diff $LOG/before.txt $LOG/after.txt"; rc=1; fi
awk -F', ' '/ app,/{print $2}' "$LOG/monitor.csv" | sort -u > "$LOG/gpu_pids.txt"
echo "PIDs seen on GPUs during the run: $(wc -l < "$LOG/gpu_pids.txt") distinct (one nesvor process per fit; all ours => clean)"
echo "end $(date -Is) rc=$rc" | tee -a "$LOG/run.txt"
exit $rc
