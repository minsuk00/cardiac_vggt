#!/bin/bash
# Timed NeSVoR recon on the five af12 cohorts (test split, scatter input), the 12 per-phase fits
# sharded across GPUs by run_nesvor.sh's GPUS= mode (docs/120 §6/§8: phase p -> GPU p mod N, one
# fit per GPU at a time; T=12 on 4 GPUs = 3 sequential fits per GPU).
#
#   GPUS=0,1,2,3 NAME=nesvor_scatter_4gpu bash tools/af12_nesvor_timing.sh
#   RESUME=1 ... same command ...     # continue after a walltime cut (complete subjects are skipped)
#
# NeSVoR sibling of tools/af12_vggt_timing.sh, same safety net: refuses if <NAME> exists for any
# af12 subject (never overwrites; RESUME=1 bypasses only that check), refuses unless every visible
# GPU is idle and the CPU load is low, snapshots every existing af12 file before/after and diffs
# them, monitors GPUs + CPU load every 2 s into $LOG/monitor.csv. Per-subject timing:
# <subject>/<NAME>/recon_breath/total_wall.sec = wall of the 12 fits with N in flight (the span
# includes each `nesvor reconstruct` process start + its own stack read: inherent to the CLI);
# time_t*.sec = each fit alone. A subject interrupted mid-run keeps its finished vol_t*.nii.gz and
# is completed on RESUME — its total_wall.sec is then the resumed invocation only (provenance says
# so); exclude such subjects from the mean or delete their <NAME>/ dir before resuming.
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS=${GPUS:-0,1,2,3}
NAME=${NAME:?set NAME (the arm name, e.g. nesvor_scatter_4gpu)}
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
MAX_LOAD=${MAX_LOAD:-8}          # node-wide load; other users' cgroups on a shared node count too
SOURCES=(${SOURCES:-cmrx2023 cmrx2024 cmrx2025 acdc mnms})
LOG=${LOG:-temp/af12_timing_${NAME}}; mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPUS
LOCAL_GPUS=$(seq -s, 0 $(( $(tr ',' '\n' <<< "$GPUS" | wc -l) - 1 )))

if [ -z "${RESUME:-}" ] && compgen -G "evaluation/volumes/*_af12/out/*/${NAME}" > /dev/null; then
    echo "REFUSING: ${NAME} already exists for an af12 subject (pick another NAME, or RESUME=1):"
    ls -d evaluation/volumes/*_af12/out/*/${NAME} | head; exit 1; fi
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "REFUSING: a process is already on a GPU"; nvidia-smi; exit 1; fi
LOAD=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$LOAD" -v m="$MAX_LOAD" 'BEGIN{exit !(l>m)}'; then
    echo "REFUSING: 1-min load average $LOAD > $MAX_LOAD (another job is using the CPU)"; exit 1; fi
[ -x "${NESVOR_BIN:-/home/minsukc/micromamba/envs/nesvor-t2/bin/nesvor}" ] || { echo "REFUSING: nesvor binary missing (docs/90)"; exit 1; }

snap() { find evaluation/volumes/*_af12/out -type f -printf '%p %s %T@\n' | grep -v "/${NAME}/" | sort; }
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

rc=0
for S in "${SOURCES[@]}"; do
    DS="${S}_af12"
    # the cohort = the bundles whose own manifest says split=test (evaluation/paths.py contract)
    SUBJECTS=$(cd evaluation && "$PY" -c "import paths; k,_=paths.filter_by_split('$DS', paths.subjects('$DS'), 'test'); print(' '.join(k))")
    echo "=== $DS  $(wc -w <<< "$SUBJECTS") subjects  $(date -Is)" | tee -a "$LOG/run.txt"
    for SUBJ in $SUBJECTS; do
        echo "--- $DS/$SUBJ $(date -Is)" >> "$LOG/run.txt"
        EVAL_DATASET="$DS" INPUT=scatter METHOD="$NAME" GPUS="$LOCAL_GPUS" \
            bash evaluation/src/engine/run_nesvor.sh "$SUBJ" breath >> "$LOG/${DS}.log" 2>&1 || rc=$?
        tail -1 "$LOG/${DS}.log" | tee -a "$LOG/run.txt"
    done
done
kill $MON
snap > "$LOG/after.txt"
if diff -q "$LOG/before.txt" "$LOG/after.txt" > /dev/null; then echo "existing files untouched: OK"
else echo "!! existing af12 files CHANGED — see diff $LOG/before.txt $LOG/after.txt"; rc=1; fi
awk -F', ' '/ app,/{print $2}' "$LOG/monitor.csv" | sort -u > "$LOG/gpu_pids.txt"
echo "PIDs seen on GPUs during the run: $(wc -l < "$LOG/gpu_pids.txt") distinct (one nesvor process per fit; all ours => clean)"
echo "end $(date -Is) rc=$rc" | tee -a "$LOG/run.txt"
exit $rc
