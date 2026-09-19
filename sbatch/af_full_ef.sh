#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --job-name=af_full_ef
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af_full_ef_%j.out
#SBATCH --open-mode=append

# ============================================================================================
# docs/111 s9 full-cohort EF chain for ONE fetal arm, across the 4 rhythm cohorts (or 2, for
# oracle_balanced -- see SOURCES below). Adapted from sbatch/eval_ef_dice.sh (the live-campaign
# EF driver) with two changes: (1) account/partition jjparkcv98/spgpu (A40) per project
# convention, not the live script's jjparkcv_owned1/spgpu2; (2) `ef_dice.py score --out` writes
# to a REPO-LOCAL json, never the shared metric_results/<split>/_ef/<method>.json the live
# campaign owns -- keeping pilot/full-cohort rows out of that file was an explicit decision
# (docs/111). No aggregate.py step: this campaign doesn't fold into per-source
# metric_results/<split>/<source>/<method>.json the way the live campaign does; af_pilot_report.py
# reads the --out json directly.
#
# Submit ONE job per arm, in parallel:
#   METHOD=fetal_cmr_4d                  sbatch sbatch/af_full_ef.sh
#   METHOD=fetal_cmr_4d_oracle            sbatch sbatch/af_full_ef.sh
#   METHOD=fetal_cmr_4d_oracle_balanced  SOURCES="cmrx2024_hrv cmrx2024_af" sbatch sbatch/af_full_ef.sh
# ============================================================================================

set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}}
METHOD=${METHOD:?set METHOD=<arm>, e.g. fetal_cmr_4d / fetal_cmr_4d_oracle / fetal_cmr_4d_oracle_balanced}
SOURCES=${SOURCES:-"cmrx2024_regular_frozen cmrx2024_regular cmrx2024_hrv cmrx2024_af"}
export SPLIT=${SPLIT:-test}
WORK=/tmp/af_full_ef_${USER}_${SLURM_JOB_ID}
PY=/home/minsukc/micromamba/envs/svr/bin/python
OUT="$REPO/temp/afsim_full_ef_${METHOD}.json"

cd "$REPO"
echo "arm     : $METHOD"
echo "sources : $SOURCES"
echo "work    : $WORK"
echo "out     : $OUT"

mkdir -p "$WORK/in" "$WORK/seg" "$REPO/temp"
t0=$(date +%s)
PYTHONPATH=training:. $PY evaluation/src/score/ef_dice.py dump "$WORK/in" \
    --method "$METHOD" --cohorts $SOURCES
t1=$(date +%s); echo "[timing] dump: $((t1-t0))s  ($(ls "$WORK/in" | wc -l) volumes)"

bash evaluation/src/engine/run_seg.sh "$WORK/in" "$WORK/seg"
t2=$(date +%s); echo "[timing] seg: $((t2-t1))s"

PYTHONPATH=training:. $PY evaluation/src/score/ef_dice.py score "$WORK/seg" --input "$WORK/in" --out "$OUT"
t3=$(date +%s); echo "[timing] score: $((t3-t2))s"

rm -rf "$WORK"
echo "DONE -> $OUT"
