#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --job-name=af_disambig_ef
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af_disambig_ef_%j.out
#SBATCH --open-mode=append

# ============================================================================================
# docs/113 cut-off vs hold disambiguation: EF/Dice for ONE arm across the two new rhythm
# cohorts. Same chain as sbatch/af_full_ef.sh (dump -> nnU-Net seg -> score).
#
# THE ONE DIFFERENCE THAT MATTERS: this writes to temp/afsim_DISAMBIG_ef_<method>.json.
# af_full_ef.sh writes temp/afsim_full_ef_<method>.json, which holds the 4-cohort AF campaign
# results docs/111/112 are built on -- reusing that script here would silently clobber them.
#
# Submit one job per arm, in parallel:
#   METHOD=fetal_cmr_4d                      sbatch sbatch/af_disambig_ef.sh
#   METHOD=vggt_final518_diff1000_ep300      sbatch sbatch/af_disambig_ef.sh
# ============================================================================================
set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}}
METHOD=${METHOD:?set METHOD=<arm>}
SOURCES=${SOURCES:-"cmrx2024_af_rvr cmrx2024_af_pause"}
export SPLIT=${SPLIT:-test}
WORK=/tmp/af_disambig_ef_${USER}_${SLURM_JOB_ID:-local}
PY=/home/minsukc/micromamba/envs/svr/bin/python
OUT="$REPO/temp/afsim_disambig_ef_${METHOD}.json"

cd "$REPO"
echo "arm     : $METHOD"
echo "sources : $SOURCES"
echo "out     : $OUT"
[ -e "$OUT" ] && { echo "REFUSING: $OUT already exists — move it aside first"; exit 2; }

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
