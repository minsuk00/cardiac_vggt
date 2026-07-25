#!/bin/bash
#SBATCH --job-name=ef_dice_ood
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=10:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/ef_dice_ood_%j.log
#
# OOD EF/Dice for the ep100 hub (gather05), method-matched via nnU-Net Task114 (2d, MMS).
# dump per-phase recon(clean/breath)+GT volumes -> segment -> EF(LV max/min)/Dice(LV/MYO/RV).
# EF is a ratio so the 12mm-pitch caveat (docs/39) does not apply. Runs on a GPU node (spgpu).
#   sbatch --account=jjparkcv0 sbatch/ef_dice_ood.sh
set -uo pipefail
VGGT=/home/minsukc/vggt; cd "$VGGT"
METHOD=${METHOD:-vggt_20260719_1f_gather05_ep99}
TAG=${METHOD#vggt_20260719_1f_}; TAG=${TAG%_ep99}
WORK=$VGGT/scratch/eval/_ef_ood/$TAG
IN=$WORK/input; SEG=$WORK/seg; OUTJ=$WORK/ef_ood_${TAG}.json
PY=/home/minsukc/micromamba/envs/svr/bin/python
ENV_SH=$VGGT/tools/nnunet_mnms_eval/env.sh

echo "[$(date +%H:%M:%S)] EF/Dice OOD  method=$METHOD  node=$(hostname)"
rm -rf "$WORK"; mkdir -p "$IN" "$SEG"

echo "[1/3] dump per-phase volumes (svr)"
$PY tools/ef_dice_1frame.py dump "$IN" --method "$METHOD" --cohorts miitt ocmr acdc
echo "  dumped $(ls "$IN"/*_0000.nii.gz 2>/dev/null | wc -l) volumes"

echo "[2/3] nnU-Net Task114 2d segment (nnunet env)"
micromamba run -n nnunet bash -c "source '$ENV_SH' && \
  nnUNet_predict -i '$IN' -o '$SEG' -t 114 -m 2d -tr nnUNetTrainerV2_MMS"
echo "  segmented $(ls "$SEG"/*.nii.gz 2>/dev/null | wc -l) volumes"

echo "[3/3] EF + Dice score (svr)"
$PY tools/ef_dice_1frame.py score "$SEG" --input "$IN" --out "$OUTJ"
echo "EF_DICE_DONE $(date +%H:%M:%S) -> $OUTJ"
