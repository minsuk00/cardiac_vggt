#!/bin/bash
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --gpu_cmode=shared
#SBATCH --requeue
#SBATCH --open-mode=append

# CORSEG-DICE ARM w100 — corseg_weight=0.1 (~37x the full-L1 gradient; Dice-DOMINANT regime).
# User-requested high-dose arm (2026-08-11): the head outputs POSITIONS, not intensities —
# V_canon can only rearrange real input pixels, so there is no free "painter" channel for the
# model to please the segmenter with; the bounded risk is aggressive pixel-shoving at the
# expense of intensity fidelity. Judge against heartl1_w000 and the w002 parity arm.
CORSEG_WEIGHT=0.1
VARIANT_TAG="w100"
# Resume the spgpu2 run (killed 2026-08-12 to free the owned L40S) in place, ~42k steps in.
# Clear this for a fresh-from-base launch.
RESUME_EXP_NAME="213515736_mri_volume_corsegdice_w100_dynamic_axial_cmrx24only"
source /home/minsukc/vggt/sbatch/corsegdice_common.sh
