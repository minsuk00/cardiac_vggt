#!/bin/bash
#SBATCH --job-name=selfgate_recon
#SBATCH --account=jjparkcv0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/selfgate_recon_%j.log
# Faithful author-parameter reconstructCardiac (1.25mm, rec 10/20, robust ON) on the LV-area
# self-gated cardphase (doc 35). CPU-only, memory-heavy -> standard partition, 128G (the 32G
# interactive alloc OOM-kills it, doc 34 §7 / doc 35 §10.7).
# Usage: sbatch sbatch/selfgate_recon_faithful.sh [Volunteer1 ...]
set -uo pipefail
cd /home/minsukc/vggt
VOLS="${@:-Volunteer1}"
echo "faithful self-gated recon: $VOLS  ($(date))"
bash baselines/fetal_cmr_4d/run_selfgate_recon.sh $VOLS
echo "done ($(date))"
