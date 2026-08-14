#!/bin/bash
# Re-reconstruct CMRxRecon2023 SAX with the FIXED ESPIRiT input domain (docs/54).
# Single task: the 2023 driver has no subject-selection flag to shard on, and it is the
# cheapest year anyway -- measured 32 s/subject (103 min for 194) on the v1 run.
# It has its own skip-guard on sax/4d_recon.nii.gz, so a requeue resumes.
#SBATCH --job-name=recon_v2_2023
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%j_recon_v2_2023.log

set -eo pipefail   # NOT -u: micromamba activate.d scripts reference unset vars
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr

echo "2023 recon on $(hostname)"
nvidia-smi -L

python tools/reconstruct_cmrx2023.py \
    --stage-dir /tmp/cmrx2023_v2_${USER}

echo "2023 finished"
