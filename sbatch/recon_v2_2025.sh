#!/bin/bash
# Re-reconstruct CMRxRecon2025 SAX with the FIXED ESPIRiT input domain (docs/54).
# 4-way shard; subject lists are pre-generated in scratch/recon_v2_shards/.
# Each shard writes its own report to avoid concurrent-write races; merge afterwards.
#SBATCH --job-name=recon_v2_2025
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%A_%a_recon_v2_2025.log

set -eo pipefail   # NOT -u: micromamba activate.d scripts reference unset vars
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr

K=${SLURM_ARRAY_TASK_ID}
D25=scratch/data/CMRxRecon2025
LIST=scratch/recon_v2_shards/2025_shard${K}.txt

echo "shard $K: $(wc -l < "$LIST") subjects on $(hostname)"
nvidia-smi -L

python tools/reconstruct_cmrx2025.py \
    --subjects $(tr '\n' ' ' < "$LIST") \
    --out-root  ${D25}/Cine_combined \
    --report    ${D25}/recon_report_v2_shard${K}.json \
    --stage-dir /tmp/cmrx2025_v2_${USER}_s${K}

echo "shard $K finished"
