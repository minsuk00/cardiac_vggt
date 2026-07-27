#!/bin/bash
# Re-reconstruct CMRxRecon2024 SAX with the FIXED ESPIRiT input domain (docs/54).
# Uses the shared recon's own main(), which takes `--subjects <Dataset>/<PID>` and writes
# in place to CMRxRecon2024/Cine_combined -- correct here, because the v1 recon NIfTIs were
# already moved to CMRxRecon2024_recon_v1_espirit_imagedomain/ and the per-subject
# heart_seg / heart_roi / dvf_* artifacts were deliberately left in place.
#SBATCH --job-name=recon_v2_2024
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%A_%a_recon_v2_2024.log

set -eo pipefail   # NOT -u: micromamba activate.d scripts reference unset vars
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr

K=${SLURM_ARRAY_TASK_ID}
LIST=scratch/recon_v2_shards/2024_shard${K}.txt

echo "shard $K: $(wc -l < "$LIST") subjects on $(hostname)"
nvidia-smi -L

python _archive/batch_reconstruct_cmrxrecon2024.py \
    --device 0 \
    --subjects $(tr '\n' ' ' < "$LIST")

echo "shard $K finished"
