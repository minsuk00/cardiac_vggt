#!/bin/bash
# Re-reconstruct CMRxRecon2024 SAX with the FIXED ESPIRiT input domain (docs/54).
# Uses tools/reconstruct_cmrx2024.py, NOT the archive's own main(): the latter does not patch
# shutil.copy2, so its packaging step hits shutil.SameFileError against the pre-existing
# sax/cine_sax.mat SYMLINK and kills the shard (jobs 55163115_[0-1] died that way).
# Writes in place to CMRxRecon2024/Cine_combined -- the v1 NIfTIs AND the stale v1-derived
# heart_seg / heart_roi / dvf_* were both moved to CMRxRecon2024_recon_v1_espirit_imagedomain/.
#SBATCH --job-name=recon_v2_2024
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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

python tools/reconstruct_cmrx2024.py \
    --subject-file "$LIST" \
    --stage-dir /tmp/cmrx2024_v2_${USER}_s${K}

echo "shard $K finished"
