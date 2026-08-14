#!/bin/bash
#SBATCH --job-name=cmrx25_extract
#SBATCH --account=jjparkcv0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/cmrx25_extract_%j.log

# Merge + extract (all modalities, nothing dropped) the CMRxRecon2025 V2 zip parts.
# CPU/IO job — no GPU. Runs all three series sequentially (TrainingData first so it can be
# inspected as soon as it finishes). Bottleneck is GPFS bandwidth, not cores.
set -euo pipefail
cd /home/minsukc/vggt
echo "host=$(hostname) start=$(date)"
# Cine only (SAX + LAX) — the only modality the model uses; merged zips retain all 9 modalities.
INCLUDE='*/Cine/*' bash tools/merge_extract_cmrxrecon2025.sh TrainingData TaskR1 TaskR2
echo "end=$(date)"
