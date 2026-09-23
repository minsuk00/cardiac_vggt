#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=192g
#SBATCH --time=12:00:00
#SBATCH --job-name=af12_nesvor_shard
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/af12_nesvor_shard_%j.out

# ============================================================================================
# Second half of the af12 NeSVoR timing cohort (docs/120 §6): run_baselines.py shards
# ${SHARDS:-7-9} of 10 (~54 subjects, ~5.4 h) on 4x L40S — the SAME allocation as the node running
# shards 0-6 (spgpu2, 4x L40S, 16 CPUs, 192 GB), GPUS=0,1,2,3, phase p on GPU p mod 4.
# Static split: the two nodes must never run overlapping shards (no per-subject lock).
# RESUME=1 only bypasses the launcher's "arm already exists" guard (the other node creates the
# arm dirs); stamped subjects are skipped and nothing is overwritten.
#
# NOT self-submitting:  sbatch sbatch/af12_nesvor_shard.sh     (SHARDS override: --export=ALL,SHARDS=7-9)
# ============================================================================================
set -uo pipefail
REPO=${REPO:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader
RESUME=1 SHARDS=${SHARDS:-7-9} N_SHARDS=10 GPUS=0,1,2,3 NAME=nesvor_4gpu \
    bash tools/af12_nesvor_timing.sh
