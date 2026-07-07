#!/bin/bash
#SBATCH --job-name=v1_4d_svrtk
#SBATCH --account=jjparkcv0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/v1_4d_%j.log
export OMP_NUM_THREADS=16
cd /home/minsukc/vggt
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"; date
bash baselines/fetal_cmr_4d/run_selfgate_recon.sh Volunteer1
echo "SBATCH_V1_4D_DONE"; date
