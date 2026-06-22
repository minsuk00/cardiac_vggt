#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --time=00:30:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%j_diagnose_4way.log
#SBATCH --job-name=diagnose_4way

cd /home/minsukc/vggt
PYTHONPATH=training:. micromamba run -n svr python tools/diagnose_4way_refiner.py
