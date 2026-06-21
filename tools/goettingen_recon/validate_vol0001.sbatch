#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=00:40:00
#SBATCH --job-name=gott_validate
#SBATCH --output=/home/minsukc/vggt/slurm_logs/gott_validate_%j.log

set -eu
DATA=/home/minsukc/vggt/scratch/data/goettingen_radial
OUT=/home/minsukc/vggt/scratch/data/goettingen_recon/vol0001_slc12
mkdir -p "$OUT"

# single mid slice (12 of 24) for a fast end-to-end validation
time bash /home/minsukc/vggt/tools/goettingen_recon/recon_vol.sh "$DATA/vol0001_vis1" "$OUT" 12

echo "=== output cfl files ==="
source /home/minsukc/vggt/tools/goettingen_recon/bart_env.sh
for f in "$OUT"/img_rt "$OUT"/img_rt_fil "$OUT"/img_rt_fil_nlmean; do
  [ -f "$f.hdr" ] && echo "$f : dims = $(head -2 "$f.hdr" | tail -1)"
done
