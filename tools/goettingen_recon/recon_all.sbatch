#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --job-name=gott_recon
#SBATCH --array=0-67%4
#SBATCH --output=/home/minsukc/vggt/slurm_logs/gott_recon_%A_%a.log

# Classical RT-NLINV recon of one Göttingen radial volume (all SAX slices) per array task,
# then convert to a magnitude NIfTI cine. No neural net.
set -eu
REPO=/home/minsukc/vggt
cd "$REPO"                                     # vol_list paths are repo-relative; resolve regardless of SLURM CWD
LIST=$REPO/tools/goettingen_recon/vol_list.txt
VOL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
NAME=$(basename "$VOL")                       # e.g. vol0001_vis1
OUT=$REPO/scratch/data/goettingen/recon/$NAME
NII=$OUT/${NAME}.nii.gz

if [ -f "$NII" ]; then echo "$NII exists, skip"; exit 0; fi
mkdir -p "$OUT"

echo "[$(date)] recon $NAME (task $SLURM_ARRAY_TASK_ID) on $(hostname)"
time bash "$REPO/tools/goettingen_recon/recon_vol.sh" "$VOL" "$OUT"

echo "[$(date)] convert -> NIfTI"
# micromamba is a shell function (not on PATH in a non-interactive batch shell) -> use the binary
/home/minsukc/.local/bin/micromamba run -n svr python "$REPO/tools/goettingen_recon/cfl_to_nifti.py" \
    "$OUT/img_rt_fil" "$NII" --matrix 160

echo "[$(date)] done $NAME"
