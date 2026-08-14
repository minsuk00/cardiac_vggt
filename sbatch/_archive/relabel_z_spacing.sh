#!/bin/bash
# Z-spacing relabel for CMRxRecon 2023 + 2024 after a re-reconstruction.
# WHY THIS IS NEEDED: the 2023/2024 drivers bake `SliceThickness` (8 or 6 mm) into the NIfTI, NOT
# the true centre-to-centre pitch (8+4=12 mm, or 6+4=10 mm; docs/27). The pitch is applied by this
# separate post-recon pass -- forget it and every volume carries a ~33% Z error. 2025 needs no pass
# (its driver bakes the pitch in-line).
# Parallel + ATOMIC (tmp + os.replace): single-threaded on GPFS ran ~1 file/s (~2 h); a kill mid-
# nib.save previously truncated a NIfTI to 0 bytes, which os.replace makes impossible.
#SBATCH --job-name=relabel_z
#SBATCH --account=jjparkcv0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%j_relabel_z.log

set -eo pipefail   # NOT -u: micromamba activate.d scripts reference unset vars
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr

export RELABEL_WORKERS=${SLURM_CPUS_PER_TASK:-16}
echo "relabel on $(hostname) with $RELABEL_WORKERS workers"
python -u tools/relabel_slice_spacing_parallel.py
