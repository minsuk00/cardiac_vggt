#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --job-name=fetal4d_nframes
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/fetal4d_nframes_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# Fetal CMR 4D frames-per-slice probe (tools/fetal4d_nframes_probe.py): task k = subject k//2,
# N = 12 or 24 (k%2). Same CPU allocation as the campaign (rhythm24_fetal.sh). Writes ONLY under
# temp/fetal4d_nframes_probe/. Run `... setup` once first.
#
#   sbatch sbatch/fetal4d_nframes_probe.sh
# ============================================================================================
set -uo pipefail
REPO=${REPO:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
export OMP=${OMP:-${SLURM_CPUS_PER_TASK:-16}} ROBUST=${ROBUST:-1} RES=${RES:-1.4}
unset T
K=${SLURM_ARRAY_TASK_ID:-0}
IDX=$((K / 2)); N=$(( K % 2 == 0 ? 12 : 24 ))
echo "=== task $K: subject $IDX  N=$N  OMP=$OMP  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) ==="
$PY tools/fetal4d_nframes_probe.py run --index "$IDX" --n "$N"
rc=$?
echo "=== task $K done (rc=$rc) ==="
exit $rc
