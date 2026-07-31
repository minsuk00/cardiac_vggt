#!/bin/bash
# Convert ACDC / M&Ms-1 4D cines into the CMRxRecon on-disk layout (docs/58 A2).
#
# WHY A BATCH JOB: M&Ms is ~345 subjects x (166 MB read + 59 MB write) = ~78 GB of GPFS IO plus
# float32 gzip encoding. Single-threaded on GPFS that is hours (cf. sbatch/relabel_z_spacing.sh,
# which measured ~1 file/s), and the interactive GPU node is cgroup-limited to 1 CPU.
#
# Writes are ATOMIC (tmp + os.replace, pid-tagged tmp name), so a kill mid-run leaves no truncated
# NIfTI and a re-run simply overwrites. Idempotent — safe to resubmit.
#
# MEMORY: each worker holds one full 4D array; the largest M&Ms cine is ~548x512x20x36 float32
# (~0.8 GB), so 24 workers peak around 20 GB. 96 G leaves headroom.
#
# Usage:  sbatch sbatch/convert_sax_layout.sh            # both sources
#         SOURCE=mnms sbatch sbatch/convert_sax_layout.sh
#SBATCH --job-name=convert_sax
#SBATCH --account=jjparkcv0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%j_convert_sax.log

set -eo pipefail   # NOT -u: micromamba activate.d scripts reference unset vars
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr

SOURCE="${SOURCE:-both}"
WORKERS="${WORKERS:-24}"

echo "host=$(hostname)  source=${SOURCE}  workers=${WORKERS}  start=$(date)"
python tools/convert_to_sax_layout.py --source "${SOURCE}" --workers "${WORKERS}" --apply
echo "done=$(date)"

echo "=== on-disk result ==="
for d in ACDC_sax MNMs_sax; do
    p="scratch/data/${d}"
    [ -d "$p" ] && echo "  ${d}: $(ls -d ${p}/*/ 2>/dev/null | wc -l) subjects, $(du -sh ${p} | cut -f1)"
done
