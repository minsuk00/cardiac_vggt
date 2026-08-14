#!/bin/bash
#SBATCH --job-name=render_7row
#SBATCH --partition=standard
#SBATCH --account=jjparkcv0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/render_7row_%j.log

# CPU-only GIF rendering for the 7-row gated sweep: renders every npz under
# result/gated_model_sweep/ with the ORIGINAL matplotlib render_7row at dpi=130 (white bg,
# aligned), parallelized across the node's cores. Decoupled from GPU generation so the slow
# matplotlib rendering runs on a dedicated, uncontended CPU node.
set -e
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr            # activate ONCE (avoids the 'micromamba run' lockfile collision)
export PYTHONPATH=training:.

FILES=($(ls result/gated_model_sweep/*/*/*/*.npz | sort))
N=${#FILES[@]}
W=${SLURM_CPUS_PER_TASK:-16}
echo "rendering $N npz across $W workers on $(hostname)"
for w in $(seq 0 $((W - 1))); do
  SUB=""
  for i in $(seq "$w" "$W" $((N - 1))); do SUB="$SUB ${FILES[$i]}"; done
  [ -z "$SUB" ] && continue
  python -u tools/miitt_viz/rerender_hires.py $SUB > "/tmp/render_worker_${w}.log" 2>&1 &
done
wait
echo "ALL RENDER WORKERS DONE ($N npz)"
