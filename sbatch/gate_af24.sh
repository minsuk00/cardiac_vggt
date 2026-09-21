#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48g
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --job-name=gate_af24
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/gate_af24_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# Self-gating front end for Fetal CMR 4D on the 24-frame `af24` cohorts, all 180 test subjects.
# dump -> seg -> assemble, one shard per array task.
#
# THIS IS THE STARTING GUN. Fetal's reconstruction is the campaign's long pole and it cannot
# begin until the gate has written cardphase.txt / rrintervals.txt / stack4d.nii.gz, so this GPU
# step sits in front of a many-hour CPU job. Everything else (VGGT recon, the 3d_fullres GT
# segmentation) is GPU work that should fill the time WHILE fetal runs, not before it.
#
# The gate stays at nnU-Net `-m 2d`. It is not a metric -- it is the baseline's OWN gating,
# segmenting the INPUT frames to find each slice's ED anchor. The scoring segmenter moved to
# 3d_fullres; this one must not follow, or we would be changing the method under comparison.
#
# PER-SHARD WORK TAG, which is not optional. fetal4d_gate.py's default work dir is SHARED across
# every source of a split and holds the live campaign's gating data; `dump` overwrites inputs with
# no guard and `seg` scans the whole directory. Sharding by subject into `af24_s<i>` keeps each
# task's nnU-Net input set disjoint AND keeps all of them clear of the campaign's dir.
#
# 180 subjects / 8 = 22-23 per shard, x24 frames = ~550 volumes each (4,320 total).
#
# NOT self-submitting -- submit with `sbatch sbatch/gate_af24.sh`.
# Dry-run the work list first:  bash sbatch/gate_af24.sh --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
export SPLIT=${SPLIT:-test}
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}

ARM=${ARM:-af24}
N_SHARDS=${N_SHARDS:-8}
IDX=${SLURM_ARRAY_TASK_ID:-0}
TAG="${ARM}_s${IDX}"
SOURCES=$($PY -c "import sys; sys.path.insert(0,'evaluation'); import paths; print(' '.join(f'{d}_${ARM}' for d in paths.DATASETS))")

# Round-robin the flat (source, subject) list so shards get a mix of small- and large-D cohorts.
SUBJ=$($PY - "$IDX" "$N_SHARDS" "$ARM" <<'EOF'
import os, sys
sys.path.insert(0, "evaluation")
import paths
idx, n, arm = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
pairs = []
for ds in paths.DATASETS:
    c = f"{ds}_{arm}"
    keep, _ = paths.filter_by_split(c, paths.subjects(c), os.environ.get("SPLIT", "test"))
    pairs += [(c, s) for s in keep]
print(" ".join(s for _, s in pairs[idx::n]))
EOF
) || { echo "failed to build the work list"; exit 2; }

if [ -z "$SUBJ" ]; then echo "task $IDX: nothing to do"; exit 0; fi
NSUB=$(echo "$SUBJ" | wc -w)
echo "=== task $IDX/$N_SHARDS  arm=$ARM  work-tag=$TAG  subjects=$NSUB  ($((NSUB*24)) frames) ==="
echo "$SUBJ" | tr ' ' '\n' | sed 's/^/    /'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

if [ "${1:-}" = "--dry-run" ]; then echo "dry-run: nothing run"; exit 0; fi

rc=0
for STAGE in dump seg assemble; do
  echo "--- $STAGE"
  $PY evaluation/src/engine/fetal4d_gate.py "$STAGE" \
      --split "$SPLIT" --sources $SOURCES --subjects $SUBJ --work-tag "$TAG" || { rc=$?; break; }
done

echo "=== task $IDX done (rc=$rc) ==="
exit $rc
