#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=2:00:00
#SBATCH --array=0-19
#SBATCH --job-name=build_af24
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/build_af24_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# Build the 24-frame (2 nominal beat) rhythm cohorts for the final paper run.
#
#   <source>_regular24   periodic R-R, frame-wise breathing   -- the CONTROL
#   <source>_hrv24       R-R ~ N(1.0, 0.15)
#   <source>_af24        R-R ~ N(1.0, 0.25)  (Markl 2015)
#
# over all 180 TEST subjects: cmrx2023 26, cmrx2024 38, cmrx2025 46, acdc 21, mnms 49.
#
# WHY 24 frames: it hands Fetal CMR 4D TWO images per output bin per slice instead of one, so
# its uniform-ramp gate averages them. Under a regular rhythm the pair is one beat apart (the
# same cardiac phase -> denoising); under af it is 2.04 of 12 phases apart on average
# (-> temporal blur). Measured bin_phase_sd 37.2deg -> 59.7deg, versus ~+2deg for any
# rhythm-parameter change. The baseline gets MORE data, not less.
#
# NEW COHORT NAMES, never --overwrite: the 12-frame cohorts behind docs/111-112 hold bundles and
# reconstructions that are NOT regenerable from git. build() also returns "skipped" when a
# manifest already exists, so re-running or requeueing only does the remainder.
#
# regular_frozen24 is deliberately NOT built here. It exists only as the bit-exactness check
# (build_af_bundle.py --check --check-suffix 24) and is wanted on 1-2 subjects, not 180.
#
# CPU-only: the simulator is numpy/torch-on-CPU plus NIfTI I/O, no GPU anywhere. Measured 110 s
# for 3 arms on the largest subject in the cohort (MNMs_L8N7Z0, D=20), so ~9 subjects/shard is
# ~15 min. The 2 h walltime is slack for GPFS contention across 20 concurrent shards.
#
# NOT self-submitting -- submit with `sbatch sbatch/build_af24_bundles.sh`.
# Dry-run the work list first:  bash sbatch/build_af24_bundles.sh --dry-run
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/tools" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
export SPLIT=${SPLIT:-test}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}

ARMS=${ARMS:-"regular24 hrv24 af24"}
N_SHARDS=${N_SHARDS:-20}
IDX=${SLURM_ARRAY_TASK_ID:-0}

# The work list is derived at runtime from each bundle's own manifest["split"], never from a
# checked-in subject file that could go stale against the cohort on disk. Round-robin over the
# flat (source, subject) list so every shard gets a mix of small- and large-D cohorts.
WORK=$("$PY" - "$IDX" "$N_SHARDS" <<'EOF'
import sys, os
sys.path.insert(0, "evaluation")
import paths
idx, n = int(sys.argv[1]), int(sys.argv[2])
pairs = []
for ds in paths.DATASETS:
    keep, _ = paths.filter_by_split(ds, paths.subjects(ds), os.environ.get("SPLIT", "test"))
    pairs += [(ds, s) for s in keep]
mine = pairs[idx::n]
by_src = {}
for ds, s in mine:
    by_src.setdefault(ds, []).append(s)
for ds, ss in by_src.items():
    print(ds, ",".join(ss))
EOF
) || { echo "failed to build the work list"; exit 2; }

if [ -z "$WORK" ]; then echo "task $IDX: nothing to do"; exit 0; fi
NSUB=$(echo "$WORK" | awk '{n+=split($2,a,",")} END{print n}')
echo "=== task $IDX/$N_SHARDS  arms='$ARMS'  subjects=$NSUB  threads=$OMP_NUM_THREADS ==="
echo "$WORK" | sed 's/^/    /'

if [ "${1:-}" = "--dry-run" ]; then echo "dry-run: nothing built"; exit 0; fi

rc=0
while read -r SRC SUBJ; do
  [ -z "$SRC" ] && continue
  echo "--- $SRC ($(echo "$SUBJ" | tr ',' '\n' | wc -l) subjects)"
  $PY tools/build_af_bundle.py --source "$SRC" --subjects "$SUBJ" \
      --arms $ARMS --eval-root "${EVAL_ROOT:-scratch/eval}" || rc=$?
done <<< "$WORK"

echo "=== task $IDX done (rc=$rc) ==="
exit $rc
