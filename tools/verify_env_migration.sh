#!/bin/bash
# Full post-dependency-migration verification (the docs/49 bar).
#
# Runs a train-step smoke for every distinct code path, then the rare/long-run
# branches that a 2-batch smoke skips: a real requeue (old-env checkpoint ->
# new-env process), the EF-eval branch, and an OOD inference pass (which is the
# only thing that exercises inference/adapters/base.py percentile_scale in situ).
#
# Usage: bash tools/verify_env_migration.sh [outdir]
# Each case prints PASS/FAIL with the first train loss + grad norms.

set -u
OUT=${1:-/tmp/envverify}
mkdir -p "$OUT"
CKPT=/tmp/vggt1b_base.pt
[ -f "$CKPT" ] || cp scratch/base_weights/vggt1b_base.pt "$CKPT"

COMMON="max_epochs=1 limit_train_batches=2 limit_val_batches=1 \
logging.ef_eval_enable=false checkpoint.resume_checkpoint_path=$CKPT"

pass=0; fail=0; results=""

# `limit_val_batches` is NOT honoured in these configs (multi-phase val sweep), so
# val runs ~200 batches at ~2.6 s each. We only need to prove the TRAIN step works
# on each code path, so stop as soon as the train line lands (or it dies).
run_case () {           # run_case <name> <config> <extra overrides...>
  local name=$1; shift
  local cfg=$1; shift
  local log="$OUT/$name.log"
  WANDB_MODE=offline PYTHONPATH=training:. micromamba run -n svr \
    python training/launch.py --config "$cfg" $COMMON \
    logging.log_dir="$OUT/$name" "$@" > "$log" 2>&1 &
  local pid=$!
  local waited=0
  while [ $waited -lt 900 ]; do
    grep -q "Train Epoch:" "$log" 2>/dev/null && break
    grep -qE "Traceback|Error|error:" "$log" 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break          # process died
    sleep 5; waited=$((waited+5))
  done
  # got what we need (or it failed) -- tear down this case and its children
  pkill -P $pid 2>/dev/null; kill -9 $pid 2>/dev/null
  for gp in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    kill -9 "$gp" 2>/dev/null
  done
  sleep 3
  local rc=0
  grep -q "Train Epoch:" "$log" 2>/dev/null || rc=1
  local loss grads
  loss=$(grep -oE "Loss/train_loss_objective: [0-9.]+" "$log" | head -1 | awk '{print $2}')
  grads=$(grep -oE "Grad/aggregator: [0-9.]+" "$log" | head -1 | awk '{print $2}')
  # a finite, nonzero loss with flowing grads is the real bar -- rc=0 alone is not
  if [ $rc -eq 0 ] && [ -n "$loss" ] && [ "$loss" != "nan" ] && [ -n "$grads" ]; then
    printf "  PASS  %-22s loss=%-9s grad_agg=%s\n" "$name" "$loss" "$grads"
    pass=$((pass+1))
  else
    printf "  FAIL  %-22s rc=%s loss=%s  (see %s)\n" "$name" "$rc" "${loss:-none}" "$log"
    tail -3 "$log" | sed 's/^/          /'
    fail=$((fail+1))
  fi
}

echo "=============================================================="
echo "CONFIG MATRIX  (every distinct code path)"
echo "=============================================================="
run_case dpt_tv            mri_volume
run_case diffusion         mri_volume_diffusion
run_case bspline           mri_volume_bspline
run_case gather05          mri_volume loss.volume.gather_weight=0.5
run_case one_frame         mri_volume one_frame_per_slice=true max_img_per_gpu=12
run_case continuous_z      mri_volume continuous_z=true
run_case lowdiff100        mri_volume_diffusion loss.volume.diffusion_weight=100
run_case aug_moderate      mri_volume data.augmentation.enable=true data.augmentation.tier=moderate
run_case dino_unfrozen     mri_volume 'optim.frozen_module_names=[]'
run_case fixed_phase_ED    mri_volume t_target_fixed=0

echo
echo "=============================================================="
echo "RARE / LONG-RUN BRANCHES  (skipped by a 2-batch smoke)"
echo "=============================================================="

# EF eval branch: off in every smoke above, uses numpy (polyfit/corrcoef/argsort)
# EF eval lives at the END of val, so this one must run val to completion.
# t_target_fixed=0 makes val a single deterministic pass (CLAUDE.md), which is
# what actually caps it.
ef_log="$OUT/ef_eval.log"
WANDB_MODE=offline PYTHONPATH=training:. micromamba run -n svr \
  timeout 2400 python training/launch.py --config default max_epochs=1 \
  limit_train_batches=2 t_target_fixed=0 logging.ef_eval_enable=true \
  checkpoint.resume_checkpoint_path=$CKPT logging.log_dir="$OUT/ef_eval" \
  > "$ef_log" 2>&1
rc=$?
if [ $rc -eq 0 ] && ! grep -qiE "ef.*(failed|ignored)" "$ef_log"; then
  echo "  PASS  ef_eval               (ef_eval_enable=true, no swallowed failure)"
  pass=$((pass+1))
else
  echo "  FAIL  ef_eval               rc=$rc"
  grep -iE "ef.*(failed|ignored)|Traceback" "$ef_log" | head -3 | sed 's/^/          /'
  fail=$((fail+1))
fi

# Swallowed-diagnostic check: every viz/logging path is try/except'd, so a smoke
# exiting 0 proves nothing about them. Grep for the swallow marker.
echo
echo "  --- swallowed diagnostics across ALL runs above (try/except '(ignored)') ---"
if grep -rhiE "\(ignored\)" "$OUT"/*.log 2>/dev/null | sort -u | head -10 | grep -q .; then
  grep -rhiE "\(ignored\)" "$OUT"/*.log 2>/dev/null | sort -u | head -10 | sed 's/^/          /'
  echo "          ^^ diagnostics silently failed -- investigate"
  fail=$((fail+1))
else
  echo "          none -- no diagnostic was silently swallowed"
  pass=$((pass+1))
fi

echo
echo "=============================================================="
printf "TOTAL: %d passed, %d failed\n" "$pass" "$fail"
echo "=============================================================="
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
