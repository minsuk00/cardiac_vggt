#!/usr/bin/env bash
# Smoke-test step 2/3 — SVRTK 3D per-phase reconstruct for one subject/variant.
# Each cardiac phase reconstructed independently (single gated stack) via the mirtk
# container shim. Parallel across phases; idempotent (skips phases already done OK).
#
# IMPORTANT: mirtk reconstruct writes FIXED-NAME intermediates (init.nii.gz, masked.nii.gz,
# tmp-file-exchange/, output-metric-*.txt) into its CWD, so parallel phases MUST each run in
# their OWN working dir or they clobber each other (seen: all died at iteration 0).
#
# Usage: [EVAL_DATASET=cmrxrecon|miitt] [MASK_FILE=...] [T=] [THICK=] bash scratch/eval/engine/run_svrtk3d.sh <subject> <clean|breath> [res_mm] [iters]
set -uo pipefail
VGGT=/home/minsukc/vggt
export PATH="$VGGT/baselines/fetal_cmr_4d/bin:$PATH"
export FCMR_BIND="$VGGT/scratch"      # bind the whole GPFS tree (covers sif + eval/)

# Faithful single-stack SVRTK config (validated by a param sweep on Train_P034, 2026-07-12):
#   -thickness 8    = the REAL slice thickness (8mm); the 12mm canonical Z is the slice PITCH
#                     (8mm thickness + 4mm gap). -thickness sets the PSF width, so 8 is correct.
#   -resolution 1.4 = match the canonical in-plane spacing (native in-plane 1.34-1.58mm).
#   -no_robust_statistics = REQUIRED for a single gated stack. Robust stats estimates slice-
#                     outlier weights from redundancy across OVERLAPPING slices; with one slice
#                     per plane (no overlap) its EM flags every slice as an outlier ("all slices
#                     are outliers") and down-weights the real data -> dim ~14-18dB recons + a
#                     hard crash at fine res. Off => 26.9dB clean (+8.4dB), stable. Applied to
#                     BOTH variants identically (breathing single-stack has no clean slice to
#                     reject against either, so robust stats can't help it regardless).
SUBJ="${1:?subject}"; VAR="${2:?clean|breath}"; RES="${3:-1.4}"; ITERS="${4:-4}"
J="${J:-8}"; T="${T:-12}"; THICK="${THICK:-8}"; METHOD="${METHOD:-svrtk3d}"
# Layout: <subject>/ holds the SHARED frozen bundle (gt/ clean/ breath/ mask_heart.nii.gz manifest.json,
# identical for every method); each method writes under <subject>/<METHOD>/ . See README "Directory layout".
SD="$VGGT/scratch/eval/${EVAL_DATASET:-cmrxrecon}/out/$SUBJ"
OUT="$SD/$METHOD/recon_$VAR"; mkdir -p "$OUT"

recon_one() {
  local p=$1; local pp=$(printf "%02d" "$p")
  local final="$OUT/vol_t${pp}.nii.gz"
  if [ -f "$final" ] && gzip -t "$final" 2>/dev/null; then echo "t$pp cached"; return; fi
  local wd="$OUT/work_t${pp}"; rm -rf "$wd"; mkdir -p "$wd"
  # -debug makes SVRTK dump per-slice rigid slice-to-volume transforms (transformation*.dof:
  # tx,ty,tz,rx,ry,rz) -- SVR's estimated motion correction, incl. through-plane tz. We keep ONLY
  # those (tiny, ~44KB/phase) + the log + wall-clock; the big --debug intermediates (averagebias*/
  # average*/slice* ~200MB/phase) are dropped. This lets us score z (and in-plane!) motion vs the
  # applied breathing later without re-running. See run's README "what is logged".
  local t0=$(date +%s)
  ( cd "$wd" && OMP_NUM_THREADS="$OMP" mirtk reconstruct vol.nii.gz 1 \
      "$SD/$VAR/stack_t${pp}.nii.gz" -thickness "$THICK" -mask "$SD/${MASK_FILE:-mask_heart.nii.gz}" \
      -resolution "$RES" -iterations "$ITERS" -no_robust_statistics $DBG > log.txt 2>&1 )
  local dt=$(( $(date +%s) - t0 ))
  cp -f "$wd/log.txt" "$OUT/log_t${pp}.txt" 2>/dev/null
  echo "$dt" > "$OUT/time_t${pp}.sec"                       # per-phase wall (with -debug: ~6x slower)
  if [ -n "$DBG" ] && ls "$wd"/transformation*.dof >/dev/null 2>&1; then  # keep per-slice motion transforms
    local td="$OUT/transforms_t${pp}"; rm -rf "$td"; mkdir -p "$td"
    cp -f "$wd"/transformation*.dof "$wd"/global-transformation*.dof "$td"/ 2>/dev/null
  fi
  if [ -f "$wd/vol.nii.gz" ] && gzip -t "$wd/vol.nii.gz" 2>/dev/null; then
    mv -f "$wd/vol.nii.gz" "$final"; rm -rf "$wd"; echo "t$pp OK (${dt}s)"
  else
    echo "t$pp FAIL (see $OUT/log_t${pp}.txt)"
  fi
}
# DEBUG=1 (default) → -debug, captures per-slice .dof motion transforms but is ~6x slower in parallel
#   (13s->81s/phase: writing + concurrent 200MB-intermediate I/O). Recon vol is IDENTICAL either way.
# DEBUG=0 → no -debug → the FAIR speed-benchmark timing (use this for the compute-cost headline + the
#   full master run; do a small DEBUG=1 subset only where you need .dof motion analysis).
OMP="${OMP:-2}"; DBG=""; [ "${DEBUG:-1}" = "1" ] && DBG="-debug"
export -f recon_one; export OUT SD VAR THICK RES ITERS OMP DBG
# Provenance (once per subject/variant) — exact engine, command, params, container, AND the hardware
# + timing needed for a fair speed comparison vs our GPU feed-forward model. See README "What is logged".
SIF="${FCMR_SIF:-$VGGT/scratch/fetal_cmr_4d/sif/svrtk.sif}"
# SLURM ALLOCATION (what our job actually got) — NOT the node total. scontrol TRES has both cpu & mem.
JINFO=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -oE 'cpu=[0-9]+,mem=[0-9]+[MG]' | head -1)
NCPU_ALLOC=$(echo "$JINFO" | grep -oE 'cpu=[0-9]+' | cut -d= -f2); NCPU_ALLOC=${NCPU_ALLOC:-$(nproc)}
MEM_ALLOC=$(echo "$JINFO" | grep -oE 'mem=[0-9]+[MG]' | cut -d= -f2); MEM_ALLOC=${MEM_ALLOC:-unknown}
{
  echo "engine          : SVRTK 'mirtk reconstruct' (3D per-phase, single gated stack, K=1)"
  echo "command         : mirtk reconstruct vol.nii.gz 1 <VAR/stack_tNN.nii.gz> -thickness $THICK \\"
  echo "                    -mask ${MASK_FILE:-mask_heart.nii.gz} -resolution $RES -iterations $ITERS \\"
  echo "                    -no_robust_statistics $DBG"
  echo "params          : thickness_mm=$THICK resolution_mm=$RES iterations=$ITERS \\"
  echo "                    robust_statistics=OFF intensity_matching=ON(default) sr_iterations=default(7,last x3)"
  echo "container(sif)  : $SIF"
  echo "container_id    : $(stat -c '%s bytes, mtime %y' "$SIF" 2>/dev/null)"
  echo "method          : $METHOD"
  echo "subject/variant : $SUBJ / $VAR   phases(T)=$T"
  echo "--- hardware / parallelism (for the compute-cost comparison) ---"
  echo "host            : $(hostname)   SLURM job ${SLURM_JOB_ID:-none}"
  echo "cpu model       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"
  echo "SLURM ALLOC     : ${NCPU_ALLOC} CPUs, ${MEM_ALLOC} RAM  (the allocation, NOT the node's 36c/186GB)"
  echo "parallelism     : J=$J phases x OMP=$OMP threads = $((J*OMP)) threads used (of ${NCPU_ALLOC} allocated)"
  echo "debug_mode      : ${DEBUG:-1}  (1=-debug, needed to emit .dof, much slower ~6x in parallel: 13s->81s/phase; 0=fair speed)"
} > "$OUT/provenance.txt"
echo "=== $METHOD : $SUBJ / $VAR : thick=${THICK} res=${RES}mm iters=$ITERS no_robust ${DBG:-nodebug} J=$J ==="
T_ALL0=$(date +%s)
seq 0 $((T-1)) | xargs -P "$J" -I{} bash -c 'recon_one {}'
T_ALL=$(( $(date +%s) - T_ALL0 ))
echo "$T_ALL" > "$OUT/total_wall.sec"                       # end-to-end wall, all phases at J (the recon time)
{ echo "--- timing ---";
  echo "total_wall_sec  : $T_ALL   (end-to-end, all $T phases at J=$J on the above hardware)";
  echo "per_phase_sec   : see time_t*.sec (mean $(cat "$OUT"/time_t*.sec 2>/dev/null | awk '{s+=$1;n++}END{if(n)printf "%.0f",s/n}')s)";
} >> "$OUT/provenance.txt"
echo "RECON_DONE $SUBJ $VAR  (total ${T_ALL}s)"
