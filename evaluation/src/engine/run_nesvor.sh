#!/usr/bin/env bash
# Step 2/3 — NeSVoR (INR SVR) per-phase reconstruct for one subject/variant, GPU.
# GPU sibling of run_svrtk3d.sh: SAME env/positional contract, SAME output layout
# (<subject>/<METHOD>/recon_<var>/vol_tNN.nii.gz), SAME idempotency + atomic-move, so
# score/image_metrics.py / score/aggregate.py consume it unchanged. Only the recon engine differs
# (CPU mirtk -> GPU `nesvor reconstruct` from the NATIVE `nesvor-t2` env, docs/90: upstream v0.5.0 +
# a 10-line torch-2 CUDA port, compiled for sm_86/sm_89 — same code the retired .sif container ran,
# but 2.8x faster on the same GPU because the container's CUDA was built for V100 only).
#
# Each cardiac phase = one independent NeSVoR fit on the single gated stack. NeSVoR needs a
# CUDA device, so phases share ONE GPU: J = concurrent fits on that GPU (default 1; a smoke-
# test concurrency sweep sets the production value). Each phase gets its OWN work dir + TMPDIR
# so concurrent fits never share scratch.
#
# Decisions baked in (see baselines/nesvor/readme.md):
#   --registration none : single stack -> stack-to-stack is a no-op (nominal geometry either
#                         way); pose is refined by NeSVoR's in-training gradient descent regardless.
#   --thicknesses 8     : real slice thickness = PSF through-plane FWHM (NOT the 12mm pitch).
#   --sample-mask/--stack-masks = the heart ROI. --sample-mask is MANDATORY on our single
#                         anisotropic stack (else NeSVoR's internal mask estimate comes out
#                         all-False -> IndexError crash after ~7min). doc 32.
#   --output-resolution 1.4 : match SVRTK; both resampled to canonical to score.
#   everything else NeSVoR default (--n-iter 6000, --n-samples 256, bias off, variances on).
# Output is stored RAW (NeSVoR's ~700-mean scale); the scorer self-percentile-normalizes it
# to [0,1] at load time (per-method rule, score/image_metrics.py).
#
# Usage: EVAL_DATASET=<src> [MASK_FILE=...] [T=] [THICK=] [J=] \
#          bash evaluation/src/engine/run_nesvor.sh <subject> <clean|breath> [res_mm]
#        EVAL_DATASET is REQUIRED (cmrx2023|cmrx2024|cmrx2025|acdc|mnms|miitt|ocmr).
set -uo pipefail
VGGT="${VGGT:-$(cd "$(dirname "$0")/../../.." && pwd)}"   # this checkout (worktree-safe); override to redirect SD
NESVOR_BIN="${NESVOR_BIN:-/home/minsukc/micromamba/envs/nesvor-t2/bin/nesvor}"   # native env (docs/90)
NESVOR_SRC="$VGGT/baselines/nesvor/NeSVoR"                                          # editable install it runs

SUBJ="${1:?subject}"; VAR="${2:?clean|breath}"; RES="${3:-1.4}"
J="${J:-1}"; T="${T:-12}"; THICK="${THICK:-8}"
# INPUT = which bundle stack feeds the recon (see run_svrtk3d.sh): default = the variant's own
# gated stack; INPUT=scatter = the SAME-INPUT arm (scatter/stack_tNN, what VGGT sees), landing
# under <METHOD>_scatter with the usual recon_<VAR>/ layout.
INPUT="${INPUT:-$VAR}"
if [ "$INPUT" = scatter ]; then METHOD="${METHOD:-nesvor_scatter}"; else METHOD="${METHOD:-nesvor}"; fi
# Layout: <subject>/ holds the SHARED frozen bundle (gt/ clean/ breath/ mask_heart.nii.gz manifest.json,
# identical for every method); each method writes under <subject>/<METHOD>/ . See README "Directory layout".
SD="$VGGT/scratch/eval/${EVAL_DATASET:?EVAL_DATASET must name a source dir: cmrx2023|cmrx2024|cmrx2025|acdc|mnms|miitt|ocmr}/out/$SUBJ"
OUT="$SD/$METHOD/recon_$VAR"; mkdir -p "$OUT"

[ -x "$NESVOR_BIN" ] || { echo "FATAL: $NESVOR_BIN missing — build the nesvor-t2 env (docs/90)"; exit 1; }
# Engine identity for the stamp: the source commit + a content hash of the working-tree diff
# (the torch-2 port), so two invocations of the same code agree and any edit to the source changes it.
engine_id() {
  local head diff
  head=$(git -C "$NESVOR_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
  diff=$(git -C "$NESVOR_SRC" diff 2>/dev/null | sha256sum | cut -c1-12)
  echo "native:nesvor-t2:${head}+${diff}"
}
mkdir -p "/tmp/vggt-nesvor_${USER}"

recon_one() {
  set -u   # xargs spawns a fresh shell that does NOT inherit the parent's set -u; re-arm it here so a
           # future unexported var in this body fails loudly instead of expanding to an empty path.
  local p=$1; local gpu=${2:-}; local pp; pp=$(printf "%02d" "$p")
  local final="$OUT/vol_t${pp}.nii.gz"
  if [ -f "$final" ] && gzip -t "$final" 2>/dev/null; then echo "t$pp cached"; return; fi
  local wd="$OUT/work_t${pp}"; rm -rf "$wd"; mkdir -p "$wd"
  # Per-phase torch/tinycudann scratch on NODE-LOCAL /tmp, not the GPFS work dir. Per-phase path
  # so J>1 fits never share it.
  local scr="/tmp/vggt-nesvor_${USER}/scr_${VAR}_t${pp}"; rm -rf "$scr"; mkdir -p "$scr"
  local t0; t0=$(date +%s)
  # GPUS mode pins this fit to one physical GPU (it then sees it as device 0); else inherit.
  env ${gpu:+CUDA_VISIBLE_DEVICES=$gpu} TMPDIR="$scr" "$NESVOR_BIN" reconstruct \
      --input-stacks  "$SD/$INPUT/stack_t${pp}.nii.gz" \
      --stack-masks   "$SD/${MASK_FILE:-mask_heart.nii.gz}" \
      --sample-mask   "$SD/${MASK_FILE:-mask_heart.nii.gz}" \
      --thicknesses   "$THICK" \
      --registration  none \
      --output-resolution "$RES" \
      --output-volume "$wd/vol.nii.gz" \
      --output-model  "$OUT/model_t${pp}.pt" \
      --output-slices "$OUT/slices_t${pp}" \
      --device 0 > "$wd/log.txt" 2>&1
      # --output-slices persists the per-slice motion-corrected NIfTIs; EACH slice's affine encodes its
      # estimated 6DOF pose (nesvor transformation2affine) — this is how we record NeSVoR's registration
      # (the analog of SVRTK's .dof) for a later motion metric. VERIFIED: model_t*.pt holds ONLY the INR
      # weights + the output-volume frame, NOT the per-slice poses. save_slices runs AFTER the volume in
      # NeSVoR's outputs(), so even if it errored the volume is already written and the gate still promotes it.
      # --output-json is NOT used: it dumps only the arg Namespace (config, already in provenance +
      # model.pt args), NOT transforms, and crashes on a numpy dtype in args in v0.5.0.
  local dt; dt=$(( $(date +%s) - t0 ))
  rm -rf "$scr"   # drop the node-local scratch (per-phase, no longer needed)
  cp -f "$wd/log.txt" "$OUT/log_t${pp}.txt" 2>/dev/null
  echo "$dt" > "$OUT/time_t${pp}.sec"
  if [ -f "$wd/vol.nii.gz" ] && gzip -t "$wd/vol.nii.gz" 2>/dev/null; then
    mv -f "$wd/vol.nii.gz" "$final"; rm -rf "$wd"; echo "t$pp OK (${dt}s)"
  else
    echo "t$pp FAIL (see $OUT/log_t${pp}.txt)"
  fi
}
export -f recon_one; export OUT SD VAR INPUT METHOD MASK_FILE THICK RES NESVOR_BIN

# GPUS=0,1,2,3 (ids relative to CUDA_VISIBLE_DEVICES, at most 4; T=12 -> 3 fits per GPU): shard
# the phases round-robin, phase p -> GPU p mod N, ONE worker per GPU running its phases one after
# another (exactly one fit per GPU at a time; J is ignored). Unset = the original J path on one GPU.
PHYS=()
if [ -n "${GPUS:-}" ]; then
  IFS=, read -ra _g <<< "$GPUS"
  IFS=, read -ra _vis <<< "${CUDA_VISIBLE_DEVICES:-}"
  for g in "${_g[@]}"; do
    [[ "$g" =~ ^[0-9]+$ ]] || { echo "FATAL: bad GPU id '$g'"; exit 1; }
    g=$((10#$g))                                     # "00"/"08" -> 0/8
    if [ ${#_vis[@]} -gt 0 ]; then
      [ "$g" -lt ${#_vis[@]} ] || { echo "FATAL: GPU id $g not in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; exit 1; }
      PHYS+=("${_vis[$g]}")
    else
      PHYS+=("$g")
    fi
  done
  if [ ${#PHYS[@]} -lt 1 ] || [ ${#PHYS[@]} -gt 4 ] || [ "$(printf '%s\n' "${PHYS[@]}" | sort -u | wc -l)" -ne ${#PHYS[@]} ]; then
    echo "FATAL: GPUS needs 1-4 distinct GPUs, got '$GPUS'"; exit 1
  fi
fi
NGPU=$(( ${#PHYS[@]} > 0 ? ${#PHYS[@]} : 1 ))

# Already complete (stamped + all T volumes valid) -> exit before rewriting provenance/total_wall.sec,
# which would otherwise record ~0 s and this invocation's GPU setup for volumes an earlier run made.
if [ -f "$OUT/stamp.json" ]; then
  _ok=0; for f in "$OUT"/vol_t*.nii.gz; do [ -f "$f" ] && gzip -t "$f" 2>/dev/null && _ok=$((_ok + 1)); done
  [ "$_ok" -eq "$T" ] && { echo "RECON_DONE $SUBJ $VAR  (already complete + stamped; nothing rewritten)"; exit 0; }
fi

# Provenance (once per subject/variant) — engine, command, params, container identity, hardware
# (GPU + SLURM allocation), parallelism, timing. Mirrors run_svrtk3d.sh's block.
JINFO=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -oE 'cpu=[0-9]+,mem=[0-9]+[MG]' | head -1)
if [ ${#PHYS[@]} -gt 0 ]; then _q=(-i "$(IFS=,; echo "${PHYS[*]}")"); else _q=(-i "${CUDA_VISIBLE_DEVICES:-0}"); _q[1]=${_q[1]%%,*}; fi
GPU=$(nvidia-smi "${_q[@]}" --query-gpu=index,name --format=csv,noheader 2>/dev/null | paste -sd';')
{
  echo "engine          : NeSVoR 'nesvor reconstruct' (INR SVR, single gated stack, per-phase)"
  echo "command         : nesvor reconstruct --input-stacks <$INPUT/stack_tNN.nii.gz> \\"
  echo "                    --stack-masks ${MASK_FILE:-mask_heart.nii.gz} --sample-mask ${MASK_FILE:-mask_heart.nii.gz} \\"
  echo "                    --thicknesses $THICK --registration none --output-resolution $RES --device 0"
  echo "params          : thickness_mm=$THICK output_resolution_mm=$RES registration=none \\"
  echo "                    n_iter=6000(default) n_samples=256(default) bias=off(default) variance=on(default)"
  echo "engine build    : native env $NESVOR_BIN  (source $NESVOR_SRC, docs/90)"
  echo "engine_id       : $(engine_id)"
  echo "method          : $METHOD"
  echo "subject/variant : $SUBJ / $VAR   phases(T)=$T   mask=${MASK_FILE:-mask_heart.nii.gz}   input_stack=$INPUT"
  echo "--- hardware / parallelism (for the compute-cost comparison) ---"
  echo "host            : $(hostname)   SLURM job ${SLURM_JOB_ID:-none}"
  echo "gpu             : ${GPU:-unknown}   SLURM alloc: ${JINFO:-unknown}"
  if [ ${#PHYS[@]} -gt 0 ]; then
    echo "parallelism     : GPUS=$GPUS -> physical GPUs ${PHYS[*]}: phase p on GPU p mod ${#PHYS[@]}, one fit per GPU at a time (each fit is single-GPU)"
  else
    echo "parallelism     : J=$J concurrent NeSVoR fits on 1 GPU (each fit is single-GPU)"
  fi
} > "$OUT/provenance.txt"

echo "=== $METHOD : $SUBJ / $VAR : thick=${THICK} res=${RES}mm reg=none $([ ${#PHYS[@]} -gt 0 ] && echo "GPUS=$GPUS" || echo "J=$J") ==="
T_ALL0=$(date +%s)
if [ ${#PHYS[@]} -gt 0 ]; then
  for i in "${!PHYS[@]}"; do
    ( for ((p = i; p < T; p += ${#PHYS[@]})); do recon_one "$p" "${PHYS[$i]}"; done ) &
  done
  wait
else
  seq 0 $((T-1)) | xargs -P "$J" -I{} bash -c 'recon_one {}'
fi
T_ALL=$(( $(date +%s) - T_ALL0 ))
echo "$T_ALL" > "$OUT/total_wall.sec"
_ndone=$(ls "$OUT"/time_t*.sec 2>/dev/null | wc -l)
_pmean=$(cat "$OUT"/time_t*.sec 2>/dev/null | awk '{s+=$1;n++}END{if(n)printf "%.0f",s/n}')
{ echo "--- timing ---";
  echo "invocation_wall_sec : $T_ALL   (THIS invocation only. Cached phases are skipped instantly, so on a";
  echo "                      resumed/partial run this is NOT the full-subject wall — do not read it as such.)";
  if [ ${#PHYS[@]} -gt 0 ]; then
  echo "                      With GPUS ($NGPU GPUs) this is the per-subject wall with phases in parallel.";
  echo "per_phase_sec       : see time_t*.sec (${_ndone} files, mean ${_pmean}s). One fit per GPU at a time, so";
  echo "                      each is a single-GPU per-phase wall (host CPU/IO shared by $NGPU concurrent fits).";
  else
  echo "per_phase_sec       : see time_t*.sec (${_ndone} files, mean ${_pmean}s). At J=1 each is the TRUE";
  echo "                      single-GPU per-phase wall (the fair compute-cost unit). At J>1 they are";
  echo "                      contention-inflated (concurrent fits share one GPU). For the headline compute";
  echo "                      cost use J=1 per-phase times from a fresh (non-resumed) run.";
  fi
} >> "$OUT/provenance.txt"
# Per-variant stamp (paths.recon_stamp): config identity of the run that wrote recon_<VAR>/.
# Written ONLY when every phase has a valid volume — a partial run stays unstamped, which the
# scorer reads as "cannot verify", never as "verified". No timestamps: two invocations with an
# identical config count as the same run, so clean/breath stamps from separate submissions match.
N_OK=$(ls "$OUT"/vol_t*.nii.gz 2>/dev/null | wc -l)
if [ "$N_OK" -eq "$T" ]; then
  printf '{"engine": "nesvor", "input_stack": "%s", "thickness_mm": %s, "output_resolution_mm": %s, "registration": "none", "container_id": "%s"}\n' \
    "$([ "$INPUT" = scatter ] && echo scatter || echo gated)" "$THICK" "$RES" "$(engine_id)" > "$OUT/stamp.json"
else
  echo "NOT stamped: only $N_OK/$T phases OK"
fi
echo "RECON_DONE $SUBJ $VAR  (total ${T_ALL}s this invocation, $N_OK/$T phases ok)"
