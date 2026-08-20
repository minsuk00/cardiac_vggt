#!/usr/bin/env bash
# Step 2/3 — NeSVoR (INR SVR) per-phase reconstruct for one subject/variant, GPU.
# GPU sibling of run_svrtk3d.sh: SAME env/positional contract, SAME output layout
# (<subject>/<METHOD>/recon_<var>/vol_tNN.nii.gz), SAME idempotency + atomic-move, so
# assemble_and_gif.py / aggregate.py consume it unchanged. Only the recon engine differs
# (CPU mirtk -> GPU Singularity `nesvor reconstruct`).
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
# to [0,1] at load time (per-method rule, assemble_and_gif.py).
#
# Usage: EVAL_DATASET=<src> [MASK_FILE=...] [T=] [THICK=] [J=] \
#          bash evaluation/src/engine/run_nesvor.sh <subject> <clean|breath> [res_mm]
#        EVAL_DATASET is REQUIRED (cmrx2023|cmrx2024|cmrx2025|acdc|mnms|miitt|ocmr).
set -uo pipefail
VGGT=/home/minsukc/vggt
module load singularity 2>/dev/null || true

SUBJ="${1:?subject}"; VAR="${2:?clean|breath}"; RES="${3:-1.4}"
J="${J:-1}"; T="${T:-12}"; THICK="${THICK:-8}"; METHOD="${METHOD:-nesvor}"
# Layout: <subject>/ holds the SHARED frozen bundle (gt/ clean/ breath/ mask_heart.nii.gz manifest.json,
# identical for every method); each method writes under <subject>/<METHOD>/ . See README "Directory layout".
SD="$VGGT/scratch/eval/${EVAL_DATASET:?EVAL_DATASET must name a source dir: cmrx2023|cmrx2024|cmrx2025|acdc|mnms|miitt|ocmr}/out/$SUBJ"
OUT="$SD/$METHOD/recon_$VAR"; mkdir -p "$OUT"

# Stage the 5.3GB .sif to node-local /tmp so container-internal torch/CUDA reads don't hit GPFS
# (mirrors baselines/nesvor/run_nesvor.sh + the project's monai-cache pattern). Durable copy stays on GPFS.
SIF_GPFS="$VGGT/scratch/nesvor/sif/nesvor.sif"
LOCAL_SIF="/tmp/vggt-nesvor_${USER}/nesvor.sif"
mkdir -p "$(dirname "$LOCAL_SIF")"
# ATOMIC + integrity-checked staging: flock serializes concurrent same-node jobs (e.g. a SLURM array
# landing two subjects on one node); stage to a temp path then atomic `mv`; re-stage if the cached size
# != the GPFS master (guards a TRUNCATED copy left by an interrupted prior `cp`). Without this, a
# poisoned node-local .sif silently fails every phase forever until /tmp is cleared.
exec 9>"${LOCAL_SIF}.lock"; flock 9
_want=$(stat -c %s "$SIF_GPFS" 2>/dev/null || echo 0)
if [ ! -f "$LOCAL_SIF" ] || [ "$(stat -c %s "$LOCAL_SIF" 2>/dev/null || echo -1)" != "$_want" ]; then
    _tmp="${LOCAL_SIF}.tmp.$$"
    cp "$SIF_GPFS" "$_tmp" && mv -f "$_tmp" "$LOCAL_SIF" || { rm -f "$_tmp"; echo "FATAL: sif staging failed"; exit 1; }
fi
exec 9>&-

recon_one() {
  set -u   # xargs spawns a fresh shell that does NOT inherit the parent's set -u; re-arm it here so a
           # future unexported var in this body fails loudly instead of expanding to an empty path.
  local p=$1; local pp; pp=$(printf "%02d" "$p")
  local final="$OUT/vol_t${pp}.nii.gz"
  if [ -f "$final" ] && gzip -t "$final" 2>/dev/null; then echo "t$pp cached"; return; fi
  local wd="$OUT/work_t${pp}"; rm -rf "$wd"; mkdir -p "$wd"
  # Per-phase torch/tinycudann scratch on NODE-LOCAL /tmp (singularity auto-mounts /tmp), not the GPFS
  # work dir — consistent with staging the sif to /tmp. Per-phase path so J>1 fits never share it.
  local scr="/tmp/vggt-nesvor_${USER}/scr_${VAR}_t${pp}"; rm -rf "$scr"; mkdir -p "$scr"
  local t0; t0=$(date +%s)
  # Bind the SUBJECT dir to a SIMPLE container path (/data) — NOT the host absolute path, which
  # contains the `scratch`->/gpfs symlink that doesn't resolve inside the container (NeSVoR's
  # makedirs walks the output path up to the missing symlink and dies). Mirrors baselines/nesvor/
  # run_nesvor.sh's /data bind. COUT = the per-method output dir as seen inside the container.
  local COUT="/data/$METHOD/recon_$VAR"
  singularity exec --nv \
      --bind "$SD:/data" \
      --env "TMPDIR=$scr" \
      "$LOCAL_SIF" \
      nesvor reconstruct \
      --input-stacks  "/data/$VAR/stack_t${pp}.nii.gz" \
      --stack-masks   "/data/${MASK_FILE:-mask_heart.nii.gz}" \
      --sample-mask   "/data/${MASK_FILE:-mask_heart.nii.gz}" \
      --thicknesses   "$THICK" \
      --registration  none \
      --output-resolution "$RES" \
      --output-volume "$COUT/work_t${pp}/vol.nii.gz" \
      --output-model  "$COUT/model_t${pp}.pt" \
      --output-slices "$COUT/slices_t${pp}" \
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
export -f recon_one; export OUT SD VAR METHOD MASK_FILE THICK RES LOCAL_SIF

# Provenance (once per subject/variant) — engine, command, params, container identity, hardware
# (GPU + SLURM allocation), parallelism, timing. Mirrors run_svrtk3d.sh's block.
JINFO=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -oE 'cpu=[0-9]+,mem=[0-9]+[MG]' | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
{
  echo "engine          : NeSVoR 'nesvor reconstruct' (INR SVR, single gated stack, per-phase)"
  echo "command         : nesvor reconstruct --input-stacks <VAR/stack_tNN.nii.gz> \\"
  echo "                    --stack-masks ${MASK_FILE:-mask_heart.nii.gz} --sample-mask ${MASK_FILE:-mask_heart.nii.gz} \\"
  echo "                    --thicknesses $THICK --registration none --output-resolution $RES --device 0"
  echo "params          : thickness_mm=$THICK output_resolution_mm=$RES registration=none \\"
  echo "                    n_iter=6000(default) n_samples=256(default) bias=off(default) variance=on(default)"
  echo "container(sif)  : $SIF_GPFS"
  echo "container_id    : $(stat -c '%s bytes, mtime %y' "$SIF_GPFS" 2>/dev/null)"
  echo "method          : $METHOD"
  echo "subject/variant : $SUBJ / $VAR   phases(T)=$T   mask=${MASK_FILE:-mask_heart.nii.gz}"
  echo "--- hardware / parallelism (for the compute-cost comparison) ---"
  echo "host            : $(hostname)   SLURM job ${SLURM_JOB_ID:-none}"
  echo "gpu             : ${GPU:-unknown}   SLURM alloc: ${JINFO:-unknown}"
  echo "parallelism     : J=$J concurrent NeSVoR fits on 1 GPU (each fit is single-GPU)"
} > "$OUT/provenance.txt"

echo "=== $METHOD : $SUBJ / $VAR : thick=${THICK} res=${RES}mm reg=none J=$J ==="
T_ALL0=$(date +%s)
seq 0 $((T-1)) | xargs -P "$J" -I{} bash -c 'recon_one {}'
T_ALL=$(( $(date +%s) - T_ALL0 ))
echo "$T_ALL" > "$OUT/total_wall.sec"
_ndone=$(ls "$OUT"/time_t*.sec 2>/dev/null | wc -l)
_pmean=$(cat "$OUT"/time_t*.sec 2>/dev/null | awk '{s+=$1;n++}END{if(n)printf "%.0f",s/n}')
{ echo "--- timing ---";
  echo "invocation_wall_sec : $T_ALL   (THIS invocation only. Cached phases are skipped instantly, so on a";
  echo "                      resumed/partial run this is NOT the full-subject wall — do not read it as such.)";
  echo "per_phase_sec       : see time_t*.sec (${_ndone} files, mean ${_pmean}s). At J=1 each is the TRUE";
  echo "                      single-GPU per-phase wall (the fair compute-cost unit). At J>1 they are";
  echo "                      contention-inflated (concurrent fits share one GPU). For the headline compute";
  echo "                      cost use J=1 per-phase times from a fresh (non-resumed) run.";
} >> "$OUT/provenance.txt"
# Per-variant stamp (paths.recon_stamp): config identity of the run that wrote recon_<VAR>/.
# Written ONLY when every phase has a valid volume — a partial run stays unstamped, which the
# scorer reads as "cannot verify", never as "verified". No timestamps: two invocations with an
# identical config count as the same run, so clean/breath stamps from separate submissions match.
N_OK=$(ls "$OUT"/vol_t*.nii.gz 2>/dev/null | wc -l)
if [ "$N_OK" -eq "$T" ]; then
  printf '{"engine": "nesvor", "thickness_mm": %s, "output_resolution_mm": %s, "registration": "none", "container_id": "%s"}\n' \
    "$THICK" "$RES" "$(stat -c '%s:%Y' "$SIF_GPFS" 2>/dev/null)" > "$OUT/stamp.json"
else
  echo "NOT stamped: only $N_OK/$T phases OK"
fi
echo "RECON_DONE $SUBJ $VAR  (total ${T_ALL}s this invocation, $N_OK/$T phases ok)"
