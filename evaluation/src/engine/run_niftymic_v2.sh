#!/usr/bin/env bash
# Step 2/3 — NiftyMIC (classical SVR + Tikhonov SRR) per-phase reconstruct for one subject/variant, CPU.
# CPU sibling of run_svrtk3d.sh: SAME env/positional contract, SAME output layout
# (<subject>/<METHOD>/recon_<var>/vol_tNN.nii.gz), SAME idempotency + atomic-move + provenance +
# stamp, SAME CPU setup (J=8 phases in parallel x OMP=2 threads on a 16-CPU allocation), so
# run_baselines.py / score/*.py consume it unchanged. Only the engine differs: `mirtk reconstruct`
# -> `niftymic_reconstruct_volume` inside the renbem/niftymic:latest (v0.9) Singularity image.
#
# Pilot + engine bugs found so far (docs/29, docs/102, baselines/niftymic/ is the n=2 ad-hoc pilot):
#   1. bind the subject dir to a SHORT container path (/data) — a deep same-path bind silently fails.
#   2. the image's joint_image_mask_builder.py crashes on SimpleITK 2.x (positional Execute()); a
#      patched copy is bind-mounted OVER the original at runtime (PATCH below).
#   3. scattered_data_approximation.py's Shepard initialiser does `slice_sitk += 1` as a bookkeeping
#      trick, meaning to operate on a COPY of the slice — but under SimpleITK 2.x, `+=` mutates in
#      place and `_get_image_slice`/`_get_mask_slice` return the slice's OWN image (not a copy), so
#      every input slice gets +1 permanently before the solver ever sees it. Real measured effect:
#      output ≈ input + 0.85 (not the clean "+1.0 pedestal" docs/29 assumed — that read a REAL bug as
#      an inherent SRR-solver property). Fixed the same way as bug 2: a patched copy
#      (`_get_image_slice`/`_get_mask_slice` wrap the return in `sitk.Image(...)`, forcing a real
#      copy) bind-mounted over the original (PATCH2 below). The self-norm gauge
#      (score/image_metrics.py SELF_NORM_METHODS) still applies on top — it corrects the reconstructor's
#      OWN intensity scale/gain, a separate, expected effect from this bug.
#
# Engine flags (from `niftymic_reconstruct_volume --help` in the container, v0.9):
#   --slice-thicknesses $THICK : real slice thickness = PSF through-plane width (NOT the 12mm
#                                pitch the header carries; same THICK rule as SVRTK/NeSVoR).
#   --isotropic-resolution $RES: 1.4mm iso output, matching SVRTK -resolution / NeSVoR
#                                --output-resolution (the pilot's default gave 1.0mm: ~2.7x voxels).
#   --iter-max $ITERMAX        : LSMR cap inside the TK1L2 solve (engine default 10 — kept; the
#                                pilot hit the cap, see docs/29 open questions).
#   everything else engine default (TK1L2, alpha 0.015, two-step-cycles 3, outlier-rejection ON,
#   intensity-correction ON = no-op for one stack, extra-frame-target 10mm).
# Single stack => volume-to-volume registration is trivial; slice-to-volume + SRR still run.
#
# Usage: EVAL_DATASET=<src> [MASK_FILE=...] [T=] [THICK=] [J=] [OMP=] \
#          bash evaluation/src/engine/run_niftymic_v2.sh <subject> <clean|breath> [res_mm] [iter_max]
#        EVAL_DATASET is REQUIRED (cmrx2023|cmrx2024|cmrx2025|acdc|mnms|miitt|ocmr).
set -uo pipefail
VGGT=/home/minsukc/vggt
SIF="${NIFTYMIC_SIF:-$VGGT/scratch/niftymic/sif/niftymic.sif}"
PATCH="${NIFTYMIC_PATCH:-$VGGT/scratch/niftymic/patch/joint_image_mask_builder.py}"
PATCH_DST=/app/NiftyMIC/niftymic/utilities/joint_image_mask_builder.py
PATCH2="${NIFTYMIC_PATCH2:-$VGGT/scratch/niftymic/patch/scattered_data_approximation.py}"
PATCH2_DST=/app/NiftyMIC/niftymic/reconstruction/scattered_data_approximation.py
[ -f "$SIF" ]   || { echo "FATAL: $SIF missing (docs/29: singularity pull docker://renbem/niftymic:latest)"; exit 1; }
[ -f "$PATCH" ]  || { echo "FATAL: $PATCH missing (docs/29 bug 2: setter-based BinaryThreshold patch)"; exit 1; }
[ -f "$PATCH2" ] || { echo "FATAL: $PATCH2 missing (docs/102 bug 3: SDA in-place-mutation +1 offset patch)"; exit 1; }
# Real singularity binary: /usr/local/bin/singularity on this cluster is a stub that only prints
# "use module load" (same resolution as baselines/fetal_cmr_4d/bin/mirtk).
SING="${NIFTYMIC_SINGULARITY:-}"
if [ -z "$SING" ]; then for p in /opt/singularity/*/bin/singularity; do [ -x "$p" ] && SING="$p"; done; fi
[ -n "$SING" ] || { echo "FATAL: no /opt/singularity/*/bin/singularity"; exit 1; }

SUBJ="${1:?subject}"; VAR="${2:?clean|breath}"; RES="${3:-1.4}"; ITERMAX="${4:-10}"
J="${J:-8}"; T="${T:-12}"; THICK="${THICK:-8}"; OMP="${OMP:-2}"
# INPUT = which bundle stack feeds the recon (see run_svrtk3d.sh): default = the variant's own
# gated stack; INPUT=scatter = the SAME-INPUT arm (scatter/stack_tNN, what VGGT sees), landing
# under <METHOD>_scatter with the usual recon_<VAR>/ layout.
INPUT="${INPUT:-$VAR}"
if [ "$INPUT" = scatter ]; then METHOD="${METHOD:-niftymic_scatter}"; else METHOD="${METHOD:-niftymic}"; fi
SD="$VGGT/scratch/eval/${EVAL_DATASET:?EVAL_DATASET must name a source dir: cmrx2023|cmrx2024|cmrx2025|acdc|mnms|miitt|ocmr}/out/$SUBJ"
SD_REAL="$(realpath "$SD")"          # scratch/ is a symlink; bind the real GPFS path
OUT="$SD/$METHOD/recon_$VAR"; mkdir -p "$OUT"
# Stage the .sif on node-local /tmp once per node (12 container starts per subject; GPFS is slow
# for repeated reads of a 2.85 GB file). Same pattern as the mirtk shim / run_nesvor.sh.
SIF_LOCAL="/tmp/vggt-niftymic_${USER}/niftymic.sif"
if [ ! -f "$SIF_LOCAL" ] || [ "$(stat -c%s "$SIF_LOCAL" 2>/dev/null || echo 0)" != "$(stat -c%s "$SIF")" ]; then
  mkdir -p "$(dirname "$SIF_LOCAL")"
  cp -f "$SIF" "$SIF_LOCAL.tmp.$$" && mv -f "$SIF_LOCAL.tmp.$$" "$SIF_LOCAL"
fi

recon_one() {
  set -u
  local p=$1; local pp; pp=$(printf "%02d" "$p")
  local final="$OUT/vol_t${pp}.nii.gz"
  if [ -f "$final" ] && gzip -t "$final" 2>/dev/null; then echo "t$pp cached"; return; fi
  local wd="$OUT/work_t${pp}"; rm -rf "$wd"; mkdir -p "$wd"
  local rel="$METHOD/recon_$VAR/work_t${pp}"       # work dir as seen from /data inside the container
  local t0; t0=$(date +%s)
  # Thread caps: ITK (registration/resampling) + OpenMP/BLAS (scipy LSMR) — J phases x OMP threads.
  SINGULARITYENV_ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$OMP" SINGULARITYENV_OMP_NUM_THREADS="$OMP" \
  SINGULARITYENV_OPENBLAS_NUM_THREADS="$OMP" SINGULARITYENV_MKL_NUM_THREADS="$OMP" \
  "$SING" exec --bind "$SD_REAL:/data" --bind "$PATCH:$PATCH_DST" --bind "$PATCH2:$PATCH2_DST" "$SIF_LOCAL" \
    niftymic_reconstruct_volume \
      --filenames "/data/$INPUT/stack_t${pp}.nii.gz" \
      --filenames-masks "/data/${MASK_FILE:-mask_heart.nii.gz}" \
      --output "/data/$rel/vol.nii.gz" \
      --slice-thicknesses "$THICK" \
      --isotropic-resolution "$RES" \
      --iter-max "$ITERMAX" \
      > "$wd/log.txt" 2>&1
  local dt; dt=$(( $(date +%s) - t0 ))
  cp -f "$wd/log.txt" "$OUT/log_t${pp}.txt" 2>/dev/null
  echo "$dt" > "$OUT/time_t${pp}.sec"
  # Keep NiftyMIC's per-slice rigid transforms + rejected_slices.json (~7 KB, the analog of SVRTK's
  # .dof / NeSVoR's slices_t*) for a later motion metric; drop the rest of the work dir.
  if [ -d "$wd/motion_correction" ]; then
    local td="$OUT/transforms_t${pp}"; rm -rf "$td"; mkdir -p "$td"
    cp -f "$wd"/motion_correction/*.tfm "$wd"/motion_correction/*.json "$td"/ 2>/dev/null
  fi
  if [ -f "$wd/vol.nii.gz" ] && gzip -t "$wd/vol.nii.gz" 2>/dev/null; then
    mv -f "$wd/vol.nii.gz" "$final"; rm -rf "$wd"; echo "t$pp OK (${dt}s)"
  else
    echo "t$pp FAIL (see $OUT/log_t${pp}.txt)"
  fi
}
export -f recon_one; export OUT SD_REAL VAR INPUT METHOD MASK_FILE THICK RES ITERMAX OMP SING SIF_LOCAL PATCH PATCH_DST PATCH2 PATCH2_DST

# Content id of the container — same "v2:<size>:<sha256 of first+last 64 MiB>[:16]" as run_svrtk3d.sh.
container_id() {
  local f=$1 size edge=$((64 << 20))
  size=$(stat -c %s "$f" 2>/dev/null) || return 0
  local sha
  sha=$( { head -c "$edge" "$f"
           if [ "$size" -gt "$edge" ]; then
             local off=$(( size - edge > edge ? size - edge : edge ))
             tail -c +$((off + 1)) "$f" | head -c "$edge"
           fi; } | sha256sum | cut -c1-16 )
  echo "v2:$size:$sha"
}
PATCH_ID=$(sha256sum "$PATCH" | cut -c1-12)
PATCH2_ID=$(sha256sum "$PATCH2" | cut -c1-12)
JINFO=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -oE 'cpu=[0-9]+,mem=[0-9]+[MG]' | head -1)
NCPU_ALLOC=$(echo "$JINFO" | grep -oE 'cpu=[0-9]+' | cut -d= -f2); NCPU_ALLOC=${NCPU_ALLOC:-$(nproc)}
MEM_ALLOC=$(echo "$JINFO" | grep -oE 'mem=[0-9]+[MG]' | cut -d= -f2); MEM_ALLOC=${MEM_ALLOC:-unknown}
{
  echo "engine          : NiftyMIC v0.9 'niftymic_reconstruct_volume' (rigid SVR + TK1L2 SRR, single gated stack, per-phase)"
  echo "command         : niftymic_reconstruct_volume --filenames <$INPUT/stack_tNN.nii.gz> \\"
  echo "                    --filenames-masks ${MASK_FILE:-mask_heart.nii.gz} --slice-thicknesses $THICK \\"
  echo "                    --isotropic-resolution $RES --iter-max $ITERMAX"
  echo "params          : thickness_mm=$THICK isotropic_resolution_mm=$RES iter_max=$ITERMAX \\"
  echo "                    reconstruction_type=TK1L2(default) alpha=0.015(default) two_step_cycles=3(default) \\"
  echo "                    outlier_rejection=ON(default) intensity_correction=ON(default, no-op for 1 stack)"
  echo "container(sif)  : $SIF"
  echo "container_id    : $(container_id "$SIF")"
  echo "patches         : $PATCH ($PATCH_ID)"
  echo "                  $PATCH2 ($PATCH2_ID)  [docs/102 +1 intensity-offset fix]"
  echo "method          : $METHOD"
  echo "subject/variant : $SUBJ / $VAR   phases(T)=$T   input_stack=$INPUT"
  echo "--- hardware / parallelism (for the compute-cost comparison) ---"
  echo "host            : $(hostname)   SLURM job ${SLURM_JOB_ID:-none}"
  echo "cpu model       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"
  echo "SLURM ALLOC     : ${NCPU_ALLOC} CPUs, ${MEM_ALLOC} RAM  (the allocation, NOT the node total)"
  echo "parallelism     : J=$J phases x OMP=$OMP threads = $((J*OMP)) threads used (of ${NCPU_ALLOC} allocated)"
} > "$OUT/provenance.txt"
echo "=== $METHOD : $SUBJ / $VAR : thick=${THICK} res=${RES}mm iter_max=$ITERMAX J=$J OMP=$OMP ==="
T_ALL0=$(date +%s)
seq 0 $((T-1)) | xargs -P "$J" -I{} bash -c 'recon_one {}'
T_ALL=$(( $(date +%s) - T_ALL0 ))
echo "$T_ALL" > "$OUT/total_wall.sec"
{ echo "--- timing ---";
  echo "total_wall_sec  : $T_ALL   (end-to-end, all $T phases at J=$J on the above hardware; THIS invocation only — cached phases skip)";
  echo "per_phase_sec   : see time_t*.sec (mean $(cat "$OUT"/time_t*.sec 2>/dev/null | awk '{s+=$1;n++}END{if(n)printf "%.0f",s/n}')s)";
} >> "$OUT/provenance.txt"
# Per-variant stamp (paths.recon_stamp): written ONLY when every phase has a valid volume — a partial
# run stays unstamped, which the scorer reads as "cannot verify", never as "verified". No timestamps.
N_OK=$(ls "$OUT"/vol_t*.nii.gz 2>/dev/null | wc -l)
if [ "$N_OK" -eq "$T" ]; then
  printf '{"engine": "niftymic", "input_stack": "%s", "thickness_mm": %s, "resolution_mm": %s, "iter_max": %s, "reconstruction_type": "TK1L2", "container_id": "%s", "patch_id": "%s", "patch2_id": "%s"}\n' \
    "$([ "$INPUT" = scatter ] && echo scatter || echo gated)" "$THICK" "$RES" "$ITERMAX" "$(container_id "$SIF")" "$PATCH_ID" "$PATCH2_ID" > "$OUT/stamp.json"
else
  echo "NOT stamped: only $N_OK/$T phases OK"
fi
echo "RECON_DONE $SUBJ $VAR  (total ${T_ALL}s, $N_OK/$T phases ok)"
