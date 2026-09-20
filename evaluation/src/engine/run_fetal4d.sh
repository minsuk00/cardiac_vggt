#!/usr/bin/env bash
# Fetal CMR 4D arm (van Amerom et al., MRM 2019) — SVRTK `mirtk reconstructCardiac` 4D-joint recon
# of ONE subject from the self-gated rolled/ stack (docs/105). Sibling of run_svrtk3d.sh /
# run_nesvor.sh / run_niftymic_v2.sh: same layout, provenance, stamp contract; the driver is
# run_baselines.py --method fetal_cmr_4d.
#
# Input is FIXED to the bundle's rolled/ stack (unknown per-slice phase) — the only input on which
# the method's self-gating has a job (docs/105). The per-frame cardiac phase + per-slice R-R come
# from fetal4d_gate.py (LV-area ED anchor, Akesson 2025, replacing the paper's overlap-based
# inter-slice sync that single-orientation SAX cannot run — docs/34/35). The recon engine and
# its parameters are the authors' verbatim (recon_cine_vol.bash; DEVIATIONS.md par.C/E):
#   -iterations 4 -rec_iterations 10 -rec_iterations_last 20 -resolution 1.4 (matches SVRTK/NeSVoR/NiftyMIC, not the author default 1.25 -- docs/105), robust statistics
#   ON, temporal PSF, continuous phases. Deviations, all disclosed: -numcardphase 12 (= GT, the
#   authors' 25 is a free output length), nominal R-R 1.0 s (no timing in the sources), recon
#   ROI = the bundle's mask_heart (same as every other baseline), single stack.
#
# Usage: EVAL_DATASET=<src> [THICK=] [T=] [ROBUST=1] [RES=1.4] bash evaluation/src/engine/run_fetal4d.sh <subject> breath
set -uo pipefail
VGGT=/home/minsukc/vggt
export PATH="$VGGT/baselines/fetal_cmr_4d/bin:$PATH"
export FCMR_BIND="$VGGT/scratch"      # bind the whole GPFS tree (covers sif + eval/)

SUBJ="${1:?subject}"; VAR="${2:-breath}"
[ "$VAR" = breath ] || { echo "fetal_cmr_4d runs on the breath bundle only (rolled/ is built from breath/)"; exit 2; }
# NF (frames per slice == output volumes) and NCP (cardiac bins == nominal beat length in frames)
# are read from the BUNDLE below, not fixed here: on a `*24` bundle they are 24 and 12. They were
# one number for every 12-frame bundle. `T` stays as the legacy override and, when set, pins BOTH.
THICK="${THICK:-8}"; RES="${RES:-1.4}"; ITERS="${ITERS:-4}"
NSR="${NSR:-10}"; NSRLAST="${NSRLAST:-20}"; ROBUST="${ROBUST:-1}"
METHOD="${METHOD:-fetal_cmr_4d}"; INPUT=rolled
# The env's python directly, not `micromamba run`: 16 concurrent array shards collide on
# micromamba's ~/.cache/mamba/proc lock ("LockFile can't be set") and abort the call.
SVR_PY="${SVR_PY:-/home/minsukc/micromamba/envs/svr/bin/python}"
SD="$VGGT/scratch/eval/${EVAL_DATASET:?EVAL_DATASET must name a source dir}/out/$SUBJ"
# GATE_ARM, not $METHOD: the fetal_cmr_4d_norobust / _motion arms deliberately SHARE the
# self-gated fetal_cmr_4d/gate, so substituting $METHOD would break them. Only an arm with its own
# gating (fetal_cmr_4d_oracle, from fetal4d_gate.py --oracle) overrides this.
GATE="$SD/${GATE_ARM:-fetal_cmr_4d}/gate"
OUT="$SD/$METHOD/recon_$VAR"; mkdir -p "$OUT"
for f in stack4d.nii.gz cardphase.txt rrintervals.txt gate.json; do
  [ -f "$GATE/$f" ] || { echo "missing $GATE/$f — run fetal4d_gate.py dump/seg/assemble first"; exit 3; }
done
# Output readout mode comes from the BUNDLE, never the invocation: a bundle with a rhythm block
# (tools/build_af_bundle.py, docs/110) is FRAME-indexed, so its targets are the reference slice's
# own 12 acquisition instants; a legacy bundle has no such key and takes the original gt_ed roll
# VERBATIM. You therefore cannot apply the wrong readout to the wrong cohort by mistake.
# Both reads FAIL CLOSED: the script has no `set -e`, so a failed $( ) would otherwise leave an
# empty string, and an empty READOUT is != gt_ed_roll -- i.e. it would silently select the NEW
# path and skip the seg_gt precondition. Exit instead.
READOUT=$("$SVR_PY" -c "import json,sys;print('ref_phase' if 'rhythm' in json.load(open(sys.argv[1])) else 'gt_ed_roll')" "$SD/manifest.json") \
  || { echo "cannot read $SD/manifest.json to determine the readout mode"; exit 6; }
REF_PLANE=$("$SVR_PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['scatter']['ref_plane'])" "$SD/manifest.json") \
  || { echo "cannot read scatter.ref_plane from $SD/manifest.json"; exit 6; }
# Same fail-closed rule as READOUT above. NF drives the OUTPUT length and the theta slice; NCP
# drives -numcardphase and the phase-cine modulus. An explicit T= pins both (legacy behaviour).
NF=$("$SVR_PY" -c "import json,sys;print(int(json.load(open(sys.argv[1]))['T']))" "$SD/manifest.json") \
  || { echo "cannot read T from $SD/manifest.json"; exit 6; }
NCP=$("$SVR_PY" -c "import json,sys;m=json.load(open(sys.argv[1]));print(int(m.get('n_cardphase',m['T'])))" "$SD/manifest.json") \
  || { echo "cannot read n_cardphase from $SD/manifest.json"; exit 6; }
[ -n "$T" ] && { NF="$T"; NCP="$T"; }
T="$NF"                     # legacy alias: provenance/echo lines below still print $T
# The legacy readout rolls the engine's NCP-long phase axis by an index taken from an NF-long GT
# curve. That is only meaningful when the two are equal, and every NF > NCP bundle carries a
# rhythm block and therefore takes the ref_phase path -- so this can only fire on a malformed one.
[ "$READOUT" = gt_ed_roll ] && [ "$NF" != "$NCP" ] && {
  echo "gt_ed_roll readout needs NF == NCP (got $NF != $NCP)"; exit 7; }
MASK="$SD/${MASK_FILE:-mask_heart.nii.gz}"
NSLICE=$(wc -w < "$GATE/rrintervals.txt"); NFRAME=$(wc -w < "$GATE/cardphase.txt")
MEANRR=$(python3 -c "import sys;v=[float(x) for x in open(sys.argv[1]).read().split()];print(sum(v)/len(v))" "$GATE/rrintervals.txt")
# The container's reconstructCardiac parses `-cardphase N v1..` / `-rrintervals N v1..` as N+1
# entries with the COUNT token as entry 0 (frame 0 / slice 0), shifting every value by one
# (measured on the P012 debug run, docs/105 par.5c; the authors' scripts pass the lists the same
# way). Workaround: count = 710 = 113*2pi (0.0004 rad) and fetal4d_gate.py puts an ED-phase frame
# at frame 0, so entry 0 is right; entries 1..NFRAME-1 = our thetas 1..NFRAME-1, zero-padded to
# 710 tokens. Per-slice R-R is not passed at all: it is a single nominal constant (rrintervals.txt
# is provenance only) and the engine's fallback sets every slice to -rrinterval.
# The count token itself READS as frame 0 / slice 0's phase, so it must encode theta[0][0].
# 710 = 113*2pi = 6e-05 rad. VERIFIED against the engine source: cardPhase is stored raw
# (reconstructCardiac.cc:400) and its only consumer CalculateAngularDifference takes a true modulo,
# so angdiff(710) == angdiff(710 mod 2pi) exactly.
# BOTH arms use this literal. Self-gated: assemble() puts an ED frame first, so theta00 == 0.
# Oracle: fetal4d_gate.py shifts every theta by a constant so theta00 == 710 mod 2pi (a constant
# offset only relabels the output bins and the ref_phase readout undoes it). Do NOT replace this
# with a search for an integer matching an arbitrary theta00 -- 710 is a convergent of 2pi, so the
# reachable residues have ~9e-3 rad gaps and no feasible search width gets within 1e-3.
CPCOUNT=710
[ "$NFRAME" -le "$CPCOUNT" ] || { echo "NFRAME=$NFRAME > $CPCOUNT: raise CPCOUNT to a larger multiple of 2pi"; exit 4; }
CPARGS=$("$SVR_PY" -c "
import sys, math
v=open(sys.argv[1]).read().split(); n=int(sys.argv[2])
# entry 0 is supplied by the count token; verify it really encodes theta[0][0].
res = abs((n - float(v[0]) + math.pi) % (2*math.pi) - math.pi)
assert res < 1e-3, f'count token {n} encodes phase off by {res:.2e} rad from theta[0][0]={v[0]} -- for an oracle gate, re-run fetal4d_gate.py assemble --oracle (it shifts theta to match)'
print(' '.join(v[1:] + ['0.0']*(n-len(v)+1)))" "$GATE/cardphase.txt" "$CPCOUNT") \
  || { echo "cardphase argument build FAILED (see the traceback above) -- refusing to run the engine"; exit 5; }
ROBUST_FLAG=""; [ "$ROBUST" = "1" ] || ROBUST_FLAG="-no_robust_statistics"
OMP="${OMP:-$(nproc)}"

# --- provenance (same format as the sibling shells) ---
SIF="${FCMR_SIF:-$VGGT/scratch/fetal_cmr_4d/sif/svrtk.sif}"
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
JINFO=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -oE 'cpu=[0-9]+,mem=[0-9]+[MG]' | head -1)
NCPU_ALLOC=$(echo "$JINFO" | grep -oE 'cpu=[0-9]+' | cut -d= -f2); NCPU_ALLOC=${NCPU_ALLOC:-$(nproc)}
MEM_ALLOC=$(echo "$JINFO" | grep -oE 'mem=[0-9]+[MG]' | cut -d= -f2); MEM_ALLOC=${MEM_ALLOC:-unknown}
{
  echo "engine          : SVRTK 'mirtk reconstructCardiac' (4D joint, temporal PSF; fetal_cmr_4d recon_cine_vol.bash params)"
  echo "command         : mirtk reconstructCardiac cine.nii.gz 1 <gate/stack4d.nii.gz> -thickness $THICK -mask mask_heart \\"
  echo "                    -iterations $ITERS -rec_iterations $NSR -rec_iterations_last $NSRLAST -resolution $RES \\"
  echo "                    -numcardphase $NCP -rrinterval $MEANRR -cardphase $CPCOUNT <theta_1..theta_$((NFRAME-1)), 0-padded> $ROBUST_FLAG"
  if [ "$READOUT" = gt_ed_roll ]; then
    echo "cardphase note  : count token $CPCOUNT = 113*2pi is read as frame 0's phase (engine off-by-one, docs/105 par.5c); frame 0 = slice-0 ED frame"
  else
    echo "cardphase note  : count token $CPCOUNT = 113*2pi is read as frame 0's phase (engine off-by-one, docs/105 par.5c); frame order is ACQUISITION order (no slice-0 ED re-roll) and thetas are offset so theta[0][0] matches the token"
  fi
  echo "rr note         : no -rrintervals (engine off-by-one); all $NSLICE slices take -rrinterval $MEANRR via the engine fallback"
  echo "stack4d dt      : $("$SVR_PY" -c "import nibabel as nib,sys;h=nib.load(sys.argv[1]).header;print(h.get_zooms()[3],'s, time units',h.get_xyzt_units()[1])" "$GATE/stack4d.nii.gz")"
  echo "params          : thickness_mm=$THICK resolution_mm=$RES iterations=$ITERS rec_iterations=$NSR/$NSRLAST \\"
  echo "                    robust_statistics=$([ "$ROBUST" = 1 ] && echo ON || echo OFF) numcardphase=$NCP rr_nominal_s=$MEANRR"
  if [ "${GATE_ARM:-fetal_cmr_4d}" = fetal_cmr_4d ]; then
    echo "gating          : fetal4d_gate.py (LV-area ED anchor, nnU-Net Task114 on rolled/ frames; docs/105) -> $GATE"
  elif [[ "${GATE_ARM:-}" == *_oracle_balanced ]]; then
    echo "gating          : fetal4d_gate.py --oracle --balanced (TRUE per-frame theta, rank-quantized onto a coverage-forced bijection with the output bins; NO nnU-Net; docs/111 s3d) -> $GATE"
  else
    echo "gating          : fetal4d_gate.py --oracle (TRUE per-frame theta from the simulation truth manifest; NO nnU-Net; docs/110) -> $GATE"
  fi
  echo "container(sif)  : $SIF"
  echo "container_id    : $(stat -c '%s bytes, mtime %y' "$SIF" 2>/dev/null)"
  echo "method          : $METHOD"
  echo "subject/variant : $SUBJ / $VAR   out_vols(NF)=$NF cardbins(NCP)=$NCP   input_stack=$INPUT   frames=$NFRAME slices=$NSLICE"
  echo "--- hardware / parallelism (for the compute-cost comparison) ---"
  echo "host            : $(hostname)   SLURM job ${SLURM_JOB_ID:-none}"
  echo "cpu model       : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"
  echo "SLURM ALLOC     : ${NCPU_ALLOC} CPUs, ${MEM_ALLOC} RAM"
  echo "parallelism     : OMP_NUM_THREADS=$OMP (one joint 4D solve, OpenMP-parallel)"
} > "$OUT/provenance.txt"

echo "=== $METHOD : $SUBJ / $VAR : thick=${THICK} res=${RES}mm iters=$ITERS rec=$NSR/$NSRLAST robust=$ROBUST T=$T frames=$NFRAME ==="
WD="$OUT/work"; rm -rf "$WD"; mkdir -p "$WD"
T0=$(date +%s)
( cd "$WD" && OMP_NUM_THREADS="$OMP" mirtk reconstructCardiac cine.nii.gz 1 "$GATE/stack4d.nii.gz" \
    -thickness "$THICK" -mask "$MASK" \
    -iterations "$ITERS" -rec_iterations "$NSR" -rec_iterations_last "$NSRLAST" \
    -resolution "$RES" -numcardphase "$NCP" -rrinterval "$MEANRR" \
    -cardphase "$CPCOUNT" $CPARGS $ROBUST_FLAG > log.txt 2>&1 )
DT=$(( $(date +%s) - T0 ))
cp -f "$WD/log.txt" "$OUT/log.txt" 2>/dev/null
echo "$DT" > "$OUT/total_wall.sec"
{ echo "--- timing ---"; echo "total_wall_sec  : $DT   (one joint 4D solve, all $T phases, on the above hardware)"; } >> "$OUT/provenance.txt"

# split the 4D cine into the per-phase files the scorer reads (vol_t00..T-1, own $RES mm grid).
# Phase convention: the engine's output phase 0 = the gater's ED (theta = 0). The GT's ED is
# frame 0 by the bundle build, but not for every subject (GT-seg ED != 0 on ~3% of CMRx23/24 and
# ~14% of CMRx25, docs/105 par.5d). Roll the output by the GT's seg-derived ED index — the SAME
# argmax the scorer uses (<subject>/seg_gt, ef_dice.py) — so "phase 0 = ED" means one thing on
# both sides. One scalar index convention, no slice-level GT information; a no-op when GT ED = 0.
SEG_GT="$SD/seg_gt"
# ref_phase needs no GT at all (it reads out at the arm's own thetas), so the seg_gt prerequisite
# applies only to the legacy roll.
if [ "$READOUT" = gt_ed_roll ]; then
  [ -f "$SEG_GT/seg_t00.nii.gz" ] || { echo "missing $SEG_GT — run ef_dice.py dump --gt-only + run_seg.sh + score first"; exit 3; }
fi
N_OK=0; GT_ED=-1
if [ -f "$WD/cine.nii.gz" ] && gzip -t "$WD/cine.nii.gz" 2>/dev/null; then
  read -r N_OK GT_ED < <("$SVR_PY" - "$WD/cine.nii.gz" "$OUT" "$NCP" "$SEG_GT" "$READOUT" "$GATE/cardphase.txt" "$REF_PLANE" "$NF" <<'EOF'
import sys, nibabel as nib, numpy as np
im = nib.load(sys.argv[1]); out = sys.argv[2]; T = int(sys.argv[3]); seg_gt = sys.argv[4]
readout, cp_path = sys.argv[5], sys.argv[6]      # argv[7] (ref) parsed inside the branch that
arr = np.asarray(im.dataobj, dtype=np.float32)   # needs it, so the legacy path stays scatter-free
# T = the engine's phase-cine length (-numcardphase); NF = frames per slice = output volumes.
# Equal for every 12-frame bundle; NF = 2T on a `*24` one, where the SAME T-bin periodic cine is
# read out at each of the NF acquisition instants.
NF = int(sys.argv[8]) if len(sys.argv) > 8 else T
if arr.ndim != 4 or arr.shape[3] != T:
    print(0, -1); sys.exit()
if readout == "gt_ed_roll":                   # LEGACY, unchanged (guarded to NF == T by the caller)
    lv = np.array([(np.asarray(nib.load(f"{seg_gt}/seg_t{t:02d}.nii.gz").dataobj) == 1).sum() for t in range(T)])
    tag = int(lv.argmax())                    # ef_dice.py: ed = gt.argmax() (LV label 1 voxel count)
    arr = np.roll(arr, tag, axis=3)           # output phase 0 (= gater ED) -> index gt_ed
else:
    # ref_phase: target f is the REFERENCE slice's acquisition instant f, so read the engine's
    # phase cine at that frame's own theta. REPLACES the gt_ed roll -- it does not compose with it.
    # (Under a periodic rhythm the gater's ED is (gt_ed + roll_ref) mod T, so this rule reduces
    # exactly to np.roll(arr, gt_ed); applying both would double-correct by gt_ed.)
    ref = int(sys.argv[7])
    th = np.array(open(cp_path).read().split(), dtype=np.float64)
    D = th.size // NF                          # cardphase.txt is slice-major over NF frames
    assert th.size == D * NF and 0 <= ref < D, (th.size, D, NF, T, ref)
    idx = th[ref * NF:(ref + 1) * NF] * T / (2.0 * np.pi)    # fractional PHASE-BIN index per frame
    frames = []
    for f in range(NF):
        i = float(idx[f]) % T
        r = int(round(i))
        if abs(i - r) < 1e-4:                 # exact bin: take it, no arithmetic (bit-exact).
            frames.append(arr[..., r % T])    # theta is stored to 6dp, so i is 1.0000003, not 1.0;
        else:                                 # blending with w=3e-7 is NOT bit-identical.
            lo = int(np.floor(i)); w = np.float32(i - lo)
            frames.append(arr[..., lo % T] * (np.float32(1.0) - w) + arr[..., (lo + 1) % T] * w)
    arr = np.stack(frames, axis=-1).astype(np.float32)
    tag = ref
nib.save(nib.Nifti1Image(arr, im.affine), sys.argv[1])
for t in range(arr.shape[3]):
    nib.save(nib.Nifti1Image(arr[..., t], im.affine), f"{out}/vol_t{t:02d}.nii.gz")
print(arr.shape[3], tag)
EOF
)
  mv -f "$WD/cine.nii.gz" "$OUT/cine.nii.gz"
  # info.tsv: the engine's final per-frame rigid pose (TranslationX/Y/Z mm, RotationX/Y/Z deg),
  # written every run (not just -debug) and confirmed identical to the last iteration's table
  # (docs/105 §11) — kept for demeaned-EPE breathing-motion diagnostics, same idea as VGGT's
  # resp_diag.json. Not produced by earlier campaign runs (deleted with $WD before this line
  # existed); re-run under a new arm name to add it without touching those results.
  [ -f "$WD/info.tsv" ] && cp -f "$WD/info.tsv" "$OUT/info.tsv"
  rm -rf "$WD"
else
  echo "FAIL recon (see $OUT/log.txt)"
fi
if [ "${N_OK:-0}" -eq "$T" ]; then
  printf '{"engine": "fetal_cmr_4d", "input_stack": "rolled", "thickness_mm": %s, "resolution_mm": %s, "iterations": %s, "rec_iterations": "%s/%s", "numcardphase": %s, "n_frames": %s, "rr_nominal_s": %s, "robust_statistics": "%s", "container_id": "%s", "cardphase_count_token": %s, "readout": "%s", "readout_arg": %s, "gate_arm": "%s"}\n' \
    "$THICK" "$RES" "$ITERS" "$NSR" "$NSRLAST" "$NCP" "$NF" "$MEANRR" "$([ "$ROBUST" = 1 ] && echo on || echo off)" "$(container_id "$SIF")" "$CPCOUNT" "$READOUT" "$GT_ED" "${GATE_ARM:-fetal_cmr_4d}" > "$OUT/stamp.json"
  if [ "$READOUT" = gt_ed_roll ]; then
    { echo "readout         : gt_ed_roll -- output rolled by $GT_ED phases so phase 0 = the GT's seg-derived ED (seg_gt LV argmax)"; } >> "$OUT/provenance.txt"
  else
    { echo "readout         : ref_phase -- output cine sampled at reference plane $GT_ED's own per-frame theta from $GATE/cardphase.txt (fractional, circular); REPLACES the gt_ed roll, uses no GT"; } >> "$OUT/provenance.txt"
  fi
else
  echo "NOT stamped: $N_OK/$T phases"
fi
echo "RECON_DONE $SUBJ $VAR  (total ${DT}s, ${N_OK:-0}/$T phases ok)"
