#!/usr/bin/env bash
# STEP 1 — plumbing test of the AUTHOR recon_cine_vol.bash wrapper on single-orientation MIITT.
#
# As faithful as possible; the ONLY departures are:
#   - resolution coarsened to $RESMM (default 3mm) — speed/memory tweak so it fits the
#     32GB interactive cap and finishes fast (real run uses author 1.25mm on SLURM).
#   - single-orientation adaptations (documented in DEVIATIONS.md): dc_vol drops
#     -stack_registration; intra-slice cardphases substituted for the inapplicable
#     interslice sync.
# Everything else is the author wrapper as-is (robust stats ON, iterations 4,
# rec_iterations 10/20, tight heart mask built by the wrapper, -remote kept).
#
# Reports whether the chain completes end-to-end + the peak reconstructCardiac RSS,
# from which we size the SLURM --mem for the faithful 1.25mm run.
#
# Usage: bash baselines/fetal_cmr_4d/scripts/run_faithful_test.sh [VOL] [RESMM]
set -uo pipefail
VGGT=/home/minsukc/vggt; FCMR=$VGGT/baselines/fetal_cmr_4d
REPO=$VGGT/scratch/fetal_cmr_4d/repo; RECROOT=$VGGT/scratch/fetal_cmr_4d/recon
VOL="${1:-Volunteer1}"; RESMM="${2:-3.0}"; RD="$RECROOT/$VOL"
export PATH="$FCMR/bin:$PATH"; export FCMR_BIND="$VGGT/scratch/fetal_cmr_4d"
module load matlab/R2024b 2>/dev/null || true

T=$(mktemp -d)
# temp RESMM copies: our patched dc_vol + the AUTHOR cine_vol (only RESOLUTION changed)
sed "s/^RESOLUTION=.*/RESOLUTION=$RESMM/" "$FCMR/scripts/recon_dc_vol_miitt.bash" > "$T/dc.bash"
sed "s/^RESOLUTION=.*/RESOLUTION=$RESMM/" "$REPO/4drecon/recon_cine_vol.bash"     > "$T/cine.bash"

# peak-RSS sampler (KB)
( m=0; while :; do r=$(ps -C reconstructCardiac -o rss= 2>/dev/null | sort -n | tail -1)
  [ -n "${r:-}" ] && [ "$r" -gt "$m" ] && m=$r; echo "$m" > "$T/maxrss"; sleep 3; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; rm -rf "$T"' EXIT

echo "[1/3] gating (reuse if present)"
[ -f "$RD/cardsync/cardphases_intraslice_cardsync.txt" ] || \
  ( cd "$FCMR/matlab" && matlab -nodisplay -batch "miitt_gating('$RD','$REPO','$FCMR/matlab')" ) || { echo "FAIL gating"; exit 1; }

echo "[2/3] dc_vol static MC @ ${RESMM}mm (single-stack patched)"
rm -rf "$RD/dc_vol"; bash "$T/dc.bash" "$RD" dc_vol || { echo "FAIL dc_vol"; exit 1; }
echo "    slice dofs produced: $(ls "$RD"/dc_vol/slice_transformations/*.dof 2>/dev/null | wc -l), stack dof: $(ls "$RD"/dc_vol/stack_transformations/*.dof 2>/dev/null | wc -l)"

echo "[3/3] cine_vol 4D recon @ ${RESMM}mm (AUTHOR wrapper, robust ON)"
cp "$RD/cardsync/cardphases_intraslice_cardsync.txt" "$RD/cardsync/cardphases_interslice_cardsync.txt"
rm -rf "$RD/cine_vol"; bash "$T/cine.bash" "$RD" cine_vol || { echo "FAIL cine_vol"; exit 1; }

PEAK=$(cat "$T/maxrss" 2>/dev/null || echo 0)
echo "PEAK_RSS_KB=$PEAK"
awk -v p="$PEAK" -v r="$RESMM" 'BEGIN{g=p/1048576; printf "PEAK ~ %.1f GB @ %smm  =>  extrapolated ~ %.0f GB @ 1.25mm\n", g, r, g*(r/1.25)^3}'
if [ -f "$RD/cine_vol/cine_vol.nii.gz" ]; then echo "STEP1 OK -> $RD/cine_vol/cine_vol.nii.gz"; else echo "STEP1 INCOMPLETE (no cine_vol.nii.gz)"; fi
