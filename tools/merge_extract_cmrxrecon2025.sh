#!/bin/bash
# Merge (only if needed) + extract CMRxRecon2025 V2 zip parts.
#
# Notes learned the hard way:
#  - Modern `unzip` falsely flags these large zip64 archives as a "zip bomb"
#    ("invalid zip file with overlapped components") -> UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE.
#    The archives are fine (per-file CRC passes, -l lists cleanly); it's a heuristic false positive.
#  - `unzip` verifies every extracted file's CRC and errors on mismatch, so extraction itself IS the
#    corruption gate (set -e aborts on a bad CRC). No separate whole-archive `unzip -t` (it also trips
#    the zip-bomb heuristic and doubles the I/O).
#  - Parts carry the rclone "Copy of " prefix; we cat them by name (originals untouched).
#  - Merged *-encrypted.zip files are KEPT as the full-modality archive (extract more later on demand).
#
#   INCLUDE='*/Cine/*' bash tools/merge_extract_cmrxrecon2025.sh TrainingData TaskR1 TaskR2   # cine only
#   bash tools/merge_extract_cmrxrecon2025.sh TrainingData                                    # all modalities
set -euo pipefail
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE

DEST=/home/minsukc/vggt/scratch/data/CMRxRecon2025
PW=CMRxRecon2025Lucky
INCLUDE="${INCLUDE:-}"          # optional in-archive glob (e.g. '*/Cine/*'); empty = everything
declare -A MAXIDX=( [TrainingData]=57 [TaskR1]=50 [TaskR2]=58 )

for S in "$@"; do
  n=${MAXIDX[$S]:?unknown series $S}
  dir="$DEST/$S"; zip="$DEST/${S}-encrypted.zip"; ext="$DEST/${S}_extracted"

  if [ -f "$zip" ]; then
    echo "=== [$S] merged zip present ($(du -sh "$zip" | cut -f1)) — skip merge ==="
  else
    echo "=== [$S] verifying + ordering parts 000..$(printf %03d "$n") ==="
    files=()
    for i in $(seq 0 "$n"); do
      f=$(printf "%s/Copy of %s-part-%03d.zip" "$dir" "$S" "$i")
      [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
      files+=("$f")
    done
    echo "=== [$S] merging ${#files[@]} parts -> $zip ($(date +%H:%M:%S)) ==="
    cat "${files[@]}" > "$zip"
    echo "  merged: $(du -sh "$zip" | cut -f1)"
  fi

  echo "=== [$S] extracting '${INCLUDE:-ALL}' -> $ext ($(date +%H:%M:%S)) ==="
  mkdir -p "$ext"
  if [ -n "$INCLUDE" ]; then
    unzip -o -q -P "$PW" "$zip" "$INCLUDE" -d "$ext"
  else
    unzip -o -q -P "$PW" "$zip" -d "$ext"
  fi
  echo "  extracted: $(du -sh "$ext" | cut -f1) | files: $(find "$ext" -type f | wc -l) ($(date +%H:%M:%S))"
  echo "=== [$S] DONE ==="
done
echo "ALL DONE"
