#!/bin/bash
# Download the Göttingen radial real-time bSSFP dataset (5 Zenodo records, 68 volumes, ~224 GB)
# into one flat dir. Resumable (wget -c + size check), parallel (6 concurrent files).
# Manifest cols: record_id \t url \t filename \t size_bytes
set -uo pipefail

DEST=/home/minsukc/vggt/scratch/data/goettingen/radial
MANIFEST=/tmp/goettingen_manifest.tsv
LOG=$DEST/download.log
PAR=6
mkdir -p "$DEST"

fetch() {
  local url="$1" key="$2" size="$3"
  local out="$DEST/$key"
  if [[ -f "$out" && "$(stat -c%s "$out")" == "$size" ]]; then return 0; fi
  wget -c -q --tries=8 --timeout=60 -O "$out" "$url"
  local have; have=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [[ "$have" == "$size" ]]; then echo "OK   $key" >>"$LOG"; else echo "FAIL $key ($have/$size)" >>"$LOG"; fi
}
export -f fetch; export DEST LOG

echo "=== parallel download start $(date) (P=$PAR) ===" >>"$LOG"
# largest files first so the long tail isn't a single 3GB straggler
sort -t$'\t' -k4 -nr "$MANIFEST" | \
  xargs -P "$PAR" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r rid url key size <<< "{}"; fetch "$url" "$key" "$size"'
echo "=== done $(date): $(grep -c '^OK' "$LOG") ok / $(grep -c '^FAIL' "$LOG") fail ===" >>"$LOG"
