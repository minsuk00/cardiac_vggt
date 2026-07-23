#!/bin/bash
# Idempotent copy of the ep99 "final" eval checkpoints into evaluation/checkpoints/<arm>/.
# Re-runnable: skips arms already byte-identical, (re)copies the rest. Temporary Phase-2 helper.
set -u
cd /home/minsukc/vggt
CK=evaluation/checkpoints
declare -A MAP=(
  [vggt_20260719_1f_aug_moderate_ep99]=20260719_1frame_aug_moderate_ep99.pt
  [vggt_20260719_1f_dino_ft_ep99]=20260719_1frame_dino_ft_ep99.pt
  [vggt_20260719_1f_gather05_ep99]=20260719_1frame_gather05_ep99.pt
  [vggt_20260719_1f_lowdiff100_ep99]=20260719_1frame_lowdiff100_ep99.pt
  [vggt_20260719_1f_no_gather_ep99]=20260719_1frame_no_gather_ep99.pt
  [vggt_20260719_1f_contz_ep99]=20260719_1frame_contz_ep99.pt
)
for arm in "${!MAP[@]}"; do
  src=scratch/checkpoints/${MAP[$arm]}
  dst=$CK/$arm/checkpoint.pt
  mkdir -p "$CK/$arm"
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
    echo "SKIP  $arm (already byte-identical)"; continue
  fi
  echo "COPY  $arm  <-  ${MAP[$arm]}"
  cp "$src" "$dst"
  if cmp -s "$src" "$dst"; then echo "  OK  byte-identical"; else echo "  FAIL mismatch $arm"; fi
done
# doubled OOD contz arm shares the contz ckpt (symlink, no extra bytes)
mkdir -p "$CK/vggt_20260719_1f_contz_ep99_contz"
ln -sfn ../vggt_20260719_1f_contz_ep99/checkpoint.pt "$CK/vggt_20260719_1f_contz_ep99_contz/checkpoint.pt"
echo "DONE  total: $(du -sh $CK | cut -f1)"
