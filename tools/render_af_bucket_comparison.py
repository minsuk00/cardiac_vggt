#!/usr/bin/env python
"""Visual check for docs/111 s3d: what SCORED reconstructions actually look like, GT vs
self-gated / oracle-raw / oracle-balanced / VGGT.

Reuses image_metrics.py's own register+resample pipeline (load_blurred -> fit_rigid -> resample)
so what's rendered is EXACTLY what gets scored -- not a naive same-index slice across arms whose
native reconstruction grids differ (fetal_cmr_4d's engine output is a completely different shape/
orientation than GT's canonical grid; slicing both at the same Z index is meaningless).

`vol_t{f}.nii.gz` (fetal_cmr_4d family) is already the AF-order READOUT -- resampled via ref_phase
onto the same frame index f as gt_t{f} -- not the engine's raw uniform-phase-bin cine
(run_fetal4d.sh:192 saves `arr` AFTER the readout reassigns it). So `vol_t{f}` vs `gt_t{f}` is
already the correct pairing; no additional re-indexing needed here.

Usage: PYTHONPATH=training:. python tools/render_af_bucket_comparison.py [SUBJECT]
"""
import sys
import os
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/minsukc/vggt-afsim/evaluation")
sys.path.insert(0, "/home/minsukc/vggt-afsim/evaluation/src/score")
import paths           # noqa: E402
import pose_psf        # noqa: E402
import image_metrics as im  # noqa: E402

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "CMRx24_Test_P017"
COHORT = "cmrx2024_af"
ARMS = [("fetal_cmr_4d", "self-gated"), ("fetal_cmr_4d_oracle", "oracle-raw"),
        ("fetal_cmr_4d_oracle_balanced", "oracle-BAL"), ("vggt_final518_diff1000_ep300", "VGGT")]
T = 12

shape_xyz, aff = im.subject_grid(COHORT, SUBJ)
D = shape_xyz[2]
manifest = __import__("json").load(open(paths.manifest(COHORT, SUBJ)))
content = im.load_canon(str(paths.fov_mask(COHORT, SUBJ)), shape_xyz, aff) > 0.5
heart = im.load_canon(str(paths.heart_mask_pad(COHORT, SUBJ)), shape_xyz, aff) > 0.5
mask = heart & content
gt = np.stack([im.load_canon(str(paths.bundle_stack(COHORT, SUBJ, "gt", t)), shape_xyz, aff)
               for t in range(T)])
zs = sorted(set([max(1, D // 5), D // 2, min(D - 2, 4 * D // 5)]))
print(f"subject={SUBJ}  D={D}  z-slices={zs}")

# ---- for each arm, replicate image_metrics.py's registered read exactly ----
recons = {"GT": gt}
for arm, label in ARMS:
    thick = pose_psf.stamp_thickness(COHORT, SUBJ, arm, "breath") if pose_psf.needs_psf(arm) else None
    rec = []
    for t in range(T):
        rp = paths.recon(COHORT, SUBJ, arm, "breath", t)
        vol_t, vaff = pose_psf.load_blurred(rp, thick, float(abs(aff[0, 0])))
        pose = pose_psf.fit_rigid(vol_t, vaff, gt[t], mask, aff)
        rec.append(pose_psf.resample(vol_t, vaff, shape_xyz, aff, pose))
    rec = np.stack(rec)
    nmap = im.norm_map(rec, arm, content, None)
    rec = im.prep_recon(rec, arm, content, stacks=None, nmap=nmap) * content[None]
    recons[label] = rec
    print(f"  {label}: registered + normalized, {rec.shape}")

vmax = np.percentile(gt, 99.5)
COLS = ["GT"] + [l for _, l in ARMS]
CELL = 140
GIF = f"/home/minsukc/vggt-afsim/figs/af_bucket_comparison_{SUBJ}.gif"
frames = []
try:
    font = ImageFont.load_default()
except Exception:
    font = None

for t in range(T):
    row_imgs = []
    for z in zs:
        strip = []
        for col in COLS:
            img = recons[col][t, :, :, z]
            img = np.clip(img / vmax * 255, 0, 255).astype(np.uint8)
            pim = Image.fromarray(img.T[::-1]).resize((CELL, CELL), Image.NEAREST).convert("RGB")
            strip.append(pim)
        row = Image.new("RGB", (CELL * len(COLS), CELL))
        for i, p in enumerate(strip):
            row.paste(p, (i * CELL, 0))
        row_imgs.append(row)
    full = Image.new("RGB", (CELL * len(COLS), CELL * len(zs) + 30), "black")
    d = ImageDraw.Draw(full)
    for i, col in enumerate(COLS):
        d.text((i * CELL + 5, 2), col, fill="white", font=font)
    for zi, row in enumerate(row_imgs):
        full.paste(row, (0, 30 + zi * CELL))
    d.text((5, full.height - 2), "", fill="white")
    d.text((CELL * len(COLS) - 60, 2), f"f={t}", fill="yellow", font=font)
    frames.append(full)

frames[0].save(GIF, save_all=True, append_images=frames[1:], duration=120, loop=0)
print(f"-> {GIF}  ({len(frames)} frames, cols={COLS}, rows z={zs})")
