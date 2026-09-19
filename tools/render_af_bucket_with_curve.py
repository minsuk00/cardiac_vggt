#!/usr/bin/env python
"""Same as render_af_bucket_comparison.py, plus a synced LV-volume curve panel at the bottom
(from seg_gt) with a moving marker showing which frame is currently displayed."""
import sys
import json
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

sys.path.insert(0, "/home/minsukc/vggt-afsim/evaluation")
sys.path.insert(0, "/home/minsukc/vggt-afsim/evaluation/src/score")
import paths           # noqa: E402
import pose_psf        # noqa: E402
import image_metrics as im  # noqa: E402

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "CMRx24_Test_P004"
COHORT = "cmrx2024_af"
ARMS = [("fetal_cmr_4d", "self-gated"), ("fetal_cmr_4d_oracle", "oracle-raw"),
        ("fetal_cmr_4d_oracle_balanced", "oracle-BAL"), ("vggt_final518_diff1000_ep300", "VGGT")]
T = 12

shape_xyz, aff = im.subject_grid(COHORT, SUBJ)
D = shape_xyz[2]
content = im.load_canon(str(paths.fov_mask(COHORT, SUBJ)), shape_xyz, aff) > 0.5
heart = im.load_canon(str(paths.heart_mask_pad(COHORT, SUBJ)), shape_xyz, aff) > 0.5
mask = heart & content
gt = np.stack([im.load_canon(str(paths.bundle_stack(COHORT, SUBJ, "gt", t)), shape_xyz, aff)
               for t in range(T)])
zs = sorted(set([max(1, D // 5), D // 2, min(D - 2, 4 * D // 5)]))

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

# GT LV curve from seg_gt (real nnU-Net segmentation of the AF-simulated targets)
segs = [np.asarray(nib.load(f"{paths.subject_dir(COHORT, SUBJ)}/seg_gt/seg_t{t:02d}.nii.gz").dataobj)
        for t in range(T)]
lv = np.array([(s == 1).sum() for s in segs], float)
ef = (lv.max() - lv.min()) / lv.max() * 100
print(f"{SUBJ}: LV voxels/frame = {lv.astype(int).tolist()}  EF={ef:.1f}%")

vmax = np.percentile(gt, 99.5)
COLS = ["GT"] + [l for _, l in ARMS]
CELL = 140
try:
    font = ImageFont.load_default()
except Exception:
    font = None

frames = []
for t in range(T):
    # --- top: image montage ---
    row_imgs = []
    for z in zs:
        strip = Image.new("RGB", (CELL * len(COLS), CELL))
        for i, col in enumerate(COLS):
            img = recons[col][t, :, :, z]
            img = np.clip(img / vmax * 255, 0, 255).astype(np.uint8)
            pim = Image.fromarray(img.T[::-1]).resize((CELL, CELL), Image.NEAREST).convert("RGB")
            strip.paste(pim, (i * CELL, 0))
        row_imgs.append(strip)
    top = Image.new("RGB", (CELL * len(COLS), CELL * len(zs) + 30), "black")
    d = ImageDraw.Draw(top)
    for i, col in enumerate(COLS):
        d.text((i * CELL + 5, 2), col, fill="white", font=font)
    for zi, r in enumerate(row_imgs):
        top.paste(r, (0, 30 + zi * CELL))
    d.text((CELL * len(COLS) - 90, 2), f"target f={t}", fill="yellow", font=font)

    # --- bottom: LV curve plot with moving marker ---
    fig, ax = plt.subplots(figsize=(CELL * len(COLS) / 100, 1.8), dpi=100)
    ax.plot(range(T), lv, "o-", color="cyan")
    ax.plot(t, lv[t], "o", color="red", markersize=12)
    ax.set_facecolor("black"); fig.patch.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xlabel("target frame f", color="white", fontsize=8)
    ax.set_ylabel("LV voxels", color="white", fontsize=8)
    ax.set_title(f"GT LV curve  (EF = {ef:.1f}%)", color="white", fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="black"); plt.close(fig)
    buf.seek(0)
    curve = Image.open(buf).convert("RGB").resize((CELL * len(COLS), int(CELL * len(COLS) / 100 * 1.8 / (CELL*len(COLS)/100) * 100 / (CELL*len(COLS)/100))))
    curve = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    curve = curve.resize((top.width, int(curve.height * top.width / curve.width)))

    full = Image.new("RGB", (top.width, top.height + curve.height), "black")
    full.paste(top, (0, 0))
    full.paste(curve, (0, top.height))
    frames.append(full)

out = f"/home/minsukc/vggt-afsim/figs/af_bucket_with_curve_{SUBJ}.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=250, loop=0)
print(f"-> {out}")
