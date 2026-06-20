"""Qualitative panel: V_gt / model V_refined / oracle-perfect / oracle-native-256 for one subject.

Visualizes the two limitations: model is blurry+displaced (geometry+resize); oracle-perfect is
sharp-ish but resize-blurred; oracle-native256 is crisp (renderer fix). Mid-heart z planes.
"""
import os, sys
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
from eval_refiner import make_refiner_model
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from limits_decomposition import build_oracle_splat
JOINT = os.path.join(REPO, "scratch", "logs", "218349151_mri_refiner_joint", "ckpts", "checkpoint_last.pt")
OUT = os.path.join(REPO, "result", "limits_eval")

device = "cuda"
ds = build_dataset()
model, _ = make_refiner_model(JOINT, device)
cfg = RespiratoryConfig(enable=True, amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3,
                        ap_ratio=0.35, ap_axis="H", per_slot=True, direction_jitter_deg=30.0)
seq = 7
data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
batch = build_batch(data, device, seq_index=seq)
batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
out = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]}, batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
V_gt = out["V_gt"][0].float()
bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
sc = batch["scanner_coords"]; z_norm = sc[0, :, 0, 0, 2]
z_idx = ((z_norm + 1) / 2 * 11).round().long().clamp(0, 11).tolist()
ys, xs = torch.meshgrid(torch.linspace(-1, 1, 518, device=device), torch.linspace(-1, 1, 518, device=device), indexing="ij")
mesh518 = torch.stack([xs, ys], dim=-1)
ys2, xs2 = torch.meshgrid(torch.linspace(-1, 1, 256, device=device), torch.linspace(-1, 1, 256, device=device), indexing="ij")
mesh256 = torch.stack([xs2, ys2], dim=-1)
with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    preds = model(batch["images"], batch=batch)
Vref = preds["V_refined"][0].float().cpu().numpy()
Vgt = V_gt.cpu().numpy()
Vor = build_oracle_splat(V_gt, z_idx, mesh518, 518, 256)[0].cpu().numpy()
Vnat = build_oracle_splat(V_gt, z_idx, mesh256, 256, 256)[0].cpu().numpy()

z0, z1 = bbox[0], bbox[1]
zs = list(range(max(z0, (z0+z1)//2 - 2), min(z1, (z0+z1)//2 + 3)))
rows = [("GT (target phase)", Vgt), ("model V_refined", Vref),
        ("oracle: perfect placement\n(518→256 splat)", Vor),
        ("oracle: NATIVE-256 splat\n(renderer fix)", Vnat)]
vmax = float(Vgt.max())
fig = plt.figure(figsize=(1.7*len(zs)+1.8, 1.7*len(rows)+0.6), dpi=125)
gs = gridspec.GridSpec(len(rows), len(zs), wspace=0.04, hspace=0.06)
for r, (lab, vol) in enumerate(rows):
    for c, z in enumerate(zs):
        ax = fig.add_subplot(gs[r, c]); ax.imshow(vol[z], cmap="gray", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(f"z={z}", fontsize=9)
        if c == 0: ax.set_ylabel(lab, fontsize=9)
fig.suptitle("Same target-phase volume, mid-heart planes — the two limitations, visualized", fontsize=11, y=1.005)
fig.savefig(os.path.join(OUT, "qual_panel.png"), bbox_inches="tight", dpi=125, facecolor="white")
print("saved qual_panel.png")
