"""Figures for the toy-experiment report: EF scatter, LV curves, segmentation overlays."""
import json, os, glob, re
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PA = "scratch/phase_analysis"
OUT = "scratch/phase_analysis/figs"; os.makedirs(OUT, exist_ok=True)
VOX = 1.4 * 1.4 * 8.0 / 1000.0; T = 12; LV = 1


def lv_curve(seg_dir, subj, kind):
    c = np.full(T, np.nan)
    for t in range(T):
        f = os.path.join(seg_dir, f"{subj}_t{t:02d}_{kind}.nii.gz")
        if os.path.exists(f):
            c[t] = (np.asarray(nib.load(f).dataobj) == LV).sum() * VOX
    return c


# ---------- Fig 1: EF scatter (model flat vs oracle perfect) ----------
mc = json.load(open(f"{PA}/model_contraction.json"))["rows"]
gt_ef = np.array([r["gt_ef"] for r in mc]); model_ef = np.array([r["pred_ef"] for r in mc])
# oracle EF
osub = sorted({re.match(r"^(.*)_t\d{2}_(oracle|gt)$", os.path.basename(s)[:-7]).group(1)
               for s in glob.glob(f"{PA}/oracle_segs/*.nii.gz")
               if re.match(r"^(.*)_t\d{2}_(oracle|gt)$", os.path.basename(s)[:-7])})
orc_ef, orc_gt = [], []
for s in osub:
    o = lv_curve(f"{PA}/oracle_segs", s, "oracle"); g = lv_curve(f"{PA}/oracle_segs", s, "gt")
    if np.isnan(o).any() or np.isnan(g).any() or (o <= 0).any() or (g <= 0).any():
        continue
    orc_ef.append((o.max() - o.min()) / o.max() * 100); orc_gt.append((g.max() - g.min()) / g.max() * 100)
orc_ef = np.array(orc_ef); orc_gt = np.array(orc_gt)

fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.plot([20, 95], [20, 95], "k--", lw=1, alpha=0.5, label="perfect (pred=true)")
ax.scatter(orc_gt, orc_ef, s=42, c="#1f77b4", alpha=0.8, label=f"ORACLE splat (perfect data): slope {np.polyfit(orc_gt,orc_ef,1)[0]:.2f}")
ax.scatter(gt_ef, model_ef, s=42, c="#cc3311", alpha=0.85, label=f"MODEL (VGGT): slope {np.polyfit(gt_ef,model_ef,1)[0]:.2f}")
ax.axhline(model_ef.mean(), color="#cc3311", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("TRUE ejection fraction (%)"); ax.set_ylabel("RECONSTRUCTED ejection fraction (%)")
ax.set_title("The model gives ~everyone the same EF (~48%),\nregardless of true contraction — the splat does NOT (slope 1.0)")
ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(25, 95); ax.set_ylim(15, 95)
fig.tight_layout(); fig.savefig(f"{OUT}/ef_scatter.png", dpi=120); plt.close(fig)
print("ef_scatter done")

# ---------- Fig 2: LV-volume curves (timing right, depth wrong) ----------
fig, ax = plt.subplots(figsize=(7.5, 4))
pick = sorted(mc, key=lambda r: -r["gt_ef"])[:3] + sorted(mc, key=lambda r: r["gt_ef"])[:1]
cols = plt.cm.viridis(np.linspace(0, 0.85, len(pick)))
for r, c in zip(pick, cols):
    g = np.array(r["gt"]); p = np.array(r["pred"])
    ax.plot(range(T), g / g.max(), "-o", color=c, ms=3, label=f"{r['subj'][:9]} GT (EF {r['gt_ef']:.0f}%)")
    ax.plot(range(T), p / p.max(), "--s", color=c, ms=3, alpha=0.7)
ax.set_xlabel("phase t"); ax.set_ylabel("LV volume / max  (solid=GT, dashed=model)")
ax.set_title("Model tracks the contraction TIMING (trough at right t) but not the DEPTH")
ax.legend(fontsize=7, ncol=2); ax.set_xticks(range(T)); fig.tight_layout()
fig.savefig(f"{OUT}/lv_curves.png", dpi=120); plt.close(fig)
print("lv_curves done")

# ---------- Fig 3: segmentation overlays (high-EF patient, ES) ----------
def midslice_overlay(ax, vol_path, seg_path, title):
    img = np.asarray(nib.load(vol_path).dataobj).astype(float)   # (X,Y,Z)
    seg = np.asarray(nib.load(seg_path).dataobj).astype(int)
    z = img.shape[2] // 2
    ax.imshow(img[:, :, z].T, cmap="gray")
    cmap = ListedColormap(["none", "#ff3333", "#ffdd33", "#33ddff"])
    ax.imshow(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]).T, cmap=cmap, vmin=0, vmax=3, alpha=0.55)
    ax.set_title(title, fontsize=9); ax.axis("off")

# pick the extreme high-EF patient
hi = max(mc, key=lambda r: r["gt_ef"]); subj = hi["subj"]; es = int(np.array(hi["gt"]).argmin()); ed = int(np.array(hi["gt"]).argmax())
fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
for col, (kind, segdir, voldir, lab) in enumerate([
        ("gt", "model_segs", "model_vols", "GROUND TRUTH (dense)"),
        ("oracle", "oracle_segs", "oracle_vols", "ORACLE splat (perfect data)"),
        ("pred", "model_segs", "model_vols", "MODEL reconstruction")]):
    for row, (ph, phn) in enumerate([(ed, "ED (filled)"), (es, "ES (contracted)")]):
        vp = f"{PA}/{voldir}/{subj}_t{ph:02d}_{kind}_0000.nii.gz"
        sp = f"{PA}/{segdir}/{subj}_t{ph:02d}_{kind}.nii.gz"
        if os.path.exists(vp) and os.path.exists(sp):
            midslice_overlay(axes[row, col], vp, sp, f"{lab}\n{phn}" if row == 0 else phn)
        else:
            axes[row, col].axis("off")
fig.suptitle(f"{subj}: true EF {hi['gt_ef']:.0f}%  →  model EF {hi['pred_ef']:.0f}%   "
             f"(red=LV blood pool; note the model's ES cavity stays large = under-contraction)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/seg_overlay.png", dpi=120); plt.close(fig)
print(f"seg_overlay done (subj {subj}, ED t{ed} ES t{es})")
print("figs ->", OUT)
