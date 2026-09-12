"""Evidence walkthrough figures for the ES 'under-contraction' diagnosis -> temp/ef_fix/."""
import json, csv, os
import numpy as np, pandas as pd, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi, stats

OUT = "temp/ef_fix"; os.makedirs(OUT, exist_ok=True)
BLUE, RED, GRAY, INKM = "#3b6fb5", "#c84b4b", "#9aa3ad", "#555b63"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": "#222",
                     "xtick.color": INKM, "ytick.color": INKM, "axes.titlesize": 10,
                     "figure.facecolor": "white", "axes.grid": True,
                     "grid.color": "#e6e8ea", "grid.linewidth": 0.6, "axes.axisbelow": True})

ROOT = "temp/dice_ef_sweep"
ef = pd.DataFrame(json.load(open(f"{ROOT}/mirror_full/results/_ef/vggt_noreg_ep300.json"))["per_subject"]).set_index("subject")
d2 = pd.read_csv(f"{ROOT}/es_diag2_noreg.csv").set_index("subj")
d1 = pd.read_csv(f"{ROOT}/es_diag_noreg.csv").set_index("subj")
sur = pd.read_csv(f"{ROOT}/es_blurtest/pred_surplus_analysis.csv").set_index("subj")
blur = pd.read_csv(f"{ROOT}/es_blurtest/blur_esv_results.csv")
j = d2.join(ef, how="inner").dropna(subset=["esv_breath"])
j["seg_dES"] = j.esv_breath - j.esv_gt
j["seg_dED"] = j.edv_breath - j.edv_gt

# ---------------- 01: the puzzle -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4))
for i, (col, lab) in enumerate([("seg_dED", "EDV error (ED)"), ("seg_dES", "ESV error (ES)")]):
    v = j[col]
    ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(v)), v, s=10, c=GRAY, alpha=0.45, edgecolor="none")
    ax.hlines(v.median(), i - 0.22, i + 0.22, color=RED if i else BLUE, lw=3)
    ax.annotate(f"median {v.median():+.1f} ml", (i + 0.26, v.median()), color=RED if i else BLUE, fontweight="bold", va="center")
ax.axhline(0, color="#333", lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["EDV (pred − GT)\nED too SMALL", "ESV (pred − GT)\nES too BIG"])
ax.set_ylabel("volume error, nnU-Net segmentation (ml)"); ax.set_ylim(-80, 80)
ax.set_title("01 — The puzzle (n=143, no-reg): both errors push EF down.\nBut is the +12 ml at ES a real bigger cavity?")
fig.tight_layout(); fig.savefig(f"{OUT}/01_the_puzzle.png", dpi=150); plt.close(fig)

# ---------------- 02: two independent measurements disagree at ES --------------------
fig, ax = plt.subplots(figsize=(5.8, 5))
r, p = stats.spearmanr(j.dES_ml, j.seg_dES)
ax.scatter(j.dES_ml, j.seg_dES, s=14, c=GRAY, alpha=0.6, edgecolor="none")
ax.axhline(0, color="#333", lw=0.8); ax.axvline(0, color="#333", lw=0.8)
ax.axhline(j.seg_dES.median(), color=RED, lw=1.5, ls="--")
ax.axvline(j.dES_ml.median(), color=BLUE, lw=1.5, ls="--")
ax.annotate(f"nnU-Net says +{j.seg_dES.median():.1f} ml", (0.02, 0.95), xycoords="axes fraction", color=RED, fontweight="bold")
ax.annotate(f"pixel count says {j.dES_ml.median():+.1f} ml", (0.02, 0.90), xycoords="axes fraction", color=BLUE, fontweight="bold")
ax.annotate(f"per-subject correlation r={r:+.2f} (p={p:.2f}) — the two\nmeasurements don't even agree on WHO is bad",
            (0.02, 0.80), xycoords="axes fraction", color=INKM)
ax.set_xlabel("ES cavity error measured from PIXELS (bright-blood count, ml)")
ax.set_ylabel("ES cavity error measured by nnU-Net (ml)")
ax.set_xlim(-60, 60); ax.set_ylim(-60, 90)
ax.set_title("02 — Same reconstructions, two rulers.\nIf the cavity were truly big, both axes would rise together.")
fig.tight_layout(); fig.savefig(f"{OUT}/02_seg_vs_pixel_disagree.png", dpi=150); plt.close(fig)

# ---------------- 03: controls -------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
a = axs[0]
dv = d1.dropna(subset=["esv_gt_seg"])
r, _ = stats.spearmanr(dv.pseudoESV_gt, dv.esv_gt_seg)
a.scatter(dv.esv_gt_seg, dv.pseudoESV_gt, s=12, c=GRAY, alpha=0.6, edgecolor="none")
a.set_xlabel("true ESV (GT segmentation, ml)"); a.set_ylabel("pixel-count ESV on GT (ml)")
a.set_title(f"Control 1: the pixel ruler works\n(tracks true ESV on GT, r={r:.2f})")
b = axs[1]
b.hist(d2.dice_bp_ES, bins=25, color=BLUE, alpha=0.85)
b.axvline(0.5, color=RED, lw=1.5, ls="--")
b.annotate("0 subjects below 0.5\n(pred pool didn't drift\nout of the measuring box)", (0.05, 0.75), xycoords="axes fraction", color=INKM)
b.set_xlabel("pred-vs-GT blood-pool overlap (Dice) at ES"); b.set_ylabel("subjects")
b.set_title("Control 2: alignment is fine")
fig.tight_layout(); fig.savefig(f"{OUT}/03_controls.png", dpi=150); plt.close(fig)

# ---------------- 04: where & how bright the surplus is ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
a = axs[0]
a.hist(sur.surplus_bright.clip(-0.2, 1.4), bins=30, color=RED, alpha=0.85)
a.axvline(0, color="#333", lw=1); a.axvline(1, color="#333", lw=1)
a.axvline(sur.inner_bright.median(), color=BLUE, lw=2)
a.annotate("muscle\nlevel", (0, a.get_ylim()[1] * 0.9), ha="center", color=INKM)
a.annotate("blood\nlevel", (1, a.get_ylim()[1] * 0.9), ha="center", color=INKM)
a.annotate(f"agreed cavity\n({sur.inner_bright.median():.2f})", (sur.inner_bright.median(), a.get_ylim()[1] * 0.55), color=BLUE, ha="center")
a.annotate(f"surplus median {sur.surplus_bright.median():.2f}\n= gray, not blood", (sur.surplus_bright.median(), a.get_ylim()[1] * 0.75), color=RED, ha="center")
a.set_xlabel("brightness of the EXTRA 'LV' nnU-Net finds (0=muscle, 1=blood)"); a.set_ylabel("subjects")
a.set_title("04a — The extra 'cavity' is gray tissue-level,\nnot bright blood (92/143 below 0.5)")
b = axs[1]
b.hist(sur.frac_surplus_in_EDlv, bins=25, color=BLUE, alpha=0.85)
b.set_xlabel("fraction of the surplus inside the ED blood footprint"); b.set_ylabel("subjects")
b.set_title("04b — ...and 97% of it sits exactly in the ring\nthe wall contracts into (blood@ED → muscle@ES)")
fig.tight_layout(); fig.savefig(f"{OUT}/04_surplus_is_gray_ring.png", dpi=150); plt.close(fig)

# ---------------- 05: example subject: images + radial profile -----------------------
subj = "MNMs_K5M7V5"
coh = {r["subj"]: r["coh"] for r in csv.DictReader(open(f"{ROOT}/difficulty_covariates.csv"))}[subj]
man = {m["subject"]: m for m in json.load(open(f"{ROOT}/es_blurtest/pred_in/manifest.json"))}
tES, sidx = man[subj]["tES"], man[subj]["sidx"]
d = f"{ROOT}/mirror_full/volumes/{coh}/out/{subj}"
seg4 = np.asarray(nib.load(f"{d}/heart_seg.nii.gz").dataobj)
def norm01(v):
    lo, hi = np.percentile(v[v > 0], [1, 99.5]); return np.clip((v - lo) / (hi - lo), 0, 1)
gt = norm01(np.asarray(nib.load(f"{d}/cine_gt.nii.gz").dataobj).astype(np.float32))[..., tES]
pr = norm01(np.asarray(nib.load(f"{d}/vggt_noreg_ep300/recon_breath/vol_t{tES:02d}.nii.gz").dataobj).astype(np.float32)[..., None])[..., 0]
pseg = np.asarray(nib.load(f"{ROOT}/es_blurtest/pred_seg/{coh}__s{sidx:03d}__pred.nii.gz").dataobj)
gLV = seg4[..., tES] == 1
z = int(np.argmax(gLV.sum((0, 1))))
cy, cx = ndi.center_of_mass(gLV[:, :, z])
req = np.sqrt(gLV[:, :, z].sum() / np.pi)  # equivalent GT LV radius (px)
yy, xx = np.mgrid[0:gt.shape[0], 0:gt.shape[1]]
rad = np.hypot(yy - cy, xx - cx) / req
bins = np.linspace(0, 2.2, 34)
def radprof(img):
    idx = np.digitize(rad.ravel(), bins); v = img[:, :, z].ravel()
    return np.array([v[idx == i].mean() if (idx == i).any() else np.nan for i in range(1, len(bins))])
y0, y1 = int(cy - 45), int(cy + 45); x0, x1 = int(cx - 45), int(cx + 45)
fig = plt.figure(figsize=(10.5, 3.8))
for i, (name, img, ps) in enumerate([("GT at ES", gt, None), ("pred at ES", pr, pseg)]):
    a = fig.add_subplot(1, 3, i + 1); a.grid(False)
    a.imshow(img[y0:y1, x0:x1, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    a.contour(gLV[y0:y1, x0:x1, z].T, levels=[.5], colors=["#00d0ff"], linewidths=1.2)
    if ps is not None:
        a.contour((ps[y0:y1, x0:x1, z] == 1).T, levels=[.5], colors=[RED], linewidths=1.2)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(name + ("\ncyan=GT LV, red=nnU-Net on pred" if i else f"\n{subj}, mid slice"), fontsize=9)
a = fig.add_subplot(1, 3, 3)
a.plot(bins[1:], radprof(gt), color=BLUE, lw=2, label="GT at ES")
a.plot(bins[1:], radprof(pr), color=RED, lw=2, label="pred at ES")
a.axvline(1.0, color="#333", lw=0.8, ls="--")
a.annotate("GT cavity edge", (1.0, 0.95), ha="center", color=INKM, fontsize=8)
a.set_xlabel("distance from LV center (× GT cavity radius)"); a.set_ylabel("brightness (0=dark, 1=blood)")
a.legend(frameon=False); a.set_title("05c — GT drops sharply at the edge;\npred descends slowly = the gray band", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/05_example_and_profile.png", dpi=150); plt.close(fig)

# ---------------- 06: blur control ---------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4))
dm = blur.esv_matched - blur.esv_ctrl; ds = blur.esv_sig10 - blur.esv_ctrl
for i, (v, lab) in enumerate([(dm, "GT blurred to\npred sharpness"), (ds, "GT blurred harder\n(σ=1.0 px)")]):
    ax.scatter(np.random.default_rng(1).normal(i, 0.06, len(v)), v, s=10, c=GRAY, alpha=0.45, edgecolor="none")
    ax.hlines(v.median(), i - 0.22, i + 0.22, color=BLUE, lw=3)
    ax.annotate(f"{v.median():+.1f} ml", (i + 0.26, v.median()), color=BLUE, fontweight="bold", va="center")
ax.axhline(j.seg_dES.median(), color=RED, lw=2, ls="--")
ax.annotate(f"the pred surplus to explain: +{j.seg_dES.median():.1f} ml", (-0.35, j.seg_dES.median() + 1.5), color=RED, fontweight="bold")
ax.axhline(0, color="#333", lw=0.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["GT blurred to\npred sharpness", "GT blurred harder\n(σ=1.0 px)"])
ax.set_ylabel("change in nnU-Net ESV vs unblurred GT (ml)"); ax.set_ylim(-15, 20)
ax.set_title("06 — Causal control: blur alone does NOT fool nnU-Net.\nSo the surplus needs the specific gray band, not smoothness.")
fig.tight_layout(); fig.savefig(f"{OUT}/06_blur_control.png", dpi=150); plt.close(fig)
print("done ->", OUT)
