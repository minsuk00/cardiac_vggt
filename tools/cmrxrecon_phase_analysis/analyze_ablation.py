"""Coverage-ablation analysis: pred-EF-vs-GT-EF slope as a function of how many
input slots observe the target phase (n_forced = 0,1,6,all-z)."""
import json, os, glob, re
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PA = "scratch/phase_analysis"; VOX = 1.4 * 1.4 * 8.0 / 1000.0; T = 12; LV = 1
FORCE = [1, 6, 12]


def lv_curve(seg_dir, subj, suffix):
    c = np.full(T, np.nan)
    for t in range(T):
        f = os.path.join(seg_dir, f"{subj}_t{t:02d}_{suffix}.nii.gz")
        if os.path.exists(f):
            c[t] = (np.asarray(nib.load(f).dataobj) == LV).sum() * VOX
    return c


def ef(c):
    return (c.max() - c.min()) / c.max() * 100 if (not np.isnan(c).any() and (c > 0).all()) else np.nan


# baseline (n_forced=0) + gt from the existing model run
mc = {r["subj"]: r for r in json.load(open(f"{PA}/model_contraction.json"))["rows"]}
subs = sorted(mc)
gt = np.array([mc[s]["gt_ef"] for s in subs])
results = {0: np.array([mc[s]["pred_ef"] for s in subs])}

# forced conditions from ablation_segs
for nf in FORCE:
    pe = []
    for s in subs:
        c = lv_curve(f"{PA}/ablation_segs", s, f"f{nf:02d}_pred")
        pe.append(ef(c))
    results[nf] = np.array(pe)

print(f"{'n_forced':>8s} {'slope':>7s} {'r':>6s} {'pred_EF_mean':>12s}   (slope 1=recovers per-patient, 0=flat)")
slopes = {}
for nf in [0] + FORCE:
    pe = results[nf]; m = ~np.isnan(pe)
    if m.sum() < 3:
        print(f"{nf:>8d}  insufficient"); continue
    sl, b = np.polyfit(gt[m], pe[m], 1); r = np.corrcoef(gt[m], pe[m])[0, 1]
    slopes[nf] = sl
    tag = "  (all-z = full target observation)" if nf == 12 else ("  (current/baseline)" if nf == 0 else "")
    print(f"{nf:>8d} {sl:>7.2f} {r:>6.2f} {np.nanmean(pe):>12.1f}{tag}")

# plot slope vs n_forced
xs = sorted(slopes); ys = [slopes[x] for x in xs]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(xs, ys, "-o", color="#cc3311", lw=2, ms=8)
ax.axhline(1.0, color="green", ls=":", label="perfect (recovers per-patient)")
ax.axhline(0.0, color="gray", ls=":", label="flat (regress-to-mean)")
ax.set_xlabel("n_forced = # input slots forced to OBSERVE the target phase")
ax.set_ylabel("pred-EF vs true-EF slope")
ax.set_title("Real model: does observing the target phase restore per-patient EF?")
ax.set_xticks(xs); ax.set_ylim(-0.2, 1.1); ax.legend(fontsize=8); fig.tight_layout()
os.makedirs(f"{PA}/figs", exist_ok=True)
fig.savefig(f"{PA}/figs/ablation_slope.png", dpi=120)
json.dump({str(k): float(v) for k, v in slopes.items()}, open(f"{PA}/ablation_slopes.json", "w"), indent=2)
print(f"\nplot -> {PA}/figs/ablation_slope.png")
