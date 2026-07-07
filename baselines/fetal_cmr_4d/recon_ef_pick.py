"""Aggregate nnU-Net Task114 segs of a recon's (phase, slice) grid -> LV volume vs cardiac phase
-> EF. LV = label 1. EF = (EDV - ESV) / EDV (voxel volume cancels in the ratio).

Run: micromamba run -n svr python baselines/fetal_cmr_4d/recon_ef_pick.py <ef_segs_dir> [label]
"""
import os, sys, glob, re, json
import numpy as np
import nibabel as nib

LV = 1


def main():
    segdir = sys.argv[1]
    lvvol = {}   # phase -> summed LV voxels over slices
    for s in glob.glob(os.path.join(segdir, "rec__p*__z*.nii.gz")):
        m = re.search(r"__p(\d+)__z(\d+)", os.path.basename(s))
        p = int(m.group(1))
        lv = int((np.asarray(nib.load(s).dataobj) == LV).sum())
        lvvol[p] = lvvol.get(p, 0) + lv
    P = max(lvvol) + 1
    vol = np.array([lvvol.get(p, 0) for p in range(P)], float)
    edv, esv = vol.max(), vol.min()
    ef = (edv - esv) / edv * 100 if edv > 0 else 0.0
    ed_p, es_p = int(vol.argmax()), int(vol.argmin())
    print(f"LV volume vs phase (voxels): {vol.astype(int).tolist()}")
    print(f"EDV={edv:.0f} (phase {ed_p})  ESV={esv:.0f} (phase {es_p})  temporal_contrast={(edv-esv)/edv:.3f}")
    print(f"EF = {ef:.1f}%")
    out = os.path.join(os.path.dirname(segdir.rstrip("/")), "recon_ef.json")
    json.dump({"lv_volume_by_phase": vol.tolist(), "EDV": edv, "ESV": esv,
               "ed_phase": ed_p, "es_phase": es_p, "EF_percent": ef}, open(out, "w"), indent=2)
    # plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(vol, "-o", ms=4); ax.axvline(ed_p, color="tab:green", ls="--", lw=1, label=f"ED (p{ed_p})")
    ax.axvline(es_p, color="tab:red", ls="--", lw=1, label=f"ES (p{es_p})")
    ax.set_xlabel("cardiac phase"); ax.set_ylabel("LV blood-pool volume (voxels)")
    ax.set_title(f"Self-gated fetal_cmr_4d recon (V1, 3mm test): EF = {ef:.1f}%"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(segdir.rstrip("/")), "recon_ef.png"), dpi=100)
    print(f"-> recon_ef.json + recon_ef.png")


if __name__ == "__main__":
    main()
