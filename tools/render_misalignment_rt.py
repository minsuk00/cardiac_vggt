"""Real-time free-breathing MIITT: long-axis reslices of the real RT stack vs our recon.

Columns: model input at query frame f | ours at frame f, for several f.  No GT exists (docs/87).
The model input is EXACTLY what run_vggt_rt fed: one fixed real frame per companion plane (the
subject-seeded draw replayed through the same MRIDataset scaffold), the mid plane at frame f.

  python tools/render_misalignment_rt.py --subject MIITT_Volunteer1 --arm vggt_augaggr224_ep300 \
      --frames 0 45 90 135 --out temp/misalign/rt/Volunteer1.png
"""
import argparse, os
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def replay_draw(subject, D, n_frames, T_gated=12):
    """The companion draw run_vggt_rt fed, replayed WITHOUT torch: MRIDataset.get_data's val
    branch with reference_slot + one_frame_per_slice + t_target_fixed=0, seeded by name_seed
    (sha256 of 'miitt/<subject>'), bbox z-range [0, D). Returns (frame index per plane z, ref z)."""
    import random, hashlib
    rng = random.Random(int(hashlib.sha256(f"miitt/{subject}".encode()).hexdigest(), 16) % (2 ** 31))
    z_mid = D // 2
    tail = [z for z in range(D) if z != z_mid]; rng.shuffle(tail)
    z_seq = [z_mid] + tail
    t_seq = [rng.randrange(T_gated) for _ in range(D)]; t_seq[0] = 0
    frame_of_t = np.round(np.linspace(0, n_frames - 1, T_gated)).astype(int)
    frame_of_z = np.empty(D, int); frame_of_z[z_seq] = frame_of_t[t_seq]
    return frame_of_z, z_mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 45, 90, 135])
    ap.add_argument("--out", required=True); ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--center", type=int, nargs=2, help="override (y, x) reslice centre")
    a = ap.parse_args()
    sd = os.path.join(ROOT, "evaluation/volumes/miitt_rt/out", a.subject)
    rt = nib.load(f"{sd}/rt_input.nii.gz"); dz = float(rt.header.get_zooms()[2]); inpl = float(rt.header.get_zooms()[0])
    rt = np.asarray(rt.dataobj, dtype=np.float32).transpose(3, 2, 1, 0)                     # (F,D,H,W)
    ours = np.asarray(nib.load(f"{sd}/{a.arm}/recon_rt.nii.gz").dataobj, dtype=np.float32).transpose(3, 2, 1, 0)
    F, D, H, W = rt.shape
    frame_of_z, ref_z = replay_draw(a.subject, D, F)
    fixed = rt[frame_of_z, np.arange(D)]                                                 # (D,H,W)
    def model_input(f):
        m = fixed.copy(); m[ref_z] = rt[f, ref_z]; return m
    if a.center:
        yc, xc = a.center
    else:   # gated bundle's heart mask, same subject, same canonical in-plane grid
        hm = np.asarray(nib.load(os.path.join(ROOT, "evaluation/volumes/miitt/out", a.subject, "mask_heart.nii.gz")).dataobj) > 0
        _, yc, xc = [int(round(v)) for v in np.argwhere(hm.transpose(2, 1, 0)).mean(0)]
    pad = 48; y0, y1 = max(yc - pad, 0), min(yc + pad, H); x0, x1 = max(xc - pad, 0), min(xc + pad, W)
    vmax = np.percentile(rt, 99.5); asp = dz / inpl
    cols = [(f"model input\nquery frame {f}", model_input(f)) for f in a.frames] + [(f"ours\nframe {f}", ours[f]) for f in a.frames]
    fig, ax = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 2 * 2.2), squeeze=False)
    for j, (name, v) in enumerate(cols):
        v = np.clip(v / vmax, 0, 1) ** a.gamma
        for i, (img, lab) in enumerate([(v[:, yc, x0:x1], f"x–z reslice, y={yc}"), (v[:, y0:y1, xc], f"y–z reslice, x={xc}")]):
            ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=1, aspect=asp, interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if j == 0: ax[i, j].set_ylabel(lab)
            if i == 0: ax[i, j].set_title(name, fontsize=9)
    fig.suptitle(f"{a.subject}  real RT free-breathing  D={D} dz={dz}mm  {F} frames  arm={a.arm}  "
                 f"companion frame per z: {frame_of_z.tolist()}  ref z={ref_z}", fontsize=8)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); fig.tight_layout(); fig.savefig(a.out, dpi=150); print("wrote", a.out)


if __name__ == "__main__":
    main()
