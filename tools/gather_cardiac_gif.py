"""Beating-heart cardiac-cycle GIFs: GT vs control (gw=0) vs gather (gw=0.5), a few val subjects.

Reconstructs the full cardiac cycle at the INFERENCE regime (reference plane = all phases, others =
5-frame burst; reconstruct_cycle) with the simulated respiratory corruption ON (the realistic
deployment task — the model must correct breathing). 3 rows (GT / control / gather) × n z-planes
spanning the anatomy bbox, animated over the 12 cardiac phases. -> one GIF per subject.

Run: micromamba run -n svr python tools/gather_cardiac_gif.py \
       --gather <gather.pt> --control <control.pt> --subjects 0 7 20 --out _html/gather_gifs
"""
import argparse, os, sys
import numpy as np, torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "training"))
from inference.inference import load_rtfb_model_reference
from inference.run_cmrxrecon import build_mri_dataset, reconstruct_cycle, _planes_across_bbox, _fig_to_pil


def three_row_gif(gt, pred_c, pred_g, bbox, path, n_slices=5, breathing=True):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, D = gt.shape[0], gt.shape[1]
    planes = _planes_across_bbox(bbox, D, n_slices)
    vmax = float(max(gt[:, planes].max(), pred_c[:, planes].max(), pred_g[:, planes].max(), 1e-3))
    rows = [(gt, "GT"), (pred_c, "control gw=0"), (pred_g, "gather gw=0.5")]
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(3, len(planes), figsize=(1.6 * len(planes), 5.0), squeeze=False)
        for r, (vol, name) in enumerate(rows):
            for c, z in enumerate(planes):
                ax = axes[r][c]; ax.imshow(vol[t, z], cmap="gray", vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0: ax.set_title(f"z={z}", fontsize=8)
                if c == 0: ax.set_ylabel(name, fontsize=10)
        fig.suptitle(f"phase t={t}   (breathing {'ON' if breathing else 'OFF'}, inference regime)", fontsize=9)
        fig.tight_layout(); frames.append(_fig_to_pil(fig)); plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gather", required=True); ap.add_argument("--control", required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=[0, 7, 20])
    ap.add_argument("--frames_per_slice", type=int, default=5)
    ap.add_argument("--breathing", action="store_true", default=True)
    ap.add_argument("--out", default=os.path.join(_ROOT, "_html", "gather_gifs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    mri_ds, rcfg = build_mri_dataset()
    print("loading models...", flush=True)
    mG = load_rtfb_model_reference(args.gather, device=dev)
    mC = load_rtfb_model_reference(args.control, device=dev)
    print("models loaded", flush=True)
    out_paths = []
    for seq in args.subjects:
        rg = reconstruct_cycle(mG, mri_ds, rcfg, seq, args.breathing, dev, args.frames_per_slice)
        rc = reconstruct_cycle(mC, mri_ds, rcfg, seq, args.breathing, dev, args.frames_per_slice)
        gt = np.asarray(rg["gt_vols"]); pg = np.asarray(rg["pred_vols"]); pc = np.asarray(rc["pred_vols"])
        subj = os.path.basename(str(mri_ds.subjects[seq])) if seq < len(mri_ds.subjects) else str(seq)
        path = os.path.join(args.out, f"cardiac_gif_seq{seq}_{subj}.gif")
        three_row_gif(gt, pc, pg, rg["bbox"], path, breathing=args.breathing)
        mm_psnr_g = np.nanmean(rg["metrics"]["motion"]); mm_psnr_c = np.nanmean(rc["metrics"]["motion"])
        print(f"seq {seq} ({subj}): motion PSNR gather={mm_psnr_g:.2f} control={mm_psnr_c:.2f}  -> {path}", flush=True)
        out_paths.append(path)
    print("GIFS:", *out_paths, sep="\n")


if __name__ == "__main__":
    main()
