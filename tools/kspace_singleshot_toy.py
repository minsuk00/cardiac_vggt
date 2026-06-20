"""Toy: what does a TRUE single-shot real-time input look like vs the clean cine we train on?

The project trains on clean gated cine slices (and the OCMR demo used k-t CS-SENSE recon =
many frames/slice). A real one-frame-per-slice real-time acquisition gives a SINGLE-SHOT,
prospectively-undersampled frame: R-fold accelerated Cartesian k-space, zero-filled (or lightly
regularized) → residual aliasing + lower SNR. This script simulates that on clean canonical
slices to (a) visualize the domain gap the model has never seen, and (b) [GPU pass] feed clean
vs aliased slices to the trained model and measure the V_canon motion-PSNR drop = brittleness.

CPU demo (default): writes result/limits_eval/kspace_singleshot.png + metrics json.
GPU model pass (--model): also runs the joint model on clean vs aliased inputs.

Run: micromamba run -n svr python tools/kspace_singleshot_toy.py [--model]
"""
import argparse, json, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, REPO)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(REPO, "result", "limits_eval")


def vista_mask(H, W, R, acs=16, seed=0):
    """1-D Cartesian undersampling mask along phase-encode (H): keep ACS center + ~H/R lines."""
    rng = np.random.RandomState(seed)
    keep = np.zeros(H, dtype=bool)
    c0, c1 = H // 2 - acs // 2, H // 2 + acs // 2
    keep[c0:c1] = True
    n_target = max(1, H // R)
    remaining = n_target - keep.sum()
    if remaining > 0:
        cand = np.where(~keep)[0]
        # variable density: prefer lines nearer the center
        w = 1.0 / (1.0 + np.abs(cand - H / 2) / (H / 6))
        w = w / w.sum()
        pick = rng.choice(cand, size=min(remaining, len(cand)), replace=False, p=w)
        keep[pick] = True
    return keep, keep.sum() / H


def single_shot(img, R, acs=16, seed=0, noise=0.02):
    """img: (H,W) magnitude in [0,1] → simulated zero-filled single-shot reconstruction."""
    H, W = img.shape
    k = np.fft.fftshift(np.fft.fft2(img))
    keep, frac = vista_mask(H, W, R, acs, seed)
    k_us = k * keep[:, None]
    # add complex Gaussian noise on the sampled lines (single-shot SNR penalty).
    # Scale by MEAN magnitude (not the DC peak) so noise is a small per-line perturbation
    # and the R-dependence is driven by aliasing/blur, not by an exploding noise floor.
    rng = np.random.RandomState(seed + 1)
    nstd = noise * np.abs(k).mean()
    k_us = k_us + (rng.randn(*k.shape) + 1j * rng.randn(*k.shape)) * nstd * keep[:, None]
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(k_us)))
    return recon, frac


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="also run the joint model on clean vs aliased")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
    from data.gpu_aug import gpu_augment_batch
    from data.respiratory import RespiratoryConfig
    from loss import compute_volume_intensity_loss, compute_motion_mask
    ds = build_dataset()

    Rs = [4, 8, 12]
    # ---- CPU demo: aliasing on representative mid-heart slices ----
    data = ds.get_data(seq_index=7, img_per_seq=NUM_SLICES)
    import numpy as _np
    imgs = _np.stack(data["images"]).astype(_np.float32) / 255.0   # (S,H,W,3) 518
    # take a single representative input slice (mid), use its grayscale
    sl = imgs[len(imgs) // 2, :, :, 0]
    # work at canonical 256 to mirror the actual data resolution
    import torch, torch.nn.functional as F
    sl256 = F.interpolate(torch.from_numpy(sl)[None, None], size=(256, 256),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
    demo = {}
    fig, axes = plt.subplots(1, len(Rs) + 1, figsize=(4 * (len(Rs) + 1), 4))
    axes[0].imshow(sl256, cmap="gray", vmin=0, vmax=1); axes[0].set_title("clean cine (training input)")
    axes[0].axis("off")
    for i, R in enumerate(Rs):
        rec, frac = single_shot(sl256, R, seed=3)
        p = psnr(rec, sl256)
        demo[f"R{R}"] = dict(psnr=p, sampled_frac=float(frac))
        axes[i + 1].imshow(rec, cmap="gray", vmin=0, vmax=1)
        axes[i + 1].set_title(f"single-shot R={R}\n({frac*100:.0f}% lines, PSNR {p:.1f} dB)")
        axes[i + 1].axis("off")
    fig.suptitle("Domain gap: clean gated cine (trained on) vs simulated single-shot real-time input", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "kspace_singleshot.png"), bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    print("clean-vs-aliased slice PSNR:", {k: round(v["psnr"], 2) for k, v in demo.items()})

    result = {"slice_demo": demo}

    # ---- GPU model pass: does the model break on aliased input? ----
    if args.model:
        import torch
        from eval_refiner import make_refiner_model
        device = "cuda"
        JOINT = os.path.join(REPO, "scratch", "logs", "218349151_mri_refiner_joint", "ckpts", "checkpoint_last.pt")
        model, _ = make_refiner_model(JOINT, device)
        cfg = RespiratoryConfig(enable=True, amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3,
                                ap_ratio=0.35, ap_axis="H", per_slot=True, direction_jitter_deg=30.0)
        for R in [8]:
            drops = []
            for seq in range(args.n):
                data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
                # CLEAN
                b = build_batch(data, device, seq_index=seq)
                b = gpu_augment_batch(b, None, device, respiratory_cfg=cfg, train=False)
                outg = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b,
                                                     grid_shape=GRID_SHAPE, tv_weight=0.0)
                Vg = outg["V_gt"][0].float().cpu().numpy()
                mm = compute_motion_mask(b["phases"])[0].cpu().numpy()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    pc = model(b["images"], batch=b)
                p_clean = psnr(pc["V_refined"][0].float().cpu().numpy()[mm], Vg[mm])
                # ALIASED: undersample each input slice's k-space (per-slot seed)
                imgs = b["images"]                       # (1,S,3,518,518)
                S = imgs.shape[1]
                aliased = imgs.clone()
                for s in range(S):
                    g = imgs[0, s, 0].cpu().numpy()
                    rec, _ = single_shot(g, R, seed=100 + s)
                    aliased[0, s] = torch.from_numpy(rec).to(device).repeat(3, 1, 1)
                b2 = dict(b); b2["images"] = aliased
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    pa = model(aliased, batch=b2)
                p_alias = psnr(pa["V_refined"][0].float().cpu().numpy()[mm], Vg[mm])
                drops.append((p_clean, p_alias))
            arr = np.array(drops)
            result[f"model_R{R}"] = dict(
                motion_clean=float(arr[:, 0].mean()), motion_aliased=float(arr[:, 1].mean()),
                drop=float((arr[:, 0] - arr[:, 1]).mean()), n=len(drops))
            print(f"R={R}: model motion PSNR clean={arr[:,0].mean():.2f} aliased={arr[:,1].mean():.2f} "
                  f"drop={(arr[:,0]-arr[:,1]).mean():+.2f} dB")

    json.dump(result, open(os.path.join(OUT, "kspace_singleshot.json"), "w"), indent=2)
    print("Wrote", os.path.join(OUT, "kspace_singleshot.json"))


if __name__ == "__main__":
    main()
