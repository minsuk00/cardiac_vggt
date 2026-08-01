"""Which way does the simulated breath actually move the heart? Render it and look.

Provenance for docs/58 §10a. The physiological requirement is fixed: the diaphragm DESCENDS on
inspiration and the heart follows it INFERIORLY (docs/01), returning to rest at end-expiration.
`respiratory.py`'s Lujan waveform is one-sided (`A·sin^{2n}(πr)`, >= 0, zero at end-expiration),
so the simulated heart always moves in ONE direction along the array's D axis as the breath
deepens. Whether that direction is inferior depends entirely on which anatomical end sits at D=0 --
and `respiratory.py` has no anatomical anchor of any kind, so nothing in the code checks.

This script drives the REAL `reslice_volume_vec` (the same function the training path uses via
`extract_slices_with_respiratory_vec`) over a full breath cycle and renders the long-axis side
view, so the direction can be read off directly instead of derived from the sign convention.

Each GIF is annotated with that subject's measured slice order and the resulting verdict:

    heart moves toward D=0.  D=0 is the APEX -> motion is INFERIOR -> physiological  OK
    heart moves toward D=0.  D=0 is the BASE -> motion is SUPERIOR -> backwards      WRONG

Usage:
    python tools/render_respiratory_direction.py                    # one subject per source
    python tools/render_respiratory_direction.py --subjects ACDC_patient011,MNMs_E3TQZ2
    python tools/render_respiratory_direction.py --amplitude 25

Outputs -> result/respiratory_direction/  (one GIF + one static filmstrip per subject)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT = "/home/minsukc/vggt"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from data.respiratory import (  # noqa: E402
    RespiratoryConfig, lujan_displacement, reslice_volume_vec,
    _build_disp_dhw, _rotate_disp,
)
from tools.render_slice_order_check import features, LV, HALF_WINDOW_MM  # noqa: E402

DATA = os.path.join(ROOT, "scratch/data")
OUT = os.path.join(ROOT, "result/respiratory_direction")

DEFAULTS = [
    os.path.join(DATA, "CMRxRecon2024/Cine_combined/CMRx24_Train_P197/sax"),
    os.path.join(DATA, "ACDC_sax/ACDC_patient011/sax"),
    os.path.join(DATA, "MNMs_sax/MNMs_I7T3U1/sax"),
    # a CMRx subject whose stack runs the OTHER way -- the ordering is not a per-source constant
    os.path.join(DATA, "CMRxRecon2025/Cine_combined/CMRx25_train_Center001_UIH_30T_umr780_P030/sax"),
]


def load_subject(sax_dir):
    """-> (V (D,H,W) torch, spacing (dz,dy,dx), feat, subject)"""
    subj = os.path.basename(os.path.dirname(sax_dir))
    im = nib.load(os.path.join(sax_dir, "3d_recon", "sax_frame_00.nii.gz"))
    vol = np.asarray(im.dataobj).astype(np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    dx, dy, dz = [float(v) for v in im.header.get_zooms()[:3]]
    # (X,Y,Z) on disk -> (D,H,W) = (Z,Y,X), the order the splat/respiratory code consumes.
    V = torch.from_numpy(np.ascontiguousarray(vol.transpose(2, 1, 0)))
    feat = features(os.path.join(sax_dir, "heart_seg.nii.gz"))
    return V, (dz, dy, dx), feat, subj


def live_config():
    """RespiratoryConfig exactly as training uses it (from the live mri_volume.yaml)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    for n, f in (("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
                 ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")):
        OmegaConf.register_new_resolver(n, f, replace=True)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(ROOT, "training", "config")):
        cfg = compose(config_name="default")
    return RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)


def real_disp_trajectory(cfg, n_frames, seed):
    """The FULL displacement vector the trainer would apply, swept over one breath.

    Reproduces `sample_displacement_vectors`'s composition exactly: SI from the Lujan waveform,
    AP = ap_ratio * SI, stacked into (d_D, d_H, d_W), then Rodrigues-tilted by a per-SUBJECT
    (theta, phi). theta/phi/amplitude are drawn ONCE per subject (fixed acquisition geometry +
    one lung capacity) and only the breath PHASE r varies -- matching group_by_burst training.
    -> (list of (3,) mm tensors, theta_deg, A_mm)
    """
    g = torch.Generator().manual_seed(int(seed))
    u = lambda: torch.rand(1, generator=g)
    tmin = cfg.tilt_min_deg or 0.0
    tmax = cfg.tilt_max_deg if cfg.tilt_max_deg is not None else cfg.direction_jitter_deg
    theta = torch.deg2rad(tmin + u() * (tmax - tmin))          # per-subject tilt
    phi = u() * 2.0 * torch.pi                                  # per-subject azimuth
    A = (cfg.amplitude_mm + (u() * 2.0 - 1.0) * cfg.amplitude_jitter).clamp_min(0.0)
    out = []
    for r in np.linspace(0.0, 1.0, n_frames, endpoint=False):
        d_si = lujan_displacement(float(r), A, n=cfg.cos2n)     # (1,)
        d_ap = cfg.ap_ratio * d_si
        v = _build_disp_dhw(d_si, d_ap, cfg.ap_axis)            # (1,3) = (d_D,d_H,d_W)
        v = _rotate_disp(v, theta, phi)                         # rigid tilt, magnitude preserved
        out.append(v.reshape(3))
    return out, float(torch.rad2deg(theta)), float(A)


def short_axis_view(V, z_index, cy, cx, half):
    """(D,H,W) -> in-plane short-axis image at plane z, cropped around the LV."""
    sl = V[z_index].numpy()                                     # (H,W)
    y0, y1 = max(0, cy - half), min(sl.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(sl.shape[1], cx + half)
    return sl[y0:y1, x0:x1]


def side_view(V, h_index, half_rows):
    """(D,H,W) -> long-axis image with z horizontal: (W_crop, D)."""
    sl = V[:, h_index, :].numpy()                    # (D, W)
    w0 = max(0, half_rows[0])
    w1 = min(sl.shape[1], half_rows[1])
    return sl[:, w0:w1].T                            # (W_crop, D)


def render(sax_dir, amplitude_mm, cos2n, n_frames, fps, real_cfg=None):
    import imageio.v2 as imageio

    V, (dz, dy, dx), feat, subj = load_subject(sax_dir)
    if feat is None:
        print(f"  {subj}: no usable segmentation, skipped")
        return None

    seg = feat["seg"]                                 # (X,Y,Z)
    mask = seg == LV
    if mask.sum() == 0:
        mask = seg > 0
    cx, cy = [int(round(c)) for c in np.array(np.nonzero(mask))[:2].mean(axis=1)]
    D, H, W = V.shape
    cy = int(np.clip(cy, 0, H - 1))
    half = int(round(HALF_WINDOW_MM / dx))
    win = (cx - half, cx + half)

    order = feat["order"]                             # 'apex-first' | 'base-first'
    d0_is_apex = order == "apex-first"
    verdict = ("motion is INFERIOR  ->  PHYSIOLOGICAL"
               if d0_is_apex else "motion is SUPERIOR  ->  BACKWARDS")
    verdict_color = "#2ecc71" if d0_is_apex else "#e74c3c"

    rs = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    if real_cfg is not None:
        vecs, theta_deg, A_mm = real_disp_trajectory(real_cfg, n_frames, abs(hash(subj)) % (2**31))
        disps = [float(v[0]) for v in vecs]                     # D-component, for the readout
        mode = (f"REAL sim: tilt={theta_deg:.0f}deg  A={A_mm:.1f}mm  "
                f"ap_ratio={real_cfg.ap_ratio}  ap_axis={real_cfg.ap_axis}")
    else:
        vecs = [torch.tensor([float(lujan_displacement(float(r), amplitude_mm, n=cos2n)), 0.0, 0.0])
                for r in rs]
        disps = [float(v[0]) for v in vecs]
        mode = f"REDUCED: pure SI (tilt=0, no AP)  A={amplitude_mm}mm"

    ref = side_view(V, cy, win)
    lo, hi = np.percentile(ref[ref > 0], [1, 99.5]) if (ref > 0).any() else (0.0, 1.0)
    extent = [0, D * dz, 0, ref.shape[0] * dx]

    # reference marker: LV centroid in z (mm) at rest, so the shift is visible against a fixed line
    z_centroid_mm = float(np.array(np.nonzero(mask))[2].mean()) * dz

    # short-axis reference plane (mid-LV) + a fixed crosshair to judge in-plane motion against
    zc = int(round(float(np.array(np.nonzero(mask))[2].mean())))
    zc = int(np.clip(zc, 0, D - 1))
    sa_half = int(round(60.0 / dx))
    sa_ref = short_axis_view(V, zc, cy, cx, sa_half)
    slo, shi = np.percentile(sa_ref[sa_ref > 0], [1, 99.5]) if (sa_ref > 0).any() else (0.0, 1.0)

    os.makedirs(OUT, exist_ok=True)
    frames = []
    for r, v in zip(rs, vecs):
        d = float(v[0])
        shifted = reslice_volume_vec(V, tuple(float(x) for x in v), spacing=(dz, dy, dx))
        img = side_view(shifted, cy, win)
        sa = short_axis_view(shifted, zc, cy, cx, sa_half)

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=110,
                                 gridspec_kw={"width_ratios": [1.55, 1.0]})
        ax = axes[0]
        ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray",
                  origin="lower", extent=extent, aspect="equal", vmin=0, vmax=1)
        ax.axvline(z_centroid_mm, color="#f1c40f", ls="--", lw=1.0, alpha=0.9)
        left, right = ("APEX", "BASE") if d0_is_apex else ("BASE", "APEX")
        ax.annotate(left, xy=(0.02, 0.94), xycoords="axes fraction", color="yellow",
                    fontsize=11, weight="bold")
        ax.annotate(right, xy=(0.98, 0.94), xycoords="axes fraction", color="yellow",
                    fontsize=11, weight="bold", ha="right")
        # content is sampled from z+d, so it APPEARS to move toward LOWER z
        ax.annotate("", xy=(0.30, 0.08), xytext=(0.52, 0.08), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#00e5ff", lw=2.2))
        ax.annotate("through-plane (SI)", xy=(0.54, 0.065), xycoords="axes fraction",
                    color="#00e5ff", fontsize=8)
        ax.set_xlabel("z (mm)   —  LONG-AXIS side view", fontsize=9)
        ax.tick_params(labelsize=7)

        ax2 = axes[1]
        ax2.imshow(np.clip((sa - slo) / max(shi - slo, 1e-6), 0, 1), cmap="gray",
                   origin="lower", aspect="equal", vmin=0, vmax=1)
        # fixed crosshair at the rest-position LV centre: in-plane drift shows as motion off it
        ax2.axhline(sa.shape[0] / 2, color="#f1c40f", ls="--", lw=0.8, alpha=0.8)
        ax2.axvline(sa.shape[1] / 2, color="#f1c40f", ls="--", lw=0.8, alpha=0.8)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlabel(f"SHORT-AXIS at z={zc}  —  in-plane (AP) component", fontsize=9)

        fig.suptitle(f"{subj}   ({order}, dz={dz:.1f}mm)   |   {mode}\n"
                     f"r={r:.2f}   d_DHW = ({v[0]:+.1f}, {v[1]:+.1f}, {v[2]:+.1f}) mm   |   {verdict}",
                     fontsize=9.5, color=verdict_color)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    tag = "_real" if real_cfg is not None else ""
    gif = os.path.join(OUT, f"{subj}_breath{tag}.gif")
    imageio.mimsave(gif, frames, duration=1.0 / fps, loop=0)

    # static filmstrip at rest / mid / peak, for the doc
    picks = [0, n_frames // 4, n_frames // 2]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.4 * len(picks), 4.0))
    for ax, k in zip(np.atleast_1d(axes), picks):
        ax.imshow(frames[k])
        ax.axis("off")
    fig.suptitle(f"{subj} — end-expiration -> peak inspiration", fontsize=11)
    fig.tight_layout()
    strip = os.path.join(OUT, f"{subj}_filmstrip{tag}.png")
    fig.savefig(strip, dpi=110)
    plt.close(fig)

    print(f"  {subj:28s} {order:11s}  peak d={max(disps):.1f}mm  -> {verdict}")
    return gif, strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma-separated subject ids")
    ap.add_argument("--amplitude", type=float, default=18.8, help="SI breath depth (mm)")
    ap.add_argument("--cos2n", type=int, default=3)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--real", action="store_true",
                    help="use the ACTUAL trainer displacement (per-subject tilt + AP + "
                         "amplitude from the live mri_volume.yaml) instead of pure SI")
    args = ap.parse_args()

    if args.subjects:
        dirs = []
        for s in [x.strip() for x in args.subjects.split(",") if x.strip()]:
            hits = glob.glob(os.path.join(DATA, "*", "*", s, "sax")) + \
                   glob.glob(os.path.join(DATA, "*", s, "sax"))
            if not hits:
                print(f"  {s}: not found")
                continue
            dirs.append(hits[0])
    else:
        dirs = DEFAULTS

    rcfg = live_config() if args.real else None
    if rcfg is not None:
        print(f"REAL sim from mri_volume.yaml: amp={rcfg.amplitude_mm}+-{rcfg.amplitude_jitter}mm  "
              f"cos2n={rcfg.cos2n}  ap_ratio={rcfg.ap_ratio} ap_axis={rcfg.ap_axis}  "
              f"tilt=({rcfg.tilt_min_deg},{rcfg.tilt_max_deg})deg  burst={rcfg.group_by_burst}")
    else:
        print(f"REDUCED: amplitude={args.amplitude}mm  sin^{2*args.cos2n}  tilt=0 (pure SI)")
    for d in dirs:
        render(d, args.amplitude, args.cos2n, args.frames, args.fps, real_cfg=rcfg)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
