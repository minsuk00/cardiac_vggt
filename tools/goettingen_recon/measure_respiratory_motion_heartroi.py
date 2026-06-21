#!/usr/bin/env python
"""Heart-ROI respiratory motion — isolate the heart, drop static background.

The whole-FOV estimate (measure_respiratory_motion.py) mixes the moving heart with static
structures (spine, posterior wall), diluting the heart's true AP excursion. Here we restrict the
tracking to a heart-centered ROI.

Per SAX slice (an independent ~4.26 s RT free-breathing cine, 127 frames):
  1. Localize the heart by per-pixel CARDIAC-band power (0.7-2.0 Hz): the myocardium/blood pool
     pulsates -> high cardiac variance. Largest high-power connected component -> heart mask + bbox.
  2. Track in-plane rigid translation of the heart ROI vs its temporal mean (subpixel phase
     cross-correlation on the cropped region).
  3. Band-separate the trajectory -> respiratory (0.1-0.5 Hz); PCA -> principal (~AP) axis + p2p amp.
  4. Negative control: shuffle frames -> the slow drift must collapse.
Reports heart-ROI AP per slice and aggregate, alongside the whole-FOV AP for comparison.
"""
import argparse, json
import numpy as np
import nibabel as nib
from skimage.registration import phase_cross_correlation
from scipy.signal import butter, filtfilt
from scipy.ndimage import label


def bandpass(x, lo, hi, fs):
    ny = 0.5 * fs
    hi = min(hi, ny * 0.99)
    b, a = butter(3, [lo / ny, hi / ny], btype='band')
    return filtfilt(b, a, x, axis=0)


def cardiac_power(cine, fs, card):
    """Per-pixel cardiac-band temporal variance -> heart localizer."""
    T = cine.shape[-1]
    f = bandpass(cine.transpose(2, 0, 1).reshape(T, -1), *card, fs)   # (T, X*Y)
    return f.var(0).reshape(cine.shape[:2])


def heart_roi(cine, fs, card, pct=92, margin=8):
    P = cardiac_power(cine, fs, card)
    mask = P > np.percentile(P, pct)
    lab, n = label(mask)
    H, W = cine.shape[:2]
    if n == 0:
        return (H // 4, 3 * H // 4, W // 4, 3 * W // 4), P, (H // 2, W // 2)
    big = 1 + int(np.argmax([(lab == i).sum() for i in range(1, n + 1)]))
    ys, xs = np.where(lab == big)
    y0, y1 = max(0, ys.min() - margin), min(H, ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(W, xs.max() + margin)
    return (y0, y1, x0, x1), P, (int(ys.mean()), int(xs.mean()))


def track(cine, ref):
    sh = np.zeros((cine.shape[-1], 2))
    for t in range(cine.shape[-1]):
        s, _, _ = phase_cross_correlation(ref, cine[..., t], upsample_factor=20, normalization=None)
        sh[t] = s
    return sh - sh.mean(0)


def ap_from(sh_mm, fs, resp):
    r = bandpass(sh_mm, *resp, fs)
    vt = np.linalg.svd(r - r.mean(0), full_matrices=False)[2]
    proj = r @ vt[0]
    return proj, float(proj.max() - proj.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nii'); ap.add_argument('out_prefix')
    ap.add_argument('--dt', type=float, default=0.03354)
    ap.add_argument('--px', type=float, default=1.6)
    ap.add_argument('--resp', type=float, nargs=2, default=[0.1, 0.5])
    ap.add_argument('--card', type=float, nargs=2, default=[0.7, 2.0])
    args = ap.parse_args()
    fs = 1.0 / args.dt

    v = nib.load(args.nii).get_fdata()
    X, Y, Z, T = v.shape
    rng = np.random.default_rng(0)
    rows = []
    for z in range(Z):
        cine = v[:, :, z, :].astype(np.float32)
        if cine.max() <= 0:
            continue
        cine = cine / np.percentile(cine, 99.5)

        # whole-FOV (for comparison)
        sh_full = track(cine, cine.mean(-1)) * args.px
        _, ap_full = ap_from(sh_full, fs, args.resp)

        # heart ROI
        (y0, y1, x0, x1), P, (cy, cx) = heart_roi(cine, fs, args.card)
        roi = cine[y0:y1, x0:x1, :]
        sh_roi = track(roi, roi.mean(-1)) * args.px
        proj, ap_roi = ap_from(sh_roi, fs, args.resp)

        # negative control on ROI
        idx = rng.permutation(T)
        sh_sh = track(roi[..., idx], roi.mean(-1)) * args.px
        _, ap_sh = ap_from(sh_sh, fs, args.resp)

        rows.append(dict(z=z, ap_roi=ap_roi, ap_full=ap_full, ap_shuffle=ap_sh,
                         roi=(int(y0), int(y1), int(x0), int(x1)), center=(cy, cx),
                         _proj=proj.tolist(), _P=P, _sh=sh_roi.tolist()))

    A = lambda k: np.array([r[k] for r in rows])
    summary = dict(
        n_slices=len(rows),
        heart_roi_ap_p2p_mm=dict(mean=float(A('ap_roi').mean()), median=float(np.median(A('ap_roi'))),
                                 min=float(A('ap_roi').min()), max=float(A('ap_roi').max())),
        whole_fov_ap_p2p_mm=dict(mean=float(A('ap_full').mean())),
        shuffle_control_ap_p2p_mm=dict(mean=float(A('ap_shuffle').mean())),
        roi_vs_shuffle_ratio=float(A('ap_roi').mean() / max(A('ap_shuffle').mean(), 1e-9)),
        roi_vs_fov_ratio=float(A('ap_roi').mean() / max(A('ap_full').mean(), 1e-9)),
    )
    json.dump({'summary': summary,
               'per_slice': [{k: r[k] for k in ('z', 'ap_roi', 'ap_full', 'ap_shuffle', 'roi', 'center')}
                             for r in rows]}, open(args.out_prefix + '.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    t = np.arange(T) * args.dt
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    zc = int(np.argsort(A('ap_roi'))[len(rows) // 2])
    R = rows[zc]; y0, y1, x0, x1 = R['roi']
    cine0 = (v[:, :, R['z'], T // 3] / np.percentile(v[:, :, R['z'], :], 99.5))
    ax[0, 0].imshow(np.rot90(np.clip(cine0, 0, 1.2)), cmap='gray')
    # draw ROI box (note rot90 flips axes for display)
    ax[0, 0].set_title(f'slice {R["z"]}: heart ROI from cardiac-power'); ax[0, 0].axis('off')
    ax[0, 1].imshow(np.rot90(R['_P']), cmap='magma'); ax[0, 1].set_title('cardiac-band power map (heart localizer)'); ax[0, 1].axis('off')
    ax[0, 2].plot(t, R['_proj'], 'b-', lw=2)
    ax[0, 2].set_title(f'heart-ROI AP respiratory drift (p2p {R["ap_roi"]:.1f} mm)')
    ax[0, 2].set_xlabel('time (s)'); ax[0, 2].set_ylabel('AP disp (mm)')

    xs = np.arange(len(rows)); w = 0.4
    ax[1, 0].bar(xs - w/2, A('ap_roi'), w, label=f"heart ROI (mean {A('ap_roi').mean():.1f})", color='crimson')
    ax[1, 0].bar(xs + w/2, A('ap_full'), w, label=f"whole FOV (mean {A('ap_full').mean():.1f})", color='steelblue')
    ax[1, 0].set_title('AP respiratory amplitude per slice: heart ROI vs whole FOV')
    ax[1, 0].set_xlabel('slice'); ax[1, 0].set_ylabel('p2p AP (mm)'); ax[1, 0].legend(fontsize=8)

    ax[1, 1].bar(xs, A('ap_roi'), color='crimson', label='heart ROI')
    ax[1, 1].bar(xs, A('ap_shuffle'), color='gray', alpha=0.7, label='shuffle control')
    ax[1, 1].axhline(A('ap_roi').mean(), color='k', ls='--')
    ax[1, 1].set_title(f"heart-ROI AP vs shuffle control (ratio {summary['roi_vs_shuffle_ratio']:.1f}x)")
    ax[1, 1].set_xlabel('slice'); ax[1, 1].set_ylabel('p2p AP (mm)'); ax[1, 1].legend(fontsize=8)

    ax[1, 2].axis('off')
    ax[1, 2].text(0.02, 0.5,
        f"HEART-ROI AP respiratory motion (vol0001, {len(rows)} slices)\n\n"
        f"heart ROI : {summary['heart_roi_ap_p2p_mm']['mean']:.2f} mm mean "
        f"({summary['heart_roi_ap_p2p_mm']['min']:.1f}-{summary['heart_roi_ap_p2p_mm']['max']:.1f})\n"
        f"whole FOV : {summary['whole_fov_ap_p2p_mm']['mean']:.2f} mm mean\n"
        f"shuffle   : {summary['shuffle_control_ap_p2p_mm']['mean']:.2f} mm (control)\n\n"
        f"ROI / FOV ratio     : {summary['roi_vs_fov_ratio']:.2f}x\n"
        f"ROI / shuffle ratio : {summary['roi_vs_shuffle_ratio']:.1f}x\n\n"
        f"implied SI (AP/0.35): ~{summary['heart_roi_ap_p2p_mm']['mean']/0.35:.1f} mm",
        fontsize=11, family='monospace', va='center')
    plt.suptitle('Heart-ROI respiratory motion — RT free-breathing radial bSSFP (vol0001)', fontsize=13)
    plt.tight_layout(); plt.savefig(args.out_prefix + '.png', dpi=100, bbox_inches='tight')
    print('saved', args.out_prefix + '.png')


if __name__ == '__main__':
    main()
