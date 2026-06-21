#!/usr/bin/env python
"""Confirm respiratory motion across ALL Göttingen subjects (not just vol0001).

For every reconstructed volume, measure heart-ROI AP respiratory drift on a sample of mid slices
(faster than all slices), report per-subject mean + the cross-subject distribution, and render a
histogram. Reuses the heart-ROI method from measure_respiratory_motion_heartroi.py.
"""
import glob, json, os
import numpy as np
import nibabel as nib
from skimage.registration import phase_cross_correlation
from scipy.signal import butter, filtfilt
from scipy.ndimage import label

DT, PX = 0.03354, 1.6
FS = 1.0 / DT
RESP, CARD = (0.1, 0.5), (0.7, 2.0)
N_SLICES = 8                                   # mid slices sampled per subject


def bandpass(x, lo, hi):
    ny = 0.5 * FS; hi = min(hi, ny * 0.99)
    b, a = butter(3, [lo / ny, hi / ny], btype='band')
    return filtfilt(b, a, x, axis=0)


def heart_roi(cine):
    T = cine.shape[-1]
    P = bandpass(cine.transpose(2, 0, 1).reshape(T, -1), *CARD).var(0).reshape(cine.shape[:2])
    mask = P > np.percentile(P, 92)
    lab, n = label(mask)
    H, W = cine.shape[:2]
    if n == 0:
        return (H // 4, 3 * H // 4, W // 4, 3 * W // 4)
    big = 1 + int(np.argmax([(lab == i).sum() for i in range(1, n + 1)]))
    ys, xs = np.where(lab == big)
    return (max(0, ys.min() - 8), min(H, ys.max() + 8), max(0, xs.min() - 8), min(W, xs.max() + 8))


def track_ap(cine):
    ref = cine.mean(-1)
    sh = np.array([phase_cross_correlation(ref, cine[..., t], upsample_factor=20, normalization=None)[0]
                   for t in range(cine.shape[-1])])
    sh = (sh - sh.mean(0)) * PX
    r = bandpass(sh, *RESP)
    vt = np.linalg.svd(r - r.mean(0), full_matrices=False)[2]
    proj = r @ vt[0]
    return float(proj.max() - proj.min())


def main():
    files = sorted(glob.glob('scratch/data/goettingen/recon/vol*_vis1/vol*_vis1.nii.gz'))
    rows = []
    for f in files:
        name = os.path.basename(f).replace('.nii.gz', '')
        v = nib.load(f).get_fdata()
        X, Y, Z, T = v.shape
        zsamp = np.linspace(Z * 0.25, Z * 0.75, N_SLICES).astype(int)
        aps = []
        for z in zsamp:
            cine = v[:, :, z, :].astype(np.float32)
            if cine.max() <= 0:
                continue
            cine = cine / np.percentile(cine, 99.5)
            y0, y1, x0, x1 = heart_roi(cine)
            aps.append(track_ap(cine[y0:y1, x0:x1, :]))
        rows.append(dict(subject=name, n_slices=Z, ap_mean=float(np.mean(aps)),
                         ap_median=float(np.median(aps)), ap_max=float(np.max(aps))))
        print(f"{name}: AP {rows[-1]['ap_mean']:.2f} mm (median {rows[-1]['ap_median']:.2f}, max {rows[-1]['ap_max']:.2f})", flush=True)

    means = np.array([r['ap_mean'] for r in rows])
    summary = dict(n_subjects=len(rows),
                   ap_across_subjects=dict(mean=float(means.mean()), median=float(np.median(means)),
                                           min=float(means.min()), max=float(means.max()), std=float(means.std())),
                   frac_above_2mm=float((means > 2).mean()), frac_above_3mm=float((means > 3).mean()))
    json.dump({'summary': summary, 'per_subject': rows},
              open('scratch/data/goettingen/analysis/breathing_all_subjects.json', 'w'), indent=2)
    print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    order = np.argsort(means)
    ax[0].bar(range(len(rows)), means[order], color='crimson')
    ax[0].axhline(means.mean(), color='k', ls='--', label=f'mean {means.mean():.2f} mm')
    ax[0].set_title(f'heart-ROI AP respiratory drift per subject (n={len(rows)})')
    ax[0].set_xlabel('subject (sorted)'); ax[0].set_ylabel('AP p2p (mm)'); ax[0].legend()
    ax[1].hist(means, bins=15, color='steelblue', edgecolor='k')
    ax[1].axvline(means.mean(), color='r', ls='--', label=f'mean {means.mean():.2f}')
    ax[1].set_title('distribution of per-subject AP'); ax[1].set_xlabel('AP p2p (mm)'); ax[1].set_ylabel('# subjects'); ax[1].legend()
    plt.tight_layout(); plt.savefig('scratch/data/goettingen/analysis/breathing_all_subjects.png', dpi=100, bbox_inches='tight')
    print('saved breathing_all_subjects.png')


if __name__ == '__main__':
    main()
