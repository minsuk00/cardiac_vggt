#!/usr/bin/env python
"""Measure / prove respiratory (breathing) motion in a Göttingen RT-FB recon.

Method
------
For each SAX slice (an independent ~4.26 s real-time free-breathing cine of 127 frames):
  1. Estimate per-frame in-plane rigid translation vs the temporal mean image
     (subpixel phase cross-correlation) -> trajectory dx(t), dy(t) in mm (1.6 mm/px).
  2. Band-separate the trajectory in the temporal-frequency domain:
       respiratory band  = 0.1-0.5 Hz   (breathing ~0.2-0.33 Hz, ~1 cycle in 4.26 s)
       cardiac band      = 0.7-2.0 Hz    (HR ~42-120 bpm, ~4 cycles in 4.26 s)
  3. PCA on the respiratory-band trajectory -> principal in-plane axis (~AP for SAX,
     since SI respiratory motion is mostly through-plane) + its peak-to-peak amplitude.
  4. Through-plane (SI) evidence: low-freq oscillation of slice *content* similarity
     (1 - NCC of each frame vs mean) -> the heart sliding through the fixed SAX plane.

Validation / proof
------------------
  * POSITIVE control: the cardiac band must recover ~4 cycles (we know HR) -> method works.
  * NEGATIVE control: temporally shuffle the frames and re-track -> the coherent slow drift
    must vanish (proves the drift is temporally structured physiology, not recon noise).
  * Global-translation variance-explained: fraction of frame-to-frame variance removed by a
    single rigid translation -> high = bulk (respiratory) motion, not local (cardiac) deform.

Outputs: a multi-panel PNG + a JSON of numbers.
"""
import argparse, json
import numpy as np
import nibabel as nib
from skimage.registration import phase_cross_correlation
from scipy.signal import butter, filtfilt, periodogram


def track(cine, ref):
    """Per-frame subpixel translation (dy,dx) vs ref image, in pixels."""
    sh = np.zeros((cine.shape[-1], 2))
    for t in range(cine.shape[-1]):
        s, _, _ = phase_cross_correlation(ref, cine[..., t], upsample_factor=20, normalization=None)
        sh[t] = s
    return sh - sh.mean(0)                  # zero-mean


def bandpass(x, lo, hi, fs):
    ny = 0.5 * fs
    if hi >= ny: hi = ny * 0.99
    b, a = butter(3, [lo / ny, hi / ny], btype='band')
    return filtfilt(b, a, x, axis=0)


def lowpass(x, hi, fs):
    b, a = butter(3, hi / (0.5 * fs), btype='low')
    return filtfilt(b, a, x, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nii')
    ap.add_argument('out_prefix')
    ap.add_argument('--dt', type=float, default=0.03354)         # s/frame (13 spokes x 2.58 ms)
    ap.add_argument('--px', type=float, default=1.6)             # in-plane mm/px
    ap.add_argument('--resp', type=float, nargs=2, default=[0.1, 0.5])
    ap.add_argument('--card', type=float, nargs=2, default=[0.7, 2.0])
    args = ap.parse_args()
    fs = 1.0 / args.dt

    v = nib.load(args.nii).get_fdata()        # [X,Y,Z,T]
    X, Y, Z, T = v.shape
    freqs = np.fft.rfftfreq(T, args.dt)

    per_slice = []
    rng = np.random.default_rng(0)
    for z in range(Z):
        cine = v[:, :, z, :].astype(np.float32)
        if cine.max() <= 0:
            continue
        cine = cine / np.percentile(cine, 99.5)
        ref = cine.mean(-1)
        sh = track(cine, ref) * args.px       # mm, (T,2) = (dy,dx)

        resp = bandpass(sh, *args.resp, fs)    # respiratory-band trajectory
        card = bandpass(sh, *args.card, fs)    # cardiac-band trajectory

        # principal (AP) axis of respiratory motion via PCA
        u, s_, vt = np.linalg.svd(resp - resp.mean(0), full_matrices=False)
        ap_axis = vt[0]
        resp_ap = resp @ ap_axis               # projection onto principal axis
        card_mag = np.linalg.norm(card, axis=1)

        # cardiac cycle count (positive control): peak of cardiac-band spectrum
        f_c, P_c = periodogram(card @ vt[0] if False else card_mag, fs)
        card_peak_hz = f_c[args.card[0] <= f_c][np.argmax(P_c[args.card[0] <= f_c])] \
            if (f_c >= args.card[0]).any() else np.nan
        card_cycles = card_peak_hz * T * args.dt

        # negative control: shuffle frames, re-track, measure residual resp amplitude
        idx = rng.permutation(T)
        sh_sh = track(cine[..., idx], ref) * args.px
        resp_sh = bandpass(sh_sh, *args.resp, fs)
        resp_sh_ap = resp_sh @ ap_axis

        # through-plane (SI) evidence: 1 - NCC(frame, mean) low-freq oscillation
        rf = ref / (np.linalg.norm(ref) + 1e-9)
        ncc = np.array([1 - float((cine[..., t] / (np.linalg.norm(cine[..., t]) + 1e-9) * rf).sum())
                        for t in range(T)])
        ncc_lp = lowpass(ncc - ncc.mean(), args.resp[1], fs)

        per_slice.append(dict(
            z=z,
            resp_ap_p2p=float(resp_ap.max() - resp_ap.min()),
            resp_ap_std=float(resp_ap.std()),
            resp_full_p2p=float(np.linalg.norm(resp, axis=1).max() * 2),
            card_ap_p2p=float(card_mag.max() - card_mag.min()),
            card_cycles=float(card_cycles),
            shuffle_resp_p2p=float(resp_sh_ap.max() - resp_sh_ap.min()),
            si_lowfreq_p2p=float(ncc_lp.max() - ncc_lp.min()),
            _resp_ap=resp_ap.tolist(), _card_mag=card_mag.tolist(),
            _sh=sh.tolist(), _ncc_lp=ncc_lp.tolist(), _shuf=resp_sh_ap.tolist(),
        ))

    # aggregate
    A = lambda k: np.array([p[k] for p in per_slice])
    summary = dict(
        n_slices=len(per_slice),
        resp_ap_p2p_mm=dict(mean=float(A('resp_ap_p2p').mean()), median=float(np.median(A('resp_ap_p2p'))),
                            min=float(A('resp_ap_p2p').min()), max=float(A('resp_ap_p2p').max())),
        resp_ap_std_mm=dict(mean=float(A('resp_ap_std').mean())),
        cardiac_cycles_recovered=dict(mean=float(A('card_cycles').mean()), median=float(np.median(A('card_cycles')))),
        shuffle_control_resp_p2p_mm=dict(mean=float(A('shuffle_resp_p2p').mean())),
        si_through_plane_lowfreq_p2p=dict(mean=float(A('si_lowfreq_p2p').mean())),
        resp_vs_shuffle_ratio=float(A('resp_ap_p2p').mean() / max(A('shuffle_resp_p2p').mean(), 1e-9)),
    )
    json.dump({'summary': summary, 'per_slice': [{k: v for k, v in p.items() if not k.startswith('_')}
               for p in per_slice]}, open(args.out_prefix + '.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2))

    # ---- figure ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    t = np.arange(T) * args.dt
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    # pick the median-amplitude slice for trajectory plots
    zc = int(np.argsort(A('resp_ap_p2p'))[len(per_slice) // 2])
    P = per_slice[zc]
    ax[0, 0].plot(t, P['_resp_ap'], 'b-', lw=2, label='respiratory band (0.1-0.5 Hz)')
    ax[0, 0].plot(t, P['_card_mag'], 'r-', lw=0.8, alpha=0.7, label='cardiac band |.| (0.7-2 Hz)')
    ax[0, 0].set_title(f'slice {P["z"]}: AP respiratory drift (p2p {P["resp_ap_p2p"]:.1f} mm) vs cardiac')
    ax[0, 0].set_xlabel('time (s)'); ax[0, 0].set_ylabel('displacement (mm)'); ax[0, 0].legend(fontsize=8)

    # spectrum of raw AP displacement
    sh = np.array(P['_sh']); apsig = sh @ np.linalg.svd(sh - sh.mean(0), full_matrices=False)[2][0]
    f, Pxx = periodogram(apsig, fs)
    ax[0, 1].semilogy(f, Pxx, 'k-'); ax[0, 1].axvspan(*args.resp, color='b', alpha=0.15, label='resp band')
    ax[0, 1].axvspan(*args.card, color='r', alpha=0.15, label='cardiac band')
    ax[0, 1].set_xlim(0, 3); ax[0, 1].set_title('AP displacement spectrum'); ax[0, 1].set_xlabel('Hz'); ax[0, 1].legend(fontsize=8)

    # negative control overlay
    ax[0, 2].plot(t, P['_resp_ap'], 'b-', lw=2, label='real (temporal order)')
    ax[0, 2].plot(t, P['_shuf'], 'gray', lw=1, label='shuffled frames (control)')
    ax[0, 2].set_title(f'negative control: shuffle kills drift\n(real {P["resp_ap_p2p"]:.1f} vs shuffle {P["shuffle_resp_p2p"]:.1f} mm)')
    ax[0, 2].set_xlabel('time (s)'); ax[0, 2].set_ylabel('AP disp (mm)'); ax[0, 2].legend(fontsize=8)

    # amplitude distribution across slices
    ax[1, 0].bar(range(len(per_slice)), A('resp_ap_p2p'), color='steelblue')
    ax[1, 0].axhline(A('resp_ap_p2p').mean(), color='k', ls='--', label=f"mean {A('resp_ap_p2p').mean():.1f} mm")
    ax[1, 0].set_title('AP respiratory amplitude per slice (24 independent acquisitions)')
    ax[1, 0].set_xlabel('slice'); ax[1, 0].set_ylabel('p2p AP (mm)'); ax[1, 0].legend(fontsize=8)

    # cardiac cycles recovered (positive control)
    ax[1, 1].bar(range(len(per_slice)), A('card_cycles'), color='indianred')
    ax[1, 1].axhline(A('card_cycles').mean(), color='k', ls='--', label=f"mean {A('card_cycles').mean():.1f} cycles")
    ax[1, 1].set_title('positive control: cardiac cycles recovered / 4.26 s\n(~4-6 expected for HR 60-90 bpm)')
    ax[1, 1].set_xlabel('slice'); ax[1, 1].set_ylabel('cycles'); ax[1, 1].legend(fontsize=8)

    # SI through-plane evidence
    ax[1, 2].plot(t, P['_ncc_lp'], 'g-', lw=2)
    ax[1, 2].set_title(f'through-plane (SI) evidence — slice {P["z"]}\nlow-freq content change (p2p {P["si_lowfreq_p2p"]:.3f})')
    ax[1, 2].set_xlabel('time (s)'); ax[1, 2].set_ylabel('1 - NCC(frame, mean)')

    plt.suptitle(f'Respiratory motion in {args.nii.split("/")[-1]} — RT free-breathing radial bSSFP', fontsize=13)
    plt.tight_layout(); plt.savefig(args.out_prefix + '.png', dpi=100, bbox_inches='tight')
    print('saved', args.out_prefix + '.png')


if __name__ == '__main__':
    main()
