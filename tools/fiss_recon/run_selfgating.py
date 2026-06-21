"""Run self-gating on a FISS .dat and visualize cardiac/respiratory signals.

Caches navigator spokes + per-spoke index to an .npz so later stages don't
re-read the 2 GB file. Run: micromamba run -n fiss-recon python run_selfgating.py <dat>
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from twix_io import read_fiss_twix
from self_gating import extract_signals, assign_bins

INTERLEAVE_DT = 24 * 2.47e-3   # s per interleave (24 readouts x 2.47 ms TR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dat')
    ap.add_argument('--out', default='/tmp/fiss_inspect/selfgating.png')
    ap.add_argument('--ncard', type=int, default=25)
    ap.add_argument('--nresp', type=int, default=4)
    args = ap.parse_args()

    d = read_fiss_twix(args.dat)
    fs = 1.0 / INTERLEAVE_DT
    sig = extract_signals(d['kdata'], d['is_nav'], fs)
    cbin, rbin = assign_bins(sig, args.ncard, args.nresp)

    print(f"HR ~ {sig['hr_bpm']:.0f} bpm (PC{sig['card_pc']}), "
          f"RR ~ {sig['rr_per_min']:.0f}/min (PC{sig['resp_pc']})")
    # occupancy
    occ = np.zeros((args.ncard, args.nresp), int)
    for c, r in zip(cbin, rbin):
        if c >= 0 and r >= 0:
            occ[c, r] += 1
    print(f"valid interleaves: {int(sig['valid'].sum())}/{len(cbin)}")
    print("bin occupancy (cardiac x resp), spokes/bin = occ*23 (non-nav per il):")
    print("  min interleaves/bin:", occ.min(), " median:", int(np.median(occ)),
          " max:", occ.max())

    T = len(sig['resp_signal'])
    t = np.arange(T) * INTERLEAVE_DT
    fig, axs = plt.subplots(4, 1, figsize=(13, 11))
    axs[0].plot(t, sig['resp_signal']); axs[0].set_title(
        f"respiratory signal (~{sig['rr_per_min']:.0f}/min)"); axs[0].set_ylabel('a.u.')
    axs[1].plot(t[:600], sig['cardiac_signal'][:600]); axs[1].set_title(
        f"cardiac signal, first {600*INTERLEAVE_DT:.0f}s (~{sig['hr_bpm']:.0f} bpm)")
    axs[2].plot(t[:600], sig['cardiac_phase'][:600], '.', ms=2)
    axs[2].set_title('cardiac phase 0..1'); axs[2].set_ylabel('phase')
    im = axs[3].imshow(occ.T, aspect='auto', origin='lower', cmap='viridis')
    axs[3].set_title('interleaves per (cardiac,resp) bin'); axs[3].set_xlabel('cardiac bin'); axs[3].set_ylabel('resp bin')
    plt.colorbar(im, ax=axs[3])
    plt.tight_layout(); plt.savefig(args.out, dpi=90)
    print('saved', args.out)

    np.savez('/tmp/fiss_inspect/gating.npz',
             cbin=cbin, rbin=rbin, **{k: v for k, v in sig.items()
                                      if isinstance(v, np.ndarray)})
    print('saved gating.npz')


if __name__ == '__main__':
    main()
