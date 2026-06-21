"""Self-gating: derive cardiac + respiratory signals from the SI navigators
and assign every interleave to a (cardiac, respiratory) bin.

Free-Running Framework (Di Sopra 2019) approach:
  - Navigator = first spoke of each interleave, oriented along SI (+z).
  - FFT each navigator along the readout -> 1-D SI projection (one per interleave,
    sampled at the interleave rate ~16.9 Hz here).
  - Stack projections over time, build a feature matrix (coils x SI-positions),
    PCA/SVD across time -> temporal components.
  - Respiratory signal = component with most spectral power in 0.1-0.5 Hz.
    Cardiac signal     = component with most spectral power in 0.6-1.8 Hz,
    taken after projecting out the respiratory component.
  - Respiratory binning: amplitude bins (end-expiration as reference bin).
  - Cardiac binning: instantaneous phase from the analytic signal -> 0..1 -> bins.

The number of bins (25 cardiac x 4 respiratory) matches the source paper.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def _bandpass(x, fs, lo, hi, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return filtfilt(b, a, x, axis=0)


def navigator_projections(kdata, is_nav):
    """(n_nav, n_coil, n_pos) magnitude SI projections from navigator spokes."""
    nav = kdata[is_nav]                                   # (n_nav, coil, read)
    proj = np.fft.fftshift(np.fft.ifft(
        np.fft.ifftshift(nav, axes=-1), axis=-1), axes=-1)
    return np.abs(proj).astype(np.float32)               # (n_nav, coil, pos)


def _dominant_component(pcs, freqs, band):
    """Index of the PC with the most spectral power in `band`."""
    P = np.abs(np.fft.rfft(pcs - pcs.mean(0), axis=0))
    m = (freqs >= band[0]) & (freqs <= band[1])
    return int(np.argmax(P[m].sum(0)))


def extract_signals(kdata, is_nav, fs, resp_band=(0.1, 0.5),
                    card_band=(0.6, 1.8), n_pc=10, skip_s=8.0):
    """Return dict with cardiac_phase[0..1), resp_signal, and diagnostics.

    Signals are per-navigator (i.e. per-interleave). The first `skip_s` seconds
    (bSSFP/FISS approach-to-steady-state transient) are excluded from the PCA
    and marked invalid in `valid` so downstream binning/recon can drop them.
    """
    mag_all = navigator_projections(kdata, is_nav)       # (T, C, P)
    T = mag_all.shape[0]
    skip = int(round(skip_s * fs))
    valid = np.zeros(T, bool); valid[skip:] = True
    mag = mag_all[skip:]                                  # steady-state only
    Tk = mag.shape[0]
    X = mag.reshape(Tk, -1)
    X = X - X.mean(0, keepdims=True)
    # keep high-variance features (body region), standardize
    v = X.var(0)
    X = X[:, v > np.percentile(v, 70)]
    X = X / (X.std(0, keepdims=True) + 1e-8)
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pcs_k = U[:, :n_pc] * S[:n_pc]                        # (Tk, n_pc)
    freqs = np.fft.rfftfreq(Tk, d=1 / fs)

    ri = _dominant_component(pcs_k, freqs, resp_band)
    resp_k = _bandpass(pcs_k[:, ri], fs, *resp_band)

    # remove respiratory from all PCs, then find cardiac in a DIFFERENT PC
    resp_n = resp_k / (np.linalg.norm(resp_k) + 1e-8)
    pcs_c = pcs_k - resp_n[:, None] * (pcs_k.T @ resp_n)[None, :]
    ci = _dominant_component(pcs_c, freqs, card_band)
    card_k = _bandpass(pcs_c[:, ci], fs, *card_band)

    # cardiac instantaneous phase -> [0,1) (on the kept, steady-state segment)
    phase_k = (np.angle(hilbert(card_k)) + np.pi) / (2 * np.pi)
    # HR / RR from the band peaks (kept segment, matching `freqs`)
    Pc = np.abs(np.fft.rfft(card_k - card_k.mean()))
    m = (freqs >= card_band[0]) & (freqs <= card_band[1])
    hr_hz = freqs[m][np.argmax(Pc[m])]
    Pr = np.abs(np.fft.rfft(resp_k - resp_k.mean()))
    mr = (freqs >= resp_band[0]) & (freqs <= resp_band[1])
    rr_hz = freqs[mr][np.argmax(Pr[mr])]

    # pad back to full length (invalid transient region zero-filled)
    def pad(a):
        out = np.zeros(T, np.float32); out[skip:] = a; return out
    return dict(cardiac_phase=pad(phase_k), cardiac_signal=pad(card_k),
                resp_signal=pad(resp_k), valid=valid,
                hr_bpm=float(hr_hz * 60), rr_per_min=float(rr_hz * 60),
                resp_pc=ri, card_pc=ci, freqs=freqs)


def assign_bins(sig, n_cardiac=25, n_resp=4):
    """Per-interleave (cardiac_bin, resp_bin).

    Cardiac: phase 0..1 -> n_cardiac uniform bins.
    Respiratory: amplitude -> n_resp quantile bins; bin 0 = end-expiration
    (defined as the resp_signal extreme with the most data on the calm side;
    here we take the lower-amplitude tail as end-expiration by convention and
    flip if needed by the caller).
    """
    valid = sig.get('valid', np.ones_like(sig['resp_signal'], bool))
    cphase = sig['cardiac_phase']
    cbin = np.minimum((cphase * n_cardiac).astype(int), n_cardiac - 1)

    r = sig['resp_signal']
    # quantile edges -> equal-occupancy respiratory bins (over valid only)
    edges = np.quantile(r[valid], np.linspace(0, 1, n_resp + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    rbin = np.clip(np.digitize(r, edges) - 1, 0, n_resp - 1)
    cbin[~valid] = -1; rbin[~valid] = -1          # invalid transient interleaves
    return cbin.astype(np.int16), rbin.astype(np.int16)
