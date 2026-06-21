"""Structural proof that CMRxRecon target_t = k/12 is NORMALIZED fractional cycle time.

Exact facts (no proxy):
  * on-disk sax_frame count per subject  (should be 12 for all)
  * native TemporalPhase per subject from cine_sax_info.csv  (varies 14-40)
  -> fixed 12 from variable native => the 12 phases are a temporal RESAMPLING of
     each subject's own R-R interval, i.e. phase k = k/12 of that subject's cycle.

Resample-vs-truncate discriminator (intensity proxy, no segmentation):
  * mean |frame_t - frame_0| curve: rises to a peak then FALLS  => 12 frames span
    systole->diastole (full cycle) => full-cycle resample, not first-12 truncation.
  * corr(intensity-ES-proxy index, native phase count) ~ 0 => resample (truncation
    would force a strong positive correlation).

Output: JSON (--out_json).
"""
import argparse, csv, glob, json, os
import numpy as np
import nibabel as nib

NUM_PHASES = 12


def native_temporal_phase(subj_sax_dir):
    csvf = os.path.join(subj_sax_dir, "cine_sax_info.csv")
    if not os.path.exists(csvf):
        return None
    for r in csv.reader(open(csvf)):
        if r and r[0] == "TemporalPhase":
            return int(r[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--proxy_n", type=int, default=0,
                    help="subjects for the intensity periodicity proxy (0 = all)")
    args = ap.parse_args()

    subs = sorted(d for d in os.listdir(args.data_root)
                  if os.path.isdir(os.path.join(args.data_root, d)))

    frame_counts = {}
    native = {}
    for s in subs:
        sd = os.path.join(args.data_root, s, "sax")
        nfr = len(glob.glob(os.path.join(sd, "3d_recon", "sax_frame_*.nii.gz")))
        frame_counts[nfr] = frame_counts.get(nfr, 0) + 1
        tp = native_temporal_phase(sd)
        if tp is not None:
            native[s] = tp

    native_vals = np.array(list(native.values()))
    native_hist = {int(k): int((native_vals == k).sum()) for k in np.unique(native_vals)}

    # intensity proxy over a subset (or all)
    proxy_subs = subs if args.proxy_n == 0 else subs[: args.proxy_n]
    curves = []; es_proxy = []; native_for_proxy = []
    for s in proxy_subs:
        rd = os.path.join(args.data_root, s, "sax", "3d_recon")
        fps = [os.path.join(rd, f"sax_frame_{t:02d}.nii.gz") for t in range(NUM_PHASES)]
        if not all(os.path.exists(f) for f in fps):
            continue
        vols = np.stack([nib.load(f).get_fdata().astype(np.float32) for f in fps], 0)
        f0 = vols[0]
        m = f0 > np.percentile(f0, 75)
        diff = np.array([np.mean(np.abs(vols[t] - f0)[m]) for t in range(NUM_PHASES)])
        diff = diff / (np.mean(f0[m]) + 1e-6)
        curves.append(diff)
        es_proxy.append(int(diff.argmax()))
        native_for_proxy.append(native.get(s, np.nan))

    curves = np.array(curves)
    mean_curve = curves.mean(0)
    es_proxy = np.array(es_proxy); native_for_proxy = np.array(native_for_proxy)
    valid = ~np.isnan(native_for_proxy)
    corr = float(np.corrcoef(es_proxy[valid], native_for_proxy[valid])[0, 1]) \
        if valid.sum() > 2 else float("nan")
    # fraction of subjects whose curve falls after its peak (diastole captured)
    falls_after_peak = float(np.mean([
        c.argmax() < NUM_PHASES - 1 and c[-1] < c.max() for c in curves]))

    out = dict(
        n_subjects=len(subs),
        frame_counts=frame_counts,
        all_have_12=frame_counts == {NUM_PHASES: len(subs)},
        native_temporal_phase_hist=native_hist,
        native_min=int(native_vals.min()), native_max=int(native_vals.max()),
        native_mean=float(native_vals.mean()),
        native_ever_12=bool((native_vals == NUM_PHASES).any()),
        proxy_n=int(curves.shape[0]),
        mean_diff_curve=mean_curve.tolist(),
        curve_peak_frame=int(mean_curve.argmax()),
        falls_after_peak_frac=falls_after_peak,
        corr_esproxy_native=corr,
    )
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"subjects: {out['n_subjects']}")
    print(f"on-disk frame counts: {frame_counts}  all_12={out['all_have_12']}")
    print(f"native TemporalPhase: min={out['native_min']} max={out['native_max']} "
          f"mean={out['native_mean']:.1f} ever==12: {out['native_ever_12']}")
    print(f"\nproxy on {out['proxy_n']} subjects:")
    print(f"  mean |frame_t - ED| curve peaks at t={out['curve_peak_frame']}, "
          f"falls after peak in {falls_after_peak*100:.0f}% of subjects (=> full-cycle resample)")
    print(f"  corr(ES_proxy_idx, native_phase_count) = {corr:+.3f}  (~0 => resample, not truncation)")


if __name__ == "__main__":
    main()
