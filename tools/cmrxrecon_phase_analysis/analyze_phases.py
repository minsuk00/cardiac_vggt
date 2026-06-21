"""Gold-standard cardiac-state analysis of CMRxRecon's 12 cine phases.

Reads Task114 (M&Ms nnU-Net) segmentations of every (subject, phase) volume and
the matching nnU-Net input volumes (for per-subject voxel spacing), computes the
LV blood-pool volume per phase, and derives:

  * per-subject EDV / ESV / EF and the ED & ES frame indices
  * the cross-subject distribution of the ES frame index  (Q: is ES fixed?)
  * the per-phase cross-subject spread of cardiac state    (Q: does target_t = k/12
    map to a fixed state?), measured two ways:
       - relative LV volume   v_rel(t)  = LV(t) / EDV          (includes EF spread)
       - contraction fraction cf(t)     = (LV(t)-ESV)/(EDV-ESV) (isolates TIMING:
            1.0 = ED-like/full, 0.0 = ES-like/empty)

Labels (Task114): 1=LV blood pool, 2=LV myocardium, 3=RV blood pool.

Output: a JSON blob (--out_json) with everything the report needs + a per-subject
CSV (--out_csv). No plotting here (the report builder does that).
"""
import argparse, glob, json, os, re
import numpy as np
import nibabel as nib

NUM_PHASES = 12
LV, MYO, RV = 1, 2, 3


def subj_from_case(path):
    # ".../<subj>_t{tt}.nii.gz"  ->  (subj, t)
    b = os.path.basename(path)[:-7]              # strip .nii.gz
    m = re.match(r"^(.*)_t(\d{2})$", b)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def voxel_ml(input_path):
    z = nib.load(input_path).header.get_zooms()[:3]
    return float(np.prod(z)) / 1000.0           # mm^3 -> mL


def subframe_es(lv, es):
    """Parabolic sub-frame refinement of the ES (LV-min) index.

    Fits a parabola through (es-1, es, es+1) and returns the vertex location, so a
    flat/quantized trough doesn't force ES onto an integer grid. Returns float es +
    a 'sharpness' = (2nd-smallest LV - smallest LV)/range, a [0,1] measure of how
    well-determined the trough is (0 = a tie/flat trough; argmin there is noise).
    Boundary es (0 or 11) is left unrefined (real ES is mid-systole, never at ED).
    """
    rng = lv.max() - lv.min()
    second = np.sort(lv)[1]
    sharp = float((second - lv.min()) / rng) if rng > 0 else 0.0
    if es <= 0 or es >= len(lv) - 1:
        return float(es), sharp
    a, b, c = lv[es - 1], lv[es], lv[es + 1]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    return es + delta, sharp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    # group seg files by subject
    segs = sorted(glob.glob(os.path.join(args.seg_dir, "*.nii.gz")))
    by_subj = {}
    for s in segs:
        subj, t = subj_from_case(s)
        if subj is None:
            continue
        by_subj.setdefault(subj, {})[t] = s

    subjects = sorted(by_subj)
    per_subj = []           # clean subjects only (12 phases, valid LV everywhere)
    skipped = []            # (subj, reason)

    for subj in subjects:
        tmap = by_subj[subj]
        if sorted(tmap) != list(range(NUM_PHASES)):
            skipped.append((subj, f"have phases {sorted(tmap)}")); continue
        lv = np.zeros(NUM_PHASES); rv = np.zeros(NUM_PHASES); myo = np.zeros(NUM_PHASES)
        ok = True
        for t in range(NUM_PHASES):
            seg = np.asarray(nib.load(tmap[t]).dataobj).astype(np.int16)
            ip = os.path.join(args.input_dir, f"{subj}_t{t:02d}_0000.nii.gz")
            if not os.path.exists(ip):
                ok = False; skipped.append((subj, f"missing input t{t}")); break
            vml = voxel_ml(ip)
            lv[t]  = (seg == LV).sum()  * vml
            rv[t]  = (seg == RV).sum()  * vml
            myo[t] = (seg == MYO).sum() * vml
        if not ok:
            continue
        if (lv <= 0).any():
            skipped.append((subj, f"LV empty at phase(s) {np.where(lv<=0)[0].tolist()}")); continue

        ed = int(lv.argmax()); es = int(lv.argmin())
        edv = float(lv[ed]); esv = float(lv[es])
        if edv <= esv:                              # constant-LV subject: ES undefined, would NaN-poison cf
            skipped.append((subj, "constant LV (edv==esv)")); continue
        ef = (edv - esv) / edv * 100.0
        ef_gating = (lv[0] - esv) / lv[0] * 100.0 if lv[0] > 0 else float("nan")  # ED := frame 0 (gating)
        es_sub, es_sharp = subframe_es(lv, es)      # robust sub-frame ES + trough sharpness
        # contraction fraction: 1=ED(full), 0=ES(empty). edv>esv now guaranteed.
        cf = (lv - esv) / (edv - esv)
        v_rel = lv / edv
        per_subj.append(dict(
            subj=subj, lv=lv.tolist(), rv=rv.tolist(), myo=myo.tolist(),
            ed_frame=ed, es_frame=es, es_subframe=es_sub, es_sharpness=es_sharp,
            edv=edv, esv=esv, ef=ef, ef_gating=ef_gating,
            lv0=float(lv[0]),                       # frame-0 LV volume (gating-ED)
            cf=cf.tolist(), v_rel=v_rel.tolist(),
        ))

    N = len(per_subj)
    es_frames = np.array([p["es_frame"] for p in per_subj])
    es_sub = np.array([p["es_subframe"] for p in per_subj])
    es_sharp = np.array([p["es_sharpness"] for p in per_subj])
    ed_frames = np.array([p["ed_frame"] for p in per_subj])
    cf_mat = np.array([p["cf"] for p in per_subj])        # (N, 12)
    vrel_mat = np.array([p["v_rel"] for p in per_subj])

    # per-phase cross-subject spread
    cf_mean = cf_mat.mean(0); cf_std = cf_mat.std(0)
    vrel_mean = vrel_mat.mean(0); vrel_std = vrel_mat.std(0)

    es_hist = {int(k): int((es_frames == k).sum()) for k in range(NUM_PHASES)}
    ed_hist = {int(k): int((ed_frames == k).sum()) for k in range(NUM_PHASES)}

    summary = dict(
        n_subjects_total=len(subjects),
        n_clean=N,
        skipped=skipped,
        ef_mean=float(np.mean([p["ef"] for p in per_subj])),
        ef_std=float(np.std([p["ef"] for p in per_subj])),
        ef_gating_mean=float(np.nanmean([p["ef_gating"] for p in per_subj])),
        ef_gating_std=float(np.nanstd([p["ef_gating"] for p in per_subj])),
        # --- ES frame (is ES fixed?) ---
        es_frame_hist=es_hist,
        es_frame_mean=float(es_frames.mean()),
        es_frame_std=float(es_frames.std()),
        es_frame_min=int(es_frames.min()),
        es_frame_max=int(es_frames.max()),
        es_frac_mean=float(es_frames.mean() / NUM_PHASES),
        es_frac_min=float(es_frames.min() / NUM_PHASES),
        es_frac_max=float(es_frames.max() / NUM_PHASES),
        # robust sub-frame ES (guards argmin-on-flat-trough) + trough sharpness
        es_subframe_mean=float(es_sub.mean()),
        es_subframe_std=float(es_sub.std()),
        es_subframe_frac_std=float((es_sub / NUM_PHASES).std()),
        es_sharpness_mean=float(es_sharp.mean()),
        es_sharpness_median=float(np.median(es_sharp)),
        es_near_tie_frac=float((es_sharp < 0.05).mean()),   # trough within 5% of range = ambiguous
        es_frame_iqr=[float(np.percentile(es_frames, 25)), float(np.percentile(es_frames, 75))],
        # --- ED frame (should be ~0 by gating) ---
        ed_frame_hist=ed_hist,
        ed_frame_is0_frac=float((ed_frames == 0).mean()),
        # --- per-phase state spread (does target_t = fixed state?) ---
        cf_mean=cf_mean.tolist(), cf_std=cf_std.tolist(),
        vrel_mean=vrel_mean.tolist(), vrel_std=vrel_std.tolist(),
    )

    with open(args.out_json, "w") as f:
        json.dump(dict(summary=summary, per_subj=per_subj), f, indent=2)

    with open(args.out_csv, "w") as f:
        f.write("subj,ed_frame,es_frame,edv_ml,esv_ml,ef_pct," +
                ",".join(f"lv_t{t:02d}" for t in range(NUM_PHASES)) + "\n")
        for p in per_subj:
            f.write(f"{p['subj']},{p['ed_frame']},{p['es_frame']},{p['edv']:.2f},"
                    f"{p['esv']:.2f},{p['ef']:.2f}," +
                    ",".join(f"{v:.2f}" for v in p["lv"]) + "\n")

    # console summary
    print(f"clean subjects: {N}/{len(subjects)}  (skipped {len(skipped)})")
    for s, r in skipped[:20]:
        print("  SKIP", s, r)
    print(f"\nEF(argmax-ED): mean={summary['ef_mean']:.1f}% std={summary['ef_std']:.1f}%  "
          f"EF(gating-ED=frame0): mean={summary['ef_gating_mean']:.1f}% std={summary['ef_gating_std']:.1f}%")
    print(f"ED frame == 0 (gating): {summary['ed_frame_is0_frac']*100:.1f}% of subjects")
    print(f"\nES frame: mean={summary['es_frame_mean']:.2f} std={summary['es_frame_std']:.2f} "
          f"range=[{summary['es_frame_min']},{summary['es_frame_max']}] IQR={summary['es_frame_iqr']} "
          f"frac=[{summary['es_frac_min']:.2f},{summary['es_frac_max']:.2f}]")
    print(f"ES sub-frame (robust): mean={summary['es_subframe_mean']:.2f} std={summary['es_subframe_std']:.2f} "
          f"(frac-time std={summary['es_subframe_frac_std']:.3f})")
    print(f"ES trough sharpness: median={summary['es_sharpness_median']:.3f}  "
          f"near-tie(<5%) fraction={summary['es_near_tie_frac']*100:.1f}%  "
          f"(low near-tie => argmin-ES is well-determined, drift is real not noise)")
    print("ES frame histogram:")
    for k in range(NUM_PHASES):
        print(f"  t{k:2d}: {es_hist[k]:4d}  {'*'*es_hist[k]}")
    print("\nper-phase contraction-fraction (cf): 1=ED-like, 0=ES-like")
    print("  t  : mean  std  (std = cross-subject STATE spread at this target_t)")
    for t in range(NUM_PHASES):
        print(f"  t{t:2d}: {cf_mean[t]:.3f} {cf_std[t]:.3f}")


if __name__ == "__main__":
    main()
