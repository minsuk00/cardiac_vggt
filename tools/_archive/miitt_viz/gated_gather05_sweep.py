"""gather05 1-frame CLEAN-REFERENCE diagnostic sweep over multiple gated subjects.
MIITT gated Volunteer1-5 + 5 OCMR gated subjects, each x {clean, normal(amp16), extreme(amp50)}.
Reference query kept breathing-clean (cleanref) so the extreme recon is readable — a DIAGNOSTIC
view (the canonical eval keeps the reference breathing; see gated_gather05_7row.py).

Organized output tree:
  result/gated_gather05_cleanref/<dataset>/<subject>/<condition>_7row.{gif,npz}
  result/gated_gather05_cleanref/metrics.json  (+ summary.csv)

Reuses the tested capture()/render_7row() from gated_gather05_7row (safe to import — __main__ guard).
Resumable: skips any condition whose npz already exists.

Run: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/gated_gather05_sweep.py
"""
import os, sys, glob, json, time, dataclasses, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "training")
from inference.adapters import MIITTGatedAdapter, OCMRAdapter
from inference.inference import load_rtfb_model_reference
from inference.run_gated_ood import load_rcfg
from tools.miitt_viz.gated_gather05_7row import capture, render_7row, CKPT

DEV = "cuda"; OUT = "result/gated_gather05_cleanref"; REGIME = "1frame"; CLEAN_REF = True
N_OCMR = 5
SEED = 72   # dynamic breathing seed (many planes caught mid-breath; ref plane naturally ~0). Viz only.


def build_subject_list():
    subs = []
    for i in range(1, 6):
        v = f"Volunteer{i}"
        p = f"scratch/data/MIITT/nifti/{v}/gated/sax/4d_recon.nii.gz"
        if os.path.exists(p):
            subs.append(("miitt", v, (lambda p=p: MIITTGatedAdapter(p))))
    ocmr = sorted(glob.glob("scratch/data/ocmr/recon/gated/*/*/sax_cine.nii.gz"))[:N_OCMR]
    for f in ocmr:
        d = os.path.dirname(f); subj = os.path.basename(d)
        subs.append(("ocmr", subj, (lambda d=d: OCMRAdapter(d))))
    return subs


def main():
    os.makedirs(OUT, exist_ok=True)
    subjects = build_subject_list()
    print(f"loading gather05: {CKPT}", flush=True)
    model = load_rtfb_model_reference(CKPT, refiner=False, device=DEV)
    rn = load_rcfg(); re_cfg = dataclasses.replace(rn, amplitude_mm=50.0)
    conds = [("clean", False, rn), ("normal", True, rn), ("extreme", True, re_cfg)]
    print(f"{len(subjects)} subjects x 3 conditions (1frame, clean_ref={CLEAN_REF})", flush=True)

    summary = []
    for ds, subj, make_adapter in subjects:
        try:
            bundle_np, bbox = make_adapter().build_canonical_bundle()
        except Exception as e:
            print(f"SKIP {ds}/{subj}: {e}", flush=True); continue
        pb = torch.from_numpy(bundle_np).to(DEV); T = bundle_np.shape[0]
        odir = os.path.join(OUT, ds, subj); os.makedirs(odir, exist_ok=True)
        print(f"\n=== {ds}/{subj}  T={T}  bbox={bbox.tolist()} ===", flush=True)
        for tag, breathing, rcfg in conds:
            base = os.path.join(odir, f"{tag}_7row")
            if os.path.exists(base + ".npz"):
                print(f"  [{tag:7s}] skip (exists)", flush=True); continue
            t0 = time.time()
            cap = capture(model, pb, bbox, breathing, rcfg, REGIME, CLEAN_REF, seq_index=SEED)
            t_fwd = time.time() - t0
            mag = np.linalg.norm(cap["rd"], axis=1)
            amp_mean, amp_max = float(mag.mean()), float(mag.max())
            means = {k: float(np.nanmean(v)) for k, v in cap["metr"].items()}
            n_breath = int((mag[1:] > 8).sum())   # #scattered planes breathing >8mm
            alab = ("clean (no breathing)" if not breathing
                    else f"{tag}: scattered |disp| mean {amp_mean:.1f} max {amp_max:.1f}mm, {n_breath} planes>8mm (ref CLEAN, seed{SEED})")
            t1 = time.time()
            vl = render_7row(cap, f"gather05 1frame CLEANREF | {ds}/{subj} | {alab}", base + ".gif")
            t_render = time.time() - t1
            np.savez_compressed(base + ".npz", gt=cap["GT"], recon=cap["RE"], inp=cap["IN"], dvf=cap["DV"],
                                cov=cap["CO"], has_slot=np.array(cap["has_slot"]), ref_zmid=cap["z_mid"],
                                zbr=cap["rd"][:, 0], sop=np.array(cap["sop"]), applied_disp=cap["rd"],
                                per_phase_motion=np.array(cap["metr"]["motion"]),
                                per_phase_bbox=np.array(cap["metr"]["bbox"]),
                                per_phase_full=np.array(cap["metr"]["full"]),
                                per_phase_ssim=np.array(cap["metr"]["ssim"]))
            summary.append(dict(dataset=ds, subject=subj, condition=tag,
                                scatter_amp_mean_mm=amp_mean, scatter_amp_max_mm=amp_max, mean=means))
            print(f"  [{tag:7s}] scat|disp| max={amp_max:.1f}mm ({n_breath}pl>8) | "
                  f"motion={means['motion']:.2f} bbox={means['bbox']:.2f} full={means['full']:.2f}dB "
                  f"ssim={means['ssim']:.3f} | fwd={t_fwd:.0f}s render={t_render:.0f}s -> {base}.gif", flush=True)
        del pb; torch.cuda.empty_cache()

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write("dataset,subject,condition,scatter_amp_max_mm,motion,bbox,full,ssim\n")
        for r in summary:
            m = r["mean"]
            f.write(f"{r['dataset']},{r['subject']},{r['condition']},{r['scatter_amp_max_mm']:.1f},"
                    f"{m['motion']:.2f},{m['bbox']:.2f},{m['full']:.2f},{m['ssim']:.3f}\n")
    print("\n=== SUMMARY (gather05 1frame cleanref sweep) ===", flush=True)
    for r in summary:
        m = r["mean"]
        print(f"{r['dataset']:5s} {r['subject']:16s} {r['condition']:7s} scatAmp={r['scatter_amp_max_mm']:4.0f}mm  "
              f"motion={m['motion']:.2f} bbox={m['bbox']:.2f} full={m['full']:.2f}dB ssim={m['ssim']:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
