"""Multi-model gated breathing-sim viz sweep: gather05 + control0, on MIITT + OCMR + ACDC gated
SAX cine, x {clean, normal}. 1-frame, clean-reference, dynamic breathing seed 72. Captions report
the SI (through-plane) breathing amplitude (NOT the 3-vector magnitude). Subsamples the cardiac
cycle to MAX_PHASES frames for speed.

Organized output:
  result/gated_model_sweep/<model>/<dataset>/<subject>/<condition>_7row.{gif,npz}
  result/gated_model_sweep/{metrics.json, summary.csv}

Reuses capture()/render_7row() from gated_gather05_7row (importable — __main__ guard). Resumable.
Run: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/gated_model_sweep.py
"""
import os, sys, glob, json, time, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "training")
from inference.adapters import MIITTGatedAdapter, OCMRAdapter, ACDCGatedAdapter
from inference.inference import load_rtfb_model_reference
from inference.run_gated_ood import load_rcfg
from tools.miitt_viz.gated_gather05_7row import capture, render_7row

DEV = "cuda"; OUT = "result/gated_model_sweep"; REGIME = "1frame"; CLEAN_REF = True
SEED = 72          # dynamic breathing seed (breathing spread across planes, ref plane ~0)
MAX_PHASES = 18    # cardiac-cycle frames (subsample of T) — cuts forward + render time
MODELS = [
    ("gather05", glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0]),
    ("control0", glob.glob("scratch/logs/216539845_*ftctrl_gather0*1frame*/ckpts/checkpoint_last.pt")[0]),
]


def dataset_list():
    subs = []
    for i in range(1, 6):
        p = f"scratch/data/MIITT/nifti/Volunteer{i}/gated/sax/4d_recon.nii.gz"
        if os.path.exists(p):
            subs.append(("miitt", f"Volunteer{i}", (lambda p=p: MIITTGatedAdapter(p))))
    for f in sorted(glob.glob("scratch/data/ocmr/recon/gated/*/*/sax_cine.nii.gz"))[:5]:
        d = os.path.dirname(f)
        subs.append(("ocmr", os.path.basename(d), (lambda d=d: OCMRAdapter(d))))
    for i in (1, 2, 3, 4):
        p = f"scratch/data/ACDC/training/patient{i:03d}/patient{i:03d}_4d.nii.gz"
        if os.path.exists(p):
            subs.append(("acdc", f"patient{i:03d}", (lambda p=p: ACDCGatedAdapter(p))))
    return subs


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = dataset_list()
    rn = load_rcfg()                                          # normal breathing (amp 16)
    conds = [("clean", False), ("normal", True)]
    print(f"{len(MODELS)} models x {len(subs)} subjects x {len(conds)} conds "
          f"(1frame cleanref seed{SEED} maxphases{MAX_PHASES})", flush=True)

    bundles = {}
    for ds, subj, mk in subs:
        try:
            bundles[(ds, subj)] = mk().build_canonical_bundle()
        except Exception as e:
            print(f"SKIP bundle {ds}/{subj}: {e}", flush=True)

    summary = []
    for mname, ckpt in MODELS:
        print(f"\n########## MODEL {mname}: {ckpt} ##########", flush=True)
        model = load_rtfb_model_reference(ckpt, refiner=False, device=DEV)
        for ds, subj, mk in subs:
            if (ds, subj) not in bundles:
                continue
            bundle_np, bbox = bundles[(ds, subj)]
            pb = torch.from_numpy(bundle_np).to(DEV); T = bundle_np.shape[0]
            odir = os.path.join(OUT, mname, ds, subj); os.makedirs(odir, exist_ok=True)
            print(f"=== {mname} {ds}/{subj}  T={T}  bbox={bbox.tolist()} ===", flush=True)
            for tag, breathing in conds:
                base = os.path.join(odir, f"{tag}_7row")
                if os.path.exists(base + ".npz"):
                    print(f"  [{tag:6s}] skip (exists)", flush=True); continue
                t0 = time.time()
                cap = capture(model, pb, bbox, breathing, rn, REGIME, CLEAN_REF,
                              seq_index=SEED, max_phases=MAX_PHASES)
                t_fwd = time.time() - t0
                si = np.abs(cap["rd"][:, 0])                  # SI (through-plane) amplitude per slot, mm
                si_scat = si[1:]                              # exclude reference (slot 0)
                si_mean = float(si_scat.mean()) if si_scat.size else 0.0
                si_max = float(si_scat.max()) if si_scat.size else 0.0
                n_breath = int((si_scat > 8).sum())
                means = {k: float(np.nanmean(v)) for k, v in cap["metr"].items()}
                alab = ("clean (no breathing)" if not breathing
                        else f"normal: SI breathing mean {si_mean:.1f} max {si_max:.1f}mm, {n_breath} planes>8mm (ref clean)")
                t1 = time.time()
                render_7row(cap, f"{mname} 1frame | {ds}/{subj} | {alab}", base + ".gif")
                t_render = time.time() - t1
                np.savez_compressed(base + ".npz", gt=cap["GT"], recon=cap["RE"], inp=cap["IN"],
                                    dvf=cap["DV"], cov=cap["CO"], has_slot=np.array(cap["has_slot"]),
                                    ref_zmid=cap["z_mid"], zbr=cap["rd"][:, 0], sop=np.array(cap["sop"]),
                                    applied_disp=cap["rd"], per_phase_motion=np.array(cap["metr"]["motion"]),
                                    per_phase_full=np.array(cap["metr"]["full"]),
                                    per_phase_ssim=np.array(cap["metr"]["ssim"]))
                summary.append(dict(model=mname, dataset=ds, subject=subj, condition=tag,
                                    si_breath_mean_mm=si_mean, si_breath_max_mm=si_max,
                                    n_planes_breathing=n_breath, mean=means))
                print(f"  [{tag:6s}] SI mean={si_mean:.1f} max={si_max:.1f}mm ({n_breath}pl>8) | "
                      f"motion={means['motion']:.2f} full={means['full']:.2f}dB ssim={means['ssim']:.3f} | "
                      f"fwd={t_fwd:.0f}s render={t_render:.0f}s -> {base}.gif", flush=True)
            del pb; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "summary.csv"), "w") as f:
        f.write("model,dataset,subject,condition,si_breath_max_mm,n_planes,motion,full,ssim\n")
        for r in summary:
            m = r["mean"]
            f.write(f"{r['model']},{r['dataset']},{r['subject']},{r['condition']},{r['si_breath_max_mm']:.1f},"
                    f"{r['n_planes_breathing']},{m['motion']:.2f},{m['full']:.2f},{m['ssim']:.3f}\n")
    print("\n=== SUMMARY ===", flush=True)
    for r in summary:
        m = r["mean"]
        print(f"{r['model']:9s} {r['dataset']:5s} {r['subject']:16s} {r['condition']:6s} "
              f"SImax={r['si_breath_max_mm']:4.0f}mm({r['n_planes_breathing']}pl) "
              f"motion={m['motion']:.2f} full={m['full']:.2f}dB ssim={m['ssim']:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
