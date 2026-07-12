"""s20contz gated breathing-sim viz sweep — the continuous-z twin of gated_model_sweep.py.

s20contz was trained MULTIFRAME (S=20) *with continuous_z=true* (off-grid slots at true physical
z, 2-plane interp). Here we eval it in the acquisition regime it was built for: **1 frame per
native slice at its TRUE fractional physical z** — no snap to the 12-plane grid, no collision-drop.
Each native acquired slice = exactly ONE slot at its own continuous canonical z; content is the
dense bundle interpolated between the two bracketing planes (identical to training's synthesis).
Slot 0 = the slice nearest mid-depth = the swept reference/query.

Datasets: MIITT + OCMR + ACDC gated SAX cine, x {clean, normal}. 1-frame-contz, clean-reference,
dynamic breathing seed 72. Captions report SI (through-plane) breathing amplitude.

Output: result/gated_model_sweep/s20contz/<dataset>/<subject>/<condition>_7row.{gif,npz}
Run: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/gated_s20contz_sweep.py
"""
import os, sys, glob, json, time, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "training")
from inference.adapters import MIITTGatedAdapter, OCMRAdapter, ACDCGatedAdapter
from inference.inference import load_rtfb_model_reference
from inference.run_gated_ood import load_rcfg
from tools.miitt_viz.gated_gather05_7row import capture, render_7row

DEV = "cuda"; OUT = "result/gated_model_sweep"; REGIME = "1frame_contz"; CLEAN_REF = True
SEED = 72          # dynamic breathing seed (breathing spread across planes, ref plane ~0)
MAX_PHASES = 18    # cardiac-cycle frames (subsample of T)
# Any reference-slot model can be run through the 1frame_contz regime. s20contz was TRAINED with
# continuous_z (matched eval); control0/gather05 are 1-frame models trained on SNAPPED /12 integer
# planes — running them on contz probes whether their Fourier z-embedding EXTRAPOLATES off-grid
# (expected to be shaky: z_embedder only ever saw the 12 integer z_norm values in training).
MODEL_CKPTS = {
    "s20contz": "scratch/logs/216949414_*s20contz*/ckpts/checkpoint_last.pt",
    "control0": "scratch/logs/216539845_*ftctrl_gather0_1frame*/ckpts/checkpoint_last.pt",
    "gather05": "scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt",
}
# Selected via argv (e.g. `... gated_s20contz_sweep.py control0`); default = s20contz (unchanged).
SELECTED = [m for m in sys.argv[1:] if m in MODEL_CKPTS] or ["s20contz"]


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
    ckpts = {m: glob.glob(MODEL_CKPTS[m])[0] for m in SELECTED}
    print(f"models={SELECTED}: {len(subs)} subjects x {len(conds)} conds "
          f"({REGIME} cleanref seed{SEED} maxphases{MAX_PHASES})", flush=True)
    for m, c in ckpts.items():
        print(f"  {m}: {c}", flush=True)

    # Continuous-z bundle (keeps ALL native slices, no collision-drop) + native slice positions
    # (the TRUE fractional-z coordinates fed as input). Both from the SAME adapter/cine slice order.
    # Built ONCE and shared across all selected models.
    bundles = {}
    for ds, subj, mk in subs:
        try:
            ad = mk()
            bundle_np, bbox = ad.build_canonical_bundle(continuous_z=True)
            positions = ad.slice_positions_mm()
            bundles[(ds, subj)] = (bundle_np, bbox, positions)
        except Exception as e:
            print(f"SKIP bundle {ds}/{subj}: {e}", flush=True)

    summary = []
    for MNAME in SELECTED:
        print(f"\n########## MODEL {MNAME}: {ckpts[MNAME]} ##########", flush=True)
        model = load_rtfb_model_reference(ckpts[MNAME], refiner=False, device=DEV)
        for ds, subj, mk in subs:
            if (ds, subj) not in bundles:
                continue
            bundle_np, bbox, positions = bundles[(ds, subj)]
            pb = torch.from_numpy(bundle_np).to(DEV); T = bundle_np.shape[0]
            # Snapped 1-frame models (control0/gather05) already have SNAPPED results under
            # result/gated_model_sweep/<model>/; write their contz-regime run to a distinct
            # <model>_contz/ dir so nothing collides. s20contz is inherently contz → keep as-is.
            out_name = MNAME if "contz" in MNAME else MNAME + "_contz"
            odir = os.path.join(OUT, out_name, ds, subj); os.makedirs(odir, exist_ok=True)
            print(f"=== {MNAME} {ds}/{subj}  T={T}  nslices={len(positions)}  bbox={bbox.tolist()} ===", flush=True)
            for tag, breathing in conds:
                base = os.path.join(odir, f"{tag}_7row")
                if os.path.exists(base + ".npz"):
                    print(f"  [{tag:6s}] skip (exists)", flush=True); continue
                t0 = time.time()
                cap = capture(model, pb, bbox, breathing, rn, REGIME, CLEAN_REF,
                              seq_index=SEED, max_phases=MAX_PHASES, positions=positions)
                t_fwd = time.time() - t0
                si = np.abs(cap["rd"][:, 0]); si_scat = si[1:]     # SI amplitude per slot (excl reference)
                si_mean = float(si_scat.mean()) if si_scat.size else 0.0
                si_max = float(si_scat.max()) if si_scat.size else 0.0
                n_breath = int((si_scat > 8).sum())
                means = {k: float(np.nanmean(v)) for k, v in cap["metr"].items()}
                alab = ("clean (no breathing)" if not breathing
                        else f"normal: SI breathing mean {si_mean:.1f} max {si_max:.1f}mm, {n_breath} planes>8mm (ref clean)")
                t1 = time.time()
                render_7row(cap, f"{MNAME} 1frame-contz | {ds}/{subj} | {alab}", base + ".gif", dpi=130)
                t_render = time.time() - t1
                np.savez_compressed(base + ".npz", gt=cap["GT"], recon=cap["RE"], inp=cap["IN"],
                                    dvf=cap["DV"], cov=cap["CO"], has_slot=np.array(cap["has_slot"]),
                                    ref_zmid=cap["z_mid"], zbr=cap["rd"][:, 0], sop=np.array(cap["sop"]),
                                    applied_disp=cap["rd"], per_phase_motion=np.array(cap["metr"]["motion"]),
                                    per_phase_full=np.array(cap["metr"]["full"]),
                                    per_phase_ssim=np.array(cap["metr"]["ssim"]))
                summary.append(dict(model=MNAME, dataset=ds, subject=subj, condition=tag,
                                    si_breath_mean_mm=si_mean, si_breath_max_mm=si_max,
                                    n_planes_breathing=n_breath, mean=means))
                print(f"  [{tag:6s}] SI mean={si_mean:.1f} max={si_max:.1f}mm ({n_breath}pl>8) | "
                      f"motion={means['motion']:.2f} full={means['full']:.2f}dB ssim={means['ssim']:.3f} | "
                      f"fwd={t_fwd:.0f}s render={t_render:.0f}s -> {base}.gif", flush=True)
            del pb; torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()

    tag_out = "_".join(SELECTED)
    with open(os.path.join(OUT, f"metrics_contz_{tag_out}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY (contz: {SELECTED}) ===", flush=True)
    for r in summary:
        m = r["mean"]
        print(f"{r['model']:9s} {r['dataset']:5s} {r['subject']:16s} {r['condition']:6s} "
              f"SImax={r['si_breath_max_mm']:4.0f}mm({r['n_planes_breathing']}pl) "
              f"motion={m['motion']:.2f} full={m['full']:.2f}dB ssim={m['ssim']:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
