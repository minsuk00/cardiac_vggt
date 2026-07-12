"""Test the NEW gated-OOD pipeline on the gather05 model, at normal (training-dist) and extreme
breathing. Loads gather05 ONCE, runs MIITT gated (Volunteer3) + OCMR gated (fs_0060), each under
three conditions: clean baseline / normal breathing (amp 16, training default) / extreme breathing
(amp 50, ~3x training). Reuses the prove-it-verified reconstruct_from_bundle + run_cmrxrecon viz.

Run: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/gated_gather05_breathing.py
"""
import dataclasses
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "training")
from inference.adapters import MIITTGatedAdapter, OCMRAdapter
from inference.adapters.base import GRID_SHAPE
from inference.inference import load_rtfb_model_reference
from inference.run_cmrxrecon import (
    reconstruct_from_bundle, save_multislice_gif, save_ed_montage, save_ed_input_png,
    save_ed_dvf_png, _nanmean, ED_PHASE,
)
from inference.run_gated_ood import load_rcfg

DEV = "cuda"
OUT = "result/gated_gather05_eval"
os.makedirs(OUT, exist_ok=True)
CKPT = glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0]

SUBJECTS = [
    ("miitt", "Volunteer3",
     lambda: MIITTGatedAdapter("scratch/data/MIITT/nifti/Volunteer3/gated/sax/4d_recon.nii.gz")),
    ("ocmr", "fs_0060",
     lambda: OCMRAdapter("scratch/data/ocmr/recon/gated/exam_fs_0060/sax__fs_0060_1_5T")),
]

print(f"loading gather05: {CKPT}", flush=True)
model = load_rtfb_model_reference(CKPT, refiner=False, device=DEV)
rcfg_normal = load_rcfg()                                             # amp 16 = training default
rcfg_extreme = dataclasses.replace(rcfg_normal, amplitude_mm=50.0)    # ~3x training ceiling
print(f"rcfg normal amp={rcfg_normal.amplitude_mm}  extreme amp={rcfg_extreme.amplitude_mm}", flush=True)

# (condition tag, breathing?, rcfg)
CONDITIONS = [
    ("clean", False, rcfg_normal),      # rcfg ignored when breathing=False
    ("normal", True, rcfg_normal),      # training-distribution breathing (amp 16)
    ("extreme", True, rcfg_extreme),    # out-of-distribution breathing (amp 50)
]

summary = []
for ds, subj, make_adapter in SUBJECTS:
    bundle_np, bbox = make_adapter().build_canonical_bundle()
    phases_bundle = torch.from_numpy(bundle_np).to(DEV)
    T = bundle_np.shape[0]
    print(f"\n=== {ds}/{subj}  T={T}  bbox={bbox.tolist()} ===", flush=True)
    for tag, breathing, rcfg in CONDITIONS:
        res = reconstruct_from_bundle(model, phases_bundle, bbox, rcfg, seq_index=0,
                                      breathing=breathing, device=DEV, grid_shape=GRID_SHAPE)
        base = os.path.join(OUT, f"{ds}_{subj}_{tag}")
        save_multislice_gif(res["pred_vols"], res["gt_vols"], res["bbox"], base + "_cycle.gif")
        save_ed_montage(res["pred_vols"][ED_PHASE], res["gt_vols"][ED_PHASE], base + "_ED.png")
        save_ed_input_png(res["ed_pack"], res["bbox"], res["z_mid"], base + "_ED_input.png")
        save_ed_dvf_png(res["ed_pack"], res["bbox"], res["z_mid"], base + "_ED_dvf.png", breathing)
        m = res["metrics"]; means = {k: _nanmean(v) for k, v in m.items()}
        # applied breathing amplitude (mean/max |disp| over slots), mm — 0 for clean
        resp = res["ed_pack"].get("resp_disp_mm")
        amp_mean = float(np.abs(resp).mean()) if resp is not None else 0.0
        amp_max = float(np.abs(resp).max()) if resp is not None else 0.0
        # Save raw arrays so ANY re-render (new layout/colormap, GT-vs-recon, ED DVF panels) is
        # CPU-only — no ~40-min GPU rerun. Holds full per-phase pred/GT cubes + the ED forward pack.
        ep = res["ed_pack"]
        np.savez_compressed(
            base + ".npz",
            pred_vols=res["pred_vols"], gt_vols=res["gt_vols"],          # (T,D,H,W) recon + GT
            bbox=res["bbox"], z_mid=res["z_mid"],
            ed_images=ep["images"], ed_delta=ep["delta"],                # ED input + predicted Δ (DVF)
            ed_slice_z=ep["slice_z"], ed_timesteps=ep["timesteps"],
            ed_resp_disp_mm=(resp if resp is not None else np.zeros((0, 3), np.float32)),
            per_phase_motion=np.array(m["motion"]), per_phase_bbox=np.array(m["bbox"]),
            per_phase_full=np.array(m["full"]), per_phase_ssim=np.array(m["ssim"]),
            applied_amp_mean_mm=amp_mean, applied_amp_max_mm=amp_max,
            dataset=ds, subject=subj, condition=tag, breathing=breathing)
        summary.append(dict(dataset=ds, subject=subj, condition=tag, breathing=breathing,
                            applied_amp_mean_mm=amp_mean, applied_amp_max_mm=amp_max,
                            mean=means, per_phase=m))
        print(f"  [{tag:7s}] applied|disp| mean={amp_mean:.1f} max={amp_max:.1f}mm | "
              f"motion={means['motion']:.2f} bbox={means['bbox']:.2f} full={means['full']:.2f}dB "
              f"ssim={means['ssim']:.3f} -> {base}_cycle.gif", flush=True)
    del phases_bundle
    torch.cuda.empty_cache()

with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY (gather05) ===", flush=True)
for r in summary:
    print(f"{r['dataset']:5s} {r['subject']:11s} {r['condition']:7s} "
          f"amp={r['applied_amp_max_mm']:4.0f}mm  motion={r['mean']['motion']:.2f}  "
          f"bbox={r['mean']['bbox']:.2f}  full={r['mean']['full']:.2f}dB  ssim={r['mean']['ssim']:.3f}", flush=True)
print("DONE", flush=True)
