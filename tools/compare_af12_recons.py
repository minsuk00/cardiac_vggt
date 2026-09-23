"""Compare a new VGGT af12 arm against the existing vggt_final518_diff1000_ep300 recons, every
subject x phase (docs/120). Read-only.

    PYTHONPATH=training:. python tools/compare_af12_recons.py vggt_final518_diff1000_ep300_3gpu

Per cohort: phases compared / missing, max|diff|, mean|diff|, worst-phase PSNR between old and new
(peak = 1; recon intensities are unit-normalized), and whether ckpt / git commit / splat_res / slot
draw match. "Essentially the same" = diffs at float32-rounding scale (~1e-6), which is what 1-GPU
runs of the same code already show run-to-run (the splat's scatter_add_ order is nondeterministic).
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import paths  # noqa: E402

OLD = "vggt_final518_diff1000_ep300"
NEW = sys.argv[1]
tot_n = tot_miss = 0
for src in ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms"):
    ds = f"{src}_af12"
    mx, sm, nvox, n, miss, worst, notes = 0.0, 0.0, 0, 0, 0, np.inf, set()
    for subj in sorted(p.name for p in paths.dataset_root(ds).iterdir() if (p / OLD).is_dir()):
        if not paths.metadata(ds, subj, NEW).exists():
            miss += json.load(open(paths.manifest(ds, subj)))["T"]
            continue
        mo, mn = (json.load(open(paths.metadata(ds, subj, a))) for a in (OLD, NEW))
        for k in ("ckpt_fingerprint", "git_commit", "splat_res"):
            if mo.get(k) != mn.get(k):
                notes.add(f"{k} differs")
        if mo.get("draw", {}).get("slot_t") != mn.get("draw", {}).get("slot_t"):
            notes.add(f"slot draw differs ({subj})")
        for t in range(json.load(open(paths.manifest(ds, subj)))["T"]):
            a = np.asarray(nib.load(paths.recon(ds, subj, OLD, "breath", t)).dataobj, dtype=np.float64)
            b = np.asarray(nib.load(paths.recon(ds, subj, NEW, "breath", t)).dataobj, dtype=np.float64)
            assert a.shape == b.shape, (ds, subj, t, a.shape, b.shape)
            d = np.abs(a - b)
            mx, sm, nvox, n = max(mx, d.max()), sm + d.sum(), nvox + d.size, n + 1
            mse = (d ** 2).mean()
            worst = min(worst, np.inf if mse == 0 else 10 * np.log10(1.0 / mse))
    tot_n, tot_miss = tot_n + n, tot_miss + miss
    print(f"{ds:15s} phases={n:4d} missing={miss:3d}  max|diff|={mx:.2e}  "
          f"mean|diff|={sm / max(nvox, 1):.2e}  worst-phase PSNR(old,new)={worst:.1f} dB  "
          f"{'; '.join(sorted(notes)) or 'metadata ok'}")
print(f"TOTAL phases compared {tot_n}, missing {tot_miss}")
