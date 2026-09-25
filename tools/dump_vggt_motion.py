#!/usr/bin/env python
"""Re-run a VGGT arm on a few eval subjects and dump its predicted motion field for EVERY target.

`run_vggt.py` keeps the per-slot displacement Δ only for ED_PHASE (`ed_dvf.npz`). This replays the
exact scored run (same checkpoint, `make_dataset`, `name_seed` draw, frozen breath/ bundle, pinned
scatter companions) through `run_vggt.reconstruct`, once per target t with ED_PHASE set to t, and
writes Δ as NIfTI. Each target's splatted volume is compared against the scored
`recon_breath/vol_t{t}` so a replay drift cannot pass silently.

Output per subject, under --out/<subject>/:
  motion_t{tt}.nii.gz  (518, 518, D, 3) float32, mm. Axes (x, y, z-plane), last dim = (dx, dy, dz)
                       of the predicted point for each input pixel of the slice at that plane.
                       Affine: in-plane spacing (256-1)*1.4/(518-1) mm, dz = subject pitch, origin 0 —
                       the same physical frame as recon_breath/vol_t*.nii.gz.
  motion_meta.json     slot order, per-target replay check, applied breathing shift per plane.

Run (GPU node):
  micromamba run -n svr env PYTHONPATH=training:. python tools/dump_vggt_motion.py \
      --out scratch/qual_fig --subjects cmrx2023_af12:CMRx23_Train_P108 ...
"""
import argparse
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
import run_vggt as rv  # noqa: E402  (sets up training/ + evaluation/ on sys.path)
paths = rv.paths

CKPT = "scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt"
ARM = "vggt_final518_diff1000_ep300"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True, help="cohort:subject pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--arm", default=ARM, help="scored arm to check the replay against")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    model, cfg = rv.load_model_from_run(a.ckpt, device=dev)
    splat_res = ((cfg.get("loss") or {}).get("volume") or {}).get("splat_res")

    for pair in a.subjects:
        ds_name, subject = pair.split(":", 1)
        subj_dir = str(paths.subject_dir(ds_name, subject))
        man = json.load(open(paths.manifest(ds_name, subject)))
        T, dz = man["T"], float(man["dz_mm"])
        disp = np.asarray(man["breath"]["disp_dhw_mm"], dtype=np.float64)
        bundle = rv.load_bundle(subj_dir, T, "breath")
        seq = rv.name_seed(man.get("source", ds_name), subject)
        od = os.path.join(a.out, subject)
        os.makedirs(od, exist_ok=True)
        meta = {"cohort": ds_name, "subject": subject, "ckpt": a.ckpt, "arm_checked": a.arm,
                "units": "mm", "axes": "(x, y, z_plane, [dx, dy, dz])",
                "applied_disp_dhw_mm_per_plane": disp.tolist(), "targets": {}}

        with tempfile.TemporaryDirectory() as tmp:
            dset = rv.make_dataset(cfg, man["rel_path"], man["split"], tmp)
            batch = rv.prepare_batch(dset, seq, bundle, dev, man["dz_mm"], man["scatter"])
            for t in range(T):
                rv.ED_PHASE = t   # reconstruct() keeps Δ only for the target equal to ED_PHASE
                vols, _, pack = rv.reconstruct(model, dset, seq, bundle, dev, disp, dz_bundle=man["dz_mm"],
                                               scatter=man["scatter"], splat_res=splat_res,
                                               phases=[t], batch=batch)
                # Replay check vs the scored recon (saved as (X,Y,Z) = pred.transpose(2,1,0)).
                saved = np.asarray(nib.load(str(paths.recon(ds_name, subject, a.arm, "breath", t))).dataobj,
                                   np.float32)
                diff = float(np.abs(vols[0].transpose(2, 1, 0) - saved).max())

                z = np.rint(pack["slot_z"]).astype(int)
                D = len(z)
                assert sorted(z.tolist()) == list(range(D)), z
                mm = pack["delta"] * np.asarray(rv.MM_PER_NORM, np.float32)       # (S, R_y, R_x, 3)
                R = mm.shape[1]
                field = np.zeros((R, R, D, 3), np.float32)
                for s, zp in enumerate(z):
                    field[:, :, zp, :] = mm[s].transpose(1, 0, 2)                   # -> (x, y, 3)
                sp = (256 - 1) * rv.INPLANE_MM / (R - 1)
                nib.save(nib.Nifti1Image(field, np.diag([sp, sp, dz, 1.0])),
                         os.path.join(od, f"motion_t{t:02d}.nii.gz"))
                meta["targets"][t] = {"slot_z": z.tolist(), "slot_t": pack["slot_t"].tolist(),
                                      "replay_max_abs_diff_vs_scored": diff}
                print(f"  {subject} t={t:02d} replay max|diff|={diff:.2e}", flush=True)

        # Cross-check the t=0 field against the scored run's own ed_dvf.npz (ED_PHASE=0 there).
        ed = np.load(os.path.join(str(paths.arm_dir(ds_name, subject, a.arm)), "ed_dvf.npz"))
        f0 = np.asarray(nib.load(os.path.join(od, "motion_t00.nii.gz")).dataobj)
        ref = ed["delta"].astype(np.float32) * np.asarray(rv.MM_PER_NORM, np.float32)
        zz = np.rint(ed["slot_z"]).astype(int)
        meta["ed_dvf_max_abs_diff_mm"] = float(max(np.abs(f0[:, :, zp, :] - ref[s].transpose(1, 0, 2)).max()
                                                   for s, zp in enumerate(zz)))
        print(f"  {subject} t=00 vs ed_dvf.npz max|diff|={meta['ed_dvf_max_abs_diff_mm']:.3f} mm", flush=True)
        json.dump(meta, open(os.path.join(od, "motion_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
