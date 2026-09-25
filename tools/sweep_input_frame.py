#!/usr/bin/env python
"""For a non-reference plane z of a subject, feed that plane each of its T real-time frames (every other slot
keeps the frozen scatter draw) and measure the predicted motion at the ED and ES targets (max / min GT LV
volume). Used to pick an input frame that moves visibly toward BOTH targets for the ED/ES motion figure
(score = min(ED motion, ES motion)).

Always writes --out/scores/<rhythm>/<subject>_z{z}.json: per frame, mean in-plane motion (mm, heart ROI,
rigid heart-ROI shift removed) toward ED and ES. With --save (all frames) or --save-frame F (one frame), also
writes --out/<rhythm>/<subject>_z{z}f{f}/motion_t{tt}.nii.gz + motion_meta.json, the tools/dump_vggt_motion.py
format (ED and ES targets only), which tools/render_motion_edes.py reads as '<source>/<subject>,z,<subject>_z{z}f{f}'.
Fields are ~32 MB per target, so score first and save only the picked frames.

Run (GPU node; one process per GPU, split the candidate list):
  micromamba run -n svr env PYTHONPATH=training:. python tools/sweep_input_frame.py \
      --out scratch/motion_edes/sweep --subjects mnms:MNMs_O5U2U7:4 cmrx2024:CMRx24_Train_P055:3
"""
import argparse
import json
import os
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
import run_vggt as rv  # noqa: E402
paths = rv.paths

CKPT = "scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt"
ARM = "vggt_final518_diff1000_ep300"


def sweep(model, cfg, splat_res, dev, src, subject, z, a):
    for rh in ("af12", "hrv12"):
        ds_name = f"{src}_{rh}"
        subj_dir = str(paths.subject_dir(ds_name, subject))
        man = json.load(open(paths.manifest(ds_name, subject)))
        T, dz = man["T"], float(man["dz_mm"])
        disp = np.asarray(man["breath"]["disp_dhw_mm"], dtype=np.float64)
        bundle = rv.load_bundle(subj_dir, T, "breath")
        seq = rv.name_seed(man.get("source", ds_name), subject)
        curve = np.array(next(r for r in json.load(open(
            f"{ROOT}/evaluation/metric_results/test/{ds_name}/ef/{ARM}.json"))["per_subject"]
            if r["subject"] == subject)["lv_curve_gt"])
        targets = {"ED": int(curve.argmax()), "ES": int(curve.argmin())}
        heart = nib.load(f"{subj_dir}/mask_heart.nii.gz").get_fdata() > 0
        m = zoom(heart[:, :, z].T.astype(np.uint8), 518 / 256, order=0) > 0
        scores = {"cohort": ds_name, "subject": subject, "z": z, "targets": targets, "frames": {}}
        with tempfile.TemporaryDirectory() as tmp:
            dset = rv.make_dataset(cfg, man["rel_path"], man["split"], tmp)
            batch = rv.prepare_batch(dset, seq, bundle, dev, man["dz_mm"], man["scatter"])
            z_slots = [int(round(float(v))) for v in batch["slice_indices"][0].tolist()]
            s_sel = z_slots.index(z)
            assert s_sel > 0, "plane is the reference slice"
            frozen = int(batch["timesteps"][0, s_sel])
            scores["frozen_frame"] = frozen
            for f in range(T):
                batch["timesteps"][0, s_sel] = f
                save = a.save or a.save_frame == f
                od = os.path.join(a.out, rh, f"{subject}_z{z}f{f}")
                meta = {"cohort": ds_name, "subject": subject, "ckpt": CKPT, "units": "mm",
                        "axes": "(x, y, z_plane, [dx, dy, dz])",
                        "override": {"z": z, "frame": f, "frozen_frame": frozen}, "targets": {}}
                mags = {}
                for kind, t in targets.items():
                    rv.ED_PHASE = t   # reconstruct() keeps Δ only for the target equal to ED_PHASE
                    _, _, pack = rv.reconstruct(model, dset, seq, bundle, dev, disp, dz_bundle=man["dz_mm"],
                                                scatter=None, splat_res=splat_res, phases=[t], batch=batch)
                    assert int(pack["slot_t"][s_sel]) == f
                    mm = pack["delta"] * np.asarray(rv.MM_PER_NORM, np.float32)   # (S, R_y, R_x, 3)
                    g = mm[s_sel][..., :2]
                    g = g - g[m].mean(0)
                    mags[kind] = float(np.hypot(g[..., 0], g[..., 1])[m].mean())
                    if save:
                        zz = np.rint(pack["slot_z"]).astype(int)
                        R = mm.shape[1]
                        field = np.zeros((R, R, len(zz), 3), np.float32)
                        for s, zp in enumerate(zz):
                            field[:, :, zp, :] = mm[s].transpose(1, 0, 2)
                        sp = (256 - 1) * rv.INPLANE_MM / (R - 1)
                        os.makedirs(od, exist_ok=True)
                        nib.save(nib.Nifti1Image(field, np.diag([sp, sp, dz, 1.0])),
                                 os.path.join(od, f"motion_t{t:02d}.nii.gz"))
                        meta["targets"][t] = {"slot_z": zz.tolist(), "slot_t": pack["slot_t"].tolist()}
                if save:
                    json.dump(meta, open(os.path.join(od, "motion_meta.json"), "w"), indent=1)
                scores["frames"][f] = mags
                print(f"{rh} {subject} z{z} frame {f:2d}{' (frozen)' if f == frozen else '         '} "
                      f"ED {mags['ED']:.1f} mm  ES {mags['ES']:.1f} mm", flush=True)
        sd = os.path.join(a.out, "scores", rh)
        os.makedirs(sd, exist_ok=True)
        json.dump(scores, open(os.path.join(sd, f"{subject}_z{z}.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True,
                    help="source:subject:z[:frame], e.g. mnms:MNMs_O5U2U7:4 (frame overrides --save-frame)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", action="store_true", help="also save the motion fields of every frame")
    ap.add_argument("--save-frame", type=int, help="also save the motion fields of this frame only")
    a = ap.parse_args()

    dev = torch.device("cuda:0")
    model, cfg = rv.load_model_from_run(CKPT, device=dev)
    splat_res = ((cfg.get("loss") or {}).get("volume") or {}).get("splat_res")
    for spec in a.subjects:
        src, subject, z, *f = spec.split(":")   # optional 4th field: per-subject --save-frame
        sweep(model, cfg, splat_res, dev, src, subject, int(z),
              argparse.Namespace(**{**vars(a), "save_frame": int(f[0]) if f else a.save_frame}))


if __name__ == "__main__":
    main()
