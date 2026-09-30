"""E15 — the paper's AF48 simulated input (tools/build_af_bundle.py arm af48) reconstructed under
full / ko_ref_L3 / self_L12_17, frame by frame, with the paper's own batch path (run_vggt.prepare_batch:
frozen breathed pixels, companions pinned to the bundle's scatter draw, reference = frame f).

Writes <out>/<arm>/recon_t{f:02d}.nii.gz (X,Y,Z) and, for nnU-Net, <out>/seg_in/<arm>__t{f:02d}_0000.nii.gz
multiplied by the bundle's mask_heart_pad10 (the paper's segmentation crop, ef_dice._dump_cine), plus the
GT frames the same way (arm 'gt'). Shard frames with --shard k --nshard n (one GPU per shard).
"""
import argparse
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nibabel as nib       # noqa: E402
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src"))
import run_vggt as rv       # noqa: E402
from e2_breathing import slot_allow   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="cmrx2024_af48")
    ap.add_argument("--subject", default="CMRx24_Test_P017")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-root", default=os.path.join(H.ROOT, "scratch/eval"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    sd = os.path.join(args.eval_root, args.cohort, "out", args.subject)
    man = json.load(open(os.path.join(sd, "manifest.json")))
    T, D, dz = man["T"], man["D"], man["dz_mm"]
    roi = np.asarray(nib.load(os.path.join(sd, "mask_heart_pad10.nii.gz")).dataobj) > 0.5      # (X,Y,Z)
    aff = np.diag([rv.INPLANE_MM, rv.INPLANE_MM, dz, 1.0])
    os.makedirs(os.path.join(args.out, "seg_in"), exist_ok=True)
    frames = list(range(T))[args.shard::args.nshard]

    if args.shard == 0:                                   # GT frames, same crop
        for f in range(T):
            g = np.asarray(nib.load(os.path.join(sd, "gt", f"gt_t{f:02d}.nii.gz")).dataobj, np.float32)
            nib.save(nib.Nifti1Image(g * roi, aff), os.path.join(args.out, "seg_in", f"gt__t{f:02d}_0000.nii.gz"))

    dev = torch.device("cuda:0")
    model, cfg = H.load("cuda:0")
    splat_res = ((cfg.get("loss") or {}).get("volume") or {}).get("splat_res")
    bundle = rv.load_bundle(sd, T, "breath")
    with tempfile.TemporaryDirectory() as tmp:
        ds = rv.make_dataset(cfg, man["rel_path"], man.get("split", "test"), tmp)
        seq = rv.name_seed(man.get("source", args.cohort), args.subject)
        batch = rv.prepare_batch(ds, seq, bundle, dev, dz_bundle=dz, scatter=man["scatter"])
    S = batch["timesteps"].shape[1]
    z_scale = float(batch["z_scale"].reshape(-1)[0])
    ko3 = torch.ones(2 * S, 2 * S, dtype=torch.bool); ko3[2:, 0:2] = False
    eye = slot_allow(torch.eye(S, dtype=torch.bool))
    arms = {"full": (None, None), "ko_ref_L3": (ko3, {3}), "self_L12_17": (eye, set(range(12, 18)))}
    fake = SimpleNamespace(R=int(batch["scanner_coords"].shape[-2]))
    for arm in arms:
        os.makedirs(os.path.join(args.out, arm), exist_ok=True)
    for f in frames:
        batch["timesteps"][0, 0] = int(f)
        rv._extract(batch, dev)
        b = {"images": batch["images"], "z_indices": batch["z_indices"], "scanner_coords": batch["scanner_coords"]}
        for arm, (allow, layers) in arms.items():
            wp = H.forward(model, fake, b, allow=allow, layers=layers)["wp"]
            V, _ = rv._splat_preds_native({"world_points": wp}, batch, (D, 256, 256), z_scale, splat_res=splat_res)
            v = np.ascontiguousarray(V[0].float().cpu().numpy().transpose(2, 1, 0))          # (X,Y,Z)
            nib.save(nib.Nifti1Image(v, aff), os.path.join(args.out, arm, f"recon_t{f:02d}.nii.gz"))
            nib.save(nib.Nifti1Image(v * roi, aff), os.path.join(args.out, "seg_in", f"{arm}__t{f:02d}_0000.nii.gz"))
        print(f"[shard {args.shard}] frame {f} done", flush=True)


if __name__ == "__main__":
    main()
