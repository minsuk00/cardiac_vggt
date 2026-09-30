"""E8 — the mechanism on REAL free-breathing real-time data (paper Sec. 4.4, docs/124).

Replicates the paper's frame-0 cine protocol (companions frozen at their frame 0, reference = mid
plane swept over all 180 real frames; `run_vggt_rt.reconstruct_rt` + docs/124 §2 wrapper) and adds
global-attention interventions:
  full          unmodified (must reproduce scratch/temp/miitt_afib_rt/vggt_frame0/*/recon_rt.nii.gz)
  ko_ref_L3     companions may not attend to any reference token in global layer 3 only
  self_L12_17   no cross-slice attention in global layers 12-17
Writes <out>/<subj>/<arm>/recon_rt.nii.gz ((X,Y,Z,T) canonical grid, same as run_vggt_rt).
LV curves: tools/rt_lv_curve_2d.py dump/seg (nnU-Net 2d), then e8_score.py.
"""
import argparse
import os
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(H.ROOT, "evaluation", "src", "analysis"))
import run_vggt_rt as rt    # noqa: E402
from data.gpu_aug import gpu_augment_batch   # noqa: E402
from e2_breathing import slot_allow          # noqa: E402

SUBJECTS = ["MIITT_Patient_2024Jan04_Cardiomyopathy_AFib", "MIITT_Volunteer1"]


@torch.no_grad()
def run(model, cfg, subj, out, dev, splat_res):
    rt_path = os.path.join(rt.RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax/4d_recon.nii.gz")
    bundle = rt.build_rt_bundle(rt_path)
    n_frames, D = bundle.shape[:2]
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = rt.rv.make_dataset(cfg, rt.make_rt_scaffold(rt_path, subj, tmpdir), "val", tmpdir)
        b0 = ds.get_data(seq_index=0, img_per_seq=ds.num_slices)
        dz_mm = float(np.asarray(b0["dz_mm"]).reshape(-1)[0])
        batch = rt.build_batch_rt(ds, rt.rv.name_seed("miitt", subj), bundle, dev)
    batch["timesteps"][0, 1:] = 0                      # docs/124: companions frozen at frame 0
    S = batch["timesteps"].shape[1]
    z_scale = float(batch["z_scale"].reshape(-1)[0])
    ko3 = torch.ones(2 * S, 2 * S, dtype=torch.bool); ko3[2:, 0:2] = False
    eye = slot_allow(torch.eye(S, dtype=torch.bool))
    arms = {"full": (None, None), "ko_ref_L3": (ko3, {3}), "self_L12_17": (eye, set(range(12, 18)))}
    fake = SimpleNamespace(R=int(batch["scanner_coords"].shape[-2]))
    ref_z = int(round(float(batch["slice_indices"][0, 0])))
    for arm, (allow, layers) in arms.items():
        vols = []
        t0 = time.time()
        for f in range(n_frames):
            batch["timesteps"][0, 0] = f
            batch.pop("images", None)
            gpu_augment_batch(batch, None, dev, respiratory_cfg=None, train=False)
            b = {"images": batch["images"], "z_indices": batch["z_indices"], "scanner_coords": batch["scanner_coords"]}
            wp = H.forward(model, fake, b, allow=allow, layers=layers)["wp"]
            V, _ = rt._splat_preds_native({"world_points": wp}, batch, (D, 256, 256), z_scale, splat_res=splat_res)
            vols.append(V[0].float().cpu().numpy())
        od = os.path.join(out, subj, arm)
        os.makedirs(od, exist_ok=True)
        rt.save_cine_xyzt(os.path.join(od, "recon_rt.nii.gz"), np.stack(vols), dz_mm)
        with open(os.path.join(od, "meta.txt"), "w") as fh:
            fh.write(f"ref_z={ref_z}\nD={D}\ndz_mm={dz_mm}\nn_frames={n_frames}\n")
        print(f"{subj} {arm}: {n_frames} frames {time.time() - t0:.0f}s ref_z={ref_z}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    model, cfg = H.load("cuda:0")
    splat_res = ((cfg.get("loss") or {}).get("volume") or {}).get("splat_res")
    for s in args.subjects:
        run(model, cfg, s, args.out, dev, splat_res)


if __name__ == "__main__":
    main()
