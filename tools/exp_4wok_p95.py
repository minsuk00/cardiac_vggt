"""Minimal p95/p99 cardiac DVF for 4wok (resp OFF) — settles whether the small MEAN in-plane DVF hides
localized myocardial motion. Also reports the same for reference + bspline for the head comparison.
Run: micromamba run -n svr python tools/exp_4wok_p95.py --seqs 0-19"""
import argparse, json, os, sys
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from exp_4wok_analysis import build_dataset_ref, build_model, build_batch, fwd, THROUGH_MM, INPLANE_MM, ANAT
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
D = "cuda"
CKPTS = {
  "4wok_diffusion": ("scratch/logs/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "dpt"),
  "reference_L1TV": ("scratch/logs/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "dpt"),
  "bspline": ("scratch/logs/217719798_mri_volume_bspline_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "bspline"),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seqs", default="0-19"); a = ap.parse_args()
    lo, hi = a.seqs.split("-"); seqs = list(range(int(lo), int(hi) + 1))
    ds = build_dataset_ref(); off = RespiratoryConfig(enable=False)
    res = {}
    for name, (ck, head) in CKPTS.items():
        m = build_model(D, ck, head)
        ip95, ip99, dz95, dz99, ipmax = [], [], [], [], []
        for t in [0, 6]:                                  # ED, ES
            ds.t_target_fixed = t
            for seq in seqs:
                data = ds.get_data(seq_index=seq, img_per_seq=12); b = build_batch(data, D, seq)
                b = gpu_augment_batch(b, None, D, respiratory_cfg=off, train=False)
                _, dvf = fwd(m, b); imgs = b["images"][0].float().mean(1)
                for s in range(1, dvf.shape[0]):
                    msk = imgs[s] > ANAT
                    if not msk.any(): continue
                    ipmm = torch.sqrt((dvf[s, :, :, 0][msk] * INPLANE_MM) ** 2 + (dvf[s, :, :, 1][msk] * INPLANE_MM) ** 2)
                    dzmm = (dvf[s, :, :, 2][msk] * THROUGH_MM).abs()
                    ip95.append(float(ipmm.quantile(0.95))); ip99.append(float(ipmm.quantile(0.99))); ipmax.append(float(ipmm.max()))
                    dz95.append(float(dzmm.quantile(0.95))); dz99.append(float(dzmm.quantile(0.99)))
        res[name] = {"inplane_p95_mm": round(float(np.mean(ip95)), 2), "inplane_p99_mm": round(float(np.mean(ip99)), 2),
                     "inplane_max_mm": round(float(np.mean(ipmax)), 2),
                     "through_p95_mm": round(float(np.mean(dz95)), 2), "through_p99_mm": round(float(np.mean(dz99)), 2)}
        print(f"{name:16s} in-plane p95={res[name]['inplane_p95_mm']} p99={res[name]['inplane_p99_mm']} max={res[name]['inplane_max_mm']} | through p95={res[name]['through_p95_mm']} p99={res[name]['through_p99_mm']} mm", flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(REPO, "result", "analysis_4wok", "p95_dvf.json"), "w"), indent=2)
    print("wrote result/analysis_4wok/p95_dvf.json")


if __name__ == "__main__":
    main()
