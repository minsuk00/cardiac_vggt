"""Cross-task evaluation matrix for the 5 respiratory-variant runs.

Re-evaluates each run's checkpoint under TWO common protocols on the SAME 30 val
subjects, deterministically (mirrors the trainer val loop exactly):

  - Protocol C (clean)     : respiratory.enable = False  (reconstruct from clean slices)
  - Protocol B (breathing) : respiratory.enable = True   (the real corrupted->clean task,
                             deterministic per seq_index, the params the resp runs trained with)

For every (checkpoint, protocol) it sweeps seq_index = 0..N_VAL-1 (N_VAL=200, identical to
limit_val_batches), each mapping to subject = seq%30 and t_target = seq%12, and records
per-sample full / bbox / motion PSNR + full SSIM. It ALSO computes the identity-Δ baseline
(world_points = scanner_coords) once per protocol — which must reproduce each run's logged
baseline_identity.json, proving the harness is faithful to the trainer.

Why this is necessary: the runs' own training-time val metrics live on DIFFERENT tasks
(resp runs score breathing-corrupted inputs, no-resp runs score clean inputs), so their
wandb curves are NOT comparable across the resp/no-resp boundary. This harness puts all 5
on one protocol so the comparison is apples-to-apples.

Outputs (result/variants_eval/):
  identity_{clean,breathing}.json      per-sample identity records + means (+ logged target)
  var{N}_{clean,breathing}.json        per-sample model records
  _validation.json                     harness-vs-logged-baseline check

Run:
  micromamba run -n svr python tools/eval_variants_matrix.py            # full matrix
  micromamba run -n svr python tools/eval_variants_matrix.py --n-val 12 # quick smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, REPO)

from data.datasets.mri_dataset import MRIDataset           # noqa: E402
from data.respiratory import RespiratoryConfig             # noqa: E402
from data.gpu_aug import gpu_augment_batch                 # noqa: E402
from vggt.models.vggt import VGGT                          # noqa: E402
from loss import compute_volume_intensity_loss             # noqa: E402

LOGS = os.path.join(REPO, "scratch", "logs")
DATA_ROOT = os.path.join(REPO, "scratch", "data", "CMRxRecon2024", "Cine_combined")
SPLIT_FILE = os.path.join(REPO, "training", "splits", "random_8_1_1.txt")
OUT_DIR = os.path.join(REPO, "result", "variants_eval")
GRID_SHAPE = (12, 256, 256)
NUM_SLICES = 12
N_VAL_DEFAULT = 200

# The 5 runs. use_t = use_t_pose_embedding (only var1 trained it on); use_z / use_target_t
# are True for all. `family`/`epoch` are tagged from the artifact inventory.
RUNS = [
    dict(var=1, name="resp_zt",          job=51695105, family="resp",   epoch=59, use_t=True,
         exp_dir="218747856_mri_volume_resp_allphases_aggft_zt"),
    dict(var=2, name="resp_no_t",        job=51695106, family="resp",   epoch=59, use_t=False,
         exp_dir="218747856_mri_volume_resp_allphases_aggft_z_no_t"),
    dict(var=3, name="resp_aug_no_t",    job=51695107, family="resp",   epoch=60, use_t=False,
         exp_dir="218747856_mri_volume_resp_aug_allphases_aggft_z_no_t"),
    dict(var=4, name="noresp_no_t",      job=51754121, family="noresp", epoch=None, use_t=False,
         exp_dir="218643188_mri_volume_noresp_allphases_aggft_z_no_t"),
    dict(var=5, name="noresp_aug_no_t",  job=51754122, family="noresp", epoch=None, use_t=False,
         exp_dir="218643188_mri_volume_noresp_aug_allphases_aggft_z_no_t"),
]

# Respiratory params for Protocol B — defaults match mri_volume.yaml (what the resp runs used).
RESP_PARAMS = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3,
                   ap_ratio=0.35, ap_axis="H", per_slot=True, direction_jitter_deg=30.0)

PROTOCOLS = {
    "clean":     RespiratoryConfig(enable=False),
    "breathing": RespiratoryConfig(enable=True, **RESP_PARAMS),
}


def build_dataset():
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    return MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                      mode="dynamic", mri_mode="axial", num_slices=NUM_SLICES, target_size=518)


def build_batch(data, device, seq_index):
    """Mirror trainer._compute_identity_baseline + the embedder keys the model forward needs."""
    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(device)

    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0  # (1,S,3,H,W) in [0,1]
    batch = {
        "images": imgs,
        "scanner_coords": st("scanner_coords"),         # (1,S,H,W,3)
        "z_indices": st("z_indices"),                    # (1,S,1)
        "t_indices": st("t_indices"),
        "target_t_indices": st("target_t_indices"),
        "timesteps": st("timesteps", np.int64),          # (1,S)
        "slice_indices": st("slice_indices", np.int64),
        "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0).to(device),
        "anatomy_bbox": torch.from_numpy(np.asarray(data["anatomy_bbox"]).astype(np.int64)).unsqueeze(0).to(device),
        "phases": torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).unsqueeze(0).to(device),
        "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=device),
    }
    return batch


def _f(x):
    return float(x.item()) if torch.is_tensor(x) else (float(x) if x is not None else None)


def eval_protocol(ds, resp_cfg, device, n_val, model=None):
    """Sweep seq_index 0..n_val-1; return per-sample records. model=None → identity-Δ."""
    records = []
    n_subj = len(ds.subjects)
    for i in range(n_val):
        data = ds.get_data(seq_index=i, img_per_seq=NUM_SLICES)
        batch = build_batch(data, device, seq_index=i)
        # Deterministic respiratory (val-mode, seeded per seq_index); no-op when disabled.
        batch = gpu_augment_batch(batch, None, device, respiratory_cfg=resp_cfg, train=False)

        if model is None:
            preds = {"world_points": batch["scanner_coords"]}
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = model(batch["images"], batch=batch)

        out = compute_volume_intensity_loss(preds, batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
        disp = batch.get("resp_disp_mm")
        records.append(dict(
            seq=i,
            subject=os.path.basename(ds.subjects[i % n_subj]),
            t_target=int(np.asarray(data["t_target"]).flatten()[0]),
            S=int(batch["images"].shape[1]),
            psnr_full=_f(out.get("metric_psnr_3d_full")),
            psnr_bbox=_f(out.get("metric_psnr_3d_bbox")),
            psnr_motion=_f(out.get("metric_psnr_3d_motion")),
            ssim_full=_f(out.get("metric_ssim_3d_full")),
            mae_full=_f(out.get("metric_mae_3d_full")),
            motion_frac=_f(out.get("metric_motion_frac")),
            resp_disp_mm_mean=(float(disp.float().norm(dim=-1).mean()) if disp is not None else None),
        ))
    return records


def summarize(records):
    def stat(key):
        vals = [r[key] for r in records if r[key] is not None and np.isfinite(r[key])]
        if not vals:
            return dict(mean=None, std=None, n=0)
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals)), n=len(vals))
    out = {k: stat(k) for k in ["psnr_full", "psnr_bbox", "psnr_motion", "ssim_full", "mae_full"]}
    # per-phase mean for the three PSNRs
    out["per_phase"] = {}
    for key in ["psnr_full", "psnr_bbox", "psnr_motion"]:
        pp = {}
        for t in range(12):
            vals = [r[key] for r in records if r["t_target"] == t and r[key] is not None and np.isfinite(r[key])]
            if vals:
                pp[t] = dict(mean=float(np.mean(vals)), n=len(vals))
        out["per_phase"][key] = pp
    return out


def make_model(use_t, ckpt_path, device):
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=use_t, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
    ).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    res = model.load_state_dict(sd, strict=False)
    model.eval()
    epoch = ck.get("epoch", None)
    load_info = dict(
        missing=len(res.missing_keys), unexpected=len(res.unexpected_keys),
        missing_sample=list(res.missing_keys)[:8], unexpected_sample=list(res.unexpected_keys)[:8],
        has_t_embedder=any("t_embedder" in k and "target" not in k for k in sd.keys()),
        ckpt_epoch=epoch,
    )
    return model, load_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=N_VAL_DEFAULT)
    ap.add_argument("--ckpt", default="checkpoint_last.pt")
    ap.add_argument("--only-identity", action="store_true", help="just validate identity baselines")
    ap.add_argument("--vars", default="1,2,3,4,5", help="comma list of var numbers to eval")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    want = {int(v) for v in args.vars.split(",") if v.strip()}
    print(f"device={device}  n_val={args.n_val}  ckpt={args.ckpt}  vars={sorted(want)}")
    print("Building val dataset (monai cache builds lazily on first access)...")
    ds = build_dataset()
    print(f"  val subjects: {len(ds.subjects)}")

    # ── Identity baselines (no model) — one per protocol → the harness-validation linchpin ──
    identity = {}
    for pname, cfg in PROTOCOLS.items():
        t0 = time.time()
        recs = eval_protocol(ds, cfg, device, args.n_val, model=None)
        s = summarize(recs)
        identity[pname] = dict(summary=s, records=recs)
        with open(os.path.join(OUT_DIR, f"identity_{pname}.json"), "w") as f:
            json.dump(identity[pname], f, indent=2)
        print(f"[identity:{pname:9s}] full={s['psnr_full']['mean']:.3f} "
              f"bbox={s['psnr_bbox']['mean']:.3f} motion={s['psnr_motion']['mean']:.3f} "
              f"({time.time()-t0:.0f}s, n={s['psnr_full']['n']})")

    if args.only_identity:
        return

    # ── Model matrix: 5 checkpoints × 2 protocols ──
    for run in RUNS:
        if run["var"] not in want:
            continue
        ckpt_path = os.path.join(LOGS, run["exp_dir"], "ckpts", args.ckpt)
        if not os.path.exists(ckpt_path):
            print(f"!! var{run['var']} ckpt missing: {ckpt_path}")
            continue
        print(f"\n=== var{run['var']} ({run['name']}, use_t={run['use_t']}) ===")
        model, load_info = make_model(run["use_t"], ckpt_path, device)
        print(f"  load: missing={load_info['missing']} unexpected={load_info['unexpected']} "
              f"has_t_embedder={load_info['has_t_embedder']} ckpt_epoch={load_info['ckpt_epoch']}")
        for pname, cfg in PROTOCOLS.items():
            t0 = time.time()
            recs = eval_protocol(ds, cfg, device, args.n_val, model=model)
            s = summarize(recs)
            payload = dict(var=run["var"], name=run["name"], family=run["family"],
                           use_t=run["use_t"], protocol=pname, ckpt=args.ckpt,
                           load_info=load_info, summary=s, records=recs)
            with open(os.path.join(OUT_DIR, f"var{run['var']}_{pname}.json"), "w") as f:
                json.dump(payload, f, indent=2)
            print(f"  [{pname:9s}] full={s['psnr_full']['mean']:.3f} "
                  f"bbox={s['psnr_bbox']['mean']:.3f} motion={s['psnr_motion']['mean']:.3f} "
                  f"ssim={s['ssim_full']['mean']:.4f}  ({time.time()-t0:.0f}s)")
        del model
        torch.cuda.empty_cache()

    print("\nDone. JSON written to", OUT_DIR)


if __name__ == "__main__":
    main()
