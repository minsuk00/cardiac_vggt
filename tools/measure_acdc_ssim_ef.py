"""ACDC EF check for the SSIM refiner (frozen-backbone, target_t) model 217891050.

For each ACDC patient: build ONE scattered input batch (legacy contract — one random frame per
in-FOV z plane, NO reference slot, since this is a target_t model), sweep the query target_t=0..11,
take the refiner's V_refined at each phase, save as nnU-Net Task114 inputs. A later seg + analyze
turns these into LV-volume curves -> predicted EF, plotted against the already-computed GT EF
(scratch/analysis/phase_analysis/acdc_analysis.json, same nnU-Net method).

Runtime-only num_freqs=6 override (matches this legacy ckpt); aggregator.py is NOT modified.

Run: micromamba run -n svr python tools/measure_acdc_ssim_ef.py --out_dir scratch/analysis/phase_analysis_acdc_ssim/pred_vols
"""
import os, sys, glob, argparse
import numpy as np
import torch
import nibabel as nib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "training"))

from inference.adapters.base import percentile_scale, assign_canonical_z, _build_batch_core, GRID_SHAPE
from tools.render_reference_ed_targeted import ACDCAdapter, ACDC_ROOT
from vggt.models.vggt import VGGT
from vggt.models.aggregator import ZIndexEmbedder, TIndexEmbedder

DEV = torch.device("cuda")
CKPT = os.path.join(_ROOT, "scratch/logs/217891050_mri_refiner_frozen_ssim_newseed/ckpts/checkpoint_last.pt")
T = 12
CANON_SPACING = (1.4, 1.4, 12.0)


def build_model():
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
             train_on_residual_dvf=True, enable_refiner=True, refiner_use_coverage=True,
             grid_shape=GRID_SHAPE).to(DEV).eval()
    m.aggregator.z_embedder = ZIndexEmbedder(embed_dim=1024, num_freqs=6).to(DEV)
    m.aggregator.target_t_embedder = TIndexEmbedder(embed_dim=1024, num_freqs=6).to(DEV)
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp, f"missing={miss[:5]} unexpected={unexp[:5]}"
    print("loaded 217891050 (refiner, target_t, num_freqs=6): clean", flush=True)
    return m


def save_nnunet(vol_dhw, out_dir, tag):
    """vol (D=12,H,W) splat-order (Z,Y,X) -> nnU-Net (X,Y,Z)."""
    arr = np.transpose(np.asarray(vol_dhw, np.float32), (2, 1, 0))
    nib.save(nib.Nifti1Image(arr, np.diag([*CANON_SPACING, 1.0])),
             os.path.join(out_dir, f"{tag}_0000.nii.gz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    allp = sorted(glob.glob(os.path.join(ACDC_ROOT, "training", "patient*")) +
                  glob.glob(os.path.join(ACDC_ROOT, "testing", "patient*")))
    # Evenly span the full set (all pathology groups + train/test) instead of the first n.
    stride = max(1, len(allp) // args.n)
    patients = allp[::stride][:args.n]
    print(f"{len(patients)} ACDC patients (stride {stride} over {len(allp)})", flush=True)
    m = build_model()

    for i, p in enumerate(patients):
        pid = os.path.basename(p)
        try:
            ad = ACDCAdapter(p)
            cine = ad.load()
            scale = percentile_scale(cine)
            z_map = assign_canonical_z(ad.slice_positions_mm())
            batch, S, _ = _build_batch_core(cine, ad.inplane_mm(), scale, z_map,
                                            np.random.default_rng(0), DEV)
        except Exception as e:
            print(f"  skip {pid}: {e}", flush=True); continue
        for k in range(T):
            batch["target_t_indices"] = torch.full((1, S, 1), k / T * 2.0 - 1.0,
                                                    dtype=torch.float32, device=DEV)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = m(batch["images"], batch=batch)
            V = preds["V_refined"][0].float().cpu().numpy()
            save_nnunet(V, args.out_dir, f"{pid}_t{k:02d}")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(patients)}] {pid} done", flush=True)
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
