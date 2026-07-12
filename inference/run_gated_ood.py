#!/usr/bin/env python
"""OOD gated (ECG breath-hold) breathing-sim validation — the OOD twin of inference/run_cmrxrecon.py.

The insight: OOD gated stacks (MIITT gated, OCMR gated) are the SAME KIND of data as CMRxRecon —
ECG-gated, breath-held, multi-phase SAX cine. So the clean gated cine serves as ground truth, our
custom respiratory simulation (docs/01, docs/05) corrupts the INPUT slices, and we measure how
well the model reconstructs the unshifted target — the IDENTICAL protocol as in-distribution val,
transferred to OOD.

Mirrors run_cmrxrecon.py exactly: the ONLY difference is the source of the canonical phase bundle.
CMRxRecon reads it from the monai cache (`MRIDataset.get_data`); here each subject's gated NIfTI is
resampled into the same canonical `(T, D=12, 256, 256)` cube + geometric bbox via
`BaseRTFBAdapter.build_canonical_bundle`. Everything downstream — deployment-realistic multi-frame
batch, reference-slot sweep, clean-vs-breathing corruption, `compute_volume_intensity_loss` metrics
(full/bbox/motion PSNR + SSIM), and the GIF/ED panels — is reused verbatim from run_cmrxrecon.

Because we APPLY the sim (unlike real prospective RT data), `resp_disp_mm` is known, so the
breathing metrics (resp_epe/corr/slope) are valid here with ground-truth displacement.

The model reads the target phase from slot-0 IMAGE CONTENT (z-only, target_t inert), so it is
agnostic to T — MIITT gated (T=30) and OCMR gated (T~21) sweep fine despite T != 12.

Usage:
  micromamba run -n svr env PYTHONPATH=training:. python inference/run_gated_ood.py --dataset miitt_gated
  micromamba run -n svr env PYTHONPATH=training:. python inference/run_gated_ood.py --dataset ocmr_gated --subjects exam_fs_0060/sax__fs_0060_1_5T
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from inference.adapters import MIITTGatedAdapter, OCMRAdapter
from inference.inference import load_rtfb_model_reference
from inference.adapters.base import DEFAULT_CKPT_REFERENCE, GRID_SHAPE
from inference.run_cmrxrecon import (
    reconstruct_from_bundle, save_multislice_gif, save_ed_montage, save_ed_input_png,
    save_ed_dvf_png, save_nnunet_nii, _nanmean, ED_PHASE,
)

# Per-dataset: gated recon root, subject discovery, adapter factory (subject -> adapter).
DATASETS = {
    "miitt_gated": dict(
        root="scratch/data/MIITT/nifti",
        discover=lambda root: sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
            if os.path.exists(os.path.join(d, "gated", "sax", "4d_recon.nii.gz"))),
        adapter=lambda root, s: MIITTGatedAdapter(os.path.join(root, s, "gated", "sax", "4d_recon.nii.gz")),
    ),
    "ocmr_gated": dict(   # recon/gated/<exam_id>/<subject>/sax_cine.nii.gz (+ meta.json)
        root="scratch/data/ocmr/recon/gated",
        discover=lambda root: sorted(
            os.path.relpath(os.path.dirname(f), root)
            for f in glob.glob(os.path.join(root, "*", "*", "sax_cine.nii.gz"))
            if not os.path.relpath(f, root).startswith("_")),
        adapter=lambda root, s: OCMRAdapter(os.path.join(root, s)),
    ),
}


def load_rcfg():
    """Build the `mri_volume.yaml` RespiratoryConfig standalone (no dataset/distributed needed) —
    the SAME breathing recipe training + in-distribution val use (docs/05)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(_ROOT, "training", "config")):
        cfg = compose(config_name="mri_volume")
    from data.respiratory import RespiratoryConfig
    return RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_REFERENCE)
    ap.add_argument("--frames-per-slice", type=int, default=5,
                    help="first N phases per non-reference in-bbox plane (short-burst sim)")
    ap.add_argument("--refiner", action="store_true", help="model has a coverage refiner head")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all discovered")
    ap.add_argument("--root", default=None, help="override the gated recon root")
    ap.add_argument("--out", default=None, help="default: result/<dataset>_eval")
    ap.add_argument("--dump-volumes", default=None,
                    help="dir to write per-phase pred/GT NIfTIs (EF/Dice seg stage)")
    ap.add_argument("--metrics-json", default=None,
                    help="path to write the per-subject/per-mode PSNR+SSIM summary")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    root = args.root or spec["root"]
    out = args.out or f"result/{args.dataset}_eval"
    os.makedirs(out, exist_ok=True)
    if args.dump_volumes:
        os.makedirs(args.dump_volumes, exist_ok=True)

    device = torch.device("cuda")
    model = load_rtfb_model_reference(args.ckpt, refiner=args.refiner, device=device)
    rcfg = load_rcfg()

    subjects = args.subjects or spec["discover"](root)
    if not subjects:
        print(f"  no {args.dataset} subjects found under {root}", flush=True)
        return

    summary = []
    for seq_index, name in enumerate(subjects):
        adapter = spec["adapter"](root, name)
        bundle_np, bbox = adapter.build_canonical_bundle()          # (T,12,256,256), (6,)
        phases_bundle = torch.from_numpy(bundle_np).to(device)
        safe = name.replace("/", "__")
        for breathing, tag in [(False, "clean"), (True, "breathing")]:
            res = reconstruct_from_bundle(
                model, phases_bundle, bbox, rcfg, seq_index, breathing, device,
                GRID_SHAPE, frames_per_slice=args.frames_per_slice)
            base = os.path.join(out, f"{safe}_{tag}")
            save_multislice_gif(res["pred_vols"], res["gt_vols"], res["bbox"], base + "_cycle.gif")
            save_ed_montage(res["pred_vols"][ED_PHASE], res["gt_vols"][ED_PHASE], base + "_ED.png")
            save_ed_input_png(res["ed_pack"], res["bbox"], res["z_mid"], base + "_ED_input.png")
            save_ed_dvf_png(res["ed_pack"], res["bbox"], res["z_mid"], base + "_ED_dvf.png", breathing)
            m = res["metrics"]
            means = {k: _nanmean(v) for k, v in m.items()}
            summary.append(dict(subject=name, seq_index=int(seq_index), mode=tag, per_phase=m, mean=means))
            print(f"[{name} {tag}] motion={means['motion']:.2f}  bbox={means['bbox']:.2f}  "
                  f"full={means['full']:.2f}dB  ssim={means['ssim']:.3f}  "
                  f"(motion/phase={['%.1f' % p for p in m['motion']]}) -> {base}_cycle.gif", flush=True)

            if args.dump_volumes:
                for t in range(res["pred_vols"].shape[0]):
                    save_nnunet_nii(res["pred_vols"][t], os.path.join(
                        args.dump_volumes, f"{safe}_{tag}_pred_t{t:02d}_0000.nii.gz"))
                if not breathing:
                    for t in range(res["gt_vols"].shape[0]):
                        save_nnunet_nii(res["gt_vols"][t], os.path.join(
                            args.dump_volumes, f"{safe}_gt_t{t:02d}_0000.nii.gz"))
        del phases_bundle
        torch.cuda.empty_cache()

    mpath = args.metrics_json or os.path.join(out, "metrics.json")
    with open(mpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote metrics -> {mpath}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
