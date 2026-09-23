"""Dangi et al. (2018) Stage A baseline over the frozen eval bundles (docs/100).

Per subject and phase k: read `<input>/stack_t{k}.nii.gz` (native 256x256xD, 1.4 mm), predict
the LV centre of every slice with the trained centre-regression CNN on Dangi's own 1.5625 mm /
192^2 grid, convert the per-slice shifts to mm, and translate each slice ON THE NATIVE GRID so
every predicted centre coincides with the anchor. Output mirrors the classical-baseline layout
(`<out_root>/<ds>/out/<subj>/<arm>/recon_<variant>/vol_t{k}.nii.gz` + stamp.json + timing.json)
so the standing scorer reads it unchanged once the arm is copied into the bundle.

Anchor (docs/100): `reference` (default) = the reference plane's predicted centre (the same
input VGGT conditions on; per-slice shifts identical to the paper's up to one whole-stack
constant); `image_center` = the paper's Sec. 2.4 literal (96,96); `mean` = the paper's Sec. 4
convention. The method's structural limit — rigid in-plane translation cannot represent an
oblique LV axis, phase mismatch, or through-plane motion — is untouched by any of these.

Generation is ONLY a rigid in-plane shift of each input slice (no splat, no z-resampling, no
intensity or phase change); through-plane and phase errors of the output equal the input's.
Checkpoint: evaluation/checkpoints/dangi_pool_v2/best.pt (trained by sbatch/train_dangi_baseline.sh
on pooled_curated_v2 [train], all 12 phases, nnU-Net LV centroids). Compute cost is I/O-bound, not
GPU-bound (~4 s/subject, A40 and CPU alike — docs/100 §10a); use --device cuda for the paper's
compute-cost timing column, CPU for a throwaway pass.

    python evaluation/src/engine/run_dangi.py --split val --input scatter      # arm dangi_scatter
    python evaluation/src/engine/run_dangi.py --split val --input gated        # arm dangi
    ... --out-root temp/dangi/eval    # dry pass outside the bundles
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import shift as nd_shift
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "baselines", "dangi-cmr-alignment"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
DEFAULT_CKPT = os.path.join(ROOT, "evaluation", "checkpoints", "dangi_pool_v2", "best.pt")
from dangi.align import anchor_point, load_model, predict_centers  # noqa: E402
from dangi.data import normalize, preprocess  # noqa: E402
import paths  # noqa: E402


def translate_native(stack_xyz, shifts_mm_xy, spacing_xy):
    """Shift each z-plane of an (X,Y,Z) array by (dx,dy) mm -> voxels; zero fill, linear."""
    out = np.empty_like(stack_xyz)
    for z in range(stack_xyz.shape[2]):
        dvox = np.asarray(shifts_mm_xy[z]) / np.asarray(spacing_xy)
        out[:, :, z] = nd_shift(stack_xyz[:, :, z], dvox, order=1, mode="constant", cval=0.0,
                                prefilter=False)
    return out


def run_subject(models, config, ds, subj, variant, input_name, anchor, out_root, batch_size):
    """models: one model copy per device; the T phases are split into contiguous blocks, one
    thread per device (phases are independent). One model -> the original sequential loop."""
    sd = paths.subject_dir(ds, subj)
    man = json.load(open(sd / "manifest.json"))
    T, ref_plane = int(man["T"]), int(man["scatter"]["ref_plane"])
    arm = f"dangi_{input_name}" if input_name != "gated" else "dangi"
    out_dir = os.path.join(out_root, ds, "out", subj, arm, f"recon_{variant}")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "stamp.json")) and not args.overwrite:
        return "cached"
    stack_dir = "scatter" if input_name == "scatter" else variant
    centres_log, t0 = {}, time.monotonic()

    def one_phase(k, model):
        img = nib.load(sd / stack_dir / f"stack_t{k:02d}.nii.gz")
        native = np.asarray(img.dataobj, dtype=np.float32)
        spacing = np.linalg.norm(img.affine[:3, :3], axis=0)
        slices, _ = preprocess(img, config)  # (Z,192,192) on Dangi's grid
        centres = predict_centers(model, normalize(slices), batch_size)  # (Z,2) x,y px @1.5625
        anchor_arg = {"reference": ref_plane, "image_center": None,
                      "mean": centres.mean(axis=0)}[anchor]
        shifts_mm = (anchor_point(centres, anchor_arg, config) - centres) * config.spacing_mm
        corrected = translate_native(native, shifts_mm, spacing[:2])
        nib.save(nib.Nifti1Image(corrected, img.affine), os.path.join(out_dir, f"vol_t{k:02d}.nii.gz"))
        return dict(centres_px=centres.tolist(), shifts_mm=shifts_mm.tolist())

    def block(r, ks):
        return [(k, one_phase(k, models[r])) for k in ks]

    blocks = [b.tolist() for b in np.array_split(np.arange(T), len(models))]
    with ThreadPoolExecutor(len(models)) as ex:
        for part in ex.map(block, range(len(models)), blocks):
            centres_log.update({f"t{k:02d}": v for k, v in part})
    total = time.monotonic() - t0
    json.dump(centres_log, open(os.path.join(out_dir, "centres.json"), "w"))
    json.dump(dict(engine="dangi_stage_a", input_stack=input_name, anchor=anchor,
                   checkpoint=os.path.abspath(args.checkpoint),
                   ckpt_fingerprint=paths.ckpt_fingerprint(args.checkpoint)),
              open(os.path.join(out_dir, "stamp.json"), "w"))
    tj = os.path.join(os.path.dirname(out_dir), "timing.json")
    timing = json.load(open(tj)) if os.path.exists(tj) else {}
    timing[variant] = dict(total_sec=total, per_phase_sec=total / T,
                           device=str(next(models[0].parameters()).device), gpus=len(models))
    json.dump(timing, open(tj, "w"), indent=2)
    return f"{total:.1f}s"


def main():
    global args
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--sources", nargs="+", default=list(paths.DATASETS))
    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--variant", default="breath", choices=paths.VARIANTS)
    ap.add_argument("--input", default="scatter", choices=["gated", "scatter"])
    ap.add_argument("--anchor", default="reference", choices=["reference", "image_center", "mean"])
    ap.add_argument("--out-root", default=os.path.join(ROOT, "scratch", "eval"),
                    help="bundle root (default) or a mirror tree such as temp/dangi/eval for a dry pass")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even subjects with an existing stamp.json (e.g. to re-time on a different device)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (at most 3): split each "
                    "subject's phases across them, one model copy per GPU; overrides --device")
    args = ap.parse_args()
    devices = [args.device]
    if args.gpus is not None:
        ids = [int(g) for g in args.gpus.split(",")]
        if len(set(ids)) != len(ids) or not 1 <= len(ids) <= 3:
            ap.error("--gpus needs 1-3 distinct ids (caesar: never all 4 GPUs)")
        devices = [f"cuda:{g}" for g in ids]
    models = []
    for d in devices:
        model, config = load_model(args.checkpoint, d)
        models.append(model)
    for ds in args.sources:
        subjects, _ = paths.filter_by_split(ds, paths.subjects(ds), args.split)
        if args.subjects:
            subjects = [s for s in subjects if s in set(args.subjects)]
        for subj in subjects:
            status = run_subject(models, config, ds, subj, args.variant, args.input, args.anchor,
                                 args.out_root, args.batch_size)
            print(f"{ds}/{subj}: {status}", flush=True)


if __name__ == "__main__":
    main()
