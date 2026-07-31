"""Headless CorSeg-CineSAX (MedNeXt-L) inference on 3D/4D SAX NIfTI volumes.

WHY this exists: the upstream release ships only a PyQt6 GUI whose loader reduces ANY volume to a
single 2D slice (`load_image` -> `np.take(..., shape[argmin]//2)`), and whose inference resizes the
whole image to 224x224 with `F.interpolate`, ignoring voxel spacing. That contradicts the paper's
own preprocessing (Methods 2.3), which resamples to 1.25 mm in-plane and THEN center-crops/pads to
224. This module does the paper-faithful thing and loops over every z-slice (and phase).

Preprocessing modes:
  paper (default) : resample in-plane to `--pixdim` (1.25 mm) -> center crop/zero-pad to 224 ->
                    per-slice z-score over non-zero voxels.   [matches training]
  gui             : bilinear-resize the whole slice to 224x224 -> z-score.  [what the shipped
                    GUI does; kept ONLY as an ablation to quantify the scale mismatch]

Labels out: 0=background, 1=LV myocardium, 2=LV cavity, 3=RV cavity  (CorSeg convention).
NOTE this differs from nnU-Net Task114 (1=LV cavity, 2=myocardium, 3=RV) -- labels 1/2 are SWAPPED.

Usage:
  micromamba run -n svr python tools/corseg/corseg_infer.py \
      --input <file.nii.gz | dir> --out <dir> [--mode paper] [--postproc] [--device cuda]
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corseg_postproc as cpp  # noqa: E402

CKPT_DEFAULT = "/home/minsukc/vggt/scratch/data/corseg/ModelWeight-CorSeg-CineSAX_MedNextL.pth"
LABEL_NAMES = {0: "background", 1: "LV_myo", 2: "LV_cav", 3: "RV_cav"}


# ─────────────────────────── model ───────────────────────────
def load_corseg(ckpt_path=CKPT_DEFAULT, device="cuda"):
    from monai.networks.nets.mednext import create_mednext
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = create_mednext(
        variant=cfg.get("mednext_variant", "L"),
        spatial_dims=cfg.get("spatial_dims", 2),
        in_channels=cfg.get("in_channels", 1),
        out_channels=cfg.get("num_classes", 4),
        kernel_size=cfg.get("mednext_kernel", 5),
        deep_supervision=False,
    )
    model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    return model.to(device).eval(), cfg


# ─────────────────── center crop / pad (invertible) ───────────────────
def _pad_crop_plan(cur, target):
    """1-D plan matching MONAI symmetric convention (floor of the deficit goes *before*).

    Returns (pad_before, pad_after, crop_start) with exactly one of pad/crop active.
    """
    if cur < target:
        d = target - cur
        return d // 2, d - d // 2, 0
    return 0, 0, (cur - target) // 2


def center_pad_crop(x, target_hw):
    """x: (N,C,H,W) -> (N,C,*target_hw). Returns (out, plan) where plan inverts it."""
    H, W = x.shape[-2:]
    ph0, ph1, cs_h = _pad_crop_plan(H, target_hw[0])
    pw0, pw1, cs_w = _pad_crop_plan(W, target_hw[1])
    out = x[..., cs_h:cs_h + min(H, target_hw[0]), cs_w:cs_w + min(W, target_hw[1])]
    # F.pad takes (left,right,top,bottom) i.e. last dim first
    out = F.pad(out, (pw0, pw1, ph0, ph1), mode="constant", value=0.0)
    return out, dict(src_hw=(H, W), ph=(ph0, ph1), pw=(pw0, pw1), cs=(cs_h, cs_w))


def invert_pad_crop(y, plan):
    """Inverse of center_pad_crop for a label map y: (N,C,Ht,Wt) -> (N,C,*src_hw)."""
    H, W = plan["src_hw"]
    ph0, ph1 = plan["ph"]
    pw0, pw1 = plan["pw"]
    cs_h, cs_w = plan["cs"]
    # undo padding
    y = y[..., ph0:y.shape[-2] - ph1 if ph1 else y.shape[-2],
            pw0:y.shape[-1] - pw1 if pw1 else y.shape[-1]]
    # undo cropping (re-insert into a zero canvas at the recorded offset)
    out = y.new_zeros(y.shape[:-2] + (H, W))
    out[..., cs_h:cs_h + y.shape[-2], cs_w:cs_w + y.shape[-1]] = y
    return out


# ─────────────────────────── core ───────────────────────────
@torch.no_grad()
def segment_stack(model, stack_zhw, spacing_xy, mode="paper", pixdim=1.25,
                  img_size=(224, 224), device="cuda", batch_size=16, amp=True):
    """stack_zhw: (Z,H,W) float32 (H,W = in-plane). Returns (Z,H,W) uint8 labels.

    spacing_xy = (in-plane spacing along H, along W) in mm.
    """
    x = torch.from_numpy(np.ascontiguousarray(stack_zhw)).float().unsqueeze(1)  # (Z,1,H,W)
    H, W = x.shape[-2:]

    if mode == "paper":
        # 1) resample to isotropic `pixdim` in-plane, preserving physical scale
        nh = max(1, int(round(H * spacing_xy[0] / pixdim)))
        nw = max(1, int(round(W * spacing_xy[1] / pixdim)))
        xr = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        # 2) center crop / zero-pad to the network's fixed 224x224
        xin, plan = center_pad_crop(xr, img_size)
    elif mode == "gui":
        xin = F.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        plan = None
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # 3) per-slice z-score over non-zero voxels only (background left at 0)
    flat = xin.reshape(xin.shape[0], -1)
    out_slices = []
    for i in range(xin.shape[0]):
        s = xin[i:i + 1]
        nz = s != 0
        if nz.any():
            mu, sd = s[nz].mean(), s[nz].std()
            if sd > 1e-8:
                s = (s - mu) / sd
            s = torch.where(nz, s, torch.zeros_like(s))
        out_slices.append(s)
    xin = torch.cat(out_slices, 0)
    del flat

    # 4) forward in batches
    logits = []
    for i in range(0, xin.shape[0], batch_size):
        chunk = xin[i:i + batch_size].to(device)
        if amp and str(device).startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(chunk)
        else:
            lg = model(chunk)
        logits.append(lg.float().argmax(1, keepdim=True).cpu())
    pred = torch.cat(logits, 0)  # (Z,1,224,224) int

    # 5) invert geometry back onto the original grid (nearest for labels)
    if mode == "paper":
        pred = invert_pad_crop(pred, plan)                          # -> (Z,1,nh,nw)
    pred = F.interpolate(pred.float(), size=(H, W), mode="nearest")  # -> (Z,1,H,W)
    return pred.squeeze(1).round().to(torch.uint8).numpy()


def apply_pp(labels_zhw, steps=("step1", "step2", "step3")):
    """Slice-wise anatomical post-processing (the paper's 3 steps, verbatim code)."""
    cfg = {s: (s in steps) for s in ("step1", "step2", "step3")}
    out = np.empty_like(labels_zhw)
    viol_before, viol_after = [], []
    for z in range(labels_zhw.shape[0]):
        sl = labels_zhw[z]
        viol_before.append(cpp.detect_violations(sl))
        res, _ = cpp.apply_postprocessing(sl, cfg)
        out[z] = res
        viol_after.append(cpp.detect_violations(res))
    agg = lambda vs, k: int(sum(bool(v[k]) for v in vs))  # noqa: E731
    stats = {
        "n_slices": int(labels_zhw.shape[0]),
        "before": {k: agg(viol_before, k) for k in ("has_fragment", "has_containment_violation", "has_gap")},
        "after": {k: agg(viol_after, k) for k in ("has_fragment", "has_containment_violation", "has_gap")},
    }
    return out, stats


def segment_nifti(model, path, mode="paper", postproc=False, device="cuda", pixdim=1.25,
                  batch_size=16):
    """Segment a 3D (X,Y,Z) or 4D (X,Y,Z,T) NIfTI. Returns (labels, affine, header, stats).

    The input HEADER is returned and propagated to the output, not just the affine. Some cohorts
    (ACDC) ship a unit affine `diag(-1,-1,1)` with the true spacing only in `pixdim`, so writing
    `Nifti1Image(lab, im.affine)` alone silently stamps 1.0 mm on the label map. That leaves Dice
    intact (voxel counts on a shared grid) but corrupts any physical-volume metric downstream
    (mL, LV mass). Passing the header through makes the output's geometry identical to the input's.
    """
    im = nib.load(path)
    data = np.asarray(im.dataobj, dtype=np.float32)
    zooms = [float(z) for z in im.header.get_zooms()]
    spacing_xy = (zooms[0], zooms[1])  # (X,Y) are the in-plane axes of a SAX stack
    squeezed = False
    if data.ndim == 3:
        data = data[..., None]
        squeezed = True
    if data.ndim != 4:
        raise ValueError(f"{path}: expected 3D or 4D, got {data.shape}")

    X, Y, Z, T = data.shape
    out = np.zeros((X, Y, Z, T), np.uint8)
    all_stats = []
    for t in range(T):
        # (X,Y,Z) -> (Z,H=X,W=Y): treat X,Y as the in-plane axes, z as the batch
        stack = np.transpose(data[..., t], (2, 0, 1))
        lab = segment_stack(model, stack, spacing_xy, mode=mode, pixdim=pixdim,
                            device=device, batch_size=batch_size)
        if postproc:
            lab, st = apply_pp(lab)
            all_stats.append(st)
        out[..., t] = np.transpose(lab, (1, 2, 0))
    if squeezed:
        out = out[..., 0]
    return out, im.affine, im.header, all_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="NIfTI file or directory of NIfTIs")
    ap.add_argument("--out", required=True, help="output directory for label maps")
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--mode", default="paper", choices=("paper", "gui"))
    ap.add_argument("--pixdim", type=float, default=1.25)
    ap.add_argument("--postproc", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--glob", default="*.nii.gz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--strip_suffix", default="_0000",
                    help="strip this from output basenames (nnU-Net input convention)")
    args = ap.parse_args()

    files = ([args.input] if os.path.isfile(args.input)
             else sorted(glob.glob(os.path.join(args.input, args.glob))))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no inputs matched {args.input}/{args.glob}")

    os.makedirs(args.out, exist_ok=True)
    model, cfg = load_corseg(args.ckpt, args.device)
    print(f"[corseg] MedNeXt-{cfg.get('mednext_variant','L')} k{cfg.get('mednext_kernel',5)} "
          f"| mode={args.mode} pixdim={args.pixdim} postproc={args.postproc} "
          f"| {len(files)} file(s) -> {args.out}", flush=True)

    stats_all = {}
    for i, f in enumerate(files):
        base = os.path.basename(f)
        base = base[:-7] if base.endswith(".nii.gz") else os.path.splitext(base)[0]
        if args.strip_suffix and base.endswith(args.strip_suffix):
            base = base[: -len(args.strip_suffix)]
        lab, affine, hdr, st = segment_nifti(model, f, mode=args.mode, postproc=args.postproc,
                                             device=args.device, pixdim=args.pixdim,
                                             batch_size=args.batch_size)
        nib.save(nib.Nifti1Image(lab, affine, header=hdr),
                 os.path.join(args.out, base + ".nii.gz"))
        if st:
            stats_all[base] = st
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] {base}  labels={np.unique(lab).tolist()}", flush=True)

    meta = {"mode": args.mode, "pixdim": args.pixdim, "postproc": args.postproc,
            "ckpt": args.ckpt, "n": len(files), "labels": LABEL_NAMES}
    if stats_all:
        meta["postproc_stats"] = stats_all
    json.dump(meta, open(os.path.join(args.out, "corseg_meta.json"), "w"), indent=2)
    print(f"[corseg] done -> {args.out}")


if __name__ == "__main__":
    main()
