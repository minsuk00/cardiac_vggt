#!/usr/bin/env python
"""Run a trained frame selector on rhythm-arm bundles -> selection tables for
`run_vggt.py --input sel` (docs/128), plus the offline phase error vs the manifests' true positions.

For each subject: every (plane, frame) of breath/stack_t* -> 518 model image (run_vggt's own
extraction) -> frozen patch tokens, computed ONCE. For queried frame f the reference is the
reference plane's frame f (slot 0 in run_vggt); every other plane's frames are scored against it
and the argmax is that plane's companion. sel[f][ref_plane] = f.

    PYTHONPATH=training:. python tools/framesel_predict.py --run scratch/framesel/runs/B \
        --cohorts cmrx2024_af12 ... --out scratch/framesel/sel_tables/B
"""
import argparse
import glob
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
from framesel_data import circ_dist                                               # noqa: E402
from framesel_model import FrameSelector, FrozenPatchEmbed                        # noqa: E402
from framesel_train import CKPT                                                   # noqa: E402
from data.gpu_aug import _resize_to_model_res, extract_slices_from_phases         # noqa: E402
from data.preprocess import Z_HALF_MM                                             # noqa: E402
from inference.load_run import load_model_from_run                                # noqa: E402


@torch.no_grad()
def embed_all(emb, stack, dev):
    """stack (F, D, H, W) tensor/array -> (D, F, N, C) frozen patch tokens of every (plane, frame),
    built from the same 518 model images run_vggt's `_extract` builds (gpu_augment_batch, aug off:
    native 256 slice, clamped to [0,1], then `_resize_to_model_res`)."""
    stack = torch.as_tensor(stack, dtype=torch.float32, device=dev)
    Fn, D = stack.shape[:2]
    planes = np.repeat(np.arange(D), Fn)
    frames = np.tile(np.arange(Fn), D)
    native = extract_slices_from_phases(stack[None], torch.as_tensor(frames, device=dev)[None],
                                        torch.as_tensor(planes, device=dev)[None],
                                        out_size=stack.shape[-1])[..., 0] / 255.0
    img = _resize_to_model_res(native, 518)[0]                                  # (D*F, 3, R, R)
    return emb(img).view(D, Fn, -1, 1024)


@torch.no_grad()
def select_from_tokens(sel, tok, ref, dz):
    """tok (D, F, N, C) -> sel (F, D) int: per queried frame f, each plane's argmax frame against
    the reference plane's frame f; the reference plane keeps frame f."""
    D, Fn = tok.shape[:2]
    znorm = torch.as_tensor((np.arange(D) - (D - 1) / 2.0) * dz / Z_HALF_MM, dtype=torch.float32,
                            device=tok.device)
    others = [z for z in range(D) if z != ref]
    out = np.zeros((Fn, D), dtype=np.int64)
    cand, _ = sel.prep(tok[others], tok[ref, 0], tok[ref])
    c = sel._tokens(cand.flatten(0, 1))                  # project candidates once, reuse for every f
    for f in range(Fn):
        _, r = sel.prep(tok[others][:, :1], tok[ref, f], tok[ref])
        logits = sel.score(c, len(others), Fn, r, znorm[others], float(znorm[ref]))  # (D-1, F)
        out[f, others] = logits.argmax(-1).cpu().numpy()
        out[f, ref] = f
    return out


def select(sel, emb, stack, ref, dz, dev):
    """stack (F, D, H, W) -> sel (F, D) int, tokens_ms, head_ms."""
    torch.cuda.synchronize(); t0 = time.perf_counter()
    tok = embed_all(emb, stack, dev)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    out = select_from_tokens(sel, tok, ref, dz)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    return out, (t1 - t0) * 1e3, (t2 - t1) * 1e3


def load_selector(run, dev, ckpt="best.pt"):
    ck = torch.load(os.path.join(run, ckpt), map_location=dev)
    a = ck["args"]
    sel = FrameSelector(a["variant"], center=a.get("center", False), norm=a.get("norm", "ln"),
                        attn_pool=a.get("attn_pool", False)).to(dev).eval()
    sel.load_state_dict(ck["model"])
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="selector run dir")
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint file inside --run (e.g. step005000.pt)")
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-root", default="scratch/eval")
    args = ap.parse_args()
    dev = torch.device("cuda")
    vggt, _ = load_model_from_run(CKPT, device=dev)
    emb = FrozenPatchEmbed(vggt).to(dev)
    del vggt
    torch.cuda.empty_cache()
    sel = load_selector(args.run, dev, args.ckpt)
    summary = {}
    for c in args.cohorts:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
        errs, orc, tms = [], [], []
        for mp in sorted(glob.glob(os.path.join(args.eval_root, c, "out", "*", "manifest.json"))):
            d = os.path.dirname(mp)
            man = json.load(open(mp))
            Fn, ref = int(man["T"]), int(man["rhythm"]["ref_plane"])
            assert ref == int(man["scatter"]["ref_plane"]), "rhythm vs run_vggt slot-0 plane differ"
            stack = np.stack([np.asarray(nib.load(os.path.join(d, "breath", f"stack_t{f:02d}.nii.gz")).dataobj,
                                         np.float32).transpose(2, 1, 0) for f in range(Fn)])
            s, tok_ms, head_ms = select(sel, emb, stack, ref, float(man["dz_mm"]), dev)
            pos = np.asarray(man["rhythm"]["pos_per_plane"])                         # (D, F)
            tgt = np.asarray(man["rhythm"]["ref_pos"])                               # (F,)
            e = circ_dist(pos[np.arange(pos.shape[0])[None, :], s], tgt[:, None])    # (F, D)
            o = circ_dist(pos[None, :, :], tgt[:, None, None]).min(-1)
            keep = np.arange(pos.shape[0]) != ref
            errs.append(e[:, keep].ravel()); orc.append(o[:, keep].ravel())
            tms.append((tok_ms, head_ms))
            json.dump({"sel": s.tolist(), "err": e.round(4).tolist(),
                       "selector": os.path.join(args.run, args.ckpt),
                       "tokens_ms": tok_ms, "head_ms": head_ms},
                      open(os.path.join(args.out, c, f"{man['subject']}.json"), "w"))
        e, o = np.concatenate(errs), np.concatenate(orc)
        summary[c] = {"n": len(tms), "err": float(e.mean()), "le1": float((e <= 1).mean()),
                      "oracle": float(o.mean()), "tokens_ms": float(np.mean([t[0] for t in tms])),
                      "head_ms": float(np.mean([t[1] for t in tms]))}
        print(c, json.dumps(summary[c]), flush=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
