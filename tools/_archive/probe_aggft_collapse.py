#!/usr/bin/env python3
"""
tools/probe_aggft_collapse.py

Why did the pooled1337 aug/noaug runs (SLURM 55996915/16) die at epoch 16-17?

The on-disk logs establish the SYMPTOMS: `train/optim/grad_aggregator` goes to exactly 0
at one step and stays < 1e-6 for the next ~70 epochs, `train/loss/diffusion` goes to
exactly 0.0 at the SAME step, `train/metric/mean_disp_norm` collapses 0.05 -> 0.012, and
`grad_point` stays healthy (~0.5). This script proves the CAUSAL CHAIN on the actual
collapsed checkpoint:

  Q1  Is the predicted DVF spatially constant?          (-> forward, spatial std of dvf)
  Q2  Is the head output input-INDEPENDENT?             (-> two different batches, compare)
  Q3  Where does spatial variance die?                  (-> forward hooks, stage by stage)
  Q4  Is dL/d(aggregator tokens) exactly 0?             (-> retain_grad on aggregator output)
  Q5  Where do PARAM gradients die?                     (-> per-module grad norms)
  CTRL Same probe on base VGGT-1B, which must show nonzero aggregator grads. Without this
       control a "gradient is 0" reading is worthless -- it could just be a broken probe.

Usage:
  PYTHONPATH=training:. python tools/probe_aggft_collapse.py \
      --ckpt /tmp/ck_aug.pt --base scratch/base_weights/vggt1b_base.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("training"))

torch.set_float32_matmul_precision("high")

S, D, HW = 12, 12, 518
DIFFUSION_W, GATHER_W, TV_W = 1000.0, 0.5, 0.0


def build_batch(device, seed):
    """Synthetic but shape/normalisation-faithful batch. Random images are deliberate:
    if the head emits the same DVF for two different noise inputs, it is input-independent."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.rand(1, S, 3, HW, HW, generator=g).to(device)
    gt = torch.rand(1, D, 256, 256, generator=g).to(device)
    z_raw = torch.arange(S, dtype=torch.float32)
    z_indices = ((z_raw / (D - 1)) * 2.0 - 1.0).unsqueeze(0).to(device)
    t_indices = torch.randint(0, 12, (1, S), generator=g).to(device)
    px = torch.linspace(-1.0, 1.0, HW, device=device)
    gy, gx = torch.meshgrid(px, px, indexing="ij")
    sc = torch.zeros(1, S, HW, HW, 3, device=device)
    sc[0, :, :, :, 0] = gx
    sc[0, :, :, :, 1] = gy
    for s in range(S):
        sc[0, s, :, :, 2] = z_indices[0, s]
    return {
        "images": images, "gt_target_volume": gt,
        "z_indices": z_indices, "t_indices": t_indices, "scanner_coords": sc,
        "anatomy_bbox": torch.tensor([[0, D, 0, 256, 0, 256]], device=device, dtype=torch.int64),
        "z_scale": torch.tensor([90.0 / 12.0], device=device),
    }


def make_model(device, ckpt):
    from vggt.models.vggt import VGGT
    torch.manual_seed(0)
    m = VGGT(depth=24, embed_dim=1024, num_heads=16,
             use_z_pose_embedding=True, reference_slot=True,
             use_reference_token=True, train_on_residual_dvf=True).to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    return m.train()


def spatial_std(t):
    """Mean over slots of the per-slot spatial std over (H, W)."""
    f = t.float()
    return f.flatten(-3, -2).std(dim=-2).mean().item() if f.dim() >= 4 else f.std().item()


def run(model, tag, device):
    from loss import compute_volume_intensity_loss

    print("\n" + "=" * 100)
    print(f"###  {tag}")
    print("=" * 100)

    # ---- forward hooks: spatial std of activations through the DPT stack ----
    acts = {}

    def hook(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o) and o.dim() >= 3:
                acts[name] = (tuple(o.shape), o.float().std().item(),
                              o.float().abs().mean().item())
        return f

    handles = []
    ph = model.point_head
    for nm, mod in [("projects.3", ph.projects[3]),
                    ("scratch.layer4_rn", ph.scratch.layer4_rn),
                    ("refinenet4", ph.scratch.refinenet4),
                    ("refinenet1", ph.scratch.refinenet1),
                    ("output_conv1", ph.scratch.output_conv1),
                    ("output_conv2", ph.scratch.output_conv2)]:
        handles.append(mod.register_forward_hook(hook(nm)))

    batch = build_batch(device, seed=1)
    tokens_ref = {}

    # capture + retain grad on the aggregator's output tokens
    orig_agg_forward = model.aggregator.forward

    def agg_wrapper(*a, **kw):
        out = orig_agg_forward(*a, **kw)
        toks, psi = out
        for t in toks:
            t.retain_grad()
        tokens_ref["list"] = toks
        return out

    model.aggregator.forward = agg_wrapper

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    dvf = preds["dvfs"]
    dvf.retain_grad()

    print(f"\n--- Q1 forward: DVF statistics  (shape {tuple(dvf.shape)}) ---")
    print(f"  dvf  mean|.|            = {dvf.float().abs().mean().item():.6e}")
    print(f"  dvf  SPATIAL std (H,W)  = {spatial_std(dvf):.6e}   <-- 0 => spatially constant")
    print(f"  dvf  global std         = {dvf.float().std().item():.6e}")
    nb = (dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]).float()
    print(f"  |neighbour diff| mean   = {nb.abs().mean().item():.6e}")
    print(f"  neighbour diff EXACT 0  = {(nb == 0).float().mean().item():.4f} of pairs")
    print(f"  per-channel spatial std = {[round(dvf[...,c].float().flatten(1,2).std(dim=1).mean().item(),8) for c in range(3)]}")

    print(f"\n--- Q3 forward: activation std through the DPT head ---")
    for k in ["projects.3", "scratch.layer4_rn", "refinenet4", "refinenet1",
              "output_conv1", "output_conv2"]:
        if k in acts:
            sh, sd_, am = acts[k]
            print(f"  {k:22s} shape={str(sh):26s} std={sd_:.6e}  mean|.|={am:.6e}")

    toks = tokens_ref["list"]
    print(f"\n  aggregator output: {len(toks)} tensors, last shape {tuple(toks[-1].shape)}")
    for i in [0, len(toks) // 2, len(toks) - 1]:
        t = toks[i]
        # tokens are (B, S, N, C) -> spatial variance across the N patch tokens
        v = t.float().std(dim=-2).mean().item()
        print(f"    tokens[{i}] std-across-patches = {v:.6e}  global std = {t.float().std().item():.6e}")

    # ---- loss + backward ----
    out = compute_volume_intensity_loss(preds, batch, tv_weight=TV_W,
                                        diffusion_weight=DIFFUSION_W, gather_weight=GATHER_W)
    loss = out["loss_volume"] + out.get("loss_diffusion", 0.0) + out.get("loss_gather", 0.0)
    print(f"\n--- loss terms ---")
    for k in ["loss_volume", "loss_diffusion", "loss_gather"]:
        if k in out:
            print(f"  {k:16s} = {float(out[k]):.6e}")
    model.zero_grad(set_to_none=True)
    loss.backward()

    print(f"\n--- Q4 backward: gradient reaching the aggregator OUTPUT tokens ---")
    print(f"  dL/d(dvf)            norm = {dvf.grad.norm().item():.6e}")
    for i in [0, len(toks) // 2, len(toks) - 1]:
        g = toks[i].grad
        print(f"  dL/d(tokens[{i}])      norm = "
              f"{'None' if g is None else f'{g.norm().item():.6e}'}"
              f"{'' if g is None else f'   exact-zero frac = {(g == 0).float().mean().item():.4f}'}")

    print(f"\n--- Q5 backward: PARAMETER grad norms by module ---")
    groups = {
        "point_head.scratch.output_conv2": [], "point_head.scratch.output_conv1": [],
        "point_head.scratch.refinenet1": [], "point_head.scratch.refinenet4": [],
        "point_head.scratch.layer4_rn": [], "point_head.projects": [],
        "point_head.norm": [],
        "aggregator.global_blocks.23": [], "aggregator.global_blocks.0": [],
        "aggregator.frame_blocks.23": [], "aggregator.frame_blocks.0": [],
        "aggregator.camera_token": [], "aggregator.z_embedder": [],
    }
    for n, p in model.named_parameters():
        for g in groups:
            if n.startswith(g):
                groups[g].append(0.0 if p.grad is None else p.grad.float().norm().item() ** 2)
                break
    for g, v in groups.items():
        tot = sum(v) ** 0.5 if v else float("nan")
        n_none = sum(1 for n, p in model.named_parameters() if n.startswith(g) and p.grad is None)
        print(f"  {g:34s} gradnorm = {tot:.6e}   n_params={len(v)}  n_grad_is_None={n_none}")

    # the exact quantity the trainer logs
    agg = [p.grad.float().norm().item() ** 2 for n, p in model.named_parameters()
           if n.startswith("aggregator") and p.requires_grad and p.grad is not None]
    pt = [p.grad.float().norm().item() ** 2 for n, p in model.named_parameters()
          if n.startswith("point_head") and p.requires_grad and p.grad is not None]
    print(f"\n  >>> TRAINER-EQUIVALENT  grad_aggregator = {sum(agg) ** 0.5:.6e}")
    print(f"  >>> TRAINER-EQUIVALENT  grad_point      = {sum(pt) ** 0.5:.6e}")

    for h in handles:
        h.remove()
    model.aggregator.forward = orig_agg_forward

    # ---- Q2 input-independence: second, different batch ----
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        b2 = build_batch(device, seed=999)
        d2 = model(b2["images"], batch=b2)["dvfs"]
    print(f"\n--- Q2 input independence (batch seed 1 vs 999) ---")
    print(f"  mean|dvf_1|            = {dvf.float().abs().mean().item():.6e}")
    print(f"  mean|dvf_2|            = {d2.float().abs().mean().item():.6e}")
    print(f"  mean|dvf_1 - dvf_2|    = {(dvf.float() - d2.float()).abs().mean().item():.6e}")
    rel = (dvf.float() - d2.float()).abs().mean().item() / (dvf.float().abs().mean().item() + 1e-12)
    print(f"  RELATIVE difference    = {rel:.6e}   <-- ~0 => output ignores the input")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/ck_aug.pt")
    ap.add_argument("--base", default="scratch/base_weights/vggt1b_base.pt")
    a = ap.parse_args()
    dev = "cuda"
    for tag, ck in [("COLLAPSED  (pooled1337 aug, epoch ~86)", a.ckpt),
                    ("CONTROL    (base VGGT-1B)", a.base)]:
        if not os.path.exists(ck):
            print(f"skip {tag}: {ck} missing")
            continue
        m = make_model(dev, ck)
        run(m, tag, dev)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
