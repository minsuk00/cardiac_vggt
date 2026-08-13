#!/usr/bin/env python3
"""
tools/exp_relu_death_vs_lr.py  —  Does the LR decide whether the DPT head's ReLU dies?

docs/64 PROVED the pooled1337 runs died because `point_head.scratch.output_conv2[1]`
(a ReLU) went 100% dead: output exactly 0 => head emits only its bias => constant,
input-independent DVF => zero gradient to the aggregator, permanently.

It did NOT establish whether the 3e-4 peak LR caused that, or merely hastened something
that 5e-5 would also reach. This script settles it by MEASURING collapse time as a
function of LR, on the real native-z pipeline, at CONSTANT LR (no warmup/cosine) so LR is
the only variable.

Why constant LR: in the real run the schedule ramped 1e-8 -> 3e-4 across 15 epochs
(~14,025 steps), so the model saw peak LR for only ~2,500 steps before dying. At constant
LR the same regime is reached from step 0, making the experiment cheap.

Read-out per arm: the step at which the ReLU becomes fully dead (or "survived N steps").
If collapse-step scales ~1/LR, the drift is LR-driven and 5e-5's collapse step can be
extrapolated and compared against the 300-epoch run length (~280k steps).

Usage:
  PYTHONPATH=training:. python tools/exp_relu_death_vs_lr.py --lr 3e-4 --steps 3000 \
      --out scratch/relu_lr_exp/lr3e-4.jsonl
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("training"))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REPO = "/home/minsukc/vggt"


def build(lr, seed):
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from train_utils.freeze import freeze_modules
    from vggt.utils.checkpoint_stage import stage_checkpoint_to_local

    for n, f in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
                 ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
        OmegaConf.register_new_resolver(n, f, replace=True)
    with initialize_config_dir(version_base=None, config_dir=f"{REPO}/training/config"):
        cfg = compose(config_name="default", overrides=[
            "one_frame_per_slice=true",
            "data.augmentation.enable=true", "data.augmentation.tier=moderate"])

    model = instantiate(cfg.model, _recursive_=False).cuda()
    # REAL base weights: the starting ReLU margin is the whole point, so random init is wrong.
    ck = stage_checkpoint_to_local("scratch/base_weights/vggt1b_base.pt")
    sd = torch.load(ck, map_location="cpu", weights_only=True)
    model.load_state_dict(sd.get("model", sd), strict=False)
    model = freeze_modules(model, patterns=cfg.optim.frozen_module_names)
    model.train()

    loss_fn = instantiate(cfg.loss, _recursive_=False)
    # Same optimizer as the real run, but CONSTANT lr (no scheduler) so LR is isolated.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05, fused=True)
    train_ds = instantiate(cfg.data.train, _recursive_=False)
    train_ds.seed = seed

    # The train dataset DEFERS the 64MB/sample `images` build (docs/61); gpu_augment_batch
    # is what creates it. Replicate trainer.py:999 exactly, or batch["images"] is missing.
    from data.gpu_aug import build_gpu_transforms
    from data.respiratory import RespiratoryConfig
    aug_cfg = cfg.data.get("augmentation", None)
    gpu_transforms = build_gpu_transforms(aug_cfg)
    resp_cfg = RespiratoryConfig.from_cfg(
        aug_cfg.get("respiratory", None) if aug_cfg is not None else None)
    return cfg, model, loss_fn, opt, train_ds, gpu_transforms, resp_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    from data.gpu_aug import gpu_augment_batch
    cfg, model, loss_fn, opt, train_ds, gpu_transforms, resp_cfg = build(a.lr, a.seed)
    oc2 = model.point_head.scratch.output_conv2
    cap = {}
    # ReLU is inplace=True => a pre-hook on the ReLU is the only place the true
    # pre-activation is still intact.
    # .clone() is LOAD-BEARING: ReLU(inplace=True) overwrites this very tensor, so a bare
    # .detach() would hand us the POST-activation and report pre_min == 0.0 always.
    oc2[1].register_forward_pre_hook(lambda m, i: cap.__setitem__("pre", i[0].detach().clone()))

    clipper = None
    if getattr(cfg.optim, "gradient_clip", None) is not None:
        from hydra.utils import instantiate
        clipper = instantiate(cfg.optim.gradient_clip, _recursive_=False)
        clipper.setup_clipping(model)

    f = open(a.out, "w")
    step, t0, dead_at = 0, time.time(), None
    print(f"[exp] lr={a.lr} steps={a.steps} out={a.out}", flush=True)
    for epoch in range(1000):
        if step >= a.steps or dead_at is not None:
            break
        for batch in train_ds.get_loader(epoch=epoch):
            if step >= a.steps or dead_at is not None:
                break
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            # trainer.py:999 — also what BUILDS batch["images"] under defer_input_images.
            batch = gpu_augment_batch(batch, gpu_transforms, "cuda",
                                      respiratory_cfg=resp_cfg, train=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                preds = model(images=batch["images"], batch=batch)
                ld = loss_fn(preds, batch)
            ld["objective"].backward()
            gnorms = clipper(model) if clipper else {}
            opt.step()

            if step % a.every == 0:
                p = cap["pre"].float()
                dead_frac = (p <= 0).float().mean().item()
                chan_dead = (p <= 0).all(0).all(-1).all(-1).float().mean().item()
                agg = sum(q.grad.float().norm().item() ** 2 for n, q in model.named_parameters()
                          if n.startswith("aggregator") and q.grad is not None) ** 0.5
                rec = {"step": step, "lr": a.lr, "epoch": epoch,
                       "pre_min": p.min().item(), "pre_mean": p.mean().item(),
                       "pre_max": p.max().item(), "dead_frac": dead_frac,
                       "chan_dead_everywhere": chan_dead,
                       "grad_aggregator": agg,
                       "grad_point": gnorms.get("point", float("nan")),
                       "objective": float(ld["objective"]),
                       "loss_diffusion": float(ld.get("loss_diffusion", 0.0)),
                       "t": time.time() - t0}
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"  step {step:5d} pre[min/mean/max]={p.min():+.3e}/{p.mean():+.3e}/"
                      f"{p.max():+.3e} dead={dead_frac:.4f} chan_dead={chan_dead:.3f} "
                      f"gagg={agg:.3e} obj={float(ld['objective']):.5f}", flush=True)
                if chan_dead >= 1.0:
                    dead_at = step
                    print(f"  *** COLLAPSED at step {step} (all channels dead) ***", flush=True)
            step += 1

    f.write(json.dumps({"final": True, "lr": a.lr, "dead_at": dead_at,
                        "steps_run": step}) + "\n")
    f.close()
    print(f"[exp] lr={a.lr} dead_at={dead_at} steps_run={step}", flush=True)


if __name__ == "__main__":
    main()
