#!/usr/bin/env python
"""Train the per-slice frame selector (docs/128). No CardioStitch in the loop.

Train = pooled_curated_v2 [train] (628), model selection = [val] (90, fixed seeded episodes).
Episodes = gated cine + continuous frame-wise breathing, generated on the fly on the GPU
(tools/framesel_data.py). Features = CardioStitch's own frozen patch embedding at 518.

    PYTHONPATH=training:. python tools/framesel_train.py --variant B --out scratch/framesel/runs/B
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
import framesel_data as fd                                                      # noqa: E402
from framesel_model import FrameSelector, FrozenPatchEmbed, selector_loss       # noqa: E402
from data.datasets.mri_dataset import MRIDataset                                # noqa: E402
from inference.load_run import load_model_from_run, mri_dataset_kwargs          # noqa: E402

CKPT = "scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt"


def make_ds(cfg, split):
    kw = dict(mri_dataset_kwargs(cfg, "train"))
    for k in ("_target_", "split", "ef_val_sweep"):
        kw.pop(k, None)
    kw["defer_input_images"] = True
    common = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                               "rescale_aug": False, "landscape_check": False,
                               "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(common, kw.pop("data_root"), split=split, **kw)


class Subjects(torch.utils.data.Dataset):
    """CPU side: one cached cine per item (random subject, or subject i when seeded for val)."""

    def __init__(self, ds, n, seeded=False, only=None):
        self.ds, self.n, self.seeded, self.only = ds, n, seeded, only

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.only is not None:                    # single-subject learnability test
            s = self.only
        else:
            s = i % len(self.ds.subjects) if self.seeded else random.randrange(len(self.ds.subjects))
        b = self.ds.get_data(seq_index=s, img_per_seq=self.ds.num_slices)
        return {k: b[k] for k in ("phases", "dz_mm", "slice_indices", "z_indices")}


def forward(sel, emb, ep, dev):
    P, Fn = ep["cand_img"].shape[:2]
    Fr = ep["ref_plane_img"].shape[0]
    tok = emb(torch.cat([ep["ref_plane_img"], ep["cand_img"].flatten(0, 1)]))
    ref_plane, cand = tok[:Fr], tok[Fr:].view(P, Fn, *tok.shape[1:])
    logits = sel(cand, ref_plane[ep["ref_idx"]],
                 torch.as_tensor(ep["cand_z"], dtype=torch.float32, device=dev), float(ep["ref_z"]),
                 ref_plane_tok=ref_plane)
    dist = torch.as_tensor(ep["label_dist"], dtype=torch.float32, device=dev)
    return logits, dist


@torch.no_grad()
def evaluate(sel, emb, val_items, resp_cfg, dev, sigma, n_planes=4):
    """Phase error of the argmax frame on fixed val episodes (seed per subject)."""
    sel.eval()
    err, rnd, losses = [], [], []
    for i, it in enumerate(val_items):
        ep = fd.make_episode(it, resp_cfg, np.random.default_rng(12345 + i), dev, n_planes=n_planes)
        logits, dist = forward(sel, emb, ep, dev)
        d = dist.cpu().numpy()
        err += d[np.arange(d.shape[0]), logits.argmax(-1).cpu().numpy()].tolist()
        rnd += d.mean(-1).tolist()                    # expected error of a random frame
        losses.append(float(selector_loss(logits, dist, sigma)))
    sel.train()
    e = np.array(err)
    return {"err": float(e.mean()), "exact": float((e == 0).mean()), "le1": float((e <= 1).mean()),
            "random": float(np.mean(rnd)), "loss": float(np.mean(losses))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B", choices=["A", "B"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=0.5, help="soft-label width, in cardiac phases")
    ap.add_argument("--n-planes", type=int, default=3, help="candidate planes per episode")
    ap.add_argument("--accum", type=int, default=4, help="episodes per optimizer step")
    ap.add_argument("--center", action="store_true",
                    help="subtract each plane's mean token over its frames (candidate planes and the "
                         "reference plane) before scoring")
    ap.add_argument("--norm", default="ln", choices=["ln", "slice"],
                    help="ln = per-token LayerNorm before the projection; slice = one RMS scale per plane")
    ap.add_argument("--attn-pool", action="store_true", help="learned attention pooling over tokens")
    ap.add_argument("--one-subject", type=int, default=None,
                    help="learnability test: train AND validate on this one train-split subject index "
                         "(val = fresh seeded draws of target phase, planes, breathing)")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=90)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)

    vggt, cfg = load_model_from_run(CKPT, device=dev)
    emb = FrozenPatchEmbed(vggt).to(dev)
    del vggt
    torch.cuda.empty_cache()
    resp_cfg = dict(cfg["data"]["augmentation"]["respiratory"])

    train_ds, val_ds = make_ds(cfg, "train"), make_ds(cfg, "val")
    print(f"train {len(train_ds.subjects)}  val {len(val_ds.subjects)}", flush=True)
    loader = torch.utils.data.DataLoader(Subjects(train_ds, args.steps * args.accum, only=args.one_subject),
                                         batch_size=None,
                                         num_workers=args.workers, collate_fn=lambda x: x,
                                         worker_init_fn=lambda w: random.seed(1000 + w),
                                         persistent_workers=True, prefetch_factor=4)
    vs = (Subjects(val_ds, args.n_val, seeded=True) if args.one_subject is None
          else Subjects(train_ds, args.n_val, seeded=True, only=args.one_subject))
    val_items = [vs[i] for i in range(args.n_val)]

    sel = FrameSelector(args.variant, center=args.center, norm=args.norm,
                        attn_pool=args.attn_pool).to(dev)
    opt = torch.optim.AdamW(sel.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=max(0.05, 2 / args.steps))
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=2)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    rng = np.random.default_rng(0)
    best, t0, it = 1e9, time.time(), iter(loader)
    for step in range(1, args.steps + 1):
        tot = 0.0
        for _ in range(args.accum):
            ep = fd.make_episode(next(it), resp_cfg, rng, dev, n_planes=args.n_planes)
            logits, dist = forward(sel, emb, ep, dev)
            loss = selector_loss(logits, dist, args.sigma) / args.accum
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(sel.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {tot:.4f} {time.time() - t0:.0f}s", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            m = evaluate(sel, emb, val_items, resp_cfg, dev, args.sigma)
            m.update(step=step, train_loss=tot, sec=time.time() - t0)
            print(json.dumps(m), flush=True)
            log.write(json.dumps(m) + "\n"); log.flush()
            ck = {"model": sel.state_dict(), "args": vars(args), "step": step}
            torch.save(ck, os.path.join(args.out, "last.pt"))
            if m["err"] < best:
                best = m["err"]
                torch.save(ck, os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
