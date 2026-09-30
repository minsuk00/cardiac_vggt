"""E9 — feature-level view: WHERE does reference-phase information enter the companions' tokens?

Final model, two runs that differ ONLY in the reference phase (ED vs the phase least similar to ED;
same companions + breathing draw as E1b/E5). For every frame/global block output, the relative change
of companion tokens ||f_b - f_a|| / ||f_a||, separately for heart-ROI patch tokens, other patch tokens
and special tokens. Before the reference is read, companion tokens cannot depend on the reference
phase, so the change should be ~0 until the reading layer.
A few subjects also dump global-block token features for VGGT-Ω-style PCA maps.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402

PCA_LAYERS = [2, 3, 4, 11, 14, 17, 23]


class Capture:
    def __init__(self, model):
        self.hs, self.out = [], {}
        agg = model.aggregator
        for kind in ("frame", "global"):
            for i, blk in enumerate(getattr(agg, f"{kind}_blocks")):
                self.hs.append(blk.register_forward_hook(self._hook(kind, i)))

    def _hook(self, kind, i):
        def f(mod, inp, out):
            self.out[(kind, i)] = out.detach().float().reshape(-1, out.shape[-1])   # (S*P, C)
        return f

    def remove(self):
        for h in self.hs:
            h.remove()


def token_groups(subj, b, P):
    S = len(b["slot_z"])
    hp = H.heart_patch_mask(subj, b).cpu()
    heart = torch.zeros(S, P, dtype=torch.bool); heart[:, H.N_SPECIAL:] = hp
    patch = torch.zeros(S, P, dtype=torch.bool); patch[:, H.N_SPECIAL:] = True
    comp = torch.zeros(S, P, dtype=torch.bool); comp[1:] = True
    return {"heart": (heart & comp).reshape(-1), "bg": (patch & ~heart & comp).reshape(-1),
            "special": (~patch & comp).reshape(-1)}


@torch.no_grad()
def run_subject(model, cfg, ds, idx, dev, save_pca):
    subj = H.Subject(ds, idx, dev)
    if subj.heart is None:
        return None
    rng = np.random.default_rng(1000 + idx)
    comp = rng.integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
    sim = [H.ncc(subj.phases[t], subj.phases[0], subj.heart) for t in range(subj.T)]
    a, b_ = 0, int(np.argmin(sim))
    ts_a = comp.copy(); ts_a[0] = a
    ts_b = comp.copy(); ts_b[0] = b_
    ba, bb = H.build_batch(subj, ts_a, disp), H.build_batch(subj, ts_b, disp)
    P = H.N_SPECIAL + (subj.R // 14) ** 2
    groups = token_groups(subj, ba, P)
    cap = Capture(model)
    H.forward(model, subj, ba); fa = dict(cap.out)
    H.forward(model, subj, bb); fb = dict(cap.out)
    cap.remove()
    change = {}
    for key in fa:
        d = (fb[key] - fa[key]).norm(dim=-1) / fa[key].norm(dim=-1).clamp_min(1e-6)
        change[key] = {g: float(d[m.to(d.device)].mean()) for g, m in groups.items()}
    rec = {"idx": idx, "S": subj.S, "slot_z": subj.slot_z.tolist(), "a": a, "b": b_, "change": change}
    if save_pca:
        rec["pca_tokens"] = {l: fa[("global", l)].half().cpu().numpy() for l in PCA_LAYERS}
        rec["pca_tokens_b"] = {l: fb[("global", l)].half().cpu().numpy() for l in PCA_LAYERS}
        rec["images_a"] = ba["images_splat"][0].cpu().numpy().astype(np.float16)
        rec["images_b"] = bb["images_splat"][0].cpu().numpy().astype(np.float16)
        rec["heart_patch"] = H.heart_patch_mask(subj, ba).cpu().numpy()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    os.makedirs(args.out, exist_ok=True)
    for i in range(0, len(ds.subjects), args.every):
        fp = os.path.join(args.out, f"s{i:03d}.pkl")
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev, save_pca=(i % 30 == 0))
        pickle.dump(rec, open(fp, "wb"))
        print(f"[{i}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
