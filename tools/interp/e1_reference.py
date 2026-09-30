"""E1 — how does the reference slice's cardiac state reach the rest of the stack?

Per test subject: fix the companion phases (random scatter, seeded) and the breathing (val-style
draw, seeded), then sweep the reference phase t = 0..T-1 (the target). Repeat under interventions
on GLOBAL attention (the only cross-slice pathway in the network):

  full            unmodified model
  ko_ref          no companion token may attend to ANY slot-0 token (all 24 global layers)
  ko_ref_sp       ... to slot-0's 5 special tokens only (camera+ref+z token, registers)
  ko_ref_pt       ... to slot-0's patch tokens only
  ko_ref_L{a}_{b} ko_ref restricted to global layers a..b
  self_only       every token attends only within its own slot in every global layer (no cross-slice)
  noref_tok       e_ref removed: slot 0 gets the 'other' camera+register tokens (no designated ref)

Outputs per (subject, condition, t): per-slot dz/dy/dx (mm, FOV + heart), obs error per slot, and
NCC(V_t, GT_t') for all t' on (a) companion planes (all planes except the reference plane) and
(b) the reference plane. Identity (Δ=0) splat is stored as a floor.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import harness as H         # noqa: E402


def conditions(S):
    allow_ko = torch.ones(2 * S, 2 * S, dtype=torch.bool)
    allow_ko[2:, 0:2] = False                        # companions -/-> slot 0 (special + patch)
    allow_sp = torch.ones(2 * S, 2 * S, dtype=torch.bool)
    allow_sp[2:, 0] = False                          # companions -/-> slot-0 special
    allow_pt = torch.ones(2 * S, 2 * S, dtype=torch.bool)
    allow_pt[2:, 1] = False                          # companions -/-> slot-0 patches
    slot = torch.arange(2 * S) // 2
    allow_self = slot[:, None] == slot[None, :]
    c = {"full": (None, None), "ko_ref": (allow_ko, None), "ko_ref_sp": (allow_sp, None),
         "ko_ref_pt": (allow_pt, None), "self_only": (allow_self, None), "noref_tok": (None, None)}
    for a in range(0, 24, 6):
        c[f"ko_ref_L{a}_{a + 5}"] = (allow_ko, set(range(a, a + 6)))
    for a in (6, 12, 18):
        c[f"ko_ref_from{a}"] = (allow_ko, set(range(a, 24)))
    for a in range(6):                              # single early layers + cumulative 0..a
        c[f"ko_ref_only{a}"] = (allow_ko, {a})
        c[f"ko_ref_upto{a}"] = (allow_ko, set(range(a + 1)))
    return c


class NoRefToken:
    """Temporarily give slot 0 the 'other' camera + register tokens (no designated reference)."""

    def __init__(self, model):
        self.agg = model.aggregator

    def __enter__(self):
        self.c = self.agg.camera_token.data.clone()
        self.r = self.agg.register_token.data.clone()
        self.agg.camera_token.data[:, 0] = self.c[:, 1]
        self.agg.register_token.data[:, 0] = self.r[:, 1]

    def __exit__(self, *a):
        self.agg.camera_token.data.copy_(self.c)
        self.agg.register_token.data.copy_(self.r)


def run_subject(model, cfg, ds, idx, dev, conds_filter=None):
    subj = H.Subject(ds, idx, dev)
    rng = np.random.default_rng(1000 + idx)
    comp = rng.integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
    ref_plane = subj.ref_plane
    comp_planes = [z for z in range(subj.D) if z != ref_plane]
    rec = {"subject": subj.name, "idx": idx, "D": subj.D, "S": subj.S, "dz": subj.dz, "T": subj.T,
           "slot_z": subj.slot_z.tolist(), "comp_t": comp.tolist(), "ref_plane": ref_plane,
           "disp": disp.cpu().numpy().tolist(), "has_heart": subj.heart is not None, "cond": {}}
    if subj.heart is None:
        return rec
    conds = conditions(subj.S)
    names = list(conds) + ["identity"]
    if conds_filter:
        names = [n for n in names if n in conds_filter]
    for name in names:
        rows = []
        for t in range(subj.T):
            ts = comp.copy()
            ts[0] = t
            b = H.build_batch(subj, ts, disp)
            if name == "identity":
                wp = b["scanner_coords"].clone()
            elif name == "noref_tok":
                with NoRefToken(model):
                    wp = H.forward(model, subj, b)["wp"]
            else:
                allow, layers = conds[name]
                wp = H.forward(model, subj, b, allow=allow, layers=layers)["wp"]
            dvf = wp - b["scanner_coords"]
            V, _ = H.splat(subj, b, wp)
            rows.append({
                "t": t,
                "disp_fov": H.slot_disp_mm(subj, b, dvf, "fov").tolist(),
                "disp_heart": H.slot_disp_mm(subj, b, dvf, "heart").tolist(),
                "obs": H.obs_err(subj, b, wp, t).tolist(),
                "ncc_comp": H.phase_ncc_profile(subj, V, comp_planes).tolist(),
                "ncc_ref": H.phase_ncc_profile(subj, V, [ref_plane]).tolist(),
                "ncc_all": H.phase_ncc_profile(subj, V).tolist(),
            })
        rec["cond"][name] = rows
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", nargs="*", default=None)
    args = ap.parse_args()
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    n = len(ds.subjects) if hasattr(ds, "subjects") else ds.len_train
    idxs = list(range(n))[args.shard::args.nshard]
    if args.limit:
        idxs = idxs[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    for i in idxs:
        fp = os.path.join(args.out, f"s{i:03d}.json")
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev, args.conds)
        json.dump(rec, open(fp, "w"))
        print(f"[{i}] {rec['subject']} S={rec['S']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
