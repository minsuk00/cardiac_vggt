"""E5 — sufficiency: is what companions READ from the reference in early global layers enough to
set the reconstructed cardiac state?

Two clean runs per subject differing ONLY in the reference frame: ref at phase a (ED = 0) and at
phase b (the GT phase least similar to ED inside the heart ROI ≈ ES). Then the "patched" run: input
identical to run a, but in global layers L the companion slots attend to run b's slot-0 keys/values
(the reference's own residual stream is untouched). If the companion planes of the patched output
move to phase b, the early-layer read of the reference is SUFFICIENT to set the cardiac state.
Only the companion->reference READS are swapped (slot 0's own queries use live K/V; its residual
stream is unchanged until the first patched layer, after which it attends to companions that carry
the donor information). Primary metric uses the COMPANION-ONLY splat (reference slice's pixels removed,
no leakage into neighbouring planes).
Done in both directions (a->b, b->a). State index on companion planes (0 = source state, 1 = donor):
  s(V) = NCC(V, GT_donor) - NCC(V, GT_source);   idx = (s(V_patch) - s(V_src)) / (s(V_donor) - s(V_src))
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

LAYER_SETS = {"L0_5": range(0, 6), "L6_11": range(6, 12), "L12_17": range(12, 18), "L18_23": range(18, 24),
              "L0_23": range(24), "L0_2": range(0, 3), "L3_5": range(3, 6), "L0_23_but3": [l for l in range(24) if l != 3]}
LAYER_SETS.update({f"L{i}": [i] for i in range(12)})
HEAD_LAYER = 3      # per-head scan in the layer found to carry the state (pilot)
N_HEADS = 16
TOP4 = [2, 10, 12, 14]   # selected on pilot subjects idx 0-2 (reported separately as selection set)


def run_subject(model, cfg, ds, idx, dev):
    subj = H.Subject(ds, idx, dev)
    rec = {"subject": subj.name, "idx": idx, "S": subj.S, "D": subj.D, "dirs": []}
    if subj.heart is None:
        return rec
    rng = np.random.default_rng(1000 + idx)
    comp = rng.integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
    comp_planes = [z for z in range(subj.D) if z != subj.ref_plane]
    m = subj.heart
    sim = [H.ncc(subj.phases[t], subj.phases[0], m) for t in range(subj.T)]
    a, b_ = 0, int(np.argmin(sim))
    rec["a"], rec["b"] = a, b_

    def batch_for(t):
        ts = comp.copy(); ts[0] = t
        return H.build_batch(subj, ts, disp)

    def ncc_ab(V, b=None, wp=None):
        """([NCC(V,GT_a), NCC(V,GT_b)] on companion planes, same on the reference plane,
            same on companion planes of the COMPANION-ONLY splat (no reference-slice leakage))."""
        out = (H.phase_ncc_profile(subj, V, comp_planes)[[a, b_]].tolist(),
               H.phase_ncc_profile(subj, V, [subj.ref_plane])[[a, b_]].tolist())
        if b is not None:
            Vc = H.splat_companions(subj, b, wp)[0]
            out = out + (H.phase_ncc_profile(subj, Vc, comp_planes)[[a, b_]].tolist(),)
        return out

    for src, don in ((a, b_), (b_, a)):
        bs, bd = batch_for(src), batch_for(don)
        d = {"src": src, "don": don}
        o_src = H.forward(model, subj, bs)
        o_don = H.forward(model, subj, bd, kv_mode="save", kv_layers=range(24))
        d["src_ncc"] = ncc_ab(H.splat(subj, bs, o_src["wp"])[0], bs, o_src["wp"])
        d["don_ncc"] = ncc_ab(H.splat(subj, bd, o_don["wp"])[0], bd, o_don["wp"])
        d["src_dz"] = H.slot_dz_mm(subj, bs, o_src["dvf"]).tolist()
        # self-patch sanity: patching the source with its OWN kv must reproduce the source
        o_self = H.forward(model, subj, bs, kv_mode="save", kv_layers=range(24))
        o_selfp = H.forward(model, subj, bs, kv_mode="patch", kv_layers=range(24))
        d["selfpatch_maxdiff"] = float((o_selfp["wp"] - o_self["wp"]).abs().max())
        # donor kv again (the self run overwrote the store)
        H.forward(model, subj, bd, kv_mode="save", kv_layers=range(24))
        d["patch"] = {}
        for name, L in LAYER_SETS.items():
            o = H.forward(model, subj, bs, kv_mode="patch", kv_layers=L)
            V = H.splat(subj, bs, o["wp"])[0]
            d["patch"][name] = {"ncc": ncc_ab(V, bs, o["wp"]), "dz": H.slot_dz_mm(subj, bs, o["dvf"]).tolist()}
        for h in range(N_HEADS):
            o = H.forward(model, subj, bs, kv_mode="patch", kv_layers=[HEAD_LAYER], kv_heads=[h])
            V = H.splat(subj, bs, o["wp"])[0]
            d["patch"][f"L{HEAD_LAYER}_h{h}"] = {"ncc": ncc_ab(V, bs, o["wp"])}
        # head subsets found in the 3-subject pilot (h2,h10,h12,h14); fixed a priori for the full run
        for nm, hs in (("L3_top4", TOP4), ("L3_other12", [h for h in range(N_HEADS) if h not in TOP4])):
            o = H.forward(model, subj, bs, kv_mode="patch", kv_layers=[HEAD_LAYER], kv_heads=hs)
            V = H.splat(subj, bs, o["wp"])[0]
            d["patch"][nm] = {"ncc": ncc_ab(V, bs, o["wp"]), "dz": H.slot_dz_mm(subj, bs, o["dvf"]).tolist()}
        rec["dirs"].append(d)
    rec["disp"] = disp.cpu().numpy().tolist()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda:0"
    model, cfg = H.load(dev)
    ds = H.make_test_dataset(cfg)
    idxs = list(range(len(ds.subjects)))[args.shard::args.nshard]
    if args.limit:
        idxs = idxs[: args.limit]
    os.makedirs(args.out, exist_ok=True)
    for i in idxs:
        fp = os.path.join(args.out, f"s{i:03d}.json")
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev)
        json.dump(rec, open(fp, "w"))
        print(f"[{i}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
