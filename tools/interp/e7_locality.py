"""E7 — is the reference read LOCALLY (companion pixel <- reference at the same in-plane location)?

Source run: reference at ED (a = 0). Donor: reference at the phase least similar to ED (b ≈ ES).
In global layer 3 only, companions read the donor's slot-0 keys/values for a SUBSET of reference
patch tokens:
  half_L / half_R : reference patch columns left / right of the heart centroid
  blk{j}          : a 3x3-token block (42x42 px of the 518 input ≈ 29 mm) at 4 positions on the heart
Outcome: companion in-plane displacement change vs the source run,
  d(x) = ||Δ_patch(x) - Δ_src(x)||,  F(x) = ||Δ_don(x) - Δ_src(x)||  (mm, heart-ROI pixels),
aggregated by where x lies (left/right of the centroid; distance from the block centre).
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np          # noqa: E402
import torch                # noqa: E402
import torch.nn.functional as F   # noqa: E402
import harness as H         # noqa: E402

LAYER = [3]


def run_subject(model, cfg, ds, idx, dev):
    subj = H.Subject(ds, idx, dev)
    if subj.heart is None:
        return None
    rng = np.random.default_rng(1000 + idx)
    comp = rng.integers(0, subj.T, subj.S)
    disp, _ = subj.sample_breathing(H.resp_cfg(cfg), seed=2000 + idx)
    sim = [H.ncc(subj.phases[t], subj.phases[0], subj.heart) for t in range(subj.T)]
    a, b_ = 0, int(np.argmin(sim))
    G, R, P = subj.R // 14, subj.R, H.N_SPECIAL + (subj.R // 14) ** 2

    def bt(t):
        ts = comp.copy(); ts[0] = t
        return H.build_batch(subj, ts, disp)

    bs, bd = bt(a), bt(b_)
    hp = H.heart_patch_mask(subj, bs)[0].view(G, G).cpu().numpy()
    rr, cc = np.nonzero(hp)
    cr, ccen = rr.mean(), cc.mean()
    o_src = H.forward(model, subj, bs)
    o_don = H.forward(model, subj, bd, kv_mode="save", kv_layers=LAYER)
    col = np.arange(G)[None, :].repeat(G, 0)
    row = np.arange(G)[:, None].repeat(G, 1)
    sets = {"half_L": col < ccen, "half_R": col >= ccen}
    offs = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
    for j, (dr, dc) in enumerate(offs):
        r0, c0 = int(round(cr + dr)), int(round(ccen + dc))
        sets[f"blk{j}"] = (np.abs(row - r0) <= 1) & (np.abs(col - c0) <= 1)
    # companion heart pixels (518 grid) and their coordinates
    heart_px = []
    for s in range(1, subj.S):
        h = subj.heart[int(bs["slot_z"][s])].float()[None, None]
        heart_px.append(F.interpolate(h, size=(R, R), mode="nearest")[0, 0] > 0)
    heart_px = torch.stack(heart_px)                          # (S-1,R,R)
    mm = torch.tensor(H.MM_PER_NORM[:2], device=dev, dtype=torch.float32)
    dsrc = o_src["dvf"][0, 1:, ..., :2] * mm                  # (S-1,R,R,2) in-plane mm (x,y)
    Fmag = ((o_don["dvf"][0, 1:, ..., :2] * mm - dsrc).norm(dim=-1))
    ys, xs = torch.meshgrid(torch.arange(R, device=dev), torch.arange(R, device=dev), indexing="ij")
    pix_col = xs.float() / 14.0 - 0.5                          # pixel -> patch-grid coordinate
    pix_row = ys.float() / 14.0 - 0.5
    rec = {"idx": idx, "a": a, "b": b_, "centroid": (float(cr), float(ccen)), "sets": {}}
    m = heart_px
    left = (pix_col < ccen)[None].expand_as(m)
    rec["F_left"] = float(Fmag[m & left].mean()); rec["F_right"] = float(Fmag[m & ~left].mean())
    for name, tokmask in sets.items():
        kv_tok = torch.zeros(P, dtype=torch.bool)
        kv_tok[H.N_SPECIAL:] = torch.from_numpy(tokmask.reshape(-1))
        o = H.forward(model, subj, bs, kv_mode="patch", kv_layers=LAYER, kv_tok=kv_tok)
        dmag = ((o["dvf"][0, 1:, ..., :2] * mm - dsrc).norm(dim=-1))
        e = {"d_left": float(dmag[m & left].mean()), "d_right": float(dmag[m & ~left].mean())}
        if name.startswith("blk"):
            r0 = float(np.mean(np.nonzero(tokmask)[0])); c0 = float(np.mean(np.nonzero(tokmask)[1]))
            dist = ((pix_row - r0) ** 2 + (pix_col - c0) ** 2).sqrt() * 14 * (255 * 1.4 / (R - 1))  # mm
            bins = np.arange(0, 60, 5)
            dist_e = dist[None].expand_as(m)
            prof, prof_F = [], []
            for lo in bins:
                sel = m & (dist_e >= lo) & (dist_e < lo + 5)
                prof.append(float(dmag[sel].mean()) if sel.any() else np.nan)
                prof_F.append(float(Fmag[sel].mean()) if sel.any() else np.nan)
            e["prof"], e["prof_F"], e["bins"] = prof, prof_F, bins.tolist()
        rec["sets"][name] = e
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
        fp = os.path.join(args.out, f"s{i:03d}.pkl")
        if os.path.exists(fp):
            continue
        t0 = time.time()
        rec = run_subject(model, cfg, ds, i, dev)
        pickle.dump(rec, open(fp, "wb"))
        print(f"[{i}] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
