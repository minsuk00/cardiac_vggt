"""Is PROXIMITY sampling worth a retrain? — warp ceiling under random vs near±1 composition.

toy_contract_levers found near±1 companions beat random by +0.8 dB with the CURRENT model (trained
on random composition → OOD on near-phase input). Question: is +0.8 a floor (untrained ceiling is
higher → a retrain on proximity composition would pay off) or is the near-phase warp CEILING also
capped at ~the wall (→ +0.8 is all proximity can ever give)?

Method: target-aware direct-opt of a free per-pixel Δ (the warp-only ceiling, same as toy_warpceiling)
on each composition at fixed budget S=8, K=0 (no exact target frame), clean protocol, identical z-set.
If ceiling(near1) >> ceiling(random) → proximity raises the achievable ceiling ⇒ train for it.
If ceiling(near1) ~ ceiling(random) → proximity is capped at the wall ⇒ +0.8 is the whole prize.

Run: micromamba run -n svr python tools/toy_proximity_ceiling.py --n 8 --steps 1500
"""
import argparse, json, os, sys, random
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
import tools.toy_contract_levers as L                                   # build_batch, etc.
from loss import compute_motion_mask                                    # noqa: E402
from vggt.utils.splat import splat_to_volume                           # noqa: E402
import tools.eval_variants_matrix as Hm                                # noqa: E402
D = 12; OUT = os.path.join(REPO, "result", "limits_eval")


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def to256(t_bshwc):  # (1,S,518,518,3)->(S,256,256,3) ; or intensity (1,S,518,518)->(S,256,256)
    if t_bshwc.dim() == 5:
        x = t_bshwc[0].permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()
    x = t_bshwc[0].unsqueeze(1)
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]


def optimize(scanner, inten, V_gt, mmask, steps, lr=0.005, tv=0.05):
    delta = torch.zeros_like(scanner, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr); Vgt_b = V_gt.unsqueeze(0); best = -1
    intf = inten.reshape(1, -1); w = (intf > 1e-3).float()
    for it in range(steps):
        world = (scanner + delta).clamp(-1.05, 1.05)
        V, _ = splat_to_volume(world.reshape(1, -1, 3), intf, (D, 256, 256), weight=w)
        l1 = (V - Vgt_b).abs().mean()
        loss = l1 + tv * ((delta[:, 1:] - delta[:, :-1]).abs().mean() + (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == steps - 1:
            with torch.no_grad():
                best = max(best, psnr(V[0].float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
    return best


def splat_id(scanner, inten):
    intf = inten.reshape(1, -1); w = (intf > 1e-3).float()
    V, _ = splat_to_volume(scanner.reshape(1, -1, 3), intf, (D, 256, 256), weight=w)
    return V[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--s", type=int, default=8)
    args = ap.parse_args(); dev = "cuda"
    ds = Hm.build_dataset(); print("val subjects:", len(ds.subjects))
    res = {"random": [], "near1": []}; idf = {"random": [], "near1": []}
    seq = 0; got = 0
    while got < args.n and seq < 200:
        i = seq; seq += 1
        data = ds.get_data(seq_index=i, img_per_seq=Hm.NUM_SLICES)
        phases = np.asarray(data["phases"]); gt = np.asarray(data["gt_target_volume"])
        bbox = [int(v) for v in np.asarray(data["anatomy_bbox"]).tolist()]
        t_target = int(np.asarray(data["t_target"]).flatten()[0]); T = phases.shape[0]
        in_bbox = list(range(bbox[0], bbox[1]))
        if len(in_bbox) < args.s:
            continue
        rng = random.Random(1000 + i); z_set = rng.sample(in_bbox, args.s)
        others = [t for t in range(T) if t != t_target]
        Vgt = torch.from_numpy(gt.astype(np.float32)).to(dev)
        mmask = compute_motion_mask(torch.from_numpy(phases.astype(np.float32)).unsqueeze(0).to(dev))[0].cpu().numpy()
        if not mmask.any():
            continue
        pools = {"random": others, "near1": [(t_target + d) % T for d in (-1, 1)]}
        for name, pool in pools.items():
            rb = random.Random(7000 + i); t_seq = [rb.choice(pool) for _ in range(args.s)]
            b = L.build_batch(phases, gt, bbox, t_target, t_seq, z_set, dev)
            sc = to256(b["scanner_coords"]); inten = to256(b["images"].mean(dim=2))
            idf[name].append(psnr(splat_id(sc, inten).float().cpu().numpy(), Vgt.cpu().numpy(), mmask))
            res[name].append(optimize(sc, inten, Vgt, mmask, args.steps))
        got += 1
        print(f"seq{i:2d} t{t_target} | random: id {idf['random'][-1]:.1f} ceil {res['random'][-1]:.1f}  "
              f"| near1: id {idf['near1'][-1]:.1f} ceil {res['near1'][-1]:.1f}")
    summ = {"n": got, "steps": args.steps, "S": args.s,
            "random": {"identity": float(np.mean(idf["random"])), "warp_ceiling": float(np.mean(res["random"]))},
            "near1": {"identity": float(np.mean(idf["near1"])), "warp_ceiling": float(np.mean(res["near1"]))}}
    summ["ceiling_gain_near1_vs_random"] = summ["near1"]["warp_ceiling"] - summ["random"]["warp_ceiling"]
    json.dump(summ, open(os.path.join(OUT, "toy_proximity_ceiling.json"), "w"), indent=2)
    print(f"\n=== WARP CEILING by composition (n={got}, clean, S={args.s}, motion PSNR) ===")
    print(f"  random : identity {summ['random']['identity']:.2f}  warp-ceiling {summ['random']['warp_ceiling']:.2f}")
    print(f"  near±1 : identity {summ['near1']['identity']:.2f}  warp-ceiling {summ['near1']['warp_ceiling']:.2f}")
    print(f"  ceiling gain (near1 - random) = {summ['ceiling_gain_near1_vs_random']:+.2f} dB")
    print("saved", os.path.join(OUT, "toy_proximity_ceiling.json"))


if __name__ == "__main__":
    main()
