"""What raises MOTION PSNR beyond the ~21 dB warp ceiling? — training-free contract probe.

Diagnosis (proven, docs 19/20): the model is warp-limited (~21 ≈ trained), the 14 dB to the
oracle (35) is the appearance wall, and ~7-9 dB of it is SUBJECT-SPECIFIC target-phase appearance
NOT in the sparse inputs. No renderer/decoder change breaks 21 (feature-splat +0.03, splat-res dead).
So the remaining lever must be on the INPUT side. This probes which input change buys motion PSNR,
WITHOUT retraining, by rebuilding input compositions straight from each subject's cached phase bundle.

Two levers (debate-designed), all clean protocol (resp off), fixed S, fixed per-subject z-set, paired:

  LEVER A — COUNT at target phase. K of S input slices placed AT the target cardiac phase (distinct z),
    the rest at fixed random OTHER phases. Sweep K. Per K we read:
      C1  identity-Δ, ONLY the K target slices         = pure OBSERVATION value (model-free)
      C2  identity-Δ, full S composition               = splat-only floor for this exact input
      C3  model,      full S composition               = realizable-now
      C4  model, motion PSNR on HELD-OUT z-planes only  = leak-free PROPAGATION skill
          (C1/C2/C3 score dynamic voxels everywhere incl. the planes where target frames were
           injected → that inflates toward the oracle by construction. C4 scores only z-planes
           that received NO target-phase slice → the honest "does it generalize to unseen planes".)

  LEVER B — PROXIMITY at fixed budget (K=0, no exact target frame, identical z-set, leak-free):
      random : S companions at uniform random phases  (current contract)
      near2  : S companions at t_target ± {1,2}
      near1  : S companions at t_target ± 1
    Tests H3: does sampling NEAR the target beat random at the SAME frame count?

Anchors: oracle (perfect placement) 35; population template 14.4; warp ceiling ~21;
subject_temporal_interp 28.1 (all from prior docs).

Run: micromamba run -n svr python tools/toy_contract_levers.py --s 8 --n 24
"""
import argparse, json, os, sys, random
import numpy as np, torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
import tools.eval_variants_matrix as H                                  # noqa: E402
from vggt.utils.splat import splat_predictions                         # noqa: E402
from loss import compute_motion_mask                                   # noqa: E402

OUT = os.path.join(REPO, "result", "limits_eval"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(H.LOGS, "218747856_mri_volume_resp_allphases_aggft_z_no_t", "ckpts", "checkpoint_last.pt")
GRID = (12, 256, 256); IMG = 518

# constant per-pixel canonical x/y meshgrid for a 518² slice (matches mri_dataset)
_py, _px = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="ij")
X_NORM = (_px.astype(np.float32) / (IMG - 1)) * 2 - 1
Y_NORM = (_py.astype(np.float32) / (IMG - 1)) * 2 - 1


def build_batch(phases, gt_vol, bbox, t_target, t_seq, z_seq, dev):
    """phases (T,D,H,W) np; build a model-ready batch for the given (t,z) composition."""
    T, D = phases.shape[0], phases.shape[1]
    S = len(t_seq)
    sl = torch.from_numpy(np.stack([phases[t, z] for t, z in zip(t_seq, z_seq)])).float()  # (S,256,256)
    up = F.interpolate(sl.unsqueeze(1), size=(IMG, IMG), mode="bilinear", align_corners=True).squeeze(1)  # (S,518,518) ~[0,1]
    imgs = up.unsqueeze(1).repeat(1, 3, 1, 1).unsqueeze(0).to(dev)                          # (1,S,3,518,518)
    sc = np.stack([np.stack([X_NORM, Y_NORM, np.full_like(X_NORM, (z / max(1, D - 1)) * 2 - 1)], -1)
                   for z in z_seq])                                                          # (S,518,518,3)
    z_idx = np.array([[(z / max(1, D - 1)) * 2 - 1] for z in z_seq], np.float32)
    t_idx = np.array([[(t / max(1, T)) * 2 - 1] for t in t_seq], np.float32)
    tt = np.full((S, 1), (t_target / max(1, T)) * 2 - 1, np.float32)
    return {
        "images": imgs,
        "scanner_coords": torch.from_numpy(sc).unsqueeze(0).to(dev),
        "z_indices": torch.from_numpy(z_idx).unsqueeze(0).to(dev),
        "t_indices": torch.from_numpy(t_idx).unsqueeze(0).to(dev),
        "target_t_indices": torch.from_numpy(tt).unsqueeze(0).to(dev),
        "gt_target_volume": torch.from_numpy(gt_vol.astype(np.float32)).unsqueeze(0).to(dev),
        "phases": torch.from_numpy(phases.astype(np.float32)).unsqueeze(0).to(dev),
    }


def mpsnr(V, Vgt, mask):
    if mask.sum() == 0:
        return None
    mse = ((V[mask] - Vgt[mask]) ** 2).mean().clamp(min=1e-10)
    return float(10 * torch.log10(1.0 / mse))


def splat_V(world_points, batch):
    V, _ = splat_predictions({"world_points": world_points}, batch, GRID)
    return V[0]                                                                              # (D,H,W)


def run(model, dev, n, S, seeds):
    ds = H.build_dataset(); print("val subjects:", len(ds.subjects))
    Ks = [0, 1, 2, 4, S]
    recs = []
    seq = 0
    while len(recs) < n and seq < 200:
        i = seq; seq += 1
        data = ds.get_data(seq_index=i, img_per_seq=H.NUM_SLICES)
        phases = np.asarray(data["phases"]); gt = np.asarray(data["gt_target_volume"])
        bbox = [int(v) for v in np.asarray(data["anatomy_bbox"]).tolist()]
        t_target = int(np.asarray(data["t_target"]).flatten()[0]); T = phases.shape[0]
        z0, z1 = bbox[0], bbox[1]; in_bbox = list(range(z0, z1))
        if len(in_bbox) < S:
            continue
        rng = random.Random(1000 + i)
        z_set = rng.sample(in_bbox, S)                                       # FIXED z-set across all conditions
        others = [t for t in range(T) if t != t_target]
        comp_phase = [rng.choice(others) for _ in range(S)]                  # FIXED companion phases
        Vgt = torch.from_numpy(gt.astype(np.float32)).to(dev)
        mmask = compute_motion_mask(torch.from_numpy(phases.astype(np.float32)).unsqueeze(0).to(dev))[0]  # (D,H,W)

        rec = dict(seq=i, subject=os.path.basename(ds.subjects[i % len(ds.subjects)]),
                   t_target=t_target, S=S, n_bbox_z=len(in_bbox), A={}, B={})

        # ── LEVER A: count at target ──
        for K in Ks:
            t_seq = [t_target] * K + comp_phase[K:]
            target_z = set(z_set[:K])
            b = build_batch(phases, gt, bbox, t_target, t_seq, z_set, dev)
            # C2 identity full, C3 model full
            Vid = splat_V(b["scanner_coords"], b)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                pr = model(b["images"], batch=b)
            Vm = splat_V(pr["world_points"].float(), b)
            # C1 identity on ONLY the K target slices (pure observation), K>=1
            if K >= 1:
                bk = build_batch(phases, gt, bbox, t_target, [t_target] * K, z_set[:K], dev)
                Vc1 = splat_V(bk["scanner_coords"], bk)
                c1 = mpsnr(Vc1, Vgt, mmask)
            else:
                c1 = None
            # held-out-z mask: dynamic voxels in planes NOT given a target slice
            zmask = torch.ones(phases.shape[1], dtype=torch.bool, device=dev)
            for z in target_z:
                zmask[z] = False
            ho = mmask & zmask.view(-1, 1, 1)
            rec["A"][str(K)] = dict(
                C1_obs=c1, C2_id=mpsnr(Vid, Vgt, mmask), C3_model=mpsnr(Vm, Vgt, mmask),
                C4_heldout=mpsnr(Vm, Vgt, ho), n_target=K)

        # ── LEVER B: proximity at fixed S, K=0 (no exact target frame), identical z-set ──
        pools = {"random": others,
                 "near2": [(t_target + d) % T for d in (-2, -1, 1, 2)],
                 "near1": [(t_target + d) % T for d in (-1, 1)]}
        for name, pool in pools.items():
            rb = random.Random(7000 + i)
            t_seq = [rb.choice(pool) for _ in range(S)]
            b = build_batch(phases, gt, bbox, t_target, t_seq, z_set, dev)
            Vid = splat_V(b["scanner_coords"], b)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                pr = model(b["images"], batch=b)
            Vm = splat_V(pr["world_points"].float(), b)
            rec["B"][name] = dict(C2_id=mpsnr(Vid, Vgt, mmask), C3_model=mpsnr(Vm, Vgt, mmask),
                                  mean_abs_dt=float(np.mean([min(abs(t - t_target), T - abs(t - t_target)) for t in t_seq])))
        recs.append(rec)
        a = rec["A"]
        print(f"seq{i:2d} {rec['subject']} t{t_target} bbz{len(in_bbox)} | "
              f"A C3: K0 {a['0']['C3_model']:.1f} K1 {a['1']['C3_model']:.1f} K2 {a['2']['C3_model']:.1f} "
              f"K4 {a['4']['C3_model']:.1f} K{S} {a[str(S)]['C3_model']:.1f}  | "
              f"C4ho K1 {a['1']['C4_heldout']:.1f} K4 {a['4']['C4_heldout']:.1f}  | "
              f"B C3: rnd {rec['B']['random']['C3_model']:.1f} n2 {rec['B']['near2']['C3_model']:.1f} "
              f"n1 {rec['B']['near1']['C3_model']:.1f}")
    return recs, Ks


def agg(recs, Ks, S):
    def m(vals): vals = [v for v in vals if v is not None and np.isfinite(v)]; return float(np.mean(vals)) if vals else None
    A = {}
    for K in Ks:
        k = str(K)
        A[k] = {c: m([r["A"][k][c] for r in recs]) for c in ("C1_obs", "C2_id", "C3_model", "C4_heldout")}
    B = {name: {c: m([r["B"][name][c] for r in recs]) for c in ("C2_id", "C3_model", "mean_abs_dt")}
         for name in ("random", "near2", "near1")}
    return dict(n=len(recs), S=S, A=A, B=B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=8); ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} S={args.s} n={args.n} ckpt={os.path.basename(CKPT)}")
    model, info = H.make_model(use_t=False, ckpt_path=CKPT, device=dev)
    print("backbone:", {k: info[k] for k in ("missing", "unexpected", "ckpt_epoch")})
    recs, Ks = run(model, dev, args.n, args.s, None)
    summ = agg(recs, Ks, args.s)
    print("\n=== LEVER A (count at target), mean motion PSNR ===")
    print(f"{'K':>3} {'C1 obs-only':>12} {'C2 id-full':>11} {'C3 model':>9} {'C4 heldout-z':>13}")
    for K in Ks:
        a = summ["A"][str(K)]
        def fmt(x): return f"{x:.2f}" if x is not None else "  -  "
        print(f"{K:>3} {fmt(a['C1_obs']):>12} {fmt(a['C2_id']):>11} {fmt(a['C3_model']):>9} {fmt(a['C4_heldout']):>13}")
    print("\n=== LEVER B (proximity at fixed S, K=0) ===")
    for name in ("random", "near2", "near1"):
        b = summ["B"][name]
        print(f"  {name:7s} <|Δt|>={b['mean_abs_dt']:.2f}  C2_id {b['C2_id']:.2f}  C3_model {b['C3_model']:.2f}")
    json.dump(dict(ckpt=CKPT, summary=summ, records=recs), open(os.path.join(OUT, "toy_contract_levers.json"), "w"), indent=2)
    print("\nsaved", os.path.join(OUT, "toy_contract_levers.json"))


if __name__ == "__main__":
    main()
