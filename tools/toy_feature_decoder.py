"""DECISIVE real-data test: does a feature-splat + 3D decoder (appearance SYNTHESIS) beat transport
and the intensity-decoder control on HELD-OUT subjects — and is it real or hallucination?

Pipeline: frozen backbone → tap DPT per-pixel features (hook on point_head.scratch.output_conv1) →
splat K features at the predicted world_points into a (K,12,256,256) feature volume → small 3D
decoder → V. Train on TRAIN subjects, eval HELD-OUT val. Guards (per red-team):
  - CONTROL: identical decoder fed [V_canon, coverage] (intensity, 2ch) = the refiner-equivalent.
    Feature-decoder must beat THIS, not just transport, to show features add synthesis.
  - SCRAMBLE ablation: at eval, zero the feature volume (keep coverage) → if motion PSNR barely
    drops, the gain was target_t leakage/memorization, not synthesis from anatomy.
  - baselines: transport (V_canon), population template (~14.4), identity floor.
Caches feature/intensity volumes to /tmp so the decoder trains many epochs (adequate convergence,
avoids a fake null). Optimizer LR vetted (lr=1e-3, logged loss must decrease).

Run: micromamba run -n svr python tools/toy_feature_decoder.py --n-train 40 --draws 3 --steps 2500
"""
import argparse, os, sys, json, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_batch, GRID_SHAPE, NUM_SLICES, DATA_ROOT, SPLIT_FILE, make_model, LOGS
from data.datasets.mri_dataset import MRIDataset
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume
from omegaconf import OmegaConf
D, H, W = GRID_SHAPE
K = 16                      # compressed feature channels
OUT = os.path.join(REPO, "result", "limits_eval")
CACHE = "/tmp/featdec_cache"
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H", per_slot=True, direction_jitter_deg=30.0)
OLD_CKPT = os.path.join(LOGS, "218747856_mri_volume_resp_allphases_aggft_z_no_t", "ckpts", "checkpoint_last.pt")


def psnr(a, b, m): a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean()); return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def make_ds(split):
    conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True, "rescale_aug": False,
                             "landscape_check": False, "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(conf, DATA_ROOT, split=split, split_file=SPLIT_FILE, mode="dynamic",
                      mri_mode="axial", num_slices=NUM_SLICES, target_size=518)


def build_cache(n_train, draws, seed0):
    os.makedirs(CACHE, exist_ok=True)
    if glob.glob(os.path.join(CACHE, "train_*.pt")):
        print("cache exists, skipping rebuild"); return
    dev = "cuda"
    model, info = make_model(False, OLD_CKPT, dev); print("backbone:", info)
    feats = {"chunks": []}
    # DPT processes frames in chunks → the hook fires once per chunk; accumulate ALL chunks.
    h = model.point_head.scratch.output_conv1.register_forward_hook(lambda m, i, o: feats["chunks"].append(o))
    rng = np.random.default_rng(seed0)

    def one(ds, seq, proj):
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        b = build_batch(data, dev, seq_index=seq)
        b = gpu_augment_batch(b, None, dev, respiratory_cfg=RespiratoryConfig(enable=True, **RESP), train=False)
        out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b, grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float(); mmask = compute_motion_mask(b["phases"])[0]
        feats["chunks"] = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(b["images"], batch=b)
        wp = preds["world_points"][0].float()                       # (S,518,518,3)
        fmap = torch.cat(feats["chunks"], dim=0).float()            # (S,Cf,hf,wf) — all frame chunks
        S = wp.shape[0]
        # downsample world_points + features to 256
        wp256 = F.interpolate(wp.permute(0, 3, 1, 2), size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        f256 = F.interpolate(fmap, size=(256, 256), mode="bilinear", align_corners=False)   # (S,Cf,256,256)
        fk = proj(f256)                                              # (S,K,256,256) compressed
        inten = b["images"][0].float().mean(dim=1)                  # (S,256? no 518)
        inten256 = F.interpolate(b["images"][0].float().mean(dim=1, keepdim=True), size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        # splat: intensity (transport) + each feature channel, at the SAME world_points
        pos = wp256.reshape(1, -1, 3); wgt = (inten256.reshape(1, -1) > 1e-3).float()
        Vc, cov = splat_to_volume(pos, inten256.reshape(1, -1), (D, 256, 256), weight=wgt)
        fvol = torch.zeros((K, D, 256, 256), device=dev)
        for k in range(K):
            fv, _ = splat_to_volume(pos, fk[:, k].reshape(1, -1), (D, 256, 256), weight=wgt)
            fvol[k] = fv[0]
        return dict(fvol=fvol.half().cpu(), Vc=Vc[0].half().cpu(), cov=cov[0].half().cpu(),
                    V_gt=V_gt.half().cpu(), mmask=mmask.cpu())

    # fixed random projection of features -> K (shared, deterministic) so train/val use same map
    Cf = None
    proj_w = None
    def proj(f):
        nonlocal Cf, proj_w
        if proj_w is None:
            Cf = f.shape[1]
            g = torch.Generator(device=f.device).manual_seed(0)
            proj_w = torch.randn(K, Cf, device=f.device, generator=g) / np.sqrt(Cf)
        return torch.einsum("kc,schw->skhw", proj_w, f)
    tr = make_ds("train"); va = make_ds("val")
    for i in range(n_train):
        for d in range(draws):
            seq = i + 30 * d   # vary draw (different RNG) same-ish subject pool
            torch.save(one(tr, seq, proj), os.path.join(CACHE, f"train_{i}_{d}.pt"))
        if i % 10 == 0: print("cached train", i)
    for vs in range(len(va.subjects)):
        torch.save(one(va, vs, proj), os.path.join(CACHE, f"val_{vs}.pt"))
    h.remove()
    print("cache built:", len(glob.glob(os.path.join(CACHE, '*.pt'))), "samples")


class Dec(nn.Module):
    def __init__(self, inch):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(min(8, o), o), nn.GELU())
        self.e1 = cb(inch, 32); self.e2 = cb(32, 64); self.pool = nn.MaxPool3d((1, 2, 2))
        self.b = cb(64, 64)
        self.d2 = cb(64 + 64, 64); self.d1 = cb(64 + 32, 32)
        self.out = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        s1 = self.e1(x); s2 = self.e2(self.pool(s1)); b = self.b(self.pool(s2))
        u2 = F.interpolate(b, size=s2.shape[2:], mode="trilinear", align_corners=False)
        u2 = self.d2(torch.cat([u2, s2], 1))
        u1 = F.interpolate(u2, size=s1.shape[2:], mode="trilinear", align_corners=False)
        u1 = self.d1(torch.cat([u1, s1], 1))
        return self.out(u1)[:, 0]


def load(split):
    files = sorted(glob.glob(os.path.join(CACHE, f"{split}_*.pt")))
    return [torch.load(f, map_location="cpu") for f in files]


def train_eval(mode, train, val, steps, lr, scramble=False):
    dev = "cuda"
    inch = {"feature": K + 1, "intensity": 2, "both": K + 2}[mode]
    dec = Dec(inch).to(dev); opt = torch.optim.Adam(dec.parameters(), lr=lr)
    rng = np.random.default_rng(0)

    def inp(s):
        cov = s["cov"].float().to(dev)[None, None]
        vc = s["Vc"].float().to(dev)[None, None]
        fv = s["fvol"].float().to(dev)[None]
        if mode == "feature":
            return torch.cat([fv, cov], 1)
        if mode == "both":
            return torch.cat([vc, fv, cov], 1)
        return torch.cat([vc, cov], 1)
    losses = []
    for it in range(steps):
        s = train[rng.integers(len(train))]
        x = inp(s); vg = s["V_gt"].float().to(dev)[None]
        pred = dec(x); loss = (pred - vg).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 400 == 0: losses.append(round(float(loss), 4))
    dec.eval(); mps = []
    with torch.no_grad():
        for s in val:
            x = inp(s)
            if scramble: x[:, :inch - 1] = 0.0    # zero features/intensity, keep coverage
            pred = dec(x)[0].cpu().numpy()
            mm = s["mmask"].numpy();
            if mm.any(): mps.append(psnr(pred, s["V_gt"].float().numpy(), mm))
    return float(np.mean(mps)), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40); ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    build_cache(args.n_train, args.draws, 0)
    train, val = load("train"), load("val")
    print(f"train {len(train)} val {len(val)} samples")
    # transport baseline on val
    transp = float(np.mean([psnr(s["Vc"].float().numpy(), s["V_gt"].float().numpy(), s["mmask"].numpy())
                            for s in val if s["mmask"].any()]))
    feat_psnr, fl = train_eval("feature", train, val, args.steps, args.lr)
    int_psnr, il = train_eval("intensity", train, val, args.steps, args.lr)
    feat_scr, _ = train_eval("feature", train, val, args.steps, args.lr, scramble=True)
    res = dict(transport_Vcanon=transp, intensity_decoder=int_psnr, feature_decoder=feat_psnr,
               feature_decoder_SCRAMBLED=feat_scr, population_template=14.4, warp_ceiling=21.0,
               recoverable_ceiling_dense=28.1, oracle=35.0,
               feat_loss_curve=fl, int_loss_curve=il, n_train=len(train), n_val=len(val))
    json.dump(res, open(os.path.join(OUT, "toy_feature_decoder.json"), "w"), indent=2)
    print("\n=== FEATURE-SPLAT DECODER (held-out val, motion PSNR) ===")
    print(f"  transport (V_canon)            {transp:.2f}")
    print(f"  intensity-decoder (refiner-eq) {int_psnr:.2f}   (control: decoder on V_canon+cov)")
    print(f"  FEATURE-decoder                {feat_psnr:.2f}   <- the synthesis test")
    print(f"  feature-decoder SCRAMBLED      {feat_scr:.2f}   (features zeroed; ~transport ⇒ no leakage)")
    print(f"  context: warp ceiling 21.0 | recoverable ceiling (dense) 28.1 | oracle 35.0")
    print(f"  feature-decoder vs transport {feat_psnr-transp:+.2f} ; vs intensity-decoder {feat_psnr-int_psnr:+.2f}")
    print(f"  loss curves feat {fl}  int {il}")
    print("Wrote", os.path.join(OUT, "toy_feature_decoder.json"))


if __name__ == "__main__":
    main()
