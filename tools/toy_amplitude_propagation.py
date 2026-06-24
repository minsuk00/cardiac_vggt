"""Is the model's MOTION error a PROPAGATABLE GLOBAL AMPLITUDE (1 reference plane fixes the volume)
or PER-PLANE appearance (a reference helps only its own plane)? — no-retrain, leak-controlled.

Reconciles: doc 21 (adding target frames doesn't raise held-out motion PSNR for the index model) vs
the flat-EF finding (model regresses contraction AMPLITUDE to the cohort mean; a target-phase
reference recovers per-patient amplitude). The user's idea = "give it a slice with the LV state I
want." Make-or-break for MOTION PSNR: can ONE reference plane's amplitude propagate to held-out planes?

DESIGN (red-team-hardened — avoids the GT-ED static-swap confound):
 - Run the model TWICE on the SAME blind input (val rng is seq-seeded ⇒ identical input slices):
   target=t → V_pred_t ; target=ED(0) → V_pred_0.  M_pred = V_pred_t − V_pred_0  (model's PREDICTED motion).
   M_gt = phases[t] − phases[0]  (TRUE motion, GT ED anchor).
 - Score the MOTION RESIDUAL directly: residual-PSNR(α·M_pred, M_gt) on dynamic voxels — NO volume is
   reconstituted, so there is no static-background leak. Baseline = α=1 (the model's own motion).
 - PREMISE CHECK (run first): is a single global α even valid? Per-plane cosine(M_pred[z],M_gt[z]) and
   per-plane LS α_oracle[z]; CV(α_oracle[z]) across planes. Small CV + high cosine ⇒ amplitude is
   global ⇒ one reference can carry it. Large ⇒ regional ⇒ a scalar/one-plane cannot.
 - α from a reference plane z_ref (LS fit on z_ref dynamic voxels, clamp [0.2,6]), applied to HELD-OUT
   planes (z≠z_ref). z_ref ∈ {max-motion (best), random (realistic), worst}. Controls: global-α oracle
   (fit on ALL dyn = global ceiling), per-plane-α oracle (regional ceiling, exposes what a scalar can't).
 - Stratify t into early-contraction {2,4} vs mixed/relaxing {6,8,10} (target_t is normalized time).

Caveat (stated in report): this proves the INFORMATION/identifiability question (is the held-out error
a globally-propagatable amplitude that 1 plane suffices to supply). It does NOT prove a retrained model
would LEARN to read+propagate it — a positive result LICENSES a retrain; given doc 21 (this model fails
to propagate), the bottleneck is propagation, so the post-hoc α-injection bypasses the broken mechanism.

Run: micromamba run -n svr python tools/toy_amplitude_propagation.py --n 20
"""
import argparse, json, os, sys
import numpy as np, torch
from omegaconf import OmegaConf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from data.datasets.mri_dataset import MRIDataset                       # noqa: E402
from vggt.models.vggt import VGGT                                      # noqa: E402
from loss import compute_volume_intensity_loss, compute_motion_mask    # noqa: E402

CKPT = os.path.join(REPO, "scratch/logs/218643188_mri_volume_noresp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt")
DATA_ROOT = os.path.join(REPO, "scratch/data/CMRxRecon2024/Cine_combined")
SPLIT = os.path.join(REPO, "training/splits/random_8_1_1.txt")
OUT = os.path.join(REPO, "result", "limits_eval"); T = 12; GRID = (T, 256, 256)
MINVOX = 20


def build_batch(data, dev):
    def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    b = {"images": imgs, "scanner_coords": st("scanner_coords"), "z_indices": st("z_indices"),
         "t_indices": st("t_indices"), "target_t_indices": st("target_t_indices"),
         "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0)}
    return {k: v.to(dev) for k, v in b.items()}


def predict(model, ds, si, t, dev):
    ds.t_target_fixed = t
    data = ds.get_data(seq_index=si, img_per_seq=12)            # val rng seq-seeded ⇒ same input for any t
    ds.t_target_fixed = None
    b = build_batch(data, dev)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        pr = model(b["images"], batch=b)
    out = compute_volume_intensity_loss({"world_points": pr["world_points"].float()}, b, grid_shape=GRID, tv_weight=0.0)
    return out["V_canon"][0].float(), np.asarray(data["phases"], np.float32)


def alpha_fit(Mp, Mg, mask):
    num = (Mp[mask] * Mg[mask]).sum(); den = (Mp[mask] ** 2).sum().clamp(min=1e-8)
    return float(num / den)


def rpsnr(Mp, Mg, mask, a):
    if mask.sum() == 0: return None
    mse = ((a * Mp[mask] - Mg[mask]) ** 2).mean().clamp(min=1e-10)
    return float(10 * torch.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--phases", default="2,4,6,8,10"); args = ap.parse_args(); dev = "cuda"
    cc = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True,
                           "rescale_aug": False, "landscape_check": False, "augs": {"scales": [1.0, 1.0]}})
    ds = MRIDataset(cc, DATA_ROOT, split="val", split_file=SPLIT, mode="dynamic",
                    mri_mode="axial", num_slices=12, target_size=518)
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024, enable_camera=False, enable_depth=False,
                 enable_point=True, enable_track=False, use_z_pose_embedding=True,
                 use_t_pose_embedding=False, use_target_t_pose_embedding=True, train_on_residual_dvf=True).to(dev)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    miss, unexp = model.load_state_dict(ck["model"], strict=False); assert not miss and not unexp
    model.eval(); print("loaded clean (noresp aggft).")
    phases = [int(p) for p in args.phases.split(",")]
    nsub = min(args.n, len(ds.subjects)); recs = []
    for si in range(nsub):
        Vp0, ph = predict(model, ds, si, 0, dev)             # model ED prediction (shared)
        phs = torch.from_numpy(ph).to(dev); Ved = phs[0]
        dyn_all = compute_motion_mask(phs.unsqueeze(0))[0]   # (D,H,W) bool
        for t in phases:
            Vpt, _ = predict(model, ds, si, t, dev)
            Mp = Vpt - Vp0; Mg = phs[t] - Ved
            dyn = dyn_all
            valid = [z for z in range(dyn.shape[0]) if int(dyn[z].sum()) >= MINVOX]
            if len(valid) < 2:
                continue
            # premise: per-plane cosine + per-plane alpha
            cos_z, a_z = [], {}
            for z in valid:
                mz = dyn[z]; mp, mg = Mp[z][mz], Mg[z][mz]
                c = float((mp * mg).sum() / (mp.norm() * mg.norm()).clamp(min=1e-8)); cos_z.append(c)
                a_z[z] = float((mp * mg).sum() / (mp ** 2).sum().clamp(min=1e-8))
            a_vals = np.array([a_z[z] for z in valid])
            cv = float(np.std(a_vals) / (abs(np.mean(a_vals)) + 1e-8))
            a_global = alpha_fit(Mp, Mg, dyn)
            # z_ref policies
            cnt = {z: int(dyn[z].sum()) for z in valid}
            z_max = max(valid, key=lambda z: cnt[z]); z_min = min(valid, key=lambda z: cnt[z])
            rng = np.random.default_rng(1000 + si * 13 + t); z_rand = int(rng.choice(valid))
            def heldout(z_ref):
                ho = dyn.clone(); ho[z_ref] = False
                a_r = float(np.clip(a_z[z_ref], 0.2, 6.0)); clamped = a_r != a_z[z_ref]
                # per-plane oracle on held-out: scale each plane by its own alpha
                Mp_pp = Mp.clone()
                for z in valid:
                    if z != z_ref: Mp_pp[z] = Mp[z] * float(np.clip(a_z[z], 0.2, 6.0))
                base = rpsnr(Mp, Mg, ho, 1.0)
                return dict(z_ref=z_ref, a_ref=a_r, clamped=clamped, base=base,
                            ref=rpsnr(Mp, Mg, ho, a_r), glob=rpsnr(Mp, Mg, ho, a_global),
                            ppl=rpsnr(Mp_pp, Mg, ho, 1.0),
                            resid=float(((a_global * Mp - Mg)[ho] ** 2).sum() / ((Mp - Mg)[ho] ** 2).sum().clamp(min=1e-10)))
            recs.append(dict(si=si, t=t, t_grp=("early" if t <= 4 else "late"),
                             cos_mean=float(np.mean(cos_z)), cv=cv, a_global=a_global,
                             n_valid=len(valid), zmax=heldout(z_max), zrand=heldout(z_rand), zmin=heldout(z_min)))
        print(f"  subj {si+1}/{nsub}")

    def gm(sel):
        v = [x for x in sel if x is not None and np.isfinite(x)]; return float(np.mean(v)) if v else None
    pol = lambda key, f: gm([r[key][f] for r in recs])
    summ = dict(n=len(recs), nsub=nsub,
                cos_mean=gm([r["cos_mean"] for r in recs]), cv_mean=gm([r["cv"] for r in recs]),
                a_global_mean=gm([r["a_global"] for r in recs]),
                corr_aref_aglob=float(np.corrcoef([r["zrand"]["a_ref"] for r in recs], [r["a_global"] for r in recs])[0, 1]),
                clamp_frac=float(np.mean([r["zrand"]["clamped"] for r in recs])))
    for pname in ("zmax", "zrand", "zmin"):
        summ[pname] = {k: pol(pname, k) for k in ("base", "ref", "glob", "ppl", "resid")}
    for grp in ("early", "late"):
        g = [r for r in recs if r["t_grp"] == grp]
        if g:
            summ[f"{grp}_base"] = gm([r["zrand"]["base"] for r in g]); summ[f"{grp}_ref"] = gm([r["zrand"]["ref"] for r in g])
            summ[f"{grp}_glob"] = gm([r["zrand"]["glob"] for r in g]); summ[f"{grp}_aglob"] = gm([r["a_global"] for r in g])
    json.dump(dict(summary=summ, records=recs), open(os.path.join(OUT, "toy_amplitude_propagation.json"), "w"), indent=2)
    print("\n=== PREMISE: is the motion error a GLOBAL amplitude scalar? ===")
    print(f"  per-plane cosine(M_pred,M_gt) mean = {summ['cos_mean']:.3f}  (high ⇒ same spatial pattern)")
    print(f"  CV of per-plane alpha across z      = {summ['cv_mean']:.3f}  (low ⇒ amplitude is spatially uniform/global)")
    print(f"  global alpha (M_pred→M_gt)          = {summ['a_global_mean']:.2f}  (>1 ⇒ model UNDER-predicts motion)")
    print("\n=== held-out motion-residual PSNR (peak=1); baseline=alpha=1 (the model) ===")
    print(f"{'z_ref policy':12s} {'base':>7s} {'+1-plane α':>11s} {'+global α(ceil)':>15s} {'+per-plane α':>13s} {'resid':>7s}")
    for pname in ("zmax", "zrand", "zmin"):
        s = summ[pname]
        print(f"{pname:12s} {s['base']:>7.2f} {s['ref']:>11.2f} {s['glob']:>15.2f} {s['ppl']:>13.2f} {s['resid']:>7.2f}")
    print(f"  corr(1-plane α[random], global α) = {summ['corr_aref_aglob']:.2f}   clamp-hit frac(random) = {summ['clamp_frac']:.2f}")
    print("\n=== by phase group (random z_ref) ===")
    for grp in ("early", "late"):
        if f"{grp}_base" in summ:
            print(f"  {grp:5s}: base {summ[f'{grp}_base']:.2f}  +1-plane α {summ[f'{grp}_ref']:.2f}  +global α {summ[f'{grp}_glob']:.2f}  (α_global {summ[f'{grp}_aglob']:.2f})")
    print("saved", os.path.join(OUT, "toy_amplitude_propagation.json"))


if __name__ == "__main__":
    main()
