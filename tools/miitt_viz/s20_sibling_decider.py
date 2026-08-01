"""DECIDING experiment: is the s20contz non-reference-plane 'frozen' defect caused by
(A) the MULTIFRAME S=20 budget or (B) CONTINUOUS-Z training?

The deciding checkpoint is the s20 sibling (multiframe + SNAPPED z = A but NOT B).
If s20 nonref aliveness ~= 1% (like s20contz) -> MULTIFRAME is the cause.
If s20 nonref aliveness ~= 31% (like gather05) -> CONTINUOUS-Z is the cause.

Uses the EXACT same machinery as the prior table:
  - capture()   from tools/miitt_viz/gated_gather05_7row.py  (per-phase sweep of the reference slot)
  - alive()     temporal-std ratio (recon / GT) per plane, split ref (z_mid) vs non-ref
CMRx in-distribution val subjects (canonical cache via mri_volume val dataset).

Regimes (matched to each model's training budget, per docs / MEMORY):
  gather05  -> '1frame'    (1 frame/plane, snapped z, NO reference cine burst)
  s20       -> 'multiframe'(S~76, full ref cine + 5-frame bursts, snapped z)  <-- the answer
  s20contz  -> 'multiframe'
Cross-check block: feed s20 and s20contz the IDENTICAL 'multiframe' batch so the ONLY
variable is weights (both are multiframe-trained; batch is snapped-z either way).

Run:
  micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/s20_sibling_decider.py
Out: result/s20_decider/summary.json  + stdout table.
"""
import os, sys, glob, gc, json, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "training")
import torch.distributed as dist
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from inference.inference import load_rtfb_model_reference
from inference.run_gated_ood import load_rcfg
from data.respiratory import RespiratoryConfig
import tools.miitt_viz.gated_gather05_7row as G   # capture(), build_1frame_batch, etc.

DEV = "cuda"; OUT = "result/s20_decider"; os.makedirs(OUT, exist_ok=True)
SUBJS = [0, 1, 2, 3, 4]        # 5 CMRx val subjects (seq_index)
BREATHING = False               # clean: purest test of motion propagation to non-ref planes
MAXPH = None                    # all cardiac phases

CKPTS = {
    "gather05": ("216539845_*ftgather05*1frame*", "1frame"),
    "s20":      ("216949759_*s20_dynamic*",       "multiframe"),
    "s20contz": ("216949414_*s20contz*",          "multiframe"),
}


def alive(RE, GT):
    """Per-plane aliveness = mean over in-FOV voxels of the temporal std across the cardiac cycle.
    RE,GT shape (T, D, 256, 256). Returns array (D,)."""
    out = []
    for p in range(GT.shape[1]):
        m = GT[:, p].max(0) > 1e-4
        out.append(float(RE[:, p][:, m].std(0).mean()) if m.sum() else 0.0)
    return np.array(out)


def load_bundles():
    """Fetch canonical phase bundle + geometric bbox for each val subject (mri_volume val ds)."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "29601")
        dist.init_process_group("gloo", rank=0, world_size=1)
    for r, fn in [("rev_ts", lambda: "0"), ("basename", lambda p: os.path.basename(p)),
                  ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}")]:
        try: OmegaConf.register_new_resolver(r, fn)
        except Exception: pass
    with initialize_config_dir(config_dir=os.path.abspath("training/config"), version_base=None):
        cfg = compose(config_name="default")
    val_wrap = instantiate(cfg.data.val, _recursive_=False)
    mri_ds = val_wrap.dataset.base_dataset.datasets[0]
    rcfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)
    bundles = {}
    for s in SUBJS:
        data = mri_ds.get_data(seq_index=s, img_per_seq=mri_ds.num_slices)
        pb = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(DEV)  # (T,D,H,W) in [0,1]
        bbox = np.asarray(data["anatomy_bbox"]).astype(np.int64)
        bundles[s] = (pb, bbox)
        print(f"  loaded subj{s}: T={pb.shape[0]} D={pb.shape[1]} bbox={bbox.tolist()}", flush=True)
    return bundles, rcfg


def eval_model(name, pat, regime, bundles, rcfg, save_probe=False):
    ck = glob.glob(f"scratch/logs/{pat}/ckpts/checkpoint_last.pt")[0]
    print(f"\n### {name}  regime={regime}\n    ckpt={ck}", flush=True)
    model = load_rtfb_model_reference(ck, refiner=False, device=DEV)
    rows = []
    for s in SUBJS:
        pb, bbox = bundles[s]
        cap = G.capture(model, pb, bbox, BREATHING, rcfg, regime=regime, clean_ref=True,
                        seq_index=s, max_phases=MAXPH)
        RE, GT, ref = cap["RE"], cap["GT"], cap["z_mid"]
        re_a, gt_a = alive(RE, GT), alive(GT, GT)
        nonref = [q for q in range(GT.shape[1]) if q != ref and gt_a[q] > 1e-4]
        ref_pct = re_a[ref] / max(gt_a[ref], 1e-6) * 100
        nonref_pct = (np.mean([re_a[q] for q in nonref]) /
                      max(np.mean([gt_a[q] for q in nonref]), 1e-6) * 100)
        full_db = float(np.nanmean(cap["metr"]["full"]))
        motion_db = float(np.nanmean(cap["metr"]["motion"]))
        rows.append(dict(subj=s, ref=int(ref), ref_pct=float(ref_pct), nonref_pct=float(nonref_pct),
                         full_db=full_db, motion_db=motion_db,
                         S=int(cap["rd"].shape[0]), nonref_planes=nonref))
        print(f"    subj{s}: ref z{ref} {ref_pct:5.1f}% | nonref {nonref_pct:5.1f}% | "
              f"full {full_db:5.2f}dB motion {motion_db:5.2f}dB S={cap['rd'].shape[0]}", flush=True)
        # ---- mechanism probe (averaging-to-mean escape) on s20 / s20contz ----
        if save_probe:
            _probe(name, s, cap, RE, GT, ref, nonref)
    del model; gc.collect(); torch.cuda.empty_cache()
    agg = dict(name=name, regime=regime, ckpt=ck,
               ref_pct=float(np.mean([r["ref_pct"] for r in rows])),
               nonref_pct=float(np.mean([r["nonref_pct"] for r in rows])),
               full_db=float(np.mean([r["full_db"] for r in rows])),
               motion_db=float(np.mean([r["motion_db"] for r in rows])), rows=rows)
    print(f"    >>> {name} MEAN: ref {agg['ref_pct']:.1f}% nonref {agg['nonref_pct']:.1f}% "
          f"full {agg['full_db']:.2f}dB", flush=True)
    return agg


_PROBE = []


def _probe(name, s, cap, RE, GT, ref, nonref):
    """Averaging-to-mean escape test. For each non-ref plane p:
      - dz_abs: mean |Δz| (mm) over the input slots that land on plane p (through-plane motion the model applies)
      - corr_static: is RE[:,p] essentially constant over the cardiac cycle? (1 - normalized temporal var proxy)
      - r_gt:  corr(RE[:,p](t), GT[:,p](t)) averaged over voxels -> does recon track the true beat?
      - r_inpmean: corr of RE[:,p].mean_t with the input-frame temporal mean of that plane
    """
    DVs = cap["DV_slots"]           # (T,S,256,256,3) Δ mm per input slot
    INs = cap["IN_slots"]           # (T,S,256,256) fed slice per slot
    zf = np.asarray(cap["slot_zf"]) # (S,) fractional canonical z per slot
    sop = cap["sop"]                # nearest slot per plane
    Tn = RE.shape[0]
    per_plane = []
    for p in nonref:
        # slots assigned to this plane (within 0.5 of integer p)
        sl = [i for i in range(len(zf)) if abs(zf[i] - p) < 0.5]
        if not sl:
            sl = [sop[p]]
        dz_abs = float(np.nanmean(np.abs(DVs[:, sl, :, :, 2])))
        m = GT[:, p].max(0) > 1e-4
        if m.sum() == 0:
            continue
        re_ts = RE[:, p][:, m]      # (T, Nvox)
        gt_ts = GT[:, p][:, m]
        # recon temporal std vs its own mean magnitude -> "how static"
        re_std = re_ts.std(0).mean()
        # per-voxel Pearson corr recon-vs-GT over time, averaged
        def vcorr(a, b):
            a = a - a.mean(0, keepdims=True); b = b - b.mean(0, keepdims=True)
            num = (a * b).sum(0)
            den = np.sqrt((a**2).sum(0) * (b**2).sum(0)) + 1e-8
            return float(np.nanmean(num / den))
        r_gt = vcorr(re_ts, gt_ts)
        # input-frame temporal mean of this plane (the companions the model could average)
        inp_mean = INs[:, sl].mean((0, 1))          # (256,256)
        re_mean = RE[:, p].mean(0)                   # (256,256)
        mm = m
        a = re_mean[mm].ravel() - re_mean[mm].mean(); b = inp_mean[mm].ravel() - inp_mean[mm].mean()
        r_inpmean = float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-8))
        per_plane.append(dict(p=int(p), dz_abs_mm=dz_abs, re_std=float(re_std),
                              r_recon_vs_gt=r_gt, r_reconmean_vs_inputmean=r_inpmean))
    if per_plane:
        agg = dict(name=name, subj=s,
                   dz_abs_mm=float(np.mean([x["dz_abs_mm"] for x in per_plane])),
                   r_recon_vs_gt=float(np.mean([x["r_recon_vs_gt"] for x in per_plane])),
                   r_reconmean_vs_inputmean=float(np.mean([x["r_reconmean_vs_inputmean"] for x in per_plane])),
                   per_plane=per_plane)
        _PROBE.append(agg)
        print(f"       [probe {name} subj{s}] |Δz|={agg['dz_abs_mm']:.2f}mm  "
              f"corr(recon,GT_beat)={agg['r_recon_vs_gt']:+.2f}  "
              f"corr(recon_mean,input_mean)={agg['r_reconmean_vs_inputmean']:+.2f}", flush=True)


def main():
    print("Loading CMRx val bundles ...", flush=True)
    bundles, rcfg = load_bundles()
    results = {}
    # native-regime eval (reproduces the prior table + gives the s20 answer)
    for name, (pat, regime) in CKPTS.items():
        results[name] = eval_model(name, pat, regime, bundles, rcfg,
                                   save_probe=(name in ("s20", "s20contz")))
    # cross-check: s20 vs s20contz on IDENTICAL multiframe batch (already both multiframe -> same as above)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({"config": dict(subjs=SUBJS, breathing=BREATHING),
                   "results": results, "probe": _PROBE}, f, indent=2)
    print("\n========== FINAL TABLE (CMRx in-dist, clean, n=%d) ==========" % len(SUBJS), flush=True)
    print(f"{'model':10s} {'regime':11s} {'ref%':>6s} {'nonref%':>8s} {'full dB':>8s}", flush=True)
    for name in ("gather05", "s20", "s20contz"):
        r = results[name]
        print(f"{name:10s} {r['regime']:11s} {r['ref_pct']:6.1f} {r['nonref_pct']:8.1f} {r['full_db']:8.2f}", flush=True)
    print("\nDONE -> result/s20_decider/summary.json", flush=True)


if __name__ == "__main__":
    main()
