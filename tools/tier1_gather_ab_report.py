"""Tier-1 offline validation of the gather-aux A/B (treatment gw=0.5 vs control gw=0.0).

Reproduces the ONLINE breathing metric offline on frozen checkpoints, on the same 30-subject
val set with the SAME deterministic breathing config the training used (matches mri_volume.yaml:
amplitude 16±8mm, cos2n=3, ap_ratio 0.35, group_by_burst, dir_jitter 30deg). Both models are run
on the IDENTICAL breathed input (breathing applied once per subject, deterministic per seq_index),
so the T-vs-C comparison is perfectly paired.

Outputs a SELF-CONTAINED HTML report (all plots base64-embedded): headline predicted-Δz-vs-applied
scatter, per-SI-bin bars, a summary table (offline vs the epoch-15 online numbers), and MANY
per-subject qualitative panels (V_gt / V_canon_T / V_canon_C / diffs / coverage-holes across z,
plus per-slice Δz vs applied breathing shift).

Run:  micromamba run -n svr python tools/tier1_gather_ab_report.py \
        --treatment <t.pt> --control <c.pt> --seqs 0-29 --out result/tier1_gather_ab
"""
import argparse, base64, io, os, sys, html
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_batch, GRID_SHAPE, DATA_ROOT, SPLIT_FILE
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from data.datasets.mri_dataset import MRIDataset
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume, splat_predictions
from vggt.models.vggt import VGGT
from inference.run_cmrxrecon import _build_multiframe_batch     # deployment-realistic multi-frame sampler
from omegaconf import OmegaConf

D, H, W = GRID_SHAPE
THROUGH_MM = (D - 1) / 2.0 * 12.0            # 66.0 mm per norm z-unit
INPLANE_MM = (256 - 1) / 2.0 * 1.4           # 178.5 mm per norm in-plane unit
ANAT = 0.05
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H",
            per_slot=True, group_by_burst=True, direction_jitter_deg=30.0)
# epoch-15 ONLINE running-avg numbers (gather A/B only; other pairs have no online resp metrics)
ONLINE = {"treatment": dict(slope=0.757, corr=0.903, epe=1.665, deep=0.010, hole=0.034),
          "control":   dict(slope=0.434, corr=0.483, epe=3.250, deep=0.398, hole=0.026)}
# Display config — set from CLI in main(); blue = model A (--treatment), red = model B (--control).
CFG = dict(LT="treatment (gw=0.5)", LC="control (gw=0.0)", show_online=True,
           title="gather-aux A/B (breathing through-plane recovery)",
           question=("does the coverage-free gather-placement auxiliary loss (treatment, gw=0.5) improve the "
                     "model's through-plane (z) correction of respiratory motion vs an identical control (gw=0.0)?"))


def build_dataset_ref(continuous_z=False):
    conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True, "rescale_aug": False,
                             "landscape_check": False, "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(conf, DATA_ROOT, split="val", split_file=SPLIT_FILE, mode="dynamic",
                      mri_mode="axial", num_slices=12, target_size=518, reference_slot=True,
                      continuous_z=continuous_z)


def build_model(device, ckpt):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024, enable_camera=False, enable_depth=False,
             enable_point=True, enable_track=False, use_z_pose_embedding=True,
             use_t_pose_embedding=False, use_target_t_pose_embedding=False, use_reference_token=True,
             train_on_residual_dvf=True, warp_head_type="dpt", bspline_grid_size=32).to(device).eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp, f"missing={miss[:4]} unexpected={unexp[:4]}"
    return m


def prep_subject_batch(ds, seq, args, cfg_on, dev):
    """Build the input batch + breathing for one subject. Returns (batch, Vgt, mm, skip, slice_z):
    `skip[s]` marks reference-plane slots to exclude from the breathing metric.

    - TRAINING regime (default): the MRIDataset S-budget sampler (img_per_seq); skip slot 0 only.
    - INFERENCE regime (--inference): the deployment multi-frame batch (reference plane = ALL T
      cardiac phases; every other in-bbox plane = a `frames_per_slice` burst). This is how the
      model is MEANT to be used (docs/28); S=20 in training is only an OOM cap. Skip = all slots on
      the reference plane (z_mid)."""
    if not args.inference:
        b = build_batch(ds.get_data(seq_index=seq, img_per_seq=args.img_per_seq), dev, seq)
        out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b, grid_shape=GRID_SHAPE, tv_weight=0.0)
        Vgt = out["V_gt"][0].float(); mm = compute_motion_mask(b["phases"])[0]
        b = gpu_augment_batch(b, None, dev, respiratory_cfg=cfg_on, train=False)
        S = b["images"].shape[1]
        skip = np.zeros(S, bool); skip[0] = True                       # slot 0 = reference
        slice_z = b["slice_indices"][0].cpu().numpy().astype(int) if "slice_indices" in b else np.full(S, -1)
        return b, Vgt, mm, skip, slice_z
    # inference regime
    data = ds.get_data(seq_index=seq, img_per_seq=ds.num_slices)
    phases_bundle = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(dev)   # (T,D,H,W)
    bbox = np.asarray(data["anatomy_bbox"]).astype(np.int64)
    b, z_mid = _build_multiframe_batch(phases_bundle, bbox, args.frames_per_slice, seq, dev)
    b["timesteps"][:, 0] = 0                                            # ED target-phase query
    b["gt_target_volume"] = phases_bundle[0].unsqueeze(0)              # V_gt = ED
    b = gpu_augment_batch(b, None, dev, respiratory_cfg=cfg_on, train=False)
    Vgt = phases_bundle[0].float(); mm = compute_motion_mask(b["phases"])[0]
    slice_z = b["slice_indices"][0].cpu().numpy().astype(int)
    skip = (slice_z == z_mid)                                           # all reference-plane slots
    return b, Vgt, mm, skip, slice_z


def fwd(model, b):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(b["images"], batch=b)
    wp = preds["world_points"].float()
    dvf = (wp[0] - b["scanner_coords"][0]).float()                    # (S,H,W,3) norm
    Vm, cov = splat_predictions({"world_points": wp}, b, GRID_SHAPE)
    return Vm[0], dvf, cov[0]


def splat_id(b):
    sc = b["scanner_coords"][0].reshape(1, -1, 3)
    it = b["images"][0].float().mean(1).reshape(1, -1)
    if it.max() > 2: it = it / 255.0
    w = (it > 1e-3).float()
    V, _ = splat_to_volume(sc, it, (D, H, W), weight=w)
    return V[0]


def b64fig(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def agg_per_plane(planes, app3, p3T, p3C):
    """Average the per-slot 3-axis vectors within each z-plane (inference mode has many frames per
    plane; per-plane bars stay readable). Ordered by applied through-plane shift."""
    planes = np.array(planes); app3, p3T, p3C = np.array(app3), np.array(p3T), np.array(p3C)
    uz = sorted(set(planes.tolist()))
    A = np.stack([app3[planes == z].mean(0) for z in uz])
    T = np.stack([p3T[planes == z].mean(0) for z in uz])
    C = np.stack([p3C[planes == z].mean(0) for z in uz])
    return A, T, C


def linfit(x, y):
    if len(x) < 3: return None, None, None
    x, y = np.asarray(x), np.asarray(y)
    slope = float(np.polyfit(x, y, 1)[0])
    xd, yd = x - x.mean(), y - y.mean()
    corr = float((xd * yd).sum() / (np.linalg.norm(xd) * np.linalg.norm(yd) + 1e-9))
    epe = float(np.abs(y - x).mean())
    return slope, corr, epe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--seqs", default="0-29")
    ap.add_argument("--out", default=os.path.join(REPO, "_html"))
    ap.add_argument("--name", default="gather_aux_ab_tier1_validation.html")
    ap.add_argument("--n_panels", type=int, default=12)
    ap.add_argument("--img_per_seq", type=int, default=12, help="training-regime S budget (12 ft, 20 s20)")
    ap.add_argument("--continuous_z", action="store_true", help="jitter non-ref slots off-grid (s20contz regime)")
    ap.add_argument("--inference", action="store_true",
                    help="deployment regime: reference plane = ALL T frames, others = frames_per_slice burst (docs/28)")
    ap.add_argument("--frames_per_slice", type=int, default=5, help="non-reference frames/plane in --inference mode")
    ap.add_argument("--label_t", default=None, help="display label for --treatment (blue)")
    ap.add_argument("--label_c", default=None, help="display label for --control (red)")
    ap.add_argument("--title", default=None); ap.add_argument("--question", default=None)
    ap.add_argument("--no_online", action="store_true", help="omit the online cross-check (no online resp metrics)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lo, hi = args.seqs.split("-"); seqs = list(range(int(lo), int(hi) + 1))
    dev = "cuda"
    if args.label_t: CFG["LT"] = args.label_t
    if args.label_c: CFG["LC"] = args.label_c
    if args.title: CFG["title"] = args.title
    if args.question: CFG["question"] = args.question
    if args.no_online: CFG["show_online"] = False

    ds = build_dataset_ref(continuous_z=args.continuous_z); ds.t_target_fixed = 0   # ED target
    cfg_on = RespiratoryConfig(enable=True, **RESP)
    print("loading models...", flush=True)
    mT = build_model(dev, args.treatment); mC = build_model(dev, args.control)
    print("models loaded", flush=True)

    # aggregate arrays (non-reference slots only)
    AG = {"si": [], "dzT": [], "dzC": []}
    subjects = []   # per-subject records for rendering

    for seq in seqs:
        b, Vgt, mm, skip, slice_z = prep_subject_batch(ds, seq, args, cfg_on, dev)
        if not bool(mm.any()):
            continue
        VmT, dvfT, covT = fwd(mT, b)
        VmC, dvfC, covC = fwd(mC, b)
        Vid = splat_id(b)
        disp = b["resp_disp_mm"][0].cpu().numpy()                    # (S,3) mm
        imgs = b["images"][0].float().mean(1)                        # (S,H,W) [0,1]

        si, dzT, dzC, planes = [], [], [], []
        app3, p3T, p3C = [], [], []      # per-slot 3-axis (x=W, y=H, z=D) applied & predicted, mm
        for s in range(dvfT.shape[0]):
            if skip[s]:                                              # reference-plane slot
                continue
            msk = imgs[s] > ANAT
            if not bool(msk.any()): continue
            si.append(float(disp[s, 0]))
            dzT.append(float(dvfT[s, :, :, 2][msk].mean() * THROUGH_MM))
            dzC.append(float(dvfC[s, :, :, 2][msk].mean() * THROUGH_MM))
            planes.append(int(slice_z[s]))
            # applied breathing (d_D,d_H,d_W) → axis order (x=W, y=H, z=D)
            app3.append([float(disp[s, 2]), float(disp[s, 1]), float(disp[s, 0])])
            p3T.append([float(dvfT[s, :, :, 0][msk].mean() * INPLANE_MM),
                        float(dvfT[s, :, :, 1][msk].mean() * INPLANE_MM),
                        float(dvfT[s, :, :, 2][msk].mean() * THROUGH_MM)])
            p3C.append([float(dvfC[s, :, :, 0][msk].mean() * INPLANE_MM),
                        float(dvfC[s, :, :, 1][msk].mean() * INPLANE_MM),
                        float(dvfC[s, :, :, 2][msk].mean() * THROUGH_MM)])
        if len(si) < 3:
            continue
        AG["si"] += si; AG["dzT"] += dzT; AG["dzC"] += dzC
        # axis bars: in inference mode many slots share a plane → aggregate per plane for readability
        app3, p3T, p3C = agg_per_plane(planes, app3, p3T, p3C) if args.inference \
            else (np.array(app3), np.array(p3T), np.array(p3C))

        slT = linfit(si, dzT); slC = linfit(si, dzC)
        heart = mm
        holeT = float((covT[heart] < 0.5).float().mean())
        holeC = float((covC[heart] < 0.5).float().mean())
        subjects.append(dict(
            seq=seq, si=np.array(si), dzT=np.array(dzT), dzC=np.array(dzC),
            slopeT=slT[0], slopeC=slC[0], epeT=slT[2], epeC=slC[2], holeT=holeT, holeC=holeC,
            si_absmax=float(np.abs(si).max()),
            app3=np.array(app3), p3T=np.array(p3T), p3C=np.array(p3C),
            Vgt=Vgt.cpu().numpy(), VmT=VmT.cpu().numpy(), VmC=VmC.cpu().numpy(),
            Vid=Vid.cpu().numpy(), covT=covT.cpu().numpy(), covC=covC.cpu().numpy(),
            mm=mm.cpu().numpy(), imgs=imgs.cpu().numpy(),
            subj_id=os.path.basename(str(ds.subjects[seq])) if hasattr(ds, "subjects") and seq < len(ds.subjects) else str(seq),
        ))
        print(f"  seq {seq}: SI|max={np.abs(si).max():.1f}mm  slope T={slT[0]:.2f} C={slC[0]:.2f}  "
              f"epe T={slT[2]:.2f} C={slC[2]:.2f}  hole T={holeT:.3f} C={holeC:.3f}", flush=True)

    # ---- aggregate stats ----
    si = np.array(AG["si"]); dzT = np.array(AG["dzT"]); dzC = np.array(AG["dzC"])
    aggT = linfit(si, dzT); aggC = linfit(si, dzC)
    deep = np.abs(si) >= 12
    deepT = float((np.abs(dzT[deep]) < 2).mean()) if deep.any() else None
    deepC = float((np.abs(dzC[deep]) < 2).mean()) if deep.any() else None
    holeT = float(np.mean([s["holeT"] for s in subjects])); holeC = float(np.mean([s["holeC"] for s in subjects]))
    off = {"treatment": dict(slope=aggT[0], corr=aggT[1], epe=aggT[2], deep=deepT, hole=holeT),
           "control":   dict(slope=aggC[0], corr=aggC[1], epe=aggC[2], deep=deepC, hole=holeC)}

    imgs_html = build_report(args, si, dzT, dzC, off, subjects)
    outpath = os.path.join(args.out, args.name)
    open(outpath, "w").write(imgs_html)
    print("\nWrote", outpath)
    print("OFFLINE agg:", off)


def build_report(args, si, dzT, dzC, off, subjects):
    figs = {}
    lim = max(20.0, float(np.abs(si).max()) * 1.05)

    # (1) headline scatter
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.axline((0, 0), slope=1, color="k", ls="--", lw=1, label="ideal (y=x)")
    ax.scatter(si, dzC, s=10, alpha=0.35, color="#d1495b", label=CFG["LC"])
    ax.scatter(si, dzT, s=10, alpha=0.45, color="#0077b6", label=CFG["LT"])
    for arr, c in [(dzC, "#d1495b"), (dzT, "#0077b6")]:
        m = np.polyfit(si, arr, 1); xs = np.array([-lim, lim])
        ax.plot(xs, m[0] * xs + m[1], color=c, lw=2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("applied breathing SI shift (mm)"); ax.set_ylabel("predicted Δz (mm)")
    ax.set_title("Predicted through-plane Δz vs the injected breathing shift\n(each point = one input slice, non-reference)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.2)
    figs["scatter"] = b64fig(fig)

    # (2) per-SI-bin mean predicted Δz
    bins = [(0, 2), (2, 8), (8, 12), (12, 40)]
    a = np.abs(si)
    mT = [np.abs(dzT[(a >= l) & (a < h)]).mean() if ((a >= l) & (a < h)).any() else 0 for l, h in bins]
    mC = [np.abs(dzC[(a >= l) & (a < h)]).mean() if ((a >= l) & (a < h)).any() else 0 for l, h in bins]
    # ideal = TRUE mean applied |SI| within each bin (not the clipped bin-midpoint)
    mIdeal = [a[(a >= l) & (a < h)].mean() if ((a >= l) & (a < h)).any() else 0 for l, h in bins]
    nbin = [int(((a >= l) & (a < h)).sum()) for l, h in bins]
    print("\nBIN TABLE  |Δz| mm (n slots per bin):")
    for i, (l, h) in enumerate(bins):
        print(f"  {l:>2}-{h:<2}mm  n={nbin[i]:<4}  applied={mIdeal[i]:5.1f}  "
              f"{CFG['LT']}={mT[i]:5.1f} ({mT[i]/max(mIdeal[i],1e-6)*100:4.0f}%)  "
              f"{CFG['LC']}={mC[i]:5.1f} ({mC[i]/max(mIdeal[i],1e-6)*100:4.0f}%)")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(bins)); wd = 0.27
    ax.bar(x - wd, mIdeal, wd, color="#adb5bd", label="ideal (=applied)")
    ax.bar(x, mC, wd, color="#d1495b", label=CFG["LC"])
    ax.bar(x + wd, mT, wd, color="#0077b6", label=CFG["LT"])
    ax.set_xticks(x); ax.set_xticklabels([f"{l}-{h}mm" for l, h in bins])
    ax.set_xlabel("applied |SI| bin"); ax.set_ylabel("mean predicted |Δz| (mm)")
    ax.set_title("Breathing correction by depth — the deep-breath tail (12-40mm)")
    ax.legend(); ax.grid(alpha=0.2, axis="y")
    figs["bins"] = b64fig(fig)

    # (3) per-subject slope + epe
    ss = sorted(subjects, key=lambda s: -s["si_absmax"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    idx = np.arange(len(ss)); wd = 0.4
    axs[0].bar(idx - wd/2, [s["slopeC"] for s in ss], wd, color="#d1495b", label=CFG["LC"])
    axs[0].bar(idx + wd/2, [s["slopeT"] for s in ss], wd, color="#0077b6", label=CFG["LT"])
    axs[0].axhline(1.0, color="k", ls="--", lw=1); axs[0].set_title("per-subject slope (→1 ideal)")
    axs[0].set_xlabel("subject (sorted by max breath depth)"); axs[0].legend(); axs[0].grid(alpha=0.2, axis="y")
    axs[1].bar(idx - wd/2, [s["epeC"] for s in ss], wd, color="#d1495b", label=CFG["LC"])
    axs[1].bar(idx + wd/2, [s["epeT"] for s in ss], wd, color="#0077b6", label=CFG["LT"])
    axs[1].set_title("per-subject EPE (mm, ↓ better)"); axs[1].set_xlabel("subject"); axs[1].grid(alpha=0.2, axis="y")
    figs["persubj"] = b64fig(fig)

    # (4) qualitative per-subject panels — deepest breaths first
    panels = []
    for s in ss[:args.n_panels]:
        panels.append((s, render_subject_panel(s), render_subject_dz(s), render_subject_axes(s)))

    return html_page(off, figs, panels)


def render_subject_axes(s):
    """Per-slice signed applied-vs-predicted displacement for each axis (x=W in-plane,
    y=H in-plane/AP, z=D through-plane) + total magnitude. GT = injected breathing shift."""
    app, pT, pC = s["app3"], s["p3T"], s["p3C"]
    order = np.argsort(app[:, 2])                       # order slices by applied through-plane shift
    app, pT, pC = app[order], pT[order], pC[order]
    n = len(app); x = np.arange(n); wd = 0.26
    titles = ["Δx  (in-plane, W)", "Δy  (in-plane, H = AP)", "Δz  (through-plane, D = SI/breathing)", "‖Δ‖  total magnitude"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 3.6))
    for k in range(4):
        if k < 3:
            g, t, c = app[:, k], pT[:, k], pC[:, k]
        else:
            g = np.linalg.norm(app, axis=1); t = np.linalg.norm(pT, axis=1); c = np.linalg.norm(pC, axis=1)
        ax = axs[k]
        ax.bar(x - wd, g, wd, color="#6c757d", label="GT (applied breathing)")
        ax.bar(x, t, wd, color="#0077b6", label=CFG["LT"])
        ax.bar(x + wd, c, wd, color="#d1495b", label=CFG["LC"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(titles[k], fontsize=9); ax.set_xlabel("input slice (sorted by breath depth)", fontsize=8)
        if k == 0: ax.set_ylabel("displacement (mm)", fontsize=9); ax.legend(fontsize=7)
        ax.grid(alpha=0.2, axis="y"); ax.tick_params(labelsize=7)
    fig.suptitle(f"subj {s['subj_id']} — per-slice GT breathing displacement vs predicted Δ  "
                 f"(in-plane also carries cardiac motion; through-plane Δz is the clean breathing channel)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return b64fig(fig)


def render_subject_panel(s):
    mm = s["mm"]; zsum = mm.reshape(mm.shape[0], -1).sum(1)
    zsel = np.argsort(-zsum)[:6]; zsel = sorted(zsel.tolist())      # 6 z-planes with most heart
    Vgt, VmT, VmC, covT, covC = s["Vgt"], s["VmT"], s["VmC"], s["covT"], s["covC"]
    rows = [("V_gt", Vgt, "gray", (0, 1)), (f"V_canon  {CFG['LT']}", VmT, "gray", (0, 1)),
            (f"V_canon  {CFG['LC']}", VmC, "gray", (0, 1)),
            ("|T − gt|", np.abs(VmT - Vgt), "magma", (0, 0.5)),
            ("|C − gt|", np.abs(VmC - Vgt), "magma", (0, 0.5)),
            ("coverage T (<0.5 red)", covT, "cov", None), ("coverage C (<0.5 red)", covC, "cov", None)]
    nr, nc = len(rows), len(zsel)
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 1.7, nr * 1.7))
    for r, (label, vol, cm, vr) in enumerate(rows):
        for c, z in enumerate(zsel):
            ax = axs[r, c]; img = vol[z]
            if cm == "cov":
                ax.imshow(np.clip(img, 0, 6), cmap="viridis")
                hole = img < 0.5
                ov = np.zeros((*img.shape, 4)); ov[hole] = [1, 0, 0, 0.55]; ax.imshow(ov)
            else:
                ax.imshow(img, cmap=cm, vmin=vr[0], vmax=vr[1])
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0: ax.set_ylabel(label, fontsize=8)
            if r == 0: ax.set_title(f"z={z}", fontsize=8)
    fig.suptitle(f"subject {s['subj_id']}  |  max breath {s['si_absmax']:.0f}mm  |  "
                 f"slope T={s['slopeT']:.2f}/C={s['slopeC']:.2f}  EPE T={s['epeT']:.1f}/C={s['epeC']:.1f}mm  "
                 f"hole T={s['holeT']:.3f}/C={s['holeC']:.3f}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return b64fig(fig)


def render_subject_dz(s):
    order = np.argsort(s["si"]); x = s["si"][order]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(x, x, "k--", lw=1, label="ideal")
    ax.plot(x, s["dzC"][order], "o-", ms=4, color="#d1495b", label=CFG["LC"])
    ax.plot(x, s["dzT"][order], "o-", ms=4, color="#0077b6", label=CFG["LT"])
    ax.set_xlabel("applied SI shift per slice (mm)"); ax.set_ylabel("predicted Δz (mm)")
    ax.set_title(f"per-slice breathing correction — subj {s['subj_id']}")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    return b64fig(fig)


def html_page(off, figs, panels):
    LT, LC, ONL = CFG["LT"], CFG["LC"], CFG["show_online"]
    def name_key(n): return {"slope (→1)": "slope", "corr (→1)": "corr", "EPE mm (↓)": "epe",
                             "deep-ignored (↓)": "deep", "hole_frac_heart (↓)": "hole"}[n]
    def row(name):
        o, c = off["treatment"], off["control"]; k = name_key(name)
        cells = f"<td class=t>{o[k]:.3f}</td><td class=c>{c[k]:.3f}</td>"
        if ONL:
            cells += f"<td>{ONLINE['treatment'][k]:.3f}</td><td>{ONLINE['control'][k]:.3f}</td>"
        return f"<tr><td>{name}</td>{cells}</tr>"
    rows = "".join(row(n) for n in ["slope (→1)", "corr (→1)", "EPE mm (↓)", "deep-ignored (↓)", "hole_frac_heart (↓)"])
    online_hdr = ("<th>online ep15<br>(blue)</th><th>online ep15<br>(red)</th>") if ONL else ""
    online_note = ("<p>If the offline numbers match the epoch-15 online running-averages, the online metric is real, "
                   "not a logging artifact.</p>") if ONL else \
                  ("<p>No online cross-check for this pair — these runs predate the online breathing metrics, so the "
                   "table is offline-only.</p>")
    panel_html = ""
    for s, panel, dz, axes in panels:
        panel_html += (f"<div class=panel><h3>subject {html.escape(str(s['subj_id']))} "
                       f"— max breath {s['si_absmax']:.0f} mm</h3>"
                       f"<img src='data:image/png;base64,{axes}'><br>"
                       f"<img src='data:image/png;base64,{dz}'><br>"
                       f"<img src='data:image/png;base64,{panel}'></div>")
    return f"""<!doctype html><meta charset=utf-8><title>Tier-1 gather-aux A/B validation</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
h1{{border-bottom:3px solid #0077b6;padding-bottom:.3rem}}
h2{{margin-top:2.2rem;color:#0077b6}}
table{{border-collapse:collapse;margin:1rem 0;font-size:.95rem}}
td,th{{border:1px solid #ccc;padding:.4rem .7rem;text-align:center}}
th{{background:#f0f4f8}} td:first-child{{text-align:left;font-weight:600}}
.t{{background:#e3f0fa;font-weight:700}} .c{{background:#fbe6ea}}
img{{max-width:100%;border:1px solid #eee;border-radius:4px}}
.panel{{margin:1.5rem 0;padding:1rem;border:1px solid #e5e5e5;border-radius:8px;background:#fafafa}}
.note{{background:#fffbe6;border-left:4px solid #f0c000;padding:.6rem 1rem;margin:1rem 0;font-size:.92rem}}
.key{{background:#eef7ee;border-left:4px solid #2a9d2a;padding:.6rem 1rem;margin:1rem 0}}
</style>
<h1>Tier-1 offline validation — {html.escape(CFG["title"])}</h1>
<p><b>Question:</b> {html.escape(CFG["question"])} Both models run on the <b>identical</b> breathed input per
subject (deterministic per seq_index) → paired comparison. <b>Blue = {html.escape(LT)}</b>;
<b>red = {html.escape(LC)}</b>.</p>

<div class=note><b>Caveats.</b> (1) These are mid-training checkpoints, not converged. (2) The "ground
truth" breathing shift is our own simulator — this validates recovery of <i>simulated</i> motion, a fair proxy but
not real free-breathing (that's Tier-3). (3) Single training run per arm; the paired-val determinism mitigates but
doesn't eliminate seed noise. (4) In-plane Δx/Δy also carry cardiac motion (see §5); Δz is the clean breathing axis.</div>

<h2>1. Aggregate{" — offline vs online cross-check" if ONL else ""}</h2>
{online_note}
<table><tr><th>metric</th><th>{html.escape(LT)}<br>(offline)</th><th>{html.escape(LC)}<br>(offline)</th>
{online_hdr}</tr>{rows}</table>

<h2>2. Headline — predicted Δz vs the injected breathing shift</h2>
<p>The pure geometry test: a perfect breathing corrector sits on y=x. The bluer model tracks the line more
closely; a flatter fit means under-correction, worst at large shifts.</p>
<img src='data:image/png;base64,{figs["scatter"]}'>

<h2>3. The deep-breath tail</h2>
<img src='data:image/png;base64,{figs["bins"]}'>

<h2>4. Per-subject</h2>
<img src='data:image/png;base64,{figs["persubj"]}'>

<h2>5. Qualitative panels ({len(panels)} subjects, deepest breaths first)</h2>
<p>Per subject, three views. <b>(a) Per-slice displacement bars</b> — for each input slice, the GT
injected breathing shift (grey) vs <b>{html.escape(LT)}</b> (blue) vs <b>{html.escape(LC)}</b> (red), broken out by axis: Δx (in-plane
W), Δy (in-plane H = AP), Δz (through-plane = SI/breathing), and total ‖Δ‖. <span class=note style="display:inline;padding:.1rem .4rem">
<b>Read the in-plane axes with care:</b> the predicted in-plane Δx/Δy legitimately carries <i>cardiac</i>
in-plane contraction on top of the breathing-AP correction, so it won't match the (breathing-only) GT as
tightly as Δz does — Δz is the clean breathing channel (little cardiac longitudinal motion), which is why
it's the headline metric.</span> Then <b>(b)</b> the per-slice Δz-vs-applied curve, and <b>(c)</b> a grid
over the 6 most-cardiac z-planes —
rows: <b>V_gt</b>, <b>V_canon {html.escape(LT)}</b>, <b>V_canon {html.escape(LC)}</b>, <b>|blue−gt|</b>, <b>|red−gt|</b>,
<b>coverage (blue)</b>, <b>coverage (red)</b> (holes, coverage&lt;0.5, in red — the guardrail against over-moving slices).</p>
{panel_html}
"""


if __name__ == "__main__":
    main()
