"""Qualitative figures + p95 DVF for the 4wok analysis HTML report.

Produces PNGs in result/analysis_4wok/figs/:
  recon_{subj}_ED.png / _ES.png : input example | V_gt mid-z | V_canon mid-z | |error|  (shows recon
                                  quality + under-contraction)
  dvf_{subj}_ES.png             : predicted Δx, Δy, Δz (mm) over a slot (shows in-plane vs ~0 through)
  breath_{subj}.png             : a deep-breath slot: clean input | breathed input | model V_canon plane
  cardiac_cycle_{subj}.png      : V_gt vs V_canon across the 12 phases at mid-z (beating heart)
Also prints p95/p99 DVF over anatomy (resp OFF) so we don't understate localized cardiac motion.

Run: micromamba run -n svr python tools/render_4wok_qualitative.py --subjects 0,3,7,12
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from exp_4wok_analysis import (build_dataset_ref, build_model, build_batch, fwd, splat_id, dvf_mm,
                               THROUGH_MM, INPLANE_MM, ANAT, CKPT, GRID_SHAPE)
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
D, H, W = GRID_SHAPE
OUT = os.path.join(REPO, "result", "analysis_4wok", "figs")
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H",
            per_slot=True, group_by_burst=True, direction_jitter_deg=30.0)
ZMID = 6


def panel(imgs, titles, path, cmaps=None, vranges=None, sup=None):
    n = len(imgs); fig, ax = plt.subplots(1, n, figsize=(3 * n, 3.2), dpi=110)
    if n == 1: ax = [ax]
    for i, (im, t) in enumerate(zip(imgs, titles)):
        cm = (cmaps or ["gray"] * n)[i]; vr = (vranges or [(None, None)] * n)[i]
        h = ax[i].imshow(im, cmap=cm, vmin=vr[0], vmax=vr[1]); ax[i].set_title(t, fontsize=9); ax[i].axis("off")
        if cm != "gray": fig.colorbar(h, ax=ax[i], fraction=0.046)
    if sup: fig.suptitle(sup, fontsize=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", default="0,3,7,12")
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    subs = [int(s) for s in args.subjects.split(",")]
    dev = "cuda"; ds = build_dataset_ref(); model = build_model(dev, args.ckpt)
    cfg_off = RespiratoryConfig(enable=False, **RESP); cfg_on = RespiratoryConfig(enable=True, **RESP)
    p95_ip, p99_ip, p95_dz, p99_dz = [], [], [], []

    for sub in subs:
        # ED / ES reconstruction + DVF
        for t, tag in [(0, "ED"), (6, "ES")]:
            ds.t_target_fixed = t
            data = ds.get_data(seq_index=sub, img_per_seq=12)
            b = build_batch(data, dev, sub)
            out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b, grid_shape=GRID_SHAPE, tv_weight=0.0)
            Vgt = out["V_gt"][0].float().cpu().numpy()
            b = gpu_augment_batch(b, None, dev, respiratory_cfg=cfg_off, train=False)
            Vm, dvf = fwd(model, b)
            Vm = Vm.cpu().numpy(); Vid = splat_id(b).cpu().numpy()
            imgs = b["images"][0].float().mean(1)                                 # (S,H,W)
            inp = imgs[1].cpu().numpy()                                            # a scattered input slice
            err = np.abs(Vm[ZMID] - Vgt[ZMID])
            panel([inp, Vgt[ZMID], Vid[ZMID], Vm[ZMID], err],
                  ["input slice (scattered)", f"GT @ {tag}", "raw splat (do-nothing)", "model V_canon", "|model-GT|"],
                  os.path.join(OUT, f"recon_s{sub}_{tag}.png"),
                  cmaps=["gray", "gray", "gray", "gray", "magma"],
                  vranges=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 0.4)],
                  sup=f"subj{sub} target={tag}  (model beats raw splat, but under-contracts vs GT)")
            # DVF maps at ES over a mid slot
            if tag == "ES":
                s = 3
                dx = dvf[s, :, :, 0].cpu().numpy() * INPLANE_MM
                dy = dvf[s, :, :, 1].cpu().numpy() * INPLANE_MM
                dz = dvf[s, :, :, 2].cpu().numpy() * THROUGH_MM
                panel([imgs[s].cpu().numpy(), dx, dy, dz],
                      ["input slice", "Δx (mm)", "Δy (mm)", "Δz through-plane (mm)"],
                      os.path.join(OUT, f"dvf_s{sub}_ES.png"),
                      cmaps=["gray", "RdBu_r", "RdBu_r", "RdBu_r"],
                      vranges=[(0, 1), (-8, 8), (-8, 8), (-8, 8)],
                      sup=f"subj{sub} predicted displacement (note Δz≈0: little through-plane cardiac motion)")
                # p95/p99 accumulation over anatomy (all slots, resp OFF)
                ip, dzs, dza = dvf_mm(dvf, imgs)
                for sl in range(1, dvf.shape[0]):
                    m = imgs[sl] > ANAT
                    if not m.any(): continue
                    ipmm = torch.sqrt((dvf[sl, :, :, 0][m] * INPLANE_MM) ** 2 + (dvf[sl, :, :, 1][m] * INPLANE_MM) ** 2)
                    dzmm = (dvf[sl, :, :, 2][m] * THROUGH_MM).abs()
                    p95_ip.append(float(ipmm.quantile(0.95))); p99_ip.append(float(ipmm.quantile(0.99)))
                    p95_dz.append(float(dzmm.quantile(0.95))); p99_dz.append(float(dzmm.quantile(0.99)))

        # cardiac cycle: V_gt vs V_canon at mid-z across 12 phases
        gts, cans = [], []
        for t in range(12):
            ds.t_target_fixed = t
            data = ds.get_data(seq_index=sub, img_per_seq=12); b = build_batch(data, dev, sub)
            out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b, grid_shape=GRID_SHAPE, tv_weight=0.0)
            gts.append(out["V_gt"][0].float().cpu().numpy()[ZMID])
            b = gpu_augment_batch(b, None, dev, respiratory_cfg=cfg_off, train=False)
            Vm, _ = fwd(model, b); cans.append(Vm.cpu().numpy()[ZMID])
        fig, ax = plt.subplots(2, 12, figsize=(20, 3.6), dpi=100)
        for t in range(12):
            ax[0, t].imshow(gts[t], cmap="gray", vmin=0, vmax=1); ax[0, t].axis("off"); ax[0, t].set_title(f"t{t}", fontsize=7)
            ax[1, t].imshow(cans[t], cmap="gray", vmin=0, vmax=1); ax[1, t].axis("off")
        ax[0, 0].set_ylabel("GT", fontsize=9); ax[1, 0].set_ylabel("model", fontsize=9)
        fig.suptitle(f"subj{sub} cardiac cycle @ mid-z (top GT, bottom model): does the chamber contract as much?", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"cardiac_cycle_s{sub}.png"), bbox_inches="tight"); plt.close(fig)

        # breathing: deep-breath slot clean vs breathed vs model
        ds.t_target_fixed = 0
        data = ds.get_data(seq_index=sub, img_per_seq=12)
        bcl = build_batch(data, dev, sub); bbr = build_batch(data, dev, sub)
        bcl = gpu_augment_batch(bcl, None, dev, respiratory_cfg=cfg_off, train=False)
        bbr = gpu_augment_batch(bbr, None, dev, respiratory_cfg=cfg_on, train=False)
        disp = bbr["resp_disp_mm"][0][:, 0].abs().cpu().numpy()
        sdeep = int(np.argmax(disp))
        icl = bcl["images"][0].float().mean(1)[sdeep].cpu().numpy()
        ibr = bbr["images"][0].float().mean(1)[sdeep].cpu().numpy()
        panel([icl, ibr, np.abs(ibr - icl)],
              [f"clean input (slot {sdeep})", f"breathed input (SI={disp[sdeep]:.0f}mm)", "|difference|"],
              os.path.join(OUT, f"breath_s{sub}.png"), cmaps=["gray", "gray", "magma"],
              vranges=[(0, 1), (0, 1), (0, 0.5)], sup=f"subj{sub} breathing shifts anatomy through-plane by {disp[sdeep]:.0f}mm")
        print(f"subj{sub} done", flush=True)

    print(f"\nP95/P99 cardiac DVF over anatomy (resp OFF, ES): "
          f"in-plane p95={np.mean(p95_ip):.2f} p99={np.mean(p99_ip):.2f} mm | "
          f"through p95={np.mean(p95_dz):.2f} p99={np.mean(p99_dz):.2f} mm")
    print("(compare to mean ~0.5/0.1: if p95 is several mm, motion is localized not absent)")
    print(f"figs -> {OUT}")


if __name__ == "__main__":
    main()
