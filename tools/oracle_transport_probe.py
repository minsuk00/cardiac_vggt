"""Oracle-transport held-out-target-phase probe (the debate's decisive gate).

QUESTION: In the one-frame-per-slice regime (each plane observed at ONE random cardiac
phase, NEVER the target phase tt), can the target-phase appearance be recovered by
TRANSPORTING the observed other-phase tissue via ORACLE motion — or is it an information
wall (the tissue simply isn't there)?

Renderers (all native grid, leak-free wrt tt: no plane is ever observed at tt):
  F  identity floor : scatter each observed slice at its OWN plane, no motion (do-nothing).
  B  oracle transport: scatter observed slices -> phase-0 template via oracle dvf_{t_k};
                       render tt by gathering template at (q + dvf_tt(q))  [pooled template
                       + backward-gather = the untested DOF the debate isolated].
  O  perfect oracle : scatter the TRUE tt slices at every plane (uses the target-phase
                       observation we DON'T have) -> the recoverable ceiling given full
                       target-phase coverage. Anchors the top.

Decision (space-robust): recov_frac_B = (MSE_F - MSE_B)/(MSE_F - MSE_O) on the cardiac
motion ROI (and heart_roi). GO if B recovers a large fraction (oracle-transport PSNR_B
clears ~+3 dB over floor / recov_frac >~0.35); NO-GO if ~0 (wall is architecture-independent).

Oracle motion = on-disk elastix DVF (T->0, mm, native grid). Convention VALIDATED in
tools/_probe_dvf_geometry.py: disp_t = dvf_t/|spacing| maps frame_t -> frame_0 by ADDITION
(f_t[q] ~= f0[q + disp_t(q)]).
"""
import os, glob, json, argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "vggt")
from vggt.utils.splat import splat_to_volume  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = "scratch/data/CMRxRecon2024/Cine_combined"
SPLIT = "training/splits/random_8_1_1.txt"
TAU = 0.05


# ── loading ────────────────────────────────────────────────────────────────
def load_subject(sax):
    ph = sorted(glob.glob(os.path.join(sax, "3d_recon", "sax_frame_*.nii.gz")))
    T = len(ph)
    aff = nib.load(ph[0]).affine
    spacing_abs = np.abs([aff[0, 0], aff[1, 1], aff[2, 2]]).astype(np.float32)
    P = np.stack([np.asarray(nib.load(p).dataobj).astype(np.float32) for p in ph], 0)  # (T,X,Y,Z)
    v0 = P[0]; nz = v0[v0 > 0]
    lo, hi = np.percentile(nz, 0.5), np.percentile(nz, 99.5)
    P = np.clip((P - lo) / (hi - lo + 1e-8), 0, 1)
    X, Y, Z = P.shape[1:]
    D = np.zeros((T, X, Y, Z, 3), np.float32)  # disp_t = dvf/|spc| (index units), T->0 add-convention
    for t in range(1, T):
        d = np.asarray(nib.load(os.path.join(sax, "dvf_elastix", f"dvf_frame_{t:02d}.nii.gz")).dataobj)[..., 0, :]
        D[t] = d / spacing_abs
    roi = None
    rp = os.path.join(sax, "heart_roi.nii.gz")
    if os.path.exists(rp):
        roi = np.asarray(nib.load(rp).dataobj).astype(bool)
    return P, D, roi


# ── primitives (grid_sample gather validated in _probe_dvf_geometry) ─────────
def _base_grid(X, Y, Z):
    ii, jj, kk = torch.meshgrid(torch.arange(X, device=DEV, dtype=torch.float32),
                                torch.arange(Y, device=DEV, dtype=torch.float32),
                                torch.arange(Z, device=DEV, dtype=torch.float32), indexing="ij")
    return ii, jj, kk


def warp(vol, disp):
    """gather: warp(vol,disp)[p] = vol(p+disp). vol (X,Y,Z), disp (X,Y,Z,3) index units."""
    X, Y, Z = vol.shape
    ii, jj, kk = _base_grid(X, Y, Z)
    si, sj, sk = ii + disp[..., 0], jj + disp[..., 1], kk + disp[..., 2]
    grid = torch.stack([sk / (Z - 1) * 2 - 1, sj / (Y - 1) * 2 - 1, si / (X - 1) * 2 - 1], -1).unsqueeze(0)
    return F.grid_sample(vol.view(1, 1, X, Y, Z), grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True).view(X, Y, Z)


def scatter(points_idx, inten, shape):
    """Forward splat of (index-position, intensity) into an (X,Y,Z) volume, matching warp's
    axis convention: splat grid (D=X,H=Y,W=Z), pos (x<->Z, y<->Y, z<->X). Intensity-gated."""
    X, Y, Z = shape
    px = points_idx[..., 2] / (Z - 1) * 2 - 1  # splat x  <-> Z
    py = points_idx[..., 1] / (Y - 1) * 2 - 1  # splat y  <-> Y
    pz = points_idx[..., 0] / (X - 1) * 2 - 1  # splat z  <-> X
    pos = torch.stack([px, py, pz], -1).view(1, -1, 3)
    val = inten.reshape(1, -1)
    w = (val > 1e-3).float()
    vol, cov = splat_to_volume(pos, val, (X, Y, Z), weight=w)  # grid (D=X,H=Y,W=Z)
    return vol.view(X, Y, Z), cov.view(X, Y, Z)


def psnr(a, b, m):
    mse = ((a - b)[m] ** 2).mean().clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item(), mse.item()


def ncc(a, b, m):
    """Masked normalized cross-correlation (Pearson) over ROI voxels — invariant to
    global intensity scale/offset, so it removes the intensity-calibration component."""
    x = a[m].float(); y = b[m].float()
    x = x - x.mean(); y = y - y.mean()
    d = (x.norm() * y.norm()).clamp(min=1e-8)
    return float((x @ y) / d)


def ssim_roi(a, b, roi):
    """2D per-slice SSIM (windowed structure) on the heart bbox crop, averaged over
    ROI-bearing planes. data_range=1 (volumes are [0,1])."""
    from skimage.metrics import structural_similarity as ssim
    A = a.cpu().numpy(); B = b.cpu().numpy(); R = roi.cpu().numpy()
    zs = np.where(R.any((0, 1)))[0]
    xs = np.where(R.any((1, 2)))[0]; ys = np.where(R.any((0, 2)))[0]
    if len(zs) == 0 or len(xs) == 0 or len(ys) == 0:
        return float("nan")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    vals = []
    for z in zs:
        pa = A[x0:x1, y0:y1, z]; pb = B[x0:x1, y0:y1, z]
        if min(pa.shape) < 7:
            continue
        vals.append(ssim(pa, pb, data_range=1.0))
    return float(np.mean(vals)) if vals else float("nan")


# ── one subject / one target phase ──────────────────────────────────────────
def run_case(P, D, roi, tt, seed, reference=False):
    T, X, Y, Z = P.shape
    Pt = torch.from_numpy(P).to(DEV)
    Dt = torch.from_numpy(D).to(DEV)
    ii, jj, kk = _base_grid(X, Y, Z)
    base = torch.stack([ii, jj, kk], -1)  # (X,Y,Z,3)

    motion = (Pt.amax(0) - Pt.amin(0)) > TAU               # (X,Y,Z) cardiac motion ROI
    roi_t = torch.from_numpy(roi).to(DEV) if roi is not None else motion
    Vgt = Pt[tt]

    # in-bbox planes = planes intersecting the heart ROI
    plane_has = roi_t.any(0).any(0)                        # (Z,)
    planes = [k for k in range(Z) if bool(plane_has[k])] or list(range(Z))

    # one-frame sampling: each plane observed at ONE phase != tt
    rng = np.random.default_rng(seed)
    t_of = {k: int(rng.choice([t for t in range(T) if t != tt])) for k in planes}
    # reference-slot (deployment): the mid-ventricular plane IS observed at the target phase tt
    if reference:
        z_mid = planes[len(planes) // 2]
        t_of[z_mid] = tt

    # gather observed slice pixels (only foreground pixels, to limit scatter points)
    def slice_pts_inten(disp_field_for_frame):
        pos_list, int_list = [], []
        for k in planes:
            tk = t_of[k]
            img = Pt[tk, :, :, k]                          # (X,Y)
            fg = img > 1e-3
            idx = torch.nonzero(fg, as_tuple=False)        # (M,2) -> (i,j)
            i, j = idx[:, 0], idx[:, 1]
            p = torch.stack([i.float(), j.float(), torch.full_like(i.float(), k)], -1)  # (M,3)
            if disp_field_for_frame is not None:
                d = Dt[tk][i, j, k]                        # (M,3) disp at those pixels
                p = p + d
            pos_list.append(p)
            int_list.append(img[i, j])
        return torch.cat(pos_list, 0), torch.cat(int_list, 0)

    # F: identity floor (each slice at its own plane, no motion)
    posF, intF = slice_pts_inten(None)
    VF, _ = scatter(posF, intF, (X, Y, Z))

    # B: oracle transport -> phase-0 template (scatter via dvf_{t_k}), then gather to tt
    posB, intB = slice_pts_inten("use_disp")             # positions carry +disp_{t_k} -> frame0
    template, _ = scatter(posB, intB, (X, Y, Z))
    VB = warp(template, Dt[tt])                            # frame0 -> tt : gather at q + disp_tt(q)

    # O: perfect oracle (true tt slices at every plane, no motion)
    posO_list, intO_list = [], []
    for k in planes:
        img = Pt[tt, :, :, k]; fg = img > 1e-3
        idx = torch.nonzero(fg, as_tuple=False); i, j = idx[:, 0], idx[:, 1]
        posO_list.append(torch.stack([i.float(), j.float(), torch.full_like(i.float(), k)], -1))
        intO_list.append(img[i, j])
    VO, _ = scatter(torch.cat(posO_list, 0), torch.cat(intO_list, 0), (X, Y, Z))

    out = {}
    for mname, m in [("motroi", motion & roi_t), ("roi", roi_t)]:
        pF, mseF = psnr(VF, Vgt, m); pB, mseB = psnr(VB, Vgt, m); pO, mseO = psnr(VO, Vgt, m)
        recov = (mseF - mseB) / (mseF - mseO + 1e-12)
        # structural metrics (higher=better); recov_frac = (B-F)/(O-F)
        nF, nB, nO = ncc(VF, Vgt, m), ncc(VB, Vgt, m), ncc(VO, Vgt, m)
        sF, sB, sO = ssim_roi(VF, Vgt, m), ssim_roi(VB, Vgt, m), ssim_roi(VO, Vgt, m)
        out[mname] = dict(psnr_F=pF, psnr_B=pB, psnr_O=pO, recov_frac_B=float(recov),
                          ncc_F=nF, ncc_B=nB, ncc_O=nO, recov_ncc=float((nB - nF) / (nO - nF + 1e-8)),
                          ssim_F=sF, ssim_B=sB, ssim_O=sO, recov_ssim=float((sB - sF) / (sO - sF + 1e-8)),
                          mseF=mseF, mseB=mseB, mseO=mseO, n=int(m.sum()))
    return out, (VF, VB, VO, Vgt, motion)


def find_ES(P):
    Pt = torch.from_numpy(P)
    mot = (Pt.amax(0) - Pt.amin(0)) > TAU
    d = ((Pt - Pt[0:1]) ** 2 * mot).sum((1, 2, 3))
    return int(torch.argmax(d))


def val_subjects(n):
    subs, cur = [], None
    for line in open(SPLIT):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"): cur = s[1:-1].lower(); continue
        if cur == "val" and s and not s.startswith("#"):
            p = os.path.join(DATA, s, "sax")
            if os.path.isdir(os.path.join(p, "3d_recon")): subs.append((s, p))
    return subs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--reference", action="store_true",
                    help="deployment mode: mid plane observes the target phase (reference slot)")
    args = ap.parse_args()
    print(f"MODE: {'reference-slot (mid plane sees target phase)' if args.reference else 'strict (no target-phase observation)'}")
    subs = val_subjects(args.n)
    print(f"running {len(subs)} val subjects x {{ED, ES}}\n")
    rows = []
    for si, (sid, sax) in enumerate(subs):
        P, D, roi = load_subject(sax)
        es = find_ES(P)
        for phase_name, tt in [("ED", 0), ("ES", es)]:
            res, vols = run_case(P, D, roi, tt, seed=1000 * si + tt, reference=args.reference)
            m = res["motroi"]
            print(f"{sid:>12} {phase_name}(t={tt:<2}) mot∩roi: "
                  f"F={m['psnr_F']:.2f}  B={m['psnr_B']:.2f}  O={m['psnr_O']:.2f}  "
                  f"B-F={m['psnr_B']-m['psnr_F']:+.2f}  recov_B={m['recov_frac_B']:+.3f}  (n={m['n']})")
            rows.append(dict(sid=sid, phase=phase_name, tt=tt, **{f"m_{k}": v for k, v in res["motroi"].items()},
                             **{f"r_{k}": v for k, v in res["roi"].items()}))
            if args.fig and si == 0:
                save_fig(vols, sid, phase_name, tt)
    # aggregate
    def agg(key):
        return float(np.mean([r[key] for r in rows]))

    def report(prefix, label):
        print(f"\n=== AGGREGATE ({label}) — n={len(rows)} cases ===")
        print("  metric      F        B        O     |  B-F   recov(B)  | ED recov  ES recov")
        for met, rk in [("psnr", "recov_frac_B"), ("ncc", "recov_ncc"), ("ssim", "recov_ssim")]:
            F, B, O = agg(f"{prefix}{met}_F"), agg(f"{prefix}{met}_B"), agg(f"{prefix}{met}_O")
            rec = agg(prefix + rk)
            edr = np.mean([r[prefix + rk] for r in rows if r["phase"] == "ED"])
            esr = np.mean([r[prefix + rk] for r in rows if r["phase"] == "ES"])
            print(f"  {met:>5}  {F:7.3f}  {B:7.3f}  {O:7.3f}  | {B-F:+6.3f}  {rec:+7.3f}  | "
                  f"{edr:+7.3f}  {esr:+7.3f}")
    report("m_", "motion ∩ heart ROI")
    report("r_", "heart ROI")
    os.makedirs("result/oracle_transport_check", exist_ok=True)
    json.dump(rows, open("result/oracle_transport_check/probe_results.json", "w"), indent=2)
    print("\nwrote result/oracle_transport_check/probe_results.json")


def save_fig(vols, sid, phase_name, tt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    VF, VB, VO, Vgt, motion = [v.cpu().numpy() for v in vols]
    # representative in-ROI plane: max motion among planes O actually covers
    covered = (VO > 1e-3).any((0, 1))
    mc = motion.sum((0, 1)) * covered
    zc = int(np.argmax(mc))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, im, ti in zip(ax, [Vgt, VF, VB, VO],
                         [f"GT {phase_name}", "F identity-floor", "B oracle-transport", "O perfect-oracle"]):
        a.imshow(im[:, :, zc].T, cmap="gray", vmin=0, vmax=1); a.set_title(ti); a.axis("off")
    p = f"result/oracle_transport_check/render_{sid}_{phase_name}_z{zc}.png"
    fig.savefig(p, dpi=90, bbox_inches="tight"); print("saved", p)


if __name__ == "__main__":
    main()
