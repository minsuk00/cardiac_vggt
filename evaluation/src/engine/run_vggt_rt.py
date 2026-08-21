#!/usr/bin/env python
"""Zero-shot VGGT reconstruction of REAL real-time free-breathing MIITT acquisitions.

Unlike run_vggt.py (gated cine + frozen SIMULATED breathing), this feeds the model the actual
free-breathing real-time recordings (`scratch/data/MIITT/nifti/<subj>/realtime/sax/4d_recon.nii.gz`,
~180 frames x D slices, golden-angle spiral R=12, 2.3x2.3 mm in-plane, 25 ms/frame). A real-time
recording is not gated to one cardiac cycle, so instead of run_vggt's 12-phase reference sweep the
reference slot sweeps EVERY real acquired frame; each companion slot keeps ONE fixed real frame of
its own z-plane (the trained one-frame-per-slice contract — same inputs, varying query).

Geometry (scanner_coords / dz / z_scale / slot draw) comes from the standard MRIDataset path,
pointed at the RT recording ITSELF: 12 RT frames are written as a temp MIITT_sax-layout subject
(`make_rt_scaffold`) and preprocessed normally, so the geometry contract has one implementation
(docs/79) and each subject is self-contained — no gated-stack dependency (the AFib patient's RT
D=12 vs gated D=15 just works). dz = 10 mm (8 mm slices + 2 mm gap) comes from the RT header.

This is a qualitative, RETROSPECTIVE demo: companions are subsampled post-hoc from a densely
sampled recording (not a prospectively sparse acquisition) and real-time data has NO ground truth
— nothing here is scored. The gated cine in the GIF is anatomy context only, NOT phase-aligned
(different scan, no shared clock). Promoted from temp/miitt_rt_probe_full/ (2026-08-18 session).

Outputs (volumes/miitt_rt/out/<subject>/):
    rt_input.nii.gz               canonical-resampled raw RT input (model-independent)
    <model_name>/recon_rt.nii.gz  (X,Y,Z,T_rt) float16 recon, one volume per real frame
    <model_name>/gif_rt.gif       gated context (when D matches) | raw RT | model input | recon,
                                  every --stride'th frame, reference plane red-starred
    <model_name>/panel_dvf_rt.png frame-0 input slices + predicted Δ (mm) at 6 z-levels

Run:
  micromamba run -n svr env PYTHONPATH=training:. python evaluation/src/engine/run_vggt_rt.py \
      --ckpt scratch/logs/<run>/ckpts/checkpoint_last.pt --model-name vggt_augaggr224hw2_ep300
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "analysis"))

import paths                                                                   # noqa: E402
import run_vggt as rv                                                          # noqa: E402
import viz                                                                     # noqa: E402
from data.gpu_aug import gpu_augment_batch                                     # noqa: E402
from data.preprocess import Z_HALF_MM                                          # noqa: E402
from loss import _splat_preds_native                                           # noqa: E402
from inference.load_run import load_model_from_run                             # noqa: E402

RT_ROOT = os.path.join(ROOT, "scratch/data/MIITT/nifti")
OUT_ROOT = paths.VOLUMES / "miitt_rt" / "out"
INPLANE_MM_RT = 2.3            # protocol: FOV 300 mm / 128 (confirmed 2026-08-20)
FRAME_MS = 25.0                # protocol: 25 ms/frame -> true playback 40 fps
PCT_LO, PCT_HI = 0.5, 99.9
DEFAULT_SUBJECTS = ["MIITT_Volunteer1", "MIITT_Volunteer2", "MIITT_Volunteer3"]


# ── RT input -> canonical grid ────────────────────────────────────────────────
def to_canonical_inplane(slice2d):
    """One RT slice -> 256x256 canonical 1.4 mm grid (bilinear resample + center crop/pad),
    the same in-plane convention as MRIDataset's cached phases."""
    H, W = slice2d.shape
    sh = int(round(H * INPLANE_MM_RT / rv.INPLANE_MM))
    sw = int(round(W * INPLANE_MM_RT / rv.INPLANE_MM))
    r = F.interpolate(torch.from_numpy(slice2d)[None, None].float(), size=(sh, sw),
                      mode="bilinear", align_corners=True)[0, 0]
    out = torch.zeros(256, 256)
    y0s, x0s = max(0, (sh - 256) // 2), max(0, (sw - 256) // 2)
    y0d, x0d = max(0, (256 - sh) // 2), max(0, (256 - sw) // 2)
    hh, ww = min(sh, 256), min(sw, 256)
    out[y0d:y0d + hh, x0d:x0d + ww] = r[y0s:y0s + hh, x0s:x0s + ww]
    return out.numpy()


def build_rt_bundle(nii_path):
    """RT NIfTI (X,Y,Z,T_rt) -> normalized canonical (T_rt, D, 256, 256) in [0,1]."""
    a = nib.load(nii_path).get_fdata().astype(np.float32)
    cine = np.transpose(a, (3, 2, 1, 0))                        # (T_rt, Z, H, W)
    nz = cine[cine > 0] if (cine > 0).any() else cine.reshape(-1)
    vmin = float(np.percentile(nz, PCT_LO))
    vmax = max(float(np.percentile(nz, PCT_HI)), vmin + 1e-6)
    n_frames, D = cine.shape[:2]
    bundle = np.zeros((n_frames, D, 256, 256), np.float32)
    for f in range(n_frames):
        for z in range(D):
            bundle[f, z] = to_canonical_inplane(np.clip((cine[f, z] - vmin) / (vmax - vmin), 0, 1))
    return bundle


# ── batch build + reconstruction ──────────────────────────────────────────────
def make_rt_scaffold(rt_path, subj, tmpdir):
    """Write 12 evenly-spaced RT frames as a temp MIITT_sax-layout subject so the standard
    MRIDataset/preprocess builds the geometry (D, dz, scanner_coords, slot draw) from the RT
    recording ITSELF — no gated-stack dependency. The scaffold's pixel content is discarded
    (the full RT bundle is swapped in); only its geometry and content mask are used. Returns
    the absolute subject dir (MRIDataset joins absolute split entries as-is)."""
    img = nib.load(rt_path)
    a = img.get_fdata().astype(np.float32)                      # (X, Y, Z, T_rt)
    rec_dir = os.path.join(tmpdir, subj, "sax", "3d_recon")
    os.makedirs(rec_dir)
    for j, f in enumerate(np.round(np.linspace(0, a.shape[3] - 1, 12)).astype(int)):
        nib.save(nib.Nifti1Image(a[..., f], img.affine), os.path.join(rec_dir, f"sax_frame_{j:02d}.nii.gz"))
    return os.path.join(tmpdir, subj)


def load_gated_context(cfg, subj, D, tmpdir):
    """The subject's own gated canonical phases (T_gated, D, H, W) for the GIF context row.
    When the gated stack has MORE slices than the RT stack (e.g. the AFib patient: gated D=15
    vs RT D=12), take the central D slices — approximate per-plane alignment, display only.
    None when the subject has no gated entry or the gated stack is smaller."""
    if not os.path.isdir(os.path.join(ROOT, "scratch/data/MIITT_sax", subj)):
        return None
    ds = rv.make_dataset(cfg, f"MIITT_sax/{subj}", "val", tmpdir)
    gated = np.asarray(ds.get_data(seq_index=0, img_per_seq=ds.num_slices)["phases"], np.float32)
    Dg = gated.shape[1]
    if Dg < D:
        return None
    z0 = (Dg - D) // 2
    return gated[:, z0:z0 + D]


def build_batch_rt(ds, seq_index, bundle, device):
    """Like run_vggt.build_batch, but the fed bundle's frame axis is the RT recording length
    (NOT required to equal the gated T=12). Each companion slot's gated-phase draw t in
    0..T_gated-1 is remapped to a fixed representative RT frame via linspace, so every
    companion keeps exactly one real frame of its own z-plane."""
    b = ds.get_data(seq_index=seq_index, img_per_seq=ds.num_slices)
    T_gated, D_ds = np.asarray(b["phases"]).shape[:2]
    if bundle.shape[1] != D_ds:
        raise ValueError(f"RT bundle D={bundle.shape[1]} != gated dataset D={D_ds}")
    _DTYPE = {"timesteps": torch.int64, "slice_indices": torch.float32}
    out = {}
    for k, v in b.items():                              # collate to batch size 1 (as run_vggt)
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)[None].to(device)
        elif torch.is_tensor(v):
            out[k] = v[None].to(device)
        elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(v))[None].to(device)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
            out[k] = torch.tensor(v, dtype=_DTYPE.get(k, torch.float32))[None].to(device)
        else:
            out[k] = v
    frame_of_t = np.round(np.linspace(0, bundle.shape[0] - 1, int(T_gated))).astype(int)
    out["timesteps"] = torch.tensor([[int(frame_of_t[int(t)]) for t in out["timesteps"][0]]],
                                    dtype=torch.int64, device=device)
    out["phases"] = torch.from_numpy(bundle)[None].to(device)   # THE swap: gated -> real RT pixels
    out.pop("images", None)                                     # force re-extraction from phases
    out.pop("images_splat", None)
    return out


@torch.no_grad()
def reconstruct_rt(model, ds, seq_index, bundle, device):
    """Sweep slot 0 (reference, z_mid) over every real frame; companions fixed.
    Returns ((T_rt, D, 256, 256) recon, mean ms/frame, frame-0 DVF pack)."""
    batch = build_batch_rt(ds, seq_index, bundle, device)
    slot_frames = batch["timesteps"][0].cpu().numpy().copy()   # fixed companion frame per slot
    n_frames, D = bundle.shape[:2]
    z_scale = float(batch["z_scale"].reshape(-1)[0])
    vols, ms, dvf_pack = [], [], None
    for f in range(n_frames):
        batch["timesteps"][0, 0] = f
        batch.pop("images", None)                       # rebuilt from phases at the new query
        gpu_augment_batch(batch, None, device, respiratory_cfg=None, train=False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        wp = preds["world_points"].float()
        V, _ = _splat_preds_native({"world_points": wp}, batch, (D, 256, 256), z_scale)
        torch.cuda.synchronize(); ms.append((time.perf_counter() - t0) * 1e3)
        vols.append(V[0].float().cpu().numpy())
        if f == 0:
            dvf_pack = dict(delta=(wp[0] - batch["scanner_coords"][0].float()).cpu().numpy(),
                            images=batch["images"][0].float().mean(1).cpu().numpy(),
                            slot_z=batch["slice_indices"][0].cpu().numpy(),
                            slot_frames=slot_frames)
    return np.stack(vols), float(np.mean(ms)), dvf_pack


# ── outputs ───────────────────────────────────────────────────────────────────
def save_cine_xyzt(path, cine_tdhw, dz_mm):
    """(T,D,H,W) splat order -> (X,Y,Z,T) NIfTI on the canonical grid (matches paths.cine).
    float32 — NIfTI has no float16; gzip absorbs most of the difference."""
    xyz_t = np.transpose(cine_tdhw, (3, 2, 1, 0)).astype(np.float32)
    affine = np.diag([rv.INPLANE_MM, rv.INPLANE_MM, dz_mm, 1.0])
    nib.save(nib.Nifti1Image(xyz_t, affine), str(path))


def render_dvf_panel(path, dvf_pack, vmax, subj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    delta, imgs, slot_z = dvf_pack["delta"], dvf_pack["images"], dvf_pack["slot_z"]
    sel = np.argsort(slot_z)[np.round(np.linspace(0, len(slot_z) - 1,
                                                  min(6, len(slot_z)))).astype(int)]
    fig, axes = plt.subplots(4, len(sel), figsize=(len(sel) * 1.3, 5.2))
    for c, s in enumerate(sel):
        shown = viz.display_gamma(imgs[s], vmax) if viz.GAMMA_ON \
            else np.clip(imgs[s] / max(vmax, 1e-3), 0, 1)
        axes[0, c].imshow(shown, cmap="gray", origin="lower", vmin=0, vmax=1)
        for r in range(3):
            d = delta[s, ..., r] * rv.MM_PER_NORM[r]
            lim = float(np.percentile(np.abs(d), 99)) or 1.0
            axes[r + 1, c].imshow(d, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
        for r in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[0, c].set_title(f"z{int(round(slot_z[s]))}", fontsize=7)
    for r, lbl in enumerate(("input", "Δx (mm)", "Δy (mm)", "Δz (mm)")):
        axes[r, 0].set_ylabel(lbl, fontsize=7)
    fig.suptitle(f"{subj} — RT input + predicted Δ at frame 0 (no GT){viz.GAMMA_TAG}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"  -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-name", required=True, help="arm dir name, e.g. vggt_augaggr224hw2_ep300")
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--stride", type=int, default=3,
                    help="GIF display stride (every frame is still reconstructed/saved)")
    args = ap.parse_args()

    device = torch.device("cuda")
    model, cfg = load_model_from_run(args.ckpt, device=device)
    print("model loaded")

    for subj in args.subjects:
        rt_path = os.path.join(RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax/4d_recon.nii.gz")
        if not os.path.exists(rt_path):
            print(f"SKIP {subj}: no RT nifti at {rt_path}"); continue
        print(f"\n=== {subj} ===")

        bundle = build_rt_bundle(rt_path)
        n_frames, D = bundle.shape[:2]
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = rv.make_dataset(cfg, make_rt_scaffold(rt_path, subj, tmpdir), "val", tmpdir)
            b0 = ds.get_data(seq_index=0, img_per_seq=ds.num_slices)
            dz_mm = float(np.asarray(b0["dz_mm"]).reshape(-1)[0])
            assert np.asarray(b0["phases"]).shape[1] == D, "scaffold D != RT D (should be impossible)"
            gated = load_gated_context(cfg, subj, D, tmpdir)
            print(f"  D={D} dz={dz_mm} mm  n_frames_rt={n_frames}  gated context: "
                  f"{'yes' if gated is not None else 'no (missing or D mismatch)'}")
            t0 = time.perf_counter()
            recon, ms_per_frame, dvf_pack = reconstruct_rt(
                model, ds, rv.name_seed("miitt", subj), bundle, device)
            print(f"  reconstructed {n_frames} frames in {time.perf_counter() - t0:.1f}s "
                  f"({ms_per_frame:.0f} ms/frame)")

        odir = OUT_ROOT / subj / args.model_name
        os.makedirs(odir, exist_ok=True)
        rt_input_path = OUT_ROOT / subj / "rt_input.nii.gz"
        if not rt_input_path.exists():
            save_cine_xyzt(rt_input_path, bundle, dz_mm)
            print(f"  -> {rt_input_path}")
        save_cine_xyzt(odir / "recon_rt.nii.gz", recon, dz_mm)
        print(f"  -> {odir / 'recon_rt.nii.gz'}")

        # GIF rows in viz.render_gif's (T,X,Y,Z) order, subsampled for display only.
        sl = slice(None, None, args.stride)
        # Model-input row: what the model actually sees — every companion z frozen at its one
        # fixed real frame, only the reference plane animating with the swept query frame.
        zi = np.round(dvf_pack["slot_z"]).astype(int)
        frame_of_z = np.empty(D, int)
        frame_of_z[zi] = dvf_pack["slot_frames"]
        ref_z = int(zi[0])
        disp_idx = np.arange(0, n_frames, args.stride)
        model_in = np.repeat(bundle[frame_of_z, np.arange(D)][None], len(disp_idx), 0)  # (n_disp,D,H,W)
        model_in[:, ref_z] = bundle[disp_idx, ref_z]

        rows = [("RT input\n(raw, no model)", np.transpose(bundle, (0, 3, 2, 1))[sl]),
                ("model input\n(1 frame/slice,\n★ = animated ref)", np.transpose(model_in, (0, 3, 2, 1))),
                ("RT recon", np.transpose(recon, (0, 3, 2, 1))[sl])]
        if gated is not None:
            rows.insert(0, ("gated cine\n(cycled, NOT\nphase-aligned)",
                            np.transpose(gated, (0, 3, 2, 1))[np.arange(n_frames) % gated.shape[0]][sl]))
        vals = np.concatenate([r[r > 0].ravel() for _, r in rows])
        vmax = float(np.percentile(vals, 99.9)) if vals.size else 1.0
        viz.render_gif(str(odir / "gif_rt.gif"), rows, list(range(D)), rows[0][1].shape[0], vmax,
                       f"{subj} — real-time free-breathing, {n_frames} frames "
                       f"(every {args.stride}, true rate 40 fps)   frame={{t}}",
                       fps=1000.0 / (FRAME_MS * args.stride),
                       ref_z=ref_z)                                      # reference plane, red-starred
        render_dvf_panel(odir / "panel_dvf_rt.png", dvf_pack, vmax, subj)

    print("\nRUN_VGGT_RT_DONE")


if __name__ == "__main__":
    main()
