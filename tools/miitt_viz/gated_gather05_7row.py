"""gather05 on the NEW gated-OOD pipeline, 7-row animated view (GT|input|recon|Dx|Dy|Dz|coverage)
per z-plane over the cardiac cycle — the gated twin of tools/miitt_viz/extreme_dvf_cov.py.
Datasets: MIITT gated (Volunteer3) + OCMR gated (fs_0060), each x {clean, normal(amp16),
extreme(amp50)}. Loads gather05 ONCE. Captures per-phase recon/input/DVF/coverage and saves a
per-condition npz so ANY re-render is CPU-only (no GPU rerun).

Run: micromamba run -n svr env PYTHONPATH=training:. python -u tools/miitt_viz/gated_gather05_7row.py
"""
import os, sys, glob, json, dataclasses, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "training")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio
from inference.adapters import MIITTGatedAdapter, OCMRAdapter
from inference.adapters.base import GRID_SHAPE, MM_PER_NORM, INPUT_IMG_SIZE, assign_canonical_z
from inference.inference import load_rtfb_model_reference
from inference.run_cmrxrecon import _build_multiframe_batch
from inference.run_gated_ood import load_rcfg
from data.gpu_aug import gpu_augment_batch, extract_slices_from_phases
from loss import compute_volume_intensity_loss

DEV = "cuda"; OUT = "result/gated_gather05_eval"; os.makedirs(OUT, exist_ok=True)
FPS = 5; D = GRID_SHAPE[0]; MM = np.array(MM_PER_NORM)
CKPT = glob.glob("scratch/logs/216539845_*ftgather05*1frame*/ckpts/checkpoint_last.pt")[0]
SUBJECTS = [
    ("miitt", "Volunteer3", lambda: MIITTGatedAdapter("scratch/data/MIITT/nifti/Volunteer3/gated/sax/4d_recon.nii.gz")),
    ("ocmr", "fs_0060", lambda: OCMRAdapter("scratch/data/ocmr/recon/gated/exam_fs_0060/sax__fs_0060_1_5T")),
]
_MK = {"motion": "metric_psnr_3d_motion", "bbox": "metric_psnr_3d_bbox",
       "full": "metric_psnr_3d_full", "ssim": "metric_ssim_3d_full"}


def to256(a):
    idx = np.linspace(0, a.shape[0] - 1, 256).astype(int)
    return a[np.ix_(idx, idx)]


def build_1frame_batch(phases_bundle, bbox, seq_index, device):
    """1-frame-per-plane batch — matches gather05's training budget (the '_1frame_' run). Slot 0 =
    swept reference at z_mid; every OTHER in-bbox plane contributes exactly ONE frame at a random
    (seeded) phase. NO reference-plane cine companions, NO multi-frame bursts. S = #in-bbox planes
    (~10 vs ~76 for multiframe), so ~4-5x cheaper. Same batch-key contract as _build_multiframe_batch."""
    T, Dd, H, W = phases_bundle.shape
    z0, z1 = int(bbox[0]), int(bbox[1]); z_mid = (z0 + z1) // 2
    in_bbox_z = list(range(z0, z1)) or [z_mid]
    rng = np.random.default_rng(seq_index)
    slots = [(z_mid, 0)]                                   # slot 0: swept reference (overwritten per phase)
    for z in in_bbox_z:
        if z == z_mid:
            continue
        slots.append((z, int(rng.integers(T))))           # ONE random-phase frame per other plane
    S = len(slots)
    hw = INPUT_IMG_SIZE
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    slot_t = torch.tensor([t for _, t in slots], dtype=torch.long)
    slot_z = torch.tensor([z for z, _ in slots], dtype=torch.long)
    canon = phases_bundle[slot_t, slot_z].unsqueeze(1)    # (S,1,256,256) on device
    up = F.interpolate(canon, size=(hw, hw), mode="bilinear", align_corners=True).squeeze(1)
    images = up.unsqueeze(1).repeat(1, 3, 1, 1)
    coords, z_idx = [], []
    for z, _t in slots:
        z_val = z / max(1, Dd - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1)); z_idx.append([z_val])
    batch = {
        "images": images.unsqueeze(0).to(device).float(),
        "scanner_coords": torch.from_numpy(np.stack(coords)).unsqueeze(0).to(device),
        "z_indices": torch.tensor(z_idx, dtype=torch.float32).unsqueeze(0).to(device),
        "timesteps": slot_t.view(1, S).to(device),
        "slice_indices": slot_z.float().view(1, S).to(device),
        "phases": phases_bundle.unsqueeze(0),
        "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=device),
        "anatomy_bbox": torch.as_tensor(bbox[:6], dtype=torch.int64).view(1, 6).to(device),
    }
    return batch, z_mid


def build_1frame_contz_batch(phases_bundle, positions, bbox, seq_index, device):
    """1-frame-per-slice at TRUE continuous physical z (no snap, no collision-drop) — the regime
    the continuous-z-trained model (s20contz) expects. Each native acquired slice contributes
    EXACTLY ONE frame at its OWN fractional canonical z (`assign_canonical_z(continuous_z=True)`),
    and its content is the dense bundle linearly interpolated between the two bracketing integer
    planes — identical to the trainer's off-grid synthesis (docs/28). Slot 0 = the slice nearest
    canonical mid-depth = the swept reference/query. Same batch-key contract as build_1frame_batch;
    returns (batch, z_mid_int, ref_zfrac) — z_mid_int for display, ref_zfrac for fractional-z sweep."""
    T, Dd, H, W = phases_bundle.shape
    zmap = assign_canonical_z(positions, continuous_z=True) or [((Dd - 1) / 2.0, 0)]
    mid = (Dd - 1) / 2.0
    ref_i = int(np.argmin([abs(zf - mid) for zf, _ in zmap]))
    ref_zfrac = float(zmap[ref_i][0])
    rng = np.random.default_rng(seq_index)
    slots_z = [ref_zfrac]; slots_t = [0]                  # slot 0: swept reference (set per phase)
    for i, (zf, _s) in enumerate(zmap):
        if i == ref_i:
            continue
        slots_z.append(float(zf)); slots_t.append(int(rng.integers(T)))  # ONE random-phase frame/slice
    S = len(slots_z)
    hw = INPUT_IMG_SIZE
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    slot_t = torch.tensor(slots_t, dtype=torch.long).view(1, S).to(device)
    slot_zf = torch.tensor(slots_z, dtype=torch.float32).view(1, S).to(device)   # FRACTIONAL canon z
    # initial clean images = bundle interpolated at fractional z (bilinear through-plane blend)
    images = extract_slices_from_phases(phases_bundle.unsqueeze(0), slot_t, slot_zf)
    images = images.permute(0, 1, 4, 2, 3).contiguous() / 255.0                  # (1,S,3,hw,hw)
    coords, z_idx = [], []
    for zf in slots_z:
        z_val = zf / max(1, Dd - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1)); z_idx.append([z_val])
    batch = {
        "images": images.to(device).float(),
        "scanner_coords": torch.from_numpy(np.stack(coords)).unsqueeze(0).to(device),
        "z_indices": torch.tensor(z_idx, dtype=torch.float32).unsqueeze(0).to(device),
        "timesteps": slot_t,
        "slice_indices": slot_zf,                          # fractional z (breathing group_ids rounds it)
        "phases": phases_bundle.unsqueeze(0),
        "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=device),
        "anatomy_bbox": torch.as_tensor(bbox[:6], dtype=torch.int64).view(1, 6).to(device),
    }
    return batch, int(round(ref_zfrac)), ref_zfrac


def capture(model, phases_bundle, bbox, breathing, rcfg, regime="multiframe", clean_ref=False,
            seq_index=0, max_phases=None, positions=None):
    """Per-phase capture on the gated batch. Returns GT/RE/IN/CO (T,D,256,256), DV (T,D,256,256,3),
    per-slot applied disp rd (S,3 mm), sop/has_slot/z_mid, per-phase metrics.
    regime: 'multiframe' (S~76, all-T ref companions + 5-frame bursts, docs/28),
            '1frame' (S~#planes, gather05's trained 1-frame-per-plane budget, integer snapped z), or
            '1frame_contz' (S=#native slices, each at its TRUE fractional physical z — the regime
               the continuous-z model expects; needs `positions` = native slice scanner positions).
    clean_ref: keep slot 0 (the reference QUERY) breathing-CLEAN — only the scattered observation
               slices get breathing.
    seq_index: seeds BOTH the deterministic burst starts AND the val breathing pattern (per-plane
               respiratory phase r). Vary it per subject / pick a dynamic seed for viz."""
    T = phases_bundle.shape[0]
    ref_zf = None
    if regime == "1frame":
        batch, z_mid = build_1frame_batch(phases_bundle, bbox, seq_index, DEV)
    elif regime == "1frame_contz":
        assert positions is not None, "1frame_contz needs native slice positions"
        batch, z_mid, ref_zf = build_1frame_contz_batch(phases_bundle, positions, bbox, seq_index, DEV)
    else:
        batch, z_mid = _build_multiframe_batch(phases_bundle, bbox, FPS, seq_index, DEV)
    ref_zf = float(z_mid) if ref_zf is None else ref_zf   # integer plane for non-contz regimes
    hw = batch["images"].shape[-1]
    sc = batch["scanner_coords"][0].cpu().numpy()
    slice_z = batch["slice_indices"][0].cpu().numpy()
    sop = [int(np.argmin(np.abs(slice_z - p))) for p in range(D)]
    has_slot = [bool(np.min(np.abs(slice_z - p)) < 0.5) for p in range(D)]; has_slot[z_mid] = True
    # Optionally subsample the cardiac cycle to `max_phases` frames (linspace) — cuts BOTH the
    # per-phase forwards AND the GIF render (viz speedup); None = all T phases.
    phase_ids = (np.linspace(0, T - 1, min(T, max_phases)).round().astype(int)
                 if max_phases else np.arange(T))
    nT = len(phase_ids)
    # Reference-query content = bundle interpolated at the reference's (fractional) z, matching
    # extract_slices_from_phases' through-plane blend. For integer z (non-contz) frac=0 → exact plane.
    z0r = int(np.floor(ref_zf)); z1r = min(z0r + 1, D - 1); frr = ref_zf - z0r
    def ref_image(t):
        sl = (1.0 - frr) * phases_bundle[t, z0r] + frr * phases_bundle[t, z1r]
        up = F.interpolate(sl[None, None].float(), size=(hw, hw), mode="bilinear", align_corners=True)
        return up.repeat(1, 3, 1, 1)
    GT = np.zeros((nT, D, 256, 256), np.float32); RE = np.zeros_like(GT); IN = np.zeros_like(GT); CO = np.zeros_like(GT)
    DV = np.zeros((nT, D, 256, 256, 3), np.float32); rd = None
    S = batch["images"].shape[1]
    IN_slots = np.zeros((nT, S, 256, 256), np.float32)   # ACTUAL fed slice per slot (not 12-col collapse)
    DV_slots = np.zeros((nT, S, 256, 256, 3), np.float32)  # per-INPUT-slot Δ (mm) — DVF is per input slice
    slot_zf = slice_z.astype(np.float32).copy()          # (S,) fractional canonical z of each fed slot
    metr = {k: [] for k in _MK}
    for ti, t in enumerate(phase_ids):
        t = int(t)
        batch["timesteps"][:, 0] = t
        if breathing:
            batch = gpu_augment_batch(batch, None, DEV, respiratory_cfg=rcfg, train=False)
            if rd is None:
                rd = batch["resp_disp_mm"][0].cpu().numpy().copy()
                if clean_ref:
                    rd[0] = 0.0                         # slot 0 fed clean -> report 0 applied disp
            if clean_ref:                                # keep the reference QUERY breathing-clean
                batch["images"][:, 0] = ref_image(t)
        else:
            batch["images"][:, 0] = ref_image(t)
        batch["gt_target_volume"] = phases_bundle[t].unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
            wp = preds["world_points"][0].float().cpu().numpy()
            out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()},
                                                batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
        RE[ti] = out["V_canon"][0].float().cpu().numpy(); GT[ti] = out["V_gt"][0].float().cpu().numpy()
        CO[ti] = out["coverage"][0].float().cpu().numpy()
        for k, mk in _MK.items():
            metr[k].append(float(out[mk]) if mk in out else float("nan"))
        im = F.interpolate(batch["images"][0, :, 0][:, None], size=(256, 256),
                           mode="bilinear", align_corners=True)[:, 0].cpu().numpy()
        IN_slots[ti] = im                                # every fed slot, at its true fractional z
        for s in range(S):                               # per-INPUT-slot Δ = world_points - scanner_coords
            d = (wp[s] - sc[s]) * MM[None, None, :]
            DV_slots[ti, s] = np.stack([to256(d[..., k]) for k in range(3)], -1)
        for p in range(D):                               # 12-plane views (nearest slot) for legacy render_7row
            IN[ti, p] = im[sop[p]]
            DV[ti, p] = DV_slots[ti, sop[p]]
    if rd is None:
        rd = np.zeros((batch["images"].shape[1], 3), np.float32)
    return dict(GT=GT, RE=RE, IN=IN, CO=CO, DV=DV, rd=rd, sop=sop, has_slot=has_slot,
                z_mid=z_mid, metr=metr, IN_slots=IN_slots, slot_zf=slot_zf, DV_slots=DV_slots)


def render_7row(cap, title_prefix, path, dpi=66):
    GT, RE, IN, CO, DV = cap["GT"], cap["RE"], cap["IN"], cap["CO"], cap["DV"]
    sop, has_slot, rz = cap["sop"], cap["has_slot"], cap["z_mid"]; zbr = cap["rd"][:, 0]
    T = GT.shape[0]
    vlx, vly, vlz = [max(1.0, np.percentile(np.abs(DV[..., k]), 99)) for k in range(3)]
    covmax = float(CO.max()); covvmax = float(np.percentile(CO[CO > 0], 95)) if (CO > 0).any() else 1.0
    gv, rv, iv = [np.percentile(x, 99.5) for x in (GT, RE, IN)]; blank = np.zeros((256, 256), np.float32)
    rows = [("GT", GT, "gray", 0, gv, False), ("input", IN, "gray", 0, iv, True),
            ("recon", RE, "gray", 0, rv, False),
            (f"Dx±{vlx:.0f}", DV[..., 0], "bwr", -vlx, vlx, True),
            (f"Dy±{vly:.0f}", DV[..., 1], "bwr", -vly, vly, True),
            (f"Dz±{vlz:.0f}mm", DV[..., 2], "bwr", -vlz, vlz, True),
            (f"cov 0-{covvmax:.0f} (max {covmax:.0f})", CO, "viridis", 0, covvmax, False)]
    nr = len(rows); ycen = [0.925 - (i + 0.5) * (0.925 - 0.01) / nr for i in range(nr)]
    frames = []
    for t in range(T):
        fig, axs = plt.subplots(nr, D, figsize=(2 * D, 1.85 * nr), dpi=dpi)
        for ri, (lab, arr, cm, lo, hi, bl) in enumerate(rows):
            for p in range(D):
                img = arr[t, p] if (has_slot[p] or not bl) else blank
                axs[ri, p].imshow(img, cmap=cm, vmin=lo, vmax=hi); axs[ri, p].axis("off")
                if ri == 0:
                    zb = f"\nbr_z {zbr[sop[p]]:+.0f}mm" if has_slot[p] else "\n(no in)"
                    axs[ri, p].set_title(f"z{p}" + ("*REF" if p == rz else "") + zb, fontsize=7)
        for yc, (lab, *_) in zip(ycen, rows):
            fig.text(0.011, yc, lab, rotation=90, va="center", ha="center", fontsize=10)
        fig.suptitle(f"{title_prefix} | col title = simulated z-breathing per slice | phase {t}/{T-1}", fontsize=11)
        fig.subplots_adjust(left=0.028, right=0.997, top=0.94, bottom=0.005, wspace=0.04, hspace=0.16)
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(path, frames, duration=0.2, loop=0)
    return dict(vlx=vlx, vly=vly, vlz=vlz, covmax=covmax, covvmax=covvmax)


def render_inputstrip(cap, title_prefix, path, dpi=130):
    """7-row view that keeps PER-OUTPUT-PLANE and PER-INPUT-SLICE quantities on their OWN axes:
      GT, recon                 — per OUTPUT canonical plane (12 columns)
      input, Δx, Δy, Δz         — per INPUT slice (S columns, sorted by depth, ALIGNED under input)
      coverage                  — per OUTPUT canonical plane (12 columns)
    The DVF (Δ = world_points − scanner_coords) is defined per input slice, so it lives on the input
    strip aligned under each fed slice — NOT crammed onto the output-plane grid (docs: per-input vs
    per-output). Needs cap['IN_slots'] (nT,S,256,256), cap['DV_slots'] (nT,S,256,256,3), cap['slot_zf']."""
    GT, RE, CO = cap["GT"], cap["RE"], cap["CO"]
    INS, DVS, zf = cap["IN_slots"], cap["DV_slots"], np.asarray(cap["slot_zf"]); rd = cap["rd"]
    has_slot, rz = cap["has_slot"], cap["z_mid"]; zbr = rd[:, 0]
    T = GT.shape[0]; S = INS.shape[1]
    order = list(np.argsort(zf))                          # display slots shallow→deep
    vlx, vly, vlz = [max(1.0, np.percentile(np.abs(DVS[..., k]), 99)) for k in range(3)]
    covmax = float(CO.max()); covvmax = float(np.percentile(CO[CO > 0], 95)) if (CO > 0).any() else 1.0
    gv, rv, iv = np.percentile(GT, 99.5), np.percentile(RE, 99.5), np.percentile(INS, 99.5)
    blank = np.zeros((256, 256), np.float32)
    # rows: kind 'out' = per-output-plane (12 cols); kind 'in' = per-input-slice strip (S cols).
    # (kind, label, array, cmap, lo, hi, blank_flag)
    rows = [
        ("out", "GT", GT, "gray", 0, gv, False),
        ("out", "recon", RE, "gray", 0, rv, False),
        ("in", f"input ({S})", INS, "gray", 0, iv, False),
        ("in", f"Dx±{vlx:.0f}", DVS[..., 0], "bwr", -vlx, vlx, False),
        ("in", f"Dy±{vly:.0f}", DVS[..., 1], "bwr", -vly, vly, False),
        ("in", f"Dz±{vlz:.0f}mm", DVS[..., 2], "bwr", -vlz, vlz, False),
        ("out", f"cov 0-{covvmax:.0f} (max {covmax:.0f})", CO, "viridis", 0, covvmax, False),
    ]
    nr = len(rows); ycen = [0.925 - (i + 0.5) * (0.925 - 0.01) / nr for i in range(nr)]
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(2 * D, 1.85 * nr), dpi=dpi)
        outer = fig.add_gridspec(nr, 1, left=0.028, right=0.997, top=0.94, bottom=0.005, hspace=0.28)
        for ri, (kind, lab, arr, cm, lo, hi, bl) in enumerate(rows):
            if kind == "in":                              # per-input-slice strip (S cols, sorted)
                inner = outer[ri].subgridspec(1, S, wspace=0.04)
                for j, s in enumerate(order):
                    ax = fig.add_subplot(inner[0, j])
                    ax.imshow(arr[t, s], cmap=cm, vmin=lo, vmax=hi); ax.axis("off")
                    if lab.startswith("input"):           # label the fed slices once (top strip row)
                        ax.set_title(f"z{zf[s]:.1f}{'*REF' if s == 0 else ''}\nbr{zbr[s]:+.0f}", fontsize=7)
            else:                                         # per-output-plane (12 cols)
                inner = outer[ri].subgridspec(1, D, wspace=0.04)
                for p in range(D):
                    ax = fig.add_subplot(inner[0, p])
                    ax.imshow(arr[t, p], cmap=cm, vmin=lo, vmax=hi); ax.axis("off")
                    if ri == 0:
                        ax.set_title(f"z{p}" + ("*REF" if p == rz else ""), fontsize=7)
        for yc, (_, lab, *_ ) in zip(ycen, rows):
            fig.text(0.011, yc, lab, rotation=90, va="center", ha="center", fontsize=10)
        fig.suptitle(f"{title_prefix} | GT/recon/cov=output planes, input+DVF=fed slices at true z | phase {t}/{T-1}", fontsize=11)
        fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()); plt.close(fig)
    imageio.mimsave(path, frames, duration=0.2, loop=0)
    return dict(vlx=vlx, vly=vly, vlz=vlz, covmax=covmax, covvmax=covvmax)


def main():
    regime = sys.argv[1] if len(sys.argv) > 1 else "multiframe"
    assert regime in ("multiframe", "1frame"), regime
    clean_ref = "cleanref" in sys.argv[2:]                # keep the reference query breathing-clean
    rtag = regime + ("_cleanref" if clean_ref else "")
    print(f"loading gather05 (regime={regime}, clean_ref={clean_ref}): {CKPT}", flush=True)
    model = load_rtfb_model_reference(CKPT, refiner=False, device=DEV)
    rn = load_rcfg(); re_cfg = dataclasses.replace(rn, amplitude_mm=50.0)
    conds = [("clean", False, rn), ("normal", True, rn), ("extreme", True, re_cfg)]
    print(f"rcfg normal amp={rn.amplitude_mm}  extreme amp={re_cfg.amplitude_mm}", flush=True)

    summary = []
    for ds, subj, make_adapter in SUBJECTS:
        bundle_np, bbox = make_adapter().build_canonical_bundle()
        phases_bundle = torch.from_numpy(bundle_np).to(DEV); T = bundle_np.shape[0]
        print(f"\n=== {ds}/{subj}  T={T}  bbox={bbox.tolist()} ===", flush=True)
        for tag, breathing, rcfg in conds:
            base = os.path.join(OUT, f"{ds}_{subj}_{tag}_{rtag}_7row")
            if os.path.exists(base + ".npz"):
                print(f"  [{tag:7s}] skip (exists) -> {base}.npz", flush=True)
                continue
            cap = capture(model, phases_bundle, bbox, breathing, rcfg, regime, clean_ref)
            mag = np.linalg.norm(cap["rd"], axis=1)
            amp_mean, amp_max = float(mag.mean()), float(mag.max())
            means = {k: float(np.nanmean(v)) for k, v in cap["metr"].items()}
            alab = "clean (no breathing)" if not breathing else f"{tag} |disp| mean {amp_mean:.1f} max {amp_max:.1f}mm"
            vl = render_7row(cap, f"gather05 | {ds}/{subj} GATED [{rtag}] | {alab}", base + ".gif")
            np.savez_compressed(base + ".npz", gt=cap["GT"], recon=cap["RE"], inp=cap["IN"],
                                dvf=cap["DV"], cov=cap["CO"], has_slot=np.array(cap["has_slot"]),
                                ref_zmid=cap["z_mid"], zbr=cap["rd"][:, 0], sop=np.array(cap["sop"]),
                                applied_disp=cap["rd"], per_phase_motion=np.array(cap["metr"]["motion"]),
                                per_phase_bbox=np.array(cap["metr"]["bbox"]), per_phase_full=np.array(cap["metr"]["full"]),
                                per_phase_ssim=np.array(cap["metr"]["ssim"]))
            summary.append(dict(dataset=ds, subject=subj, condition=tag, regime=rtag, breathing=breathing,
                                applied_amp_mean_mm=amp_mean, applied_amp_max_mm=amp_max, mean=means,
                                dvf_vlims_mm=[vl["vlx"], vl["vly"], vl["vlz"]], cov_max=vl["covmax"]))
            print(f"  [{tag:7s}] applied|disp| mean={amp_mean:.1f} max={amp_max:.1f}mm | "
                  f"motion={means['motion']:.2f} bbox={means['bbox']:.2f} full={means['full']:.2f}dB "
                  f"ssim={means['ssim']:.3f} | DVF dx/dy/dz={vl['vlx']:.0f}/{vl['vly']:.0f}/{vl['vlz']:.0f}mm "
                  f"cov_max={vl['covmax']:.0f} -> {base}.gif", flush=True)
        del phases_bundle; torch.cuda.empty_cache()

    with open(os.path.join(OUT, f"metrics_{rtag}_7row.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY (gather05, 7-row, regime={rtag}) ===", flush=True)
    for r in summary:
        print(f"{r['dataset']:5s} {r['subject']:11s} {r['condition']:7s} amp={r['applied_amp_max_mm']:4.0f}mm  "
              f"motion={r['mean']['motion']:.2f} bbox={r['mean']['bbox']:.2f} full={r['mean']['full']:.2f}dB "
              f"ssim={r['mean']['ssim']:.3f}  DVF={r['dvf_vlims_mm'][0]:.0f}/{r['dvf_vlims_mm'][1]:.0f}/{r['dvf_vlims_mm'][2]:.0f}mm", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
