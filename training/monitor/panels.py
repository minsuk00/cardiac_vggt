# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The wandb figures `Monitor` logs. Each `log_*` function renders one kind of figure from
the arrays it is given and logs it under a fixed wandb key; none of them reads or writes
training state. matplotlib / wandb are imported inside the functions so importing this
module stays cheap.
"""

import logging

import torch

from training.data.preprocess import Z_HALF_MM
from training.utils.val_logging import N_FILM_PLANES, pick_planes, subject_id


def display_gamma(image, vmax: float, gamma: float = 0.7):
    """Map an MRI intensity array to [0, 1] for display only."""
    import numpy as np

    return np.clip(image / max(float(vmax), 1e-8), 0.0, 1.0) ** gamma


def _gray_slices(images):
    """(S, 3, H, W) model-input slices -> (S, H, W) gray in [0, 1] (numpy)."""
    imgs = images.detach().float().cpu()
    if imgs.min() < 0:
        imgs = (imgs + 1.0) / 2.0
    return imgs.clamp(0, 1).mean(dim=1).numpy()


def log_motion_mask_example(writer, val_ds, subj_indices, step: int):
    """Motion mask for the visual val subjects under `media_others/`.

    One row per subject; columns at 3 z-planes spanning the stack: V_gt(t=0) | motion
    magnitude (max-min over phases) | mask overlay. Data-derived only and static across
    training, so it is logged once at startup to document which voxels `val/psnr/motion` is
    computed over. vmin/vmax are per subject so small-FOV rows are not washed out.
    """
    if not writer or val_ds is None:
        return
    try:
        import wandb
        import numpy as np
        import matplotlib.pyplot as plt
        from training.metrics import compute_motion_mask, MOTION_MASK_TAU
    except ImportError:
        return

    try:
        # Drop out-of-range picks so tiny val sets (e.g. synthetic test data) still render.
        subj_indices = [i for i in subj_indices if i < len(val_ds.subjects)]
        if not subj_indices:
            return
        n_rows = len(subj_indices)

        fig, axes = plt.subplots(n_rows, 3, figsize=(18.0, 2.4 * n_rows), dpi=90)
        try:
            axes = np.atleast_2d(axes)  # (n_rows, 3) even when n_rows == 1
            for row, subj_idx in enumerate(subj_indices):
                data = val_ds.get_data(seq_index=subj_idx, img_per_seq=val_ds.num_slices)
                phases = np.asarray(data["phases"]).astype(np.float32)   # (T, D, H, W)
                phase0 = phases[0]                                       # (D, H, W)
                motion_mag = phases.max(0) - phases.min(0)              # (D, H, W)
                mask = compute_motion_mask(
                    torch.from_numpy(phases).unsqueeze(0)
                )[0].cpu().numpy()                                      # (D, H, W) bool
                bbox = np.asarray(data["anatomy_bbox"]).astype(int)
                z0, z1 = int(bbox[0]), int(bbox[1])
                # Apex/mid/base, not the mid-bbox plane: that one is the reference slot
                # (see pick_planes), and one plane cannot show the mask degenerating.
                zs = pick_planes(phase0.shape[0], 3)
                frac = float(mask[z0:z1].mean())
                vmax = max(float(phase0.max()), 1e-3)
                mmax = max(float(motion_mag.max()), 1e-3)

                ax = axes[row]
                phase0_strip = np.concatenate([phase0[z] for z in zs], axis=1)
                ax[0].imshow(display_gamma(phase0_strip, vmax),
                             cmap="gray", vmin=0, vmax=1)
                ax[0].set_title(f"subj {subj_idx}: V_gt t=0 (z={list(zs)})", fontsize=9)
                ax[1].imshow(np.concatenate([motion_mag[z] for z in zs], axis=1),
                             cmap="magma", vmin=0, vmax=mmax)
                ax[1].set_title("motion = max-min", fontsize=9)
                ax[2].imshow(display_gamma(phase0_strip, vmax),
                             cmap="gray", vmin=0, vmax=1)
                m3 = np.concatenate([mask[z] for z in zs], axis=1)
                overlay = np.zeros((*m3.shape, 4))
                overlay[m3] = [1, 0, 0, 0.45]
                ax[2].imshow(overlay)
                ax[2].set_title(f"mask tau={MOTION_MASK_TAU} ({frac*100:.1f}% of bbox)", fontsize=9)
                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
            fig.suptitle("Motion mask (val subjects; apex | mid | base)", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            writer.log("media_others/val_motion_mask_example", wandb.Image(fig), step)
        finally:
            plt.close(fig)
    except Exception as e:
        logging.warning(f"motion mask example log failed (ignored): {e}")


def log_cardiac_cycle_gif(writer, gt_frames, canon_frames, vmax: float, val_ds, subj_idx: int,
                          note: str, step: int):
    """Animated GIF over the cardiac cycle: per frame, top row = V_gt's N_FILM_PLANES planes,
    bottom row = V_canon's. Each stored frame is (N*h, W) — the planes stacked vertically —
    and is re-tiled to 2 x N. wandb.Video (moviepy) needs 3-channel RGB."""
    import numpy as np

    def _tile_2xn(gt_stack, model_stack, n=N_FILM_PLANES):
        h = gt_stack.shape[0] // n; W = gt_stack.shape[1]
        def _row(stack):                                    # (n·h, W) -> (h, n·W)
            planes = stack.reshape(n, h, W)
            return np.concatenate([planes[i] for i in range(n)], axis=1)
        grid = np.concatenate([_row(gt_stack), _row(model_stack)], axis=0)   # (2h, n·W)
        g = (display_gamma(grid, vmax) * 255.0).astype(np.uint8)
        return np.stack([g, g, g], axis=0)                  # (3, 2h, n·W)

    try:
        import wandb
        frames = np.stack([_tile_2xn(gt_frames[t], canon_frames[t])
                           for t in range(len(gt_frames))], axis=0)   # (T, 3, 2h, n·W)
        sid = subject_id(val_ds.subjects[subj_idx])
        writer.log(
            f"media_filmstrip/Val_Visuals_subj{subj_idx}_{sid}_cardiac_cycle_gif",
            wandb.Video(frames, fps=4, format="gif",
                        caption=f"{sid} — step={step} — rows: V_gt (top) / V_canon (bottom); "
                                f"cols: {N_FILM_PLANES} planes evenly spanning z=0..D-1"
                                f"{note}"),
            step,
        )
    except Exception as e:
        logging.warning(f"cardiac cycle gif log failed (ignored): {e}")


def log_augmentation(writer, orig_images, aug_images, caption: str, step: int) -> None:
    """Before/after panel of the train-time augmentation for one subject: top row = the
    original input slices, bottom row = the augmented slices the model actually trains on."""
    import wandb
    import matplotlib.pyplot as plt
    from matplotlib import gridspec as _gs

    def _gray(t):  # (B,S,3,H,W) or (S,3,H,W) -> (S,H,W) in [0,1]
        t = t[0] if t.dim() == 5 else t
        return _gray_slices(t)

    orig, aug = _gray(orig_images), _gray(aug_images)
    S = min(orig.shape[0], aug.shape[0], 6)   # a few slots is enough
    fig = plt.figure(figsize=(1.6 * S + 0.4, 3.6), dpi=90)
    try:
        gs = _gs.GridSpec(2, S, wspace=0.04, hspace=0.12)
        fig.suptitle("Data augmentation -- original (top) vs augmented (bottom)", fontsize=8)
        for s in range(S):
            ax0 = fig.add_subplot(gs[0, s])
            ax0.imshow(display_gamma(orig[s], 1.0), cmap="gray", vmin=0, vmax=1)
            ax0.set_xticks([]); ax0.set_yticks([])
            if s == 0:
                ax0.set_ylabel("original", fontsize=8)
            ax1 = fig.add_subplot(gs[1, s])
            ax1.imshow(display_gamma(aug[s], 1.0), cmap="gray", vmin=0, vmax=1)
            ax1.set_xticks([]); ax1.set_yticks([])
            if s == 0:
                ax1.set_ylabel("augmented", fontsize=8)
        writer.log("media_others/Train_Visuals_Augmentation", wandb.Image(fig, caption=caption), step)
    finally:
        plt.close(fig)


def log_volume_and_dvf(writer, batch: dict, name: str, step: int, caption: str,
                       group: str = "media_others"):
    """Two figures under `{group}/`:
      {name}_Volume : 4 rows x max(S,D) cols — input slices, V_gt, V_canon, signed diff (per z).
      {name}_DVF    : 4 rows x S cols       — input intensity + Δx/Δy/Δz (mm) per slot.
    """
    if not writer:
        return
    try:
        import wandb
        import matplotlib.pyplot as plt
        from matplotlib import gridspec as _gs
        import numpy as np
    except ImportError:
        return

    def _slot_title(s, t_picks, z_picks, resp_r, resp_dmag):
        ttl = f"t={int(t_picks[s])}, z={int(z_picks[s])}"
        if resp_r is not None:
            ttl += f"\nr={resp_r[s]:.2f} |d|={resp_dmag[s]:.0f}mm"
        return ttl

    t_picks = batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None
    z_picks = batch["slice_indices"][0].cpu().numpy() if "slice_indices" in batch else None
    # Per-slot breathing (phase r + displacement |d| mm), present only when breathing is on.
    resp_dmag = (batch["resp_disp_mm"][0].float().norm(dim=-1).cpu().numpy()
                 if "resp_disp_mm" in batch else None)
    resp_r = batch["resp_r"][0].float().cpu().numpy() if "resp_r" in batch else None

    # ── Volume figure (per-z grid) ─────────────────────────────────────
    if "V_canon" in batch and "V_gt" in batch:
        V_canon = batch["V_canon"][0].detach().float().cpu().numpy()   # (D, H, W)
        V_gt = batch["V_gt"][0].detach().float().cpu().numpy()
        D, _, _ = V_canon.shape
        imgs = _gray_slices(batch["images"][0])                         # (S, H, W)
        S = imgs.shape[0]
        diff = V_canon - V_gt
        v_vmax = float(max(V_canon.max(), V_gt.max(), 1e-3))
        ERR = 0.1  # signed-diff color range ±ERR for the V_canon-V_gt row
        n_cols = max(S, D)

        fig = plt.figure(figsize=(1.6 * n_cols + 1.6, 7.5), dpi=90)
        try:
            gs = _gs.GridSpec(4, n_cols + 1, width_ratios=[1.0] * n_cols + [0.05], wspace=0.04, hspace=0.18)
            fig.suptitle(f"Volumes — {caption}", fontsize=8)

            # Row 0: input slices
            for s in range(S):
                ax = fig.add_subplot(gs[0, s])
                ax.imshow(display_gamma(imgs[s], 1.0), cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if t_picks is not None and z_picks is not None:
                    ax.set_title(_slot_title(s, t_picks, z_picks, resp_r, resp_dmag), fontsize=7)
                if s == 0:
                    ax.set_ylabel("input slice", fontsize=8)
            for s in range(S, n_cols):
                fig.add_subplot(gs[0, s]).axis("off")
            fig.add_subplot(gs[0, n_cols]).axis("off")

            def _vol_row(r, vol, cmap, vmin, vmax, ylabel, show_titles=False,
                         gamma=False):
                last_im = None
                for d in range(D):
                    ax = fig.add_subplot(gs[r, d])
                    shown = display_gamma(vol[d], vmax) if gamma else vol[d]
                    last_im = ax.imshow(shown, cmap=cmap,
                                        vmin=(0 if gamma else vmin),
                                        vmax=(1 if gamma else vmax))
                    ax.set_xticks([]); ax.set_yticks([])
                    if show_titles:
                        ax.set_title(f"z={d}", fontsize=7)
                    if d == 0:
                        ax.set_ylabel(ylabel, fontsize=8)
                for d in range(D, n_cols):
                    fig.add_subplot(gs[r, d]).axis("off")
                plt.colorbar(last_im, cax=fig.add_subplot(gs[r, n_cols]))

            _vol_row(1, V_gt,    "gray",   0,     v_vmax, "V_gt", show_titles=True, gamma=True)
            _vol_row(2, V_canon, "gray",   0,     v_vmax, "V_canon", gamma=True)
            _vol_row(3, diff,    "RdBu_r", -ERR,  ERR,    f"V_canon-V_gt\n(±{ERR})")

            writer.log(f"{group}/{name}_Volume", wandb.Image(fig, caption=caption), step)
        finally:
            plt.close(fig)

    # ── DVF figure (per-slot Δx/Δy/Δz) ─────────────────────────────────
    # The predicted Δ is recovered as (pred_world_points - scanner_coords).
    if "pred_world_points" in batch and "scanner_coords" in batch:
        pred_dvf = (batch["pred_world_points"][0] - batch["scanner_coords"][0]).detach().float().cpu().numpy()
        imgs = _gray_slices(batch["images"][0])
        S = imgs.shape[0]
        p50 = float(np.percentile(np.abs(pred_dvf), 50))
        p95 = float(np.percentile(np.abs(pred_dvf), 95))
        p99 = float(np.percentile(np.abs(pred_dvf), 99))
        # Normalized [-1,1] residual → physical mm (align_corners: 1 norm unit =
        # (size-1)/2 * spacing). In-plane and through-plane mm/norm differ a lot, so the
        # colorbars are per axis.
        IN_PLANE_MM = (256 - 1) / 2.0 * 1.4      # ≈178.5 mm per norm unit (Δx, Δy)
        THROUGH_MM = Z_HALF_MM                   # z_norm = z_mm / Z_HALF_MM (docs/58), same for every subject
        IN_PLANE_R = 15.0                         # in-plane colorbar half-range (mm)
        THROUGH_R = 25.0                          # through-plane colorbar half-range (mm)

        fig = plt.figure(figsize=(1.6 * S + 1.6, 7.5), dpi=90)
        try:
            gs = _gs.GridSpec(4, S + 1, width_ratios=[1.0] * S + [0.05], wspace=0.04, hspace=0.18)
            fig.suptitle(
                f"DVF — {caption}    |Δ|(norm) p50={p50:.3f} p95={p95:.3f} p99={p99:.3f}",
                fontsize=8,
            )
            rows = [
                ("input intensity", display_gamma(imgs, 1.0),    "gray",   0,           1.0,        True),
                ("Δx (mm)",         pred_dvf[..., 0] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
                ("Δy (mm)",         pred_dvf[..., 1] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
                ("Δz (mm)",         pred_dvf[..., 2] * THROUGH_MM,  "RdBu_r", -THROUGH_R,  THROUGH_R,  False),
            ]
            for r, (lbl, data, cmap, vmin, vmax, is_top) in enumerate(rows):
                last_im = None
                for s in range(S):
                    ax = fig.add_subplot(gs[r, s])
                    last_im = ax.imshow(data[s], cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([])
                    if is_top and t_picks is not None and z_picks is not None:
                        ax.set_title(_slot_title(s, t_picks, z_picks, resp_r, resp_dmag), fontsize=7)
                    if s == 0:
                        ax.set_ylabel(lbl, fontsize=8)
                plt.colorbar(last_im, cax=fig.add_subplot(gs[r, S]))

            writer.log(f"{group}/{name}_DVF", wandb.Image(fig, caption=caption), step)
        finally:
            plt.close(fig)


def log_ed_es_panels(writer, stash, subj_indices, val_ds, step: int) -> None:
    """One 6-row ED-vs-ES figure per subject that has both phases stashed:
    rows = input / V_gt / V_canon at ED, then the same at ES; columns = the S input slices
    (input rows) or the D z-planes (volume rows). Per z, not the mid plane (the reference
    slot, where the model does nothing): contraction reads as the ED cavities shrinking to
    the ES cavities across planes."""
    if not writer:
        return
    try:
        import wandb
        import matplotlib.pyplot as plt
        from matplotlib import gridspec as _gs
    except ImportError:
        return
    for subj_idx in subj_indices:
        rec = stash.get(subj_idx)
        if not rec or "ED" not in rec or "ES" not in rec:
            continue
        fig = None
        try:
            ed, es = rec["ED"], rec["ES"]
            D = ed["V_gt"].shape[0]
            S = ed["images"].shape[0]
            n_cols = max(S, D)
            vmax = float(max(ed["V_gt"].max(), ed["V_canon"].max(),
                             es["V_gt"].max(), es["V_canon"].max(), 1e-3))
            # (row-label, kind, data, phase-record) — kind "in"=S input slices, "vol"=D z-planes.
            rows = [
                ("input ED", "in", ed["images"], ed),
                ("V_gt ED", "vol", ed["V_gt"], ed),
                ("V_canon ED", "vol", ed["V_canon"], ed),
                ("input ES", "in", es["images"], es),
                ("V_gt ES", "vol", es["V_gt"], es),
                ("V_canon ES", "vol", es["V_canon"], es),
            ]
            fig = plt.figure(figsize=(1.1 * n_cols + 1.4, 1.5 * len(rows) + 0.8), dpi=90)
            gs = _gs.GridSpec(len(rows), n_cols, wspace=0.04, hspace=0.22)
            fig.suptitle(f"ED vs ES — val subj {subj_idx} (ED t={ed['t']}, ES t={es['t']}) — step={step}",
                         fontsize=9)
            for r, (label, kind, data, prec) in enumerate(rows):
                ncol = S if kind == "in" else D
                ts, zs = prec.get("timesteps"), prec.get("slices")
                for c in range(ncol):
                    ax = fig.add_subplot(gs[r, c])
                    vm = 1.0 if kind == "in" else vmax
                    ax.imshow(display_gamma(data[c], vm), cmap="gray", vmin=0, vmax=1)
                    ax.set_xticks([]); ax.set_yticks([])
                    if c == 0:
                        ax.set_ylabel(label, fontsize=8)
                    # (t,z) on both input rows (different samples), z on the two V_gt rows.
                    if kind == "in" and ts is not None and zs is not None:
                        ax.set_title(f"t{int(ts[c])} z{int(zs[c])}", fontsize=6)
                    elif kind == "vol" and r in (1, 4):
                        ax.set_title(f"z{c}", fontsize=6)
                for c in range(ncol, n_cols):
                    fig.add_subplot(gs[r, c]).axis("off")
            sid = subject_id(val_ds.subjects[subj_idx]) if val_ds is not None else None
            writer.log(
                f"media_val_ED_ES/Val_Visuals_subj{subj_idx}_{sid or 'unknown'}_ED_ES",
                wandb.Image(fig, caption=f"{sid} — per-z (mid plane = reference slot)"), step)
        except Exception as e:
            logging.warning(f"[ed/es] panel subj {subj_idx} failed (ignored): {e}")
        finally:
            if fig is not None:
                plt.close(fig)


def log_lookup(writer, batch: dict, name: str, step: int, caption: str,
               group: str = "media_others", reference_slot: bool = False):
    """Round-trip (analysis-by-synthesis) panel, val only.

    For up to 4 input slices at distinct z-depths, sample V_canon AND V_gt back at the
    predicted per-pixel coords p = pred_world_points and show them beside the input slice
    and |V_canon−V_gt|@p. col1 ≈ col2 by construction (their gap is the splat's coverage
    blur); col2 ≈ col3 by training (their gap is the reconstruction error). With
    `reference_slot`, slot 0 (Δ≈0, phase-matched) is pinned as the first row as a control.
    """
    needed = ("V_canon", "V_gt", "pred_world_points", "images", "slice_indices")
    if not writer or any(k not in batch for k in needed):
        return
    try:
        import wandb
        import matplotlib.pyplot as plt
        from matplotlib import gridspec as _gs
        import numpy as np
        from vggt.utils.splat import sample_volume
    except ImportError:
        return

    V_canon = batch["V_canon"][0].detach().float()                    # (D, H, W)
    V_gt = batch["V_gt"][0].detach().float()
    wp = batch["pred_world_points"][0].detach().float()               # (S, Hs, Ws, 3)
    imgs = _gray_slices(batch["images"][0])                           # (S, Hs, Ws)
    z_picks = batch["slice_indices"][0].cpu().numpy()                 # (S,)
    t_picks = batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None
    S, Hs, Ws = imgs.shape

    present = np.unique(z_picks)
    n_pick = min(4, len(present))
    zsel = present[np.linspace(0, len(present) - 1, n_pick).round().astype(int)]
    slots = [int(np.where(z_picks == zt)[0][0]) for zt in zsel]       # first slot at each z
    ref_slot = 0 if reference_slot else None
    if ref_slot is not None:                                          # pin reference first, ≤4 total
        slots = ([ref_slot] + [s for s in slots if s != ref_slot])[:4]

    vmax = float(max(V_canon.max().item(), V_gt.max().item(), 1e-3))
    ERR = 0.1
    dev = V_canon.device
    z_scale = float(batch["z_scale"].reshape(-1)[0])

    fig = plt.figure(figsize=(4 * 2.6, len(slots) * 2.6 + 0.6), dpi=110)
    try:
        gs = _gs.GridSpec(len(slots), 4, wspace=0.06, hspace=0.14)
        fig.suptitle(f"Lookup (round-trip @pred) — {caption}", fontsize=8)
        col_titles = ["input I", "V_canon @ pred", "V_gt @ pred", "|V_canon−V_gt| @ pred"]

        for r, s in enumerate(slots):
            pos = wp[s].reshape(1, -1, 3).to(dev)                      # (1, Hs*Ws, 3)
            rc = sample_volume(V_canon.unsqueeze(0), pos, z_scale).reshape(Hs, Ws).cpu().numpy()
            rg = sample_volume(V_gt.unsqueeze(0), pos, z_scale).reshape(Hs, Ws).cpu().numpy()
            err = np.abs(rc - rg)
            zt = int(z_picks[s]); tt = int(t_picks[s]) if t_picks is not None else -1
            is_ref = (ref_slot is not None and s == ref_slot)
            cells = [
                (display_gamma(imgs[s], 1.0), "gray",  0, 1.0),
                (display_gamma(rc, vmax),     "gray",  0, 1.0),
                (display_gamma(rg, vmax),     "gray",  0, 1.0),
                (err,                         "magma", 0, ERR),
            ]
            for c, (data, cmap, vmin, vm) in enumerate(cells):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vm)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(col_titles[c], fontsize=7)
                if c == 0:
                    lbl = ("REF " if is_ref else "") + f"slot{s}\nt={tt} z={zt}"
                    ax.set_ylabel(lbl, fontsize=7)

        writer.log(f"{group}/{name}_Lookup", wandb.Image(fig, caption=caption), step)
    finally:
        plt.close(fig)
