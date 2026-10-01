"""Training-episode generator for the per-slice frame selector (docs/128).

Gated cine + continuous breathing. One EPISODE = one training subject:
  * every candidate plane z offers its T=12 gated cine phases as frames (cyclic order from a random
    start), each breathed frame-wise like the eval bundles: one breathing direction + depth per
    subject, per-plane start phase r0, r advancing DT/T_BREATH per frame (build_af_bundle.simulate);
  * the reference is plane `ref_plane` (the dataset's slot-0 plane, z_mid of the anatomy bbox) at a
    random target phase, breathed at a random frame of its own breathing trace;
  * frames are resliced + resized to the model resolution by `extract_slices_with_respiratory_vec`
    (the trainer's own breathing core). `render_frames` also renders fractional positions (blend
    of the two bracketing phases, as build_af_bundle.render_phase) -- used only by
    tools/framesel_check_data.py to reproduce the eval bundles' pixels.

Labels for every candidate frame are its circular phase distance to the reference's phase.
The selector scores each frame independently, so frame order never matters.
"""
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
import build_af_bundle as afb                                                     # noqa: E402
from data.gpu_aug import _resize_to_model_res                                     # noqa: E402
from data.respiratory import (_build_disp_dhw, _rotate_disp, lujan_displacement,  # noqa: E402
                              extract_slices_with_respiratory_vec)

T = afb.T
INPLANE_MM = afb.INPLANE_MM


def circ_dist(a, b, period=T):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % period
    return np.minimum(d, period - d)


def sample_breathing(D, n_frames, resp_cfg, rng):
    """(D, n_frames, 3) mm frame-wise displacement, sampled like the trainer's group_by_burst
    draw (one depth + one tilted direction per subject, per-plane start phase), advanced
    frame-wise like build_af_bundle.simulate."""
    A = max(resp_cfg["amplitude_mm"] + rng.uniform(-1, 1) * resp_cfg["amplitude_jitter"], 0.0)
    unit = _build_disp_dhw(torch.ones(1, 1), torch.full((1, 1), float(resp_cfg["ap_ratio"])),
                           resp_cfg["ap_axis"])                                    # (1,1,3) per unit SI
    lo, hi = math.radians(resp_cfg["tilt_min_deg"]), math.radians(resp_cfg["tilt_max_deg"])
    theta = torch.tensor([[lo + rng.uniform() * (hi - lo)]])
    phi = torch.tensor([[rng.uniform() * 2 * math.pi]])
    unit = _rotate_disp(unit, theta, phi)[0, 0].numpy()                            # (3,)
    r0 = rng.uniform(size=D)
    r = (r0[:, None] + np.arange(n_frames)[None, :] * afb.DT / afb.T_BREATH) % 1.0  # (D, F)
    si = lujan_displacement(torch.from_numpy(r), A, n=int(resp_cfg["cos2n"])).numpy()
    return si[..., None] * unit[None, None, :]


def render_frames(phases, planes, pos, disp, dz, out_size):
    """Render frames on the GPU.

    phases: (T, D, H, W) float tensor (canonical cine, [0,1]) on device
    planes: (N,) int plane per frame;  pos: (N,) cardiac position;  disp: (N, 3) mm
    -> (N, 3, out_size, out_size) model images in [0, 1] (the `images` contract), built like the
    eval path (run_vggt `_extract`): native-resolution slice, clamped to [0,1], then
    `_resize_to_model_res`.
    A fractional position is the blend of the two bracketing phases (render_phase); blend-then-
    reslice == reslice-then-blend since grid_sample is linear in its input. Integer positions
    reduce to a single phase exactly (weight 0 on the second).
    """
    dev = phases.device
    p = np.asarray(pos, float) % T
    lo = np.floor(p).astype(np.int64)
    w = torch.from_numpy(p - lo).float().to(dev).view(1, -1, 1, 1, 1)
    z = torch.as_tensor(np.asarray(planes), dtype=torch.long, device=dev)[None]
    d = torch.as_tensor(np.asarray(disp), dtype=torch.float32, device=dev)[None]
    spacing = (float(dz), INPLANE_MM, INPLANE_MM)
    ph = phases[None]
    H = phases.shape[-1]
    a = extract_slices_with_respiratory_vec(ph, torch.as_tensor(lo % T, device=dev)[None], z, d,
                                            spacing, out_size=H)
    b = extract_slices_with_respiratory_vec(ph, torch.as_tensor((lo + 1) % T, device=dev)[None], z,
                                            d, spacing, out_size=H)
    native = (a * (1 - w) + b * w)[..., 0] / 255.0         # (1, N, H, W) in [0, 1]
    return _resize_to_model_res(native, out_size)[0]       # (N, 3, R, R)


def make_episode(sample, resp_cfg, rng, device, out_size=518, n_planes=3):
    """One gated-cine episode from one `MRIDataset.get_data` sample.

    Returns dict:
      ref_img (1,3,R,R), ref_plane_img (T,3,R,R) the reference plane's whole window,
      ref_pos int phase, ref_z float (physical z_norm of the ref plane)
      cand_img (P, T, 3, R, R), cand_pos (P, T), cand_z (P,) z_norm, label_dist (P, T) circ dist
    """
    phases = torch.as_tensor(np.asarray(sample["phases"]), dtype=torch.float32, device=device)
    D = phases.shape[1]
    dz = float(np.asarray(sample["dz_mm"]).reshape(-1)[0])
    zs = [int(round(float(v))) for v in sample["slice_indices"]]
    znorm = {z: float(np.asarray(v).reshape(-1)[0]) for z, v in zip(zs, sample["z_indices"])}
    ref_z, others = zs[0], zs[1:]
    planes = list(rng.choice(others, size=min(n_planes, len(others)), replace=False))

    disp = sample_breathing(D, T, resp_cfg, rng)                       # (D, T frames, 3)
    # Every plane, the reference one included, is a frame window over its 12 gated phases from a
    # random start; the reference is one random frame of the reference plane's own window.
    pos = np.stack([(int(rng.integers(T)) + np.arange(T)) % T for _ in [ref_z] + planes])  # (1+P, T)
    f_ref = int(rng.integers(T))
    ref_pos = int(pos[0, f_ref])
    all_planes = np.repeat([ref_z] + planes, T)
    all_disp = disp[np.asarray([ref_z] + planes)].reshape(-1, 3)
    imgs = render_frames(phases, all_planes, pos.ravel(), all_disp, dz, out_size)
    imgs = imgs.view(1 + len(planes), T, *imgs.shape[1:])
    pos = pos[1:]
    return {
        "ref_img": imgs[0, f_ref:f_ref + 1], "ref_plane_img": imgs[0], "ref_idx": f_ref,
        "ref_pos": ref_pos,
        "ref_z": znorm[ref_z],
        "cand_img": imgs[1:],
        "cand_pos": pos, "cand_z": np.array([znorm[z] for z in planes]),
        "label_dist": circ_dist(pos, ref_pos),
    }
