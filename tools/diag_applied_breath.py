"""What is the actual per-slot APPLIED breathing displacement distribution (deterministic val
draws), and how much does the direction-tilt reduce the through-plane (D) component vs the raw SI?
CPU-only, no model. Answers: does applied d_D really reach ~24 mm, or is it capped lower?
"""
import os, sys
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "training"))
from data.respiratory import RespiratoryConfig, sample_resp_disp, sample_displacements

# mri_volume.yaml respiratory defaults
cfg = RespiratoryConfig(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35,
                        ap_axis="H", direction_jitter_deg=30.0, per_slot=True)
dev = torch.device("cpu")
S = 12
dD, dSI_raw, rs = [], [], []
for seq in range(30):
    seq_t = torch.tensor([[seq]], dtype=torch.int64)
    disp, r = sample_resp_disp(1, S, cfg, dev, train=False, seq_index=seq_t)  # (1,S,3),(1,S)
    dD.append(disp[0, :, 0].numpy())          # post-tilt through-plane (what we plotted)
    rs.append(r[0].numpy())
    # pre-tilt raw SI for the SAME seed (re-seed identically)
    g = torch.Generator(device=dev).manual_seed(int(seq))
    d_si, d_ap, r2 = sample_displacements(1, S, cfg, dev, generator=g)
    dSI_raw.append(d_si[0].numpy())

dD = np.concatenate(dD); dSI = np.concatenate(dSI_raw); rs = np.concatenate(rs)
A_max = cfg.amplitude_mm + cfg.amplitude_jitter
print(f"amplitude A in [{cfg.amplitude_mm-cfg.amplitude_jitter:.0f}, {A_max:.0f}] mm; "
      f"d_si = A*sin(pi*r)^{2*cfg.cos2n}; tilt up to {cfg.direction_jitter_deg:.0f} deg\n")
for name, v in [("raw SI  d_si (pre-tilt)", dSI), ("through-plane d_D (post-tilt, plotted)", np.abs(dD))]:
    print(f"{name:42s}  max={v.max():5.1f}  p99={np.percentile(v,99):5.1f}  "
          f"p95={np.percentile(v,95):5.1f}  p50={np.percentile(v,50):4.1f}  mean={v.mean():4.1f}")
print(f"\nfraction of slots with d_si > 12mm: {(dSI>12).mean():.2f}   > 20mm: {(dSI>20).mean():.2f}")
print(f"fraction of slots with |d_D| > 12mm: {(np.abs(dD)>12).mean():.2f}  > 20mm: {(np.abs(dD)>20).mean():.2f}")
print(f"E[sin(pi r)^{2*cfg.cos2n}] over slots = {(np.sin(np.pi*rs)**(2*cfg.cos2n)).mean():.3f} "
      f"(so typical d_si is a small fraction of A)")
