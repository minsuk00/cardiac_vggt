#!/usr/bin/env python3
"""Native-z identity-splat gate (docs/58 §8.1g).

Fault-injected acceptance gate for the native-z refactor: for real pooled subjects
spanning 5/8/10/12mm pitch, splat every plane of the REAL `gt_target_volume` back at
its own EXACT canonical position (Delta=0) using the REAL production `dz_mm`/`z_scale`
(pulled straight from `MRIDataset`), and require near-exact recovery.

Built directly at the 256x256 canonical resolution (NOT through the 518x518 DINOv2
upsample) so this isolates z-axis correctness specifically. Routing through the real
518-px input pipeline instead measures a DIFFERENT, pre-existing, already-accepted
lossy step (the 256->518->256 bilinear round trip needed for the DINOv2 patch
embedding) that swamps the z-axis signal with ~35-45 dB of unrelated noise -- verified
directly: a single fixed z-plane (z_scale has NO effect there) already shows the same
~39 dB ceiling. See tools/probe_zgrid_alignment.py for the same z-only methodology this
gate reuses.

The gate is deliberately run TWICE:
  1. FAULT-INJECTED: z_scale is corrupted on purpose. Must FAIL (low PSNR). A gate
     that has never been shown to fail on broken input is not proof of anything.
  2. REAL: z_scale from the actual pipeline. Must PASS (>= 100 dB on every subject).

Usage:
    PYTHONPATH=training:. micromamba run -n svr python tools/gate_native_z_identity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from data.datasets.mri_dataset import MRIDataset
from vggt.utils.splat import splat_to_volume

PASS_THRESHOLD_DB = 100.0

# ~30 real subjects spanning the pool's pitch range (5/8/10/12mm), by (data_root, subj_id).
SUBJECTS = [
    ("scratch/data/CMRxRecon2024/Cine_combined", f"CMRx24_Test_P00{i}") for i in range(1, 6)
] + [
    ("scratch/data/CMRxRecon2025/Cine_combined", "CMRx25_R1test_Center012_Philips_30T_IngeniaCX_P001"),
    ("scratch/data/CMRxRecon2025/Cine_combined", "CMRx25_R1test_Center012_Philips_30T_IngeniaCX_P002"),
    ("scratch/data/CMRxRecon2025/Cine_combined", "CMRx25_R1test_Center006_Siemens_30T_Prisma_P023"),
    ("scratch/data/CMRxRecon2025/Cine_combined", "CMRx25_R2test_Center001_Siemens_30T_Vida_P013"),
] + [
    ("scratch/data/ACDC_sax", f"ACDC_patient{i:03d}") for i in
    [1, 2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
] + [
    ("scratch/data/MNMs_sax", sid) for sid in
    ["MNMs_A9C5P4", "MNMs_D4N6W6", "MNMs_E4I9O7", "MNMs_E6H0V9", "MNMs_J9L4S2",
     "MNMs_E9L4N2", "MNMs_M2P1R1", "MNMs_G7S6V0"]
]


def _make_dataset(data_root, subj_id):
    split_path = f"/tmp/_gate_split_{os.getpid()}.txt"
    with open(split_path, "w") as f:
        f.write(f"[train]\n{subj_id}\n[val]\n{subj_id}\n")
    ds = MRIDataset(
        common_conf=None, data_root=data_root, split="train", split_file=split_path,
        mode="static", one_frame_per_slice=True, reference_slot=False,
    )
    os.remove(split_path)
    return ds


def _identity_psnr(item, z_scale_override=None):
    """Splat EVERY canonical plane at its own exact z-position (Delta=0, 256x256
    resolution -- no 518px upsample) and compare to gt_target_volume."""
    gt = torch.from_numpy(item["gt_target_volume"]).float()  # (D, H, W)
    D, H, W = gt.shape
    dz = float(item["dz_mm"][0])
    z_scale = z_scale_override if z_scale_override is not None else float(item["z_scale"][0])

    py, px = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    x_norm = (px.float() / (W - 1)) * 2.0 - 1.0
    y_norm = (py.float() / (H - 1)) * 2.0 - 1.0

    pos_list, inten_list = [], []
    # z_norm must match mri_dataset.py's own formula exactly: (k-(D-1)/2)*dz/Z_HALF_MM.
    # Recover Z_HALF_MM from item['z_scale'] (= Z_HALF_MM/dz) rather than hardcoding it.
    Z_HALF_MM = float(item["z_scale"][0]) * dz
    for k in range(D):
        z_norm = (k - (D - 1) / 2.0) * dz / Z_HALF_MM
        pos_list.append(torch.stack([x_norm, y_norm, torch.full_like(x_norm, z_norm)], dim=-1).reshape(-1, 3))
        inten_list.append(gt[k].reshape(-1))
    pos = torch.cat(pos_list, dim=0).unsqueeze(0)      # (1, D*H*W, 3)
    inten = torch.cat(inten_list, dim=0).unsqueeze(0)  # (1, D*H*W)
    weight = (inten > 1e-3).float()

    V, cov = splat_to_volume(pos, inten, (D, H, W), z_scale, weight=weight)
    m = (cov[0] > 0.5) & (gt > 1e-3)
    if not bool(m.any()):
        return None
    mse = ((V[0] - gt) ** 2)[m].mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def run(fault_inject: bool) -> bool:
    label = "FAULT-INJECTED (z_scale deliberately wrong)" if fault_inject else "REAL (production z_scale)"
    print(f"\n=== {label} ===")
    all_pass = True
    n_ok = 0
    for data_root, subj_id in SUBJECTS:
        try:
            ds = _make_dataset(data_root, subj_id)
            if not ds.subjects:
                print(f"  {subj_id:55s} SKIP (not found)")
                continue
            item = ds.get_data(seq_index=0)
            dz = float(item["dz_mm"][0])
            D = item["gt_target_volume"].shape[0]
            override = (float(item["z_scale"][0]) * 0.4) if fault_inject else None
            psnr = _identity_psnr(item, z_scale_override=override)
            if psnr is None:
                print(f"  {subj_id:55s} SKIP (no covered anatomy)")
                continue
            ok = psnr >= PASS_THRESHOLD_DB
            all_pass &= ok
            n_ok += int(ok)
            status = "PASS" if ok else "FAIL"
            print(f"  {subj_id:55s} dz={dz:5.2f}mm D={D:3d}  identity={psnr:7.2f} dB  [{status}]")
        except Exception as e:
            print(f"  {subj_id:55s} ERROR: {e}")
            all_pass = False
    print(f"  -> {n_ok} passed >= {PASS_THRESHOLD_DB} dB")
    return all_pass


if __name__ == "__main__":
    fault_all_pass = run(fault_inject=True)
    if fault_all_pass:
        print("\n*** GATE INVALID: fault-injected run did not fail. The gate proves nothing. ***")
        sys.exit(1)
    print("\nFault injection correctly FAILED the gate (as required).")

    real_all_pass = run(fault_inject=False)
    if not real_all_pass:
        print("\n*** GATE FAILED on real z_scale. ***")
        sys.exit(1)
    print(f"\n*** GATE PASSED: every subject >= {PASS_THRESHOLD_DB} dB identity recovery. ***")
