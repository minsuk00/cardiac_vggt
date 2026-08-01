"""ACDC-gated STEP 1 — build the frozen breathing bundle for one ACDC patient.

Mirror of scratch/eval/{miitt,ocmr}/build_inputs.py: ECG-gated breath-hold SAX cine is clean GT;
simulate the trainer's respiratory motion on it; VGGT reconstructs; score PSNR/SSIM/NCC + resp_diag
vs the SAME VGGT-identical canonical GT. ACDC is the PATHOLOGY OOD cohort (5 classes: DCM/HCM/MINF/
NOR/RV) — the widest EF range for the EF endpoint.

Source = `scratch/data/ACDC/{training,testing}/patientXXX/`:
  patientXXX_4d.nii.gz (X,Y,Z,T), heart_seg.nii.gz (Task114 4D), heart_roi.nii.gz, Info.cfg (ED/ES/Group).
Loaded via inference.adapters.acdc.ACDCGatedAdapter (reorients cine to LPS, per-patient spacing).

ACDC-SPECIFIC (the LPS trap — CLAUDE.md "ALL data must be LPS"): the adapter reorients the CINE to
LPS, so heart_roi/heart_seg (file-native orientation, 114 LPS / 36 LAS across the 150) MUST be
reoriented the SAME way or the masks misalign with the canonical GT bundle. `reorient_to_lps` below
applies each file's own io_orientation -> LPS, exactly as ACDCGatedAdapter does for the cine.

Writes under scratch/eval/acdc/out/patientXXX/ : same layout as ocmr/miitt
  (gt/, clean/, breath/, mask_heart.nii.gz, mask_fov.nii.gz, heart_seg_native.nii.gz, manifest.json).
manifest adds: group (pathology class), ED/ES (1-based Info.cfg indices), native_source (4d path).

Run: micromamba run -n svr python evaluation/engine/build_inputs/acdc.py patient001 [patient002 ...]
     (default: all training patients)
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
import torch
from omegaconf import OmegaConf

VGGT = "/home/minsukc/vggt"
sys.path.insert(0, VGGT)
sys.path.insert(0, os.path.join(VGGT, "training"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # local geom.py (copied into git)

from inference.adapters.acdc import ACDCGatedAdapter  # noqa: E402
from inference.adapters.base import percentile_scale  # noqa: E402
from data.respiratory import RespiratoryConfig, sample_resp_disp, reslice_volume_vec  # noqa: E402
from geom import place_to_canonical  # noqa: E402

ACDC_ROOT = f"{VGGT}/scratch/data/ACDC"
OUT_ROOT = f"{VGGT}/scratch/eval/acdc/out"
CANON_SPACING_XYZ = (1.4, 1.4, 12.0)
DATASET = "acdc_gated"


def canon_affine():
    return np.diag([*CANON_SPACING_XYZ, 1.0])


def name_seed(name):
    return int(hashlib.sha256(f"{DATASET}/{name}".encode()).hexdigest(), 16) % (2 ** 31)


def reorient_to_lps(path):
    """Load a (X,Y,Z[,T]) NIfTI and reorient its 3 spatial axes to LPS (matching ACDCGatedAdapter).
    Returns the reoriented array (spatial axes only touched)."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    xfm = ornt_transform(io_orientation(img.affine), axcodes2ornt(("L", "P", "S")))
    return np.ascontiguousarray(apply_orientation(data, xfm))


def find_patient_dir(name):
    for split in ("training", "testing"):
        d = os.path.join(ACDC_ROOT, split, name)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(name)


def read_info(pdir):
    info = {}
    for ln in open(os.path.join(pdir, "Info.cfg")):
        if ":" in ln:
            k, v = ln.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def build_respiratory_config():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29574")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver(
        "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(VGGT, "training", "config")):
        cfg = compose(config_name="default")
    return RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)


def save_xyz(dhw, affine, path):
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.asarray(dhw).transpose(2, 1, 0)), affine), path)


def build_one(name, rcfg):
    pdir = find_patient_dir(name)
    nii_path = os.path.join(pdir, f"{name}_4d.nii.gz")
    info = read_info(pdir)
    adapter = ACDCGatedAdapter(nii_path)

    inpl = adapter.inplane_mm()
    pitch = float(np.diff(np.asarray(adapter.slice_positions_mm())[:, 2]).mean()) \
        if adapter._a.shape[2] > 1 else 10.0
    native_spacing_dhw = (pitch, inpl[1], inpl[0])
    native_affine = np.diag([inpl[0], inpl[1], pitch, 1.0])

    bundle, bbox = adapter.build_canonical_bundle()          # (T,12,256,256)
    T = bundle.shape[0]

    cine = adapter.load().astype(np.float32)                 # (T,Z,H,W) LPS-reoriented
    _, Z, H, W = cine.shape
    vmin, vmax = percentile_scale(cine)
    norm = np.clip((cine - vmin) / (vmax - vmin), 0.0, 1.0)

    seed = name_seed(name)
    disp, r = sample_resp_disp(1, Z, rcfg, "cpu", train=False,
                               seq_index=torch.tensor([[seed]], dtype=torch.int64))
    disp0 = disp[0]
    mean_abs = disp0.norm(dim=-1).mean().item()
    planes = int((bundle[0] > 0).any(axis=(1, 2)).sum())
    print(f"[{name}] grp={info.get('Group')} T={T} native(Z,H,W)=({Z},{H},{W}) pitch={pitch:.2f}mm "
          f"inpl={inpl[0]:.2f} bundle_planes={planes}/12 seed={seed} mean|disp|={mean_abs:.2f}mm", flush=True)

    subj_dir = os.path.join(OUT_ROOT, name)
    for sub in ("gt", "clean", "breath"):
        os.makedirs(os.path.join(subj_dir, sub), exist_ok=True)

    for t in range(T):
        save_xyz(bundle[t], canon_affine(), os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz"))
        vt = torch.from_numpy(norm[t])
        save_xyz(vt.numpy(), native_affine, os.path.join(subj_dir, "clean", f"stack_t{t:02d}.nii.gz"))
        breathed = torch.stack(
            [reslice_volume_vec(vt, disp0[z], spacing=native_spacing_dhw)[z] for z in range(Z)], dim=0)
        save_xyz(breathed.numpy(), native_affine, os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz"))

    # masks: reorient heart_roi to LPS (adapter frame) BEFORE placing, else misaligned
    roi_path = os.path.join(pdir, "heart_roi.nii.gz")
    if os.path.isfile(roi_path):
        roi_xyz = reorient_to_lps(roi_path)                  # (X,Y,Z[,T]) LPS
        if roi_xyz.ndim == 4:
            roi_xyz = roi_xyz.max(axis=3)
        roi_zhw = np.transpose(roi_xyz, (2, 1, 0)).astype(np.float32)
    else:
        roi_zhw = np.ones((Z, H, W), np.float32)
    heart_canon = place_to_canonical(roi_zhw, adapter, normalize=False, binary=True)
    fov_canon = place_to_canonical(np.ones((Z, H, W), np.float32), adapter, binary=True)
    save_xyz(heart_canon, canon_affine(), os.path.join(subj_dir, "mask_heart.nii.gz"))
    save_xyz(fov_canon, canon_affine(), os.path.join(subj_dir, "mask_fov.nii.gz"))

    # per-phase seg (Task114 4D), reoriented to LPS -> save native (X,Y,Z,T) for later EF/Dice
    seg_src = os.path.join(pdir, "heart_seg.nii.gz")
    if os.path.isfile(seg_src):
        seg_lps = reorient_to_lps(seg_src)                   # (X,Y,Z,T) LPS
        nib.save(nib.Nifti1Image(seg_lps.astype(np.int16), native_affine),
                 os.path.join(subj_dir, "heart_seg_native.nii.gz"))

    manifest = {
        "dataset": DATASET, "subject": name, "group": info.get("Group"),
        "ED": int(info.get("ED", 0)), "ES": int(info.get("ES", 0)), "NbFrame": int(info.get("NbFrame", T)),
        "native_source": nii_path, "seed": seed,
        "T": T, "native_ZHW": [Z, H, W],
        "native_spacing_xyz_mm": [float(inpl[0]), float(inpl[1]), float(pitch)],
        "canonical_spacing_xyz_mm": list(CANON_SPACING_XYZ),
        "intensity_scale": {"vmin": float(vmin), "vmax": float(vmax)},
        "bbox_zyx": [int(x) for x in bbox],
        "bundle_planes_filled": planes,
        "breath": {
            "mean_abs_disp_mm": mean_abs,
            "disp_dhw_mm": disp0.tolist(),
            "r_per_plane": r[0].tolist(),
            "amplitude_mm": rcfg.amplitude_mm, "amplitude_jitter": rcfg.amplitude_jitter,
            "ap_ratio": rcfg.ap_ratio, "cos2n": rcfg.cos2n, "group_by_burst": rcfg.group_by_burst,
            "tilt_min_deg": rcfg.tilt_min_deg, "tilt_max_deg": rcfg.tilt_max_deg,
            "direction_jitter_deg": rcfg.direction_jitter_deg,
        },
    }
    json.dump(manifest, open(os.path.join(subj_dir, "manifest.json"), "w"), indent=2)
    print(f"done -> {subj_dir}  (gt/clean/breath x {T} + masks + manifest)", flush=True)


def main():
    names = sys.argv[1:] or sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(ACDC_ROOT, "training", "patient*")))
    rcfg = build_respiratory_config()
    assert rcfg.enable
    rcfg.per_slot = False
    print(f"building {len(names)} ACDC patient(s)", flush=True)
    for name in names:
        build_one(name, rcfg)


if __name__ == "__main__":
    main()
