"""OCMR-gated STEP 1 — build the frozen breathing bundle for one OCMR gated SAX subject.

Mirror of scratch/eval/miitt/build_inputs.py (the proven gated-OOD pattern), so the VGGT
head-to-head harness (engine/run_vggt.py --dataset ocmr) + engine/assemble_and_gif.py scorer
consume OCMR identically to MIITT/CMRx: ONE frozen breathing realization per subject, GT = the
VGGT-identical canonical bundle, PSNR/SSIM/NCC + resp_diag against the same target.

Source = OCMR fully-sampled ECG-gated recon `scratch/data/ocmr/recon/gated/<exam>/<subj>/`
(sax_cine.nii.gz 4D + meta.json + heart_seg.nii.gz 4D all-phase + heart_roi.nii.gz), produced by
scratch/data/ocmr/ocmr_recon_gated.py (iFFT+RSS, no CS -> clean cardiac motion GT). Loaded via
inference.adapters.ocmr.OCMRAdapter (build_canonical_bundle, inplane_mm/slice_positions_mm from meta).

KEY differences from miitt/build_inputs.py:
  * adapter = OCMRAdapter(<recon_subj_dir>) (reads sax_cine.nii.gz + meta.json), NOT MIITTGatedAdapter.
  * native spacing is per-SUBJECT from the adapter (meta.json), NOT MIITT's hard-coded 1.5/10 mm.
  * subject name is the safe-flattened `<exam>__<subj>` (path has '/'); the real recon dir is
    stored in manifest.native_source so engine/run_vggt.py:prep_ocmr can rebuild the adapter.

Writes under scratch/eval/ocmr/out/<exam>__<subj>/ :
  gt/gt_tNN.nii.gz          (256,256,12) canonical GT phases (VGGT-identical)
  clean/stack_tNN.nii.gz    native (X,Y,Z) normalized gated slices
  breath/stack_tNN.nii.gz   native, per-slice respiratory-shifted
  mask_heart.nii.gz         canonical heart ROI (placed) -> scoring
  mask_fov.nii.gz           canonical FOV occupancy (placed) -> scoring (drops empty planes)
  heart_seg_native.nii.gz   native per-phase seg (for later EF/Dice)
  manifest.json             T, native shape, per-native-slice disp, bbox, scale, native_source

Run: micromamba run -n svr python scratch/eval/ocmr/build_inputs.py <exam>/<subj>
     e.g. ... exam_fs_0074/sax__fs_0074_1_5T   (default: all discovered SAX gated subjects)
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import nibabel as nib
import torch
from omegaconf import OmegaConf

VGGT = "/home/minsukc/vggt"
sys.path.insert(0, VGGT)
sys.path.insert(0, os.path.join(VGGT, "training"))
sys.path.insert(0, os.path.join(VGGT, "scratch", "eval", "miitt"))

from inference.adapters.ocmr import OCMRAdapter  # noqa: E402
from inference.adapters.base import percentile_scale  # noqa: E402
from data.respiratory import RespiratoryConfig, sample_resp_disp, reslice_volume_vec  # noqa: E402
from geom import place_to_canonical  # noqa: E402

GATED_ROOT = f"{VGGT}/scratch/data/ocmr/recon/gated"
OUT_ROOT = f"{VGGT}/scratch/eval/ocmr/out"
CANON_SPACING_XYZ = (1.4, 1.4, 12.0)
DATASET = "ocmr_gated"


def canon_affine():
    return np.diag([*CANON_SPACING_XYZ, 1.0])


def name_seed(name):
    return int(hashlib.sha256(f"{DATASET}/{name}".encode()).hexdigest(), 16) % (2 ** 31)


def native_pitch_mm(adapter):
    """Center-to-center slice pitch from the meta slice positions (per-subject)."""
    pos = np.asarray(adapter.slice_positions_mm(), dtype=np.float64)  # (Z,3)
    if pos.shape[0] < 2:
        return 8.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).mean())


def build_respiratory_config():
    """RespiratoryConfig from the LIVE mri_volume.yaml; main() forces per_slot=False (one breath
    amplitude/subject, matching training; phase stays per-slice). Verbatim from miitt/build_inputs."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29573")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver(
        "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(VGGT, "training", "config")):
        cfg = compose(config_name="mri_volume")
    return RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)


def save_xyz(dhw, affine, path):
    """(D,H,W)=(Z,Y,X) splat-order array -> (X,Y,Z) NIfTI."""
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.asarray(dhw).transpose(2, 1, 0)), affine), path)


def discover():
    return sorted(
        os.path.relpath(os.path.dirname(f), GATED_ROOT)
        for f in glob.glob(os.path.join(GATED_ROOT, "*", "sax__*", "sax_cine.nii.gz")))


def build_one(rel, rcfg):
    """rel = '<exam>/<subj>' relative to GATED_ROOT."""
    recon_dir = os.path.join(GATED_ROOT, rel)
    name = rel.replace("/", "__")
    adapter = OCMRAdapter(recon_dir)

    # per-subject native spacing (splat D,H,W order = pitch, inplane_y, inplane_x)
    inpl = adapter.inplane_mm()
    pitch = native_pitch_mm(adapter)
    native_spacing_dhw = (pitch, inpl[1], inpl[0])
    native_affine = np.diag([inpl[0], inpl[1], pitch, 1.0])   # cosmetic (VGGT reads array only)

    # ---- GT: VGGT-identical canonical bundle ----
    bundle, bbox = adapter.build_canonical_bundle()          # (T,12,256,256), bbox(6,)
    T = bundle.shape[0]

    # ---- native normalized phases (same [0,1] scale as the GT bundle) ----
    cine = adapter.load().astype(np.float32)                 # (T,Z,H,W)=(T,Z,Y,X)
    _, Z, H, W = cine.shape
    vmin, vmax = percentile_scale(cine)
    norm = np.clip((cine - vmin) / (vmax - vmin), 0.0, 1.0)

    # ---- deterministic breathing (one realization/subject; native grid) ----
    seed = name_seed(name)
    disp, r = sample_resp_disp(1, Z, rcfg, "cpu", train=False,
                               seq_index=torch.tensor([[seed]], dtype=torch.int64))
    disp0 = disp[0]                                          # (Z,3) mm (d_D,d_H,d_W) per native slice
    mean_abs = disp0.norm(dim=-1).mean().item()
    planes = int((bundle[0] > 0).any(axis=(1, 2)).sum())
    print(f"[{name}] T={T} native(Z,H,W)=({Z},{H},{W}) pitch={pitch:.2f}mm inpl={inpl[0]:.2f} "
          f"bundle_planes={planes}/12 seed={seed} mean|disp|={mean_abs:.2f}mm "
          f"max|disp|={disp0.norm(dim=-1).max().item():.2f}mm", flush=True)

    subj_dir = os.path.join(OUT_ROOT, name)
    for sub in ("gt", "clean", "breath"):
        os.makedirs(os.path.join(subj_dir, sub), exist_ok=True)

    for t in range(T):
        save_xyz(bundle[t], canon_affine(), os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz"))
        vt = torch.from_numpy(norm[t])                       # (Z,H,W)
        save_xyz(vt.numpy(), native_affine, os.path.join(subj_dir, "clean", f"stack_t{t:02d}.nii.gz"))
        breathed = torch.stack(
            [reslice_volume_vec(vt, disp0[z], spacing=native_spacing_dhw)[z] for z in range(Z)], dim=0)
        save_xyz(breathed.numpy(), native_affine, os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz"))

    # ---- canonical scoring masks (placed, co-located with GT bundle) ----
    roi_path = os.path.join(recon_dir, "heart_roi.nii.gz")
    if os.path.isfile(roi_path):
        roi_xyz = np.asarray(nib.load(roi_path).dataobj, dtype=np.float32)   # (X,Y,Z[,T])
        if roi_xyz.ndim == 4:
            roi_xyz = roi_xyz.max(axis=3)                    # static ROI from any-phase union
        roi_zhw = np.transpose(roi_xyz, (2, 1, 0)).astype(np.float32)        # (Z,H,W)
    else:
        roi_zhw = np.ones((Z, H, W), np.float32)             # fall back to full FOV
    heart_canon = place_to_canonical(roi_zhw, adapter, normalize=False, binary=True)
    fov_canon = place_to_canonical(np.ones((Z, H, W), np.float32), adapter, binary=True)
    save_xyz(heart_canon, canon_affine(), os.path.join(subj_dir, "mask_heart.nii.gz"))
    save_xyz(fov_canon, canon_affine(), os.path.join(subj_dir, "mask_fov.nii.gz"))

    # native per-phase seg (for later EF/Dice)
    seg_src = os.path.join(recon_dir, "heart_seg.nii.gz")
    if os.path.isfile(seg_src):
        import shutil
        shutil.copyfile(seg_src, os.path.join(subj_dir, "heart_seg_native.nii.gz"))

    manifest = {
        "dataset": DATASET, "subject": name, "native_source": recon_dir, "seed": seed,
        "T": T, "native_ZHW": [Z, H, W],
        "native_spacing_xyz_mm": [float(inpl[0]), float(inpl[1]), float(pitch)],
        "canonical_spacing_xyz_mm": list(CANON_SPACING_XYZ),
        "intensity_scale": {"vmin": float(vmin), "vmax": float(vmax)},
        "bbox_zyx": [int(x) for x in bbox],
        "bundle_planes_filled": planes,
        "breath": {
            "mean_abs_disp_mm": mean_abs,
            "disp_dhw_mm": disp0.tolist(),          # (Z,3) per native slice
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
    args = sys.argv[1:]
    rels = args or discover()
    rcfg = build_respiratory_config()
    assert rcfg.enable, "respiratory sim disabled in mri_volume.yaml?!"
    rcfg.per_slot = False   # per-SUBJECT breath amplitude (match training); phase stays per-slice
    print(f"building {len(rels)} OCMR-gated subject(s)", flush=True)
    for rel in rels:
        build_one(rel, rcfg)


if __name__ == "__main__":
    main()
