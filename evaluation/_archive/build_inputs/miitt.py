"""MIITT-gated STEP 1/3 — build the frozen SVRTK bundle for one MIITT gated subject.

Framing A (mirror of the CMRxRecon harness): the gated breath-held cine is clean
ground truth; we simulate the trainer's respiratory motion on it, reconstruct with
SVRTK, and score vs the SAME canonical GT the VGGT model is scored against.

KEY differences from cmrxrecon/build_inputs.py (see docs / handoff):
  * Source = MIITT `<subj>/gated/sax/4d_recon.nii.gz` (native 224x180xZ, T=30), NOT
    the CMRx MRIDataset canonical cache.
  * GT = `MIITTGatedAdapter.build_canonical_bundle()` VERBATIM -> byte-identical to
    what VGGT scores against (in-plane->1.4, z-snap to 12mm, 10/12 planes filled).
  * SVRTK input = NATIVE slices (1.5x1.5x10mm, thickness 8) -> clinically faithful
    (user decision 2026-07-12). Breathing applied on the NATIVE grid.
  * Recon is scored by resampling native->placement (see geom.py) so it co-locates
    with the placed GT; NO smooth canonical resample of the recon.

Writes under scratch/eval/miitt/out/<subj>/ :
  gt/gt_tNN.nii.gz          (256,256,12) canonical GT phases (VGGT-identical)
  clean/stack_tNN.nii.gz    native (X,Y,Z) normalized gated slices  (SVRTK clean input)
  breath/stack_tNN.nii.gz   native, per-slice respiratory-shifted    (SVRTK breath input)
  mask_heart_native.nii.gz  native heart ROI  -> SVRTK `-mask`
  mask_heart.nii.gz         canonical heart ROI (placed) -> scoring
  mask_fov.nii.gz           canonical FOV occupancy (placed) -> scoring (drops empty planes)
  heart_seg_native.nii.gz   native per-phase seg (for later EF/Dice)
  manifest.json             T, native shape, per-native-slice disp, bbox, scale, group

Run: micromamba run -n svr python evaluation/engine/build_inputs/miitt.py <subject>
"""
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # local geom.py (copied into git)

from inference.adapters.miitt import MIITTGatedAdapter, SLICE_SPACING_MM, GATED_INPLANE_MM  # noqa: E402
from inference.adapters.base import percentile_scale  # noqa: E402
from data.respiratory import RespiratoryConfig, sample_resp_disp, reslice_volume_vec  # noqa: E402
from geom import place_to_canonical  # noqa: E402

MIITT_ROOT = f"{VGGT}/scratch/data/MIITT/nifti"
OUT_ROOT = f"{VGGT}/scratch/eval/miitt/out"
CANON_SPACING_XYZ = (1.4, 1.4, 12.0)
# native breathing reslice spacing in splat (D=Z, H=Y, W=X) order
NATIVE_SPACING_DHW = (SLICE_SPACING_MM, GATED_INPLANE_MM[1], GATED_INPLANE_MM[0])
DATASET = "miitt_gated"


def canon_affine():
    return np.diag([*CANON_SPACING_XYZ, 1.0])


def name_seed(name):
    return int(hashlib.sha256(f"{DATASET}/{name}".encode()).hexdigest(), 16) % (2 ** 31)


def group_of(name):
    return "patient" if "patient" in name.lower() else "volunteer"


def build_respiratory_config():
    """RespiratoryConfig from the LIVE mri_volume.yaml. NOTE: main() forces `rcfg.per_slot=False`
    so eval draws ONE breath amplitude per SUBJECT (uniform 18.8 +/- 7.35 mm) -- matching training,
    which is already per-subject (its burst branch uses A_subj.expand, amplitude_breath_jitter=0).
    Breath PHASE stays per-slice (each slice = a different breath moment). This makes the eval
    corruption consistent with the already-per-subject-trained VGGT (train==eval)."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29572")
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
    """(D,H,W)=(Z,Y,X) splat-order array -> (X,Y,Z) NIfTI."""
    nib.save(nib.Nifti1Image(np.ascontiguousarray(np.asarray(dhw).transpose(2, 1, 0)), affine), path)


def main():
    name = sys.argv[1]
    sax = os.path.join(MIITT_ROOT, name, "gated", "sax")
    recon_path = os.path.join(sax, "4d_recon.nii.gz")
    assert os.path.isfile(recon_path), recon_path
    rcfg = build_respiratory_config()
    assert rcfg.enable, "respiratory sim disabled in mri_volume.yaml?!"
    rcfg.per_slot = False   # per-SUBJECT breath amplitude (match training); phase stays per-slice

    adapter = MIITTGatedAdapter(recon_path)
    native_affine = nib.load(recon_path).affine        # (X,Y,Z) world frame SVRTK reconstructs in

    # ---- GT: VGGT-identical canonical bundle (in-plane->1.4, z-snap, normalized) ----
    bundle, bbox = adapter.build_canonical_bundle()     # (T,12,256,256) splat, bbox(6,)
    T = bundle.shape[0]

    # ---- native normalized phases (SVRTK input space; SAME scale as the GT bundle) ----
    cine = adapter.load().astype(np.float32)            # (T,Z,H,W)=(T,Z,Y,X)
    _, Z, H, W = cine.shape
    vmin, vmax = percentile_scale(cine)
    norm = np.clip((cine - vmin) / (vmax - vmin), 0.0, 1.0)   # (T,Z,H,W) in [0,1]

    # ---- deterministic breathing (one realization/subject; native grid) ----
    seed = name_seed(name)
    disp, r = sample_resp_disp(1, Z, rcfg, "cpu", train=False,
                               seq_index=torch.tensor([[seed]], dtype=torch.int64))
    disp0 = disp[0]                                      # (Z,3) mm (d_D,d_H,d_W) per native slice
    mean_abs = disp0.norm(dim=-1).mean().item()
    print(f"[{name}] group={group_of(name)} T={T} native(Z,H,W)=({Z},{H},{W}) "
          f"bundle_planes={int((bundle[0] > 0).any(axis=(1,2)).sum())}/12 seed={seed} "
          f"mean|disp|={mean_abs:.2f}mm max|disp|={disp0.norm(dim=-1).max().item():.2f}mm", flush=True)

    subj_dir = os.path.join(OUT_ROOT, name)
    for sub in ("gt", "clean", "breath"):
        os.makedirs(os.path.join(subj_dir, sub), exist_ok=True)

    # GT phases (canonical)
    for t in range(T):
        save_xyz(bundle[t], canon_affine(), os.path.join(subj_dir, "gt", f"gt_t{t:02d}.nii.gz"))

    # native SVRTK stacks: clean == normalized gated slices; breath == per-slice reslice
    for t in range(T):
        vt = torch.from_numpy(norm[t])                  # (Z,H,W)
        save_xyz(vt.numpy(), native_affine, os.path.join(subj_dir, "clean", f"stack_t{t:02d}.nii.gz"))
        breathed = torch.stack(
            [reslice_volume_vec(vt, disp0[z], spacing=NATIVE_SPACING_DHW)[z] for z in range(Z)], dim=0)
        save_xyz(breathed.numpy(), native_affine, os.path.join(subj_dir, "breath", f"stack_t{t:02d}.nii.gz"))

    # ---- masks ----
    # native heart ROI -> SVRTK -mask (native recon space)
    roi_img = nib.load(os.path.join(sax, "heart_roi.nii.gz"))
    roi_xyz = np.asarray(roi_img.dataobj)               # (X,Y,Z)
    nib.save(nib.Nifti1Image((roi_xyz > 0.5).astype(np.float32), roi_img.affine),
             os.path.join(subj_dir, "mask_heart_native.nii.gz"))
    # canonical scoring masks (placed, co-located with GT bundle)
    roi_zhw = np.transpose(roi_xyz, (2, 1, 0)).astype(np.float32)    # (Z,H,W)
    heart_canon = place_to_canonical(roi_zhw, adapter, normalize=False, binary=True)  # (12,256,256)
    fov_canon = place_to_canonical(np.ones((Z, H, W), np.float32), adapter, binary=True)
    save_xyz(heart_canon, canon_affine(), os.path.join(subj_dir, "mask_heart.nii.gz"))
    save_xyz(fov_canon, canon_affine(), os.path.join(subj_dir, "mask_fov.nii.gz"))

    # native per-phase seg (for later EF/Dice)
    seg_src = os.path.join(sax, "heart_seg.nii.gz")
    if os.path.isfile(seg_src):
        import shutil
        shutil.copyfile(seg_src, os.path.join(subj_dir, "heart_seg_native.nii.gz"))

    manifest = {
        "dataset": DATASET, "subject": name, "group": group_of(name), "seed": seed,
        "T": T, "native_ZHW": [Z, H, W], "native_spacing_xyz_mm": [GATED_INPLANE_MM[0], GATED_INPLANE_MM[1], SLICE_SPACING_MM],
        "canonical_spacing_xyz_mm": list(CANON_SPACING_XYZ),
        "intensity_scale": {"vmin": float(vmin), "vmax": float(vmax)},
        "bbox_zyx": [int(x) for x in bbox],
        "bundle_planes_filled": int((bundle[0] > 0).any(axis=(1, 2)).sum()),
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


if __name__ == "__main__":
    main()
