"""Export the respiratory-corrupted same-phase SAX stack for classical/ML SVR baselines.

Per docs/30 sec4 step 2 / docs/31 sec8 step 2: NiftyMIC/NeSVoR/SVRTK can't self-gate, so a
fair test needs input at ONE known real cardiac phase (unlike VGGT's native scattered
input) but with REAL misalignment for their registration to actually correct. The clean
stack baselines/niftymic/export_stack.py exports IS V_gt -- nothing to correct there. This
script applies the trainer's own respiratory sim (training/data/respiratory.py) -- the
exact rigid SI/AP shift applied at val time, per real z-slice -- to that same clean t=0
stack, giving every SVR baseline a genuine registration problem to solve.

GT is UNCHANGED: the original clean stack (scratch/niftymic/data/<tag>_stack.nii.gz)
remains V_gt, matching the trainer's own contract (target stays at the unshifted
end-expiration reference; only the INPUT is corrupted).

Mask: reuses the same unshifted content_mask (<tag>_mask.nii.gz, already exported) for
the corrupted variant too -- content_mask marks native-FOV vs zero-pad, an
acquisition-geometry property that doesn't change under a modest anatomical shift within
the FOV.

Output: scratch/niftymic/data/<subject>_t<phase>_resp_stack.nii.gz (same dir/mask as the
clean export -- baselines/nesvor/run_nesvor.sh's STACK_SUFFIX picks this up).

Usage: micromamba run -n svr python baselines/export_resp_stack.py
"""
import os
import sys

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from niftymic.export_stack import (  # noqa: E402
    DATA_ROOT, SPLIT_FILE, TARGET_PHASE, SUBJECT_INDICES, SPACING_XYZ, _affine,
)
from data.datasets.mri_dataset import MRIDataset  # noqa: E402
from data.respiratory import RespiratoryConfig, sample_resp_disp, reslice_volume_vec  # noqa: E402

OUT_DIR = "/home/minsukc/vggt/scratch/niftymic/data"  # same dir as the clean export
N_CANON_Z = 12  # canonical grid depth D -- one "slot" per real z-plane


def _build_respiratory_config():
    """Load RespiratoryConfig from the LIVE mri_volume.yaml (not hand-copied defaults) --
    mirrors inference/run_cmrxrecon.py's build_mri_dataset(), robust to config drift."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29566")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from hydra import compose, initialize_config_dir
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver(
        "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "config"))
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="mri_volume")
    return RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)


def main():
    rcfg = _build_respiratory_config()
    print(f"RespiratoryConfig: enable={rcfg.enable} amplitude_mm={rcfg.amplitude_mm} "
          f"ap_ratio={rcfg.ap_ratio} direction_jitter_deg={rcfg.direction_jitter_deg}")
    device = "cpu"  # tiny (12,256,256) volumes -- no GPU needed for a one-off export

    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(
        common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
        mode="dynamic", mri_mode="axial", num_slices=1, target_size=518,
        t_target_fixed=TARGET_PHASE,
    )
    affine = _affine(SPACING_XYZ)

    for idx in SUBJECT_INDICES:
        subject_path = ds.subjects[idx % len(ds.subjects)]
        name = os.path.basename(os.path.dirname(subject_path))
        data = ds.get_data(seq_index=idx, img_per_seq=1)
        phases = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32))  # (T,D,H,W)
        t_target = int(data["t_target"][0])
        assert t_target == TARGET_PHASE, (t_target, TARGET_PHASE)

        V = phases[t_target]  # (D,H,W) clean t=0 volume -- the same tensor V_gt uses

        # Val-time deterministic respiratory draw: one displacement PER Z-PLANE "slot"
        # (S=12), seeded by seq_index=idx exactly like the trainer's val path.
        seq_index = torch.tensor([[idx]], dtype=torch.int64)
        disp, r = sample_resp_disp(1, N_CANON_Z, rcfg, device, train=False, seq_index=seq_index)
        # disp: (1, 12, 3) mm (d_D, d_H, d_W) per z-plane; r: (1, 12) respiratory phase

        corrupted_planes = []
        for z in range(N_CANON_Z):
            shifted_vol = reslice_volume_vec(V, disp[0, z])  # (D,H,W), default canonical spacing
            corrupted_planes.append(shifted_vol[z])          # take THIS plane's own resliced content
        corrupted_dhw = torch.stack(corrupted_planes, dim=0).numpy().astype(np.float32)  # (D,H,W)

        mean_abs_diff = np.abs(corrupted_dhw - V.numpy()).mean()
        print(f"[{idx}] {name}  t={t_target}  mean|corrupted-clean|={mean_abs_diff:.5f}  "
              f"r range=({r.min().item():.3f},{r.max().item():.3f})")
        assert mean_abs_diff > 0, "respiratory corruption produced no change -- check enable=True"

        # splat (D=Z,H=Y,W=X) -> nibabel (X,Y,Z), same convention as export_stack.py.
        stack_xyz = corrupted_dhw.transpose(2, 1, 0)
        resp_path = os.path.join(OUT_DIR, f"{name}_t{t_target}_resp_stack.nii.gz")
        nib.save(nib.Nifti1Image(stack_xyz, affine), resp_path)
        print(f"     -> {resp_path}")


if __name__ == "__main__":
    main()
