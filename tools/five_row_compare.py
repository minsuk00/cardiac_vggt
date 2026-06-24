"""5-row OOD-vs-in-dist comparison across all 4 modes:

  Row 1: OLD V_canon            (218747856, backbone only) — OOD baseline
  Row 2: newseed V_canon        (217891050, OLD backbone FROZEN; sanity ≡ Row 1)
  Row 3: newseed V_refined      (SSIM refiner on top of frozen OLD backbone)
  Row 4: joint V_canon          (218349151, jointly-trained backbone, pre-refiner)
  Row 5: joint V_refined        (jointly-trained refiner)

Reads:
  Row 1 ≡ Row 2 → backbone freeze confirmed (sanity).
  Row 2 → Row 3 → pure SSIM-refiner contribution on OLD backbone.
  Row 1 → Row 4 → joint training's damage to the backbone alone.
  Row 3 vs Row 5 → the joint-vs-separate headline.

Outputs (one per mode, native 256x256, pct99.5):
  result/4way_refiner/five_row_val_ON.png
  result/4way_refiner/five_row_val_OFF.png
  result/4way_refiner/five_row_OCMR.png
  result/4way_refiner/five_row_Goett.png
  result/4way_refiner/five_row_MIITT.png   (3rd OOD set, no GT — cross-validates OCMR/Göttingen)
"""
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))


from vggt.models.vggt import VGGT
from vggt.utils.splat import splat_predictions
from tools.eval_ocmr_inference import load_cine, percentile_scale, assign_canonical_z, build_batch
from tools.diagnose_ood_clean_paradox import build_val_dataset
from eval.adapters.goettingen import GoettingenAdapter
from eval.adapters.miitt import MIITTAdapter

OLD_CKPT  = "/home/minsukc/vggt/scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
NEWSEED_CKPT = "/home/minsukc/vggt/scratch/logs/217891050_mri_refiner_frozen_ssim_newseed/ckpts/checkpoint_last.pt"
JOINT_CKPT = "/home/minsukc/vggt/scratch/logs/218349151_mri_refiner_joint/ckpts/checkpoint_last.pt"
OUT = os.path.join(_ROOT, "result", "4way_refiner")
DEV = torch.device("cuda")
GRID_SHAPE = (12, 256, 256)

OCMR_SUBJECTS = ["us_0084_1_5T", "us_0173_pt_1_5T", "us_0183_pt_1_5T",
                 "us_0197_pt_1_5T", "us_0169_pt_1_5T"]
GOTT_RECON = "/home/minsukc/vggt/scratch/data/goettingen/recon"  # native RT recons (direct adapter)
GOTT_SUBJECTS = ["vol0001_vis1", "vol0002_vis1", "vol0003_vis1",
                 "vol0009_vis1", "vol0023_vis1"]
MIITT_RECON = "/home/minsukc/vggt/scratch/data/MIITT/nifti"  # RT free-breathing arm (PLACEHOLDER spacing)
MIITT_SUBJECTS = ["Volunteer1", "Volunteer2", "Volunteer3", "Volunteer4", "Volunteer5"]
VAL_SEQS = [0, 1, 2, 3, 4]


def build_model(use_refiner):
    return VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
        enable_refiner=use_refiner, refiner_use_coverage=use_refiner, grid_shape=GRID_SHAPE,
    ).to(DEV).eval()


def load_state(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    return model


def _forward(model, batch, target_t=-1.0):
    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), target_t, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    V_can = preds.get("V_canon")
    if V_can is None:
        wp = preds["world_points"].float()
        V_can, _cov = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    V_can = V_can[0].float().cpu().numpy()
    V_ref = preds["V_refined"][0].float().cpu().numpy() if "V_refined" in preds else None
    return V_can, V_ref


def val_batch(ds, rcfg, seq_index, breathing):
    from data.gpu_aug import gpu_augment_batch
    S0 = ds.num_slices
    data = ds.get_data(seq_index=seq_index, img_per_seq=S0)
    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(DEV)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    batch = {"images": imgs, "scanner_coords": st("scanner_coords"),
             "z_indices": st("z_indices"), "t_indices": st("t_indices"),
             "timesteps": st("timesteps", np.int64),
             "slice_indices": st("slice_indices", np.int64),
             "phases": torch.from_numpy(np.asarray(data["phases"]).astype(np.float32))
                        .to(DEV).unsqueeze(0),
             "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=DEV)}
    return gpu_augment_batch(batch, None, DEV,
                              respiratory_cfg=(rcfg if breathing else None), train=False)


def ocmr_batch(subj_dir):
    cine, meta = load_cine(subj_dir)
    scale = percentile_scale(cine)
    z_map = assign_canonical_z(meta["slice_positions_mm"])
    rng = np.random.default_rng(0)
    batch, _S, _picks = build_batch(cine, meta, scale, z_map, rng, DEV)
    return batch


def goettingen_batch(ds, subj):
    # Direct RTFB adapter on the native recon (real slices, no 6->8mm interp); `ds` unused.
    nii = os.path.join(GOTT_RECON, subj, subj + ".nii.gz")
    return GoettingenAdapter(nii).build_batch(np.random.default_rng(0), DEV)[0]


def build_gott_dataset():
    return None  # Göttingen now uses the direct adapter; no MRIDataset needed


def miitt_batch(subj):
    # Direct RTFB adapter on the MIITT RT arm (placeholder spacing — qualitative only).
    nii = os.path.join(MIITT_RECON, subj, "realtime", "sax", "4d_recon.nii.gz")
    return MIITTAdapter(nii).build_batch(np.random.default_rng(0), DEV)[0]


def window_pct(sl, V):
    hi = float(np.percentile(V, 99.5))
    lo = float(np.percentile(V, 1.0))
    return np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1)


def save_refined_niftis(new_outs, out_dir):
    """Dump the newseed-ckpt V_refined volume (row 3) per mode/subject as NIfTI.

    Identity affine — V_refined lives in the dimensionless canonical [-1,1] grid
    (12,256,256)=(Z,Y,X), same convention as trainer.save_val_volumes.
    """
    import nibabel as nib
    os.makedirs(out_dir, exist_ok=True)
    labels_by_mode = {"val_ON": [f"seq{i}" for i in VAL_SEQS],
                      "val_OFF": [f"seq{i}" for i in VAL_SEQS],
                      "OCMR": OCMR_SUBJECTS, "Goett": GOTT_SUBJECTS,
                      "MIITT": MIITT_SUBJECTS}
    affine = np.eye(4, dtype=np.float32)
    for mode, outs in new_outs.items():
        for (_V_canon, V_ref), lbl in zip(outs, labels_by_mode[mode]):
            if V_ref is None:
                continue
            path = os.path.join(out_dir, f"{mode}_{lbl}_Vrefined.nii.gz")
            nib.save(nib.Nifti1Image(V_ref.astype(np.float32), affine), path)
            print(f"  wrote {path}")


def render_5row(rows_data, labels, title, path):
    N = len(labels)
    fig, axes = plt.subplots(5, N, figsize=(N * 3.0, 15.0))
    if N == 1:
        axes = np.array(axes).reshape(5, 1)
    row_labels = [
        "OLD ckpt\nV_canon\n(no refiner)\n[OOD baseline]",
        "newseed ckpt\nV_canon\n(OLD bb FROZEN)\n[≡ Row 1?]",
        "newseed ckpt\nV_refined\n(SSIM refiner on\nOLD backbone)",
        "joint ckpt\nV_canon\n(jointly trained,\npre-refiner)",
        "joint ckpt\nV_refined\n(jointly trained\nrefiner)",
    ]
    for r, vols in enumerate(rows_data):
        for i, (lbl, V) in enumerate(zip(labels, vols)):
            mid = V.shape[0] // 2
            axes[r, i].imshow(window_pct(V[mid], V), cmap="gray", vmin=0, vmax=1)
            axes[r, i].axis("off")
            if r == 0:
                axes[r, i].set_title(lbl, fontsize=9)
            if i == 0:
                axes[r, i].text(-0.25, 0.5, row_labels[r],
                                transform=axes[r, i].transAxes,
                                fontsize=8, ha="right", va="center")
    fig.suptitle(f"{title}  (native 256x256, pct99.5 per volume)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def run_canon_across_modes(model, val_ds, rcfg, gott_ds):
    """Returns dict[mode] -> list of (V_canon, V_refined) tuples."""
    out = {"val_ON": [], "val_OFF": [], "OCMR": [], "Goett": [], "MIITT": []}
    for i in VAL_SEQS:
        out["val_ON"].append(_forward(model, val_batch(val_ds, rcfg, i, True)))
    for i in VAL_SEQS:
        out["val_OFF"].append(_forward(model, val_batch(val_ds, rcfg, i, False)))
    for sub in OCMR_SUBJECTS:
        sd = os.path.join(_ROOT, "scratch/data/ocmr/recon", sub)
        if os.path.isdir(sd):
            out["OCMR"].append(_forward(model, ocmr_batch(sd)))
    for sub in GOTT_SUBJECTS:
        out["Goett"].append(_forward(model, goettingen_batch(gott_ds, sub)))
    for sub in MIITT_SUBJECTS:
        nii = os.path.join(MIITT_RECON, sub, "realtime", "sax", "4d_recon.nii.gz")
        if os.path.exists(nii):
            out["MIITT"].append(_forward(model, miitt_batch(sub)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    gott_ds = build_gott_dataset()

    print("\n[1/3] OLD ckpt (no refiner) -> V_canon ...")
    m = load_state(build_model(use_refiner=False), OLD_CKPT)
    old_outs = run_canon_across_modes(m, val_ds, rcfg, gott_ds)
    del m; torch.cuda.empty_cache()

    print("\n[2/3] newseed ckpt (frozen OLD bb + SSIM refiner) -> V_canon, V_refined ...")
    m = load_state(build_model(use_refiner=True), NEWSEED_CKPT)
    new_outs = run_canon_across_modes(m, val_ds, rcfg, gott_ds)
    del m; torch.cuda.empty_cache()

    print("\n[3/3] joint ckpt -> V_canon, V_refined ...")
    m = load_state(build_model(use_refiner=True), JOINT_CKPT)
    joint_outs = run_canon_across_modes(m, val_ds, rcfg, gott_ds)
    del m; torch.cuda.empty_cache()

    val_labels = [f"seq{i}" for i in VAL_SEQS]
    modes = [
        ("val_ON",  val_labels,      "val ON (in-dist, breathing sim)"),
        ("val_OFF", val_labels,      "val OFF (in-dist, no breathing)"),
        ("OCMR",    OCMR_SUBJECTS,   "OCMR (OOD real-time free-breathing)"),
        ("Goett",   GOTT_SUBJECTS,   "Göttingen (OOD radial RT free-breathing)"),
        ("MIITT",   MIITT_SUBJECTS,  "MIITT (OOD paired RT free-breathing; placeholder spacing)"),
    ]
    for key, labels, title in modes:
        if not old_outs[key]:
            print(f"  skip {key} (no subjects found)"); continue
        rows = [
            [V for V, _ in old_outs[key]],     # row 1: OLD V_canon
            [V for V, _ in new_outs[key]],     # row 2: newseed V_canon (≡ row 1?)
            [Vr for _, Vr in new_outs[key]],   # row 3: newseed V_refined
            [V for V, _ in joint_outs[key]],   # row 4: joint V_canon
            [Vr for _, Vr in joint_outs[key]], # row 5: joint V_refined
        ]
        render_5row(rows, labels, title,
                    os.path.join(OUT, f"five_row_{key}.png"))

    print("\nSaving newseed V_refined NIfTIs ...")
    save_refined_niftis(new_outs, os.path.join(OUT, "nifti_newseed_refined"))


if __name__ == "__main__":
    main()
