"""5 val subjects × {canonical val draw + 5 random global-RNG draws} on the OLD ckpt.
Breathing ON. One RESULTS PNG and one INPUTS PNG per subject (10 PNGs total) so each
panel is large enough to read.

Outputs:
  result/4way_refiner/draws_<subj>_RESULTS.png   1 row × 6 V_canon mid-z (ED)
  result/4way_refiner/draws_<subj>_INPUTS.png    6 rows × S input slices

Subject identities (val split, random_8_1_1.txt):
  seq0 Train_P053, seq1 Val_P055, seq2 Train_P034, seq3 Test_P028, seq4 Train_P178
"""
import os
import random as pyrandom
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
from tools.diagnose_ood_clean_paradox import build_val_dataset

OLD_CKPT = "/home/minsukc/vggt/scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
OUT = os.path.join(_ROOT, "result", "4way_refiner")
DEV = torch.device("cuda")
GRID_SHAPE = (12, 256, 256)

SUBJECTS = [
    (0, "Train_P053"), (1, "Val_P055"), (2, "Train_P034"),
    (3, "Test_P028"), (4, "Train_P178"),
]
DRAWS = [("val_fixed", None), ("seed_11", 11), ("seed_22", 22),
         ("seed_33", 33), ("seed_44", 44), ("seed_55", 55)]


def load_old_model():
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True, enable_refiner=False,
    ).to(DEV).eval()
    ck = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state, strict=False)
    print("  OLD ckpt loaded", flush=True)
    return model


def one_draw(model, ds, rcfg, seq_index, seed):
    """seed=None → canonical val draw (random.Random(seq_index)).
    seed=int    → flip split='train' and seed global RNG for a fresh draw.
    Breathing ON in both cases."""
    from data.gpu_aug import gpu_augment_batch
    saved_split = ds.split
    try:
        if seed is None:
            ds.split = "val"
        else:
            ds.split = "train"
            pyrandom.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        data = ds.get_data(seq_index=seq_index, img_per_seq=ds.num_slices)
    finally:
        ds.split = saved_split

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
    # Breathing ON.
    batch = gpu_augment_batch(batch, None, DEV, respiratory_cfg=rcfg, train=False)

    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), -1.0, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    V = preds.get("V_canon")
    if V is None:
        wp = preds["world_points"].float()
        V, _cov = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    V_can = V[0].float().cpu().numpy()
    inputs = batch["images"][0].mean(1).cpu().numpy()
    z = np.asarray(data["z_indices"]).ravel()
    t = np.asarray(data["t_indices"]).ravel()
    return V_can, inputs, z, t


def window_pct(sl, V):
    hi = float(np.percentile(V, 99.5))
    lo = float(np.percentile(V, 1.0))
    return np.clip((sl - lo) / (hi - lo + 1e-9), 0, 1)


def render_results(outputs, names, subj_label, path):
    N = len(outputs)
    fig, axes = plt.subplots(1, N, figsize=(N * 4.0, 4.5))
    if N == 1:
        axes = [axes]
    for ax, V, nm in zip(axes, outputs, names):
        mid = V.shape[0] // 2
        ax.imshow(window_pct(V[mid], V), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{nm}\nV_canon mid-z @ ED", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{subj_label} — OLD ckpt — V_canon at ED, breathing ON  "
                 "(native 256x256, pct99.5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def render_inputs(inputs_list, zts, names, subj_label, path):
    N = len(inputs_list)
    S = max(im.shape[0] for im in inputs_list)
    fig, axes = plt.subplots(N, S, figsize=(S * 2.0, N * 2.1))
    if N == 1:
        axes = np.array([axes])
    if S == 1:
        axes = axes.reshape(N, 1)
    for r, (im, (zi, ti), nm) in enumerate(zip(inputs_list, zts, names)):
        for s in range(S):
            ax = axes[r, s]
            if s < im.shape[0]:
                hi = float(np.percentile(im[s], 99.5))
                lo = float(np.percentile(im[s], 1.0))
                ax.imshow(np.clip((im[s] - lo) / (hi - lo + 1e-9), 0, 1),
                          cmap="gray", aspect="equal")
                ax.set_title(f"z={zi[s]:+.2f} t={ti[s]:+.2f}", fontsize=7)
            ax.axis("off")
        axes[r, 0].text(-0.10, 0.5, nm, transform=axes[r, 0].transAxes,
                        fontsize=11, ha="right", va="center")
    fig.suptitle(f"{subj_label} — input slices per draw (breathing ON, val_fixed = canonical "
                 "random.Random(seq_index))", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    model = load_old_model()
    val_ds, rcfg = build_val_dataset()

    for seq, subj in SUBJECTS:
        print(f"\n=== seq{seq} ({subj}) ===")
        outs, ins, zts, names = [], [], [], []
        for nm, seed in DRAWS:
            V, inp, zi, ti = one_draw(model, val_ds, rcfg, seq, seed)
            outs.append(V); ins.append(inp); zts.append((zi, ti)); names.append(nm)
            print(f"  {nm}: S={inp.shape[0]}", flush=True)
        lbl = f"{subj} (val seq {seq})"
        render_results(outs, names, lbl,
                       os.path.join(OUT, f"draws_seq{seq}_{subj}_RESULTS.png"))
        render_inputs(ins, zts, names, lbl,
                      os.path.join(OUT, f"draws_seq{seq}_{subj}_INPUTS.png"))


if __name__ == "__main__":
    main()
