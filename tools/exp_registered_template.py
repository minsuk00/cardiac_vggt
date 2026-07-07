"""E2 — Was doc-20's "population template useless (14.4 dB, below floor)" a grid-MISALIGNMENT artifact?

doc-20 built the population prior as a raw voxel-average of TRAIN subjects' canonical phases and got
14.4 dB (below the 16.8 identity floor), concluding "no free population prior" -> a learned decoder
can't lean on a template. But that average was over hearts that are NOT anatomically aligned in the
canonical grid (position/size vary). This re-runs it two ways:

  naive   : template[t] = mean_train phases[t]  in the raw canonical grid (= doc-20's number).
  aligned : first affine-align each subject's heart (anatomy_bbox -> a common target box, translate +
            per-axis scale) into a COMMON frame, average there, then score aligned template vs aligned
            held-out subject (same aligned frame).

If aligned >> naive and clears the floor, a population prior DOES carry usable subject-agnostic
appearance once hearts are registered -> the learned-prior propagation lever (crux for cardiac
through-plane / unobserved-plane synthesis) has headroom doc-20 under-counted. If aligned ~ naive,
doc-20 stands: hearts are too individually-varying for a template to help.

Run: micromamba run -n svr python tools/exp_registered_template.py
"""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, GRID_SHAPE, NUM_SLICES
from data.datasets.mri_dataset import MRIDataset
from eval_variants_matrix import DATA_ROOT, SPLIT_FILE
from data.respiratory import RespiratoryConfig  # noqa
from loss import compute_motion_mask
from omegaconf import OmegaConf
T = 12; D, H, W = GRID_SHAPE
OUT = os.path.join(REPO, "result", "registered_template")


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def make_ds(split):
    conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True, "rescale_aug": False,
                             "landscape_check": False, "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(conf, DATA_ROOT, split=split, split_file=SPLIT_FILE, mode="dynamic",
                      mri_mode="axial", num_slices=NUM_SLICES, target_size=518)


def get(ds, seq):
    data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
    ph = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).cuda()   # (T,D,H,W) splat order
    bbox = np.asarray(data["anatomy_bbox"]).astype(np.float32)                    # (z0,z1,y0,y1,x0,x1)
    return ph, bbox


def align_grid(bbox, tgt, device):
    """Build a grid_sample grid that resamples a native (D,H,W) volume into the aligned frame where the
    subject's bbox maps onto the common target box `tgt`. For each aligned output voxel, find the native
    coord: n = blo + (a - tlo) * (bhi-blo)/(thi-tlo), then normalize to [-1,1]."""
    (z0, z1, y0, y1, x0, x1) = bbox
    (tz0, tz1, ty0, ty1, tx0, tx1) = tgt
    zs = torch.arange(D, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    def m(a, tlo, thi, blo, bhi):
        return blo + (a - tlo) * (bhi - blo) / max(thi - tlo, 1e-3)
    nz = m(zs, tz0, tz1, z0, z1); ny = m(ys, ty0, ty1, y0, y1); nx = m(xs, tx0, tx1, x0, x1)
    nz = nz / (D - 1) * 2 - 1; ny = ny / (H - 1) * 2 - 1; nx = nx / (W - 1) * 2 - 1
    gz = nz.view(D, 1, 1).expand(D, H, W); gy = ny.view(1, H, 1).expand(D, H, W); gx = nx.view(1, 1, W).expand(D, H, W)
    return torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)                          # (1,D,H,W,3) xyz


def warp(vol_tdhw, grid):
    """vol (T,D,H,W) -> aligned (T,D,H,W) via one grid (shared across T)."""
    out = F.grid_sample(vol_tdhw.unsqueeze(1), grid.expand(vol_tdhw.shape[0], -1, -1, -1, -1),
                        mode="bilinear", padding_mode="zeros", align_corners=True)
    return out[:, 0]


def main():
    os.makedirs(OUT, exist_ok=True)
    tr, va = make_ds("train"), make_ds("val")
    N_TMPL = min(60, len(tr.subjects))
    # target box = mean bbox over template subjects
    bxs = []
    phs_tr = []
    for s in range(N_TMPL):
        ph, bb = get(tr, s); phs_tr.append(ph); bxs.append(bb)
    tgt = np.mean(np.stack(bxs), 0)
    dev = "cuda"
    # naive template (raw grid) + aligned template (common frame)
    naive_acc = torch.zeros((T, D, H, W), device=dev)
    align_acc = torch.zeros((T, D, H, W), device=dev)
    for ph, bb in zip(phs_tr, bxs):
        naive_acc += ph
        align_acc += warp(ph, align_grid(bb, tgt, dev))
    naive_tmpl = naive_acc / N_TMPL
    align_tmpl = align_acc / N_TMPL
    print(f"templates from {N_TMPL} train subjects; target bbox={np.round(tgt,1).tolist()}")

    rows = {"naive": [], "aligned": []}
    for vs in range(len(va.subjects)):
        ph, bb = get(va, vs)
        mmask = compute_motion_mask(ph.unsqueeze(0))[0].cpu().numpy()
        if not mmask.any():
            continue
        ph_al = warp(ph, align_grid(bb, tgt, dev))
        mmask_al = compute_motion_mask(ph_al.unsqueeze(0))[0].cpu().numpy()
        ph_np, phal_np = ph.cpu().numpy(), ph_al.cpu().numpy()
        nt, at = naive_tmpl.cpu().numpy(), align_tmpl.cpu().numpy()
        for t in range(T):
            rows["naive"].append(psnr(nt[t], ph_np[t], mmask))
            if mmask_al.any():
                rows["aligned"].append(psnr(at[t], phal_np[t], mmask_al))
    summary = {"naive_template": round(float(np.mean(rows["naive"])), 2),
               "aligned_template": round(float(np.mean(rows["aligned"])), 2),
               "identity_floor_ref": 16.8, "transport_ceiling_ref": 21.0, "trained_model_ref": 20.6,
               "doc20_naive_ref": 14.4, "n_val_subjects": len(va.subjects), "n_template": N_TMPL,
               "note": "aligned>>naive and >floor => population prior carries usable appearance once "
                       "registered (doc-20 undercounted); aligned~naive => doc-20 stands."}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("Wrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
