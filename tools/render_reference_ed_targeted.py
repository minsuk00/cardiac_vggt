"""ED-targeted IO + DVF panels for the reference-conditioned models, on datasets where we can
fix the target to end-diastole: ACDC (labeled ED in Info.cfg) and the RT OOD sets
(OCMR/Goettingen/MIITT, ED detected by nnU-Net max-LV-area, read from ed_frames.json).

Reference contract: slot 0 = the ED frame at the mid-ventricular canonical plane (the anchor);
the remaining slots are scattered (random frames at the other in-FOV z planes). So every panel
reconstructs the heart AT ED, conditioned on a real ED reference slice.

Outputs result/reference_models_io/{model}/{DATASET}_{subj}_io.png and _dvf.png.

Run: micromamba run -n svr python tools/render_reference_ed_targeted.py --datasets acdc[,ocmr,goett,miitt]
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "training"))

from eval.adapters.base import (
    percentile_scale, assign_canonical_z, to_canonical_inplane,
    INPUT_IMG_SIZE, D_CANON, GRID_SHAPE,
)
from eval.adapters.ocmr import OCMRAdapter
from eval.adapters.goettingen import GoettingenAdapter
from eval.adapters.miitt import MIITTAdapter
from vggt.models.vggt import VGGT
from vggt.utils.splat import splat_predictions

DEV = torch.device("cuda")
OUT_ROOT = os.path.join(_ROOT, "result", "reference_models_io")
ED_JSON = os.path.join(OUT_ROOT, "ed_frames.json")
MODELS = [
    ("reference", "scratch/logs/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "dpt"),
    ("diffusion", "scratch/logs/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "dpt"),
    ("bspline",   "scratch/logs/217719798_mri_volume_bspline_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "bspline"),
]
OCMR_SUBJECTS = ["us_0084_1_5T", "us_0173_pt_1_5T", "us_0183_pt_1_5T", "us_0169_pt_1_5T", "us_0197_pt_1_5T"]
GOTT_SUBJECTS = ["vol0001_vis1", "vol0002_vis1", "vol0003_vis1", "vol0009_vis1", "vol0023_vis1"]
MIITT_SUBJECTS = ["Volunteer1", "Volunteer2", "Volunteer3", "Volunteer4", "Volunteer5"]
GOTT_RECON = os.path.join(_ROOT, "scratch/data/goettingen/recon")
MIITT_RECON = os.path.join(_ROOT, "scratch/data/MIITT/nifti")
OCMR_RECON = os.path.join(_ROOT, "scratch/data/ocmr/recon")
ACDC_ROOT = os.path.join(_ROOT, "scratch/data/ACDC")
IN_PLANE_MM = (256 - 1) / 2.0 * 1.4
THROUGH_MM = (12 - 1) / 2.0 * 12.0
IN_PLANE_R, THROUGH_R = 15.0, 25.0
D = D_CANON


# ── ACDC adapter (labeled ED) — same interface as the RT adapters ───────────
class ACDCAdapter:
    def __init__(self, pdir):
        pid = os.path.basename(pdir)
        im = nib.load(os.path.join(pdir, f"{pid}_4d.nii.gz"))
        self._a = np.asarray(im.dataobj, np.float32)              # (X, Y, Z, T)
        z = im.header.get_zooms()
        self._inplane = (float(z[0]), float(z[1]))
        self._zsp = float(z[2])
        self.ed = self._cfg_ed(os.path.join(pdir, "Info.cfg")) - 1  # 1-indexed -> 0-indexed

    @staticmethod
    def _cfg_ed(path):
        for line in open(path):
            if line.startswith("ED:"):
                return int(float(line.split(":")[1]))
        raise KeyError("ED")

    def load(self):
        return np.transpose(self._a, (3, 2, 1, 0))                # (T, Z, H=Y, W=X)

    def inplane_mm(self):
        return self._inplane

    def slice_positions_mm(self):
        nS = self._a.shape[2]
        return np.stack([np.zeros(nS), np.zeros(nS), np.arange(nS) * self._zsp], axis=1)


def mid_ventricular_entry(z_map):
    """z_map sorted by canonical z; pick the median plane as mid-ventricular."""
    zc = [z for z, _ in z_map]
    mid_z = zc[len(zc) // 2]
    mid = next(e for e in z_map if e[0] == mid_z)
    return mid, [e for e in z_map if e != mid]


def build_ed_batch(cine, inplane, slice_positions, ed_frame, rng, device):
    """slot 0 = ED frame @ mid-ventricular plane; other slots = random frames @ other z planes."""
    T = cine.shape[0]
    vmin, vmax = percentile_scale(cine)
    z_map = assign_canonical_z(slice_positions)
    if not z_map:
        raise ValueError("no in-FOV slices")
    mid, rest = mid_ventricular_entry(z_map)
    slots = [(mid, int(ed_frame))] + [(e, int(rng.integers(T))) for e in rest]
    py, px = np.meshgrid(np.arange(INPUT_IMG_SIZE), np.arange(INPUT_IMG_SIZE), indexing="ij")
    x_norm = (px / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)
    imgs, coords, zidx, picks = [], [], [], []
    for (z_canon, slice_idx), f in slots:
        raw = cine[f, slice_idx]
        norm = np.clip((raw - vmin) / (vmax - vmin), 0.0, 1.0)
        canon = to_canonical_inplane(norm, inplane)
        up = F.interpolate(canon[None, None], size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                           mode="bilinear", align_corners=True)[0, 0].numpy()
        imgs.append(np.repeat(up[None], 3, axis=0))
        z_val = z_canon / max(1, D - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1))
        zidx.append([z_val])
        picks.append((z_canon, slice_idx, f))
    batch = {
        "images": torch.from_numpy(np.stack(imgs)).float()[None].to(device),
        "scanner_coords": torch.from_numpy(np.stack(coords)).float()[None].to(device),
        "z_indices": torch.tensor(zidx, dtype=torch.float32)[None].to(device),
    }
    return batch, picks


# ── render helpers (input | V_canon IO; DVF) ────────────────────────────────
def input_volume(batch):
    imgs = batch["images"][0]
    z = batch["z_indices"][0, :, 0].float().cpu().numpy()
    V = np.zeros((D, 256, 256), np.float32)
    for s in range(imgs.shape[0]):
        zi = int(round((z[s] + 1) / 2 * (D - 1)))
        if 0 <= zi <= D - 1:
            V[zi] = F.interpolate(imgs[s, 0].float().cpu()[None, None], size=(256, 256),
                                  mode="bilinear", align_corners=True)[0, 0].numpy()
    return V


def window_pct(V):
    nz = V[V > 0]; ref = nz if nz.size else V
    hi = float(np.percentile(ref, 99.5)); lo = float(np.percentile(ref, 1.0))
    return np.clip((V - lo) / (hi - lo + 1e-9), 0, 1)


def render_io(Vin, Vout, title, path):
    Vin_w, Vout_w = window_pct(Vin), window_pct(Vout)
    fig = plt.figure(figsize=(8 * 2.6 + 0.5, 3 * 2.6))
    gs = gridspec.GridSpec(3, 9, figure=fig,
                           width_ratios=[1, 1, 1, 1, 0.12, 1, 1, 1, 1], wspace=0.05, hspace=0.12)
    for Vw, c0, tag in [(Vin_w, 0, "in"), (Vout_w, 5, "V_canon")]:
        for k in range(D):
            ax = fig.add_subplot(gs[k // 4, c0 + (k % 4)])
            ax.imshow(Vw[k], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{tag} z={k}", fontsize=8); ax.axis("off")
    sep = fig.add_subplot(gs[:, 4]); sep.set_xlim(0, 1); sep.set_ylim(0, 1)
    sep.axvline(0.5, color="red", lw=3); sep.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def render_dvf(batch, pred_dvf, title, path, picks):
    imgs = batch["images"][0].detach().float().cpu().clamp(0, 1).mean(dim=1).numpy()
    S = imgs.shape[0]
    p50, p95, p99 = (float(np.percentile(np.abs(pred_dvf), q)) for q in (50, 95, 99))
    fig = plt.figure(figsize=(1.9 * S + 1.6, 8.5), dpi=160)
    gs = gridspec.GridSpec(4, S + 1, width_ratios=[1.0] * S + [0.05], wspace=0.04, hspace=0.18)
    fig.suptitle(f"{title}    |Δ|(norm) p50={p50:.3f} p95={p95:.3f} p99={p99:.3f}", fontsize=10)
    rows = [("input intensity", imgs, "gray", 0, 1.0, True),
            ("Δx (mm)", pred_dvf[..., 0] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
            ("Δy (mm)", pred_dvf[..., 1] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
            ("Δz (mm)", pred_dvf[..., 2] * THROUGH_MM, "RdBu_r", -THROUGH_R, THROUGH_R, False)]
    for r, (lbl, data, cmap, vmin, vmax, is_top) in enumerate(rows):
        last = None
        for s in range(S):
            ax = fig.add_subplot(gs[r, s])
            last = ax.imshow(data[s], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if is_top:
                zc, sidx, f = picks[s]
                ax.set_title(f"z={zc} f={f}" + ("  [ref·ED]" if s == 0 else ""), fontsize=8)
            if s == 0:
                ax.set_ylabel(lbl, fontsize=9)
        plt.colorbar(last, cax=fig.add_subplot(gs[r, S]))
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def forward(model, batch):
    S = batch["images"].shape[1]
    batch.setdefault("target_t_indices", torch.full((1, S, 1), -1.0, dtype=torch.float32, device=DEV))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    wp = preds["world_points"].float()
    V = preds.get("V_canon")
    if V is None:
        V, _ = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    V = V[0].float().cpu().numpy()
    dvf = (wp[0] - batch["scanner_coords"][0]).detach().float().cpu().numpy()
    return V, dvf


# ── dataset → list of (subject, adapter-factory, ed_frame) ──────────────────
def acdc_jobs():
    pats = sorted(glob.glob(os.path.join(ACDC_ROOT, "training", "patient*")))[:5]
    out = []
    for p in pats:
        ad = ACDCAdapter(p)
        out.append((os.path.basename(p), ad, ad.ed))
    return out


def ood_jobs(name, subjects, ed_map):
    factory = {"OCMR": lambda s: OCMRAdapter(os.path.join(OCMR_RECON, s)),
               "Goett": lambda s: GoettingenAdapter(os.path.join(GOTT_RECON, s, s + ".nii.gz")),
               "MIITT": lambda s: MIITTAdapter(os.path.join(MIITT_RECON, s, "realtime", "sax", "4d_recon.nii.gz"))}[name]
    out = []
    for s in subjects:
        if s not in ed_map:
            print(f"  skip {name}/{s}: no ED in ed_frames.json"); continue
        out.append((s, factory(s), int(ed_map[s])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="acdc")  # comma list: acdc,ocmr,goett,miitt
    args = ap.parse_args()
    want = [d.strip().lower() for d in args.datasets.split(",")]
    ed_map = json.load(open(ED_JSON)) if os.path.exists(ED_JSON) else {}

    DATASETS = {}
    if "acdc" in want:
        DATASETS["ACDC"] = acdc_jobs()
    if "ocmr" in want:
        DATASETS["OCMR"] = ood_jobs("OCMR", OCMR_SUBJECTS, ed_map.get("OCMR", {}))
    if "goett" in want:
        DATASETS["Goett"] = ood_jobs("Goett", GOTT_SUBJECTS, ed_map.get("Goett", {}))
    if "miitt" in want:
        DATASETS["MIITT"] = ood_jobs("MIITT", MIITT_SUBJECTS, ed_map.get("MIITT", {}))

    rng = np.random.default_rng(0)
    for name, ckpt, head in MODELS:
        out_dir = os.path.join(OUT_ROOT, name); os.makedirs(out_dir, exist_ok=True)
        print(f"=== {name} ({head}) ===", flush=True)
        model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                     enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                     use_z_pose_embedding=True, use_t_pose_embedding=False,
                     use_target_t_pose_embedding=False, use_reference_token=True,
                     train_on_residual_dvf=True, warp_head_type=head).to(DEV).eval()
        ck = torch.load(os.path.join(_ROOT, ckpt), map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(ck["model"], strict=False)
        assert not miss and not unexp, f"{name}: missing={miss[:4]} unexpected={unexp[:4]}"
        for ds, jobs in DATASETS.items():
            for subj, adapter, ed in jobs:
                try:
                    cine = adapter.load()
                    batch, picks = build_ed_batch(cine, adapter.inplane_mm(),
                                                  adapter.slice_positions_mm(), ed,
                                                  np.random.default_rng(0), DEV)
                except Exception as e:
                    print(f"  skip {ds}/{subj}: {e}"); continue
                V, dvf = forward(model, batch)
                Vin = input_volume(batch)
                render_io(Vin, V, f"{name} · {ds}_{subj} (target ED, f={ed})  —  INPUT | V_canon",
                          os.path.join(out_dir, f"{ds}_{subj}_io.png"))
                render_dvf(batch, dvf, f"DVF — {name} · {ds}_{subj} (target ED)",
                           os.path.join(out_dir, f"{ds}_{subj}_dvf.png"), picks)
    print(f"done -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
