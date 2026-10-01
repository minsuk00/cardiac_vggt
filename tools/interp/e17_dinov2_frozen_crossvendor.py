"""E17 — does FROZEN DINOv2 (VGGT's patch_embed) encode heart anatomy independent of contrast? (docs/128 §2)

40 CMRx24 (Siemens) subjects, each paired with a distinct non-Siemens M&Ms subject (GE/Philips/Canon) of the
closest ED LV volume (greedy, without replacement). Per subject: ED mid-ventricular slice (middle of the LV's
z-extent), cropped to a 160 mm square centred on the LV centroid, resampled to 518², percentile-normalised.
NOTE: native 3d_recon/heart_seg files, NOT the training canonical pipeline (that is e18).

Conditions (patch labels transferred onto A by 1-nearest-neighbour from the other image):
  cross   : B = the volume-matched other-vendor subject
  ctrl    : A' = A with a non-monotonic contrast remap, same pose (identical anatomy)
  ctrl_rot: A' rotated 90° (removes any positional shortcut)
Methods: DINOv2 1-NN | raw-pixel 1-NN (14x14 pixel patches) | same location (B's label at the same patch index).
Metric: per-class Dice (LV/MYO/RV) on the 37x37 PATCH grid vs A's own patch labels (majority label per patch).

Run from the repo root (CPU is fine, ~5 min):
    CUDA_VISIBLE_DEVICES= micromamba run -n svr python tools/interp/e17_dinov2_frozen_crossvendor.py [N_PAIRS]
Outputs: temp/dinov2_probe/results.csv (per pair x cond x method), temp/dinov2_probe/maps.pt (for e17_viz.py).
"""
import glob, os, shutil, sys
import numpy as np, nibabel as nib, pandas as pd, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from vggt.layers.vision_transformer import vit_large  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
D, CROP_MM, NAMES = os.path.join(ROOT, "scratch/data"), 160.0, ["bg", "LV", "MYO", "RV"]
OUT = os.path.join(ROOT, "temp/dinov2_probe")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16)

def ed(d):
    im = nib.load(f"{d}/3d_recon/sax_frame_00.nii.gz")
    return im.get_fdata(dtype=np.float32), np.asanyarray(nib.load(f"{d}/heart_seg.nii.gz").dataobj)[..., 0], im.header.get_zooms()[:3]

def lv_ml(d):
    s = np.asanyarray(nib.load(f"{d}/heart_seg.nii.gz").dataobj)[..., 0]
    return (s == 1).sum() * np.prod(nib.load(f"{d}/heart_seg.nii.gz").header.get_zooms()[:3]) / 1000

def crop(d):
    v, s, (sx, sy, _) = ed(d)
    zs = np.where((s == 1).any((0, 1)))[0]; z = zs[len(zs) // 2]
    v, s = v[:, :, z].T, s[:, :, z].T
    cy, cx = np.argwhere(s == 1).mean(0); hy, hx = CROP_MM / 2 / sy, CROP_MM / 2 / sx
    ys = np.linspace(cy - hy, cy + hy, 518); xs = np.linspace(cx - hx, cx + hx, 518)
    grid = torch.stack(torch.meshgrid(torch.tensor(xs / (v.shape[1] - 1) * 2 - 1), torch.tensor(ys / (v.shape[0] - 1) * 2 - 1), indexing="xy"), -1)[None].float()
    img = F.grid_sample(torch.from_numpy(v).float()[None, None], grid, align_corners=True)[0, 0]
    seg = F.grid_sample(torch.from_numpy(s.astype(np.float32))[None, None], grid, mode="nearest", align_corners=True)[0, 0].long()
    lo, hi = np.percentile(img[img > 0], [0.5, 99.5]); return ((img - lo) / (hi - lo)).clamp(0, 1), seg

remap = lambda a: (1 - (2 * a - 1) ** 2) ** 0.7 * 0.9 + 0.1 * (1 - a)  # non-monotonic: mid-grey -> bright, bright blood -> dark

# --- pairing: greedy, without replacement, closest ED LV volume
info = pd.read_csv(glob.glob(f"{D}/MNMs/MNMs1/*information*.csv")[0])
vendor = dict(zip(info["External code"], info["VendorName"]))
mnms = {p: lv_ml(p) for p in sorted(glob.glob(f"{D}/MNMs_sax/MNMs_*/sax")) if vendor.get(p.split("/")[-2][5:], "Siemens") != "Siemens"}
cmrx = sorted(glob.glob(f"{D}/CMRxRecon2024/Cine_combined/CMRx24_Train_*/sax/heart_seg.nii.gz"))[:N]
pairs = []
for p in cmrx:
    a = os.path.dirname(p); va = lv_ml(a)
    b = min(mnms, key=lambda q: abs(mnms[q] - va)); pairs.append((a, b, va, mnms.pop(b)))
print(f"{len(pairs)} pairs, |ΔLV| median {np.median([abs(x[2] - x[3]) for x in pairs]):.1f} mL")

# --- frozen DINOv2 = the aggregator.patch_embed weights of the VGGT-1B base ckpt (stage to /tmp: GPFS torch.load is slow)
dst = f"/tmp/vggt1b_base_{os.environ.get('USER', 'user')}.pt"
if not os.path.exists(dst): shutil.copy(os.path.join(ROOT, "scratch/base_weights/vggt1b_base.pt"), dst)
sd = torch.load(dst, map_location="cpu", weights_only=True); sd = sd.get("model", sd)
vit = vit_large(img_size=518, patch_size=14, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0, block_chunks=0, init_values=1.0)
vit.load_state_dict({k.split("aggregator.patch_embed.", 1)[1]: v for k, v in sd.items() if "aggregator.patch_embed." in k}); vit.eval()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # aggregator's _resnet_mean/std

def plabels(seg): return F.avg_pool2d(F.one_hot(seg, 4).permute(2, 0, 1).float()[None], 14).argmax(1).reshape(-1)

def dice(pred, gt):
    return {NAMES[c]: (2 * ((pred == c) & (gt == c)).sum() / ((pred == c).sum() + (gt == c).sum()).clamp(min=1)).item() for c in (1, 2, 3)}

def nn_transfer(fa, fb, lb):
    # subtract the joint mean token first: without it every cos-sim is ~0.8 (shared DC component)
    m = torch.cat([fa, fb]).mean(0); fa, fb = F.normalize(fa - m, dim=-1), F.normalize(fb - m, dim=-1)
    return lb[(fa @ fb.T).argmax(1)]

rows, maps = [], []
for k, (a, b, va, vb) in enumerate(pairs):
    ai, asg = crop(a); bi, bsg = crop(b); ci = remap(ai)
    ri, rsg = torch.rot90(ci, 1), torch.rot90(asg, 1)
    imgs = torch.stack([ai, bi, ci, ri])[:, None].repeat(1, 3, 1, 1)
    with torch.no_grad():
        tok = vit((imgs - mean) / std, is_training=True)["x_norm_patchtokens"]  # (4,1369,1024)
    pix = F.unfold(imgs[:, :1], 14, stride=14).transpose(1, 2)
    la = plabels(asg)
    keep = {"a": (ai * 255).byte(), "b": (bi * 255).byte(), "r": (ri * 255).byte(), "la": la, "names": (a.split("/")[-2], b.split("/")[-2], vendor[b.split("/")[-2][5:]])}
    for cond, j, lj in [("cross", 1, plabels(bsg)), ("ctrl", 2, la), ("ctrl_rot", 3, plabels(rsg))]:
        for meth, pred in [("DINOv2", nn_transfer(tok[0], tok[j], lj)), ("raw pixels", nn_transfer(pix[0], pix[j], lj)), ("same location", lj)]:
            rows.append({"pair": k, "cond": cond, "method": meth, **dice(pred, la)})
            if cond != "ctrl" and meth != "raw pixels": keep[f"{cond}/{meth}"] = pred
        keep[f"{cond}/src"] = lj
    maps.append(keep)
    print(f"[{k + 1}/{len(pairs)}] {a.split('/')[-2]} ({va:.0f} mL) ↔ {b.split('/')[-2]} {vendor[b.split('/')[-2][5:]]} ({vb:.0f} mL)", flush=True)

df = pd.DataFrame(rows); df.to_csv(f"{OUT}/results.csv", index=False)
torch.save(maps, f"{OUT}/maps.pt")
g = df.groupby(["cond", "method"])[["LV", "MYO", "RV"]]
print("\nper-class Dice of transferred patch labels, mean ± sd over pairs")
print((g.mean().round(2).astype(str) + " ± " + g.std().round(2).astype(str)).to_string())
