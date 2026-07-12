"""ONE GPU pass: load each model once, compute recon(3 planes)+DVF(3 slots)+coverage(3 planes).
Saves per-model npz immediately (robust to GPFS stalls). As-trained z per model."""
import numpy as np, glob, sys, gc, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from inference.inference import load_rtfb_model_reference, forward
from inference.adapters.miitt import MIITTAdapter
from inference.adapters.base import to_canonical_inplane, MM_PER_NORM, D_CANON, INPUT_IMG_SIZE

dev = "cuda"; NF = 30; subj = "Volunteer1"
ad = MIITTAdapter(f"scratch/data/MIITT/nifti/{subj}/realtime/sax/4d_recon.nii.gz")
_, _, _, rc0 = ad.build_batch_multiframe(dev, 1, 1, continuous_z=False)
vmin, vmax = rc0["scale"]; inpl = rc0["inplane"]
def to256(a): idx = np.linspace(0, a.shape[0]-1, 256).astype(int); return a[np.ix_(idx, idx)]

specs = [("control0", "216539845_*ftctrl_gather0*1frame*", False),
         ("gather05", "216539845_*ftgather05*1frame*", False),
         ("s20contz", "216949414_*s20contz*", True)]
for name, pat, cz in specs:
    ck = glob.glob(f"scratch/logs/{pat}/ckpts/checkpoint_last.pt")[0]
    print(f"loading {name} ...", flush=True)
    model = load_rtfb_model_reference(ck, refiner=False, device=dev)
    b, S, _, rcx = ad.build_batch_multiframe(dev, 1, 1, continuous_z=cz)
    sc = b["scanner_coords"][0].cpu().numpy()
    slot_plane = (sc[:, 0, 0, 2] + 1) / 2 * (D_CANON - 1)
    refp = int(round(float(slot_plane[0])))
    nlo = int(np.argmin(np.abs(slot_plane - (refp - 1)))); nhi = int(np.argmin(np.abs(slot_plane - (refp + 1))))
    slots = [nlo, 0, nhi]; planes = [max(0, refp-1), refp, min(D_CANON-1, refp+1)]
    cs = rcx["cine_slice"]; fr = list(np.linspace(0, cs.shape[0]-1, NF).astype(int))
    RE = np.zeros((NF, 3, 256, 256), np.float32); DV = np.zeros((NF, 3, 256, 256, 3), np.float32); CO = np.zeros((NF, 3, 256, 256), np.float32)
    for ti, f in enumerate(fr):
        up = F.interpolate(to_canonical_inplane(np.clip((cs[f]-vmin)/(vmax-vmin), 0, 1), inpl)[None, None],
                           size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE), mode="bilinear", align_corners=True)[0, 0].numpy()
        b["images"][:, 0] = torch.from_numpy(up).to(dev).repeat(3, 1, 1)
        r = forward(model, b, want=("V", "world_points", "coverage"), device=dev)
        RE[ti] = r["V"][planes]
        for si, slot in enumerate(slots):
            d = (r["world_points"][slot] - sc[slot]) * np.array(MM_PER_NORM)[None, None, :]
            DV[ti, si] = np.stack([to256(d[..., k]) for k in range(3)], -1)
        CO[ti] = r["coverage"][planes]
    np.savez_compressed(f"result/viz_{name}_V1.npz", recon=RE, dvf=DV, cov=CO,
                        refp=refp, planes=np.array(planes), cz=cz)
    print(f"{name} DONE refp={refp} planes={planes} S={S}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
print("ALL DONE", flush=True)
