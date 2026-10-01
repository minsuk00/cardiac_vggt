"""E18 — where does anatomy live in the TRAINED model's tokens? Same 1-NN cross-subject label-transfer probe as
e17, but on the final model's aggregator outputs, layer by layer (docs/128 §3).

Model: final518_diff1000_curated898 (harness.CKPT, or INTERP_CKPT). Inputs built exactly like val
(harness.Subject + build_batch): canonical 1.4 mm / 256² grid -> 518², all slots at ED (t=0), no breathing.
Only the aggregator is run (the DPT point head is not needed for features).
Subjects: test split of pooled_curated_v2 with heart_seg_canonical, paired ACROSS sources (different
dataset) by closest ED LV volume (greedy, random_state=0, without replacement).
Probed slots: slot 0 (reference, mid plane) and the companion slot at plane ref+1.
Layers: frozen DINOv2 output (patch_embed) and global-block outputs 0,3,7,11,15,19,23 (patch tokens only;
the 5 special tokens per slot are dropped).
Labels: heart_seg_canonical (1 LV, 2 MYO, 3 RV) at ED, majority label per 14x14 patch of the 518 grid.
Per layer: 1-NN label transfer B->A (cos-sim after subtracting the joint mean token), per-class patch Dice,
plus collapse diagnostics (uniq_nn = #distinct B patches used as a NN; nn_offset_heart = mean grid distance in
patches between an A heart patch and its NN in B).

Run from the repo root (~3 min on one GPU; sharing an already-busy GPU is fine, see ~/CLAUDE.md GPU rule):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=training:. micromamba run -n svr python tools/interp/e18_aggregator_anatomy_probe.py [N_PAIRS]
Env: PROBE_DEV (default cuda:0), PROBE_SAVE_PAIRS (default "3,13,15": pairs whose ref-slot features are saved for e18_viz.py).
Outputs: temp/aggft_probe/results.csv, temp/aggft_probe/pairNN.pt
"""
import os, sys, time
import numpy as np, pandas as pd, nibabel as nib, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

NPAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
PICK = {int(x) for x in os.environ.get("PROBE_SAVE_PAIRS", "3,13,15").split(",") if x}
OUT = os.path.join(H.ROOT, "temp/aggft_probe")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(48)
DEV, G, NAMES = os.environ.get("PROBE_DEV", "cuda:0"), 37, ["bg", "LV", "MYO", "RV"]
GLOBAL = [0, 3, 7, 11, 15, 19, 23]

model, cfg = H.load_model_from_run(H.CKPT, device=DEV)  # plain model, no attention hooks
model.float().eval()
ds = H.make_test_dataset(cfg, "test")
print("test subjects:", len(ds.subjects), flush=True)

def source(p): return [s for s in p.split("/") if s.endswith("_sax") or s.startswith("CMRxRecon")][0]
def seg_ed(p):  # (D,H,W) at ED, splat order (same (X,Y,Z)->(Z,Y,X) transpose as MRIDataset's heart_roi_canonical)
    return np.transpose(np.asarray(nib.load(os.path.join(p, "heart_seg_canonical.nii.gz")).dataobj)[..., 0], (2, 1, 0))

meta = []
for i, p in enumerate(ds.subjects):
    if not os.path.exists(os.path.join(p, "heart_seg_canonical.nii.gz")): continue
    im = nib.load(os.path.join(p, "heart_seg_canonical.nii.gz")); dz = float(im.header.get_zooms()[2])
    meta.append((i, p, source(p), (seg_ed(p) == 1).sum() * 1.4 * 1.4 * dz / 1000))
meta = pd.DataFrame(meta, columns=["idx", "path", "src", "lv_ml"])
print(meta.src.value_counts().to_string(), flush=True)

# greedy cross-source pairing by closest LV volume, without replacement
pool, pairs = meta.copy(), []
for _, a in meta.sample(frac=1, random_state=0).iterrows():
    if a.idx not in set(pool.idx) or len(pairs) >= NPAIRS: continue
    pool = pool[pool.idx != a.idx]
    c = pool[pool.src != a.src]
    if not len(c): continue
    b = c.loc[(c.lv_ml - a.lv_ml).abs().idxmin()]
    pool = pool[pool.idx != b.idx]; pairs.append((a, b))
print(f"{len(pairs)} pairs, median |dLV| {np.median([abs(a.lv_ml - b.lv_ml) for a, b in pairs]):.1f} mL", flush=True)

class Capture:
    def __init__(self, agg):
        self.out, self.hs = {}, [agg.patch_embed.register_forward_hook(self._pe)]
        for i in GLOBAL: self.hs.append(agg.global_blocks[i].register_forward_hook(self._g(i)))
    def _pe(self, m, inp, out): self.out["dino"] = (out["x_norm_patchtokens"] if isinstance(out, dict) else out).detach().float().cpu()
    def _g(self, i):
        def f(m, inp, out): self.out[f"g{i}"] = out.detach().float().cpu()
        return f
cap = Capture(model.aggregator)

cache = {}
def feats(row):
    """-> ({layer: {slot_name: (1369,C)}, "_img": {slot_name: (518,518)}}, {slot_name: patch labels (1369,)}, name)"""
    if row.idx in cache: return cache[row.idx]
    subj = H.Subject(ds, int(row.idx), DEV)
    batch = H.build_batch(subj, np.zeros(subj.S, dtype=int))
    t0 = time.time()
    with torch.no_grad():
        model.aggregator(batch["images"], z_indices=batch["z_indices"])
    dt = time.time() - t0
    seg = torch.from_numpy(seg_ed(row.path).astype(np.int64))
    slots = {"ref": 0}
    comp = np.where(subj.slot_z == subj.ref_plane + 1)[0]
    if len(comp): slots["comp"] = int(comp[0])
    f, lab = {}, {}
    for name, s in slots.items():
        sl = F.interpolate(seg[int(subj.slot_z[s])][None, None].float(), size=(518, 518), mode="nearest")[0, 0].long()
        lab[name] = F.avg_pool2d(F.one_hot(sl, 4).permute(2, 0, 1).float()[None], 14).argmax(1).reshape(-1)
    for L, x in cap.out.items():
        x = x.reshape(subj.S, -1, x.shape[-1])
        x = x if L == "dino" else x[:, H.N_SPECIAL:]
        assert x.shape[1] == G * G, (L, x.shape)
        f[L] = {name: x[s] for name, s in slots.items()}
    f["_img"] = {name: batch["images"][0, s, 0].float().cpu() for name, s in slots.items()}  # (518,518) model input
    cache[row.idx] = (f, lab, subj.name)
    print(f"  forward {subj.name}: S={subj.S} {dt:.0f}s", flush=True)
    return cache[row.idx]

def dice(pred, gt):
    return {NAMES[c]: (2 * ((pred == c) & (gt == c)).sum() / ((pred == c).sum() + (gt == c).sum()).clamp(min=1)).item() for c in (1, 2, 3)}

def nn_transfer(fa, fb, lb):
    m = torch.cat([fa, fb]).mean(0); fa, fb = F.normalize(fa - m, dim=-1), F.normalize(fb - m, dim=-1)
    return lb[(fa @ fb.T).argmax(1)]

def diag(fa, fb, la):
    """collapse diagnostics: #distinct B patches used as a NN; mean grid offset (patches) of A heart patches to their NN in B."""
    m = torch.cat([fa, fb]).mean(0); na, nb = F.normalize(fa - m, dim=-1), F.normalize(fb - m, dim=-1)
    nn_ = (na @ nb.T).argmax(1)
    yx = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"), -1).reshape(-1, 2).float()
    off = (yx[nn_] - yx)[la > 0].norm(dim=1).mean().item()
    return {"uniq_nn": nn_.unique().numel(), "nn_offset_heart": off}

rows = []
for k, (a, b) in enumerate(pairs):
    fa, la, na = feats(a); fb, lb, nb = feats(b)
    for slot in ("ref", "comp"):
        if slot not in la or slot not in lb or not all((la[slot] == c).any() and (lb[slot] == c).any() for c in (1, 2, 3)):
            continue  # need all three classes present on both slices
        rows.append({"pair": k, "A": na, "B": nb, "slot": slot, "layer": "same location", **dice(lb[slot], la[slot])})
        for L in fa:
            if L == "_img": continue
            rows.append({"pair": k, "A": na, "B": nb, "slot": slot, "layer": L, **dice(nn_transfer(fa[L][slot], fb[L][slot], lb[slot]), la[slot]),
                         **diag(fa[L][slot], fb[L][slot], la[slot])})
    if k in PICK:  # ref-slot features + labels + images for visualisation
        torch.save({"A": na, "B": nb, "fa": {L: v["ref"].half() for L, v in fa.items()}, "fb": {L: v["ref"].half() for L, v in fb.items()},
                    "la": la["ref"], "lb": lb["ref"]}, os.path.join(OUT, f"pair{k:02d}.pt"))
    print(f"[{k + 1}/{len(pairs)}] {na} ({a.src}, {a.lv_ml:.0f} mL) <-> {nb} ({b.src}, {b.lv_ml:.0f} mL)", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "results.csv"), index=False)

df = pd.DataFrame(rows)
order = ["same location", "dino"] + [f"g{i}" for i in GLOBAL]
for slot in ("ref", "comp"):
    d = df[df.slot == slot]
    if not len(d): continue
    g = d.groupby("layer")[["LV", "MYO", "RV", "uniq_nn", "nn_offset_heart"]]
    print(f"\n== slot {slot}: per-class Dice, mean ± sd over {d.pair.nunique()} pairs")
    print((g.mean().round(2).astype(str) + " ± " + g.std().round(2).astype(str)).reindex(order).to_string())
