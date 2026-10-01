"""Interpretability harness for the final CardioStitch model (final518_diff1000, docs/125+).

One place that (1) loads the model, (2) builds a fully-controlled input stack for a test subject
(per-slot cardiac phase, per-slot breathing displacement, optional z-token override), (3) runs the
forward pass with optional GLOBAL-ATTENTION interventions (group-level key masks), (4) records
global-attention statistics without materialising the full (S*P)^2 matrix, and (5) scores the
output (per-slot through-plane shift, splatted volume, NCC vs every GT phase).

Everything geometric is taken from `MRIDataset.get_data` (the training implementation); inputs are
re-extracted with the training code's own `extract_slices_with_respiratory_vec` +
`_resize_to_model_res`, so an unmodified call reproduces the trainer's val path.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "training"), os.path.join(ROOT, "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                              # noqa: E402
import torch                                                                    # noqa: E402
import torch.nn.functional as F                                                 # noqa: E402
from omegaconf import OmegaConf                                                 # noqa: E402

from data.datasets.mri_dataset import MRIDataset                                # noqa: E402
from data.gpu_aug import _resize_to_model_res                                   # noqa: E402
from data.respiratory import (RespiratoryConfig, extract_slices_with_respiratory_vec,  # noqa: E402
                              sample_displacement_vectors)
from data.preprocess import Z_HALF_MM                                           # noqa: E402
from loss import _splat_preds_native, _resize_field                             # noqa: E402
from vggt.utils.splat import sample_volume                                      # noqa: E402
from inference.load_run import load_model_from_run, mri_dataset_kwargs          # noqa: E402

CKPT = os.environ.get("INTERP_CKPT") or os.path.join(
    ROOT, "scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt")
INPLANE_MM = 1.4
MM_PER_NORM = np.array([0.5 * 255 * INPLANE_MM, 0.5 * 255 * INPLANE_MM, Z_HALF_MM])  # (x, y, z)
N_SPECIAL = 5          # 1 camera(+z+ref) token + 4 registers per slot (patch_start_idx)


# ───────────────────────────── model + data ─────────────────────────────
def load(device="cuda:0", ckpt=CKPT):
    model, cfg = load_model_from_run(ckpt, device=device)
    install_global_attention_hooks(model)
    return model, cfg


def make_test_dataset(cfg, split="test"):
    kw = dict(mri_dataset_kwargs(cfg, "val"))
    kw.pop("ef_val_sweep", None)
    kw.pop("_target_", None)
    kw["split"] = split
    kw["t_target_fixed"] = 0
    if kw.get("split_file") and not os.path.isabs(kw["split_file"]):
        kw["split_file"] = os.path.join(ROOT, kw["split_file"])   # MRIDataset silently returns [] if not found
    data_root = kw.pop("data_root")
    common = OmegaConf.create({"img_size": int(cfg.get("img_size", 518)), "patch_size": 14,
                               "rescale": True, "rescale_aug": False, "landscape_check": False,
                               "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(common, data_root, **kw)


def resp_cfg(cfg):
    return RespiratoryConfig.from_cfg(((cfg.get("data") or {}).get("augmentation") or {}).get("respiratory"))


class Subject:
    """Static per-subject tensors (clean cine on the canonical grid + slot geometry)."""

    def __init__(self, ds, idx, device):
        b = ds.get_data(seq_index=idx, img_per_seq=ds.num_slices)
        self.idx = idx
        self.name = b["seq_name"]
        self.device = device
        self.phases = torch.from_numpy(np.asarray(b["phases"])).float().to(device)      # (T,D,H,W)
        self.T, self.D, self.H, self.W = self.phases.shape
        self.dz = float(np.asarray(b["dz_mm"]).reshape(-1)[0])
        self.z_scale = float(np.asarray(b["z_scale"]).reshape(-1)[0])
        self.slot_z = np.array([int(round(float(z))) for z in b["slice_indices"]])      # (S,)
        self.S = len(self.slot_z)
        self.scanner_coords = torch.from_numpy(np.stack(b["scanner_coords"])).float().to(device)  # (S,R,R,3)
        self.z_indices = torch.from_numpy(np.stack(b["z_indices"])).float().to(device)   # (S,1)
        self.R = self.scanner_coords.shape[1]
        roi = b.get("heart_roi_canonical")
        self.heart = None if roi is None else torch.from_numpy(np.asarray(roi)).bool().to(device)  # (D,H,W)
        self.content = torch.from_numpy(np.asarray(b["content_mask"])).bool().to(device)
        self.ref_plane = int(self.slot_z[0])

    def sample_breathing(self, rcfg, seed):
        """Per-slot (S,3) mm displacement drawn exactly like val: per-plane phase, per-subject amp/tilt."""
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        gid = torch.from_numpy(self.slot_z)[None].to(self.device)
        v, r = sample_displacement_vectors(1, self.S, rcfg, self.device, generator=g,
                                           group_ids=gid, n_planes=self.D)
        return v[0], r[0]


def build_batch(subj, t_slots, disp=None, z_override=None, slot_z=None):
    """Controlled input stack.
    t_slots: (S,) cardiac phase per slot (slot 0 = reference phase = target).
    disp:    (S,3) mm breathing shift per slot (None = none). Model must undo it.
    z_override: (S,) mm to ADD to the z-token only (image + scanner_coords unchanged) — a lie about
                the slice position used for the z-token dose-response experiment.
    slot_z:  permuted/custom plane per slot (default: the dataset draw).
    """
    dev = subj.device
    sz = subj.slot_z if slot_z is None else np.asarray(slot_z)
    S = len(sz)
    t = torch.as_tensor(np.asarray(t_slots), dtype=torch.int64, device=dev)[None]
    z = torch.as_tensor(sz, dtype=torch.int64, device=dev)[None]
    d = torch.zeros(1, S, 3, device=dev) if disp is None else torch.as_tensor(disp, dtype=torch.float32, device=dev).reshape(1, S, 3)
    native = extract_slices_with_respiratory_vec(subj.phases[None], t, z, d, spacing=(subj.dz, 1.4, 1.4),
                                                 out_size=subj.H)[..., 0] / 255.0   # (1,S,H,W)
    # scanner_coords / z_indices follow the plane (pure geometry of the nominal plane)
    idx = [int(np.where(subj.slot_z == zz)[0][0]) for zz in sz]
    sc = subj.scanner_coords[idx].clone()
    zi = subj.z_indices[idx].clone()
    if z_override is not None:
        zi = zi + torch.as_tensor(np.asarray(z_override), dtype=torch.float32, device=dev).view(S, 1) / Z_HALF_MM
    return {"images": _resize_to_model_res(native, subj.R), "images_splat": native,
            "scanner_coords": sc[None], "z_indices": zi[None], "slot_z": sz, "t_slots": np.asarray(t_slots),
            "disp": d[0], "z_scale": torch.tensor([subj.z_scale], device=dev)}


# ───────────────────────────── attention control ─────────────────────────────
class AttnCtl:
    """Global-attention intervention + recording state, read by the patched forward.

    Groups: every token gets a group id = slot*2 + (0 special | 1 patch).  `allow` is a
    (2S, 2S) bool matrix [query_group, key_group]; None = unmasked. `layers` restricts the
    mask to a set of global-block indices (None = all 24).
    `record`: set of global-block indices whose attention statistics are accumulated.
    """

    def __init__(self):
        self.allow = None
        self.layers = None
        self.record = set()
        self.stats = {}
        self.S = None
        self.P = None
        self.query_points = []   # list of (slot, patch_row, patch_col) whose full maps are saved
        self.maps = {}
        self.heart_patch = None  # (S, P-5) bool: patch token covers heart ROI (for heart-query stats)
        self.grid = None
        self.key_hide = None     # (N,) bool token-level key mask (with query_sel)
        self.query_sel = None    # (N,) bool
        self.kv_mode = None      # None | 'save' | 'patch' — slot-0 key/value activation patching
        self.kv_layers = set()
        self.kv_heads = None     # None = all heads; else list of head indices to patch
        self.kv_tok = None       # None = all slot-0 tokens; else (P,) bool of slot-0 tokens to patch
        self.kv_store = KV_STORE
        self.full_layers = set()  # global-block indices whose full (N,N) head-mean attention is saved (pooled)
        self.full_pool = 6
        self.full = {}

    def reset(self):
        self.__init__()


KV_STORE = {}     # survives CTL.reset() so a 'save' run can feed a later 'patch' run
CTL = AttnCtl()


def _group_ids(S, P, device):
    slot = torch.arange(S, device=device).repeat_interleave(P)
    typ = (torch.arange(P, device=device) >= N_SPECIAL).long().repeat(S)
    return slot * 2 + typ


def install_global_attention_hooks(model):
    for li, blk in enumerate(model.aggregator.global_blocks):
        attn = blk.attn
        attn._li = li

        def fwd(x, pos=None, _a=attn):
            B, N, C = x.shape
            qkv = _a.qkv(x).reshape(B, N, 3, _a.num_heads, _a.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = _a.q_norm(q), _a.k_norm(k)
            if _a.rope is not None:
                q = _a.rope(q, pos)
                k = _a.rope(k, pos)
            mask = None
            if CTL.allow is not None and (CTL.layers is None or _a._li in CTL.layers):
                g = _group_ids(CTL.S, CTL.P, x.device)
                mask = CTL.allow.to(x.device)[g][:, g]            # (N,N) bool, True = may attend
            if CTL.key_hide is not None and (CTL.layers is None or _a._li in CTL.layers):
                # token-level: queries in `query_sel` may not attend keys in `key_hide`
                tm = ~(CTL.query_sel.to(x.device)[:, None] & CTL.key_hide.to(x.device)[None, :])
                mask = tm if mask is None else (mask & tm)
            if CTL.kv_mode is not None and _a._li in CTL.kv_layers:
                P0 = CTL.P                                        # slot-0 tokens are [0, P)
                if CTL.kv_mode == "save":
                    CTL.kv_store[_a._li] = (k[:, :, :P0].clone(), v[:, :, :P0].clone())
                elif CTL.kv_mode == "patch":
                    ks, vs = CTL.kv_store[_a._li]
                    # companions read the donor run's slot-0 keys/values; slot 0's OWN queries
                    # also see them, so restrict: patch via a separate attention for slot 0.
                    k2, v2 = k.clone(), v.clone()
                    hs = slice(None) if CTL.kv_heads is None else list(CTL.kv_heads)
                    if CTL.kv_tok is None:
                        k2[:, hs, :P0], v2[:, hs, :P0] = ks[:, hs], vs[:, hs]
                    else:                                        # only a subset of slot-0 tokens
                        ti = CTL.kv_tok.nonzero()[:, 0].to(x.device)
                        for h in (range(k2.shape[1]) if CTL.kv_heads is None else CTL.kv_heads):
                            k2[:, h, ti], v2[:, h, ti] = ks[:, h, ti], vs[:, h, ti]
                    xc = F.scaled_dot_product_attention(q[:, :, P0:], k2, v2, attn_mask=None if mask is None else mask[P0:])
                    x0 = F.scaled_dot_product_attention(q[:, :, :P0], k, v, attn_mask=None if mask is None else mask[:P0])
                    x = torch.cat([x0, xc], dim=2).transpose(1, 2).reshape(B, N, C)
                    return _a.proj_drop(_a.proj(x))
            if _a._li in CTL.record:
                _record(_a._li, q, k, mask)
            if _a._li in CTL.full_layers:
                _record_full(_a._li, q, k, mask)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x.transpose(1, 2).reshape(B, N, C)
            return _a.proj_drop(_a.proj(x))

        attn.forward = fwd


@torch.no_grad()
def _record_full(li, q, k, mask, chunk=600):
    """Full head-mean attention matrix (B=1), pooled by CTL.full_pool along both axes: rows averaged over
    p consecutive query tokens, columns summed over p consecutive key tokens (so each row still sums to 1)."""
    p = CTL.full_pool
    qf, kf = q[0].float(), k[0].float()
    N = qf.shape[1]
    assert N % p == 0 and chunk % p == 0
    scale = q.shape[-1] ** -0.5
    out = torch.zeros(N // p, N // p)
    for s0 in range(0, N, chunk):
        s1 = min(N, s0 + chunk)
        logits = torch.einsum("hqd,hkd->hqk", qf[:, s0:s1], kf) * scale
        if mask is not None:
            logits = logits.masked_fill(~mask[s0:s1][None], float("-inf"))
        a = logits.softmax(-1).mean(0)                                   # (n,N), mean over heads
        a = a.view((s1 - s0) // p, p, N).mean(1).view(-1, N // p, p).sum(2)
        out[s0 // p:s1 // p] = a.cpu()
    CTL.full[li] = out


@torch.no_grad()
def _record(li, q, k, mask, chunk=512):
    """Accumulate per-head slot->slot attention mass without storing the (N,N) matrix.
    For every query token (B=1) we compute softmax rows chunk-wise and reduce to:
      mass[h, qslot, kslot]           (patch queries only, all keys of kslot incl. special)
      mass_sp[h, qslot, kslot]        (mass on kslot's 5 special tokens)
      same[h, qslot, kslot]           (mass on the key patch at the SAME (row,col) in kslot)
      nb1[h, qslot, kslot]            (mass on the 3x3 neighbourhood of the same (row,col))
    and the same four restricted to heart-ROI query patches (suffix _h). Counts for normalising.
    """
    S, P = CTL.S, CTL.P
    Hh = q.shape[1]
    G = int(round((P - N_SPECIAL) ** 0.5))
    qf, kf = q[0].float(), k[0].float()                   # (h,N,d)
    scale = q.shape[-1] ** -0.5
    N = qf.shape[1]
    st = CTL.stats.setdefault(li, {n: torch.zeros(Hh, S, S, device=q.device, dtype=torch.float64)
                                   for n in ("mass", "mass_sp", "same", "nb1", "mass_h", "mass_sp_h",
                                             "same_h", "nb1_h")})
    st.setdefault("cnt", torch.zeros(S, device=q.device, dtype=torch.float64))
    st.setdefault("cnt_h", torch.zeros(S, device=q.device, dtype=torch.float64))
    st.setdefault("ent", torch.zeros(Hh, S, device=q.device, dtype=torch.float64))
    # neighbourhood offsets (in patch-grid units)
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    hp = CTL.heart_patch                                   # (S, G*G) bool or None
    for s0 in range(0, N, chunk):
        s1 = min(N, s0 + chunk)
        tok = torch.arange(s0, s1, device=q.device)
        qslot = tok // P
        qp = tok % P
        keep = qp >= N_SPECIAL                             # patch queries only
        if not keep.any():
            continue
        tok, qslot, qp = tok[keep], qslot[keep], qp[keep]
        logits = torch.einsum("hqd,hkd->hqk", qf[:, tok], kf) * scale
        if mask is not None:
            logits = logits.masked_fill(~mask[tok][None], float("-inf"))
        a = logits.softmax(-1)                             # (h,n,N)
        n = a.shape[1]
        ent = -(a.clamp_min(1e-12).log() * a).sum(-1)      # (h,n)
        a4 = a.view(Hh, n, S, P)
        m_slot = a4.sum(-1)                                # (h,n,S)
        m_sp = a4[..., :N_SPECIAL].sum(-1)
        pr = (qp - N_SPECIAL) // G
        pc = (qp - N_SPECIAL) % G
        same_idx = (N_SPECIAL + pr * G + pc)               # (n,)
        m_same = a4.gather(-1, same_idx.view(1, n, 1, 1).expand(Hh, n, S, 1)).squeeze(-1)
        m_nb = torch.zeros_like(m_same)
        for dr, dc in offs:
            rr, cc = pr + dr, pc + dc
            ok = (rr >= 0) & (rr < G) & (cc >= 0) & (cc < G)
            idx = N_SPECIAL + rr.clamp(0, G - 1) * G + cc.clamp(0, G - 1)
            v = a4.gather(-1, idx.view(1, n, 1, 1).expand(Hh, n, S, 1)).squeeze(-1)
            m_nb += v * ok.view(1, n, 1)
        oh = F.one_hot(qslot, S).double()                  # (n,S)
        st["mass"] += torch.einsum("hnk,nq->hqk", m_slot.double(), oh)
        st["mass_sp"] += torch.einsum("hnk,nq->hqk", m_sp.double(), oh)
        st["same"] += torch.einsum("hnk,nq->hqk", m_same.double(), oh)
        st["nb1"] += torch.einsum("hnk,nq->hqk", m_nb.double(), oh)
        st["cnt"] += oh.sum(0)
        st["ent"] += torch.einsum("hn,nq->hq", ent.double(), oh)
        if hp is not None:
            inh = hp[qslot, qp - N_SPECIAL]
            if inh.any():
                ohh = oh * inh.double()[:, None]
                st["mass_h"] += torch.einsum("hnk,nq->hqk", m_slot.double(), ohh)
                st["mass_sp_h"] += torch.einsum("hnk,nq->hqk", m_sp.double(), ohh)
                st["same_h"] += torch.einsum("hnk,nq->hqk", m_same.double(), ohh)
                st["nb1_h"] += torch.einsum("hnk,nq->hqk", m_nb.double(), ohh)
                st["cnt_h"] += ohh.sum(0)
        # full maps for requested query points
        for (qs, r_, c_) in CTL.query_points:
            t_id = qs * P + N_SPECIAL + r_ * G + c_
            hit = (tok == t_id).nonzero()
            if len(hit):
                CTL.maps[(li, qs, r_, c_)] = a[:, int(hit[0])].view(Hh, S, P).cpu()


def finalize_stats():
    out = {}
    for li, st in CTL.stats.items():
        d = {}
        c = st["cnt"].clamp_min(1)[None, :, None]
        ch = st["cnt_h"].clamp_min(1)[None, :, None]
        for n in ("mass", "mass_sp", "same", "nb1"):
            d[n] = (st[n] / c).cpu().numpy()
            d[n + "_h"] = (st[n + "_h"] / ch).cpu().numpy()
        d["ent"] = (st["ent"] / st["cnt"].clamp_min(1)[None]).cpu().numpy()
        d["cnt_h"] = st["cnt_h"].cpu().numpy()
        out[li] = d
    return out


def heart_patch_mask(subj, batch):
    """(S, G*G) bool: patch token's 14x14 footprint (on the 518 grid) overlaps the heart ROI of
    that slot's nominal plane."""
    G = subj.R // 14
    if subj.heart is None:
        return None
    m = subj.heart[torch.as_tensor(batch["slot_z"], device=subj.device)].float()[:, None]  # (S,1,H,W)
    m = F.interpolate(m, size=(subj.R, subj.R), mode="nearest")
    m = F.max_pool2d(m, 14, 14)                                                             # (S,1,G,G)
    return m[:, 0].reshape(len(batch["slot_z"]), G * G) > 0


# ───────────────────────────── forward + scoring ─────────────────────────────
@torch.no_grad()
def forward(model, subj, batch, allow=None, layers=None, record=None, query_points=None,
            key_hide=None, query_sel=None, kv_mode=None, kv_layers=(), kv_heads=None, kv_tok=None,
            full_layers=()):
    """kv_mode='save': store slot-0 post-RoPE keys/values at `kv_layers` (global blocks).
    kv_mode='patch': companions (slots>=1) attend to the STORED slot-0 keys/values at `kv_layers`
    while slot 0's own queries keep its live keys/values — i.e. only the companion->reference READS
    are swapped. (Slot 0's residual stream is unchanged up to the first patched layer; after it,
    slot 0 attends to companions that already carry the donor information.)"""
    CTL.reset()
    S = batch["images"].shape[1]
    CTL.S, CTL.P = S, N_SPECIAL + (subj.R // 14) ** 2
    CTL.allow, CTL.layers = allow, layers
    CTL.key_hide, CTL.query_sel = key_hide, query_sel
    CTL.kv_mode, CTL.kv_layers, CTL.kv_heads, CTL.kv_tok = kv_mode, set(kv_layers), kv_heads, kv_tok
    if kv_mode == "save":
        KV_STORE.clear()
    CTL.record = set(record or [])
    CTL.query_points = list(query_points or [])
    CTL.full_layers = set(full_layers)
    if CTL.record:
        CTL.heart_patch = heart_patch_mask(subj, batch)
    mb = {"z_indices": batch["z_indices"], "scanner_coords": batch["scanner_coords"]}
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=mb)
    wp = preds["world_points"].float()                         # (1,S,R,R,3)
    out = {"wp": wp, "dvf": (wp - batch["scanner_coords"]).float()}
    if CTL.record:
        out["attn"] = finalize_stats()
        out["maps"] = dict(CTL.maps)
    if CTL.full_layers:
        out["full"] = dict(CTL.full)
    CTL.reset()
    return out


def splat(subj, batch, wp, splat_res=518):
    V, cov = _splat_preds_native({"world_points": wp}, batch, (subj.D, subj.H, subj.W), subj.z_scale,
                                 splat_res=splat_res)
    return V[0].float(), cov[0].float()


def splat_companions(subj, batch, wp, splat_res=518):
    """Volume splatted from companion slots ONLY (slot 0's intensities zeroed -> the splat's
    intensity>1e-3 weight gate drops them). Scoring cardiac transfer on this volume removes the
    reference slice's own pixels, which otherwise leak into neighbouring planes via its predicted Δz."""
    b2 = dict(batch)
    ims = batch["images_splat"].clone()
    ims[:, 0] = 0
    b2["images_splat"] = ims
    return splat(subj, b2, wp, splat_res)


def slot_dz_mm(subj, batch, dvf, thr=0.05):
    """Per-slot mean predicted through-plane shift (mm) over in-FOV pixels (resp_diag convention)."""
    img = batch["images"][0, :, 0]                               # (S,R,R)
    out = []
    for s in range(img.shape[0]):
        m = img[s] > thr
        out.append(float(dvf[0, s, ..., 2][m].mean() * MM_PER_NORM[2]) if m.any() else np.nan)
    return np.array(out)


def slot_disp_mm(subj, batch, dvf, region="fov", thr=0.05):
    """Per-slot mean predicted (dz, dy, dx) mm. region: 'fov' | 'heart'."""
    S = dvf.shape[1]
    res = np.full((S, 3), np.nan)
    img = batch["images"][0, :, 0]
    for s in range(S):
        m = img[s] > thr
        if region == "heart" and subj.heart is not None:
            h = subj.heart[int(batch["slot_z"][s])].float()[None, None]
            h = F.interpolate(h, size=(subj.R, subj.R), mode="nearest")[0, 0] > 0
            m = m & h
        if m.any():
            d = dvf[0, s][m]                                     # (n,3) x,y,z normalized
            res[s] = [float(d[:, 2].mean() * MM_PER_NORM[2]), float(d[:, 1].mean() * MM_PER_NORM[1]),
                      float(d[:, 0].mean() * MM_PER_NORM[0])]
    return res


def ncc(a, b, m):
    a, b = a[m].double(), b[m].double()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def phase_ncc_profile(subj, V, planes=None):
    """NCC(V, GT_t) inside the heart ROI for every GT phase t (optionally restricted to planes)."""
    m = subj.heart.clone() if subj.heart is not None else subj.content.clone()
    if planes is not None:
        keep = torch.zeros(subj.D, dtype=torch.bool, device=m.device)
        keep[list(planes)] = True
        m &= keep[:, None, None]
    return np.array([ncc(V, subj.phases[t], m) for t in range(subj.T)])


def obs_err(subj, batch, wp, t_target, per_slot=True):
    """Observation consistency (the paper's L_obs, heart ROI only): |GT_target(p_i) - I_i| per slot,
    over input pixels whose NOMINAL location is in the heart ROI."""
    imsp = batch["images_splat"]                                 # (1,S,H,W)
    w = _resize_field(wp, subj.H, subj.W)                        # (1,S,H,W,3)
    S = imsp.shape[1]
    g = sample_volume(subj.phases[t_target][None], w.reshape(1, -1, 3), subj.z_scale).view(S, subj.H, subj.W)
    err = (g - imsp[0]).abs()
    res = []
    for s in range(S):
        m = (imsp[0, s] > 1e-3)
        if subj.heart is not None:
            m = m & subj.heart[int(batch["slot_z"][s])]
        res.append(float(err[s][m].mean()) if m.any() else np.nan)
    return np.array(res)
