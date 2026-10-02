"""Per-slice frame selector (docs/130).

Features: CardioStitch's OWN frozen DINOv2 patch embedding (`model.aggregator.patch_embed` of the
diff1000 run, frozen in aggft), on the same 518x518 model images -> (1369, 1024) patch tokens per
frame. Identical tokens to what the reconstruction computes, so they can be shared at eval.

Selector (variant "B" of docs/130 §4; the only one kept): each candidate frame of a plane is
  centered (minus its plane's mean token over frames), RMS-scaled once per plane, projected,
  2x2-pooled, plus a z embedding -> cross-attends to the (centered, scaled) reference tokens ->
  learned attention pooling -> MLP -> one score; softmax over that plane's frames.
Every frame is scored independently of the other frames (order-free), so any number of frames works.

Loss: soft circular-Gaussian labels on phase distance, q_j ~ exp(-d_j^2 / 2 sigma^2); CE(q, softmax).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# The frozen reconstruction model whose patch embed the selector reads (paper model, ARM=diff1000).
VGGT_CKPT = "scratch/logs/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt"


class FrozenPatchEmbed(nn.Module):
    """aggregator.patch_embed + its input normalisation, lifted from a loaded VGGT."""

    def __init__(self, vggt):
        super().__init__()
        agg = vggt.aggregator
        self.patch_embed = agg.patch_embed
        self.register_buffer("mean", agg._resnet_mean.clone())
        self.register_buffer("std", agg._resnet_std.clone())
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images, chunk=32):
        """images (N, 3, R, R) in [0,1] -> (N, P, 1024) patch tokens (bf16 autocast)."""
        out = []
        for i in range(0, images.shape[0], chunk):
            x = (images[i:i + chunk] - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tok = self.patch_embed(x)
            if isinstance(tok, dict):
                tok = tok["x_norm_patchtokens"]
            out.append(tok.float())
        return torch.cat(out)


def sincos(x, dim):
    """x (...,) -> (..., dim) sinusoidal embedding of a value in ~[-2, 2]."""
    freqs = torch.exp(torch.linspace(0, math.log(64.0), dim // 2, device=x.device))
    a = x[..., None] * freqs * math.pi
    return torch.cat([a.sin(), a.cos()], -1)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.nq, self.nk, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, kv):
        k = self.nk(kv)
        x = x + self.attn(self.nq(x), k, k, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class AttnPool(nn.Module):
    """Learned attention pooling over tokens: a shared per-token scorer -> softmax over the frame's
    tokens -> weighted average. Content-based (no position input), so it can find the heart wherever
    it sits in a slice."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, x):                                   # (N, n, d) -> (N, d)
        w = torch.softmax(self.score(x).squeeze(-1), -1)
        self.last_w = w                                     # (N, n), read by the heart-ROI attention loss
        return (w.unsqueeze(-1) * x).sum(1)


class FrameSelector(nn.Module):
    """Tokens are centered per plane and divided by one scalar per plane (their RMS over the plane's
    frames x tokens x channels) instead of a per-token LayerNorm -- keeps heart tokens (large
    frame-to-frame change) large relative to background ones (docs/130 §4)."""

    def __init__(self, in_dim=1024, dim=256, layers=2, heads=8, pool=2):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(in_dim, dim)
        self.zemb = nn.Linear(2 * 32, dim)          # [z of the candidate plane, z offset to ref]
        self.cpool = AttnPool(dim)
        self.blocks = nn.ModuleList([CrossBlock(dim, heads) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def _tokens(self, tok):
        """(N, 1369, C) -> projected (N, n, dim), the 37x37 grid avg-pooled by `pool`."""
        x = self.proj(tok)
        if self.pool > 1:
            n, P, c = x.shape
            g = int(round(P ** 0.5))
            x = F.avg_pool2d(x.transpose(1, 2).reshape(n, c, g, g), self.pool, ceil_mode=True)
            x = x.flatten(2).transpose(1, 2)
        return x

    def forward(self, cand_tok, ref_tok, cand_z, ref_z, ref_plane_tok):
        """cand_tok (P, F, N, C), ref_tok (N, C), cand_z (P,), ref_z float,
        ref_plane_tok (F_ref, N, C) all frames of the reference plane -> logits (P, F)."""
        Pn, Fn = cand_tok.shape[:2]
        cand_tok, ref_tok = self.prep(cand_tok, ref_tok, ref_plane_tok)
        return self.score(self._tokens(cand_tok.flatten(0, 1)), Pn, Fn, ref_tok, cand_z, ref_z)

    def prep(self, cand_tok, ref_tok, ref_plane_tok):
        """Subtract each plane's mean token over its own frames (candidates: their plane; reference:
        the reference plane), keeping only the phase-varying part, then divide by one RMS per plane.
        Order-free, any F."""
        cand_tok = cand_tok - cand_tok.mean(1, keepdim=True)
        ref_mean = ref_plane_tok.mean(0)
        ref_tok, ref_plane_tok = ref_tok - ref_mean, ref_plane_tok - ref_mean
        cand_tok = cand_tok / cand_tok.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        ref_tok = ref_tok / ref_plane_tok.pow(2).mean().sqrt().clamp_min(1e-6)
        return cand_tok, ref_tok

    def score(self, c, Pn, Fn, ref_tok, cand_z, ref_z):
        """`forward` with the candidates already projected (c = _tokens(cand), (P*F, n, d)) -- lets
        inference project every candidate ONCE and score it against many references."""
        r = self._tokens(ref_tok[None])                                # (1, n, d)
        z = torch.cat([sincos(cand_z, 32), sincos(cand_z - ref_z, 32)], -1)
        z = self.zemb(z).repeat_interleave(Fn, 0)                      # (P*F, d)
        x = c + z[:, None]
        kv = r.expand(x.shape[0], -1, -1)
        for blk in self.blocks:
            x = blk(x, kv)
        return self.head(self.cpool(x)).view(Pn, Fn)


def soft_labels(dist, sigma):
    """dist (P, F) circular phase distance -> (P, F) target distribution."""
    return torch.softmax(-dist ** 2 / (2 * sigma ** 2), -1)


def selector_loss(logits, dist, sigma):
    return -(soft_labels(dist, sigma) * F.log_softmax(logits, -1)).sum(-1).mean()
