"""CorSeg soft-Dice auxiliary loss (ARM corseg-dice — docs/69 follow-up, 2026-08-11).

Motivation: the heart-L1 arm upweights heart-ROI *intensity* fidelity; this arm instead
rewards putting the *endocardial boundary* in the right place — which is literally what
contraction amplitude is. A frozen CorSeg-CineSAX (MedNeXt-L, 2D per-slice) segments every
z-slice of the predicted volume V_canon, and a soft Dice against the on-disk GT labels
(`heart_seg_canonical[..., t_target]`, batch key `heart_seg_t`) is added to the objective.
Gradients flow through the frozen segmenter into V_canon -> world_points.

Preprocessing replicates tools/corseg/corseg_infer.py `segment_stack(mode="paper")` —
resample 1.4mm -> 1.25mm in-plane, center crop/pad to 224, per-slice z-score over nonzero —
but differentiably and BRANCHLESS (no tensor->python bool: no GPU sync / graph break under
cuda.compile_attention_blocks; same hazard loss.py documents for the heart-L1 term).

Label spaces differ and are REMAPPED here (verified empirically 2026-08-11, 3 val subjects,
IoU 0.902/0.706/0.890 on the matched pairs):
    GT (heart_seg_canonical): 1=LV_cav, 2=LV_myo, 3=RV
    CorSeg output channels:   1=LV_myo, 2=LV_cav, 3=RV
Dice is computed volume-level (summed over all slices) on classes 1..3; a class absent from
the GT contributes Dice=1 (a constant, no gradient) — the MRI2CT `mask_absent` pattern, which
removes the empty-class smoothing-cliff gradient.

⚠️ Hygiene: this campaign's amp_ratio verdict metric is CorSeg-derived. Any arm trained with
this loss must have its decisive checkpoints re-scored with nnU-Net (Task114) so the verdict
stays independent of the training signal.
"""
import torch
import torch.nn.functional as F

CORSEG_CKPT = "/home/minsukc/vggt/scratch/data/corseg/ModelWeight-CorSeg-CineSAX_MedNextL.pth"
PIXDIM_CANON = 1.4    # canonical in-plane spacing (mm) — preprocess.py
PIXDIM_CORSEG = 1.25  # CorSeg paper spacing (mm)
IMG = 224             # CorSeg fixed input size
_CHUNK = 4            # slices per checkpointed MedNeXt forward (memory bound, see loss fn)

# GT label -> CorSeg class index (background stays 0). See docstring for verification.
_GT2CORSEG = torch.tensor([0, 2, 1, 3], dtype=torch.long)

_MODEL = None

# Fraction of Dice-branch grad elements pinned at ±grad_clamp in the most recent backward
# (detached GPU scalar; None until the first backward). Read by loss.py for logging —
# one step LAGGED, since the hook fires in backward, after the step's scalars are logged.
# Diagnostic for weight tuning: if most elements saturate, raising corseg_weight no longer
# strengthens the term (grad degenerates to a constant-magnitude push) — raise grad_clamp.
LAST_SAT = None


def _get_model(device):
    """Lazy per-process singleton; frozen + eval. Params never require grad — input grads
    still flow through, which is all the loss needs."""
    global _MODEL
    if _MODEL is None:
        from monai.networks.nets.mednext import create_mednext
        ck = torch.load(CORSEG_CKPT, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        m = create_mednext(
            variant=cfg.get("mednext_variant", "L"),
            spatial_dims=cfg.get("spatial_dims", 2),
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("num_classes", 4),
            kernel_size=cfg.get("mednext_kernel", 5),
            deep_supervision=False,
        )
        m.load_state_dict(ck.get("model_state_dict", ck), strict=True)
        m.eval().requires_grad_(False)
        _MODEL = m.to(device)
    return _MODEL


def _resample_crop(x, mode):
    """(N,1,256,256) -> (N,1,224,224): 1.4->1.25mm resample then center crop/pad.
    Same symmetric floor-first convention as corseg_infer.center_pad_crop."""
    n = max(1, int(round(x.shape[-1] * PIXDIM_CANON / PIXDIM_CORSEG)))
    if mode == "bilinear":
        x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
    else:
        # nearest-exact: half-pixel-center convention, same grid as the bilinear image
        # path (legacy "nearest" floors, shifting labels ~0.44 px off the image).
        x = F.interpolate(x, size=(n, n), mode="nearest-exact")
    if n >= IMG:
        o = (n - IMG) // 2
        x = x[..., o:o + IMG, o:o + IMG]
    else:
        pad = IMG - n
        lo = pad // 2
        x = F.pad(x, (lo, pad - lo, lo, pad - lo))
    return x


def _zscore_nonzero(x):
    """Per-slice z-score over nonzero pixels, branchless. Matches segment_stack step 3
    (up to biased-vs-unbiased std, ~sqrt(n/(n-1)) on thousands of pixels — negligible;
    empirically validated by the dice-on-GT check). Near-constant slices (sd below the
    floor) are left unnormalized, zeros stay exactly 0.

    ⚠️ SD FLOOR IS 1e-3, NOT the reference's 1e-8 (NaN post-mortem, 2026-08-11): a
    degenerate early-training V_canon can emit a near-constant nonzero slice; dividing by
    sd~1e-7 produces ~1e6-scale inputs that overflow MedNeXt's bf16 activations → NaN
    objective → the trainer's NaN-guard skips every backward and the run freezes (jobs
    57023101 + the clamped smoke, NaN onset step 33/17, control w=0 clean). Real slices
    have sd >> 1e-3 after [0,1] normalization, so the floor only fires on degenerate
    inputs — where "don't normalize" is the right answer anyway."""
    nz = (x != 0)
    nzf = nz.float()
    cnt = nzf.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    mu = (x * nzf).sum(dim=(1, 2, 3), keepdim=True) / cnt
    var = (((x - mu) ** 2) * nzf).sum(dim=(1, 2, 3), keepdim=True) / cnt
    # +1e-12 before sqrt: sqrt'(0)=inf, so an exactly-constant/all-zero slice (var==0,
    # e.g. a zero-coverage z-plane in V_canon) emits NaN in BACKWARD even though the
    # forward `where`s route around it — and the grad-clamp hook passes NaN through.
    sd = (var + 1e-12).sqrt()
    normed = torch.where(sd > 1e-3, (x - mu) / sd.clamp(min=1e-3), x)
    return torch.where(nz, normed, torch.zeros_like(x))


def corseg_dice_loss(V_canon, seg_t, weight, smooth=1e-5, grad_clamp=5e-6):
    """weight · (1 - mean soft Dice (classes 1..3, CorSeg space)) of CorSeg(V_canon) vs GT.

    Args:
        V_canon: (B, D, H, W) float, canonical-grid predicted volume, intensities ~[0,1].
        seg_t:   (B, D, H, W) integer labels in GT convention (heart_seg_canonical at
                 t_target, already co-warped by the train-time affine aug).
        weight:  corseg_weight. Applied INSIDE this function so that `grad_clamp` is in
                 absolute (post-weight) units — see below.
        grad_clamp: per-voxel cap (absolute value) on the gradient this term sends into
                 V_canon, enforced by a backward hook where the Dice branch leaves V_canon.
    Returns: scalar tensor (fp32), differentiable w.r.t. V_canon, ALREADY weighted.

    ⚠️ WHY THE CLAMP EXISTS (2026-08-11, job 57023101 post-mortem). The Dice gradient is
    boundary-CONCENTRATED: measured on a real early-training prediction, its per-voxel max is
    ~16,000x the full-L1's per-voxel max (32x even after weight=0.002), while its p50 is
    ~0.1x — a heavy spike tail pushing the same boundary channels every step. Global
    norm-clipping (max_norm=1.0, already active) does NOT stop that persistent directional
    push, and it killed the DPT head's final ReLU within 33 steps — the docs/64 death
    signature (grad_aggregator == 0, loss_corseg frozen) via a new driver. The per-voxel
    clamp caps the Dice pull at ~3.5x the L1's per-voxel scale (L1 max ~1.4e-6): the p50 of
    the Dice gradient passes untouched, only the ~1% spike tail is cut. It also bounds HIGH
    weights by construction — raising corseg_weight then saturates more voxels at the cap
    instead of scaling the spikes.
    """
    B, D, H, W = V_canon.shape
    model = _get_model(V_canon.device)

    x = V_canon.reshape(B * D, 1, H, W).float()
    # The hook sees the POST-weight gradient (chain rule applies `weight` upstream of this
    # node), so grad_clamp is an absolute per-voxel bound regardless of corseg_weight. `x`
    # feeds only the Dice branch, so the L1/gather/diffusion gradients are unaffected.
    if x.requires_grad and grad_clamp is not None:
        def _clamp_hook(g):
            global LAST_SAT
            with torch.no_grad():
                LAST_SAT = (g.abs() >= grad_clamp).float().mean()
            return g.clamp(-grad_clamp, grad_clamp)
        x.register_hook(_clamp_hook)
    # fp32 ISLAND — the whole branch runs with autocast disabled (defense in depth against
    # low-precision overflow on out-of-distribution inputs from a degenerate early model;
    # the fp32-softmax precision guard is subsumed).
    #
    # CHUNKED + CHECKPOINTED forward — MedNeXt-L activations for a whole stack peak at
    # ~20 GB (D=12, measured) and OOM'd a 44 GB L40S on large-D subjects next to the 23 GB
    # trainer. Slices are processed in chunks of 4 with recompute-in-backward
    # (use_reentrant=False), bounding activation memory to one chunk (~7 GB) at ~1.5x the
    # term's forward compute. Val/no-grad paths skip checkpointing (nothing is stored).
    with torch.amp.autocast("cuda", enabled=False):
        x = _zscore_nonzero(_resample_crop(x, "bilinear"))
        probs_chunks = []
        for i in range(0, x.shape[0], _CHUNK):
            xi = x[i:i + _CHUNK]
            if torch.is_grad_enabled() and xi.requires_grad:
                lg = torch.utils.checkpoint.checkpoint(model, xi, use_reentrant=False)
            else:
                lg = model(xi)
            probs_chunks.append(F.softmax(lg.float(), dim=1))
        probs = torch.cat(probs_chunks, 0)              # (N, 4, 224, 224), fp32

    lut = _GT2CORSEG.to(seg_t.device)
    seg_c = lut[seg_t.long()]                           # -> CorSeg label space
    s = _resample_crop(seg_c.reshape(B * D, 1, H, W).float(), "nearest")
    s = s.round().long().clamp(0, 3).squeeze(1)         # (N, 224, 224)
    onehot = F.one_hot(s, num_classes=4).permute(0, 3, 1, 2).float()

    inter = (probs * onehot).sum(dim=(0, 2, 3))         # (4,) volume-level
    union = probs.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    present = onehot.sum(dim=(0, 2, 3)) > 0
    dice = torch.where(present, dice, torch.ones_like(dice))
    return (1.0 - dice[1:].mean()) * weight
