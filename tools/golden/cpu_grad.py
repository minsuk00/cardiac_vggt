"""G4: one forward + backward of the paper model and loss on CPU (deterministic kernels), on a
fixed synthetic batch. Fingerprints every loss term and every parameter gradient, so old vs new
code can be compared bit for bit where the GPU backward (grid_sample 3D) is nondeterministic.

  python tools/golden/cpu_grad.py --out temp/golden/<tag>/g4.json
"""
import argparse
import inspect

from common import dump, sha

import torch

# Old code needs the flags spelled out (its defaults were off); new code rejects them, so
# `_accepted` drops whatever the current signature does not take.
MODEL_KW = dict(img_size=518, patch_size=14, enable_point=True, gradient_checkpointing=True,
                use_z_pose_embedding=True, use_reference_token=True, train_on_residual_dvf=True,
                warp_head_type="dpt")
# Paper diff1000 loss; tv/motion_l1 at their paper value 0 (old default tv_weight was 0.1).
LOSS_KW = dict(weight=1.0, splat_res=518, diffusion_weight=1000.0, gather_weight=0.5,
               heart_weight=0.5, tv_weight=0.0, motion_l1_weight=0.0)


def _accepted(fn, kw, extra=()):
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params or k in extra}


def make_batch(S=3, R=224, D=6, N=256):
    g = torch.Generator().manual_seed(0)
    lin = torch.arange(R, dtype=torch.float32) / (R - 1) * 2 - 1
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    z = torch.linspace(-0.3, 0.3, S)
    sc = torch.stack([torch.stack([xx, yy, torch.full_like(xx, float(zi))], -1) for zi in z])[None]
    roi = torch.zeros(1, D, N, N, dtype=torch.uint8)
    roi[:, 1:5, 90:170, 90:170] = 1
    return {
        "images": torch.rand(1, S, 3, R, R, generator=g),
        "images_splat": torch.rand(1, S, N, N, generator=g),
        "scanner_coords": sc,
        "z_indices": z.view(1, S, 1),
        "gt_target_volume": torch.rand(1, D, N, N, generator=g),
        "z_scale": torch.tensor([90.0 / 12.0]),
        "heart_roi_canonical": roi,
        "anatomy_bbox": torch.tensor([[0, D, 40, 216, 40, 216]]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()

    from loss import MultitaskLoss, compute_volume_intensity_loss
    from vggt.models.vggt import VGGT

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    model = VGGT(**_accepted(VGGT.__init__, MODEL_KW)).train()
    batch = make_batch()
    preds = model(batch["images"], batch=batch)
    loss_kw = _accepted(compute_volume_intensity_loss, LOSS_KW, extra=("weight",))
    out = MultitaskLoss(volume=loss_kw)(preds, batch)
    out["objective"].backward()

    res = {"losses": {k: sha(v) for k, v in sorted(out.items())
                      if k.startswith("loss_") or k == "objective"},
           "world_points": sha(preds["world_points"]),
           "grads": {n: sha(q.grad) for n, q in model.named_parameters() if q.grad is not None},
           "n_params": sum(1 for _ in model.parameters())}
    res["losses_value"] = {k: float(v) for k, v in out.items()
                           if (k.startswith("loss_") or k == "objective") and v.numel() == 1}
    dump(res, a.out)
    print(f"objective={float(out['objective']):.8f}  grads={len(res['grads'])}")


if __name__ == "__main__":
    main()
