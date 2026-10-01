"""G6: model construction fingerprint (CPU). For each backbone variant, build VGGT with seed 0 and
record every state_dict entry (key, shape, dtype, sha) plus the global RNG state afterwards, so
a removed/added init draw or a renamed key shows up. Optionally strict-loads paper checkpoints.

  python tools/golden/model_init.py --out temp/golden/<tag>/g6.json [--ckpts /tmp/vggt_golden/ckpts/*.pt]
"""
import argparse
import inspect

from common import dump, sha

import torch

VARIANTS = {
    "dinov2_518": dict(img_size=518, patch_size=14, backbone="dinov2_vitl14_reg"),
    "dinov2_224": dict(img_size=224, patch_size=14, backbone="dinov2_vitl14_reg"),
    "dinov3_256": dict(img_size=256, patch_size=16, backbone="dinov3_vitl16"),
}
# Old trees need the always-on flags spelled out; newer trees reject them.
LEGACY_KW = dict(enable_point=True, use_z_pose_embedding=True, use_reference_token=True,
                 train_on_residual_dvf=True, warp_head_type="dpt")


def build(VGGT, kw):
    params = inspect.signature(VGGT.__init__).parameters
    var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    kw = {**kw, **LEGACY_KW}
    return VGGT(**{k: v for k, v in kw.items() if var_kw or k in params})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--ckpts", nargs="*", default=[])
    a = p.parse_args()

    from vggt.models.vggt import VGGT

    res = {}
    for name, kw in VARIANTS.items():
        torch.manual_seed(0)
        model = build(VGGT, kw)
        sd = model.state_dict()
        res[name] = {"rng_after": sha(torch.get_rng_state()),
                     "n": len(sd),
                     "params": {k: [list(v.shape), str(v.dtype), sha(v)] for k, v in sd.items()}}
        print(f"{name}: {len(sd)} entries, rng {res[name]['rng_after']}")
        if name == "dinov2_518":
            for c in a.ckpts:
                state = torch.load(c, map_location="cpu", weights_only=False)["model"]
                model.load_state_dict(state, strict=True)
                res.setdefault("strict_load", {})[c.rsplit("/", 1)[-1]] = "ok"
                print(f"  strict load ok: {c}")
        del model
    dump(res, a.out)


if __name__ == "__main__":
    main()
