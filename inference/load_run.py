"""Build a model from the run that produced the checkpoint — never from the live `default.yaml`.

**This is the single most important design point in the eval rebuild.** Every previous eval script
read its protocol by `compose(config_name="default")`, i.e. from whatever `training/config/` looks
like today. That silently mis-scores any checkpoint trained before the config moved. Concretely, at
the time of writing the live default resolves to `img_size 518` / aug tier `moderate`, while the
checkpoint under test (`213338187_augaggr224hw2_pooled1337`) trained at `img_size 224` / tier
`aggressive`. Nothing crashes — you just get a model fed at the wrong resolution and a val protocol
that never matches training.

Every run writes its own fully-resolved config to `<log_dir>/run_meta.jsonl` (the `event=="launch"`
row), so the run IS its own source of truth. `load_model_from_run` reads that and nothing else.

## Handling older runs

`img_size` / `patch_size` / `backbone` only became `model:` keys in commit 40b652a; before that
`model:` carried just the behaviour flags and `img_size` lived at the config top level. Both layouts
are handled, in this precedence: `config.model.<k>` → `config.<k>` → default. `patch_size` finally
falls back to `backbone_patch_size(backbone)` because it is a fixed property of the backbone, not a
free knob (docs/77) — a wrong value there fails loudly at build time (docs/76), which is what we
want.

## Dead kwargs

`VGGT.__init__` absorbs unknown keys through `**kwargs`, so `enable_camera` / `enable_depth` /
`enable_track` / `enable_refiner` / `refiner_use_coverage` / `grid_shape` — all present in old
configs and all retired — pass silently and do nothing. They are dropped explicitly here so a
future reader does not think `enable_refiner=True` still builds a refiner. (That is also why there
is no `--refiner` flag anywhere in the new harness: it was a no-op.)
"""
from __future__ import annotations

import json
import os

import torch

from vggt.models.vggt import VGGT
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local

# Retired VGGT kwargs that old configs still carry; `**kwargs` would swallow them silently.
DEAD_MODEL_KWARGS = ("enable_camera", "enable_depth", "enable_track", "enable_refiner",
                     "refiner_use_coverage", "grid_shape", "use_t_pose_embedding",
                     "use_target_t_pose_embedding", "_target_")

# Pre-40b652a runs recorded neither; every such run was DINOv2/14.
LEGACY_BACKBONE = "dinov2_vitl14_reg"


def find_run_meta(ckpt_path):
    """`<log_dir>/run_meta.jsonl` for a `<log_dir>/ckpts/<name>.pt` checkpoint."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for cand in (os.path.join(d, "..", "run_meta.jsonl"), os.path.join(d, "run_meta.jsonl")):
        p = os.path.normpath(cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no run_meta.jsonl next to or one level above {ckpt_path}. The eval harness reads the "
        "protocol from the run that produced the checkpoint; without it the arm cannot be scored "
        "faithfully. Point --ckpt at a checkpoint inside its original log_dir.")


def read_run_config(ckpt_path):
    """The fully-resolved Hydra config the run launched with (the `event=='launch'` row).

    On a requeue there is one launch row per process start; the FIRST is the one that defined the
    run (later rows are resumes of it), and the model architecture cannot change across a resume.
    """
    meta_path = find_run_meta(ckpt_path)
    with open(meta_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    launches = [r for r in rows if "config" in r]
    if not launches:
        raise ValueError(f"{meta_path} has no row carrying a 'config' blob")
    return launches[0]["config"], meta_path


def model_kwargs_from_config(cfg):
    """-> the exact kwargs to build this run's VGGT, plus the resolved (img_size, backbone)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "training"))
    from data import backbone_patch_size          # noqa: E402  (also registers backbone_ps)

    m = dict(cfg.get("model") or {})
    for k in DEAD_MODEL_KWARGS:
        m.pop(k, None)

    def pick(key, default=None):
        """model block wins, then config top level, then the default. `None` counts as absent —
        pre-40b652a runs record `backbone: null` / `patch_size: null` at the top level."""
        for src in (m, cfg):
            if src.get(key) is not None:
                return src[key]
        return default

    backbone = pick("backbone", LEGACY_BACKBONE)
    kw = dict(m)
    kw["backbone"] = backbone
    kw["img_size"] = pick("img_size", 518)
    kw["patch_size"] = pick("patch_size") or backbone_patch_size(backbone)
    return kw


def load_model_from_run(ckpt_path, device="cuda", verbose=True):
    """-> (model.eval() on `device`, run_cfg).

    `run_cfg` is the run's whole config, so callers also get the data knobs (`reference_slot`,
    `one_frame_per_slice`, `continuous_z`, `z_jitter`, `split_file`, `data_root`) and the
    respiratory block — all of which must match the run, for the same reason the model kwargs must.
    """
    cfg, meta_path = read_run_config(ckpt_path)
    kw = model_kwargs_from_config(cfg)
    model = VGGT(**kw)

    # GPFS torch.load of an ~9 GB checkpoint is ~266 s vs ~5 s from node-local /tmp (docs/50).
    ck = torch.load(stage_checkpoint_to_local(ckpt_path), map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    # strict=False is required (heads the config disables are absent), so guard the parts that
    # MUST be present — a silently half-loaded aggregator would still produce plausible volumes.
    critical = [k for k in missing
                if any(s in k for s in ("aggregator", "point_head", "z_embedder"))]
    if critical:
        raise RuntimeError(
            f"{len(critical)} critical weights missing from {ckpt_path}: {critical[:5]} ... "
            "The model kwargs do not match the checkpoint's architecture.")
    if verbose:
        print(f"  model: img_size={kw['img_size']} backbone={kw['backbone']} "
              f"patch_size={kw['patch_size']} warp_head={kw.get('warp_head_type', 'dpt')}\n"
              f"  protocol from {os.path.relpath(meta_path)} "
              f"(exp {cfg.get('exp_name')})\n"
              f"  loaded {ckpt_path}  (missing={len(missing)}, unexpected={len(unexpected)})",
              flush=True)
    return model.to(device).eval(), cfg


def mri_dataset_kwargs(cfg, split="val"):
    """The run's own `MRIDataset` kwargs for `split`, straight out of its config.

    Reads `data.<split>.dataset.dataset_configs[0]` — the ComposedDataset entry the run actually
    built — so sampling knobs (`reference_slot`, `one_frame_per_slice`, `target_size`, `z_jitter`,
    `continuous_z`, `num_slices`) come from the run rather than from today's default.yaml.
    `_target_` and `defer_input_images` are dropped: the harness needs real `images` back, because
    unlike the trainer it does not always route through `gpu_augment_batch`.
    """
    node = (cfg.get("data") or {}).get(split) or {}
    dsc = ((node.get("dataset") or {}).get("dataset_configs") or [])
    if not dsc:
        raise ValueError(f"run config has no data.{split}.dataset.dataset_configs entry")
    kw = {k: v for k, v in dsc[0].items() if k not in ("_target_", "defer_input_images")}
    return kw
