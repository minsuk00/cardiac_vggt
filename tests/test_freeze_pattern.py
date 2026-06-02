"""Guard the freeze pattern in mri_finetune.yaml / mri_volume.yaml.

Current contract: the entire aggregator subtree is frozen (incl. z_embedder + t_embedder).
Only `point_head` is trainable (~32.65M params).

Why z/t embedders are also frozen, despite being only 28K params:
  We A/B-measured the cost of making them trainable while keeping the rest of the
  aggregator frozen. Step time jumps from ~1.32 sec to ~3.25 sec (2.5× slower)
  because backward must traverse all 48 frozen attention blocks via gradient
  checkpointing recomputation to reach the upstream embedders. The 4-day baseline
  ran with embedders frozen at random init and still reached 31+ dB PSNR — the
  point_head memorizes the (random but consistent) codes. Frozen wins on throughput
  with no proven quality cost in our discrete multi-phase setup.

When this contract should change: if you ever unfreeze the aggregator's attention
blocks (e.g., for fundamental retraining on free-breathing data or Option B
continuous-phase queries), backward already traverses them all, so trainable
embedders become free. In that case, enumerate the aggregator subparts and let
z/t embedders learn. Until then, the wildcard freeze stays.
"""
import sys
import os
import pytest
from omegaconf import OmegaConf

# Make training/ importable for direct imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def model_with_freeze():
    from hydra import compose, initialize_config_dir
    from train_utils.freeze import freeze_modules
    from vggt.models.vggt import VGGT

    OmegaConf.register_new_resolver("rev_ts", lambda: "test", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: p.rstrip("/").split("/")[-1], replace=True)

    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "training", "config")
    cfg_dir = os.path.abspath(cfg_dir)
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="mri_volume")

    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=True,
        use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
    )
    freeze_modules(model, patterns=list(cfg.optim.frozen_module_names), recursive=True)
    return model


def _counts(model, prefix):
    nt = nf = 0
    for n, p in model.named_parameters():
        if n.startswith(prefix):
            if p.requires_grad:
                nt += p.numel()
            else:
                nf += p.numel()
    return nt, nf


def test_t_embedder_is_frozen(model_with_freeze):
    """t_embedder is currently frozen (with the rest of the aggregator) for 2.5× faster
    training. See module docstring for the rationale. Flip this assertion if you
    enumerate the freeze subparts (e.g., when unfreezing the aggregator too).
    """
    nt, nf = _counts(model_with_freeze, "aggregator.t_embedder")
    assert nt == 0, f"t_embedder should be frozen for throughput; got {nt} trainable params"
    assert nf > 0, "t_embedder has no params at all — wrong module path?"


def test_z_embedder_is_frozen(model_with_freeze):
    """Same — z_embedder frozen alongside t_embedder. See module docstring."""
    nt, nf = _counts(model_with_freeze, "aggregator.z_embedder")
    assert nt == 0, f"z_embedder should be frozen for throughput; got {nt} trainable params"
    assert nf > 0, "z_embedder has no params at all"


def test_target_t_embedder_is_frozen(model_with_freeze):
    """target_t_embedder frozen alongside z/t embedders under the `*aggregator*` wildcard
    (head-only freeze). In aggft runs (`*patch_embed*` only) it trains — same as z/t."""
    nt, nf = _counts(model_with_freeze, "aggregator.target_t_embedder")
    assert nt == 0, f"target_t_embedder should be frozen for throughput; got {nt} trainable params"
    assert nf > 0, "target_t_embedder has no params at all — wrong module path?"


def test_heavy_aggregator_blocks_are_frozen(model_with_freeze):
    """DINOv2 patch embed + 24×24 attention blocks should stay frozen."""
    for prefix in ["aggregator.patch_embed", "aggregator.frame_blocks", "aggregator.global_blocks"]:
        nt, nf = _counts(model_with_freeze, prefix)
        assert nt == 0, f"{prefix}: expected fully frozen but got {nt} trainable params"
        assert nf > 0, f"{prefix}: has no params at all"


def test_register_and_camera_tokens_frozen(model_with_freeze):
    """Register tokens (slot-0 distinguishing) and the legacy camera_token stay frozen."""
    for prefix in ["aggregator.register_token", "aggregator.camera_token"]:
        nt, nf = _counts(model_with_freeze, prefix)
        assert nt == 0, f"{prefix}: expected frozen, got {nt} trainable"


def test_point_head_is_trainable(model_with_freeze):
    nt, _ = _counts(model_with_freeze, "point_head")
    assert nt > 30_000_000, f"point_head trainable count seems wrong: {nt}"
