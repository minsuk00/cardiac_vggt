"""Guard the freeze pattern in mri_finetune.yaml / mri_volume.yaml.

The 4-day baseline pipeline accidentally swept up `z_embedder` and `t_embedder` via
the `*aggregator*` wildcard, leaving them at random init for the whole training. The
point_head then had to memorize meaningless random codes. The fix is to enumerate
the heavy aggregator subparts explicitly so the small (z, t) Fourier-projection
embedders stay trainable. This test ensures we don't regress on that.
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


def test_t_embedder_is_trainable(model_with_freeze):
    """The cyclic Fourier projection for cardiac phase MUST learn — it was accidentally
    frozen via `*aggregator*` in the original 4-day pipeline; don't let that recur.
    """
    nt, nf = _counts(model_with_freeze, "aggregator.t_embedder")
    assert nt > 0, "t_embedder must have trainable params"
    assert nf == 0, f"t_embedder has frozen params: {nf}"


def test_z_embedder_is_trainable(model_with_freeze):
    """Same for the z Fourier projection."""
    nt, nf = _counts(model_with_freeze, "aggregator.z_embedder")
    assert nt > 0, "z_embedder must have trainable params"
    assert nf == 0, f"z_embedder has frozen params: {nf}"


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
