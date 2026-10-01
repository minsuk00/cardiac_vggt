"""Guard the freeze pattern in mri_volume.yaml.

Current contract (reference-slice conditioning, docs/24/25 — aggft): mri_volume.yaml trains the
aggregator finetune (aggft) so the camera_token (the slot-0 reference anchor) and z_embedder can
specialize. `optim.frozen_module_names = ["*patch_embed*"]` freezes ONLY the DINOv2 patch_embed;
the 24×24 attention blocks, z_embedder, camera_token, register_token, and point_head all TRAIN.

`use_t_pose_embedding` / `use_target_t_pose_embedding` are OFF in mri_volume, so `t_embedder` /
`target_t_embedder` are not even constructed (the broken content-free target_t index is gone).

History: the pre-reference pipeline was head-only (the `*aggregator*` wildcard froze the whole
aggregator incl. z/t embedders for ~2.5× throughput). That contract moved to the aggft regime
when reference conditioning became primary — the backward already traverses the attention blocks,
so trainable embedders/camera_token are free. The model is built from the mri_volume config here
so this test tracks the config's flags automatically.
"""
import sys
import os
import pytest

# Make training/ importable for direct imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_cfg():
    from hydra import compose, initialize_config_dir
    from resolvers import register_all

    register_all()

    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "config"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name="default")


def _build_from_cfg(cfg):
    from vggt.models.vggt import VGGT

    return VGGT(img_size=518, patch_size=14, embed_dim=1024)


@pytest.fixture(scope="module")
def model_with_freeze():
    from train_utils.freeze import freeze_modules

    cfg = _load_cfg()
    model = _build_from_cfg(cfg)
    freeze_modules(model, patterns=list(cfg.optim.frozen_module_names), recursive=True)
    return model


@pytest.fixture(scope="module")
def cfg_only():
    return _load_cfg()


def _counts(model, prefix):
    nt = nf = 0
    for n, p in model.named_parameters():
        if n.startswith(prefix):
            if p.requires_grad:
                nt += p.numel()
            else:
                nf += p.numel()
    return nt, nf


def test_mri_volume_is_aggft(cfg_only):
    """Sanity: mri_volume freezes ONLY patch_embed (aggft), not the whole aggregator."""
    frozen = list(cfg_only.optim.frozen_module_names)
    assert frozen == ["*patch_embed*"], f"expected aggft freeze, got {frozen}"


def test_mri_volume_uses_reference_conditioning(cfg_only):
    """Config wiring: the dataset puts the reference slice in slot 0."""
    assert cfg_only.reference_slot is True


def test_patch_embed_is_frozen(model_with_freeze):
    """The DINOv2 patch embed stays frozen (the only frozen subtree in aggft)."""
    nt, nf = _counts(model_with_freeze, "aggregator.patch_embed")
    assert nt == 0, f"patch_embed should be frozen; got {nt} trainable params"
    assert nf > 0, "patch_embed has no params at all — wrong module path?"


def test_aggregator_blocks_are_trainable(model_with_freeze):
    """aggft: the 24×24 attention blocks train (this is what corrects motion, docs/09)."""
    for prefix in ["aggregator.frame_blocks", "aggregator.global_blocks"]:
        nt, _ = _counts(model_with_freeze, prefix)
        assert nt > 0, f"{prefix}: expected trainable under aggft, got 0"


def test_z_embedder_is_trainable(model_with_freeze):
    """z_embedder trains under aggft (free — backward already traverses the blocks)."""
    nt, _ = _counts(model_with_freeze, "aggregator.z_embedder")
    assert nt > 0, "z_embedder should be trainable under aggft"


def test_camera_token_is_trainable(model_with_freeze):
    """The reference anchor: camera_token MUST train so it can specialize to
    'slot 0 is the target-phase reference' (docs/25)."""
    nt, _ = _counts(model_with_freeze, "aggregator.camera_token")
    assert nt > 0, "camera_token should be trainable for reference conditioning"


def test_point_head_is_trainable(model_with_freeze):
    nt, _ = _counts(model_with_freeze, "point_head")
    assert nt > 30_000_000, f"point_head trainable count seems wrong: {nt}"
