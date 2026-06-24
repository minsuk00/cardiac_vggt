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
from omegaconf import OmegaConf

# Make training/ importable for direct imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_cfg():
    from hydra import compose, initialize_config_dir

    OmegaConf.register_new_resolver("rev_ts", lambda: "test", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: p.rstrip("/").split("/")[-1], replace=True)

    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "config"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name="mri_volume")


def _build_from_cfg(cfg, warp_head_type=None):
    from vggt.models.vggt import VGGT

    m = cfg.model
    return VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=m.enable_camera, enable_depth=m.enable_depth,
        enable_point=m.enable_point, enable_track=m.enable_track,
        use_z_pose_embedding=m.use_z_pose_embedding,
        use_t_pose_embedding=m.use_t_pose_embedding,
        use_target_t_pose_embedding=m.use_target_t_pose_embedding,
        use_reference_token=m.use_reference_token,
        train_on_residual_dvf=m.train_on_residual_dvf,
        warp_head_type=warp_head_type or m.warp_head_type,
        bspline_grid_size=m.bspline_grid_size,
    )


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
    """Config wiring: reference ON, the obsolete input/target-phase indices OFF."""
    m = cfg_only.model
    assert m.use_reference_token is True
    assert m.use_z_pose_embedding is True
    assert m.use_t_pose_embedding is False
    assert m.use_target_t_pose_embedding is False
    assert cfg_only.reference_slot is True
    assert cfg_only.distributed.find_unused_parameters is True


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


def test_obsolete_phase_embedders_absent(model_with_freeze):
    """use_t/use_target_t are OFF → those embedders aren't even built."""
    agg = model_with_freeze.aggregator
    assert not hasattr(agg, "t_embedder"), "t_embedder should not exist (use_t off)"
    assert not hasattr(agg, "target_t_embedder"), "target_t_embedder should not exist (use_target_t off)"


def test_use_reference_token_wired(model_with_freeze):
    assert getattr(model_with_freeze.aggregator, "use_reference_token", False) is True


def test_point_head_is_trainable(model_with_freeze):
    nt, _ = _counts(model_with_freeze, "point_head")
    assert nt > 30_000_000, f"point_head trainable count seems wrong: {nt}"


@pytest.fixture(scope="module")
def bspline_model_with_freeze():
    """Same aggft contract, but with the B-spline warp head (warp_head_type='bspline')."""
    from train_utils.freeze import freeze_modules

    cfg = _load_cfg()
    model = _build_from_cfg(cfg, warp_head_type="bspline")
    freeze_modules(model, patterns=list(cfg.optim.frozen_module_names), recursive=True)
    return model


def test_bspline_head_trainable_and_patch_embed_frozen(bspline_model_with_freeze):
    """The B-spline head stays trainable (named point_head) and is far smaller than the
    32.65M DPT head; patch_embed frozen, aggregator blocks trainable (aggft)."""
    nt, _ = _counts(bspline_model_with_freeze, "point_head")
    assert 0 < nt < 5_000_000, f"bspline point_head trainable count unexpected: {nt}"
    pe_t, pe_f = _counts(bspline_model_with_freeze, "aggregator.patch_embed")
    assert pe_t == 0 and pe_f > 0, "patch_embed must stay frozen"
    blk_t, _ = _counts(bspline_model_with_freeze, "aggregator.frame_blocks")
    assert blk_t > 0, "aggregator blocks must be trainable under aggft"
