import os
import inspect
import sys

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _compose(name):
    from hydra import compose, initialize_config_dir
    import data  # noqa: F401  (registers the backbone_ps resolver)

    OmegaConf.register_new_resolver("rev_ts", lambda: "test", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: p.rstrip("/").split("/")[-1], replace=True)
    OmegaConf.register_new_resolver(
        "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True
    )
    OmegaConf.register_new_resolver(
        "backbone_tag",
        lambda name: "dinov3" if str(name).startswith("dinov3_") else "dinov2",
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "aug_tag",
        lambda enabled, tier: (
            "noaug" if not enabled else {
                "conservative": "aug_cons",
                "moderate": "aug_mod",
                "aggressive": "aug_agg",
            }[str(tier)]
        ),
        replace=True,
    )
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "config"))
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=name)


def test_default_and_dinov3_config_contracts():
    default = _compose("default")
    assert (default.backbone, default.img_size, default.patch_size) == (
        "dinov2_vitl14_reg",
        518,
        14,
    )
    assert default.model.backbone == default.backbone
    assert default.model.img_size == default.img_size
    assert default.model.patch_size == default.patch_size
    assert list(default.logging.wandb_writer.tags) == ["dinov2", "aug_mod", 518]

    dinov3 = _compose("exp_dinov3")
    assert (dinov3.backbone, dinov3.img_size, dinov3.patch_size) == (
        "dinov3_vitl16",
        256,
        16,
    )
    assert dinov3.checkpoint.strict is True
    assert dinov3.checkpoint.resume_checkpoint_path.endswith("vggt1b_dinov3_vitl16_seed.pt")
    assert dinov3.data.train.dataset.dataset_configs[0].patch_size == 16
    assert dinov3.data.val.dataset.dataset_configs[0].patch_size == 16
    assert list(dinov3.logging.wandb_writer.tags) == ["dinov3", "aug_mod", 256]


def test_wandb_augmentation_tags_follow_overrides():
    from hydra import compose, initialize_config_dir

    _compose("default")  # register the same resolvers used by this standalone compose
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "config"))
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        aggressive = compose(config_name="default", overrides=["data.augmentation.tier=aggressive"])
        disabled = compose(config_name="default", overrides=["data.augmentation.enable=false"])
        res224 = compose(config_name="default", overrides=["img_size=224"])
        res336 = compose(config_name="default", overrides=["img_size=336"])

    assert list(aggressive.logging.wandb_writer.tags) == ["dinov2", "aug_agg", 518]
    assert list(disabled.logging.wandb_writer.tags) == ["dinov2", "noaug", 518]
    assert list(res224.logging.wandb_writer.tags) == ["dinov2", "aug_mod", 224]
    assert list(res336.logging.wandb_writer.tags) == ["dinov2", "aug_mod", 336]


def test_dinov3_special_token_removal_and_grid_validation():
    from vggt.models.dinov3 import extract_patch_tokens

    hidden = torch.arange(261 * 8).reshape(1, 261, 8)
    patches = extract_patch_tokens(hidden, height=256, width=256)
    assert patches.shape == (1, 256, 8)
    assert torch.equal(patches, hidden[:, 5:])

    with pytest.raises(ValueError, match="divisible"):
        extract_patch_tokens(hidden, height=255, width=256)
    with pytest.raises(ValueError, match="patch_size=0"):
        extract_patch_tokens(hidden, height=256, width=256, patch_size=0)
    with pytest.raises(ValueError, match="returned 260 tokens"):
        extract_patch_tokens(hidden[:, :-1], height=256, width=256)


def test_dpt_patch16_reshape_and_output():
    from vggt.heads.dpt_head import DPTHead

    head = DPTHead(
        dim_in=8,
        patch_size=16,
        output_dim=4,
        features=8,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
        pos_embed=False,
    )
    tokens = [torch.randn(1, 1, 261, 8) for _ in range(4)]
    images = torch.randn(1, 1, 3, 256, 256)
    prediction, confidence = head(tokens, images, patch_start_idx=5)
    assert prediction.shape == (1, 1, 256, 256, 3)
    assert confidence.shape == (1, 1, 256, 256)


@pytest.mark.parametrize("target_size,patch_size", [(255, 16), (256, 0), (-256, 16)])
def test_dataset_rejects_invalid_patch_grid(target_size, patch_size):
    from data.datasets.mri_dataset import validate_patch_grid

    with pytest.raises(ValueError):
        validate_patch_grid(target_size, patch_size)


def test_dataset_accepts_default_and_dinov3_patch_grids():
    from data.datasets.mri_dataset import validate_patch_grid

    assert validate_patch_grid(518, 14) == (518, 14)
    assert validate_patch_grid(256, 16) == (256, 16)


def test_patch_embed_freeze_pattern_with_fake_backbone():
    from train_utils.freeze import freeze_modules

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = nn.Module()
            self.aggregator.patch_embed = nn.Linear(2, 2)
            self.aggregator.frame_blocks = nn.Linear(2, 2)
            self.point_head = nn.Linear(2, 2)

    model = FakeModel()
    freeze_modules(model, patterns=["*patch_embed*"], recursive=True)
    model.train()
    assert not any(p.requires_grad for p in model.aggregator.patch_embed.parameters())
    assert model.aggregator.patch_embed.training is False
    assert all(p.requires_grad for p in model.aggregator.frame_blocks.parameters())
    assert all(p.requires_grad for p in model.point_head.parameters())


def test_new_constructor_options_preserve_positional_apis():
    from data.datasets.mri_dataset import MRIDataset
    from vggt.models.vggt import VGGT

    model_args = inspect.signature(VGGT).bind(518, 14, 1024)
    assert model_args.arguments["embed_dim"] == 1024
    assert "backbone" not in model_args.arguments

    dataset_args = inspect.signature(MRIDataset).bind(
        object(), "/data", "val", "/split", "dynamic", 12, 518, "axial"
    )
    assert dataset_args.arguments["mri_mode"] == "axial"
    assert "patch_size" not in dataset_args.arguments


def test_seed_builder_accepts_raw_and_wrapped_states(tmp_path):
    from tools.build_dinov3_seed import _load_model_state

    state = {"weight": torch.ones(2)}
    raw = tmp_path / "raw.pt"
    wrapped = tmp_path / "wrapped.pt"
    torch.save(state, raw)
    torch.save({"model": state}, wrapped)
    assert torch.equal(_load_model_state(raw)["weight"], state["weight"])
    assert torch.equal(_load_model_state(wrapped)["weight"], state["weight"])


def test_seed_builder_filters_source_only_keys_and_injects_deterministic_z():
    from tools.build_dinov3_seed import (
        NEW_BACKBONE_PREFIX,
        _assemble_hybrid,
        _z_embedder_state,
    )

    z_state = _z_embedder_state()
    target_shapes = {
        "aggregator.camera_token": (1,),
        **{name: tuple(tensor.shape) for name, tensor in z_state.items()},
        f"{NEW_BACKBONE_PREFIX}embeddings.cls_token": (1, 1, 4),
    }
    base = {
        "aggregator.camera_token": torch.ones(1),
        "aggregator.patch_embed.old": torch.ones(1),
        "camera_head.retired": torch.ones(1),
    }
    dinov3 = {"embeddings.cls_token": torch.ones(1, 1, 4)}
    hybrid = _assemble_hybrid(base, dinov3, target_shapes)

    assert set(hybrid) == set(target_shapes)
    for name, tensor in z_state.items():
        assert torch.equal(hybrid[name], tensor)
    second = _assemble_hybrid(base, dinov3, target_shapes)
    for name in z_state:
        assert torch.equal(hybrid[name], second[name])


def test_seed_builder_no_clobber_is_enforced_at_publication(tmp_path):
    from tools.build_dinov3_seed import _save_weights_only

    output = tmp_path / "seed.pt"
    torch.save({"sentinel": True}, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        _save_weights_only({"weight": torch.ones(1)}, output, overwrite=False)
    assert output.read_bytes() == original
