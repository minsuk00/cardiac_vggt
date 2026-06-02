"""Unit tests for the decoupled target-phase embedding (`target_t_embedder`).

These build a TINY conv-patch-embed Aggregator (no 300M DINOv2) so they run fast on CPU.
They guard the load-bearing properties of the surgery:
  - the target_t query actually wires through to the aggregated tokens (not silently dropped),
  - it's required when the flag is on (fail-loud, like z/t),
  - t_target = T/2 (norm 0.0) does NOT trip any guard,
  - flag-off keeps the module absent and the forward target_t-agnostic,
  - old checkpoints lacking the module load fine under strict=False (warm-start path).
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vggt.models.aggregator import Aggregator


def _tiny_aggregator(**kw):
    return Aggregator(
        img_size=28, patch_size=14, embed_dim=64, depth=2, num_heads=4,
        patch_embed="conv", use_z_pose_embedding=True, use_t_pose_embedding=True,
        **kw,
    ).eval()


def _inputs(B=1, S=2):
    torch.manual_seed(0)
    images = torch.rand(B, S, 3, 28, 28)
    z = torch.zeros(B, S, 1)
    t = torch.full((B, S, 1), 0.1)
    return images, z, t


def test_aggregator_raises_without_target_t():
    agg = _tiny_aggregator(use_target_t_pose_embedding=True)
    images, z, t = _inputs()
    with pytest.raises(ValueError, match="target_t_indices"):
        agg(images, z_indices=z, t_indices=t, target_t_indices=None)


def test_aggregator_target_t_changes_output():
    """Different target query, SAME images/z/t → different aggregated tokens. Guards against
    the embedder being constructed but never reaching the output (silent no-op)."""
    agg = _tiny_aggregator(use_target_t_pose_embedding=True)
    images, z, t = _inputs()
    tt_a = torch.full((1, 2, 1), -0.5)
    tt_b = torch.full((1, 2, 1), 0.5)
    with torch.no_grad():
        out_a, _ = agg(images, z_indices=z, t_indices=t, target_t_indices=tt_a)
        out_b, _ = agg(images, z_indices=z, t_indices=t, target_t_indices=tt_b)
    assert not torch.allclose(out_a[-1], out_b[-1], atol=1e-5), \
        "target_t_indices does not affect aggregator output — signal dropped"


def test_aggregator_embeddings_separate_and_additive():
    """The three pose embeddings must be SEPARATE modules summed into the camera token:
      - target_t_embedder is a DISTINCT instance from t_embedder (reusing it would make the
        camera token symmetric in (t_self, t_target) and destroy motion direction),
      - varying z, t, OR target_t each independently changes the output → all three CONTRIBUTE
        additively (none overwrites another)."""
    agg = _tiny_aggregator(use_target_t_pose_embedding=True)
    assert agg.target_t_embedder is not agg.t_embedder, \
        "target_t_embedder must be a SEPARATE instance from t_embedder"
    assert agg.target_t_embedder is not agg.z_embedder

    images, z, t = _inputs()
    tt = torch.full((1, 2, 1), 0.2)
    with torch.no_grad():
        base, _ = agg(images, z_indices=z, t_indices=t, target_t_indices=tt)
        out_z, _ = agg(images, z_indices=z + 0.5, t_indices=t, target_t_indices=tt)
        out_t, _ = agg(images, z_indices=z, t_indices=t + 0.3, target_t_indices=tt)
        out_tt, _ = agg(images, z_indices=z, t_indices=t, target_t_indices=torch.full((1, 2, 1), -0.4))
    for out, name in [(out_z, "z"), (out_t, "t_self"), (out_tt, "target_t")]:
        assert not torch.allclose(base[-1], out[-1], atol=1e-6), \
            f"varying {name} did not change output — it is being overwritten, not summed"


def test_aggregator_target_t_midpoint_no_error():
    """t_target = T/2 normalizes to 0.0 — must run cleanly (no `== 0.0` guard like z)."""
    agg = _tiny_aggregator(use_target_t_pose_embedding=True)
    images, z, t = _inputs()
    tt0 = torch.zeros(1, 2, 1)
    with torch.no_grad():
        out, _ = agg(images, z_indices=z, t_indices=t, target_t_indices=tt0)
    assert out[-1].shape[0] == 1


def test_aggregator_target_t_off_by_default():
    """Flag off → no module, no requirement; forward ignores target_t entirely."""
    agg = _tiny_aggregator()  # use_target_t_pose_embedding defaults False
    assert not hasattr(agg, "target_t_embedder")
    images, z, t = _inputs()
    with torch.no_grad():
        out, _ = agg(images, z_indices=z, t_indices=t, target_t_indices=None)
    assert out[-1].shape[0] == 1


def test_checkpoint_warmstart_tolerates_missing_target_t():
    """An OLD checkpoint (no target_t_embedder) must load into a target_t model under
    strict=False without raising — the new module lands in missing_keys (warm-start path)."""
    model_has = _tiny_aggregator(use_target_t_pose_embedding=True)
    sd_without = _tiny_aggregator(use_target_t_pose_embedding=False).state_dict()
    result = model_has.load_state_dict(sd_without, strict=False)  # must not raise
    assert any("target_t_embedder" in k for k in result.missing_keys), \
        f"expected target_t_embedder.* in missing_keys; got {result.missing_keys}"
    # No target_t key should be flagged unexpected (it simply isn't in the old ckpt).
    assert not any("target_t_embedder" in k for k in result.unexpected_keys)
