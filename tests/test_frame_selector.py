"""frame_selector (docs/130): selector shapes, table invariants, oracle, checkpoint guard. CPU, tiny."""
import numpy as np
import pytest
import torch

from frame_selector.model import FrameSelector
from frame_selector.oracle import oracle_sel
from frame_selector.predict import load_selector, select_from_tokens

N, C = 16, 1024          # 4x4 token grid stands in for DINO's 37x37


def test_selector_scores_every_frame_of_every_plane():
    torch.manual_seed(0)
    sel = FrameSelector().eval()
    P, F = 3, 5
    logits = sel(torch.randn(P, F, N, C), torch.randn(N, C), torch.randn(P), 0.1,
                 ref_plane_tok=torch.randn(7, N, C))
    assert logits.shape == (P, F) and torch.isfinite(logits).all()


def test_select_keeps_the_reference_plane_on_the_queried_frame():
    torch.manual_seed(0)
    D, F, ref = 6, 4, 2
    out = select_from_tokens(FrameSelector().eval(), torch.randn(D, F, N, C), ref, dz=8.0)
    assert out.shape == (F, D) and out.dtype == np.int64
    assert (out[:, ref] == np.arange(F)).all() and ((0 <= out) & (out < F)).all()


def test_oracle_picks_the_circularly_closest_frame():
    pos = [[0, 3, 6, 9], [11, 2, 5, 8]]                    # (D=2, frames=4) true cardiac positions
    man = {"n_cardphase": 12, "rhythm": {"pos_per_plane": pos, "ref_pos": [0, 6, 10, 3]}}
    sel, err = oracle_sel(man)
    assert sel.tolist() == [[0, 0], [2, 2], [3, 0], [1, 1]]   # target 10: 9 (d=1) / 11 (d=1, circular)
    assert err.tolist() == [[0, 1], [0, 1], [1, 1], [0, 1]]


def _save(tmp_path, args):
    torch.save({"model": FrameSelector().state_dict(), "args": args, "step": 1}, tmp_path / "best.pt")


def test_load_selector_accepts_the_kept_architecture(tmp_path):
    _save(tmp_path, {"variant": "B", "center": True, "norm": "slice", "attn_pool": True, "roi_weight": 1.0})
    assert isinstance(load_selector(str(tmp_path), "cpu"), FrameSelector)
    _save(tmp_path, {"roi_weight": 0.0})                    # post-port ckpt: no architecture keys
    assert isinstance(load_selector(str(tmp_path), "cpu"), FrameSelector)


@pytest.mark.parametrize("bad", [{"variant": "A"}, {"norm": "ln"}, {"attn_pool": False}, {"center": False}])
def test_load_selector_rejects_dropped_variants(tmp_path, bad):
    _save(tmp_path, {"variant": "B", "center": True, "norm": "slice", "attn_pool": True, **bad})
    with pytest.raises(ValueError, match="kept architecture"):
        load_selector(str(tmp_path), "cpu")
