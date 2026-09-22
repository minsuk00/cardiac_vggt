"""Run explicitly on a CUDA node after scripts/setup.sh."""
import os
from pathlib import Path
import pytest
import torch
from cinevol.encodings import TorchHashGrid, Grid4DHashGrid

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def test_grid4d_forward_coordinate_and_parameter_second_gradients():
    torch.manual_seed(3)
    cpu = TorchHashGrid(3,2,[4]*3,[12,12,8],7).cuda()
    cuda = Grid4DHashGrid(3,2,[4]*3,[12,12,8],7).cuda()
    with torch.no_grad():
        cpu.embeddings.normal_(0, .2)
        cuda.grid.embeddings.copy_(cpu.embeddings)
    x = torch.tensor([[.13,.27,.43],[.71,.62,.33]], device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = cpu(x), cuda(y)
    torch.testing.assert_close(a,b,rtol=1e-3,atol=2e-6)
    ga = torch.autograd.grad(a.square().sum(),x,create_graph=True)[0]
    gb = torch.autograd.grad(b.square().sum(),y,create_graph=True)[0]
    torch.testing.assert_close(ga,gb,rtol=2e-3,atol=2e-6)
    ga.square().sum().backward()
    gb.square().sum().backward()
    torch.testing.assert_close(cpu.embeddings.grad,cuda.grid.embeddings.grad,rtol=3e-3,atol=2e-6)
    assert cuda.grid.embeddings.grad.abs().sum() > 0


def test_grid4d_complete_model_backward():
    from cinevol.config import configuration
    from cinevol.model import CiNeVol
    from cinevol.fit import batch_gradient
    c = configuration("phantom", True)
    c["backend"] = "grid4d"
    m = CiNeVol(c, [[-20]*3, [20]*3], 2, 3).cuda()
    n = 8
    b = {"xyz":torch.randn(n,3,device="cuda"), "value":torch.rand(n,device="cuda"),
         "cardiac":torch.rand(n,device="cuda"), "respiratory":torch.rand(n,device="cuda"),
         "slice_index":torch.arange(n,device="cuda")%2, "frame_index":torch.arange(n,device="cuda")%3,
         "rotation":torch.eye(3,device="cuda").repeat(n,1,1),
         "spacing":torch.tensor([2.,2.,4.],device="cuda").repeat(n,1)}
    result = batch_gradient(m,b,torch.randn(n,16,3,device="cuda"),3)
    assert result["total"] > 0
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
