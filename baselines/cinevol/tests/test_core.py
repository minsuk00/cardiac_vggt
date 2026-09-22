import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from cinevol.config import configuration
from cinevol.encodings import TorchHashGrid
from cinevol.model import CiNeVol
from cinevol.psf import training_samples, FWHM_TO_SIGMA
from cinevol.losses import pair_tv, rigidity, loss_terms, weighted_loss
from cinevol.fit import batch_gradient

torch.set_num_threads(2)


def batch(n=5):
    return {"xyz": torch.randn(n, 3), "value": torch.rand(n),
            "cardiac": torch.rand(n), "respiratory": torch.rand(n),
            "slice_index": torch.arange(n) % 3, "frame_index": torch.arange(n) % 4,
            "spacing": torch.tensor([2., 2., 4.]).repeat(n, 1),
            "rotation": torch.eye(3).repeat(n, 1, 1)}


def model():
    return CiNeVol(configuration("phantom", True), [[-20]*3, [20]*3], 3, 4)


def test_published_settings_and_dimensions():
    c = configuration("invivo")
    assert c["optimization"] == dict(steps_per_subject=500, batch_observed_pixels=32768,
                                     optimizer="AdamW", learning_rate=.005, mlp_weight_decay=.01)
    assert c["loss_weights"]["jacobian_cardiac"] == .0001
    assert c["psf_samples"] == {"fitting": 16, "inference": 16}
    m = CiNeVol(c, [[-20]*3, [20]*3], 3, 4)
    assert m.cardiac_net[0].in_features == 224
    assert m.intensity_net[-1].out_features == 17
    assert m.correction_net[0].in_features == 48
    assert m.bias_net[0].in_features == 20
    assert len(m.cardiac_grids) == len(m.respiratory_grids) == 3


def test_hash_coordinate_and_second_gradients():
    torch.manual_seed(1)
    m = TorchHashGrid(2, 2, [3]*3, [6,6,4], 5).double()
    x = torch.tensor([[.23,.31,.42],[.71,.53,.61]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(m, (x,), eps=1e-6, atol=1e-5)
    grad = torch.autograd.grad(m(x).square().sum(), x, create_graph=True)[0]
    loss = grad.square().sum()
    loss.backward()
    assert torch.isfinite(m.embeddings.grad).all()
    assert m.embeddings.grad.abs().sum() > 0
    assert torch.equal(m(torch.tensor([[-.1, .2, .3]], dtype=torch.double)), torch.zeros(1,4,dtype=torch.double))


def test_psf_orientation_and_thickness():
    b = batch(1)
    b["xyz"].zero_()
    b["rotation"] = torch.tensor([[[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]])
    out = training_samples(b, torch.eye(3)[None])
    expected = torch.tensor([[[0.,2.4,0.],[0.,0.,2.4],[4.,0.,0.]]]) * FWHM_TO_SIGMA
    torch.testing.assert_close(out, expected)


def test_half_pair_reduction():
    xyz = torch.tensor([[[0.,0.,0.],[0.,0.,0.],[2.,0.,0.],[2.,0.,0.]]])
    values = torch.tensor([[1.,1.,3.,3.]])
    assert pair_tv(values, xyz).item() == .5


def test_rigidity_of_full_map():
    x = torch.randn(4,3,requires_grad=True)
    rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    assert rigidity(x, x).item() == 0
    assert rigidity(x + 10, x).item() == 0
    assert rigidity(x @ rotation, x).item() == 0
    assert rigidity(2*x, x).item() == 27


def test_chunking_preserves_global_bias_loss_and_gradients():
    torch.manual_seed(12)
    a = model()
    # Nontrivial signed bias values are essential to distinguish mean(|b|)^2.
    torch.nn.init.normal_(a.bias_net[-1].weight, std=.2)
    b = copy.deepcopy(a)
    data = batch(5)
    noise = torch.randn(5,16,3)
    whole = loss_terms(a, data, training_samples(data, noise))
    weighted_loss(whole, a.config).backward()
    divided = batch_gradient(b, data, noise, 2)
    for k, value in whole.items():
        assert divided[k] == pytest.approx(float(value.detach()), rel=1e-4, abs=1e-7)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-3, atol=2e-6, msg=name)


def test_dynamic_inference_ignores_nuisance_heads():
    torch.manual_seed(17)
    m = model()
    x = torch.randn(4,3)
    phi, psi = torch.rand(4), torch.rand(4)
    before = m.intensity(x, phi, psi)[0]
    with torch.no_grad():
        m.scale_logits.fill_(30)
        m.bias_net[-1].bias.fill_(8)
        m.correction_net[-1].bias.fill_(100)
    torch.testing.assert_close(before, m.intensity(x,phi,psi)[0])
    with torch.no_grad():
        # Ensure a non-negligible learned state dependence for this fixture.
        for g in m.cardiac_grids:
            g.embeddings.normal_(0, .5)
        m.cardiac_net[-1].weight.normal_(0, .1)
        m.spatial.embeddings.normal_(0, .5)
    assert not torch.allclose(m.intensity(x,phi,psi)[0], m.intensity(x,phi*.2,psi)[0])
