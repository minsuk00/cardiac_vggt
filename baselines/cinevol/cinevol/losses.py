import torch


def pair_tv(value, xyz, epsilon=1e-6):
    """Eq.10: denominator B*U, with U/2 disjoint pairs (not pair mean)."""
    half = value.shape[1] // 2
    distance = (xyz[:, :half] - xyz[:, half:]).norm(dim=-1).clamp_min(epsilon)
    return ((value[:, :half] - value[:, half:]).abs() / distance).sum() / value.numel()


def rigidity(mapped, xyz):
    rows = [torch.autograd.grad(mapped[:, i].sum(), xyz, create_graph=True, retain_graph=True)[0] for i in range(3)]
    jac = torch.stack(rows, dim=1)
    identity = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
    return ((jac.transpose(-1, -2) @ jac - identity).square().sum((-1, -2))).mean()


def loss_terms(model, batch, samples, global_abs_bias=None, jacobian=True):
    n, u, _ = samples.shape
    repeat = lambda x: x[:, None].expand(n, u)
    results = model(samples, repeat(batch["cardiac"]), repeat(batch["respiratory"]),
                    repeat(batch["slice_index"]), repeat(batch["frame_index"]))
    abs_bias = results["bias"].abs().mean()
    # The tangent has the exact full-batch gradient when chunks are accumulated.
    # The detached offset also gives the exact global value after weighted summation.
    bias_zero = abs_bias.square() if global_abs_bias is None else 2 * global_abs_bias * abs_bias - global_abs_bias.square()
    eps = model.config["implementation_choices"]["distance_epsilon_mm"]
    terms = {
        "mae": (results["prediction"].mean(1) - batch["value"]).abs().mean(),
        "tv": pair_tv(results["raw"], results["warped"], eps),
        "bias": bias_zero + pair_tv(results["bias"], results["warped"], eps),
        "correction": results["correction"].abs().mean(),
    }
    if jacobian:
        x = batch["xyz"].detach().requires_grad_(True)
        dc, dr = model.offsets(x, batch["cardiac"], batch["respiratory"])
        terms["jacobian_cardiac"] = rigidity(x + dc, x)
        terms["jacobian_respiratory"] = rigidity(x + dr, x)
    return terms


def weighted_loss(terms, config):
    return sum(config["loss_weights"][k] * v for k, v in terms.items())
