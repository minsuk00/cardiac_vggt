import torch
from torch import nn
from torch.nn import functional as F
from .encodings import make_grid


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, outputs))


class CiNeVol(nn.Module):
    def __init__(self, config, bounds, n_slices, n_frames):
        super().__init__()
        self.config = config
        bounds = torch.as_tensor(bounds, dtype=torch.float32)
        self.register_buffer("centre", bounds.mean(0))
        self.register_buffer("extent", (bounds[1] - bounds[0]).max())
        self.spatial = make_grid(config["spatial_grid"], config["backend"])
        self.cardiac_grids = nn.ModuleList([make_grid(config["spatiotemporal_grids"], config["backend"]) for _ in range(3)])
        self.respiratory_grids = nn.ModuleList([make_grid(config["spatiotemporal_grids"], config["backend"]) for _ in range(3)])
        spatial_dim = self.spatial.output_dim
        motion_dim = spatial_dim + 3 * self.cardiac_grids[0].output_dim
        width = config["mlps"]["hidden_width"]
        self.cardiac_net = mlp(motion_dim, 3, width)
        self.respiratory_net = mlp(motion_dim, 3, width)
        self.intensity_net = mlp(spatial_dim, 17, width)
        self.bias_net = mlp(20, 1, width)
        self.correction_net = mlp(48, 1, width)
        self.slice_embedding = nn.Embedding(n_slices, 16)
        self.frame_embedding = nn.Embedding(n_frames, 16)
        self.scale_logits = nn.Parameter(torch.zeros(n_slices))
        for network in (self.cardiac_net, self.respiratory_net):
            nn.init.uniform_(network[-1].weight, -1e-4, 1e-4)
            nn.init.zeros_(network[-1].bias)
        # Start nuisance heads near neutral; all layers remain trainable.
        for network in (self.bias_net, self.correction_net):
            nn.init.uniform_(network[-1].weight, -1e-4, 1e-4)
            nn.init.zeros_(network[-1].bias)

    def normalized(self, xyz):
        return (xyz - self.centre) / self.extent + 0.5

    def offsets(self, xyz, cardiac, respiratory):
        x = self.normalized(xyz)
        spatial = self.spatial(x)
        out = []
        for grids, network, state in ((self.cardiac_grids, self.cardiac_net, cardiac),
                                      (self.respiratory_grids, self.respiratory_net, respiratory)):
            encoded = [g(torch.cat((x[..., list(axes)], state[..., None]), -1))
                       for g, axes in zip(grids, ((0, 1), (0, 2), (1, 2)))]
            out.append(network(torch.cat([spatial] + encoded, -1)) * self.extent)
        return out

    def intensity(self, xyz, cardiac, respiratory):
        dc, dr = self.offsets(xyz, cardiac, respiratory)
        warped = xyz + dc + dr
        features = self.spatial(self.normalized(warped))
        z = self.intensity_net(features)
        return F.softplus(z[..., 0]), z[..., 1:], features, warped

    def forward(self, xyz, cardiac, respiratory, slices, frames):
        raw, f, features, warped = self.intensity(xyz, cardiac, respiratory)
        se, fe = self.slice_embedding(slices), self.frame_embedding(frames)
        b = self.bias_net(torch.cat((features[..., :4], se), -1))[..., 0]
        c = self.correction_net(torch.cat((f, se, fe), -1))[..., 0]
        scale = (self.scale_logits.softmax(0) * len(self.scale_logits))[slices]
        return {"raw": raw, "bias": b, "correction": c, "warped": warped,
                "prediction": scale * b.exp() * raw + c}

    def optimizer_groups(self):
        nets = (self.cardiac_net, self.respiratory_net, self.intensity_net, self.bias_net, self.correction_net)
        mlps = [p for net in nets for p in net.parameters()]
        ids = {id(p) for p in mlps}
        return [{"params": mlps, "weight_decay": self.config["optimization"]["mlp_weight_decay"]},
                {"params": [p for p in self.parameters() if id(p) not in ids], "weight_decay": 0.0}]
