# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch.nn as nn


class OptimizerWrapper:
    """A torch optimizer plus one scheduler per option (lr, weight_decay, ...), each a
    function of training progress `where` in [0, 1]."""

    def __init__(self, optimizer, schedulers):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.step_schedulers(0.0)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step_schedulers(self, where: float) -> None:
        for param_group in self.optimizer.param_groups:
            for option, scheduler in self.schedulers.items():
                param_group[option] = scheduler(where)


def construct_optimizer(model: nn.Module, optim_conf) -> OptimizerWrapper:
    """ONE param group with every model parameter (frozen ones included, so the
    optimizer state_dict layout matches existing checkpoints), and one scheduler per
    option from `optim_conf.options.<option>[0].scheduler`."""
    options = hydra.utils.instantiate(optim_conf.options)
    for option, cfgs in options.items():
        assert len(cfgs) == 1 and set(cfgs[0].keys()) == {"scheduler"}, (
            f"optim.options.{option}: per-parameter schedulers are not supported")
    schedulers = {option: cfgs[0]["scheduler"] for option, cfgs in options.items()}
    optimizer = hydra.utils.instantiate(optim_conf.optimizer,
                                        [{"params": list(model.parameters())}])
    for option in schedulers:
        assert option in optimizer.defaults, f"{option} is not an option of {optimizer}"
    return OptimizerWrapper(optimizer, schedulers)
