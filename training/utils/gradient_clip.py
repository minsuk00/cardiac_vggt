# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class GradientClipper:
    """Per-module-group gradient-norm clipping."""

    def __init__(self, configs):
        """
        Args:
            configs: List of dictionaries, each containing:
                - module_name: list of str; a parameter belongs to the first config with
                  one of these substrings in its name
                - max_norm: float, maximum norm for gradient clipping
                - norm_type: int, type of norm (default: 2)
        """
        self.configs = [{
            'module_names': list(config['module_name']),
            'max_norm': float(config['max_norm']),
            'norm_type': config.get('norm_type', 2),
        } for config in configs]
        self.params_to_clip_by_config = None

    def setup_clipping(self, model: nn.Module) -> None:
        """
        Collect each config's trainable parameters by module name and check that every
        trainable parameter is covered. Call once at the beginning of each train epoch.
        """
        params_to_clip_by_config = []
        all_clipped_params = set()

        for config in self.configs:
            current_config_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    for module_name in config['module_names']:
                        if module_name in name:
                            current_config_params.append(param)
                            all_clipped_params.add(param)
                            break
            params_to_clip_by_config.append((config, current_config_params))

        remaining = [name for name, param in model.named_parameters()
                     if param.requires_grad and param not in all_clipped_params]
        if remaining:
            raise ValueError(f"{len(remaining)} trainable parameters are not configured for "
                             f"gradient clipping: {remaining}")

        self.params_to_clip_by_config = params_to_clip_by_config

    def __call__(self) -> dict:
        """Clip each config's gradients in place; return {",".join(module_names): norm}."""
        grad_norms = {}
        for config, params_to_clip in self.params_to_clip_by_config:
            # Empty under a head-only freeze (the aggregator group has no trainable params).
            if not params_to_clip:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=config['max_norm'],
                norm_type=config['norm_type']
            )
            grad_norms[",".join(config['module_names'])] = grad_norm.item()

        return grad_norms
