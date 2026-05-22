from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.prune as prune


def get_parameters_to_prune(model, prune_bias: bool = False):
    """Collect all weight parameters, and optionally bias parameters, for global pruning."""

    parameters_to_prune = []

    for _, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            parameters_to_prune.append((module, "weight"))

        if prune_bias and hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            parameters_to_prune.append((module, "bias"))

    return parameters_to_prune


def apply_global_l1_pruning(parameters_to_prune, amount: float):
    """Apply global L1 unstructured pruning and make pruning permanent."""

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for module, parameter_name in parameters_to_prune:
        prune.remove(module, parameter_name)
