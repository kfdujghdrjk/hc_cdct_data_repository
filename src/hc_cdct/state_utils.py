from __future__ import annotations

import torch


def unwrap_model(model):
    """Return the underlying module when a model is wrapped by DataParallel."""

    return model.module if isinstance(model, torch.nn.DataParallel) else model


def resolve_state_key(model, weight_name: str) -> str:
    """Resolve a state_dict key for both normal and DataParallel-wrapped models."""

    state_dict = model.state_dict()

    if weight_name in state_dict:
        return weight_name

    prefixed = f"module.{weight_name}"
    if prefixed in state_dict:
        return prefixed

    stripped = weight_name.removeprefix("module.")
    if stripped in state_dict:
        return stripped

    raise KeyError(
        f"Cannot find {weight_name!r} in model state_dict. "
        f"Example keys: {list(state_dict.keys())[:5]}"
    )


def get_state_tensor(model, weight_name: str) -> torch.Tensor:
    key = resolve_state_key(model, weight_name)
    return model.state_dict()[key]


def copy_state_tensor(model, weight_name: str, value: torch.Tensor):
    """Copy a tensor into a model state_dict entry and reload the state."""

    state_dict = model.state_dict()
    key = resolve_state_key(model, weight_name)
    state_dict[key].copy_(value)
    model.load_state_dict(state_dict)


def get_module_by_weight_name(model, weight_name: str):
    """
    Convert a weight name such as 'layer1.1.conv2.weight' into the owning module.

    This function is used when pruning only the watermark embedding layer.
    """

    module_path = weight_name.removeprefix("module.").removesuffix(".weight")
    module = unwrap_model(model)

    try:
        return module.get_submodule(module_path)
    except AttributeError as exc:
        raise AttributeError(
            f"Cannot resolve module path {module_path!r}. "
            "Check whether the embedding layer name matches the model architecture."
        ) from exc
