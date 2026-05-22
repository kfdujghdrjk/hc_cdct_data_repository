from __future__ import annotations

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn


def _import_local_models():
    try:
        import models  # type: ignore
        return models
    except Exception as exc:
        raise ImportError(
            "Cannot import local `models` package. Copy the original `models/` directory "
            "from your PyTorch-CIFAR project into the repository root, or install it as a package."
        ) from exc


def build_model(name: str, num_classes: int, device: str):
    """Build a model from the local PyTorch-CIFAR style `models` package."""

    models = _import_local_models()

    if not hasattr(models, name):
        raise ValueError(
            f"Model {name!r} was not found in local `models` package. "
            f"Available public names include: {[k for k in dir(models) if not k.startswith('_')][:20]}"
        )

    constructor = getattr(models, name)

    try:
        model = constructor(num_classes=num_classes)
    except TypeError:
        # Some PyTorch-CIFAR constructors do not expose num_classes.
        model = constructor()

    return model.to(device)


def load_checkpoint_if_needed(model, checkpoint_path: str | None, device: str, resume: bool = True):
    """Load a checkpoint when requested. Supports raw state_dict and {'net': state_dict} formats."""

    if not resume:
        return model

    if not checkpoint_path:
        raise ValueError("resume=True but checkpoint_path is empty.")

    checkpoint_path = str(checkpoint_path)
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}. "
            "Place the checkpoint under checkpoints/ or update configs/default.yaml."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state_dict = checkpoint["net"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def maybe_wrap_data_parallel(model, device: str, data_parallel: bool = True):
    if data_parallel and device.startswith("cuda"):
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    return model
