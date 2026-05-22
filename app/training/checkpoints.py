import os
from typing import Any

import torch


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, path: str
) -> None:
    """
    Saves a training checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    """
    Loads a training checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
