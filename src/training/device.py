"""Which torch device to train on, shared by every training script."""

from typing import Optional

import torch


def pick_device(name: Optional[str] = None) -> torch.device:
    """`name` if given, else CUDA, then MPS, then CPU."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
