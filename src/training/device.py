"""
Which torch device to train on, and the choices that depend on it; shared by
every training script.
"""

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


def attention_for(device: torch.device) -> Optional[str]:
    """
    The attention implementation a transformer backbone should use on
    `device`: "eager" on MPS, whose scaled_dot_product_attention refuses
    dropout on inputs that need no grad (every frozen layer in training
    mode); None (the library default, SDPA) everywhere else.
    """
    return "eager" if device.type == "mps" else None
