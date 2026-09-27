"""Fine-tuning checkpoints in the format src/training/finetune.py writes, for demo tests."""

from pathlib import Path
from typing import Optional, Sequence

import torch

from src.demo.kws import CHECKPOINT_CLASSES
from src.model.bcresnet import BCResNet


def save_checkpoint(path: Path, tau: float = 1.0, classes: Sequence[str] = CHECKPOINT_CLASSES,
                    model: Optional[BCResNet] = None) -> Path:
    model = model if model is not None else BCResNet(tau, len(classes))
    torch.save({"model_state_dict": model.state_dict(), "tau": float(tau), "seed": 0,
                "classes": list(classes), "pretrained": "pretrain.pt", "stages": []}, path)
    return path
