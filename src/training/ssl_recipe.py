"""
The fixed training settings of the SSL backbones (step 2), identical for every
backbone, and where runs go. Fixed in advance and never tuned on LOSO
results: there is no development set, the held-out speaker is the only
unseen data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from src.config import Config
from src.model.architecture import SSLCommandClassifier
from src.model.backbones import unfreeze_count
from src.training.schedule import warmup_cosine_scheduler

SAMPLE_RATE = Config.SAMPLE_RATE
WINDOW_S = Config.MAX_AUDIO_LENGTH
WINDOW_SAMPLES = Config.MAX_AUDIO_SAMPLES


@dataclass(frozen=True)
class AdamWStage:
    """One stage: AdamW, per-step cosine from the stage's rates to 0, no warmup."""

    epochs: int
    head_lr: float
    encoder_lr: Optional[float]  # None exactly when the backbone stays frozen
    unfreeze: bool               # False: head only; True: head + top layers
    batch_size: int = 8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    class_weighted: bool = True  # "balanced" cross-entropy weights

    def __post_init__(self):
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError(f"epochs and batch_size must be positive: {self}")
        if self.head_lr <= 0:
            raise ValueError(f"head_lr must be positive: {self}")
        if self.unfreeze != (self.encoder_lr is not None):
            raise ValueError(f"encoder_lr must be set exactly when unfreeze is True: {self}")
        if self.encoder_lr is not None and self.encoder_lr <= 0:
            raise ValueError(f"encoder_lr must be positive: {self}")


CONTROL_HEAD_WARMUP = AdamWStage(epochs=5, head_lr=1e-4, encoder_lr=None, unfreeze=False)
CONTROL_FINETUNE = AdamWStage(epochs=10, head_lr=1e-5, encoder_lr=1e-6, unfreeze=True)
DYSARTHRIC_FINETUNE = AdamWStage(epochs=15, head_lr=5e-5, encoder_lr=5e-6, unfreeze=True)


def stage_top_n(model: SSLCommandClassifier, stage: AdamWStage) -> int:
    """Transformer layers this stage trains: 0, or the top min(4, layers)."""
    return unfreeze_count(model.num_layers) if stage.unfreeze else 0


def param_groups(model: SSLCommandClassifier, stage: AdamWStage) -> List[dict]:
    """
    [head group, trainable-backbone group], read from requires_grad so the
    optimizer can never disagree with the freezing. Call set_trainable first.
    """
    head = [p for p in model.head.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone and not stage.unfreeze:
        raise ValueError("backbone parameters are trainable in a head-only stage")
    if stage.unfreeze and not backbone:
        raise ValueError("an unfreeze stage has no trainable backbone parameters")
    groups = [{"params": head, "lr": stage.head_lr}]
    if stage.unfreeze:
        groups.append({"params": backbone, "lr": stage.encoder_lr})
    grouped = [id(p) for g in groups for p in g["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if len(grouped) != len(set(grouped)) or set(grouped) != trainable:
        raise ValueError("optimizer groups must hold every trainable parameter exactly once")
    return groups


def make_optimizer(model: SSLCommandClassifier, stage: AdamWStage) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups(model, stage), weight_decay=stage.weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, stage: AdamWStage,
                   steps_per_epoch: int) -> LambdaLR:
    """Cosine from each group's rate to 0 over the stage; call step() after every batch."""
    if steps_per_epoch <= 0:
        raise ValueError("the training set is smaller than one batch; "
                         f"batch_size={stage.batch_size}")
    return warmup_cosine_scheduler(optimizer, stage.epochs * steps_per_epoch, 0)


def run_name(backbone: str) -> str:
    """The model's name in eval-harness tables: the backbone's name."""
    return backbone


def controls_run_name(backbone: str) -> str:
    """The control-stage checkpoint scored on dysarthric speakers, e.g. hubert-base-controls."""
    return f"{backbone}-controls"


def seed_dir(runs_dir: Path, name: str, seed: int) -> Path:
    return Path(runs_dir) / name / f"seed{seed}"
