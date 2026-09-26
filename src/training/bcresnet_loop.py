"""
BC-ResNet training and prediction on waveform batches: log-Mel on the device,
SpecAugment on training batches only, then the model.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.bcresnet import BCResNet
from src.model.frontend import LogMel, SpecAugParams, spec_augment
from src.training.bcresnet_recipe import SgdStage, make_optimizer, make_scheduler


def _batch_parts(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """(audio, labels) from a TORGO dict batch or an (audio, labels) pair."""
    if isinstance(batch, dict):
        return batch["input_values"], batch["labels"]
    audio, labels = batch
    return audio, torch.as_tensor(labels)


def set_head_only(model: BCResNet, head_only: bool) -> None:
    """
    Freeze every weight but the class head's (head_only) or unfreeze everything.
    Only gradients are switched off: the model still trains in train mode, so
    batch-norm statistics keep adapting to the new data while the head warms up.
    """
    for p in model.parameters():
        p.requires_grad_(not head_only)
    for p in model.head.parameters():
        p.requires_grad_(True)


def train_epoch(model: BCResNet, loader: DataLoader, logmel: LogMel,
                spec_params: Optional[SpecAugParams], optimizer: torch.optim.Optimizer,
                scheduler, device: torch.device, generator: torch.Generator,
                class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """One pass over `loader`; the scheduler steps after every batch. Returns (mean loss, accuracy)."""
    model.train()  # also in head-only stages: batch norm adapts, frozen weights get no gradients
    total_loss, correct, seen = 0.0, 0, 0
    for batch in loader:
        audio, labels = _batch_parts(batch)
        audio, labels = audio.to(device), labels.to(device)
        with torch.no_grad():
            features = logmel(audio)
            if spec_params is not None:
                features = spec_augment(features, spec_params, generator)
        logits = model(features)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        seen += len(labels)
    if seen == 0:
        raise ValueError("the training loader yielded no batches")
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model: BCResNet, loader: DataLoader, logmel: LogMel,
            device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """(predicted class ids, labels) in loader order. No augmentation."""
    model.eval()
    preds, labels = [], []
    for batch in loader:
        audio, batch_labels = _batch_parts(batch)
        logits = model(logmel(audio.to(device)))
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(batch_labels.cpu())
    return torch.cat(preds).numpy().astype(np.int64), torch.cat(labels).numpy().astype(np.int64)


def run_stage(model: BCResNet, stage: SgdStage, loader: DataLoader, logmel: LogMel,
              spec_params: Optional[SpecAugParams], device: torch.device,
              generator: torch.Generator,
              class_weights: Optional[torch.Tensor] = None) -> List[dict]:
    """Train `model` in place for one stage; the last epoch is kept. Per-epoch history."""
    set_head_only(model, stage.head_only)
    try:
        optimizer = make_optimizer(model, stage)
        scheduler = make_scheduler(optimizer, stage, len(loader))
        history = []
        for epoch in range(stage.epochs):
            loss, acc = train_epoch(model, loader, logmel, spec_params, optimizer, scheduler,
                                    device, generator, class_weights)
            history.append({"epoch": epoch + 1, "loss": loss, "acc": acc})
        return history
    finally:
        set_head_only(model, False)
