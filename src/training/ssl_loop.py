"""
Training and prediction for the SSL classifiers on waveform batches
(TORGOCommandDataset + collate_fn dicts). The backbone masks time spans
itself in train mode (src/model/backbones.py), so there is no SpecAugment here.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.architecture import SSLCommandClassifier, set_trainable
from src.training.ssl_recipe import AdamWStage, make_optimizer, make_scheduler, stage_top_n


def train_epoch(model: SSLCommandClassifier, loader: DataLoader,
                optimizer: torch.optim.Optimizer, scheduler, device: torch.device,
                max_grad_norm: float,
                class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    """One pass over `loader`; the scheduler steps after every batch. Returns (mean loss, accuracy)."""
    model.train()  # also in the head-only stage: backbone masking and dropout stay on
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_loss, correct, seen = 0.0, 0, 0
    for batch in loader:
        audio, labels = batch["input_values"].to(device), batch["labels"].to(device)
        out = model(audio, labels=labels, class_weights=class_weights)
        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += out["loss"].item() * len(labels)
        correct += (out["logits"].argmax(dim=-1) == labels).sum().item()
        seen += len(labels)
    if seen == 0:
        raise ValueError("the training loader yielded no batches")
    return total_loss / seen, correct / seen


def run_stage(model: SSLCommandClassifier, stage: AdamWStage, loader: DataLoader,
              device: torch.device,
              class_weights: Optional[torch.Tensor] = None) -> List[dict]:
    """Train `model` in place for one stage; the last epoch is kept. Per-epoch history."""
    set_trainable(model, stage_top_n(model, stage))
    optimizer = make_optimizer(model, stage)
    scheduler = make_scheduler(optimizer, stage, len(loader))
    history = []
    for epoch in range(stage.epochs):
        loss, acc = train_epoch(model, loader, optimizer, scheduler, device,
                                stage.max_grad_norm, class_weights)
        history.append({"epoch": epoch + 1, "loss": loss, "acc": acc})
        print(f"    epoch {epoch + 1}/{stage.epochs}: loss {loss:.4f} acc {acc:.4f}", flush=True)
    return history


@torch.no_grad()
def predict(model: SSLCommandClassifier, loader: DataLoader,
            device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """(predicted class ids, labels) in loader order. Eval mode, no augmentation."""
    model.eval()
    preds, labels = [], []
    for batch in loader:
        logits = model(batch["input_values"].to(device))["logits"]
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(batch["labels"].cpu())
    return torch.cat(preds).numpy().astype(np.int64), torch.cat(labels).numpy().astype(np.int64)
