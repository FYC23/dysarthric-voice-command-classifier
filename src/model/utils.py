"""
Model utility functions.
"""

import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def compute_class_weights_tensor(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Uses sklearn's 'balanced' strategy:
    weight = n_samples / (n_classes * n_samples_for_class)
    """
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    
    # Create full weight tensor (in case some classes are missing)
    full_weights = np.ones(num_classes)
    for label, weight in zip(unique_labels, class_weights):
        full_weights[int(label)] = weight
    
    return torch.tensor(full_weights, dtype=torch.float32).to(device)


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.
    
    Returns:
        dict with 'total', 'trainable', 'frozen' counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen
    }