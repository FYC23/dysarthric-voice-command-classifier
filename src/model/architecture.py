"""Classifier heads on pretrained self-supervised speech backbones."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over the time dimension.
    
    Instead of simple mean pooling, this learns which frames are most
    important for classification.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            
        Returns:
            pooled: (batch, hidden_size)
        """
        attention_scores = self.attention(hidden_states)
        attention_weights = F.softmax(attention_scores, dim=1)
        pooled = (hidden_states * attention_weights).sum(dim=1)
        return pooled


class LayerWeightedSum(nn.Module):
    """
    Softmax-weighted sum of every hidden state (the SUPERB setup). Each state
    is layer-normalised first, without learned scale: raw magnitudes differ a
    lot between layers (most in pre-norm stacks, where only the last state is
    normalised), and the loudest layer would otherwise dominate. The weights
    start at zero, so training starts from a plain average.
    """

    def __init__(self, num_states: int):
        super().__init__()
        if num_states < 1:
            raise ValueError(f"need at least one hidden state, got {num_states}")
        self.weights = nn.Parameter(torch.zeros(num_states))

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(hidden_states) != self.weights.numel():
            raise ValueError(f"expected {self.weights.numel()} hidden states, "
                             f"got {len(hidden_states)}")
        normed = torch.stack([F.layer_norm(h, h.shape[-1:]) for h in hidden_states])
        mix = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (mix * normed).sum(dim=0)


class CommandHead(nn.Module):
    """Weighted sum of layers -> attention pooling -> MLP. Trained in every stage."""

    def __init__(self, num_states: int, hidden_size: int, num_labels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_weights = LayerWeightedSum(num_states)
        self.attention_pooling = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.attention_pooling(self.layer_weights(hidden_states)))


class SSLCommandClassifier(nn.Module):
    """
    A pretrained backbone (src/model/backbones.py) and a CommandHead over all
    of its hidden states. Inputs are fixed 2 s windows, so no attention mask.
    """

    def __init__(self, backbone: HubertModel, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        config = backbone.config
        self.head = CommandHead(config.num_hidden_layers + 1, config.hidden_size,
                                num_labels, dropout)

    @property
    def num_layers(self) -> int:
        return self.backbone.config.num_hidden_layers

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None,
                class_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hidden = self.backbone(input_values=input_values, output_hidden_states=True)
        logits = self.head(hidden.hidden_states)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels, weight=class_weights)
        return result


def set_trainable(model: SSLCommandClassifier, top_n: int) -> None:
    """
    Set requires_grad on every parameter: the head and the last `top_n`
    transformer layers train, everything else is frozen (CNN front end,
    feature projection, positional convolution, final encoder layer norm and
    the lower layers). Never leaves a parameter in its previous state.

    Also clears the CNN front end's own `_requires_grad` flag (via
    `_freeze_parameters`, HubertFeatureEncoder's usual entry point): its
    forward force-sets its output tensor's requires_grad whenever that flag
    is set and the model is training, regardless of its parameters'
    requires_grad, which would otherwise build a live autograd graph through
    a "frozen" CNN on every step.
    """
    layers = model.backbone.encoder.layers
    if not 0 <= top_n <= len(layers):
        raise ValueError(f"top_n must be in [0, {len(layers)}], got {top_n}")
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.backbone.feature_extractor._freeze_parameters()
    for layer in layers[len(layers) - top_n:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for p in model.head.parameters():
        p.requires_grad_(True)
