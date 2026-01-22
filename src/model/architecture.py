"""
HuBERT-based model architecture for voice command classification.
"""

from typing import Dict, Optional

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


class HuBERTForCommandClassification(nn.Module):
    """
    HuBERT model with a classification head for voice command recognition.
    
    Architecture:
    - HuBERT encoder (frozen initially, then fine-tuned)
    - Attention pooling over time dimension
    - MLP classifier
    """
    
    def __init__(
        self,
        model_path: str,
        num_labels: int,
        hidden_size: int = 1024,
        classifier_dropout: float = 0.1,
        freeze_encoder: bool = True,
        freeze_feature_extractor: bool = True
    ):
        super().__init__()
        
        self.hubert = HubertModel.from_pretrained(model_path)
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        self.attention_pooling = AttentionPooling(hidden_size)
        
        classifier_hidden = hidden_size // 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_labels)
        )
        
        if freeze_encoder:
            self.freeze_encoder()
        
        if freeze_feature_extractor:
            self.freeze_feature_extractor()
    
    def freeze_encoder(self):
        """Freeze all HuBERT encoder parameters."""
        for param in self.hubert.parameters():
            param.requires_grad = False
        print("HuBERT encoder frozen")
    
    def unfreeze_encoder(self, num_layers: Optional[int] = None):
        """
        Unfreeze HuBERT encoder parameters.
        
        Args:
            num_layers: If specified, only unfreeze the top N transformer layers.
        """
        if num_layers is None:
            for param in self.hubert.encoder.parameters():
                param.requires_grad = True
            print("All HuBERT encoder layers unfrozen")
        else:
            total_layers = len(self.hubert.encoder.layers)
            for i, layer in enumerate(self.hubert.encoder.layers):
                if i >= total_layers - num_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            print(f"Top {num_layers} HuBERT encoder layers unfrozen")
    
    def freeze_feature_extractor(self):
        """Freeze the CNN feature extractor (always recommended)."""
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.hubert.feature_projection.parameters():
            param.requires_grad = False
        print("HuBERT feature extractor frozen")
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_values: (batch, seq_len) raw audio waveform
            attention_mask: (batch, seq_len) optional attention mask
            labels: (batch,) optional labels for loss computation
            class_weights: (num_classes,) optional weights for CrossEntropyLoss
            
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        outputs = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        hidden_states = outputs.last_hidden_state
        pooled = self.attention_pooling(hidden_states)
        logits = self.classifier(pooled)
        
        result = {'logits': logits}
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())