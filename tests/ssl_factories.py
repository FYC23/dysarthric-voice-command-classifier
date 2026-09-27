"""Tiny random HuBERT checkpoints for tests: no downloads, fast on CPU."""

from pathlib import Path

import torch
from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

from src.model.backbones import BackboneSpec

TINY_HIDDEN = 32


def tiny_backbone_dir(root: Path, layers: int = 3, stable: bool = False) -> Path:
    """
    Save a random HuBERT with the real models' 320x downsampling (100 frames
    per 2 s window) and masking defaults that differ from MASKING_OVERRIDES,
    so tests can see load_backbone replace them. `stable` picks the pre-norm
    encoder that HuBERT-large uses.
    """
    path = Path(root) / f"tiny-hubert-{layers}{'-stable' if stable else ''}"
    if path.exists():
        return path
    config = HubertConfig(
        hidden_size=TINY_HIDDEN, num_hidden_layers=layers, num_attention_heads=2,
        intermediate_size=64, conv_dim=(8, 8, 8), conv_kernel=(10, 8, 4),
        conv_stride=(5, 8, 8), num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=stable, feat_extract_norm="layer" if stable else "group",
        apply_spec_augment=False, mask_time_prob=0.3, mask_time_min_masks=2, layerdrop=0.1,
    )
    torch.manual_seed(0)
    HubertModel(config).save_pretrained(path)
    Wav2Vec2FeatureExtractor(do_normalize=not stable).save_pretrained(path)
    return path


def tiny_spec(root: Path, layers: int = 3, stable: bool = False,
              name: str = "tiny-hubert") -> BackboneSpec:
    return BackboneSpec(name=name, hf_id=str(tiny_backbone_dir(root, layers, stable)))
