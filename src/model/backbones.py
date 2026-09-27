"""
Pretrained self-supervised speech backbones for the accuracy-vs-size curve
(step 2), and how they load. All three are self-supervised only (no ASR
fine-tuning), so the curve varies size, not training objective. Every
backbone gets the same masking and layer-drop settings, so the curve does
not vary regularisation either.
"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional

from transformers import AutoFeatureExtractor, HubertModel

UNFREEZE_TOP_LAYERS = 4

# Overrides each checkpoint's defaults. On a 2 s window (99 frames) this is
# exactly one 10-frame (200 ms) time mask per clip, about 10% of the window.
# The library default mask_time_min_masks=2 would force two masks (19%)
# whatever mask_time_prob says. Layer drop is off: most layers are frozen, it
# would remove half of DistilHuBERT, and a dropped layer repeats a hidden
# state in the weighted sum.
MASKING_OVERRIDES: Mapping[str, object] = MappingProxyType({
    "apply_spec_augment": True,
    "mask_time_prob": 0.05,
    "mask_time_length": 10,
    "mask_time_min_masks": 1,
    "mask_feature_prob": 0.0,
    "layerdrop": 0.0,
})


@dataclass(frozen=True)
class BackboneSpec:
    name: str   # the run name in eval-harness tables and under runs/
    hf_id: str  # Hugging Face model id, or a local checkpoint directory


BACKBONES: Dict[str, BackboneSpec] = {spec.name: spec for spec in (
    BackboneSpec("hubert-large", "facebook/hubert-large-ll60k"),  # 315M, 24 layers
    BackboneSpec("hubert-base", "facebook/hubert-base-ls960"),    # 95M, 12 layers
    BackboneSpec("distilhubert", "ntu-spml/distilhubert"),        # 24M, 2 layers
)}


def unfreeze_count(num_hidden_layers: int) -> int:
    """How many top transformer layers fine-tuning trains: 4, or all if fewer."""
    if num_hidden_layers <= 0:
        raise ValueError(f"a backbone needs at least one layer, got {num_hidden_layers}")
    return min(UNFREEZE_TOP_LAYERS, num_hidden_layers)


def _cache(cache_dir: Optional[Path]) -> Optional[str]:
    return str(cache_dir) if cache_dir is not None else None


def load_backbone(spec: BackboneSpec, cache_dir: Optional[Path] = None,
                  attn_implementation: Optional[str] = None) -> HubertModel:
    """
    The pretrained backbone with MASKING_OVERRIDES applied. Downloads from
    Hugging Face on first use; set HF_ENDPOINT to use a mirror.
    `attn_implementation` ("eager", "sdpa", ...) picks the attention kernel;
    None keeps the library default (see src/training/device.attention_for).
    """
    return HubertModel.from_pretrained(spec.hf_id, cache_dir=_cache(cache_dir),
                                       attn_implementation=attn_implementation,
                                       **MASKING_OVERRIDES)


def load_feature_extractor(spec: BackboneSpec, cache_dir: Optional[Path] = None):
    """The checkpoint's own input preprocessing (whether it normalises each window)."""
    return AutoFeatureExtractor.from_pretrained(spec.hf_id, cache_dir=_cache(cache_dir))
