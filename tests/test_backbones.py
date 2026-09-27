"""Backbone table and loading: one masking setting for every backbone."""

import pytest
from transformers.models.hubert.modeling_hubert import _compute_mask_indices

from ssl_factories import tiny_spec
from src.model.backbones import (
    BACKBONES, MASKING_OVERRIDES, BackboneSpec, load_backbone, load_feature_extractor,
    unfreeze_count,
)

REAL_WINDOW_FRAMES = 99  # HuBERT frames in a 2 s window at 16 kHz


def test_the_three_backbones_and_their_checkpoints():
    assert BACKBONES == {
        "hubert-large": BackboneSpec("hubert-large", "facebook/hubert-large-ll60k"),
        "hubert-base": BackboneSpec("hubert-base", "facebook/hubert-base-ls960"),
        "distilhubert": BackboneSpec("distilhubert", "ntu-spml/distilhubert"),
    }


def test_masking_overrides_are_the_agreed_values():
    assert dict(MASKING_OVERRIDES) == {
        "apply_spec_augment": True, "mask_time_prob": 0.05, "mask_time_length": 10,
        "mask_time_min_masks": 1, "mask_feature_prob": 0.0, "layerdrop": 0.0,
    }


@pytest.mark.parametrize("stable", [False, True])
def test_load_backbone_replaces_the_checkpoint_masking(tmp_path, stable):
    config = load_backbone(tiny_spec(tmp_path, stable=stable)).config
    for key, value in MASKING_OVERRIDES.items():
        assert getattr(config, key) == value, key


def test_overrides_give_exactly_one_200ms_mask_per_2s_clip():
    o = MASKING_OVERRIDES
    mask = _compute_mask_indices((64, REAL_WINDOW_FRAMES), o["mask_time_prob"],
                                 o["mask_time_length"], min_masks=o["mask_time_min_masks"])
    assert (mask.sum(axis=1) == 10).all()


@pytest.mark.parametrize("stable", [False, True])
def test_feature_extractor_follows_the_checkpoint(tmp_path, stable):
    extractor = load_feature_extractor(tiny_spec(tmp_path, stable=stable))
    assert extractor.do_normalize is (not stable)


@pytest.mark.parametrize("layers, expected", [(24, 4), (12, 4), (4, 4), (2, 2), (1, 1)])
def test_unfreeze_count_is_top_4_capped(layers, expected):
    assert unfreeze_count(layers) == expected


def test_unfreeze_count_rejects_an_empty_encoder():
    with pytest.raises(ValueError):
        unfreeze_count(0)


def test_load_backbone_uses_eager_attention(tmp_path):
    # PyTorch's MPS scaled_dot_product_attention refuses dropout, and the
    # checkpoints keep attention_dropout=0.1, so SDPA cannot train on a Mac.
    assert load_backbone(tiny_spec(tmp_path)).config._attn_implementation == "eager"
