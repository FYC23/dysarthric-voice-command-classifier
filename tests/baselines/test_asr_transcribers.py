"""Model registry and device choice. The real models are exercised by hand (Task 6)."""

import pytest
import torch

from src.baselines.asr.transcribers import DTYPES, MODELS, build_transcriber, resolve_device


def test_registry_holds_exactly_the_two_agreed_models():
    assert {k: m.hf_id for k, m in MODELS.items()} == {
        "whisper-large-v3": "openai/whisper-large-v3",
        "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
    }
    assert "30 s" in MODELS["whisper-large-v3"].cost_note


def test_float32_is_the_default_choice():
    assert list(DTYPES)[0] == "float32" and DTYPES["float32"] is torch.float32


def test_unknown_model_is_rejected_before_any_download():
    with pytest.raises(ValueError, match="parakeet-tdt-0.6b-v2"):
        build_transcriber("parakeet-tdt-0.6b-v2", torch.device("cpu"), torch.float32)


def test_explicit_device_is_used():
    assert resolve_device("cpu") == torch.device("cpu")


def test_unavailable_device_fails_clearly(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="mps"):
        resolve_device("mps")


def test_default_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device(None) == torch.device("mps")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device(None) == torch.device("cpu")
