"""
Off-the-shelf ASR models behind one interface: a batch of 16 kHz float32
windows in, one transcript per window out. transformers + PyTorch only,
the same stack as the trained models; weights cache in Config.MODEL_CACHE_DIR.
"""

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
import torch

from src.config import Config

DTYPES = {"float32": torch.float32, "float16": torch.float16}  # first = default


class Transcriber(Protocol):
    model: torch.nn.Module
    cost_note: str

    def transcribe(self, batch: Sequence[np.ndarray]) -> list:
        ...


class WhisperTranscriber:
    """Greedy, English, no prompt: the plain off-the-shelf behaviour."""

    def __init__(self, hf_id: str, device: torch.device, dtype: torch.dtype, cost_note: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        cache = str(Config.MODEL_CACHE_DIR)
        self.processor = WhisperProcessor.from_pretrained(hf_id, cache_dir=cache)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            hf_id, cache_dir=cache, dtype=dtype).to(device).eval()
        self.device, self.dtype, self.cost_note = device, dtype, cost_note

    def transcribe(self, batch: Sequence[np.ndarray]) -> list:
        # The processor pads every window to Whisper's fixed 30 s input
        features = self.processor(list(batch), sampling_rate=Config.SAMPLE_RATE,
                                  return_tensors="pt").input_features
        with torch.no_grad():
            ids = self.model.generate(features.to(self.device, self.dtype), language="en",
                                      task="transcribe", num_beams=1, do_sample=False)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


class ParakeetTranscriber:
    """Parakeet-TDT greedy decoding. v3 is multilingual; it cannot be forced to English."""

    def __init__(self, hf_id: str, device: torch.device, dtype: torch.dtype, cost_note: str):
        from transformers import AutoProcessor, ParakeetForTDT

        cache = str(Config.MODEL_CACHE_DIR)
        self.processor = AutoProcessor.from_pretrained(hf_id, cache_dir=cache)
        self.model = ParakeetForTDT.from_pretrained(
            hf_id, cache_dir=cache, dtype=dtype).to(device).eval()
        self.device, self.dtype, self.cost_note = device, dtype, cost_note

    def transcribe(self, batch: Sequence[np.ndarray]) -> list:
        inputs = self.processor(list(batch), sampling_rate=Config.SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["input_features"] = inputs["input_features"].to(self.dtype)
        with torch.no_grad():
            out = self.model.generate(**inputs)
        return [t.strip() for t in self.processor.batch_decode(out.sequences,
                                                               skip_special_tokens=True)]


@dataclass(frozen=True)
class AsrModel:
    hf_id: str
    cls: type
    cost_note: str


MODELS = {
    "whisper-large-v3": AsrModel(
        "openai/whisper-large-v3", WhisperTranscriber,
        "encoder padded to 30 s; greedy decode of one clip"),
    "parakeet-tdt-0.6b-v3": AsrModel(
        "nvidia/parakeet-tdt-0.6b-v3", ParakeetTranscriber,
        "greedy TDT decode of one clip; multilingual, language not forced"),
}


def build_transcriber(name: str, device: torch.device, dtype: torch.dtype) -> Transcriber:
    if name not in MODELS:
        raise ValueError(f"unknown ASR model {name!r}; choose from {sorted(MODELS)}")
    spec = MODELS[name]
    return spec.cls(spec.hf_id, device, dtype, spec.cost_note)


def resolve_device(requested: Optional[str]) -> torch.device:
    """The requested device if it exists; by default cuda, then mps, then cpu."""
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
                 "cpu": True}
    if requested is None:
        return torch.device(next(d for d, ok in available.items() if ok))
    if not available.get(requested, False):
        raise ValueError(f"device {requested!r} is not available here; "
                         f"available: {[d for d, ok in available.items() if ok]}")
    return torch.device(requested)
