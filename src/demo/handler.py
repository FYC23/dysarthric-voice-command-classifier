"""
One demo request: audio in, both models' answers out. The two panels fail
independently: a Parakeet error never hides the BC-ResNet answer.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

from src.baselines.asr.scoring import lenient_pred
from src.demo.audio_input import prepare_input
from src.eval.constants import OOV

logger = logging.getLogger(__name__)

ASR_OFF = "Parakeet is turned off (--no-asr)."


class Spotter(Protocol):
    def predict(self, window: np.ndarray) -> Tuple[Dict[str, float], float]:
        ...


class Transcriber(Protocol):
    def transcribe(self, batch: Sequence[np.ndarray]) -> list:
        ...


@dataclass(frozen=True)
class AsrAnswer:
    transcript: Optional[str] = None
    command: Optional[str] = None      # None when nothing was transcribed
    latency_ms: Optional[float] = None
    error: Optional[str] = None        # set instead of the fields above


@dataclass(frozen=True)
class DemoResult:
    heard: np.ndarray                  # the 2 s window both models received
    speech_found: bool
    probabilities: Dict[str, float]
    kws_latency_ms: float
    asr: AsrAnswer


def _transcribe(asr: Optional[Transcriber], window: np.ndarray,
                load_error: Optional[str]) -> AsrAnswer:
    if asr is None:
        return AsrAnswer(error=load_error or ASR_OFF)
    start = time.perf_counter()
    try:
        transcript = asr.transcribe([window])[0]
    except Exception as err:  # one failing clip must not take down the BC-ResNet answer
        logger.exception("Parakeet failed on a clip")
        return AsrAnswer(error=f"Parakeet failed on this clip: {err}")
    latency_ms = 1000.0 * (time.perf_counter() - start)
    command = lenient_pred(transcript)
    return AsrAnswer(transcript=transcript, command=None if command == OOV else command,
                     latency_ms=latency_ms)


def recognize(sample_rate: int, samples: np.ndarray, kws: Spotter,
              asr: Optional[Transcriber], asr_load_error: Optional[str] = None) -> DemoResult:
    """Raises AudioInputError (user-facing message) before running any model."""
    model_input = prepare_input(sample_rate, samples)
    probabilities, kws_latency_ms = kws.predict(model_input.window)
    return DemoResult(heard=model_input.window, speech_found=model_input.speech_found,
                      probabilities=probabilities, kws_latency_ms=kws_latency_ms,
                      asr=_transcribe(asr, model_input.window, asr_load_error))
