"""Tests for one demo request through both models, with stub models."""

import numpy as np
import pytest

from src.config import Config
from src.demo.audio_input import AudioInputError
from src.demo.handler import ASR_OFF, recognize
from tests.conftest import word_clip

SR = Config.SAMPLE_RATE
PROBS = {"four": 0.9, "for": 0.0}


class StubKws:
    def __init__(self):
        self.windows = []

    def predict(self, window):
        self.windows.append(window)
        return dict(PROBS), 1.5


class StubAsr:
    def __init__(self, transcript="for", error=None):
        self.transcript, self.error, self.batches = transcript, error, []

    def transcribe(self, batch):
        self.batches.append(batch)
        if self.error:
            raise self.error
        return [self.transcript]


def test_both_models_hear_the_same_window():
    kws, asr = StubKws(), StubAsr()
    result = recognize(SR, word_clip(1.5), kws, asr)
    assert result.heard.shape == (Config.MAX_AUDIO_SAMPLES,)
    np.testing.assert_array_equal(kws.windows[0], result.heard)
    np.testing.assert_array_equal(asr.batches[0][0], result.heard)
    assert result.speech_found


def test_transcript_is_mapped_leniently():
    result = recognize(SR, word_clip(1.5), StubKws(), StubAsr("for"))
    assert result.probabilities == PROBS
    assert result.kws_latency_ms == 1.5
    assert result.asr.transcript == "for"
    assert result.asr.command == "four"
    assert result.asr.latency_ms >= 0
    assert result.asr.error is None


def test_empty_transcript_has_no_command():
    result = recognize(SR, word_clip(1.5), StubKws(), StubAsr(""))
    assert result.asr.transcript == ""
    assert result.asr.command is None


def test_asr_failure_keeps_the_kws_answer():
    result = recognize(SR, word_clip(1.5), StubKws(), StubAsr(error=RuntimeError("boom")))
    assert result.probabilities == PROBS
    assert "boom" in result.asr.error
    assert result.asr.command is None


def test_asr_that_failed_to_load_reports_why():
    result = recognize(SR, word_clip(1.5), StubKws(), None, "Parakeet could not be loaded: x")
    assert result.asr.error == "Parakeet could not be loaded: x"


def test_asr_turned_off():
    result = recognize(SR, word_clip(1.5), StubKws(), None)
    assert result.asr.error == ASR_OFF


def test_bad_audio_raises_before_any_model_runs():
    kws, asr = StubKws(), StubAsr()
    with pytest.raises(AudioInputError):
        recognize(SR, np.zeros(10), kws, asr)
    assert kws.windows == [] and asr.batches == []
