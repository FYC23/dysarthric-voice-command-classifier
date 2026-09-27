"""Tests for the Gradio app's wiring and rendering, with stub models (no model loads)."""

from pathlib import Path

import gradio as gr
import numpy as np
import pytest

import app
from src.config import Config
from src.demo.handler import AsrAnswer
from src.demo.results_table import DemoResults, ModelCost
from tests.conftest import word_clip

RESULTS = DemoResults(table_markdown="| t |\n", kws_cost=ModelCost("323k", "171M"),
                      asr_cost=ModelCost("627M", "16.7G"))
PROBS = {"four": 0.9, "zero": 0.1}


class StubKws:
    def predict(self, window):
        return dict(PROBS), 2.0


class StubAsr:
    def transcribe(self, batch):
        return ["for"]


def handler(asr=StubAsr(), asr_error=None):
    return app.make_handler(StubKws(), asr, asr_error, RESULTS, "CPU")


def test_default_checkpoint_prefers_the_space_bundle(tmp_path, monkeypatch):
    space, local = tmp_path / "weights" / "bcresnet-8.pt", tmp_path / "deploy.pt"
    space.parent.mkdir()
    space.write_bytes(b"x")
    local.write_bytes(b"x")
    monkeypatch.setattr(app, "SPACE_CHECKPOINT", space)
    monkeypatch.setattr(app, "LOCAL_CHECKPOINT", local)
    assert app.default_checkpoint() == space


def test_default_checkpoint_falls_back_to_runs(tmp_path, monkeypatch):
    local = tmp_path / "deploy.pt"
    local.write_bytes(b"x")
    monkeypatch.setattr(app, "SPACE_CHECKPOINT", tmp_path / "missing.pt")
    monkeypatch.setattr(app, "LOCAL_CHECKPOINT", local)
    assert app.default_checkpoint() == local


def test_no_checkpoint_anywhere_says_how_to_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "SPACE_CHECKPOINT", tmp_path / "a.pt")
    monkeypatch.setattr(app, "LOCAL_CHECKPOINT", tmp_path / "b.pt")
    with pytest.raises(FileNotFoundError, match="--checkpoint"):
        app.default_checkpoint()


def test_explicit_checkpoint_wins():
    assert app.resolve_checkpoint("runs/x/fold1_F01.pt") == Path("runs/x/fold1_F01.pt")


def test_header_lists_all_twenty_commands_and_the_closed_set():
    header = app.header_markdown(RESULTS)
    for word in Config.TARGET_COMMANDS:
        assert word in header
    assert "323k" in header and "627M" in header
    assert "always picks one of the 20" in header


def test_no_audio_asks_for_a_recording():
    heard, notice, label, kws_md, asr_md = handler()(None)
    assert heard is None and label is None
    assert notice == app.EMPTY_NOTICE


def test_bad_audio_shows_the_reason():
    heard, notice, label, _, _ = handler()((Config.SAMPLE_RATE, np.zeros(10)))
    assert heard is None and label is None
    assert "at least 0.1 s" in notice


def test_happy_path_renders_both_panels():
    heard, notice, label, kws_md, asr_md = handler()((Config.SAMPLE_RATE, word_clip(1.5)))
    assert heard[0] == Config.SAMPLE_RATE and heard[1].shape == (Config.MAX_AUDIO_SAMPLES,)
    assert notice == ""
    assert label == PROBS
    assert kws_md == "323k params · 171M MACs · 2 ms (CPU)"
    assert "“for”" in asr_md and "→ four" in asr_md and "627M params · 16.7G MACs" in asr_md


def test_silence_shows_the_no_speech_notice():
    _, notice, _, _, _ = handler()((Config.SAMPLE_RATE, np.zeros(Config.SAMPLE_RATE)))
    assert notice == app.NO_SPEECH_NOTICE


def test_asr_error_is_shown_in_its_panel():
    _, _, label, _, asr_md = handler(asr=None, asr_error="Parakeet could not be loaded: x")(
        (Config.SAMPLE_RATE, word_clip(1.5)))
    assert label == PROBS
    assert "Parakeet could not be loaded: x" in asr_md


def test_empty_transcript_says_nothing_recognised():
    md = app.asr_markdown(AsrAnswer(transcript="", latency_ms=5.0), RESULTS.asr_cost, "CPU")
    assert "nothing recognised" in md and "→" not in md


def test_create_app_builds_blocks():
    assert isinstance(app.create_app(handler(), RESULTS), gr.Blocks)


def test_cpu_is_the_default_device():
    assert app.parse_args([]).device == "cpu"
