"""End to end with a fake transcriber: clips -> cached transcripts -> harness runs + cost."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from conftest import word_clip, write_wav
from src.baselines.asr.pipeline import (
    COST_FILE, TRANSCRIPTS_FILE, clip_window, load_cost, reference_clip, run_baseline, run_dir,
    save_cost, scored_run, transcribe_clips, transcription_cost,
)
from src.eval.constants import ARRAY_MIC, CONTROL_SPEAKERS, DYSARTHRIC_SPEAKERS, OOV
from src.eval.io import load_run

SPOKEN = {"yes": "Yes.", "two": "to", "menu": ""}  # label -> what the fake model "hears"


def clips_table():
    """Every dysarthric speaker plus one control, three words each, array mic."""
    rows = []
    for spk in DYSARTHRIC_SPEAKERS + CONTROL_SPEAKERS[:1]:
        for i, label in enumerate(SPOKEN):
            rows.append({"file_path": f"/fake/{spk}/{i}.wav", "speaker_id": spk,
                         "session": "Session1", "utterance_id": f"{i:04d}", "mic": ARRAY_MIC,
                         "label": label, "is_dysarthric": spk in DYSARTHRIC_SPEAKERS,
                         "seg_start": np.nan, "seg_end": np.nan})
    return pd.DataFrame(rows)


class FakeTranscriber:
    """Windows are the labels themselves (see load_label); counts what it transcribes."""

    def __init__(self):
        self.model = nn.Linear(4, 4)
        self.cost_note = "fake"
        self.seen = []

    def transcribe(self, batch):
        self.seen += list(batch)
        return [SPOKEN[w] for w in batch]


def load_label(row):
    return row["label"]


def test_runs_are_valid_zero_shot_and_load_back(tmp_path):
    fake = FakeTranscriber()
    runs = run_baseline("fake-asr", clips_table(), lambda: fake, tmp_path, batch_size=4,
                        load_window=load_label)
    assert [r.model for r in runs] == ["fake-asr-strict", "fake-asr-lenient"]
    for run in runs:
        loaded = load_run(run_dir(tmp_path, run.model))
        assert loaded.zero_shot and loaded.seed == 0
        assert "transcript" in loaded.predictions.columns
    strict, lenient = (load_run(run_dir(tmp_path, r.model)).predictions for r in runs)
    by_label = lambda p: p.groupby("label")["pred"].first().to_dict()
    assert by_label(strict) == {"yes": "yes", "two": OOV, "menu": OOV}
    assert by_label(lenient) == {"yes": "yes", "two": "two", "menu": OOV}


def test_cost_is_written_once_from_the_reference_clip(tmp_path):
    fake = FakeTranscriber()
    clips = clips_table()
    run_baseline("fake-asr", clips, lambda: fake, tmp_path, batch_size=4, load_window=load_label)
    cost = load_cost(tmp_path / "fake-asr" / COST_FILE)
    assert cost.params == 4 * 4 + 4 and cost.note == "fake"
    assert fake.seen[-1] == load_label(reference_clip(clips))


def test_complete_cache_and_cost_never_build_a_model(tmp_path):
    fake = FakeTranscriber()
    run_baseline("fake-asr", clips_table(), lambda: fake, tmp_path, batch_size=4,
                 load_window=load_label)

    def no_model():
        raise AssertionError("model built although transcripts and cost are cached")

    runs = run_baseline("fake-asr", clips_table(), no_model, tmp_path, batch_size=4,
                        load_window=load_label)
    assert len(runs) == 2


def test_resume_transcribes_only_missing_clips_without_duplicates(tmp_path):
    clips = clips_table()
    cache = tmp_path / TRANSCRIPTS_FILE
    first = FakeTranscriber()
    transcribe_clips(clips.head(5), lambda: first, cache, batch_size=2, load_window=load_label)
    second = FakeTranscriber()
    out = transcribe_clips(clips, lambda: second, cache, batch_size=2, load_window=load_label)
    assert len(second.seen) == len(clips) - 5
    assert len(out) == len(clips)
    assert len(pd.read_csv(cache)) == len(clips)


def test_crash_mid_batch_keeps_finished_batches(tmp_path):
    clips = clips_table()
    cache = tmp_path / TRANSCRIPTS_FILE

    class Crashing(FakeTranscriber):
        def transcribe(self, batch):
            if len(self.seen) >= 4:
                raise RuntimeError("out of memory")
            return super().transcribe(batch)

    with pytest.raises(RuntimeError):
        transcribe_clips(clips, Crashing, cache, batch_size=2, load_window=load_label)
    assert len(pd.read_csv(cache)) == 4


def test_changed_segment_is_retranscribed(tmp_path):
    clips = clips_table().head(3)
    cache = tmp_path / TRANSCRIPTS_FILE
    transcribe_clips(clips, FakeTranscriber, cache, batch_size=3, load_window=load_label)
    relabelled = clips.assign(seg_start=[np.nan, np.nan, 0.5], seg_end=[np.nan, np.nan, 1.1])
    again = FakeTranscriber()
    out = transcribe_clips(relabelled, lambda: again, cache, batch_size=3, load_window=load_label)
    assert again.seen == ["menu"]
    assert len(out) == 3 and out["segment"].iloc[2] == "0.500-1.100"


def test_empty_transcript_survives_the_cache(tmp_path):
    clips = clips_table()
    cache = tmp_path / TRANSCRIPTS_FILE
    transcribe_clips(clips, FakeTranscriber, cache, batch_size=8, load_window=load_label)
    out = transcribe_clips(clips, FakeTranscriber, cache, batch_size=8, load_window=load_label)
    menu = out[out.label == "menu"]["transcript"]
    assert (menu == "").all()
    for scorer in ("strict", "lenient"):
        preds = scored_run(out, "fake-asr", scorer).predictions
        assert (preds[preds.label == "menu"]["pred"] == OOV).all()


def test_wrong_number_of_transcripts_is_an_error(tmp_path):
    class Short(FakeTranscriber):
        def transcribe(self, batch):
            return super().transcribe(batch)[:-1]

    with pytest.raises(ValueError, match="transcripts"):
        transcribe_clips(clips_table(), Short, tmp_path / TRANSCRIPTS_FILE, batch_size=4,
                         load_window=load_label)


def test_clip_window_is_the_classifier_window(tmp_path):
    path = tmp_path / "clip.wav"
    write_wav(path, word_clip(seconds=3.0))
    row = pd.Series({"file_path": str(path), "seg_start": np.nan, "seg_end": np.nan})
    window = clip_window(row)
    assert window.dtype == np.float32 and len(window) == 32000


def test_unreadable_audio_names_the_file(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF\x00\x00not really a wav")
    row = pd.Series({"file_path": str(path), "seg_start": np.nan, "seg_end": np.nan})
    with pytest.raises(RuntimeError, match="broken.wav"):
        clip_window(row)


def test_reference_clip_is_the_first_dysarthric_array_clip():
    clips = clips_table().sample(frac=1, random_state=0)
    ref = reference_clip(clips)
    assert (ref.speaker_id, ref.utterance_id, ref.mic) == (DYSARTHRIC_SPEAKERS[0], "0000", ARRAY_MIC)


def test_cost_counts_the_model_and_round_trips(tmp_path):
    class Tiny:
        model = nn.Linear(10, 5)
        cost_note = "tiny"

        def transcribe(self, batch):
            self.model(torch.zeros(1, 10))
            return ["yes"]

    profile = transcription_cost(Tiny(), np.zeros(32000, dtype=np.float32))
    assert (profile.params, profile.macs, profile.input_seconds) == (55, 50, 2.0)
    save_cost(profile, tmp_path / COST_FILE)
    assert load_cost(tmp_path / COST_FILE) == profile
