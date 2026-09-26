"""Speech Commands for BC-ResNet pretraining: classes, scanning, word cache, silence."""

import dataclasses
import json
import pickle
import shutil

import numpy as np
import pytest
import torch

from src.audio import DEFAULT_VAD, extract_word, prepare_waveform
from src.data.noise import NoiseBank
from src.data.speech_commands import (
    CLASSES, SILENCE, SILENCE_ID, SpeechCommandsEvalSet, SpeechCommandsTrainSet, WordCache,
    ensure_word_cache, read_wav, scan_split, silence_count, silence_eval_windows,
)
from src.data.worker_rng import WorkerRng
from src.eval.constants import SPEECH_COMMANDS_V2_WORDS

SR = 16000
WINDOW = 32000


def test_classes_are_the_35_words_then_silence():
    assert len(CLASSES) == 36
    assert CLASSES[:35] == tuple(sorted(SPEECH_COMMANDS_V2_WORDS))
    assert SILENCE_ID == 35 and CLASSES[SILENCE_ID] == SILENCE == "_silence_"


def test_scan_lists_every_word_clip_with_its_class(fake_speech_commands):
    df = scan_split(fake_speech_commands, "train")
    assert len(df) == 70
    assert set(df.label) == set(SPEECH_COMMANDS_V2_WORDS)
    assert (df.label.map(CLASSES.index) == df.label_id).all()
    assert not df.file_path.str.contains(SILENCE).any()


def test_missing_split_says_how_to_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_speech_commands"):
        scan_split(tmp_path, "train")


def test_missing_word_folder_is_named(fake_speech_commands):
    shutil.rmtree(fake_speech_commands / "test" / "visual")
    with pytest.raises(ValueError, match="visual"):
        scan_split(fake_speech_commands, "test")


def test_read_wav_rejects_another_sample_rate(fake_speech_commands):
    path = next((fake_speech_commands / "train" / "yes").glob("*.wav"))
    with pytest.raises(ValueError, match="8000"):
        read_wav(path, 8000)


def test_cache_round_trips_the_extracted_words(fake_speech_commands, tmp_path):
    cache = WordCache(ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR))
    df = scan_split(fake_speech_commands, "train")
    assert len(cache) == len(df)
    assert (cache.labels == df.label_id.to_numpy()).all()
    for i in (0, 37, 69):
        expected = extract_word(read_wav(df.file_path[i], SR), SR)
        np.testing.assert_allclose(cache.word(i), expected, atol=1 / 32767)


def test_cache_is_reused_when_nothing_changed(fake_speech_commands, tmp_path):
    directory = ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    stamp = (directory / "audio.npy").stat().st_mtime_ns
    ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert (directory / "audio.npy").stat().st_mtime_ns == stamp


def test_cache_rebuilds_when_clips_change(fake_speech_commands, tmp_path):
    directory = ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    before = len(WordCache(directory))
    next((fake_speech_commands / "train" / "yes").glob("*.wav")).unlink()
    ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert len(WordCache(directory)) == before - 1


def test_interrupted_build_is_not_trusted(fake_speech_commands, tmp_path):
    directory = ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    (directory / "meta.json").unlink()                 # the marker is written last
    (directory / "labels.npy").write_bytes(b"garbage")  # a half-written file
    ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert len(WordCache(directory)) == 70


def test_cache_rebuilds_when_the_vad_settings_change(fake_speech_commands, tmp_path):
    directory = ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["vad"]["pad_s"] = 0.5  # as if the cache were built with other VAD settings
    meta_path.write_text(json.dumps(meta))
    ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert json.loads(meta_path.read_text())["vad"] == dataclasses.asdict(DEFAULT_VAD)


def test_unreadable_marker_or_file_list_means_rebuild(fake_speech_commands, tmp_path):
    directory = ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    (directory / "meta.json").write_text("{")  # a torn write
    ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert json.loads((directory / "meta.json").read_text())["count"] == 70
    (directory / "files.npy").unlink()
    ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert len(np.load(directory / "files.npy")) == 70


def _leftovers(directory):
    return sorted(p.name for p in directory.parent.iterdir() if p.name != directory.name)


def test_rebuild_leaves_a_mapped_cache_untouched(fake_speech_commands, tmp_path):
    directory = ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    old = WordCache(directory)
    first = old.word(0).copy()  # maps the audio
    next((fake_speech_commands / "train" / "backward").glob("*.wav")).unlink()
    ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert not np.array_equal(WordCache(directory).word(0), first)  # the new cache differs
    np.testing.assert_array_equal(old.word(0), first)               # the old map does not
    assert _leftovers(directory) == []


def test_failed_build_keeps_the_old_cache_and_no_partial(fake_speech_commands, tmp_path,
                                                         monkeypatch):
    directory = ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    next((fake_speech_commands / "train" / "yes").glob("*.wav")).unlink()

    def broken(*args, **kwargs):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr("src.data.speech_commands.extract_word", broken)
    with pytest.raises(RuntimeError, match="extraction failed"):
        ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR)
    assert (directory / "meta.json").exists()  # the old cache is still complete
    assert len(WordCache(directory)) == 70
    assert _leftovers(directory) == []


def test_pickled_cache_carries_no_audio(fake_speech_commands, tmp_path):
    cache = WordCache(ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR))
    cache.word(0)  # maps the audio
    clone = pickle.loads(pickle.dumps(cache))
    assert clone._audio is None
    np.testing.assert_array_equal(clone.word(3), cache.word(3))


def test_silence_count_is_the_average_word_class():
    assert silence_count(84848) == 2424
    assert silence_count(70) == 2


def test_silence_eval_windows_fill_the_window(fake_speech_commands):
    val = silence_eval_windows(fake_speech_commands, "validation", WINDOW, SR)
    test = silence_eval_windows(fake_speech_commands, "test", WINDOW, SR)
    assert len(val) == 2   # 5 s of running_tap -> two whole 2 s windows
    assert len(test) == 2  # each 1 s clip repeated to 2 s
    assert all(len(w) == WINDOW and w.dtype == np.float32 for w in val + test)
    assert all(np.count_nonzero(w) > 0.99 * WINDOW for w in test)


def test_silence_eval_windows_need_the_silence_folder(fake_speech_commands):
    shutil.rmtree(fake_speech_commands / "test" / SILENCE)
    with pytest.raises(FileNotFoundError, match="download_speech_commands"):
        silence_eval_windows(fake_speech_commands, "test", WINDOW, SR)


@pytest.fixture
def train_words(fake_speech_commands, tmp_path):
    return WordCache(ensure_word_cache(fake_speech_commands, "train", tmp_path / "cache", SR))


@pytest.fixture
def noise():
    return NoiseBank([np.random.default_rng(0).normal(0, 0.1, 48000).astype(np.float32)])


def test_train_set_is_every_word_then_fresh_silence(train_words, noise):
    ds = SpeechCommandsTrainSet(train_words, noise, WINDOW, n_silence=2)
    assert len(ds) == 72
    word, label = ds[0]
    assert label == int(train_words.labels[0])
    assert word.shape == (WINDOW,) and word.dtype == torch.float32
    silence, label = ds[71]
    assert label == SILENCE_ID and silence.shape == (WINDOW,)
    assert torch.count_nonzero(silence) > 0.99 * WINDOW
    again, _ = ds[71]
    assert not torch.equal(silence, again)  # a new draw on every access


def test_train_set_rejects_negative_silence(train_words, noise):
    with pytest.raises(ValueError, match="silence"):
        SpeechCommandsTrainSet(train_words, noise, WINDOW, n_silence=-1)


def test_eval_items_are_the_prepared_waveform(fake_speech_commands, tmp_path):
    words = WordCache(ensure_word_cache(fake_speech_commands, "validation", tmp_path / "c", SR))
    windows = silence_eval_windows(fake_speech_commands, "validation", WINDOW, SR)
    ds = SpeechCommandsEvalSet(words, windows, WINDOW)
    assert len(ds) == len(words) + len(windows)
    df = scan_split(fake_speech_commands, "validation")
    for i in (0, 50):
        audio, label = ds[i]
        expected = prepare_waveform(read_wav(df.file_path[i], SR), SR, WINDOW)
        np.testing.assert_allclose(audio.numpy(), expected, atol=1 / 32767)
        assert label == df.label_id[i]
    audio, label = ds[len(words)]
    assert label == SILENCE_ID
    np.testing.assert_array_equal(audio.numpy(), windows[0])
    assert torch.equal(ds[0][0], ds[0][0])  # deterministic


def test_eval_set_rejects_silence_of_the_wrong_length(train_words):
    with pytest.raises(ValueError, match="length"):
        SpeechCommandsEvalSet(train_words, [np.zeros(100, np.float32)], WINDOW)


def test_worker_rng_reseeds_only_when_the_process_seed_changes():
    rng = WorkerRng()
    torch.manual_seed(1)
    first = rng.get()
    torch.manual_seed(1)
    assert rng.get() is first
    torch.manual_seed(2)
    assert rng.get() is not first
