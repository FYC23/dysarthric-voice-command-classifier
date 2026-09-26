"""Integration test: scan -> dataset item, with and without augmentation."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

from src.audio import prepare_waveform
from src.config import config
from src.data.dataset import TORGOCommandDataset, collate_fn
from src.data.noise import NoiseBank
from src.data.preprocessing import create_label_mapping, scan_torgo_dataset

WINDOW = config.MAX_AUDIO_SAMPLES


@pytest.fixture
def df(fake_torgo):
    frame = scan_torgo_dataset(fake_torgo, ["one", "yes", "no", "up", "down"],
                               ("wav_arrayMic", "wav_headMic"))
    label2id, _ = create_label_mapping(frame)
    return frame.assign(label_id=frame.label.map(label2id))


@pytest.fixture
def noise_bank():
    return NoiseBank([np.random.default_rng(0).normal(0, 0.05, 3 * config.SAMPLE_RATE)])


@pytest.mark.parametrize("augment", [False, True])
def test_items_have_fixed_length_and_label(df, noise_bank, augment):
    ds = TORGOCommandDataset(df, Wav2Vec2FeatureExtractor(), config,
                             max_length=WINDOW, augment=augment, noise=noise_bank)
    for i in range(len(ds)):
        item = ds[i]
        assert item["input_values"].shape == (WINDOW,)
        assert int(item["label"]) == df.iloc[i].label_id


def test_labelled_segment_rows_are_cropped_to_the_segment(df):
    labelled = df.assign(seg_start=np.nan, seg_end=np.nan)
    labelled.loc[0, ["seg_start", "seg_end"]] = [0.8, 1.2]  # the word in word_clip()
    ds = TORGOCommandDataset(labelled, Wav2Vec2FeatureExtractor(do_normalize=False), config,
                             max_length=WINDOW, augment=False)
    values = ds[0]["input_values"].numpy()
    assert values.shape == (WINDOW,)
    nonzero = np.flatnonzero(values)
    assert abs((nonzero[-1] - nonzero[0]) / config.SAMPLE_RATE - (0.4 + 2 * 0.15)) < 0.02


def test_window_is_two_seconds():
    assert config.MAX_AUDIO_SAMPLES == int(2.0 * config.SAMPLE_RATE)


def test_eval_items_are_the_prepared_waveform_unchanged(df):
    ds = TORGOCommandDataset(df, None, config, max_length=WINDOW, augment=False)
    for i in range(len(ds)):
        audio = ds.load_audio(df.iloc[i].file_path)
        expected = prepare_waveform(audio, config.SAMPLE_RATE, WINDOW)
        np.testing.assert_array_equal(ds[i]["input_values"].numpy(), expected)
        np.testing.assert_array_equal(ds[i]["input_values"].numpy(), expected)  # and every time


def test_augmented_items_are_a_new_view_each_time(df, noise_bank):
    ds = TORGOCommandDataset(df, None, config, max_length=WINDOW, augment=True, noise=noise_bank)
    first, second = ds[0]["input_values"], ds[0]["input_values"]
    assert first.shape == (WINDOW,) and first.dtype == torch.float32
    assert not torch.equal(first, second)


def test_augment_without_noise_bank_fails_at_construction(df):
    with pytest.raises(ValueError, match="NoiseBank"):
        TORGOCommandDataset(df, None, config, max_length=WINDOW, augment=True)


def test_augmentation_is_reproducible_from_the_torch_seed(df, noise_bank):
    def first_view():
        torch.manual_seed(123)
        ds = TORGOCommandDataset(df, None, config, max_length=WINDOW, augment=True,
                                 noise=noise_bank)
        return ds[0]["input_values"]
    assert torch.equal(first_view(), first_view())


def test_dataloader_workers_draw_different_augmentations(df, noise_bank):
    one_clip = df.iloc[[0, 0, 0, 0]]
    ds = TORGOCommandDataset(one_clip, None, config, max_length=WINDOW, augment=True,
                             noise=noise_bank)
    ds[0]  # the main process creates its generator first; workers must not reuse it
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=collate_fn)
    views = [batch["input_values"][0] for batch in loader]
    assert not torch.equal(views[0], views[1])  # each worker's first draw
