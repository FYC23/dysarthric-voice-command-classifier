"""
PyTorch Dataset for TORGO dysarthric voice commands.

Evaluation items (augment=False) are prepare_waveform output, unchanged.
Training items (augment=True) are a fresh random view of the word on every
access (src/data/torgo_augment.py). Nothing augmented is written to disk.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
from transformers import Wav2Vec2FeatureExtractor

from ..audio import extract_word, prepare_waveform
from .noise import NoiseBank
from .torgo_augment import DEFAULT_TORGO_AUG, TorgoAugParams, augment_word
from .worker_rng import WorkerRng


class TORGOCommandDataset(Dataset):
    """
    PyTorch Dataset for TORGO dysarthric voice commands.

    feature_extractor: HuBERT's Wav2Vec2FeatureExtractor, or None to return the
        raw float32 window (BC-ResNet computes log-Mel on the batch).
    noise: background-noise bank; required when augment=True and the recipe
        adds noise.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: Optional[Wav2Vec2FeatureExtractor],
        config,
        max_length: int = 24000,
        target_sr: int = 16000,
        augment: bool = False,
        noise: Optional[NoiseBank] = None,
        aug_params: TorgoAugParams = DEFAULT_TORGO_AUG,
    ):
        if augment and aug_params.noise_prob > 0 and noise is None:
            raise ValueError("augment=True needs a NoiseBank, e.g. "
                             "NoiseBank.from_dir(config.NOISE_DIR, config.SAMPLE_RATE)")
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.config = config
        self.max_length = max_length
        self.target_sr = target_sr
        self.augment = augment
        self.noise = noise
        self.aug_params = aug_params
        self._worker_rng = WorkerRng()

    def __len__(self) -> int:
        return len(self.df)

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample to target sample rate."""
        try:
            audio, _ = librosa.load(file_path, sr=self.target_sr, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros(self.max_length, dtype=np.float32)

    @staticmethod
    def segment_of(row: pd.Series) -> Optional[Tuple[float, float]]:
        """Hand-labelled word location (see src/data/segments.py), if the row has one."""
        start, end = row.get('seg_start'), row.get('seg_end')
        if start is None or end is None or pd.isna(start) or pd.isna(end):
            return None
        return float(start), float(end)

    def _get_rng(self) -> np.random.Generator:
        """
        One generator per process, seeded from torch. DataLoader workers each
        have their own torch seed (and a new one every epoch), so a generator
        copied in from the main process is replaced, not reused. With
        num_workers=0, set_seed() makes the augmentation reproducible.
        """
        return self._worker_rng.get()

    def _featurize(self, audio: np.ndarray) -> torch.Tensor:
        if self.feature_extractor is None:
            return torch.from_numpy(audio)
        inputs = self.feature_extractor(
            audio, sampling_rate=self.target_sr, return_tensors="pt", padding=False
        )
        return inputs.input_values.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio = self.load_audio(row['file_path'])
        segment = self.segment_of(row)
        if self.augment:
            word = extract_word(audio, self.target_sr, segment=segment)
            audio = augment_word(word, self.max_length, self._get_rng(), self.noise,
                                 self.aug_params)
        else:
            audio = prepare_waveform(audio, self.target_sr, self.max_length, segment=segment)

        return {
            'input_values': self._featurize(audio),
            'label': torch.tensor(row['label_id'], dtype=torch.long),
            'speaker_id': row['speaker_id'],
            'file_path': row['file_path']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    input_values = torch.stack([item['input_values'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'input_values': input_values,
        'labels': labels,
        'speaker_ids': [item['speaker_id'] for item in batch],
        'file_paths': [item['file_path'] for item in batch]
    }
