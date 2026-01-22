"""
PyTorch Dataset for TORGO dysarthric voice commands.
"""

import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
from transformers import Wav2Vec2FeatureExtractor

from .augmentation import create_augmentation_pipeline, apply_spec_augment


class TORGOCommandDataset(Dataset):
    """
    PyTorch Dataset for TORGO dysarthric voice commands with advanced augmentation.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: Wav2Vec2FeatureExtractor,
        config,
        max_length: int = 48000,
        target_sr: int = 16000,
        augment: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.config = config
        self.max_length = max_length
        self.target_sr = target_sr
        self.augment = augment
        
        if augment:
            self.augmentation_pipeline = create_augmentation_pipeline(config)
        else:
            self.augmentation_pipeline = None
        
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
    
    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is exactly max_length samples."""
        if len(audio) > self.max_length:
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio
    
    def apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply comprehensive data augmentation pipeline."""
        if not self.augment or self.augmentation_pipeline is None:
            return audio
        
        audio = audio.astype(np.float32)
        audio = self.augmentation_pipeline(samples=audio, sample_rate=self.target_sr)
        
        if random.random() < 0.5:
            audio = apply_spec_augment(audio, self.config)
        
        return audio
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        audio = self.load_audio(row['file_path'])
        audio = self.pad_or_truncate(audio)
        audio = self.apply_augmentation(audio)
        audio = self.pad_or_truncate(audio)
        
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=False
        )
        
        input_values = inputs.input_values.squeeze(0)
        
        return {
            'input_values': input_values,
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