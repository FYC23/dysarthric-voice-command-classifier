"""
Inference module for voice command prediction.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor

from ..model.architecture import HuBERTForCommandClassification


class VoiceCommandPredictor:
    """
    Voice command predictor for inference.
    
    Loads a trained model and provides an easy interface for prediction.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        label_mapping_path: str,
        model_cache_dir: str,
        device: Optional[str] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (.pt file)
            label_mapping_path: Path to label_mapping.json
            model_cache_dir: Path to cached HuBERT model
            device: 'cuda', 'cpu', or None for auto-detect
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
        self.label2id = label_mapping['label2id']
        self.id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
        self.num_labels = len(self.label2id)
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_cache_dir)
        
        # Initialize and load model
        self.model = HuBERTForCommandClassification(
            model_path=model_cache_dir,
            num_labels=self.num_labels,
            hidden_size=1024,
            classifier_dropout=0.1,
            freeze_encoder=True,
            freeze_feature_extractor=True
        )
        
        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Audio settings
        self.sample_rate = 16000
        self.max_length = 48000  # 3 seconds
        
        print(f"Model loaded on {self.device}")
        print(f"Number of classes: {self.num_labels}")
    
    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio for model input.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the input audio
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Resample if necessary
        if sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0) if audio.shape[0] == 2 else audio.mean(axis=1)
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Pad or truncate to fixed length
        if len(audio) > self.max_length:
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False
        )
        
        return inputs.input_values
    
    def predict(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Predict the voice command from audio.
        
        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate of the input audio
            
        Returns:
            Dictionary with:
                - 'label': Predicted command string
                - 'confidence': Confidence score (0-1)
                - 'probabilities': Dict of all labels with their probabilities
        """
        # Preprocess
        input_values = self.preprocess_audio(audio, sample_rate)
        input_values = input_values.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
        
        # Get prediction
        pred_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_idx].item()
        
        # Get all probabilities
        all_probs = {self.id2label[i]: probs[0, i].item() for i in range(self.num_labels)}
        
        # Sort by probability
        sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'label': self.id2label[pred_idx],
            'confidence': confidence,
            'probabilities': sorted_probs
        }
    
    def predict_file(self, audio_path: str) -> Dict:
        """
        Predict from an audio file path.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Prediction dictionary
        """
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        return self.predict(audio, sr)