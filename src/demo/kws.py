"""
The demo's keyword spotter: a BC-ResNet fine-tuning checkpoint in,
probabilities over the 20 commands out, through the same log-Mel front end
as training (src/model/frontend.py).
"""

import time
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch

from src.config import Config
from src.eval.constants import COMMANDS
from src.model.bcresnet import BCResNet
from src.model.frontend import LogMel

# The class order checkpoints are trained with: src/training/loso.py::TORGO_CLASSES.
# Not imported from there: loso.py pulls in pandas and the training data code.
CHECKPOINT_CLASSES = tuple(sorted(COMMANDS))
REQUIRED_KEYS = ("model_state_dict", "tau", "classes")


class CheckpointError(ValueError):
    """The file is not a 20-command BC-ResNet fine-tuning checkpoint."""


class KeywordSpotter:
    def __init__(self, model: BCResNet, tau: float, device: torch.device):
        self.model = model.to(device).eval()
        self.logmel = LogMel().to(device).eval()
        self.tau = tau
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path],
                        device: torch.device = torch.device("cpu")) -> "KeywordSpotter":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {path} not found")
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as err:  # pickle, zip and weights_only errors all mean "not a checkpoint"
            raise CheckpointError(f"{path} could not be read as a PyTorch checkpoint: {err}") from err
        if not isinstance(checkpoint, dict) or any(k not in checkpoint for k in REQUIRED_KEYS):
            raise CheckpointError(f"{path} is not a BC-ResNet fine-tuning checkpoint "
                                  f"(needs keys {list(REQUIRED_KEYS)})")
        classes = tuple(checkpoint["classes"])
        if classes != CHECKPOINT_CLASSES:
            raise CheckpointError(f"{path} predicts {len(classes)} classes {list(classes)}, "
                                  f"not the 20 commands in training order")
        tau = float(checkpoint["tau"])
        model = BCResNet(tau, len(classes))
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as err:
            raise CheckpointError(f"{path}: weights do not fit BC-ResNet tau={tau:g}: {err}") from err
        return cls(model, tau, device)

    def predict(self, window: np.ndarray) -> Tuple[Dict[str, float], float]:
        """(command -> probability, milliseconds for front end + model) for one waveform."""
        window = np.array(window, dtype=np.float32)
        if window.ndim != 1:
            raise ValueError(f"expected a 1-D waveform, got shape {window.shape}")
        start = time.perf_counter()
        with torch.no_grad():
            x = torch.from_numpy(window).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(self.logmel(x)), dim=-1)[0].cpu().numpy()
        latency_ms = 1000.0 * (time.perf_counter() - start)
        return {c: float(p) for c, p in zip(CHECKPOINT_CLASSES, probs)}, latency_ms

    def warm_up(self) -> None:
        """One prediction on silence, so the first visitor's latency excludes lazy setup."""
        self.predict(np.zeros(Config.MAX_AUDIO_SAMPLES, dtype=np.float32))
