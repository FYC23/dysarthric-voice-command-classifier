"""Shared test setup: make the repo root importable and provide a fake TORGO tree."""

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SR = 16000


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    """Write a float waveform in [-1, 1] as 16-bit mono PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def word_clip(seconds: float = 2.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Quiet noise with a 0.4 s tone burst starting at 0.8 s."""
    rng = np.random.default_rng(seed)
    audio = rng.normal(0, 1e-3, int(seconds * sr))
    t = np.arange(int(0.4 * sr)) / sr
    start = int(0.8 * sr)
    audio[start:start + len(t)] += 0.1 * np.sin(2 * np.pi * 300 * t)
    return audio


def add_utterance(root: Path, group: str, speaker: str, session: str, utt: str,
                  prompt: str, mics: dict) -> None:
    """Create prompts/<utt>.txt and one wav per mic. mics maps mic dir -> audio or raw bytes."""
    sess = root / group / speaker / session
    (sess / "prompts").mkdir(parents=True, exist_ok=True)
    (sess / "prompts" / f"{utt}.txt").write_text(prompt)
    for mic, content in mics.items():
        path = sess / mic / f"{utt}.wav"
        if isinstance(content, bytes):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        else:
            write_wav(path, content)


@pytest.fixture
def fake_torgo(tmp_path: Path) -> Path:
    """
    A tiny TORGO-shaped tree covering the labelling and mic edge cases:

    F01/Session1
      0001 "one"                       both mics          -> 2 rows
      0002 "One validated acts ..."    both mics          -> excluded (sentence)
      0003 "Yes\\n"                     arrayMic           -> 1 row, label "yes"
      0004 "xxx"                       arrayMic           -> excluded
      0005 "up"                        headMic only       -> 1 row
      0006 "down"                      arrayMic too short, headMic ok -> 1 row (head)
      0007 "left"                      arrayMic corrupt   -> excluded
      0008 "right"                     no wav             -> excluded
      0009 "two"                       headMic 12 s long (a TORGO head-mic
                                       "no" is 194 s) -> 1 row (array)
    FC01/Session1
      0001 "no"                        arrayMic           -> 1 row
    """
    root = tmp_path / "TORGO"
    clip = word_clip()
    both = {"wav_arrayMic": clip, "wav_headMic": clip}
    add_utterance(root, "F", "F01", "Session1", "0001", "one", both)
    add_utterance(root, "F", "F01", "Session1", "0002",
                  "One validated acts of school districts.", both)
    add_utterance(root, "F", "F01", "Session1", "0003", "Yes\n", {"wav_arrayMic": clip})
    add_utterance(root, "F", "F01", "Session1", "0004", "xxx", {"wav_arrayMic": clip})
    add_utterance(root, "F", "F01", "Session1", "0005", "up", {"wav_headMic": clip})
    add_utterance(root, "F", "F01", "Session1", "0006", "down",
                  {"wav_arrayMic": clip[:80], "wav_headMic": clip})
    add_utterance(root, "F", "F01", "Session1", "0007", "left",
                  {"wav_arrayMic": b"RIFF\x00\x00not really a wav"})
    add_utterance(root, "F", "F01", "Session1", "0008", "right", {})
    add_utterance(root, "F", "F01", "Session1", "0009", "two",
                  {"wav_arrayMic": clip, "wav_headMic": word_clip(seconds=12.0)})
    add_utterance(root, "FC", "FC01", "Session1", "0001", "no", {"wav_arrayMic": clip})
    # Non-session folders and hidden folders must be ignored
    (root / "F" / "F01" / "Notes").mkdir(parents=True)
    (root / ".F.partial").mkdir()
    return root
