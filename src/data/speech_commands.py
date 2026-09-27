"""
Speech Commands v0.02 for BC-ResNet pretraining (stage 1): the 36 classes
(35 words + _silence_), the extracted-word cache, and silence windows.

Waveforms follow the TORGO path so every stage sees the same kind of input:
extract_word (DC removal + VAD trim), then a random place in the 2 s window
for training (speech_commands_augment.py) or the centre for evaluation.
Extracted words are deterministic, so they are cached once as int16 .npy
files; augmentation stays on the fly.
"""

import dataclasses
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..audio import DEFAULT_VAD, extract_word, fit_to_length, remove_dc
from ..eval.constants import SPEECH_COMMANDS_V2_WORDS
from .noise import NoiseBank
from .speech_commands_augment import (
    DEFAULT_BCRESNET_AUG, BCResNetAugParams, augment_word_in_window, make_silence,
)
from .worker_rng import WorkerRng

SILENCE = "_silence_"
CLASSES: Tuple[str, ...] = tuple(SPEECH_COMMANDS_V2_WORDS) + (SILENCE,)
SILENCE_ID = CLASSES.index(SILENCE)
# Bump when the cache layout or extract_word's code changes. The VAD settings
# (DEFAULT_VAD) are stored in meta.json and checked automatically.
CACHE_VERSION = 1
DOWNLOAD_HINT = "run scripts/download_speech_commands.sh"
_INT16_SCALE = 32767
_META = "meta.json"  # written last: its presence marks a complete cache
_INSTALL_ATTEMPTS = 5  # renames retried when concurrent runs race to install


def read_wav(path: Path, sr: int) -> np.ndarray:
    """Mono float32 audio; the file must already be at `sr`."""
    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if file_sr != sr:
        raise ValueError(f"{path}: sample rate {file_sr}, expected {sr}")
    return audio[:, 0]


def scan_split(root: Path, split: str) -> pd.DataFrame:
    """One row per word clip (file_path, label, label_id), in class then path order."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"{split_dir} not found; {DOWNLOAD_HINT}")
    words = tuple(sorted(d.name for d in split_dir.iterdir()
                         if d.is_dir() and not d.name.startswith("_")))
    if words != tuple(SPEECH_COMMANDS_V2_WORDS):
        missing = sorted(set(SPEECH_COMMANDS_V2_WORDS) - set(words))
        extra = sorted(set(words) - set(SPEECH_COMMANDS_V2_WORDS))
        raise ValueError(f"{split_dir}: word folders do not match Speech Commands v0.02 "
                         f"(missing {missing}, unexpected {extra}); {DOWNLOAD_HINT}")
    rows = [{"file_path": str(path), "label": word, "label_id": label_id}
            for label_id, word in enumerate(SPEECH_COMMANDS_V2_WORDS)
            for path in sorted((split_dir / word).glob("*.wav"))]
    return pd.DataFrame(rows, columns=["file_path", "label", "label_id"])


def word_cache_dir(cache_dir: Path, split: str) -> Path:
    return Path(cache_dir) / "speech_commands_words" / split


def _relative_files(clips: pd.DataFrame, root: Path) -> List[str]:
    return [str(Path(p).relative_to(root)) for p in clips["file_path"]]


def _to_int16(audio: np.ndarray) -> np.ndarray:
    return np.round(np.clip(audio, -1.0, 1.0) * _INT16_SCALE).astype(np.int16)


def _cache_matches(directory: Path, clips: pd.DataFrame, root: Path, sr: int) -> bool:
    """True if `directory` holds a complete cache of exactly these clips and settings."""
    try:
        meta = json.loads((directory / _META).read_text())
        if (not isinstance(meta, dict) or meta.get("version") != CACHE_VERSION
                or meta.get("sample_rate") != sr
                or meta.get("vad") != dataclasses.asdict(DEFAULT_VAD)):
            return False
        return np.load(directory / "files.npy").tolist() == _relative_files(clips, root)
    except (OSError, ValueError):  # missing, or torn by an interrupted write
        return False


def _write_word_cache(clips: pd.DataFrame, root: Path, target: Path, sr: int,
                      split: str) -> None:
    """Every cache file, written into the empty `target`; meta.json last."""
    words = [_to_int16(extract_word(read_wav(p, sr), sr))
             for p in tqdm(clips["file_path"], desc=f"Caching {split} words")]
    offsets = np.concatenate([[0], np.cumsum([len(w) for w in words])]).astype(np.int64)
    audio = np.concatenate(words) if words else np.zeros(0, np.int16)
    np.save(target / "audio.npy", audio)
    np.save(target / "offsets.npy", offsets)
    np.save(target / "labels.npy", clips["label_id"].to_numpy(np.int64))
    np.save(target / "files.npy", np.array(_relative_files(clips, root)))
    (target / _META).write_text(json.dumps(
        {"version": CACHE_VERSION, "sample_rate": sr, "vad": dataclasses.asdict(DEFAULT_VAD),
         "count": len(clips)}))


def _install(built: Path, directory: Path, clips: pd.DataFrame, root: Path, sr: int) -> None:
    """
    Put the finished `built` directory at `directory` by renames only. An old
    cache is moved aside and deleted rather than overwritten, so a run that has
    it memory-mapped keeps reading its files (they live on until unmapped).
    If another run installed a matching cache meanwhile, that one is kept.
    """
    for _ in range(_INSTALL_ATTEMPTS):
        if _cache_matches(directory, clips, root, sr):
            return
        stale = directory.with_name(f".{directory.name}.stale-{uuid.uuid4().hex}")
        try:
            directory.replace(stale)
        except FileNotFoundError:
            pass
        try:
            built.replace(directory)
            return
        except OSError:  # another run installed its cache between the two renames
            continue
        finally:
            shutil.rmtree(stale, ignore_errors=True)
    raise RuntimeError(f"could not install the word cache at {directory}: "
                       "other runs kept replacing it")


def _build_word_cache(clips: pd.DataFrame, root: Path, directory: Path, sr: int) -> None:
    """
    Build in a private sibling directory, then swap it in, so runs sharing the
    cache never see a half-written one. The partial build is removed on failure.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{directory.name}.partial-", dir=directory.parent))
    try:
        _write_word_cache(clips, root, partial, sr, directory.name)
        _install(partial, directory, clips, root, sr)
    finally:
        shutil.rmtree(partial, ignore_errors=True)  # gone already once installed


def ensure_word_cache(root: Path, split: str, cache_dir: Path, sr: int) -> Path:
    """The split's word-cache directory, built now if missing, stale or incomplete."""
    clips = scan_split(root, split)
    directory = word_cache_dir(cache_dir, split)
    if not _cache_matches(directory, clips, Path(root), sr):
        print(f"Building the {split} word cache in {directory} ({len(clips)} clips)")
        _build_word_cache(clips, Path(root), directory, sr)
    return directory


class WordCache:
    """
    The extracted words of one split. The audio is memory-mapped on first use
    and left out of pickles, so DataLoader workers share it instead of copying
    ~2 GB each (macOS starts workers by pickling the dataset).
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.offsets = np.load(self.directory / "offsets.npy")
        self.labels = np.load(self.directory / "labels.npy")
        self._audio = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_audio"] = None
        return state

    def word(self, index: int) -> np.ndarray:
        """Word `index` as a new float32 array."""
        if self._audio is None:
            self._audio = np.load(self.directory / "audio.npy", mmap_mode="r")
        start, end = self.offsets[index], self.offsets[index + 1]
        return self._audio[start:end].astype(np.float32) / _INT16_SCALE


def silence_count(n_words: int) -> int:
    """Silence examples per epoch: as many as the average word class has."""
    return int(round(n_words / len(SPEECH_COMMANDS_V2_WORDS)))


def silence_eval_windows(root: Path, split: str, length: int, sr: int) -> List[np.ndarray]:
    """
    Evaluation _silence_ items that fill the window: long recordings (the
    validation running_tap) are cut into consecutive windows; short clips (test)
    are repeated to the window length. No VAD: it would trim noise to bursts.
    """
    folder = Path(root) / split / SILENCE
    paths = sorted(folder.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no {SILENCE} clips in {folder}; {DOWNLOAD_HINT}")
    windows = []
    for path in paths:
        audio = remove_dc(read_wav(path, sr)).astype(np.float32)
        if len(audio) >= length:
            windows += [audio[s:s + length].copy()
                        for s in range(0, len(audio) - length + 1, length)]
        else:
            windows.append(np.resize(audio, length).astype(np.float32))
    return windows


class SpeechCommandsTrainSet(Dataset):
    """
    Every word clip as a fresh random 2 s view, then `n_silence` _silence_
    examples drawn fresh on every access (noise over the whole window).
    """

    def __init__(self, words: WordCache, noise: NoiseBank, length: int, n_silence: int,
                 params: BCResNetAugParams = DEFAULT_BCRESNET_AUG):
        if n_silence < 0:
            raise ValueError(f"n_silence must be >= 0, got {n_silence}")
        self.words = words
        self.noise = noise
        self.length = length
        self.n_silence = n_silence
        self.params = params
        self._rng = WorkerRng()

    def __len__(self) -> int:
        return len(self.words) + self.n_silence

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        rng = self._rng.get()
        if index < len(self.words):
            audio = augment_word_in_window(self.words.word(index), self.length, rng,
                                           self.noise, self.params)
            return torch.from_numpy(audio), int(self.words.labels[index])
        return torch.from_numpy(make_silence(self.length, rng, self.noise, self.params)), SILENCE_ID


class SpeechCommandsEvalSet(Dataset):
    """Word clips centred in the window (prepare_waveform), then fixed silence windows."""

    def __init__(self, words: WordCache, silence: Sequence[np.ndarray], length: int):
        wrong = [len(w) for w in silence if len(w) != length]
        if wrong:
            raise ValueError(f"silence windows must have length {length}, got {wrong[:3]}")
        self.words = words
        self.silence = tuple(np.asarray(w, dtype=np.float32) for w in silence)
        self.length = length

    def __len__(self) -> int:
        return len(self.words) + len(self.silence)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        if index < len(self.words):
            audio = fit_to_length(self.words.word(index), self.length).astype(np.float32)
            return torch.from_numpy(audio), int(self.words.labels[index])
        return torch.from_numpy(self.silence[index - len(self.words)].copy()), SILENCE_ID
