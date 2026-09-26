"""
TORGO dataset scanning and preprocessing utilities.
"""

import wave
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd

MIC_TYPES = ("wav_arrayMic", "wav_headMic")
MIN_DURATION_S = 0.1   # M05 Session2 has ~10 ms head-mic stubs
MAX_DURATION_S = 10.0  # single-word takes are <= 8.4 s; one F03 head-mic "no" is 194 s
COLUMNS = ['file_path', 'speaker_id', 'session', 'utterance_id', 'mic',
           'label', 'gender', 'is_dysarthric']


def parse_speaker_info(speaker_id: str) -> Tuple[str, bool]:
    """
    Parse speaker ID to extract gender and dysarthria status.
    
    Speaker ID format:
    - F01, F03, F04: Female dysarthric
    - FC01, FC02, FC03: Female control (no dysarthria)
    - M01-M05: Male dysarthric
    - MC01-MC04: Male control (no dysarthria)
    """
    is_dysarthric = 'C' not in speaker_id
    gender = 'F' if speaker_id.startswith('F') else 'M'
    return gender, is_dysarthric


def _iter_sessions(torgo_root: Path) -> Iterator[Tuple[str, Path]]:
    """Yield (speaker_id, session_dir) for <group>/<speaker>/Session*/ folders."""
    for group_dir in sorted(p for p in torgo_root.iterdir() if p.is_dir() and not p.name.startswith('.')):
        for speaker_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(speaker_dir.glob("Session*")):
                if (session_dir / "prompts").is_dir():
                    yield speaker_dir.name, session_dir


def _read_prompt(prompt_file: Path) -> Optional[str]:
    """Prompt text, lowercased and stripped; None if the file can't be read."""
    try:
        return prompt_file.read_text(errors="replace").strip().lower()
    except OSError as e:
        print(f"Warning: could not read prompt {prompt_file}: {e}")
        return None


def _wav_duration(wav_file: Path) -> Optional[float]:
    """Duration in seconds from the WAV header; None if the file is not a readable WAV."""
    try:
        with wave.open(str(wav_file)) as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, OSError):
        return None


def _duration_problem(duration: Optional[float], min_s: float, max_s: float) -> Optional[str]:
    """Reason to skip a file based on its duration, or None if it is usable."""
    if duration is None:
        return "unreadable"
    if duration < min_s:
        return "too_short"
    if duration > max_s:
        return "too_long"
    return None


def scan_torgo_dataset(
    torgo_root: Path,
    target_commands: List[str],
    mic_types: Union[str, Sequence[str]] = MIC_TYPES,
    min_duration_s: float = MIN_DURATION_S,
    max_duration_s: float = MAX_DURATION_S,
) -> pd.DataFrame:
    """
    Scan TORGO for single-word prompts that exactly match a target command.

    A prompt is kept only if its whole text (lowercased, stripped) is a target
    word, so sentences such as "Two other cases also were under advisement."
    are not labelled "two". The label is the prompt shown to the speaker; TORGO
    has no check of what was actually said.

    Each utterance yields one row per available mic (array and head mic record
    the same take), so the `mic` column must be kept in mind when grouping.
    Unreadable WAVs and files outside [min_duration_s, max_duration_s] are skipped.

    Returns DataFrame with columns: file_path, speaker_id, session, utterance_id,
                                    mic, label, gender, is_dysarthric
    """
    mics = (mic_types,) if isinstance(mic_types, str) else tuple(mic_types)
    targets = {cmd.lower() for cmd in target_commands}
    samples, skipped = [], Counter()

    for speaker_id, session_dir in _iter_sessions(torgo_root):
        gender, is_dysarthric = parse_speaker_info(speaker_id)
        for prompt_file in sorted((session_dir / "prompts").glob("*.txt")):
            label = _read_prompt(prompt_file)
            if label not in targets:
                continue
            for mic in mics:
                wav_file = session_dir / mic / f"{prompt_file.stem}.wav"
                if not wav_file.exists():
                    continue
                duration = _wav_duration(wav_file)
                problem = _duration_problem(duration, min_duration_s, max_duration_s)
                if problem:
                    skipped[problem] += 1
                    continue
                samples.append({
                    'file_path': str(wav_file),
                    'speaker_id': speaker_id,
                    'session': session_dir.name,
                    'utterance_id': prompt_file.stem,
                    'mic': mic,
                    'label': label,
                    'gender': gender,
                    'is_dysarthric': is_dysarthric,
                })

    if skipped:
        print(f"Skipped {sum(skipped.values())} target-word WAVs: {dict(skipped)}")
    return pd.DataFrame(samples, columns=COLUMNS)


def create_speaker_splits(df: pd.DataFrame, val_speakers: List[str], test_speakers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by speakers for proper evaluation.
    """
    train_df = df[~df['speaker_id'].isin(val_speakers + test_speakers)].copy()
    val_df = df[df['speaker_id'].isin(val_speakers)].copy()
    test_df = df[df['speaker_id'].isin(test_speakers)].copy()
    
    return train_df, val_df, test_df


def create_label_mapping(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label to ID mappings from dataset.
    """
    labels_in_dataset = sorted(df['label'].unique())
    label2id = {label: idx for idx, label in enumerate(labels_in_dataset)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label