"""
Hand-labelled segments for TORGO clips that don't fit the model window.

Dysarthric clips often contain struggle sounds, breaths and partial attempts
before (or after) the word. For clips still longer than the window after
silence trimming, a human marks where the word is (see the segment labeler,
which exports data/labels/torgo_manual_segment_labels.json). This module turns those labels
into training rows:

- each "word" segment becomes its own sample (two complete attempts -> two rows)
- "non_speech" segments are collected for a future "unknown" class
- "partial" segments are ignored for now
- "no_word" clips and over-window clips nobody has labelled yet are dropped,
  never cropped blindly (a blind crop keeps the loudest part, often the struggle)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import soundfile as sf

from ..audio import DEFAULT_VAD, VadParams, kept_length_s

FORMAT, VERSION = "torgo-segment-labels", 1
SEGMENT_TYPES = {"word", "partial", "non_speech"}
STATUSES = {"todo", "done", "no_word"}


@dataclass(frozen=True)
class SegmentLabelResult:
    samples: pd.DataFrame      # training rows; seg_start/seg_end set for labelled words, NaN otherwise
    non_speech: pd.DataFrame   # labelled non-speech segments (candidates for an "unknown" class)
    dropped: pd.DataFrame      # rows removed, with a `reason` column


def _validate_clip(clip_id: str, clip: dict) -> None:
    if clip.get("status") not in STATUSES:
        raise ValueError(f"{clip_id}: status must be one of {sorted(STATUSES)}, got {clip.get('status')!r}")
    for seg in clip.get("segments", []):
        start, end, kind = seg.get("start"), seg.get("end"), seg.get("type")
        if kind not in SEGMENT_TYPES:
            raise ValueError(f"{clip_id}: segment type must be one of {sorted(SEGMENT_TYPES)}, got {kind!r}")
        if not (isinstance(start, (int, float)) and isinstance(end, (int, float)) and 0 <= start < end):
            raise ValueError(f"{clip_id}: segment needs 0 <= start < end, got start={start!r} end={end!r}")


def load_segment_labels(path: Path) -> Dict[str, dict]:
    """
    Read a labeler export. Returns {path relative to the TORGO root: clip entry}.
    A missing file means nothing has been labelled yet and returns {}.
    """
    path = Path(path)
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        raise ValueError(f"{path}: expected format {FORMAT!r} version {VERSION}, "
                         f"got {doc.get('format')!r} version {doc.get('version')!r}")
    clips = doc.get("clips", {})
    for clip_id, clip in clips.items():
        _validate_clip(clip_id, clip)
    return clips


def measure_kept_lengths(df: pd.DataFrame, params: VadParams = DEFAULT_VAD) -> pd.DataFrame:
    """Return a copy of df with `kept_s`: seconds left after silence trimming."""
    def kept(path: str) -> float:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return kept_length_s(audio.mean(axis=1), sr, params)

    return df.assign(kept_s=[kept(p) for p in df["file_path"]])


def _segment_rows(row: pd.Series, segments: list, kind: str) -> list:
    return [{**row.to_dict(), "seg_start": float(s["start"]), "seg_end": float(s["end"])}
            for s in sorted(segments, key=lambda s: s["start"]) if s["type"] == kind]


def apply_segment_labels(df: pd.DataFrame, labels: Dict[str, dict],
                         torgo_root: Path, window_s: float) -> SegmentLabelResult:
    """
    Split scanned rows (with `kept_s`) into samples, non-speech segments and drops.

    Only labels with status "done" or "no_word" count; "todo" is unfinished work.
    """
    torgo_root = Path(torgo_root)
    samples, non_speech, dropped = [], [], []
    for _, row in df.iterrows():
        clip_id = Path(row["file_path"]).relative_to(torgo_root).as_posix()
        clip = labels.get(clip_id, {})
        status = clip.get("status", "todo")
        segments = clip.get("segments", []) if status != "todo" else []
        non_speech += _segment_rows(row, segments, "non_speech")

        if status == "no_word":
            dropped.append({**row.to_dict(), "reason": "no_word"})
        elif status == "done":
            words = _segment_rows(row, segments, "word")
            samples += words or []
            if not words:
                dropped.append({**row.to_dict(), "reason": "no_word_segment"})
        elif row["kept_s"] > window_s:
            dropped.append({**row.to_dict(), "reason": "unlabeled_over_window"})
        else:
            samples.append({**row.to_dict(), "seg_start": np.nan, "seg_end": np.nan})

    columns = list(df.columns) + ["seg_start", "seg_end"]
    return SegmentLabelResult(
        samples=pd.DataFrame(samples, columns=columns).reset_index(drop=True),
        non_speech=pd.DataFrame(non_speech, columns=columns).reset_index(drop=True),
        dropped=pd.DataFrame(dropped, columns=list(df.columns) + ["reason"]).reset_index(drop=True),
    )
