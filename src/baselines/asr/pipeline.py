"""
Zero-shot ASR baseline: TORGO clips -> transcripts -> eval-harness runs.

Transcripts are cached per clip (runs/<model>/transcripts.csv) as they are
produced, so a crash resumes where it stopped, and rescoring never re-runs
the model. Each model yields two zero-shot runs, <model>-strict and
<model>-lenient, and one cost profile.
"""

from pathlib import Path
from typing import Callable, List

import librosa
import numpy as np
import pandas as pd

from src.audio import prepare_waveform
from src.baselines.asr.scoring import SCORERS
from src.baselines.asr.transcribers import Transcriber
from src.config import Config
from src.data.dataset import TORGOCommandDataset
from src.eval.constants import ARRAY_MIC, DYSARTHRIC_SPEAKERS
from src.eval.cost import COST_FILE, CostProfile, count_macs_of, count_params, load_cost, save_cost
from src.eval.io import save_run
from src.eval.schema import CLIP_KEY, PRED_COLUMNS, Run

SEED = 0  # greedy decoding is deterministic: one run per model
TRANSCRIPTS_FILE = "transcripts.csv"
# A clip's audio depends on its hand-labelled segment too; if the segment
# changes, the cached transcript is for different audio and is not reused.
CACHE_KEY = list(CLIP_KEY) + ["segment"]


def clip_window(row: pd.Series) -> np.ndarray:
    """The 2 s window the classifiers see for this clip (src/audio.py)."""
    try:
        audio, _ = librosa.load(row["file_path"], sr=Config.SAMPLE_RATE, mono=True)
    except Exception as e:  # never score silence in place of an unreadable file
        raise RuntimeError(f"could not read {row['file_path']}: {e}") from e
    return prepare_waveform(audio, Config.SAMPLE_RATE, Config.MAX_AUDIO_SAMPLES,
                            segment=TORGOCommandDataset.segment_of(row))


def segment_key(row: pd.Series) -> str:
    segment = TORGOCommandDataset.segment_of(row)
    return "" if segment is None else f"{segment[0]:.3f}-{segment[1]:.3f}"


def read_cache(path: Path) -> pd.DataFrame:
    """Everything as text; an empty transcript stays "" (not NaN)."""
    if not path.exists():
        return pd.DataFrame(columns=CACHE_KEY + ["transcript"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def transcribe_clips(clips: pd.DataFrame, get_transcriber: Callable[[], Transcriber],
                     cache_path: Path, batch_size: int,
                     load_window: Callable = clip_window) -> pd.DataFrame:
    """
    `clips` plus `segment` and `transcript`. Only clips missing from the cache
    are transcribed, and each batch is appended as soon as it is done. The
    model is only built (get_transcriber) if something is missing.
    """
    cache_path = Path(cache_path)
    keyed = clips.assign(segment=[segment_key(r) for _, r in clips.iterrows()])
    done = set(read_cache(cache_path)[CACHE_KEY].itertuples(index=False, name=None))
    todo = keyed[[k not in done for k in keyed[CACHE_KEY].itertuples(index=False, name=None)]]
    if len(todo):
        transcriber = get_transcriber()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(todo), batch_size):
            _transcribe_batch(todo.iloc[start:start + batch_size], transcriber, cache_path,
                              load_window)
            print(f"  transcribed {min(start + batch_size, len(todo))}/{len(todo)}")
    cache = read_cache(cache_path).drop_duplicates(CACHE_KEY, keep="last")
    return keyed.merge(cache, on=CACHE_KEY, how="left", validate="one_to_one")


def _transcribe_batch(batch: pd.DataFrame, transcriber: Transcriber, cache_path: Path,
                      load_window: Callable) -> None:
    texts = transcriber.transcribe([load_window(r) for _, r in batch.iterrows()])
    if len(texts) != len(batch):
        raise ValueError(f"got {len(texts)} transcripts for a batch of {len(batch)} clips")
    rows = batch[CACHE_KEY].assign(transcript=[" ".join(str(t).split()) for t in texts])
    rows.to_csv(cache_path, mode="a", header=not cache_path.exists(), index=False)


def scored_run(transcripts: pd.DataFrame, model: str, scorer: str) -> Run:
    """A zero-shot run named <model>-<scorer>; the transcript is kept as an extra column."""
    columns = [c for c in PRED_COLUMNS if c != "pred"]
    preds = transcripts[columns + ["transcript"]].assign(
        pred=transcripts["transcript"].map(SCORERS[scorer]))
    return Run(model=f"{model}-{scorer}", seed=SEED, predictions=preds, zero_shot=True)


def run_dir(runs_dir: Path, run_name: str) -> Path:
    return Path(runs_dir) / run_name / f"seed{SEED}" / "eval"


def reference_clip(clips: pd.DataFrame) -> pd.Series:
    """The clip cost is measured on: the first dysarthric array-mic clip in key order."""
    dys = clips[(clips["mic"] == ARRAY_MIC) & clips["speaker_id"].isin(DYSARTHRIC_SPEAKERS)]
    return dys.sort_values(list(CLIP_KEY)).iloc[0]


def transcription_cost(transcriber: Transcriber, window: np.ndarray) -> CostProfile:
    """Params of the whole model; MACs of transcribing one window, decoding included."""
    return CostProfile(
        params=count_params(transcriber.model),
        macs=count_macs_of(lambda: transcriber.transcribe([window])),
        input_seconds=len(window) / Config.SAMPLE_RATE,
        note=transcriber.cost_note,
    )


def run_baseline(model: str, clips: pd.DataFrame, get_transcriber: Callable[[], Transcriber],
                 runs_dir: Path, batch_size: int,
                 load_window: Callable = clip_window) -> List[Run]:
    """Transcribe (or reuse the cache), save strict and lenient runs, write the cost once."""
    model_dir = Path(runs_dir) / model
    transcripts = transcribe_clips(clips, get_transcriber, model_dir / TRANSCRIPTS_FILE,
                                   batch_size, load_window)
    runs = [scored_run(transcripts, model, scorer) for scorer in SCORERS]
    for run in runs:
        save_run(run, run_dir(runs_dir, run.model))
    cost_path = model_dir / COST_FILE
    if not cost_path.exists():
        window = load_window(reference_clip(clips))
        save_cost(transcription_cost(get_transcriber(), window), cost_path)
    return runs
