"""
Leave-one-speaker-out helpers shared by every trained model (HuBERT in
scripts/train.py, BC-ResNet in src/training/finetune.py): the fold split,
eval-harness rows, building and saving a run, class weights and seeding.
"""

import random
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.data.segments import evaluation_clips
from src.eval.io import save_run
from src.eval.schema import PRED_COLUMNS, EvalValidationError, Run

UNVALIDATED_PREDICTIONS = "predictions_unvalidated.csv"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_fold(dysarthric_df: pd.DataFrame, test_speaker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    (train, test) rows for one LOSO fold. Training keeps every labelled word
    attempt; the test side scores each recording once (its first attempt), as
    every model in the eval harness does.
    """
    held_out = dysarthric_df['speaker_id'] == test_speaker
    return dysarthric_df[~held_out].copy(), evaluation_clips(dysarthric_df[held_out]).copy()


def fold_predictions(test_df: pd.DataFrame, pred_ids, label_ids,
                     id2label: Mapping[int, str]) -> pd.DataFrame:
    """
    Eval-harness rows for one fold: each clip's identity from test_df plus its
    predicted word. Evaluation loaders do not shuffle, so predictions come back
    in table order; the returned labels are checked to prove it.
    """
    clips = test_df.reset_index(drop=True)
    returned = [int(i) for i in label_ids]
    if len(pred_ids) != len(clips) or returned != clips['label_id'].astype(int).tolist():
        raise ValueError("predictions are not in the order of the test table; "
                         "cannot attach them to clips")
    clip_columns = [c for c in PRED_COLUMNS if c != 'pred']
    return clips[clip_columns].assign(pred=[id2label[int(i)] for i in pred_ids])


def build_loso_run(frames, fold_train_speakers: Mapping[str, frozenset], control_speakers,
                   seed: int, model: str) -> Run:
    """
    One validated eval-harness run from the LOSO folds. Every fold starts from
    a checkpoint trained on all control speakers, so they are recorded as
    training speakers of every fold.
    """
    controls = frozenset(control_speakers)
    return Run(
        model=model,
        seed=seed,
        predictions=pd.concat(frames, ignore_index=True),
        fold_train_speakers={s: frozenset(t) | controls for s, t in fold_train_speakers.items()},
    )


def save_loso_run(frames, fold_train_speakers: Mapping[str, frozenset], control_speakers,
                  seed: int, model: str, out_dir: Path) -> Path:
    """
    Validate and save the LOSO run for src.eval.io.load_run. If validation
    fails, the raw predictions are kept first: eight folds are expensive.
    """
    out_dir = Path(out_dir)
    try:
        run = build_loso_run(frames, fold_train_speakers, control_speakers, seed, model)
    except EvalValidationError:
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / UNVALIDATED_PREDICTIONS
        pd.concat(frames, ignore_index=True).to_csv(raw_path, index=False)
        print(f"Eval-harness validation failed; raw predictions kept at {raw_path}")
        raise
    save_run(run, out_dir)
    return out_dir


def balanced_class_weights(label_ids: Sequence[int], num_classes: int) -> torch.Tensor:
    """
    sklearn's "balanced" weights, n_samples / (n_present_classes * count), for
    the classes present; 1.0 for classes absent from this training set.
    """
    labels = np.asarray(label_ids, dtype=np.int64)
    if len(labels) and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(f"label ids must be in [0, {num_classes}), got {sorted(set(labels))}")
    counts = np.bincount(labels, minlength=num_classes)
    present = counts > 0
    weights = np.ones(num_classes)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)
