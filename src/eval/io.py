"""
Save and load runs as a directory: predictions.csv plus run.json metadata.

This is how predictions move between machines (e.g. HuBERT folds on the GPU
box, scored on the Mac). Loading re-validates, so a hand-edited or stale run
cannot slip a leak into a table.
"""

import json
from pathlib import Path

import pandas as pd

from src.eval.schema import PRED_COLUMNS, Run

PREDICTIONS_FILE = "predictions.csv"
METADATA_FILE = "run.json"


def save_run(run: Run, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run.predictions.to_csv(run_dir / PREDICTIONS_FILE, index=False)
    meta = {
        "model": run.model,
        "seed": run.seed,
        "zero_shot": run.zero_shot,
        "fold_train_speakers": {k: sorted(v) for k, v in run.fold_train_speakers.items()},
    }
    (run_dir / METADATA_FILE).write_text(json.dumps(meta, indent=2))


def load_run(run_dir: Path) -> Run:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / METADATA_FILE).read_text())
    # Read the key columns as strings: utterance ids like "0001" must not become 1
    preds = pd.read_csv(run_dir / PREDICTIONS_FILE,
                        dtype={c: str for c in PRED_COLUMNS}, keep_default_na=False,
                        na_values=[""])
    return Run(
        model=meta["model"],
        seed=int(meta["seed"]),
        predictions=preds,
        zero_shot=bool(meta["zero_shot"]),
        fold_train_speakers={k: frozenset(v) for k, v in meta["fold_train_speakers"].items()},
    )
