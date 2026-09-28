"""
The numbers the demo shows, read from the harness's combined report
(outputs/results/results.csv), so nothing on the page is typed by hand.
csv module only: the Space does not install pandas.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from src.config import Config
from src.eval.constants import HEADLINE_MIC
from src.eval.units import human_count

RESULTS_CSV = Config.OUTPUT_DIR / "results" / "results.csv"
KWS_RUN = "bcresnet-8"
ASR_RUN = "parakeet-tdt-0.6b-v3-lenient"
# (run name in results.csv, name on the page), in table order
TABLE_ROWS = (
    ("whisper-large-v3-lenient", "Whisper large-v3 (zero-shot)"),
    (ASR_RUN, "Parakeet-TDT-0.6B-v3 (zero-shot)"),
    (KWS_RUN, "BC-ResNet-8"),
)
NUMERIC_COLUMNS = ("accuracy", "ci_low", "ci_high", "severe", "mild", "params", "macs")
REQUIRED_COLUMNS = ("model", "mic") + NUMERIC_COLUMNS
HEADER = ("Model", "Params", "MACs / 2 s", "Accuracy", "95% CI", "Severe", "Mild")


@dataclass(frozen=True)
class ModelCost:
    params: str  # e.g. "323k"
    macs: str    # per 2 s window, e.g. "171M"


@dataclass(frozen=True)
class DemoResults:
    table_markdown: str
    kws_cost: ModelCost
    asr_cost: ModelCost


def format_params(n: float) -> str:
    """human_count, with billions written B as the README does (1543490560 -> 1.5B)."""
    text = human_count(n)
    return text[:-1] + "B" if text.endswith("G") else text


def _number(text: str, path: Path, run: str, column: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if math.isnan(value):
        raise ValueError(f"{path}: {run} has no value for {column}")
    return value


def _read_rows(path: Path) -> Dict[str, Dict[str, float]]:
    """Run name -> numeric columns, for the TABLE_ROWS runs on the headline mic."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/report_results.py first")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks columns {missing}")
        rows = {r["model"]: r for r in reader if r["mic"] == HEADLINE_MIC}
    numbers = {}
    for run, _ in TABLE_ROWS:
        if run not in rows:
            raise ValueError(f"{path} has no {HEADLINE_MIC} row for {run}")
        numbers[run] = {c: _number(rows[run][c], path, run, c) for c in NUMERIC_COLUMNS}
    return numbers


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _table(rows: Dict[str, Dict[str, float]]) -> str:
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    for run, name in TABLE_ROWS:
        r = rows[run]
        cells = (name, format_params(r["params"]), human_count(r["macs"]),
                 f"{_pct(r['accuracy'])}%", f"[{_pct(r['ci_low'])}, {_pct(r['ci_high'])}]",
                 f"{_pct(r['severe'])}%", f"{_pct(r['mild'])}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _cost(row: Dict[str, float]) -> ModelCost:
    return ModelCost(params=format_params(row["params"]), macs=human_count(row["macs"]))


def load_results(path: Path = RESULTS_CSV) -> DemoResults:
    rows = _read_rows(Path(path))
    return DemoResults(table_markdown=_table(rows), kws_cost=_cost(rows[KWS_RUN]),
                       asr_cost=_cost(rows[ASR_RUN]))
