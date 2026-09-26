"""
Paired comparison of two models scored on the same clips.

Both models see the same speakers, so compare speaker by speaker ("better on
7 of 8") rather than two means with overlapping error bars. The interval is
a bootstrap over speakers of the mean per-speaker difference.
"""

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.eval.aggregate import (
    N_BOOT, bootstrap_mean_ci, check_model_runs, mean_per_speaker, scored_fingerprint,
)
from src.eval.constants import HEADLINE_MIC
from src.eval.metrics import compute_run_metrics
from src.eval.schema import EvalValidationError, Run

TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PairedComparison:
    baseline: str
    candidate: str
    mic: str
    per_speaker_diff: pd.Series  # candidate - baseline, seed-averaged accuracy
    n_better: int
    n_worse: int
    n_tied: int
    mean_diff: float
    ci_low: float
    ci_high: float


def compare(baseline_runs: Sequence[Run], candidate_runs: Sequence[Run],
            mic: str = HEADLINE_MIC, n_boot: int = N_BOOT, rng_seed: int = 0) -> PairedComparison:
    check_model_runs(baseline_runs)
    check_model_runs(candidate_runs)
    if scored_fingerprint(baseline_runs[0], mic) != scored_fingerprint(candidate_runs[0], mic):
        raise EvalValidationError(
            f"{baseline_runs[0].model} and {candidate_runs[0].model} were not scored on "
            "the same clips with the same labels")

    base = _speaker_accuracy(baseline_runs, mic)
    cand = _speaker_accuracy(candidate_runs, mic)
    diff = cand - base
    ci_low, ci_high = bootstrap_mean_ci(diff.to_numpy(), n_boot, rng_seed)
    return PairedComparison(
        baseline=baseline_runs[0].model,
        candidate=candidate_runs[0].model,
        mic=mic,
        per_speaker_diff=diff,
        n_better=int((diff > TIE_TOLERANCE).sum()),
        n_worse=int((diff < -TIE_TOLERANCE).sum()),
        n_tied=int((diff.abs() <= TIE_TOLERANCE).sum()),
        mean_diff=float(diff.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _speaker_accuracy(runs: Sequence[Run], mic: str) -> pd.Series:
    return mean_per_speaker([compute_run_metrics(r, mic) for r in runs])["accuracy"]
