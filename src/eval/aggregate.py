"""
Combine one model's seeds into a summary with two kinds of uncertainty.

- seed_std: how much the headline moves between training runs.
- ci_low/ci_high: bootstrap 95% interval from resampling the 8 speakers,
  i.e. how much it would move for a different set of users. With 8 speakers
  this is the wide one, and the one that matters for "works for a new user".
"""

import warnings
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.eval.constants import DYSARTHRIC_SPEAKERS, HEADLINE_MIC
from src.eval.metrics import RunMetrics, compute_run_metrics
from src.eval.schema import EvalValidationError, Run, check_same_clips, clip_fingerprint

N_BOOT = 10_000
CI_LEVEL = 0.95
MIN_SEEDS = 3


@dataclass(frozen=True)
class ModelSummary:
    model: str
    mic: str
    n_seeds: int
    headline: float                  # mean over seeds of speaker-averaged accuracy
    seed_std: float                  # sample std over seeds (NaN for one seed)
    ci_low: float                    # bootstrap over speakers
    ci_high: float
    per_speaker: pd.DataFrame        # index speaker; n, accuracy (mean over seeds)
    severity: Mapping[str, float]
    pooled: float
    pretraining_overlap: Mapping[str, float]
    oov_rate: float
    reject_rate: float
    confusion: pd.DataFrame          # summed over seeds
    control_headline: Optional[float]
    clip_fingerprint: str            # dysarthric clips and labels scored on `mic`


def summarize(runs: Sequence[Run], mic: str = HEADLINE_MIC,
              n_boot: int = N_BOOT, rng_seed: int = 0) -> ModelSummary:
    runs = list(runs)
    check_model_runs(runs)
    metrics = [compute_run_metrics(r, mic) for r in runs]
    headlines = np.array([m.headline for m in metrics])
    per_speaker = mean_per_speaker(metrics)
    ci_low, ci_high = bootstrap_mean_ci(per_speaker["accuracy"].to_numpy(), n_boot, rng_seed)
    controls = [m.control_headline for m in metrics if m.control_headline is not None]

    return ModelSummary(
        model=runs[0].model,
        mic=mic,
        n_seeds=len(runs),
        headline=float(headlines.mean()),
        seed_std=float(headlines.std(ddof=1)) if len(runs) > 1 else float("nan"),
        ci_low=ci_low,
        ci_high=ci_high,
        per_speaker=per_speaker,
        severity=_mean_of_dicts([m.severity for m in metrics]),
        pooled=float(np.mean([m.pooled for m in metrics])),
        pretraining_overlap=_mean_of_dicts([m.pretraining_overlap for m in metrics]),
        oov_rate=float(np.mean([m.oov_rate for m in metrics])),
        reject_rate=float(np.mean([m.reject_rate for m in metrics])),
        confusion=sum((m.confusion for m in metrics[1:]), metrics[0].confusion),
        control_headline=float(np.mean(controls)) if controls else None,
        clip_fingerprint=scored_fingerprint(runs[0], mic),
    )


def check_model_runs(runs: Sequence[Run]) -> None:
    """The runs are the seeds of one model, each seed once, all on the same clips."""
    if not runs:
        raise EvalValidationError("no runs to summarize")
    models = sorted({r.model for r in runs})
    if len(models) > 1:
        raise EvalValidationError(f"summarize one model at a time, got {models}")
    seeds = [r.seed for r in runs]
    if len(set(seeds)) != len(seeds):
        raise EvalValidationError(f"repeated seed in {seeds}")
    check_same_clips(runs)
    if len(runs) < MIN_SEEDS and not runs[0].zero_shot:
        warnings.warn(f"{models[0]}: {len(runs)} seed(s); report trained models over "
                      f"at least {MIN_SEEDS} seeds", UserWarning, stacklevel=3)


def scored_fingerprint(run: Run, mic: str) -> str:
    """
    Fingerprint of what the headline scores: dysarthric clips on `mic`.
    Zero-shot runs also score control speakers, which must not make them
    incomparable with trained models.
    """
    p = run.predictions
    return clip_fingerprint(p[(p["mic"] == mic) & p["speaker_id"].isin(DYSARTHRIC_SPEAKERS)])


def check_summaries_comparable(summaries: Sequence[ModelSummary]) -> None:
    """Models share a table or figure only if scored on the same clips and mic."""
    if not summaries:
        raise EvalValidationError("no model summaries to report")
    names = [s.model for s in summaries]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise EvalValidationError(f"model listed more than once: {repeated}")
    mics = sorted({s.mic for s in summaries})
    if len(mics) > 1:
        raise EvalValidationError(f"models summarised on different mics: {mics}")
    if len({s.clip_fingerprint for s in summaries}) > 1:
        raise EvalValidationError(
            f"models were not scored on the same clips with the same labels: {names}")


def mean_per_speaker(metrics: Sequence[RunMetrics]) -> pd.DataFrame:
    """Per-speaker clip count and accuracy averaged over seeds."""
    acc = pd.concat([m.per_speaker["accuracy"] for m in metrics], axis=1).mean(axis=1)
    return pd.DataFrame({"n": metrics[0].per_speaker["n"], "accuracy": acc})


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT, rng_seed: int = 0,
                      level: float = CI_LEVEL) -> Tuple[float, float]:
    """Percentile interval of the mean, resampling `values` with replacement."""
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(lo), float(hi)


def _mean_of_dicts(dicts: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}
