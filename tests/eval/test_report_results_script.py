"""scripts/report_results.py: finds the saved runs and costs (the report itself is tested
in test_combined.py)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src.eval.cost import CostProfile, save_cost
from src.eval.io import save_run
from tests.eval.test_combined import asr_run, trained_run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "report_results.py"
ASR = ("whisper-large-v3", "parakeet-tdt-0.6b-v3")


def load_script():
    spec = importlib.util.spec_from_file_location("report_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_runs(runs_dir: Path, seeds=(0, 1), with_metrics=True) -> Path:
    for m in ASR:
        save_run(asr_run({}, model=f"{m}-lenient"), runs_dir / f"{m}-lenient" / "seed0" / "eval")
        save_cost(CostProfile(params=600_000_000, macs=10**10, input_seconds=2.0),
                  runs_dir / m / "cost.json")
    for tau in (1, 8):
        model = f"bcresnet-{tau}"
        save_cost(CostProfile(params=9_000 * tau, macs=5_000_000 * tau, input_seconds=2.0),
                  runs_dir / model / "cost.json")
        for seed in seeds:
            seed_dir = runs_dir / model / f"seed{seed}"
            save_run(trained_run(model, seed, {}), seed_dir / "eval")
            if with_metrics:
                (seed_dir / "pretrain_metrics.json").write_text(
                    json.dumps({"test_acc": 0.9 + seed / 100}))
    return runs_dir


def test_defaults_report_every_width_against_lenient_asr():
    args = load_script().parse_args([])
    assert (args.scorer, args.seeds, args.taus) == ("lenient", None, [1, 2, 3, 8])


def test_speech_commands_accuracy_is_the_mean_over_the_reported_seeds(tmp_path):
    runs_dir = fake_runs(tmp_path)
    acc = load_script().speech_commands_accuracy(runs_dir, "bcresnet-8", [0, 1])
    assert acc == pytest.approx(0.905)


def test_speech_commands_accuracy_is_none_if_a_seed_lacks_metrics(tmp_path):
    runs_dir = fake_runs(tmp_path, with_metrics=False)
    assert load_script().speech_commands_accuracy(runs_dir, "bcresnet-8", [0]) is None


def test_main_writes_the_report_for_the_requested_widths(tmp_path, capsys):
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out), "--taus", "1", "8"])
    notes = [l for l in capsys.readouterr().out.splitlines() if l.startswith("note:")]
    assert len(notes) == 2  # one "fewer than 3 seeds" note per width, not one per table
    table = pd.read_csv(out / "results.csv")
    assert table["model"].tolist() == ["whisper-large-v3-lenient", "parakeet-tdt-0.6b-v3-lenient",
                                       "bcresnet-1", "bcresnet-8"]
    assert table["n_seeds"].tolist() == [1, 1, 2, 2]
    sc = pd.read_csv(out / "speech_commands.csv")
    assert sc["model"].tolist() == ["bcresnet-1", "bcresnet-8"]
    assert len(pd.read_csv(out / "comparisons.csv")) == 4


def test_main_can_pin_the_seeds(tmp_path):
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out),
                        "--taus", "8", "--seeds", "0"])
    assert pd.read_csv(out / "results.csv")["n_seeds"].tolist() == [1, 1, 1]
