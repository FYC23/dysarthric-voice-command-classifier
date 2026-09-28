"""scripts/report_results.py: finds the saved runs and costs (the report itself is tested
in test_combined.py)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import config
from src.eval.cost import CostProfile, save_cost
from src.eval.io import save_run
from tests.eval.test_combined import asr_run, trained_run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "report_results.py"
ASR = ("whisper-large-v3", "parakeet-tdt-0.6b-v3")
SSL = ("hubert-large", "hubert-base", "distilhubert")
SSL_SEEDS = (0, 1, 2)


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
    for backbone in SSL:
        for model in (backbone, f"{backbone}-controls"):
            save_cost(CostProfile(params=10**8, macs=10**10, input_seconds=2.0,
                                  note="CNN front end, every transformer layer and the head"),
                      runs_dir / model / "cost.json")
            for seed in SSL_SEEDS:
                save_run(trained_run(model, seed, {}), runs_dir / model / f"seed{seed}" / "eval")
    return runs_dir


def test_defaults_report_every_width_and_backbone_against_lenient_asr():
    args = load_script().parse_args([])
    assert (args.scorer, args.seeds, args.taus) == ("lenient", None, [1, 2, 3, 8])
    assert args.backbones == list(SSL)
    assert args.out == config.OUTPUT_DIR / "results"


def test_speech_commands_accuracy_is_the_mean_over_the_reported_seeds(tmp_path):
    runs_dir = fake_runs(tmp_path)
    acc = load_script().speech_commands_accuracy(runs_dir, "bcresnet-8", [0, 1])
    assert acc == pytest.approx(0.905)


def test_speech_commands_accuracy_is_none_if_a_seed_lacks_metrics(tmp_path):
    runs_dir = fake_runs(tmp_path, with_metrics=False)
    assert load_script().speech_commands_accuracy(runs_dir, "bcresnet-8", [0]) is None


def test_main_writes_the_report_for_the_requested_widths(tmp_path, capsys):
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out), "--taus", "1", "8",
                        "--backbones"])
    notes = [l for l in capsys.readouterr().out.splitlines() if l.startswith("note:")]
    assert len(notes) == 2  # one "fewer than 3 seeds" note per width, not one per table
    table = pd.read_csv(out / "results.csv")
    assert table["model"].tolist() == ["whisper-large-v3-lenient", "parakeet-tdt-0.6b-v3-lenient",
                                       "bcresnet-1", "bcresnet-8"]
    assert table["n_seeds"].tolist() == [1, 1, 2, 2]
    sc = pd.read_csv(out / "speech_commands.csv")
    assert sc["model"].tolist() == ["bcresnet-1", "bcresnet-8"]
    assert len(pd.read_csv(out / "comparisons.csv")) == 4


def test_main_adds_each_backbone_and_its_controls_only_run(tmp_path):
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out), "--taus", "8",
                        "--backbones", "hubert-large", "distilhubert"])
    table = pd.read_csv(out / "results.csv").set_index("model")
    assert table.index.tolist()[2:] == ["bcresnet-8", "hubert-large", "distilhubert",
                                        "hubert-large-controls", "distilhubert-controls"]
    assert table.loc["hubert-large", "n_seeds"] == 3
    assert table.loc["hubert-large", "note"].startswith("MACs include the CNN front end")
    assert "control speakers only" in table.loc["hubert-large-controls", "note"]
    assert "not counted" not in table.loc["hubert-large", "note"]


def test_comparisons_answer_three_questions(tmp_path):
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out), "--taus", "1", "8",
                        "--backbones", "hubert-large", "distilhubert"])
    table = pd.read_csv(out / "comparisons.csv")
    pairs = {q: list(zip(g["candidate"], g["baseline"])) for q, g in table.groupby("question")}
    assert len(pairs["Trained models vs zero-shot ASR"]) == 6 * 2
    assert pairs["Effect of dysarthric fine-tuning"] == [
        ("hubert-large", "hubert-large-controls"), ("distilhubert", "distilhubert-controls")]
    assert pairs["Small models vs the large pretrained reference"] == [
        ("bcresnet-1", "hubert-large"), ("bcresnet-8", "hubert-large")]
    md = (out / "comparisons.md").read_text()
    assert "## Effect of dysarthric fine-tuning" in md


def test_without_hubert_large_there_is_no_reference_question(tmp_path):
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out), "--taus", "8",
                        "--backbones", "distilhubert"])
    questions = set(pd.read_csv(out / "comparisons.csv")["question"])
    assert questions == {"Trained models vs zero-shot ASR", "Effect of dysarthric fine-tuning"}


def test_controls_only_runs_stay_off_the_cost_figures(tmp_path, monkeypatch):
    plotted = []

    def record(summaries, costs, path, axis="macs", secondary=None, secondary_label=None,
               labels=None):
        plotted.append((axis, [s.model for s in summaries], labels))

    monkeypatch.setattr("src.eval.plots.plot_accuracy_vs_cost", record)
    monkeypatch.setattr("src.eval.combined.plot_accuracy_vs_cost", record)
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out), "--taus", "8"])
    assert len(plotted) == 4  # MACs and params, per mic
    for _, models, labels in plotted:
        assert models[2:] == ["bcresnet-8", "hubert-large", "hubert-base", "distilhubert"]
        assert (labels["hubert-large"], labels["distilhubert"]) == ("HuBERT-large",
                                                                    "DistilHuBERT")
        assert labels["hubert-base-controls"] == "HuBERT-base (controls only)"
    assert (out / "confusion_hubert-large-controls.png").stat().st_size > 0


def test_main_can_pin_the_seeds(tmp_path):
    runs_dir, out = fake_runs(tmp_path / "runs"), tmp_path / "out"
    load_script().main(["--runs-dir", str(runs_dir), "--out", str(out),
                        "--taus", "8", "--seeds", "0", "--backbones", "distilhubert"])
    # --seeds pins BC-ResNet only; the SSL models use every finished seed
    assert pd.read_csv(out / "results.csv")["n_seeds"].tolist() == [1, 1, 1, 3, 3]
